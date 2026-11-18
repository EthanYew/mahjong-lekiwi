"""Wrist-camera photos of every face-up tile on the mat, taken from the hover, for a tile-name check at the
hover (ROADMAP 10): the arm hovers over each tile --views times, each from a different place, height and
wrist roll, and saves the wrist camera's frames under the tile's name.

    python scripts/wrist_capture.py                       # 8 views a tile, then asks for the next set
    python scripts/wrist_capture.py --views 12 --jitter-mm 10
    python scripts/wrist_capture.py --sim --sets 1        # simulated arm and mat, no hardware

Each set: the arm rests, the RealSense scans the mat, and the tiles are listed left to right with the
detector's names (and drawn, numbered, on the RealSense view). Enter accepts the names; "3=2C" renames
tile 3, "-3" leaves it out, "r" scans again. Then for every tile and view: lift, hover over the tile
(measured hover correction, settled from above) moved by a random offset within --jitter-mm (the first
view stays centred), at the next height in --clearances-mm, the wrist rolled by up to --roll-deg, and
save --frames wrist frames. The tiles are never touched. When every tile of the set has its views, the
arm rests, the terminal rings (and a desktop notification pops up), and it waits for you to put a new
set of tiles down.

Saved to sessions/<time>_wrist/wrist/<NAME>/<set>_<tile>_v<view>_f<k>.jpg (one folder per name: an
image-classification layout), with a `wrist_view` event per view (name, detector read, hover, jitter,
height, roll). After each set it prints how many views every name has over all wrist sessions, and
which still have fewer than --target.

    python scripts/wrist_capture.py --empty even-hand14   # the same hovers over a layout's bare spots

--empty LAYOUT takes photos of nothing between the jaws, which the wrist camera's back check (mahjong/back_check.py)
needs as much as backs, and the arm otherwise only hovers where a tile is. Lay face-down tiles on some of the layout's
spots and leave the rest bare: the scan lists the spots with no tile within EMPTY_CLEAR_M (a back too, with weights that
see backs; check the list by eye with older ones), and the arm hovers over each bare spot as over a tile, so its
neighbours show at the edge of the view as they do at a real look. Saved to
sessions/<time>_wrist/wrist_empty/empty/<set>_<spot>_v<view>_f<k>.jpg, kept out of wrist/ so the name check never
reads them. Tiles are never touched.
"""
import argparse
import math
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import back_check, config, detector, hardware, layout, perception, wrist_photos
from mahjong.arm import ADJUST_TIME_S, SETTLE_S
from mahjong.camera import grab_frames
from mahjong.hover_correction import HoverCorrection
from mahjong.placements import parse_answer
from mahjong.session import Session, read_jsonl

WINDOW = "wrist capture"
ROLL_MARGIN_DEG = 5.0        # a rolled view stays this far inside the roll's command limits
RETRIES = 3                  # fresh jitter draws for a view whose hover is out of reach
# --empty: a spot is bare with no detection this close (centres). Layout neighbours are 50-51mm apart, so a tile on the
# next spot doesn't make one look taken, and one knocked half-way does.
EMPTY_CLEAR_M = 0.035


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--views", type=int, default=8, help="hovers per tile (default 8)")
    p.add_argument("--jitter-mm", type=float, default=8.0,
                   help="each view's hover moves by a random offset within this radius (default 8; record.py's "
                        "training jitter is 6)")
    p.add_argument("--clearances-mm", default="40,55,70",
                   help="hover heights above the grasp point, cycled over the views (default 40,55,70; the "
                        "policies hover at 40)")
    p.add_argument("--roll-deg", type=float, default=15.0, help="wrist roll varied by up to this (default 15)")
    p.add_argument("--frames", type=int, default=2, help="wrist frames saved per view (default 2)")
    p.add_argument("--target", type=int, default=10, help="views per name to aim for over all sessions (default 10)")
    p.add_argument("--sets", type=int, default=None, help="stop after this many sets of tiles (default: until q)")
    p.add_argument("--seed", type=int, default=None, help="random seed for the jitter and roll")
    p.add_argument("--no-view", action="store_true", help="no RealSense / wrist windows")
    p.add_argument("--empty", metavar="LAYOUT", default=None,
                   help="hover over this layout's bare spots instead of over tiles, for the back check's empty photos")
    p.add_argument("--sim", action="store_true", help="simulated arm, camera and mat")
    return p.parse_args(argv)


def pumping_sleep(seconds, pump=lambda: cv2.waitKey(1), clock=time.monotonic, sleep=time.sleep):
    """Sleep that keeps the OpenCV windows answering: the arm's moves and settles wait through Arm.sleep,
    and a window whose events nobody handles for a few seconds is marked "not responding"."""
    end = clock() + seconds
    pump()
    sleep(max(0.0, end - clock()))


def ask_pumping(ask, pump=lambda: cv2.waitKey(50)):
    """ask(prompt) on a thread while this one keeps the windows answering (input() blocks them otherwise)."""
    def asking(prompt):
        answer = {}

        def run():
            try:
                answer["text"] = ask(prompt)
            except BaseException as error:        # e.g. EOFError: raised here, never read as Enter
                answer["error"] = error
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        while thread.is_alive():
            pump()
        if "error" in answer:
            raise answer["error"]
        return answer["text"]
    return asking


def plan_views(n, jitter_m, clearances_m, roll_deg, rng):
    """n views of one tile: (dx, dy, clearance, roll offset). The first is centred at the first height,
    unrolled; the rest draw uniformly over the jitter disc and the roll range, heights cycled."""
    views = []
    for v in range(n):
        if v == 0:
            views.append((0.0, 0.0, clearances_m[0], 0.0))
            continue
        r, a = jitter_m * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        views.append((r * math.cos(a), r * math.sin(a), clearances_m[v % len(clearances_m)], rng.uniform(-roll_deg, roll_deg)))
    return views


def overview(image, tiles, intrinsics, T_cam_to_base):
    """The RealSense frame with each tile numbered and named, as listed."""
    from mahjong.layout import table_to_pixel
    image = image.copy()
    for i, t in enumerate(tiles, 1):
        (u, v), = table_to_pixel([(*t["xy"], config.grasp_z(*t["xy"]))], intrinsics, T_cam_to_base)
        colour = (120, 120, 120) if t.get("skip") else (0, 200, 0)
        cv2.circle(image, (int(u), int(v)), 14, colour, 1)
        cv2.putText(image, f"{i} {t['tile_name']}", (int(u) - 14, int(v) - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    return image


def confirm_names(realsense, ask, show):
    """Scan, list, and let the person fix the names. Returns tiles to capture (left to right), or None to quit."""
    while True:
        tiles = [{**t, "detected_as": t["tile_name"]} for t in realsense.detect_tiles()]
        while True:
            print(f"\n{len(tiles)} tiles, left to right:")
            for i, t in enumerate(tiles, 1):
                note = " (left out)" if t.get("skip") else (f" (was {t['renamed_from']})" if t.get("renamed_from") else "")
                print(f"  {i:>2} {t['tile_name']:<3} {t['confidence']:.2f}{note}")
            if show:
                cv2.imshow(WINDOW, overview(realsense.snapshot(), tiles, realsense.intrinsics, realsense.T_cam_to_base))
                cv2.waitKey(1)
            action, result = parse_answer(ask("Names right? Enter to start, 3=2C rename, -3 leave out, r rescan, "
                                              "q quit: "), tiles)
            if action == "error":
                print(result)
            elif action == "edit":
                tiles = result
            elif action == "rescan":
                break
            elif action == "quit":
                return None
            else:
                return [t for t in tiles if not t.get("skip")]


def empty_spots(tiles, spots, clear_m=EMPTY_CLEAR_M):
    """The layout's spots with no scanned tile within clear_m: [{"tile_name": "empty", "id", "xy", ...}], in layout order."""
    return [{"tile_name": back_check.EMPTY, "id": s["id"], "xy": list(s["xy"]), "detected_as": None, "confidence": None}
            for s in spots if not any(math.dist(t["xy"], s["xy"]) <= clear_m for t in tiles)]


def confirm_empty(realsense, spots, ask, show):
    """Scan, list the bare spots, and let the person leave some out. Returns the spots to capture, or None to quit."""
    if not perception.sees_backs(realsense):
        print("The detector's weights see no backs: a spot with a back on it is listed as bare. Check the list by eye.")
    while True:
        tiles = list(realsense.detect_tiles()) + list(getattr(realsense, "faint", None) or [])
        bare = empty_spots(tiles, spots)
        while True:
            print(f"\n{len(bare)} bare spots of {len(spots)}: "
                  + " ".join(s["id"] + (" (left out)" if s.get("skip") else "") for s in bare))
            if show:
                ids = {s["id"] for s in bare if not s.get("skip")}
                cv2.imshow(WINDOW, layout.draw(realsense.snapshot(), [s for s in spots if s["id"] in ids],
                                               realsense.intrinsics, realsense.T_cam_to_base))
                cv2.waitKey(1)
            action, result = parse_answer(ask("Bare spots right? Enter to start, -S3 leave one out, r rescan, q quit: "),
                                          bare)
            if action == "edit" and any(s.get("renamed_from") for s in result):
                action, result = "error", "a bare spot has no name to change: -S3 leaves one out"
            if action == "error":
                print(result)
            elif action == "edit":
                bare = result
            elif action == "rescan":
                break
            elif action == "quit":
                return None
            else:
                return [s for s in bare if not s.get("skip")]


def roll_to(arm, offset_deg):
    """Turn the wrist by offset_deg from the hover's roll, kept ROLL_MARGIN_DEG inside its limits. Returns
    the offset applied."""
    if not offset_deg:
        return 0.0
    command = arm.commanded()
    low, high = arm.joint_limits()["wrist_roll"]
    target = min(max(command["wrist_roll"] + offset_deg, low + ROLL_MARGIN_DEG), high - ROLL_MARGIN_DEG)
    arm.move_to({"wrist_roll": target}, ADJUST_TIME_S)
    arm.sleep(SETTLE_S)
    return target - command["wrist_roll"]


def capture_tile(arm, camera, session, correction, tile, views, set_n, tile_n, n_frames, show, rng, jitter_m,
                 roll_deg, root="wrist"):
    """Every planned view of one tile (or bare spot, under root wrist_empty/); returns (views saved, the camera: reopened
    if it was lost)."""
    folder = session.dir / root / tile["tile_name"]
    folder.mkdir(parents=True, exist_ok=True)
    x, y = tile["xy"]
    hx, hy = correction.apply(x, y) if correction else (x, y)
    saved = 0
    for v, (dx, dy, clearance, roll) in enumerate(views):
        result = None
        for attempt in range(RETRIES + 1):
            arm.lift()
            result = arm.hover(hx + dx, hy + dy, clearance + (config.HOVER_OFFSET_M - config.HOVER_CLEARANCE_M),
                               clearance, config.HOVER_GRIPPER, log=lambda *_: None)
            if result is not None or v == 0:
                break
            r, a = jitter_m * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)    # out of reach: draw again
            dx, dy = r * math.cos(a), r * math.sin(a)
        if result is None:
            print(f"    view {v + 1}: out of reach, skipped")
            session.event("wrist_view_skipped", tile=tile["tile_name"], set=set_n, tile_n=tile_n, view=v,
                          reason="unreachable", detection_xy=[x, y])
            continue
        applied = roll_to(arm, roll)
        frames = grab_frames(camera, n_frames)
        if len(frames) < n_frames:              # lost (camera.LOST_S without a frame): reopen, try the view again
            camera = hardware.reopen_wrist_camera(camera)
            session.event("wrist_camera_reopened", tile=tile["tile_name"], set=set_n, tile_n=tile_n, view=v)
            frames = grab_frames(camera, n_frames)
        files = []
        for k, frame in enumerate(frames):
            name = f"{set_n:02d}_{tile.get('id') or f'{tile_n:02d}'}_v{v:02d}_f{k}.jpg"
            cv2.imwrite(str(folder / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            files.append(name)
            if show:
                cv2.imshow("wrist", frame)
                cv2.waitKey(1)
        if not files:
            print(f"    view {v + 1}: no wrist frames even after reopening the camera, skipped")
            session.event("wrist_view_skipped", tile=tile["tile_name"], set=set_n, tile_n=tile_n, view=v,
                          reason="wrist camera lost", detection_xy=[x, y])
            continue
        saved += 1
        session.event("wrist_view", tile=tile["tile_name"], detected_as=tile["detected_as"],
                      confidence=tile["confidence"], set=set_n, tile_n=tile_n, view=v, detection_xy=[x, y],
                      hover_xy=result["hover_xy"], hover_jitter_xy=[dx, dy], clearance_m=result["clearance_m"],
                      target_clearance_m=clearance, roll_offset_deg=applied, wrist_roll=arm.commanded()["wrist_roll"],
                      spot=tile.get("id"), files=[f"{root}/{tile['tile_name']}/{f}" for f in files])
    return saved, camera


def views_per_name(sessions_dir=None, recordings_dir=None, review=None):
    """Wrist views at the hover per tile name, by where they came from: {"wrist_capture", "bench",
    "recordings": Counter}. The first two are mahjong.wrist_photos' photos as reviewed (wrist_review.py:
    a photo marked wrong counts under its right name, or not at all); a recorded episode starts at the
    hover too (its episode log names the tile). Simulated sessions don't count."""
    sources = {**wrist_photos.counts(sessions_dir or config.SESSIONS, review), "recordings": Counter()}
    if recordings_dir is None:
        try:
            from lerobot.utils.constants import HF_LEROBOT_HOME
            recordings_dir = Path(HF_LEROBOT_HOME)
        except ImportError:
            recordings_dir = None
    if recordings_dir is not None:
        from mahjong.dataset import EPISODE_LOGS
        for log in sorted(p for name in EPISODE_LOGS for p in Path(recordings_dir).glob(f"*/*/{name}")):
            for record in read_jsonl(log):
                sources["recordings"][record.get("tile_name")] += 1
    return sources


def report(sources, target):
    """Print each name's views (all sources) and which are below target; returns those."""
    counts = sum(sources.values(), Counter())
    print("Wrist views at the hover per name (" + ", ".join(f"{k} {sum(c.values())}" for k, c in sources.items()) + "):")
    for suit in ("B", "C", "D", "F"):
        names = [n for n in detector.TILE_NAMES if n.endswith(suit) and len(n) == 2 and n[0].isdigit()]
        print("  " + "  ".join(f"{n} {counts[n]:>3}" for n in names))
    print("  " + "  ".join(f"{n} {counts[n]:>3}" for n in ("EW", "SW", "WW", "NW", "RD", "GD", "WD")))
    short = [n for n in detector.TILE_NAMES if counts[n] < target]
    print(f"Below {target}: {', '.join(short)}" if short else f"Every name has {target} views or more.")
    return short


def notify(message, sim=False):
    """Ring the terminal and, where there is one, pop up a desktop notification."""
    print(f"\a\n{'=' * 60}\n{message}\n{'=' * 60}")
    if not sim and shutil.which("notify-send"):
        try:
            subprocess.run(["notify-send", "-u", "critical", "Wrist capture", message], timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            pass


def main(argv=None, ask=None):
    args = parse_args(argv)
    ask = ask or input
    clearances = [float(c) / 1000 for c in args.clearances_mm.split(",")]
    rng = random.Random(args.seed)
    spots = layout.load(args.empty)["spots"] if args.empty else None      # an unknown layout stops before the arm moves
    if args.sim:
        from mahjong import sim
        devices = sim.connect()
    else:
        devices = hardware.connect(["arm", "wrist_camera", "realsense"])
    arm, camera, realsense = devices["arm"], devices["wrist_camera"], devices["realsense"]
    session = Session("wrist", ask_setup=not args.sim, sim=args.sim)
    session.event("wrist_capture_start", views=args.views, jitter_mm=args.jitter_mm, clearances_m=clearances,
                  roll_deg=args.roll_deg, frames=args.frames, seed=args.seed, empty_layout=args.empty)
    if arm.at_connect:
        session.event("connect", **arm.at_connect)
    correction = HoverCorrection.load()
    print(f"Hover correction: {correction.summary() if correction else 'off'}")
    show = not (args.no_view or args.sim)
    if show:       # windows stay answering through the arm's moves and while a question waits
        arm.sleep, ask = pumping_sleep, ask_pumping(ask)
    set_n = 0
    try:
        while args.sets is None or set_n < args.sets:
            arm.rest()
            tiles = confirm_empty(realsense, spots, ask, show) if args.empty else confirm_names(realsense, ask, show)
            if tiles is None:
                break
            if not tiles:
                print("No bare spots to capture." if args.empty else "No tiles to capture.")
                continue
            set_n += 1
            session.event("wrist_set", set=set_n, tiles=[{k: t.get(k) for k in ("tile_name", "detected_as", "xy",
                                                                                  "confidence", "id")} for t in tiles])
            done = Counter()
            for i, tile in enumerate(tiles, 1):
                key = tile.get("id") or tile["tile_name"]
                print(f"  {'spot' if args.empty else 'tile'} {i}/{len(tiles)}: {key}")
                views = plan_views(args.views, args.jitter_mm / 1000, clearances, args.roll_deg, rng)
                saved, camera = capture_tile(arm, camera, session, correction, tile, views, set_n, i, args.frames,
                                             show, rng, args.jitter_mm / 1000, args.roll_deg,
                                             root=back_check.EMPTY_FOLDER if args.empty else "wrist")
                done[key] += saved
            arm.rest()
            short_of = [t.get("id") or t["tile_name"] for t in tiles if done[t.get("id") or t["tile_name"]] < args.views]
            if args.empty:
                print("Back check photos over all sessions: "
                      + back_check.summary([] if args.sim else back_check.photos(session.dir.parent)))
            else:
                report(views_per_name(session.dir.parent) if not args.sim else {"this set": done}, args.target)
            notify(f"Set {set_n} done: {sum(done.values())} views of {len(tiles)} {'spots' if args.empty else 'tiles'}"
                   + (f" (fewer than {args.views} for {', '.join(sorted(set(short_of)))})" if short_of else "")
                   + (". Move the tiles so other spots are bare." if args.empty else ". Put the next set of tiles down."),
                   sim=args.sim)
            session.event("wrist_set_done", set=set_n, views=dict(done), short_of=short_of)
            if (args.sets is None or set_n < args.sets) and \
                    ask("Swap in the next set of tiles, then Enter to scan (q to finish): ").strip().lower() == "q":
                break
    except KeyboardInterrupt:
        print("Interrupted.")
        session.intervention("interrupted", command="wrist_capture")
    finally:
        arm.rest()
        if show:
            cv2.destroyAllWindows()
        session.close()
        if not args.sim:
            camera.release()
            realsense.close()
            arm.disconnect()


if __name__ == "__main__":
    main()
