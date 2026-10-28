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

from mahjong import config, detector
from mahjong.arm import ADJUST_TIME_S, SETTLE_S
from mahjong.hover_correction import HoverCorrection
from mahjong.session import Session

WINDOW = "wrist capture"
ROLL_MARGIN_DEG = 5.0        # a rolled view stays this far inside the roll's command limits
RETRIES = 3                  # fresh jitter draws for a view whose hover is out of reach


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


def parse_answer(answer, tiles):
    """The names step: "" accept, "r" rescan, "q" quit, else edits "3=2C -5" (1-based). Returns
    ("accept"|"rescan"|"quit"|"edit", tiles after the edits) or ("error", message)."""
    answer = answer.strip()
    if answer.lower() in ("", "r", "q"):
        return {"": "accept", "r": "rescan", "q": "quit"}[answer.lower()], tiles
    tiles = [dict(t) for t in tiles]
    for part in answer.replace(",", " ").split():
        try:
            if part.startswith("-"):
                tiles[int(part[1:]) - 1]["skip"] = True
            else:
                n, name = part.split("=")
                name = name.strip().upper()
                if name not in detector.TILE_NAMES:
                    return "error", f"{name} isn't a tile name ({', '.join(detector.TILE_NAMES)})"
                tiles[int(n) - 1]["tile_name"] = name
                tiles[int(n) - 1]["renamed_from"] = tiles[int(n) - 1].get("renamed_from") or tiles[int(n) - 1]["detected_as"]
        except (ValueError, IndexError):
            return "error", f"can't read {part!r}: use e.g. 3=2C to rename tile 3, -3 to leave it out"
    return "edit", tiles


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
                 roll_deg):
    """Every planned view of one tile; returns how many were saved."""
    folder = session.dir / "wrist" / tile["tile_name"]
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
        camera.flush()
        files = []
        for k in range(n_frames):
            ok, frame = camera.read()
            if not ok:
                continue
            name = f"{set_n:02d}_{tile_n:02d}_v{v:02d}_f{k}.jpg"
            cv2.imwrite(str(folder / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            files.append(name)
            if show:
                cv2.imshow("wrist", frame)
                cv2.waitKey(1)
        if not files:
            print(f"    view {v + 1}: the wrist camera gave no frame")
            continue
        saved += 1
        session.event("wrist_view", tile=tile["tile_name"], detected_as=tile["detected_as"],
                      confidence=tile["confidence"], set=set_n, tile_n=tile_n, view=v, detection_xy=[x, y],
                      hover_xy=result["hover_xy"], hover_jitter_xy=[dx, dy], clearance_m=result["clearance_m"],
                      target_clearance_m=clearance, roll_offset_deg=applied, wrist_roll=arm.commanded()["wrist_roll"],
                      files=[f"wrist/{tile['tile_name']}/{f}" for f in files])
    return saved


def views_per_name(sessions_dir=None):
    """Views saved per tile name over every (not simulated) wrist session."""
    counts = Counter()
    for folder in sorted((sessions_dir or config.SESSIONS).glob("*_wrist")):
        if detector.is_sim(folder) or not (folder / "wrist").is_dir():
            continue
        for name_dir in (folder / "wrist").iterdir():
            counts[name_dir.name] += len(list(name_dir.glob("*_f0.jpg")))
    return counts


def report(counts, target):
    print("Views per name over all wrist sessions:")
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
    if args.sim:
        from mahjong.arm import Arm
        from mahjong.sim import SimCamera, SimPerception, SimRobot
        realsense = SimPerception()
        arm, camera = Arm(SimRobot(), sleep=lambda s: None), SimCamera()
        session = Session("wrist", ask_setup=False, sim=True)
    else:
        from mahjong import hardware
        devices = hardware.connect(["arm", "wrist_camera", "realsense"])
        arm, camera, realsense = devices["arm"], devices["wrist_camera"], devices["realsense"]
        session = Session("wrist", sim=False)
    session.event("wrist_capture_start", views=args.views, jitter_mm=args.jitter_mm, clearances_m=clearances,
                  roll_deg=args.roll_deg, frames=args.frames, seed=args.seed)
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
            tiles = confirm_names(realsense, ask, show)
            if tiles is None:
                break
            if not tiles:
                print("No tiles to capture.")
                continue
            set_n += 1
            session.event("wrist_set", set=set_n, tiles=[{k: t.get(k) for k in ("tile_name", "detected_as", "xy",
                                                                                  "confidence")} for t in tiles])
            done = Counter()
            for i, tile in enumerate(tiles, 1):
                print(f"  tile {i}/{len(tiles)}: {tile['tile_name']}")
                views = plan_views(args.views, args.jitter_mm / 1000, clearances, args.roll_deg, rng)
                done[tile["tile_name"]] += capture_tile(arm, camera, session, correction, tile, views, set_n, i,
                                                        args.frames, show, rng, args.jitter_mm / 1000, args.roll_deg)
            arm.rest()
            short_of = [t["tile_name"] for t in tiles if done[t["tile_name"]] < args.views]
            report(views_per_name(session.dir.parent) if not args.sim else done, args.target)
            notify(f"Set {set_n} done: {sum(done.values())} views of {len(tiles)} tiles"
                   + (f" (fewer than {args.views} for {', '.join(sorted(set(short_of)))})" if short_of else "")
                   + ". Put the next set of tiles down.", sim=args.sim)
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
