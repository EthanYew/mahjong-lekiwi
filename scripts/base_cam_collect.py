"""The arm collects base-camera photos of face-down tiles (docs/face_down_plan.md, tests 1b and 2): pick up a tile a person laid
face down on a layout spot, hold it up to the base camera at the taught poses, save frames, set it back down.

    python scripts/base_cam_collect.py --empty                      # first: visit the saved poses with NOTHING in the jaws
    python scripts/base_cam_collect.py                              # collect what's short, 7 default tiles, 6 holds per way up
    python scripts/base_cam_collect.py --tiles 1B,EW,7C --spots S5,S9 --rounds 3
    python scripts/base_cam_collect.py --sim                        # simulated arm and cameras, no hardware

`--empty` is the safety pass: for each pose in data/camera_poses.json (`base_view_*`) it asks, moves there, and saves a base
and a wrist frame to the session's `dry_run/`, with the pose's height above the table, its reach and its joint margins printed
(a pose with the tip under 30mm up is flagged). Look at the sheet before putting a tile in play: are the jaws in the base
view, does the arm clear the base on the way? A tile is never in the jaws.

The collecting run reads what the hold-up sessions already hold (by hand and by arm), works out which tiles are furthest short
of `--target-holds`, and lays them in groups, one tile per `--spots` spot. Each round the arm picks every spot once (pick, rest,
every pose with a small random nudge, rest, set it back down, `--frames` frames at each pose), so holds of different tiles
alternate; after `--rounds` rounds every tile moves one spot along and is laid the other way. Collected a tile at a time, its
holds came in a row from one spot, and the scene named the tile better than its face did (2026-11-06: 99% with the tile
blacked out; `base_cam_train.py`'s control). A tile is laid one of two ways: as the layout's face-up tile would lie (flipped
sideways about its long edge, so the top of its picture stays at the same end), or that turned 180 degrees on the mat. The arm
can't turn a tile, so that is what decides the way up the camera sees, and which is which depends on the poses: after the first
hold of the session it shows you the photo and asks whether the picture reads right way up (`upright`) or upside down
(`turned`), and labels every later hold from that. A live RealSense window
shows where to lay each tile: the spot's outline, the tile's name and the end where the picture's top goes (TOP; `--layout-top`
says which end of a spot a face-up layout tile's picture points to). Nothing turns green, since the detector can't see a back. A miss, a drop or an unreachable spot stops that tile and asks you, since nothing can
see where a face-down tile is. Esc during a pickup stops the policy. Photos go where the hand-held ones go
(`sessions/<time>_holdup/photos/<NAME>/`) with `holdup_shot` events, so `holdup_score.py` and `base_cam_train.py` read them.
Run `setup_check.py` first, and from the main checkout (sessions are per-checkout).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import bench
from mahjong import camera_poses, config, base_cam, base_cam_collect, kinematics, layout, privacy
from mahjong.camera import grab_frames
from mahjong.hover_correction import HoverCorrection
from mahjong.return_correction import ReturnCorrection
from mahjong.session import Session
from mahjong.terminal_keys import TerminalKeys, new_events

LOW_POSE_M = 0.030          # the plan's floor for a pose's tip above the table (docs/face_down_plan.md, "What the base camera sees")
STANDARD_SETUP = config.DATA / "standard_setup.json"


def default_layout():
    return json.loads(STANDARD_SETUP.read_text())["layout"] if STANDARD_SETUP.exists() else "even-hand14"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--empty", action="store_true", help="the safety pass: visit the saved poses with nothing in the jaws, no policy")
    p.add_argument("--tiles", default=",".join(base_cam.DEFAULT_TILES), help="tiles to collect, comma separated (default: %(default)s)")
    p.add_argument("--target-holds", type=int, default=base_cam_collect.DEFAULT_TARGET_HOLDS,
                   help="holds per tile and way up to reach, counting every hold-up session (default %(default)s)")
    p.add_argument("--rounds", type=int, default=2,
                   help="at most this many pickups of each tile before the tiles move one spot along (default %(default)s): "
                        "each sets the tile down by memory")
    p.add_argument("--layout", default=default_layout(), help="the layout whose spots you lay tiles on (default: the standard setup's)")
    p.add_argument("--spots", default="S5,S7,S9",
                   help="spot ids: a tile on each at once, picked in turn (default %(default)s: S8 missed every pickup on "
                        "2026-11-05). More spots, more tiles interleaved")
    p.add_argument("--max-misses", type=int, default=2,
                   help="misses in a row on a spot before the tile is given up for this run (default %(default)s)")
    p.add_argument("--poses", default=None, help="saved poses to visit, comma separated (default: every base_view_N)")
    p.add_argument("--frames", type=int, default=3, help="frames saved at each pose (default %(default)s)")
    p.add_argument("--settle-s", type=float, default=1.0, help="seconds at a pose before its frames, for the tile to stop swinging (default %(default)s)")
    p.add_argument("--gap-s", type=float, default=0.25, help="seconds between a pose's frames (default %(default)s)")
    p.add_argument("--jitter-deg", type=float, default=base_cam_collect.DEFAULT_JITTER_DEG,
                   help="each pose is nudged by up to this many degrees of pan, wrist flex and wrist roll; 0 = exactly as taught (default %(default)s)")
    p.add_argument("--seed", type=int, default=0, help="seed of the nudges (default %(default)s)")
    p.add_argument("--layout-top", choices=("far", "near"), default="far",
                   help="which end of a spot the top of a face-up layout tile's picture points to, far = away from the robot "
                        "(default %(default)s); the window marks it TOP for the as-layout laying, the other end for rotated")
    p.add_argument("--no-view", action="store_true", help="no RealSense window: the terminal prompt only (--sim never opens one)")
    p.add_argument("--hover-lower-mm", type=float, default=None,
                   help="hover this many mm lower than the policy's own, as bench.py --hover-lower-mm (for a spot where it closes high)")
    p.add_argument("--policy", default=None, help="policy id (default: the registry's default)")
    p.add_argument("--sim", action="store_true", help="simulated arm, cameras and policy: no hardware")
    return p.parse_args(argv)


class Rig:
    """The arm, its cameras, the policy and the corrections: what picking a tile at a spot and showing it needs."""

    def __init__(self, args, spots, pose_list):
        self.args, self.spots, self.pose_list = args, spots, pose_list
        self.realsense = None           # opened only to show where to lay tiles (not --sim, --no-view or --empty)
        if args.sim:
            from mahjong import sim
            tiles = [("5C", *s["xy"]) for s in spots]
            devices = sim.connect(tiles=tiles, wrist_view=True)
            self.arm, self.wrist, self.mat = devices["arm"], devices["wrist_camera"], devices["realsense"]
            self.base = sim.SimBaseCamera()
            self.policy = sim.SimPolicy(mat=self.mat, policy_id=args.policy or "sim")
        else:
            from mahjong import hardware
            from mahjong.policy import Policy
            self.policy = Policy(args.policy)             # before connecting, so a mistyped id fails fast
            guided = not (args.no_view or args.empty)
            devices = hardware.connect(["arm", "wrist_camera", "base_camera"] + (["realsense"] if guided else []))
            self.arm, self.wrist, self.base = devices["arm"], devices["wrist_camera"], devices["base_camera"]
            self.realsense = devices.get("realsense")
        self.events = new_events()
        self.terminal = TerminalKeys(self.events)
        self.abort = bench.Abort(self.events)
        self.correction = HoverCorrection.load()
        self.returns = None
        self.session = None
        self.view_map = None            # {laying: upright / turned}, learned from the person's answer on the session's first photo

    def open_session(self, kind, **setup):
        self.session = Session(kind, ask_setup=False, sim=self.args.sim, policy=self.policy.id, layout=self.args.layout, **setup)
        path = self.session.dir / "return_corrections.jsonl" if self.args.sim else config.RETURN_CORRECTIONS
        self.returns = ReturnCorrection.load(path)     # simulated returns land where they're set down: the real model doesn't apply
        if self.arm.at_connect:
            self.session.event("connect", **self.arm.at_connect)

    def close(self):
        holding = self.arm.holding()[0]
        try:
            self.arm.rest()
        finally:
            self.arm.disconnect()
            self.wrist.release()
            self.base.release()
            if self.realsense is not None:
                self.realsense.close()
        if holding:
            print("WARNING: the jaws still read as holding a tile. Take it from the arm before the next run.")

    # --- one hold ---

    def pick(self, name, spot, raise_m=0.0, back_m=0.0):
        """Hover over the spot (the layout's xy stands for a detection: the detector can't see a back) and run the policy.
        raise_m: a hover this much higher, for a tile on top of another; back_m: this much nearer the robot along x (wall_test.py).
        Returns {"hover", "run", "attempt", "holding", ...} or {"unreachable": True}. The arm is left holding at rest."""
        x, y = spot["xy"]
        hx, hy = self.correction.apply(x, y)
        hx -= back_m
        h = dict(self.policy.hover)
        lower = (self.args.hover_lower_mm or 0.0) / 1000 - raise_m      # bench.py --hover-lower-mm: offset and clearance together
        h["offset_m"] -= lower
        h["clearance_m"] -= lower
        attempt = self.session.attempt_dir(name)
        self.terminal.start()
        try:
            hover = self.arm.hover(hx, hy, h["offset_m"], h["clearance_m"], h["gripper"])
            if hover is None:
                return {"unreachable": True, "attempt": attempt}
            self.abort.clear()
            run = self.policy.run(self.arm, self.wrist, attempt, abort=self.abort)
            squeezed = self.arm.squeeze()
        finally:
            self.terminal.stop()
        self.arm.rest()
        holding, command, measured = self.arm.holding()
        return {"hover": hover, "run": run, "attempt": attempt, "holding": holding, "squeezed_from": squeezed,
                "gripper_command": command, "gripper_measured": measured}

    def show(self, rng, still_held=None):
        """Visit every pose with the tile in the jaws; returns (frames, which pose each is from, "" or why it stopped).
        Between poses the arm is checked: a tile that slipped ends the hold. still_held: that check, for a grip Arm.holding can't see."""
        still_held = still_held or (lambda: self.arm.holding()[0])
        frames, from_pose = [], []
        limits = self.arm.joint_limits()
        for name in self.pose_list:
            pose = base_cam_collect.jitter(camera_poses.get(name)["pose"], rng, self.args.jitter_deg)
            pose = {j: min(max(v, limits[j][0]), limits[j][1]) for j, v in pose.items()}     # a nudge never passes a joint's limit
            self.arm.go_to_pose(pose)
            self.arm.sleep(self.args.settle_s)
            if not still_held():
                return frames, from_pose, f"the tile was lost at {name}"
            got = []
            for i in range(self.args.frames):
                if i:
                    self.arm.sleep(self.args.gap_s)
                got += grab_frames(self.base, 1)
            if not got:
                return frames, from_pose, "the base camera gave no frames"
            frames += got
            from_pose += [name] * len(got)
        return frames, from_pose, ""

    def put_back(self, spot, pick):
        """Set the tile down at the spot, aimed as a return is (the spot is its home and its detection). None if out of reach."""
        home = [float(v) for v in spot["xy"]]
        x, y = self.returns.target(home, home, pick["run"]["grasp_xy"] or pick["hover"]["hover_xy"])
        self.arm.rest()
        return self.arm.place(x, y, pick["hover"]["offset_m"], grasp_clearance_m=pick["run"].get("grasp_clearance_m")), (x, y)


GUIDE_WINDOW = "lay the tile (RealSense view)"


def close_windows():
    """Close every OpenCV window now: with the arm moving nothing pumps them, and the desktop marks them 'not responding'."""
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def guide(realsense, placement, spot, sign, show=None, wait_key=None, text=None, overlay=None):
    """A live RealSense window until the tile is laid: the spot's outline, the end where the picture's top goes, the tile's name.
    Returns "" (laid, go), "s" or "q". The caller closes the window before the arm moves (close_windows): nothing pumps it then.
    `text` replaces the default title; `overlay(frame)` draws on each frame (wall_test.py's picture of what to lay)."""
    show = show or (lambda image: cv2.imshow(GUIDE_WINDOW, image))
    wait_key = wait_key or (lambda: cv2.waitKey(30) & 0xFF)
    text = text or (f"Lay {placement['tile']} FACE DOWN on {spot['id']}"
                    + (", then turn it 180" if placement["laying"] == "rotated" else ""))
    while True:
        out = base_cam_collect.guide_frame(realsense.snapshot(), spot, realsense.intrinsics, realsense.T_cam_to_base, text, sign)
        if overlay is not None:
            overlay(out)
        show(out)
        key = wait_key()
        answer = {13: "", 10: "", 32: "", ord("s"): "s", ord("q"): "q", 27: "q"}.get(key)
        if answer is not None:
            return answer


def put_back_by_hand(rig, placement, spot, ask, why, prompt):
    """The tile has to go back on its spot by hand (a miss, a drop): the same window as when it was first laid, same TOP end,
    so it goes back as it was. Returns "" (back, go on), "s" or "q"; the terminal prompt `prompt` without the RealSense."""
    if rig.realsense is None:
        return ask(prompt).strip().lower()
    print(f"{why}\n  (the RealSense window shows where: Space/Enter when it is back, s skip this tile, q quit)")
    sign = base_cam_collect.top_end_sign(placement["laying"], rig.args.layout_top)
    answer = guide(rig.realsense, placement, spot, sign,
                   text=f"{why}: put {placement['tile']} back FACE DOWN on {spot['id']}, TOP end as marked")
    close_windows()
    return answer


def ask_view(rig, laying, frame, ask):
    """Once per session: show the first photo and ask what the camera sees, which fixes upright / turned for both layings.
    Returns the way up this laying shows, or None if the person quit."""
    path = rig.session.dir / "view_check.jpg"
    cv2.imwrite(str(path), privacy.blur_faces(frame)[0])        # the file is blurred; the window below is your own screen
    question = "Does the tile's picture read right way up (u) or upside down (d) in this photo? (q = quit)"
    in_window = not rig.args.sim and not rig.args.no_view
    if in_window:
        shown = frame.copy()
        cv2.putText(shown, question, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow("base camera: the first hold", shown)
    print(f"First photo of the session: {path}")
    while True:
        if in_window:           # keys in the window: nothing else pumps it while the terminal waits
            key = cv2.waitKey(100) & 0xFF
            answer = chr(key) if key != 255 else ""
        else:
            answer = ask(question + ": ").strip().lower()
        if answer in ("u", "d"):
            break
        if answer in ("q", "\x1b"):
            close_windows()
            return None
    close_windows()
    rig.view_map = base_cam_collect.learn_view(laying, "upright" if answer == "u" else "turned")
    rig.session.event("holdup_view_map", view=rig.view_map, answered=answer, frame=path.name)
    print(f"So {laying} shows {rig.view_map[laying]} and the other laying shows "
          f"{rig.view_map[base_cam_collect.LAYINGS[1 - base_cam_collect.LAYINGS.index(laying)]]}.")
    return rig.view_map[laying]


def save_hold(rig, counters, pick, spot, tile, kind, frames, from_pose):
    session = rig.session
    hold = counters["holds"][(tile, kind)] = counters["holds"].get((tile, kind), 0) + 1
    counters["shots"] += 1
    files = base_cam.save_shot(session.dir, {"tile": tile, "kind": kind, "hold": hold}, frames)
    session.event("holdup_shot", **base_cam_collect.shot_event(
        tile, kind, hold, files, from_pose, shot=counters["shots"], spot=spot["id"], attempt_dir=pick["attempt"].name,
        grasp_clearance_m=pick["run"].get("grasp_clearance_m"), jitter_deg=rig.args.jitter_deg))
    print(f"{tile} {kind} hold {hold}: {len(frames)} frames from {len(set(from_pose))} pose(s) saved.")


def do_placement(rig, rng, placement, spot, ask, counters):
    """Pick the laid tile up `holds` times, showing it each time. Returns "ok", "skip" or "quit"."""
    tile, laying = placement["tile"], placement["laying"]
    session = rig.session
    done = misses = 0
    while done < placement["holds"]:
        pick = rig.pick(tile, spot)
        if pick.get("unreachable"):
            session.event("holdup_pick", tile=tile, spot=spot["id"], success=False, reason="unreachable")
            print(f"{spot['id']} is out of the hover's reach: skipping this tile.")
            return "skip"
        run = pick["run"]
        session.event("holdup_pick", tile=tile, spot=spot["id"], spot_xy=spot["xy"], attempt_dir=pick["attempt"].name,
                      success=pick["holding"], stop_reason=run["stop_reason"], grasp_clearance_m=run.get("grasp_clearance_m"),
                      gripper_command=pick["gripper_command"], gripper_measured=pick["gripper_measured"],
                      policy=rig.policy.id, hover_clearance_m=pick["hover"]["clearance_m"])
        if run["stop_reason"] == "camera" and not pick["holding"]:
            from mahjong import hardware
            rig.wrist = hardware.reopen_wrist_camera(rig.wrist)
            print("The wrist camera dropped out and was reopened: trying this pickup again.")
            continue
        if not pick["holding"]:
            misses += 1
            if misses >= rig.args.max_misses:       # a spot the policy can't pick from stalls the whole run (S8, 2026-11-05)
                session.event("holdup_gave_up", tile=tile, spot=spot["id"], misses=misses)
                print(f"{tile}: missed {misses} times in a row on {spot['id']}: giving this tile up for this run. Take it off "
                      f"{spot['id']}. If the spot keeps failing, leave it out with --spots.")
                return "skip"
            answer = put_back_by_hand(
                rig, placement, spot, ask, f"{tile}: missed (stop: {run['stop_reason']})",
                f"{tile}: missed (stop: {run['stop_reason']}). Check the tile on {spot['id']} (face down, where it was). "
                "Enter = try again, s = skip this tile, q = quit: ")
            if answer == "q":
                return "quit"
            if answer == "s":
                return "skip"
            continue
        misses = 0
        frames, from_pose, problem = rig.show(rng)
        holding = placed = asked = None
        if problem:
            rig.arm.rest()
            session.event("holdup_stopped", tile=tile, reason=problem)
            holding = rig.arm.holding()[0]
            print(f"Stopped: {problem}." + (" The jaws still read as holding it." if holding else " Find the tile before going on."))
        else:
            placed, asked = rig.put_back(spot, pick)        # before the question, so the arm isn't left holding while you look
            if placed is None:
                session.event("holdup_stopped", tile=tile, reason="return out of reach")
                print("Couldn't set the tile back down (out of reach): it is still in the jaws.")
        if frames:
            kind = rig.view_map[laying] if rig.view_map else ask_view(rig, laying, frames[0], ask)
            if kind is None:
                return "quit"
            save_hold(rig, counters, pick, spot, tile, kind, frames, from_pose)
        if problem:
            answer = "q" if holding else put_back_by_hand(
                rig, placement, spot, ask, f"Stopped: {problem}",
                "Enter = carry on once the tile is back face down on its spot, s = skip this tile, q = quit: ")
            if holding:
                return "quit"           # putting it down is the return below, not a recovery from here
            return "quit" if answer == "q" else "skip" if answer == "s" else "ok"
        if placed is None:
            return "quit"
        session.event("holdup_return", tile=tile, spot=spot["id"], asked_xy=list(asked), attempt_dir=pick["attempt"].name, **placed)
        done += 1
    return "ok"


def lay(rig, arrangement, spots, where, ask):
    """Asks for the arrangement's tiles to be laid face down, a spot at a time (the RealSense window, or the terminal).
    Returns [(spot, tile)] for the spots laid (s leaves a spot out of this arrangement), or None to quit."""
    laying = arrangement["laying"]
    print(f"[{where}] Lay these, {laying.replace('_', ' ')}: " + ", ".join(f"{t} on {s['id']}" for s, t in zip(spots, arrangement["tiles"]))
          + " (moving any tile already there)")
    laid = []
    for spot, tile in zip(spots, arrangement["tiles"]):
        text = f"[{where}] {base_cam_collect.lay_text(tile, laying, spot)}"
        if rig.realsense is not None:
            print(text + "\n  (the RealSense window shows where: Space/Enter when laid, s skip, q quit)")
            answer = guide(rig.realsense, {"tile": tile, "laying": laying}, spot,
                           base_cam_collect.top_end_sign(laying, rig.args.layout_top))
        else:
            answer = ask(text + ", then Enter (s = skip, q = quit): ").strip().lower()
        if answer == "q":
            close_windows()
            return None
        if answer != "s":
            laid.append((spot, tile))
    close_windows()
    return laid


def run_rounds(rig, rng, arrangement, laid, ask, counters):
    """Each round picks every laid spot once, so holds of different tiles alternate. A spot given up (misses, out of reach,
    skipped) sits out the rest of this arrangement. Returns "ok" or "quit"."""
    out = set()
    for _ in range(arrangement["rounds"]):
        for spot, tile in laid:
            if spot["id"] in out:
                continue
            result = do_placement(rig, rng, {"tile": tile, "laying": arrangement["laying"], "holds": 1}, spot, ask, counters)
            if result == "quit":
                return "quit"
            if result == "skip":
                out.add(spot["id"])
    return "ok"


def empty_pass(rig, ask):
    """Visit each saved pose with nothing in the jaws, one Enter at a time; save a base and a wrist frame at each."""
    out = rig.session.dir / "dry_run"
    out.mkdir(exist_ok=True)
    limits = rig.arm.joint_limits()
    print(f"{len(rig.pose_list)} pose(s). The jaws hold nothing; the gripper stays as it is.")
    tiles = []
    for name in rig.pose_list:
        entry = camera_poses.get(name)
        pose = entry["pose"]
        tip = kinematics.pincer_tip(pose)
        up = kinematics.tip_clearance(pose)
        reach = float(np.hypot(tip[0] - kinematics.SHOULDER_PAN_X, tip[1]))
        margin = min(min(pose[j] - limits[j][0], limits[j][1] - pose[j]) for j in config.ARM_JOINTS)
        flag = f"  <-- under {LOW_POSE_M * 1000:.0f}mm" if up < LOW_POSE_M else ""
        print(f"{name}: tip x {tip[0] * 100:+.1f} y {tip[1] * 100:+.1f} cm, {up * 1000:.0f}mm above the table, "
              f"{reach * 100:.0f}cm from the shoulder, nearest joint limit {margin:.0f} deg{flag}")
        answer = ask("  Enter = move there, s = skip, q = stop: ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            continue
        rig.arm.go_to_pose(pose)
        rig.arm.sleep(rig.args.settle_s)
        base, wrist = (grab_frames(c, 1) for c in (rig.base, rig.wrist))
        base = [privacy.blur_faces(frame)[0] for frame in base]     # the base camera sees the room
        for label, got in (("base", base), ("wrist", wrist)):
            if got:
                cv2.imwrite(str(out / f"{name}_{label}.jpg"), got[0])
        tiles.append(np.hstack([cv2.resize(g[0], (320, 240)) for g in (base, wrist) if g]) if base or wrist else None)
        rig.session.event("holdup_empty_pose", pose=name, tip_xyz_m=[float(v) for v in tip], tip_above_table_m=float(up),
                          reach_m=reach, joint_margin_deg=float(margin), base_frame=bool(base), wrist_frame=bool(wrist))
        rig.arm.rest()          # each pose is reached from rest, as in a run
    sheet = [t for t in tiles if t is not None]
    if sheet:
        cv2.imwrite(str(rig.session.dir / "dry_run_sheet.jpg"), np.vstack(sheet))
        print(f"Base (left) and wrist (right) frames: {out}, one sheet at {rig.session.dir / 'dry_run_sheet.jpg'}")


def main(argv=None, ask=input):
    args = parse_args(argv)
    privacy.require()           # the weights that blur faces in the base camera's photos: before any hardware
    if args.hover_lower_mm is not None and not -15 <= args.hover_lower_mm <= 10:
        raise SystemExit("--hover-lower-mm is from -15 (higher) to 10 (lower): the settle only goes 10mm under the hover's offset")
    spot_ids = [s.strip() for s in args.spots.split(",") if s.strip()]
    layout_spots = {s["id"]: s for s in layout.load(args.layout)["spots"]}
    unknown = [s for s in spot_ids if s not in layout_spots]
    if unknown:
        raise SystemExit(f"layout {args.layout} has no spot {', '.join(unknown)} (has: {', '.join(layout_spots)})")
    spots = [layout_spots[s] for s in spot_ids]
    names = [n.strip() for n in args.poses.split(",")] if args.poses else base_cam_collect.pose_names()
    for n in names:
        camera_poses.get(n)             # a wrong name fails before any hardware
    if not names:
        raise SystemExit(f"no {base_cam_collect.POSE_PREFIX}* poses in {config.CAMERA_POSES}: teach some with scripts/teleop_view.py")
    tiles = list(dict.fromkeys(t.strip().upper() for t in args.tiles.split(",") if t.strip()))
    if not args.empty and len(tiles) > 1 and len(spots) < 2:
        raise SystemExit("--spots needs two or more spots: with one, each tile's holds come in a row, and the scene names the tile")
    plan = [] if args.empty else base_cam_collect.arrangements(
        base_cam_collect.wanted(tiles, base_cam_collect.hold_counts(), args.target_holds), tiles, len(spots), args.rounds)
    if not args.empty and not plan:
        return print(f"Every tile in {', '.join(tiles)} already has {args.target_holds} holds each way up: nothing to collect.")
    rig = Rig(args, spots, names)
    rng = base_cam_collect.rng_for(args.seed)
    counters = {"shots": 0, "holds": {}}
    try:
        if args.empty:
            rig.open_session("holdup_dry")
            empty_pass(rig, ask)
            return rig.session.dir
        conditions = base_cam.ask_lighting(ask)
        rig.open_session("holdup")
        rig.session.conditions = conditions
        rig.session.event("holdup_conditions", conditions=conditions, shots=sum(len(a["tiles"]) * a["rounds"] for a in plan),
                          source="arm", poses=names, jitter_deg=args.jitter_deg, target_holds=args.target_holds,
                          spots=spot_ids, order="interleaved")
        print(f"To collect: {len(plan)} arrangements, the tiles one per spot ({', '.join(spot_ids)}), each picked in turn; "
              "between arrangements every tile moves one spot along and is laid the other way:")
        for i, a in enumerate(plan, 1):
            print(f"  {i}. {a['laying']}: " + ", ".join(f"{s} {t}" for s, t in zip(spot_ids, a["tiles"])) + f", {a['rounds']} round(s)")
        rig.arm.rest()
        for i, arrangement in enumerate(plan, 1):
            laid = lay(rig, arrangement, spots, f"{i}/{len(plan)}", ask)
            if laid is None:
                break
            rig.session.event("holdup_arrangement", n=i, laying=arrangement["laying"], rounds=arrangement["rounds"],
                              tiles={spot["id"]: tile for spot, tile in laid})
            if run_rounds(rig, rng, arrangement, laid, ask, counters) == "quit":
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        folder = rig.session.dir if rig.session else None
        rig.close()
        if rig.session:
            rig.session.event("holdup_done", shots_taken=counters["shots"])
            print(f"{counters['shots']} hold(s) saved in {folder}" + ("" if args.empty else
                  f"\nScore a reader trained on them: python scripts/base_cam_train.py, then python scripts/holdup_score.py for every result"))
            rig.session.close()
    return folder


if __name__ == "__main__":
    main()
