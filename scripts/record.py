"""Record pickup demonstrations: closed-loop hover over a chosen tile, then gamepad teleop.

Each episode: the arm rests, the RealSense detects tiles, you pick one, the arm hovers over it
(measured hover correction, settled from above to the clearance), and holds. → hands you the gamepad
and starts recording from exactly that pose; → again saves, ← discards, Esc ends. After a discard that
moved nothing, Enter at "Same tile, same hover again?" lifts and hovers over the same tile again, skipping
the trip to rest and the tile choice. A tile still in the jaws when an episode ends is set back down at its
home, where it was first seen this session (as bench.py's return, aimed with the return model), and a
RealSense scan logs where it landed, a sample for data/return_corrections.jsonl; then the arm rests.
--no-return drops it at rest instead.

The hover settings (offset, clearance, gripper opening) are written per episode to
<dataset>/episode_log.jsonl; make_policy.py stores them with the trained policy so it's deployed from
the same hover. Only the wrist camera is recorded (the policy's only input). Each record also carries
the session's setup answers (lighting etc.) and the episode's --slice, so offline evaluation can break
scores down by them.

With --layout NAME (taught with scripts/layout.py), the RealSense view shows where the next tile goes —
an outline that turns green once the tile is within 5mm / 6° — and the tile at that spot is picked
without asking. Each spot is recorded --repeats times in a row; the episode log stores the spot and
where the tile actually was.

Hover jitter (ROADMAP 7): --hover-jitter-mm R moves each episode's hover by a random offset within R mm
(uniform over the disc; --jitter-zero-fraction of episodes stay clean), so the demos have to steer to the
tile from wherever the hover lands, not repeat one move per mat position (what v2 learned). For an eval
slice, --hover-offsets-mm cycles a fixed list instead, so the set can be recorded again identically. An
episode keeps its offset through ← re-hovers and discards — rejecting the hard ones would bias the
dataset back toward the clean hover. Each episode log row has hover_jitter_xy and the unjittered target.

DAgger (ROADMAP 9): with --policy ID the policy drives from its own hover once → starts the episode, and
pressing A (the rightmost face button) hands it to you, from the pose the policy last commanded
(gripper held); sticks and other buttons do nothing until then. Every frame is recorded; the episode log row has the policy and takeover_frame,
and training (make_policy.py --add) keeps only your frames, as deltas from the hover like every other
episode. Record these into a dataset of their own.

    python scripts/record.py --dataset Yewy/mahjong_mat_pickup_v3 --task "Pick up the mahjong tile from the mat"
    python scripts/record.py --dataset Yewy/mahjong_mat_eval_v1 --slice new --layout eval_new --repeats 2
    python scripts/record.py --dataset Yewy/mahjong_mat_pickup_v3_jitter --hover-jitter-mm 6
    python scripts/record.py --dataset Yewy/mahjong_mat_eval_v1 --slice jitter --layout eval_new --repeats 2 \
        --hover-offsets-mm 6x,-6y
    python scripts/record.py --dataset Yewy/mahjong_mat_dagger_v3c --policy pickup_mat_cl_v3c
"""
import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import config, hardware, kinematics, layout
from mahjong.camera import LOST_S
from mahjong.dataset import EPISODE_LOG
from mahjong.gamepad import DEADZONE, draw_input, warn_if_at_reach, warn_off_centre
from mahjong.hover_correction import HoverCorrection, parse_offsets, random_offset
from mahjong.perception import choose_tile, drain_stdin
from mahjong.return_correction import Homes, ReturnCorrection, check_landing
from mahjong.session import Session, ask_one

JOINT_NAMES = [f"{j}.pos" for j in config.JOINTS]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},          # delta from episode start
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},                     # delta from episode start
    "observation.state_raw": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action_raw": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["h", "w", "c"]},
}
DISCARD_REASONS = ["bad demo", "knocked a tile", "hover was wrong", "other"]    # asked after ←, Enter skips
SAME_TILE_NO = {"knocked a tile", "hover was wrong"}      # discards after which a same-tile retry defaults to no
SLICES = ["train", "familiar", "new", "dim", "tilt", "jitter"]   # train: for training; the rest: eval-only (offline_eval.py)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, help="repo id to create or append to, e.g. Yewy/mahjong_mat_pickup_v3")
    p.add_argument("--task", default="Pick up the mahjong tile from the mat", help="task text stored with the episodes")
    p.add_argument("--episodes", type=int, default=200, help="stop after this many saved episodes")
    p.add_argument("--hover-offset-mm", type=float, default=config.HOVER_OFFSET_M * 1000,
                   help="pincer tip commanded this far above the grasp point (default %(default)s)")
    p.add_argument("--clearance-mm", type=float, default=config.HOVER_CLEARANCE_M * 1000,
                   help="...then settled until it measures this far above it (default %(default)s)")
    p.add_argument("--hover-gripper", type=float, default=config.HOVER_GRIPPER,
                   help="gripper opening at the hover (%%). Every dataset so far used 90 and spent ~3.5s narrowing "
                        "before descending; v2n's trimmed episodes start at 27.2 (record at that to compare with v2n)")
    p.add_argument("--stick-commit-s", type=float, default=0.1,
                   help="a gamepad stick moves the arm only once held this long, so flicks don't (default %(default)s)")
    p.add_argument("--deadzone", type=float, default=DEADZONE,
                   help="stick deadzone, fraction of full travel (default %(default)s, as collect_corrections: a resting "
                        "thumb past XLeRobot's 0.15 pulled the arm 87mm; lower it if gentle pushes don't register)")
    p.add_argument("--no-correction", action="store_true", help="hover over the raw detection")
    p.add_argument("--slice", choices=SLICES, default="train",
                   help="what these episodes are for: train, or an eval slice (familiar spots / new spots / dim light / "
                        "tilted tiles / jittered hovers) — eval datasets are never trained on")
    jitter = p.add_mutually_exclusive_group()
    jitter.add_argument("--hover-jitter-mm", type=float, default=0.0,
                        help="move each episode's hover by a random offset within this radius (mm), so the policy "
                             "learns to steer to the tile")
    jitter.add_argument("--hover-offsets-mm", type=parse_offsets, default=None,
                        help="cycle these hover offsets through the saved episodes instead, e.g. 0,6x,-6y (an eval "
                             "slice that can be recorded again identically)")
    p.add_argument("--jitter-zero-fraction", type=float, default=0.2,
                   help="with --hover-jitter-mm: share of episodes recorded from the unjittered hover")
    p.add_argument("--jitter-seed", type=int, default=None, help="with --hover-jitter-mm: random seed")
    p.add_argument("--layout", default=None, help="place tiles to this layout's spots (scripts/layout.py)")
    p.add_argument("--repeats", type=int, default=2, help="with --layout: episodes per spot")
    p.add_argument("--start-spot", default=None, help="with --layout: start at this spot id")
    p.add_argument("--device", default=None, help="gamepad evdev path")
    p.add_argument("--no-view", action="store_true", help="no wrist camera window")
    p.add_argument("--policy", default=None,
                   help="DAgger: this registered policy drives from its own hover (replacing --hover-offset-mm, "
                        "--clearance-mm, --hover-gripper) until you press A on the gamepad; only your frames are trained on")
    p.add_argument("--no-return", action="store_true",
                   help="don't set a held tile back down after an episode: it's released at rest, as before")
    return p.parse_args(argv)


def open_dataset(repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    root = HF_LEROBOT_HOME / repo_id
    info = root / "meta" / "info.json"
    if info.exists() and json.loads(info.read_text()).get("total_episodes", 0) == 0:
        shutil.rmtree(root)            # stub from a run interrupted before its first save
    if info.exists():
        existing = json.loads(info.read_text())["features"]
        missing = set(FEATURES) - set(existing)
        if missing:
            raise SystemExit(f"{repo_id} has a different layout (missing {sorted(missing)}); record into a new dataset")
        dataset = LeRobotDataset.resume(repo_id, root=root)
        print(f"Appending to {repo_id} ({dataset.meta.total_episodes} episodes)")
        return dataset
    print(f"Creating {repo_id}")
    return LeRobotDataset.create(repo_id=repo_id, fps=config.FPS, features=FEATURES, robot_type="so101_follower",
                                 use_videos=True)


def main(argv=None):
    args = parse_args(argv)
    from lerobot.common.control_utils import init_keyboard_listener
    hover = {"offset_m": args.hover_offset_mm / 1000, "clearance_m": args.clearance_mm / 1000, "gripper": args.hover_gripper}
    policy = None
    if args.policy:             # before connecting anything, so a typo fails fast
        if args.slice != "train":
            raise SystemExit("--policy records DAgger corrections, which are training data: leave --slice at train")
        from mahjong.policy import Policy, next_command
        policy = Policy(args.policy)
        hover = dict(policy.hover)
        print(f"DAgger: {policy.id} drives from its hover {hover}; press A on the gamepad to take over")
    spots, todo, ghost = [], [], None
    if args.layout:            # before connecting anything, so a typo fails fast
        spots = layout.load(args.layout)["spots"]
        todo = layout.schedule(spots, args.repeats, args.start_spot)
        ghost = layout.load_ghost(args.layout)
    devices = hardware.connect(["arm", "wrist_camera", "realsense", "gamepad"],
                               gamepad_kwargs={"device": args.device, "commit_s": args.stick_commit_s,
                                               "deadzone": args.deadzone})
    arm, camera, realsense, gamepad = (devices[k] for k in ("arm", "wrist_camera", "realsense", "gamepad"))
    print(f"Gamepad: {gamepad.name}")
    dataset = open_dataset(args.dataset)      # before the session folder: it refuses a dataset of another layout
    session = Session("record", dataset=args.dataset, task=args.task, hover=hover, stick_commit_s=args.stick_commit_s,
                      deadzone=args.deadzone,
                      slice=args.slice, layout=args.layout, hover_jitter_mm=args.hover_jitter_mm,
                      hover_offsets_m=args.hover_offsets_mm, jitter_zero_fraction=args.jitter_zero_fraction,
                      jitter_seed=args.jitter_seed, policy=args.policy)
    session.event("connect", **arm.at_connect)
    correction = None if args.no_correction else HoverCorrection.load()
    print(f"Hover correction: {correction.summary() if correction else 'off'}")
    returns, homes = None if args.no_return else ReturnCorrection.load(), Homes()
    if returns:
        print(f"Return correction: {returns.summary()}")
    listener, events = init_keyboard_listener()
    saved, dt = 0, 1.0 / config.FPS
    rng, offset = random.Random(args.jitter_seed), None     # offset: the next saved episode's, kept until it's saved
    retry = None               # (tile, placement) to hover over again after a discard, skipping rest
    try:
        while saved < args.episodes:
            for k in ("exit_early", "rerecord_episode", "stop_recording"):
                events[k] = False
            retrying = retry is not None
            if retrying:               # ← on a demo that moved nothing: straight back to the same hover (straight up first)
                tile, placement = retry
                retry = None
            else:
                rest_and_release(arm, hover)
                placement = {}
                if args.layout:
                    if not todo:
                        print(f"Every spot of layout {args.layout} recorded.")
                        break
                    spot = next(s for s in spots if s["id"] == todo[0])
                    print(f"Spot {spot['id']} ({len(todo)} episodes to go): {layout.PLACE_HELP}")
                    placed = layout.place(realsense.snapshot, spots, spot["id"], realsense.intrinsics,
                                          realsense.T_cam_to_base, ghost=ghost)
                    cv2.destroyWindow("layout")
                    for k in ("exit_early", "rerecord_episode", "stop_recording"):   # keys pressed while placing
                        events[k] = False
                    if placed["action"] == "quit":
                        break
                    if placed["action"] == "skip":
                        todo = [s for s in todo if s != spot["id"]]
                        continue
                    found = placed["found"]
                    tiles = realsense.detect_tiles()
                    tile = layout.nearest_tile(tiles, found["xy"] if found else spot["xy"])
                    if tile is None:
                        print(f"No tile detected at {spot['id']} — place it again.")
                        continue
                    if placed["status"] != "ok":
                        print(f"  (placed {placed['status']}, not within tolerance — logged)")
                    placement = {"layout": args.layout, "spot": spot["id"], "spot_xy": spot["xy"],
                                 "spot_yaw_deg": spot.get("yaw_deg", 0.0), "placement_status": placed["status"],
                                 "placed_xy": found["xy"] if found else None,
                                 "placed_yaw_deg": found["yaw_deg"] if found else None}
                else:
                    tiles = realsense.detect_tiles()
                    tile = choose_tile(tiles)
                if tile is None:
                    if input("No tile chosen. q to quit, Enter to scan again: ").strip().lower() == "q":
                        break
                    session.event("attempt_skipped", reason="cancelled" if tiles else "no_tiles")
                    continue
            x, y = tile["xy"]
            home = homes(tile, {"id": placement["spot"]} if placement else None)
            hx, hy = correction.apply(x, y) if correction else (x, y)
            if offset is None:
                offset = next_offset(args, rng, saved)
            if any(offset):
                print(f"  hover jittered by ({offset[0] * 1000:+.1f}, {offset[1] * 1000:+.1f})mm on purpose")
            result = arm.hover(hx + offset[0], hy + offset[1], hover["offset_m"], hover["clearance_m"], hover["gripper"])
            if result is None:
                print("Unreachable from the closed-loop hover — choose another tile.")
                session.event("attempt_skipped", reason="unreachable", tile=tile["tile_name"],
                              hover_xy=[hx + offset[0], hy + offset[1]], hover_jitter_xy=list(offset))
                if args.hover_jitter_mm:
                    offset = None          # a random draw may be what put it out of reach: draw again
                continue
            session.event("hover", tile=tile["tile_name"], detection_xy=[x, y], hover_jitter_xy=list(offset),
                          same_tile_retry=retrying, **result)
            warn_if_at_reach(*result["hover_xy"])

            print(f"Hovering. → {'START recording with ' + policy.id + ' driving' if policy else 'take control and START recording'}"
                  f", ← re-hover, Esc quit")
            while not events["exit_early"]:
                show(camera, args.no_view, "hovering: → to record")
                time.sleep(dt)
            if events["stop_recording"]:
                break
            if events["rerecord_episode"]:     # the hover looked wrong: worth knowing how often (ROADMAP 1, 3)
                session.intervention("hover_rejected", tile=tile["tile_name"], detection_xy=[x, y],
                                     hover_xy=result["hover_xy"], clearance_m=result["clearance_m"])
                continue
            events["exit_early"] = False

            if policy is None:
                gamepad.start(arm)
                warn_off_centre(gamepad, session, tile["tile_name"])
            else:
                policy.reset()
                camera.flush()
                limits, last_safe = arm.joint_limits(), arm.commanded()
                gamepad.arm_takeover()
            anchor = np.array([arm.measured()[j] for j in config.JOINTS], dtype=np.float32)
            print("RECORDING — → save, ← discard, Esc save nothing and quit"
                  + (f"; {policy.id} drives, A on the gamepad takes over" if policy else ""))
            frames, roll, actions, states, takeover = 0, RollTrace(), [], [], None
            started = time.time()
            failed_reads, camera_lost = 0, False
            while True:
                if events["exit_early"]:
                    if time.time() - started >= KEY_REPEAT_S or events["rerecord_episode"] or events["stop_recording"]:
                        break
                    events["exit_early"] = False      # the → that started it, counted twice: not a save
                t0 = time.time()
                if policy is not None and takeover is None and gamepad.takeover_pressed():
                    takeover = frames
                    gamepad.start(arm)          # from the policy's last command: no drop, the gripper held
                    warn_off_centre(gamepad, session, tile["tile_name"])
                    print(f"  t={frames / config.FPS:4.1f}s you have it")
                    session.intervention("takeover", tile=tile["tile_name"], frame=frames)
                if policy is not None and takeover is None:
                    ok, frame = camera.read()
                else:
                    command = gamepad.step(arm, dt)
                    ok, frame = camera.read()
                if not ok:             # an episode with a gap in its frames can't be trained on
                    failed_reads += 1
                    if failed_reads >= LOST_S * config.FPS:
                        print(f"  no wrist camera frames for {LOST_S:.0f}s — the episode is discarded")
                        camera_lost = events["rerecord_episode"] = True
                        break
                    continue
                failed_reads = 0
                state = np.array([arm.measured()[j] for j in config.JOINTS], dtype=np.float32)
                if policy is not None and takeover is None:
                    command, _ = next_command(policy, frame, state, anchor, limits, last_safe)
                    arm.send(command)
                    last_safe = command
                action = np.array([command[j] for j in config.JOINTS], dtype=np.float32)
                roll.add(command["wrist_roll"], float(state[ROLL]))
                actions.append(action)
                states.append(state)
                dataset.add_frame({"observation.state": state - anchor, "action": action - anchor,
                                   "observation.state_raw": state, "action_raw": action,
                                   "observation.images.wrist": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "task": args.task})
                frames += 1
                driver = "" if policy is None else (" POLICY drives: A takes over" if takeover is None else " YOU")
                yours = policy is None or takeover is not None
                show(camera, args.no_view, f"REC {frames / config.FPS:4.1f}s{driver}", frame,
                     getattr(gamepad, "last", None) if yours else None)
                time.sleep(max(0.0, dt - (time.time() - t0)))
            roll.report()
            fps = round(frames / max(time.time() - started, 1e-6), 1)
            if frames and fps < 0.9 * config.FPS:     # the policy and the gamepad both assume 30 steps a second
                print(f"  recorded at {fps:.0f} fps, not {config.FPS}: the loop is running slow")
            held = {"tile_name": tile["tile_name"], "detection_xy": [x, y], "home_xy": home, "attempt_dir": None,
                    "spot": placement.get("spot")}
            arm.squeeze()              # jaws a person stopped closing on the tile: shut, so holding() sees it
            returned = not args.no_return and return_tile(arm, actions, result, held, returns, realsense, session,
                                                          correction, hover, args.policy or "teleop", states)
            dagger = {"policy": policy.id, "takeover_frame": takeover, "fps": fps} if policy else {}
            if events["stop_recording"] or events["rerecord_episode"]:
                dataset.clear_episode_buffer()
                reason = "wrist camera lost" if camera_lost else None
                if not events["stop_recording"] and not camera_lost:
                    drain_stdin()
                    reason = ask_one({"ask": "Why discard it", "choices": DISCARD_REASONS}, None, input)
                session.event("episode_discarded", tile=tile["tile_name"], frames=frames, reason=reason, **dagger,
                              **roll.summary())
                if camera_lost:
                    camera = hardware.reopen_wrist_camera(camera)
                if events["stop_recording"]:
                    break
                if not returned and ask_same_tile(arm, reason):      # a returned tile has moved
                    retry = (tile, placement)
                continue
            episode = dataset.meta.total_episodes
            dataset.save_episode()
            saved += 1
            record = {"episode_index": episode, "timestamp": datetime.now().isoformat(timespec="seconds"),
                      "tile_name": tile["tile_name"], "detection_xy": [x, y],
                      "hover_xy": result["hover_xy"], "hover_unjittered_xy": [hx, hy], "hover_jitter_xy": list(offset),
                      "hover": hover, "hover_clearance_m": result["clearance_m"],
                      "hover_correction": correction.name if correction else "off", "stick_commit_s": args.stick_commit_s,
                      "frames": frames, "session": session.dir.name, "slice": args.slice,
                      "conditions": session.conditions, **placement, **dagger, **roll.summary()}
            with open(Path(dataset.root) / EPISODE_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
            session.event("episode_saved", **record)
            print(f"Saved episode {episode} ({frames / config.FPS:.1f}s"
                  + (")" if not policy else f", yours from {takeover / config.FPS:.1f}s)" if takeover is not None
                     else f", no takeover: {policy.id}'s alone, not trained on)"))
            offset = None          # the next episode draws its own
            if placement:
                todo.pop(0)
    except KeyboardInterrupt:          # Esc and q are the ways out; Ctrl-C is stepping in
        print("Interrupted.")
        session.intervention("interrupted", command="record")
    finally:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        dataset.finalize()
        rest_and_release(arm, hover)
        arm.disconnect()
        camera.release()
        realsense.close()
        gamepad.close()
        listener.stop()
        cv2.destroyAllWindows()
        print(f"{saved} episode(s) saved this session; {args.dataset} has {dataset.meta.total_episodes}.")
        session.close()


ROLL = config.JOINTS.index("wrist_roll")
KEY_REPEAT_S = 0.3     # a → this soon after the → that started recording is the same press counted twice


class RollTrace:
    """Wrist roll over an episode: how far the command moved (only the right stick's left/right moves it)
    and how far the servo strayed from the command (pushed or twisted). Tells a stick leak from the
    wrist being turned when the roll changes without being asked for (2026-10-27, 7B)."""

    def __init__(self):
        self.first = self.last = None
        self.travel = self.lag = 0.0

    def add(self, commanded, measured):
        if self.last is not None:
            self.travel += abs(commanded - self.last)
        self.first = commanded if self.first is None else self.first
        self.last = commanded
        self.lag = max(self.lag, abs(measured - commanded))

    def report(self):
        s = self.summary()
        if s and (s["roll_command_travel_deg"] >= 2 or s["roll_lag_max_deg"] >= 6):
            print(f"  wrist roll: command moved {s['roll_command_travel_deg']:.0f}deg (net {s['roll_command_change_deg']:+.0f}), "
                  f"servo up to {s['roll_lag_max_deg']:.0f}deg off its command")

    def summary(self):
        if self.first is None:
            return {}
        return {"roll_command_change_deg": round(self.last - self.first, 1),
                "roll_command_travel_deg": round(self.travel, 1), "roll_lag_max_deg": round(self.lag, 1)}


RETURN_TOWARD_HOVER = (0.0, 0.25, 0.5, 0.75, 1.0)     # fractions of the way from the return target to the hover


def return_tile(arm, actions, hovered, held, returns, realsense, session, correction, hover, policy_id, states=()):
    """Set a tile still in the jaws back down so it lands at its home (held["home_xy"]: where it was first
    seen this session), aimed with the return model from where the jaws closed (the first command shut; the
    hover if none was), then check where it landed. Returning to the grasp point instead walked tiles 25-31mm
    in one recording (2026-10-28). Teleop reaches further than the IK plans (9B at the left edge, 29cm out,
    2026-10-27), so an unreachable target walks back toward the hover, which the IK reached. True if it was
    set down. It's let go just above where the jaws closed on it (states: the measured joints per frame)."""
    if not arm.holding()[0]:
        return False
    _, grasp_xy = kinematics.grasp_point(actions)
    closed_at = kinematics.grasp_clearance(states, actions)
    held = {**held, "grasp_xy": grasp_xy or hovered["hover_xy"]}
    home, name = held["home_xy"], held["tile_name"]
    print(f"Returning {name} home, to ({home[0]:.3f}, {home[1]:+.3f}) as the RealSense sees it")
    tx, ty = returns.target(home, held["detection_xy"], held["grasp_xy"])
    hx, hy = hovered["hover_xy"]
    for k in RETURN_TOWARD_HOVER:
        x, y = tx + (hx - tx) * k, ty + (hy - ty) * k
        placed = arm.place(x, y, hovered["offset_m"], log=print if k == 0 else lambda *a: None,
                           grasp_clearance_m=closed_at)
        if placed is not None:
            break
    else:
        print("  Out of reach even at the hover: it's released at rest instead.")
        session.event("return_skipped", reason="unreachable", tile=name, xy=[tx, ty])
        return False
    if k:
        print(f"  set down {np.hypot(x - tx, y - ty) * 1000:.0f}mm toward the hover, at ({x:.3f}, {y:+.3f})")
    session.event("return", tile=name, asked_xy=[tx, ty], home_xy=home, toward_hover=k, grasp_clearance_m=closed_at,
                  return_correction=returns.name, **placed)
    tiles = realsense.detect_tiles()
    session.event("detections", label="after_return", target_xy=home,
                  tiles=[{f: t[f] for f in ("tile_name", "xy", "confidence") if f in t} for t in tiles])
    _, reset = check_landing(held, placed["xy"], tiles, session, config.RETURN_CORRECTIONS, correction, hover,
                             policy=policy_id, source="record", return_correction=returns.name)
    if reset:
        drain_stdin()
        input(f"  Put {name} back on its home, ({home[0]:.3f}, {home[1]:+.3f}), then Enter: ")
        session.intervention("tile_reset", tile=name, home_xy=home)
    return True


def rest_and_release(arm, hover):
    """Rest, then let go of a tile still held (--no-return, or a return out of reach) by opening to the hover
    gripper: a gripper left stalled on it at 0 overloads."""
    arm.rest()
    arm.squeeze()
    if arm.holding()[0]:
        print("Releasing the tile from the last episode at rest.")
        arm.set_gripper(hover["gripper"])


def ask_same_tile(arm, reason):
    """After ←: hover over the same tile again without going to rest and choosing it? Only offered with the
    jaws empty (a held tile is released at rest). Enter says yes unless the reason says the mat changed."""
    if arm.holding()[0]:
        print("Holding the tile: back to rest to release it.")
        return False
    drain_stdin()          # after a lost camera nothing has flushed the ← / → pressed while recording
    return ask_one({"ask": "Same tile, same hover again (nothing moved)", "choices": "yn"},
                   reason not in SAME_TILE_NO, input)


def next_offset(args, rng, saved):
    """The hover offset (m) for saved episode number `saved`."""
    if args.hover_offsets_mm:
        return args.hover_offsets_mm[saved % len(args.hover_offsets_mm)]
    return random_offset(rng, args.hover_jitter_mm / 1000, args.jitter_zero_fraction)


def show(camera, disabled, text, frame=None, last=None):
    """The wrist frame with `text`, and under it what the gamepad reads (`last`) while you drive."""
    if disabled:
        return
    if frame is None:
        ok, frame = camera.read()
        if not ok:
            return
    frame = frame.copy()
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if last:
        draw_input(frame, last)
    cv2.imshow("wrist", frame)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
