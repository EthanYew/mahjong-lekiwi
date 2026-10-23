"""Record pickup demonstrations: closed-loop hover over a chosen tile, then gamepad teleop.

Each episode: the arm rests, the RealSense detects tiles, you pick one, the arm hovers over it
(measured hover correction, settled from above to the clearance), and holds. → hands you the gamepad
and starts recording from exactly that pose; → again saves, ← discards and re-hovers, Esc ends.

The hover settings (offset, clearance, gripper opening) are written per episode to
<dataset>/episode_log.jsonl; make_policy.py stores them with the trained policy so it's deployed from
the same hover. Only the wrist camera is recorded (the policy's only input). Each record also carries
the session's setup answers (lighting etc.) and the episode's --slice, so offline evaluation can break
scores down by them.

With --layout NAME (taught with scripts/layout.py), the RealSense view shows where the next tile goes —
an outline that turns green once the tile is within 4mm / 5° — and the tile at that spot is picked
without asking. Each spot is recorded --repeats times in a row; the episode log stores the spot and
where the tile actually was.

    python scripts/record.py --dataset Yewy/mahjong_mat_pickup_v3 --task "Pick up the mahjong tile from the mat"
    python scripts/record.py --dataset Yewy/mahjong_mat_eval_v1 --slice new --layout eval_new --repeats 2
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import config, hardware, layout
from mahjong.hover_correction import HoverCorrection
from mahjong.perception import choose_tile
from mahjong.session import Session

JOINT_NAMES = [f"{j}.pos" for j in config.JOINTS]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},          # delta from episode start
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},                     # delta from episode start
    "observation.state_raw": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action_raw": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["h", "w", "c"]},
}
EPISODE_LOG = "episode_log.jsonl"
SLICES = ["train", "familiar", "new", "dim", "tilt"]     # train: for training; the rest: eval-only (offline_eval.py)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, help="repo id to create or append to, e.g. Yewy/mahjong_mat_pickup_v3")
    p.add_argument("--task", default="Pick up the mahjong tile from the mat")
    p.add_argument("--episodes", type=int, default=200, help="stop after this many saved episodes")
    p.add_argument("--hover-offset-mm", type=float, default=config.HOVER_OFFSET_M * 1000)
    p.add_argument("--clearance-mm", type=float, default=config.HOVER_CLEARANCE_M * 1000)
    p.add_argument("--hover-gripper", type=float, default=config.HOVER_GRIPPER,
                   help="gripper opening at the hover (%%). Every dataset so far used 90 and spent ~3.5s narrowing to "
                        "~21%% before descending; ~23 starts where that narrowing ends")
    p.add_argument("--stick-commit-s", type=float, default=0.1)
    p.add_argument("--no-correction", action="store_true", help="hover over the raw detection")
    p.add_argument("--slice", choices=SLICES, default="train",
                   help="what these episodes are for: train, or an eval slice (familiar spots / new spots / dim light / "
                        "tilted tiles) — eval datasets are never trained on")
    p.add_argument("--layout", default=None, help="place tiles to this layout's spots (scripts/layout.py)")
    p.add_argument("--repeats", type=int, default=2, help="with --layout: episodes per spot")
    p.add_argument("--start-spot", default=None, help="with --layout: start at this spot id")
    p.add_argument("--device", default=None, help="gamepad evdev path")
    p.add_argument("--no-view", action="store_true", help="no wrist camera window")
    return p.parse_args()


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


def main():
    args = parse_args()
    from lerobot.common.control_utils import init_keyboard_listener
    hover = {"offset_m": args.hover_offset_mm / 1000, "clearance_m": args.clearance_mm / 1000, "gripper": args.hover_gripper}
    spots, todo, ghost = [], [], None
    if args.layout:            # before connecting anything, so a typo fails fast
        spots = layout.load(args.layout)["spots"]
        todo = layout.schedule(spots, args.repeats, args.start_spot)
        ghost = layout.load_ghost(args.layout)
    devices = hardware.connect(["arm", "wrist_camera", "realsense", "gamepad"],
                               gamepad_kwargs={"device": args.device, "commit_s": args.stick_commit_s})
    arm, camera, realsense, gamepad = (devices[k] for k in ("arm", "wrist_camera", "realsense", "gamepad"))
    print(f"Gamepad: {gamepad.name}")
    session = Session("record", dataset=args.dataset, task=args.task, hover=hover, stick_commit_s=args.stick_commit_s,
                      slice=args.slice, layout=args.layout)
    correction = None if args.no_correction else HoverCorrection.load()
    print(f"Hover correction: {correction.summary() if correction else 'off'}")
    dataset = open_dataset(args.dataset)
    listener, events = init_keyboard_listener()
    saved, dt = 0, 1.0 / config.FPS
    try:
        while saved < args.episodes:
            for k in ("exit_early", "rerecord_episode", "stop_recording"):
                events[k] = False
            arm.rest()
            if arm.holding()[0]:
                print("Releasing the tile from the last episode at rest.")
                arm.set_gripper(hover["gripper"])
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
                tile = layout.nearest_tile(realsense.detect_tiles(), found["xy"] if found else spot["xy"])
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
                tile = choose_tile(realsense.detect_tiles())
            if tile is None:
                if input("No tile chosen. q to quit, Enter to scan again: ").strip().lower() == "q":
                    break
                continue
            x, y = tile["xy"]
            hx, hy = correction.apply(x, y) if correction else (x, y)
            result = arm.hover(hx, hy, hover["offset_m"], hover["clearance_m"], hover["gripper"])
            if result is None:
                print("Unreachable from the closed-loop hover — choose another tile.")
                continue
            session.event("hover", tile=tile["tile_name"], detection_xy=[x, y], **result)

            print("Hovering. → take control and START recording, ← re-hover, Esc quit")
            while not events["exit_early"]:
                show(camera, args.no_view, "hovering: → to record")
                time.sleep(dt)
            if events["stop_recording"]:
                break
            if events["rerecord_episode"]:
                continue
            events["exit_early"] = False

            gamepad.start(arm)
            anchor = np.array([arm.measured()[j] for j in config.JOINTS], dtype=np.float32)
            print("RECORDING — → save, ← discard, Esc save nothing and quit")
            frames = 0
            while not events["exit_early"]:
                t0 = time.time()
                command = gamepad.step(arm, dt)
                ok, frame = camera.read()
                if not ok:
                    continue
                state = np.array([arm.measured()[j] for j in config.JOINTS], dtype=np.float32)
                action = np.array([command[j] for j in config.JOINTS], dtype=np.float32)
                dataset.add_frame({"observation.state": state - anchor, "action": action - anchor,
                                   "observation.state_raw": state, "action_raw": action,
                                   "observation.images.wrist": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "task": args.task})
                frames += 1
                show(camera, args.no_view, f"REC {frames / config.FPS:4.1f}s", frame)
                time.sleep(max(0.0, dt - (time.time() - t0)))
            if events["stop_recording"] or events["rerecord_episode"]:
                dataset.clear_episode_buffer()
                session.event("episode_discarded", tile=tile["tile_name"], frames=frames)
                if events["stop_recording"]:
                    break
                continue
            episode = dataset.meta.total_episodes
            dataset.save_episode()
            saved += 1
            record = {"episode_index": episode, "timestamp": datetime.now().isoformat(timespec="seconds"),
                      "tile_name": tile["tile_name"], "detection_xy": [x, y],
                      "hover_xy": result["hover_xy"], "hover": hover, "hover_clearance_m": result["clearance_m"],
                      "hover_correction": correction.kind if correction else "off", "stick_commit_s": args.stick_commit_s,
                      "frames": frames, "session": session.dir.name, "slice": args.slice,
                      "conditions": session.conditions, **placement}
            with open(Path(dataset.root) / EPISODE_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
            session.event("episode_saved", **record)
            print(f"Saved episode {episode} ({frames / config.FPS:.1f}s)")
            if placement:
                todo.pop(0)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        dataset.finalize()
        arm.rest()
        arm.disconnect()
        camera.release()
        realsense.close()
        gamepad.close()
        listener.stop()
        cv2.destroyAllWindows()
        print(f"{saved} episode(s) saved this session; {args.dataset} has {dataset.meta.total_episodes}.")
        session.close()


def show(camera, disabled, text, frame=None):
    if disabled:
        return
    if frame is None:
        ok, frame = camera.read()
        if not ok:
            return
    frame = frame.copy()
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("wrist", frame)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
