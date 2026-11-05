"""Drive the arm with the gamepad while watching the wrist camera and the base camera side by side (face-down plan, test 2).

    python scripts/teleop_view.py                          # drive from the raised rest; s saves a pose, c saves both photos
    python scripts/teleop_view.py --go manual_1            # start at a saved pose and adjust it
    python scripts/teleop_view.py --name base_view_        # saved poses are base_view_1, base_view_2, ...
    python scripts/teleop_view.py --sim                    # simulated arm and cameras, no hardware

Use it to find where the arm can hold a tile up to the base camera: the window shows the wrist view on the left, the base view
on the right, the commanded pose, the tool point and its distance from the shoulder, and what the gamepad reads. Put a tile in the jaws first. Keys, in the window:
  s  save the commanded pose as the next name (data/camera_poses.json) and print it
  c  save both camera frames to the session folder (captures/)
  g  go back to --go's pose  (only with --go)
  q  rest the arm and quit
Saved poses are taught ones: the commanded pose, not the measured (the servos sag below their command), the way the
discard pose is. The arm is never started from the measured pose: the teleop is seeded from what was last commanded.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import camera_poses, config, hardware, kinematics
from mahjong.gamepad import draw_input, warn_off_centre
from mahjong.session import Session

WINDOW = "wrist (left) | base (right)"
SCALE = 0.75


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--go", default=None, metavar="POSE", help="start at this saved pose (data/camera_poses.json)")
    p.add_argument("--name", default="base_view_", help="saved poses are named this plus a number (default base_view_)")
    p.add_argument("--device", default=None, help="gamepad evdev path (default: the first gamepad found)")
    p.add_argument("--sim", action="store_true", help="simulated arm, cameras and gamepad")
    return p.parse_args(argv)


class SimPad:
    """--sim: a gamepad that never moves the arm."""
    last = {}

    def start(self, arm):
        pass

    def step(self, arm, dt):
        return arm.commanded()

    def off_centre(self, fraction=0.5):
        return {}

    def close(self):
        pass


class SimBase:
    """--sim: the base camera as a grey frame."""
    def read(self):
        return True, np.full((480, 640, 3), 90, np.uint8)

    def flush(self, frames=10):
        pass

    def release(self):
        pass


def connect(args):
    if args.sim:
        from mahjong import sim
        devices = sim.connect(wrist_view=True)
        return {"arm": devices["arm"], "wrist_camera": devices["wrist_camera"], "base_camera": SimBase(), "gamepad": SimPad()}
    return hardware.connect(["arm", "gamepad", "wrist_camera", "base_camera"],
                            gamepad_kwargs={"device": args.device, "commit_s": 0.0})


def panel(wrist, base, pose, tool, reach_m, last):
    """Both frames side by side at SCALE, titled, with the commanded pose and the reach under them."""
    def prep(frame, title):
        frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), np.uint8)
        cv2.putText(frame, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame
    left = prep(wrist, "wrist")
    if last:
        draw_input(left, last)
    right = prep(base, "base")
    both = np.hstack([left, right])
    both = cv2.resize(both, None, fx=SCALE, fy=SCALE)
    text = " ".join(f"{j[:5]} {pose[j]:.0f}" for j in config.ARM_JOINTS) + f"  gripper {pose.get('gripper', 0):.0f}"
    cv2.putText(both, text, (8, both.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(both, f"tool point x {tool[0] * 100:+.1f} y {tool[1] * 100:+.1f} z {tool[2] * 100:+.1f} cm, "
                      f"{reach_m * 100:.0f} cm from the shoulder", (8, both.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 200, 0), 1)
    return both


def commanded_pose(arm):
    c = arm.commanded()
    return {j: float(c[j]) for j in config.JOINTS if j in c}


def main(argv=None, keys=None, max_steps=None):
    """`keys`: a function returning the next key press ('s', 'c', 'g', 'q') or None, for tests and --no-view."""
    args = parse_args(argv)
    if args.go:
        camera_poses.get(args.go)           # fail before any hardware if the name is wrong
    devices = connect(args)
    arm, gamepad = devices["arm"], devices["gamepad"]
    wrist, base = devices["wrist_camera"], devices["base_camera"]
    session = Session("teleop_view", ask_setup=False, sim=args.sim)
    view = not args.sim
    saved = captured = steps = 0
    try:
        arm.raise_from_rest()
        if args.go:
            arm.go_to_pose(camera_poses.get(args.go)["pose"])
        gamepad.start(arm)
        warn_off_centre(gamepad, session, None)
        print(__doc__.split("Keys, in the window")[1].split("Saved poses")[0])
        while True:
            t0 = time.time()
            gamepad.step(arm, 1.0 / config.FPS)
            ok_w, frame_w = wrist.read()
            ok_b, frame_b = base.read()
            pose = commanded_pose(arm)
            tool = kinematics.tool_point(arm.commanded())
            reach = float(np.hypot(tool[0] - kinematics.SHOULDER_PAN_X, tool[1]))
            image = panel(frame_w if ok_w else None, frame_b if ok_b else None, pose, tool, reach,
                          getattr(gamepad, "last", None))
            key = None
            if view:
                cv2.imshow(WINDOW, image)
                code = cv2.waitKey(1) & 0xFF
                key = chr(code) if code != 255 else None
            elif keys:
                key = keys()
            if key == "q":
                break
            if key == "s":
                name = camera_poses.next_name(args.name)
                camera_poses.save(name, pose, source="taught with teleop_view.py: the commanded pose", session=session.dir.name)
                saved += 1
                session.event("camera_pose_saved", name=name, pose=pose)
                print(f"saved '{name}': {pose}")
            elif key == "c":
                captured += 1
                out = session.dir / "captures"
                out.mkdir(exist_ok=True)
                for label, frame, ok in (("wrist", frame_w, ok_w), ("base", frame_b, ok_b)):
                    if ok:
                        cv2.imwrite(str(out / f"{captured:03d}_{label}.jpg"), frame)
                session.event("captured", n=captured, pose=pose)
                print(f"saved captures/{captured:03d}_wrist.jpg and _base.jpg")
            elif key == "g" and args.go:
                arm.go_to_pose(camera_poses.get(args.go)["pose"])
                gamepad.start(arm)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            time.sleep(max(0.0, 1.0 / config.FPS - (time.time() - t0)))
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        arm.rest()
        arm.disconnect()
        gamepad.close()
        wrist.release()
        base.release()
        if view:
            cv2.destroyAllWindows()
        session.event("teleop_view_done", poses_saved=saved, captures=captured)
        print(f"{saved} pose(s) saved to {config.CAMERA_POSES}, {captured} capture(s) in {session.dir}")
        session.close()
    return session.dir


if __name__ == "__main__":
    main()
