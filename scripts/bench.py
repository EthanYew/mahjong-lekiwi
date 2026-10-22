"""Test bench for isolated pickups: closed-loop hover + an ACT policy, one tile at a time or in batches.

    1 hover          detect, choose a tile, hover over it (corrected, settled from above)
    2 pickup         hover, then the policy picks it up; parks at rest holding it
    3 return         put the held tile back where the jaws closed on it, then rest
    4 discard        carry the held tile to the taught discard pose and release it, then rest
    5 teach-discard  drive the arm over the discard container with the gamepad and save that pose
    6 trials         pickup -> success check -> return, N times on one tile; prints the success rate
    7 rest           lift if low, then park (never opens the gripper)
    8 quit

Type a number or a name. During a pickup, Esc stops the policy. Everything goes in a session folder
(sessions/<time>_bench/): the terminal log, events.jsonl, and per attempt the wrist frames,
act_trace.npz and RealSense overview frames before and after. `python scripts/session_report.py`
summarises the latest one.

    python scripts/bench.py                         # interactive, default policy
    python scripts/bench.py --trials 5 --tile 8D    # batch, then quit
    python scripts/bench.py --sim --trials 3        # no hardware
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import config
from mahjong.arm import Arm
from mahjong.hover_correction import HoverCorrection
from mahjong.perception import choose_tile, drain_stdin
from mahjong.session import Session


class Abort:
    """Esc, from lerobot's keyboard listener, as an Event-like flag for policy.run."""
    def __init__(self, events):
        self.events = events

    def clear(self):
        self.events["stop_recording"] = self.events["exit_early"] = self.events["rerecord_episode"] = False

    def is_set(self):
        return self.events["stop_recording"]


class Bench:
    def __init__(self, args):
        self.args = args
        self.session = Session("bench", ask_notes=not args.sim, policy=args.policy or "default", sim=args.sim)
        if args.sim:
            from mahjong.sim import SimCamera, SimPerception, SimPolicy, SimRobot
            self.arm = Arm(SimRobot(), sleep=lambda s: None)
            self.camera, self.realsense, self.policy = SimCamera(), SimPerception(), SimPolicy()
            self.events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
            self.listener = None
        else:
            from lerobot.common.control_utils import init_keyboard_listener
            from mahjong.camera import WristCamera
            from mahjong.perception import RealSense
            from mahjong.policy import Policy
            self.policy = Policy(args.policy)
            self.arm, self.camera, self.realsense = Arm.connect(), WristCamera(), RealSense()
            self.listener, self.events = init_keyboard_listener()
        self.abort = Abort(self.events)
        self.correction = HoverCorrection.load()
        self.held = None        # {"tile_name", "return_xy", "approach_offset_m"} while holding a tile
        print(f"Policy: {self.policy.id}, hover {self.policy.hover}\nHover correction: {self.correction.summary()}")
        self.session.event("bench_setup", policy=self.policy.id, hover=self.policy.hover,
                           correction=self.correction.summary())

    # --- pieces ---

    def choose(self, tile_name=None):
        if not self.arm.is_at_rest():
            self.arm.rest()
        tiles = self.realsense.detect_tiles()
        if not tiles:
            print("No tiles detected.")
            return None
        if tile_name is None:
            drain_stdin()
        tile = choose_tile(tiles, tile_name=tile_name)
        if tile is None and tile_name is not None:
            print(f"{tile_name} not detected (seen: {', '.join(t['tile_name'] for t in tiles)})")
        return tile

    def hover_over(self, tile):
        x, y = tile["xy"]
        hx, hy = self.correction.apply(x, y)
        h = self.policy.hover
        result = self.arm.hover(hx, hy, h["offset_m"], h["clearance_m"], h["gripper"])
        if result is None:
            print("Unreachable from the closed-loop hover.")
        return result

    def snapshot(self, path):
        cv2.imwrite(str(path), self.realsense.snapshot())

    # --- commands ---

    def hover(self):
        if self.held:
            return print(f"Holding {self.held['tile_name']} — return or discard it first.")
        tile = self.choose()
        if tile and (result := self.hover_over(tile)):
            self.session.event("hover", tile=tile["tile_name"], detection_xy=tile["xy"], **result)

    def pickup(self, tile_name=None, ask=True):
        """One attempt. Returns True if the gripper ended up holding the tile."""
        if self.held:
            print(f"Holding {self.held['tile_name']} — return or discard it first.")
            return None
        tile = self.choose(tile_name)
        if tile is None:
            return None
        attempt = self.session.attempt_dir(tile["tile_name"])
        self.snapshot(attempt / "overview_before.jpg")
        hover = self.hover_over(tile)
        if hover is None:
            return None
        self.abort.clear()
        run = self.policy.run(self.arm, self.camera, attempt, abort=self.abort)
        self.arm.rest()
        holding, command, measured = self.arm.holding()
        self.snapshot(attempt / "overview_after.jpg")
        success = holding
        if ask:
            drain_stdin()
            answer = input(f"Picked it up? (gripper commanded {command:.0f}, measured {measured:.0f}: looks "
                           f"{'held' if holding else 'empty'}) [{'Y/n' if holding else 'y/N'}]: ").strip().lower()
            success = holding if not answer else answer.startswith("y")
        self.session.event("attempt", tile=tile["tile_name"], policy=self.policy.id, detection_xy=tile["xy"],
                           hover_xy=hover["hover_xy"], hover_offset_m=hover["offset_m"],
                           hover_clearance_m=hover["clearance_m"], gripper_command=command, gripper_measured=measured,
                           gripper_says_held=holding, success=success, attempt_dir=attempt.name, **run)
        if holding:
            self.held = {"tile_name": tile["tile_name"], "return_xy": run["grasp_xy"] or hover["hover_xy"],
                         "approach_offset_m": hover["offset_m"]}
        print(f"{tile['tile_name']}: {'picked up' if success else 'missed'}")
        return success

    def return_tile(self):
        if not self.held:
            return print("Not holding a tile.")
        x, y = self.held["return_xy"]
        print(f"Returning {self.held['tile_name']} to ({x:.3f}, {y:+.3f})")
        released_at = self.arm.place(x, y, self.held["approach_offset_m"])
        if released_at is not None:
            self.session.event("return", tile=self.held["tile_name"], xy=[x, y], released_at_m=released_at)
            self.held = None

    def discard(self):
        if not config.DISCARD_POSE.exists():
            return print("No discard pose yet — use teach-discard first.")
        pose = json.loads(config.DISCARD_POSE.read_text())["pose"]
        self.arm.go_to_pose(pose)
        self.arm.set_gripper(config.GRIPPER_RELEASE)
        self.arm.rest()
        self.session.event("discard", tile=self.held["tile_name"] if self.held else None)
        self.held = None

    def teach_discard(self):
        from mahjong.gamepad import GamepadTeleop
        import time
        gamepad = GamepadTeleop(self.args.device, commit_s=0.0)
        self.arm.raise_from_rest()
        gamepad.start(self.arm)
        self.abort.clear()
        print("Drive over the discard container's opening, then → to save (Esc cancels).")
        while not (self.events["exit_early"] or self.events["stop_recording"]):
            t0 = time.time()
            gamepad.step(self.arm, 1.0 / config.FPS)
            time.sleep(max(0.0, 1.0 / config.FPS - (time.time() - t0)))
        gamepad.close()
        if not self.events["stop_recording"]:
            pose = {j: self.arm.commanded()[j] for j in config.ARM_JOINTS}
            config.DISCARD_POSE.write_text(json.dumps({"pose": pose, "session": self.session.dir.name}, indent=2) + "\n")
            self.session.event("discard_pose_taught", pose=pose)
            print(f"Saved {config.DISCARD_POSE}")
        self.arm.rest()

    def trials(self, n=None, tile_name=None):
        if n is None:
            drain_stdin()
            n = int(input("How many trials? ").strip() or 0)
            tile_name = input("Which tile (name)? ").strip() or None
        results = []
        for i in range(n):
            print(f"\n=== trial {i + 1}/{n} ===")
            success = self.pickup(tile_name, ask=False)
            if success is None:
                print("Couldn't attempt (tile not found or unreachable) — stopping the trials.")
                break
            results.append(success)
            if self.held:
                self.return_tile()
        rate = f"{sum(results)}/{len(results)}"
        self.session.event("trials_done", tile=tile_name, policy=self.policy.id, results=results)
        print(f"\n{tile_name or 'trials'}: {rate} picked up")

    def close(self):
        try:
            self.arm.rest()
        finally:
            self.arm.robot.disconnect()
            self.camera.release()
            self.realsense.close()
            if self.listener:
                self.listener.stop()
            self.session.close()


COMMANDS = ["hover", "pickup", "return", "discard", "teach-discard", "trials", "rest", "quit"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--policy", default=None, help="policy id from data/policies.json (default: its 'default')")
    p.add_argument("--trials", type=int, default=None, help="run this many pickup/return trials, then quit")
    p.add_argument("--tile", default=None, help="tile name for --trials")
    p.add_argument("--sim", action="store_true", help="simulated arm, camera, perception and policy")
    p.add_argument("--device", default=None, help="gamepad evdev path (teach-discard)")
    args = p.parse_args()
    bench = Bench(args)
    actions = {"hover": bench.hover, "pickup": bench.pickup, "return": bench.return_tile, "discard": bench.discard,
               "teach-discard": bench.teach_discard, "trials": bench.trials, "rest": bench.arm.rest}
    try:
        bench.arm.rest()
        if args.trials:
            bench.trials(args.trials, args.tile)
            return
        while True:
            print("\n" + "\n".join(f"  {i} {name}" for i, name in enumerate(COMMANDS, start=1)))
            print(f"[{'holding ' + bench.held['tile_name'] if bench.held else 'gripper empty'}]")
            drain_stdin()
            choice = input("> ").strip().lower()
            if choice.isdigit() and 1 <= int(choice) <= len(COMMANDS):
                choice = COMMANDS[int(choice) - 1]
            if choice in ("quit", "q"):
                break
            if choice not in actions:
                print(f"unknown command {choice!r}")
                continue
            try:
                actions[choice]()
            except KeyboardInterrupt:
                print("\ninterrupted — holding position")
    except KeyboardInterrupt:
        pass
    finally:
        bench.close()


if __name__ == "__main__":
    main()
