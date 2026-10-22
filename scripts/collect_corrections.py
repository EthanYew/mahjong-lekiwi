"""Measure the hover's real error with the gamepad, for mahjong/hover_correction.py.

For each detected tile (left to right) the arm hovers as recording does, plus the correction measured
so far; you line the jaws up over the tile with the gamepad (torque on, wrist roll and pitch locked —
the IK sets both) and press →. The commanded pose's tool point minus the detection is recorded, so
every record is valid fitting data whatever the hover started from. ← skips a tile, Esc finishes.
Put the midpoint between the jaw tips over the centre of the tile's top face; lowering close to the
tile makes it easier to judge. Aim for 15-20 tiles over the whole table, far corners included.

    python scripts/collect_corrections.py
    python -m mahjong.hover_correction        # the fit
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, kinematics
from mahjong.arm import Arm, ADJUST_TIME_S
from mahjong.gamepad import GamepadTeleop
from mahjong.hover_correction import HoverCorrection, append_record
from mahjong.perception import RealSense, drain_stdin
from mahjong.session import Session


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", action="store_true", help="hover over the raw detection, not the current correction")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    from lerobot.common.control_utils import init_keyboard_listener
    session = Session("corrections")
    gamepad = GamepadTeleop(args.device, commit_s=0.0, locked_sticks={"rx"}, locked_buttons={"l", "r"})
    arm, realsense = Arm.connect(), RealSense()
    listener, events = init_keyboard_listener()
    correction = HoverCorrection.load()
    print(f"Hover correction: {'off (--raw)' if args.raw else correction.summary()}")
    recorded, dt = 0, 1.0 / config.FPS
    try:
        while True:
            arm.rest()
            tiles = realsense.detect_tiles()
            print(f"\n{len(tiles)} tile(s): {', '.join(t['tile_name'] for t in tiles) or 'none'}")
            stop = False
            for i, tile in enumerate(tiles, start=1):
                x, y = tile["xy"]
                print(f"\n[{i}/{len(tiles)}] {tile['tile_name']} at ({x:.3f}, {y:+.3f})")
                hx, hy = (x, y) if args.raw else correction.apply(x, y)
                hover = arm.hover(hx, hy, config.HOVER_OFFSET_M, config.HOVER_CLEARANCE_M, config.HOVER_GRIPPER)
                if hover is None:
                    print("  unreachable — skipped")
                    continue
                for k in ("exit_early", "rerecord_episode", "stop_recording"):
                    events[k] = False
                print("  line up, then → record / ← skip / Esc finish")
                gamepad.start(arm)
                while not events["exit_early"]:
                    t0 = time.time()
                    gamepad.step(arm, dt)
                    time.sleep(max(0.0, dt - (time.time() - t0)))
                commanded = arm.commanded()
                arm.move_to(hover["command"], ADJUST_TIME_S)     # straight back up before anything moves sideways
                if events["stop_recording"]:
                    stop = True
                    break
                if events["rerecord_episode"]:
                    print("  skipped")
                    continue
                aligned = kinematics.tool_point(commanded)
                delta = [float(aligned[0] - x), float(aligned[1] - y)]
                record = append_record({
                    "tile_name": tile["tile_name"], "confidence": tile["confidence"], "detection_xy": [x, y],
                    "hover_target_xy": [hx, hy], "hover_correction": "off" if args.raw else correction.kind,
                    "aligned_commanded_joints": commanded, "aligned_measured_joints": arm.measured(),
                    "aligned_tool_xyz": [float(v) for v in aligned], "delta_xy": delta, "session": session.dir.name})
                session.event("correction_recorded", **record)
                recorded += 1
                missed = math.hypot(aligned[0] - hx, aligned[1] - hy)
                print(f"  recorded: jaws - detection ({delta[0] * 1000:+.1f}, {delta[1] * 1000:+.1f})mm; "
                      f"this hover missed by {missed * 1000:.1f}mm")
                correction = HoverCorrection.load()
                print(f"  correction now: {correction.summary()}")
            if stop:
                break
            drain_stdin()
            if input(f"\n{recorded} recorded. Rearrange tiles and Enter to scan again, q to finish: ").strip().lower() == "q":
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        arm.rest()
        arm.disconnect()
        realsense.close()
        gamepad.close()
        listener.stop()
        print(f"{recorded} correction(s) recorded. `python -m mahjong.hover_correction` shows the fit.")
        session.close()


if __name__ == "__main__":
    main()
