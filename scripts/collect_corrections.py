"""Measure the hover's real error with the gamepad, for mahjong/hover_correction.py.

For each detected tile (left to right) the arm hovers as recording does, plus the correction measured
so far; you line the jaws up over the tile with the gamepad (torque on, wrist roll and pitch locked —
the IK sets both) and press →. The commanded pose's tool point minus the detection is recorded, so
every record is valid fitting data whatever the hover started from. ← skips a tile, Esc finishes.
Put the midpoint between the jaw tips over the centre of the tile's top face; lowering close to the
tile makes it easier to judge. A "wrist" window shows the wrist camera while you line up (--no-view: none),
and the frame at → is saved in the session (aligned/), so how each correction was lined up can be checked
afterwards. Aim for 15-20 tiles over the whole table, far corners included.

    python scripts/collect_corrections.py
    python -m mahjong.hover_correction        # the fit
"""
import argparse
import math
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, hardware, kinematics
from mahjong.arm import ADJUST_TIME_S
from mahjong.camera import LOST_S
from mahjong.gamepad import warn_if_at_reach
from mahjong.hover_correction import HoverCorrection, append_record
from mahjong.perception import drain_stdin
from mahjong.session import Session


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", action="store_true", help="hover over the raw detection, not the current correction")
    p.add_argument("--device", default=None, help="gamepad evdev path (default: the first gamepad found)")
    p.add_argument("--no-view", action="store_true", help="no wrist camera window (and no wrist camera needed)")
    args = p.parse_args(argv)
    from lerobot.common.control_utils import init_keyboard_listener
    devices = hardware.connect(["arm", "realsense", "gamepad"] + ([] if args.no_view else ["wrist_camera"]),
                               gamepad_kwargs={"device": args.device, "commit_s": 0.0, "locked_sticks": {"rx"},
                                               "locked_buttons": {"l", "r"}})
    arm, realsense, gamepad = devices["arm"], devices["realsense"], devices["gamepad"]
    camera = devices.get("wrist_camera")
    session = Session("corrections")
    session.event("connect", **arm.at_connect)
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
                    session.event("attempt_skipped", reason="unreachable", tile=tile["tile_name"], hover_xy=[hx, hy])
                    continue
                for k in ("exit_early", "rerecord_episode", "stop_recording"):
                    events[k] = False
                warn_if_at_reach(hx, hy)
                print("  line up, then → record / ← skip / Esc finish")
                gamepad.start(arm)
                label = f"[{i}/{len(tiles)}] {tile['tile_name']}: line up, -> record / <- skip / Esc finish"
                frame, failed_reads = None, 0
                while not events["exit_early"]:
                    t0 = time.time()
                    gamepad.step(arm, dt)
                    if camera is not None:
                        ok, latest = camera.read()
                        if ok:
                            frame, failed_reads = latest, 0
                            show(frame, label)
                        else:
                            failed_reads += 1
                            if failed_reads >= LOST_S * config.FPS:     # the arm holds its command meanwhile
                                camera, failed_reads = hardware.reopen_wrist_camera(camera), 0
                                gamepad.start(arm)
                    time.sleep(max(0.0, dt - (time.time() - t0)))
                commanded = arm.commanded()
                arm.move_to(hover["command"], ADJUST_TIME_S)     # straight back up before anything moves sideways
                if events["stop_recording"]:
                    stop = True
                    break
                if events["rerecord_episode"]:
                    print("  skipped")
                    session.event("correction_skipped", tile=tile["tile_name"], detection_xy=[x, y],
                                  hover_target_xy=[hx, hy])
                    continue
                aligned = kinematics.tool_point(commanded)
                delta = [float(aligned[0] - x), float(aligned[1] - y)]
                record = append_record({
                    "source": "gamepad", "tile_name": tile["tile_name"], "confidence": tile["confidence"], "detection_xy": [x, y],
                    "hover_target_xy": [hx, hy], "hover_correction": "off" if args.raw else correction.kind,
                    "aligned_commanded_joints": commanded, "aligned_measured_joints": arm.measured(),
                    "aligned_tool_xyz": [float(v) for v in aligned], "delta_xy": delta, "session": session.dir.name})
                session.event("correction_recorded", **record, **save_frame(session, recorded + 1, tile["tile_name"], frame))
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
    except KeyboardInterrupt:          # Esc and q are the ways out; Ctrl-C is stepping in
        print("Interrupted.")
        session.intervention("interrupted", command="collect_corrections")
    finally:
        arm.rest()
        arm.disconnect()
        realsense.close()
        gamepad.close()
        if camera is not None:
            camera.release()
            cv2.destroyAllWindows()
        listener.stop()
        print(f"{recorded} correction(s) recorded. `python -m mahjong.hover_correction` shows the fit.")
        session.close()


def show(frame, text):
    frame = frame.copy()
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("wrist", frame)
    cv2.waitKey(1)


def save_frame(session, n, tile_name, frame):
    """The wrist frame the jaws were lined up on, as session/aligned/<n>_<tile>.jpg: {"wrist_frame": its path
    in the session}, or {} without one."""
    if frame is None:
        return {}
    path = session.dir / "aligned" / f"{n:02d}_{tile_name}.jpg"
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), frame)
    return {"wrist_frame": str(path.relative_to(session.dir))}


if __name__ == "__main__":
    main()
