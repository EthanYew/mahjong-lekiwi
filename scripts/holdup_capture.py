"""Test 1 of docs/face_down_plan.md: hold tiles up to the LeKiwi base camera by hand and save what it sees. No arm.

    python scripts/holdup_capture.py                                   # 7 tiles x 3 distances x 2 ways up
    python scripts/holdup_capture.py --tiles 5C,3C,2C,9C --distances-cm 15,30 --frames 4
    python scripts/holdup_capture.py --sim                             # a synthetic camera, to try the flow

Run it once by day and once in the evening: the lighting questions are asked first and saved with the session. For each
tile, distance and way up (right way up, then turned 180 degrees) it names the shot; hold the tile face to the camera at
that distance, then press space (or Enter in the terminal with --no-view). `s` skips the shot, `b` takes the last one
again, `q` quits. The tile faces are checked by you: the name you hold is the truth. Score the session after with
`python scripts/holdup_score.py`.

Saved to sessions/<time>_holdup/photos/<NAME>/<distance>cm_<upright|turned>_<n>.jpg with a `holdup_shot` event per shot.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import config, holdup
from mahjong.camera import BaseCamera, read_within
from mahjong.session import Session, ask_conditions, previous_conditions

WINDOW = "base camera: hold a tile up"
LIGHT_KEYS = ("daylight", "lamp", "ceiling_lights")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tiles", default=",".join(holdup.DEFAULT_TILES),
                   help="the tiles you will hold, comma separated (default: %(default)s)")
    p.add_argument("--distances-cm", default=",".join(str(d) for d in holdup.DEFAULT_DISTANCES_CM),
                   help="distances from the camera to the tile (default: %(default)s)")
    p.add_argument("--frames", type=int, default=3, help="frames saved per shot (default 3)")
    p.add_argument("--gap-s", type=float, default=0.25, help="seconds between a shot's frames (default 0.25)")
    p.add_argument("--no-view", action="store_true", help="no preview window: Enter in the terminal takes the shot")
    p.add_argument("--sim", action="store_true", help="a synthetic camera, no hardware")
    return p.parse_args(argv)


class SimBaseCamera:
    """Noise with a tile-sized lighter patch: enough for the flow and the tests."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def read(self):
        frame = self.rng.integers(0, 60, (480, 640, 3), dtype=np.uint8)
        frame[160:320, 240:400] = 220
        return True, frame

    def flush(self, frames=10):
        pass

    def release(self):
        pass


def open_camera(sim):
    if sim:
        return SimBaseCamera()
    try:
        camera = BaseCamera()
    except RuntimeError as e:
        raise SystemExit(f"{e}\nCheck `ls -l /dev/basecam /dev/wristcam` (docs/face_down_plan.md, 'The camera fix'): "
                         "each must point at its own camera.")
    ok, _ = read_within(camera)
    if not ok:
        camera.release()
        raise SystemExit("the base camera opened but gives no frames: replug it")
    return camera


def ask_lighting(ask=input):
    questions = [q for q in config.SETUP_QUESTIONS if q["key"] in LIGHT_KEYS]
    return ask_conditions(previous_conditions(), questions=questions, ask=ask)


def instruction(shot, n, total):
    way = "right way up" if shot["kind"] == "upright" else "turned 180"
    return f"[{n}/{total}] hold {shot['tile']} {way}, face to the camera, {shot['distance_cm']} cm away"


def next_action(camera, text, view, ask):
    """'take', 'skip', 'back' or 'quit', from the preview window's keys or the terminal."""
    if not view:
        answer = ask(f"{text}  (Enter = take, s skip, b redo last, q quit): ").strip().lower()
        return {"": "take", "s": "skip", "b": "back", "q": "quit"}.get(answer, "take")
    print(text)
    while True:
        ok, frame = camera.read()
        if ok:
            shown = frame.copy()
            cv2.putText(shown, text[:70], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(shown, "space take, s skip, b redo last, q quit", (8, 466), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(30) & 0xFF
        action = {ord(" "): "take", ord("s"): "skip", ord("b"): "back", ord("q"): "quit", 13: "take"}.get(key)
        if action:
            return action


def take_frames(camera, n, gap_s, sleep=time.sleep):
    camera.flush()
    frames = []
    for i in range(n):
        if i:
            sleep(gap_s)
        ok, frame = read_within(camera)
        if not ok:
            break
        frames.append(frame)
    return frames


def main(argv=None, ask=input, camera=None):
    args = parse_args(argv)
    shots = holdup.plan([t.strip().upper() for t in args.tiles.split(",") if t.strip()],
                        [int(d) for d in args.distances_cm.split(",")])
    camera = camera or open_camera(args.sim)
    view = not (args.no_view or args.sim)
    conditions = ask_lighting(ask)
    session = Session("holdup", ask_setup=False)
    session.conditions = conditions
    session.event("holdup_conditions", conditions=conditions, shots=len(shots))
    taken, i, last = 0, 0, None
    try:
        while i < len(shots):
            shot = shots[i]
            action = next_action(camera, instruction(shot, i + 1, len(shots)), view, ask)
            if action == "quit":
                break
            if action == "skip":
                i += 1
            elif action == "back":
                if last is not None:
                    for f in last["files"]:
                        (session.dir / f).unlink(missing_ok=True)
                    session.event("holdup_shot_discarded", shot=last["shot"])
                    last = None
                    i = max(i - 1, 0)
            else:
                frames = take_frames(camera, args.frames, args.gap_s)
                if not frames:
                    print("No frames from the camera: replug it, then take this shot again.")
                    continue
                files = holdup.save_shot(session.dir, shot, frames)
                taken += 1
                last = {"shot": taken, "files": files}
                session.event("holdup_shot", shot=taken, files=files, **shot)
                i += 1
    finally:
        camera.release()
        if view:
            cv2.destroyAllWindows()
        session.event("holdup_done", shots_taken=taken, planned=len(shots))
        print(f"{taken} of {len(shots)} shots saved in {session.dir}\nScore them: python scripts/holdup_score.py")
        session.close()
    return session.dir


if __name__ == "__main__":
    main()
