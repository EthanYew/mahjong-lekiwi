"""What play is asking about, on screen: the RealSense view with every hand slot named, the tile in question picked out, and the
wrist camera's view of it when the arm has looked. Answers still go in the terminal.

The window lives in its own process. Play's main thread sits in the policy's control loop and in input() for long stretches, and
an OpenCV window nobody pumps for a few seconds gets marked "not responding". So this process only sends pictures; the other one
does nothing but keep the window answering, and leaves when the pipe closes, however play ended.
"""
import struct
import subprocess
import sys
import threading
import textwrap
from pathlib import Path

import cv2
import numpy as np

WINDOW = "play: the hand"
WRIST_CROP_PX = 360         # wider than the reader's crop, so a person sees the tile even when it sits off-centre
PANEL_PX = 480
ZOOM_MARGIN_PX = 25         # around the tiles, in the frame: room for the labels above the far row
CAPTION_PX_PER_CHAR = 10
LINE_PX = 24
OK, UNSURE, FOCUS, ZONE = (0, 200, 0), (0, 170, 255), (0, 255, 255), (255, 180, 0)


def _text(image, text, xy, colour, scale=0.5, thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def zoom_on(image, outlines):
    """The part of the frame holding the outlines, enlarged to the frame's width or height, and the function taking frame pixels
    to the enlarged picture's. On the real layout a tile is ~30px wide in the whole frame: too small to read a label against."""
    h, w = image.shape[:2]
    points = np.vstack(outlines)
    x0, y0 = np.maximum(np.floor(points.min(axis=0)).astype(int) - ZOOM_MARGIN_PX, 0)
    x1, y1 = np.minimum(np.ceil(points.max(axis=0)).astype(int) + ZOOM_MARGIN_PX, (w, h))
    scale = min(w / (x1 - x0), h / (y1 - y0))
    picture = cv2.resize(image[y0:y1, x0:x1], (round((x1 - x0) * scale), round((y1 - y0) * scale)), interpolation=cv2.INTER_CUBIC)
    return picture, lambda pixels: (np.asarray(pixels) - (x0, y0)) * scale


def compose(overview, tiles, caption, intrinsics, T_cam_to_base, focus=None, wrist=None):
    """The picture. tiles: [{"id", "xy", "name", "unsure"}], name None if there's nothing to say yet. focus: the id asked about.
    wrist: the wrist camera's frame of it, or None."""
    from .layout import table_to_pixel, tile_corners
    from .name_check import crop
    outlines = [table_to_pixel(tile_corners(*tile["xy"], 0.0), intrinsics, T_cam_to_base) for tile in tiles]
    picture, to_picture = zoom_on(overview, outlines)
    for tile, outline in zip(tiles, outlines):
        corners = np.round(to_picture(outline)).astype(np.int32)
        colour = FOCUS if tile["id"] == focus else ZONE if tile["id"] == "zone" else UNSURE if tile.get("unsure") else OK
        cv2.polylines(picture, [corners], True, colour, 3 if tile["id"] == focus else 1, cv2.LINE_AA)
        label = f"{tile['id']} {tile.get('name') or '?'}"
        _text(picture, label, (int(corners[:, 0].min()), int(corners[:, 1].min()) - 5), colour, 0.5, 2 if tile["id"] == focus else 1)
    if wrist is not None:
        panel = crop(wrist, side=WRIST_CROP_PX, size=PANEL_PX)
        _text(panel, f"wrist camera: {focus}" if focus else "wrist camera", (10, 25), FOCUS, 0.6, 2)
        height = max(picture.shape[0], panel.shape[0])
        pad = lambda image: cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        picture = np.hstack([pad(picture), pad(panel)])
    lines = textwrap.wrap(caption, picture.shape[1] // CAPTION_PX_PER_CHAR) or [""]
    bar =np.zeros((LINE_PX * len(lines) + 34, picture.shape[1], 3), np.uint8)
    for i, line in enumerate(lines):
        _text(bar, line, (10, LINE_PX * (i + 1)), (255, 255, 255), 0.55)
    _text(bar, "Answer in the terminal", (10, bar.shape[0] - 10), (170, 170, 170), 0.45)
    return np.vstack([bar, picture])


class Window:
    """Shows pictures in a window of its own process. If the window can't open (no display), play carries on without it."""

    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable, "-m", "mahjong.hand_view"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,      # Ctrl-C in the terminal stops play, not the window
            cwd=Path(__file__).resolve().parents[1])

    def show(self, image):
        if self.process is None:
            return
        data = cv2.imencode(".png", image)[1].tobytes()
        try:
            self.process.stdin.write(struct.pack(">I", len(data)) + data)
            self.process.stdin.flush()
        except OSError:
            print("  The play window is gone: the questions carry on in the terminal.")
            self.process = None

    def close(self):
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None


def read_pictures(stream, deliver):
    """Each length-prefixed picture on `stream` to deliver(image), until the stream ends."""
    while (head := stream.read(4)) and len(head) == 4:
        data = stream.read(struct.unpack(">I", head)[0])
        deliver(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))


def run_window(stream):
    """The window's process: show the newest picture and keep the window answering until the pipe closes."""
    latest, done = [], threading.Event()

    def reader():
        read_pictures(stream, latest.append)
        done.set()
    threading.Thread(target=reader, daemon=True).start()
    while not done.is_set():
        if latest:
            cv2.imshow(WINDOW, latest[-1])
            latest.clear()
        cv2.waitKey(50)


if __name__ == "__main__":
    run_window(sys.stdin.buffer)
