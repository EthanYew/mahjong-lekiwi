"""What play is asking about, on screen: the RealSense view with every hand slot named, the tiles a person should look at picked out
with what they might be, and the wrist camera's view of the one asked about when the arm has looked. Answers still go in the terminal.

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
PICTURE_FIT_PX = (1100, 640)    # the mat at its largest on screen: at the frame's own 848 wide, neighbouring labels collide
ZOOM_MARGIN_PX = 25         # around the tiles, in the frame: room for the labels above the far row
CAPTION_PX_PER_CHAR = 10
LINE_PX = 24
LOUD_TEXT = 0.6             # a label picked out
LABEL_RIVALS = 1            # other names on the picture itself: there's room for one between neighbours; the list has them all
DIM = 0.45                  # how much light the rest of the mat keeps when some tiles need attention
OK, LOOK, FOCUS, ZONE = (0, 200, 0), (0, 170, 255), (0, 255, 255), (255, 180, 0)


def _text(image, text, xy, colour, scale=0.5, thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def zoom_on(image, outlines, fit=None):
    """The part of the frame holding the outlines, enlarged to fit (width, height; the frame's own by default), and the function
    taking frame pixels to the enlarged picture's. On the real layout a tile is ~30px wide in the whole frame: too small to read a
    label against."""
    h, w = image.shape[:2]
    points = np.vstack(outlines)
    x0, y0 = np.maximum(np.floor(points.min(axis=0)).astype(int) - ZOOM_MARGIN_PX, 0)
    x1, y1 = np.minimum(np.ceil(points.max(axis=0)).astype(int) + ZOOM_MARGIN_PX, (w, h))
    fit_w, fit_h = fit or (w, h)
    scale = min(fit_w / (x1 - x0), fit_h / (y1 - y0))
    picture = cv2.resize(image[y0:y1, x0:x1], (round((x1 - x0) * scale), round((y1 - y0) * scale)), interpolation=cv2.INTER_CUBIC)
    return picture, lambda pixels: (np.asarray(pixels) - (x0, y0)) * scale


def suspected(tile):
    """A tile's name as the window puts it: "5C", or "5C or 9C" once it needs a look and has another name it could be."""
    others = (tile.get("could_be") or [])[:LABEL_RIVALS] if tile.get("why") else []
    return " or ".join([tile.get("name") or "?", *others])


def attention_list(tiles, focus):
    """A row of text for each tile needing a look: its whole list of suspects and why, for what the picture has no room for."""
    rows = []
    for tile in tiles:
        if tile.get("why"):
            others = tile.get("could_be") or []
            name = (tile.get("name") or "?") + (" or " + ", ".join(others) if others else "")
            rows.append((f"{'> ' if tile['id'] == focus else ''}{tile['id']}: {name}. {tile['why']}", FOCUS if tile["id"] == focus else LOOK))
    return rows


def text_bar(width, rows, footer=None):
    """Wrapped rows of (text, colour) on black."""
    lines = [(line, colour) for text, colour in rows for line in textwrap.wrap(text, width // CAPTION_PX_PER_CHAR) or [""]]
    bar = np.zeros((LINE_PX * len(lines) + (34 if footer else 10), width, 3), np.uint8)
    for i, (line, colour) in enumerate(lines):
        _text(bar, line, (10, LINE_PX * (i + 1)), colour, 0.55)
    if footer:
        _text(bar, footer, (10, bar.shape[0] - 10), (170, 170, 170), 0.45)
    return bar


def outline_of(tile, intrinsics, T_cam_to_base):
    """Where a tile is in the frame: the detector's own box if it was seen, as that's what's really there and needs no calibration;
    else where its position projects to."""
    from .layout import table_to_pixel, tile_corners
    if tile.get("box"):
        u, v, w, h = tile["box"]
        return np.array([[u - w / 2, v - h / 2], [u + w / 2, v - h / 2], [u + w / 2, v + h / 2], [u - w / 2, v + h / 2]], float)
    return table_to_pixel(tile_corners(*tile["xy"], 0.0), intrinsics, T_cam_to_base)


def compose(overview, tiles, caption, intrinsics, T_cam_to_base, focus=None, wrist=None):
    """The picture. tiles: [{"id", "xy", "name", "box"}], name None if there's nothing to say yet, box the detection's (u, v, w, h) in
    pixels if there is one; a tile needing a look also has "why" and "could_be" (other names it may be). With any of those the rest of the mat dims, so the eye goes straight to them, and they're
    listed under the picture. focus: the id asked about right now. wrist: the wrist camera's frame of it, or None."""
    from .name_check import crop
    outlines = [outline_of(tile, intrinsics, T_cam_to_base) for tile in tiles]
    picture, to_picture = zoom_on(overview, outlines, PICTURE_FIT_PX)
    corners = [np.round(to_picture(outline)).astype(np.int32) for outline in outlines]
    picked_out = [c for tile, c in zip(tiles, corners) if tile.get("why") or tile["id"] == focus]
    if any(tile.get("why") for tile in tiles):
        lit = cv2.fillPoly(np.zeros(picture.shape[:2], np.uint8), picked_out, 1).astype(bool)
        picture[~lit] = (picture[~lit] * DIM).astype(np.uint8)
    # every outline before any label, and the labels picked out last: a person checks all 13, so a thick box mustn't hide a name
    drawn = sorted(zip(tiles, corners), key=lambda pair: bool(pair[0].get("why") or pair[0]["id"] == focus))
    colour_of = lambda tile: FOCUS if tile["id"] == focus else LOOK if tile.get("why") else ZONE if tile["id"] == "zone" else OK
    for tile, c in drawn:
        cv2.polylines(picture, [c], True, colour_of(tile), 3 if tile.get("why") or tile["id"] == focus else 1, cv2.LINE_AA)
    for tile, c in drawn:
        loud = tile.get("why") or tile["id"] == focus
        _text(picture, f"{tile['id']} {suspected(tile)}", (int(c[:, 0].min()), int(c[:, 1].min()) - 5), colour_of(tile),
              LOUD_TEXT if loud else 0.5, 2 if loud else 1)
    if wrist is not None:
        panel = crop(wrist, side=WRIST_CROP_PX, size=PANEL_PX)
        _text(panel, f"wrist camera: {focus}" if focus else "wrist camera", (10, 25), FOCUS, 0.6, 2)
        height = max(picture.shape[0], panel.shape[0])
        pad = lambda image: cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        picture = np.hstack([pad(picture), pad(panel)])
    width = picture.shape[1]
    parts = [text_bar(width, [(caption, (255, 255, 255))], "Answer in the terminal"), picture]
    rows = attention_list(tiles, focus)
    if rows:
        parts.append(text_bar(width, rows))
    return np.vstack(parts)


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
