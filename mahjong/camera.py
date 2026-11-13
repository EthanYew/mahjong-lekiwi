"""The wrist camera, at the resolution and rate every dataset was recorded at (640x480, 30fps).

It can stop giving frames mid-session (2026-10-28: from the 3rd of 10 bench trials every read failed;
the same during wrist_capture.py). No frames for LOST_S means it's lost: every script then releases it
and opens it again with hardware.reopen_wrist_camera, and redoes what it was doing.
"""
import os
import time

import cv2

from . import config

LOST_S = 1.0       # no wrist frames for this long: the camera is lost, reopen it
# Reruns of a pickup the camera stopped (policy.run's stop_reason "camera") before it counts as a miss. A camera sending torn
# frames reopens at once with no prompt (hardware.reopen_wrist_camera), so without a limit the same pick reran with no end.
RERUNS = 1

# A torn frame: the camera keeps giving frames but the USB link lost data, so bands of rows are shifted or smeared
# (YUYV, no JPEG to fail to decode). 2026-11-13 (20261113_100710_bench): 34 attempts with torn frames picked 10, the 19
# without them 18, both policies closing 35-78mm in the air. A tear changes whole rows at once from one frame to the
# next, which the arm's own motion at 30fps doesn't: in 1591 wrist videos, sustained tears had up to 100% of rows jump,
# clean ones at most 6% (20261112_174429_bench), and motion blur on a fast lift 1-3 frames.
TORN_ROW_JUMP = 40          # grey levels: a row's mean change from the previous frame that counts as a jump
TORN_ROWS = 0.3             # the share of rows that jump in a torn frame
TORN_STOP_FRAMES = 5        # torn frames before the hold that stop a pickup: no blur reached 4, every sustained tear did


def torn(previous, frame):
    """Whether `frame` (BGR) tore from `previous` (the frame before it): TORN_ROWS of its rows jumped."""
    if previous is None or previous.shape != frame.shape:
        return False
    grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (previous, frame)]
    rows = cv2.reduce(cv2.absdiff(*grey), 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F)
    return float((rows > TORN_ROW_JUMP).mean()) >= TORN_ROWS


class WristCamera:
    label = "wrist camera"

    def __init__(self, path=config.WRIST_CAM):
        if not os.path.exists(path):
            raise RuntimeError(f"{self.label} {path} not found — check the udev rule / replug it")
        self.cap = cv2.VideoCapture(os.path.realpath(path), cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"{self.label} {path} failed to open")
        width, height = config.WRIST_RESOLUTION
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)

    def read(self):
        """(ok, BGR frame)"""
        return self.cap.read()

    def flush(self, frames=10):
        """Drop buffered frames so the next read is current."""
        for _ in range(frames):
            self.cap.grab()

    def release(self):
        self.cap.release()


class BaseCamera(WristCamera):
    """The LeKiwi base's front camera, same model and settings as the wrist camera (docs/face_down_plan.md)."""
    label = "base camera"

    def __init__(self, path=config.BASE_CAM):
        super().__init__(path)


def read_within(camera, lost_s=LOST_S, clock=time.monotonic):
    """A frame from `camera`, trying for up to lost_s: (True, frame), or (False, None) if it's lost. For
    reads outside a control loop (a loop counts its own failed reads against LOST_S)."""
    end = clock() + lost_s
    while True:
        ok, frame = camera.read()
        if ok:
            return True, frame
        if clock() >= end:
            return False, None


def grab_frames(camera, n):
    """Up to n current frames: fewer if the camera is lost (no frame within LOST_S)."""
    camera.flush()
    frames = []
    for _ in range(n):
        ok, frame = read_within(camera)
        if not ok:
            break
        frames.append(frame)
    return frames
