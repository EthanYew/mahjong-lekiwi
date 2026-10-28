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


class WristCamera:
    def __init__(self, path=config.WRIST_CAM):
        if not os.path.exists(path):
            raise RuntimeError(f"wrist camera {path} not found — check the udev rule / replug it")
        self.cap = cv2.VideoCapture(os.path.realpath(path), cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"wrist camera {path} failed to open")
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
