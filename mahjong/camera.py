"""The wrist camera, at the resolution and rate every dataset was recorded at (640x480, 30fps)."""
import os

import cv2

from . import config


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
