"""Tile detection with the RealSense behind the arm: YOLO boxes, depth at the box centre, through the
hand-eye transform into the arm's base frame; tile heights come from the table plane.

One RealSense pipeline is opened per session and kept (MahjongBot hardware-reset the camera before
every detection, ~5s each, because its recorder handed the device back and forth with lerobot's
camera; nothing here does that).
"""
import sys
import time

import numpy as np

from . import config


class RealSense:
    """Depth + colour RealSense pipeline, aligned to colour, with the standard filter chain."""

    def __init__(self, reset=True, warmup_frames=30):
        import pyrealsense2 as rs
        self.rs = rs
        if reset:   # clears a device left wedged by a previous crash
            for device in rs.context().query_devices():
                device.hardware_reset()
            time.sleep(5)
        width, height = config.REALSENSE_RESOLUTION
        cfg = rs.config()
        cfg.enable_device(config.REALSENSE_SERIAL)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, config.FPS)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, config.FPS)
        self.pipeline = rs.pipeline()
        self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.filters = [rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()]
        self.T_cam_to_base = np.load(config.T_CAM_TO_BASE)
        self._model = None
        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames()

    def close(self):
        self.pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _frames(self):
        aligned = self.align.process(self.pipeline.wait_for_frames(timeout_ms=10000))
        depth = aligned.get_depth_frame()
        for f in self.filters:
            depth = f.process(depth)
        return np.asanyarray(aligned.get_color_frame().get_data()), depth.as_depth_frame()

    def snapshot(self):
        """A BGR colour frame (for session logs)."""
        return self._frames()[0].copy()

    def detect_tiles(self, n_frames=config.DETECTION_FRAMES):
        """Tiles seen over n_frames, merged into one entry per physical tile."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(config.TILE_DETECTOR))
        raw = []
        for _ in range(n_frames):
            color, depth = self._frames()
            intrinsics = depth.profile.as_video_stream_profile().intrinsics
            for box in self._model(color, iou=0.5, verbose=False)[0].boxes:
                confidence = float(box.conf[0])
                if confidence < config.DETECTION_CONFIDENCE:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
                d = robust_depth(depth, u, v)
                if d == 0.0:
                    continue
                p_cam = np.array([(u - intrinsics.ppx) * d / intrinsics.fx, (v - intrinsics.ppy) * d / intrinsics.fy, d, 1.0])
                x, y = (self.T_cam_to_base @ p_cam)[:2]
                raw.append({"tile_name": self._model.names[int(box.cls[0])], "confidence": confidence,
                            "xy": (float(x), float(y)), "pixel": (u, v), "box_px": (float(x2 - x1), float(y2 - y1))})
        return merge_tiles(raw)


def robust_depth(depth_frame, u, v, window=5):
    depths = [depth_frame.get_distance(u + du, v + dv)
              for du in range(-window, window + 1, 2) for dv in range(-window, window + 1, 2)]
    depths = [d for d in depths if d > 0.0]
    return float(np.median(depths)) if depths else 0.0


def merge_tiles(detections, radius_m=config.TILE_MERGE_RADIUS_M):
    """One entry per physical tile: detections within radius_m of a cluster's first are the same tile
    (by position, not label — labels flicker between frames). Label from the most confident read,
    position averaged."""
    remaining, merged = list(detections), []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed] + [t for t in remaining if np.hypot(t["xy"][0] - seed["xy"][0], t["xy"][1] - seed["xy"][1]) <= radius_m]
        remaining = [t for t in remaining if t not in cluster]
        best = max(cluster, key=lambda t: t["confidence"])
        x, y = float(np.mean([t["xy"][0] for t in cluster])), float(np.mean([t["xy"][1] for t in cluster]))
        merged.append({**best, "xy": (x, y), "z": config.grasp_z(x, y), "reads": len(cluster)})
    return sorted(merged, key=lambda t: -t["xy"][1])      # left to right


def drain_stdin():
    """Drop keystrokes buffered while a keyboard listener was active (arrow keys leak escape codes)."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def choose_tile(tiles, prompt="Which tile? (number or name, Enter to cancel): ", tile_name=None):
    """A tile from `tiles`: by name if tile_name is given (most confident match), else interactively."""
    if tile_name is not None:
        matches = [t for t in tiles if t["tile_name"].lower() == tile_name.lower()]
        return max(matches, key=lambda t: t["confidence"]) if matches else None
    for i, t in enumerate(tiles, start=1):
        print(f"  [{i}] {t['tile_name']:<4} conf {t['confidence']:.2f}  ({t['xy'][0]:.3f}, {t['xy'][1]:+.3f})")
    while True:
        drain_stdin()
        choice = input(prompt).strip()
        if not choice:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(tiles):
            return tiles[int(choice) - 1]
        matches = [t for t in tiles if t["tile_name"].lower() == choice.lower()]
        if len(matches) == 1:
            return matches[0]
        print(f"  {'several' if matches else 'no'} tiles named {choice!r} — use the number")
