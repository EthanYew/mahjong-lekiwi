"""Tile detection with the RealSense behind the arm: YOLO boxes, depth at the box centre, through the
hand-eye transform into the arm's base frame; tile heights come from the table plane.

One RealSense pipeline is opened per session and kept (MahjongBot hardware-reset the camera before
every detection, ~5s each, because its recorder handed the device back and forth with lerobot's
camera; nothing here does that).
"""
import sys
import time
from collections import defaultdict

import numpy as np

from . import config

MIN_FRAMES_SEEN = 2     # of a scan's frames: a tile in just one is more likely a flicker than a tile


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
        profile = self.pipeline.start(cfg)
        self.depth_scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = {"fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy}   # colour = aligned depth
        self.align = rs.align(rs.stream.color)
        self.filters = [rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()]
        self.T_cam_to_base = np.load(config.T_CAM_TO_BASE)
        self._model = None
        self._boxes_of = (None, [])  # the last frame tile_boxes ran on, and its boxes
        self.last_frames = []       # the latest scan's frames and boxes (detector.save_scan)
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

    def depth_snapshot(self):
        """(BGR colour frame, depth in metres aligned to it; 0 where there's no reading)."""
        color, depth = self._frames()
        return color.copy(), np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale_m

    @property
    def names(self):
        """The detector's class names, in class order (once it has loaded)."""
        return [self._model.names[i] for i in sorted(self._model.names)] if self._model else []

    def _load_detector(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(config.TILE_DETECTOR))

    def tile_boxes(self, image_bgr):
        """The confident detector boxes in a frame, [(x1, y1, x2, y2)], for layout.find_tile. Names aren't used: only
        that a tile is there. One detector run per frame, however many spots ask (~7ms on the GPU)."""
        if self._boxes_of[0] is not image_bgr:
            self._load_detector()
            boxes = [b["xyxy"] for b in run_detector(self._model, image_bgr) if b["confidence"] >= config.DETECTION_CONFIDENCE]
            self._boxes_of = (image_bgr, boxes)
        return self._boxes_of[1]

    def detect_tiles(self, n_frames=config.DETECTION_FRAMES):
        """Tiles seen over n_frames, merged into one entry per physical tile by a vote (merge_tiles), which
        counts every box down to config.DETECTION_SAVE_CONFIDENCE. Each frame, its filtered depth and
        every box YOLO found in it stay in self.last_frames, for the detector's data (detector.save_scan)."""
        self._load_detector()
        raw, self.last_frames = [], []
        for k in range(n_frames):
            color, depth = self._frames()
            intrinsics = depth.profile.as_video_stream_profile().intrinsics
            boxes = run_detector(self._model, color)
            self.last_frames.append({"color": color.copy(), "depth": np.asanyarray(depth.get_data()).copy(),
                                     "depth_scale_m": self.depth_scale_m, "boxes": boxes})
            for box in boxes:
                x1, y1, x2, y2 = box["xyxy"]
                u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
                d = robust_depth(depth, u, v)
                if d == 0.0:
                    continue
                p_cam = np.array([(u - intrinsics.ppx) * d / intrinsics.fx, (v - intrinsics.ppy) * d / intrinsics.fy, d, 1.0])
                x, y = (self.T_cam_to_base @ p_cam)[:2]
                raw.append({"tile_name": box["tile_name"], "confidence": box["confidence"], "frame": k,
                            "xy": (float(x), float(y)), "pixel": (u, v), "box_px": (float(x2 - x1), float(y2 - y1))})
        return merge_tiles(raw, n_frames)


def run_detector(model, color):
    """Every box the YOLO `model` finds in a BGR frame, down to config.DETECTION_SAVE_CONFIDENCE:
    [{"tile_name", "confidence", "xyxy"}]. The live scan and the detector's exam both go through here."""
    return [{"tile_name": model.names[int(box.cls[0])], "confidence": round(float(box.conf[0]), 4),
             "xyxy": [round(float(c), 1) for c in box.xyxy[0]]}
            for box in model(color, conf=config.DETECTION_SAVE_CONFIDENCE, iou=0.5, verbose=False)[0].boxes]


def robust_depth(depth_frame, u, v, window=5):
    """The median depth around pixel (u, v), over the samples inside the frame: a box centred within `window`
    of the edge (a tile half in view) made get_distance raise and ended a bench session (2026-10-30)."""
    w, h = depth_frame.get_width(), depth_frame.get_height()
    depths = [depth_frame.get_distance(u + du, v + dv)
              for du in range(-window, window + 1, 2) for dv in range(-window, window + 1, 2)
              if 0 <= u + du < w and 0 <= v + dv < h]
    depths = [d for d in depths if d > 0.0]
    return float(np.median(depths)) if depths else 0.0


def merge_tiles(detections, n_frames=1, radius_m=config.TILE_MERGE_RADIUS_M, min_frames=MIN_FRAMES_SEEN,
                confident=config.DETECTION_CONFIDENCE):
    """One entry per physical tile: detections within radius_m of a cluster's first are the same tile
    (by position, not label — labels flicker between frames), position averaged. The name is a vote:
    the one with the most confidence summed over the frames; "vote_confidence" is that sum over the
    frames the tile was seen in, "confidence" stays its best single read (what interventions' 0.6 was
    measured on). A tile seen in fewer than min_frames frames, or voted below `confident`, is dropped.
    "votes" keeps every name's sum, so a split read (8C 2.46 vs 9C 1.53) can be told from a sure one.
    The single most confident read named a 9C as 8C on the exam (8C 0.69 vs 9C 0.68 in one frame); the
    vote doesn't name it 9C either (8C read in 4 of 5 frames) but leaves it out. v6x3 on the exam's
    scans: 98.0% right both ways, mix-ups 1 -> 0, found 99.0% -> 98.0% (2026-10-28)."""
    remaining, merged = list(detections), []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed] + [t for t in remaining if np.hypot(t["xy"][0] - seed["xy"][0], t["xy"][1] - seed["xy"][1]) <= radius_m]
        remaining = [t for t in remaining if t not in cluster]
        seen = {t.get("frame", i) for i, t in enumerate(cluster)}
        votes = defaultdict(float)
        for t in cluster:
            votes[t["tile_name"]] += t["confidence"]
        name = max(votes, key=votes.get)
        confidence = votes[name] / len(seen)
        if len(seen) < min(min_frames, n_frames) or confidence < confident:
            continue
        best = max((t for t in cluster if t["tile_name"] == name), key=lambda t: t["confidence"])
        x, y = float(np.mean([t["xy"][0] for t in cluster])), float(np.mean([t["xy"][1] for t in cluster]))
        merged.append({**best, "xy": (x, y), "z": config.grasp_z(x, y), "vote_confidence": round(confidence, 4),
                       "reads": len(cluster), "frames_seen": len(seen),
                       "votes": {n: round(v, 4) for n, v in sorted(votes.items(), key=lambda nv: -nv[1])}})
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
