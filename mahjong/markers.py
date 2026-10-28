"""The drop zone for drawn tiles, taught by an ArUco marker (docs/autonomy_plan.md, section 2).

The marker lies flat on the mat and the drawn tile goes on top of it, as in MahjongBot's autonomous play
(marker id 3, DICT_6X6_250). A tile on it hides it, so its position is found while it's uncovered and
saved (data/drop_zone.json) with the calibration fingerprint it was found under: a recalibrated RealSense
makes the saved position mean nothing. scripts/play.py looks again each time the zone is empty, and a
marker that moved more than ZONE_MOVED_M means the camera or the LeKiwi base moved: it stops.

Under the tile was chosen to match MahjongBot, with two things to watch that would move it beside the
zone instead: first-try pickups at the zone below the ~95% seen on the mat (no policy has seen a tile on
a black-and-white pattern), and the detector (no marker class) reading the marker's edge as a tile.
"""
import json
import math
from datetime import datetime

import cv2
import numpy as np

from . import config
from .hover_correction import setup_fingerprint
from .layout import pixel_to_table

DRAWN_TILE_MARKER_ID = 3
DICTIONARY = cv2.aruco.DICT_6X6_250
ZONE_FILE = config.DATA / "drop_zone.json"
ZONE_RADIUS_M = config.TILE_MERGE_RADIUS_M   # a drawn tile this close to the zone is on it (and pickable by spot)
ZONE_MOVED_M = 0.005                    # the uncovered marker read this far from where it was taught: the camera moved
LOCATE_FRAMES = 10


def find(image_bgr, marker_id=DRAWN_TILE_MARKER_ID):
    """The marker's 4 corners in pixels (4 x 2), or None."""
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if i == marker_id:
            return c.reshape(4, 2)
    return None


def centre_xy(corners, intrinsics, T_cam_to_base):
    """Table-frame (x, y) of the marker's centre: it lies on the table, not at a tile's top face."""
    point = pixel_to_table(np.mean(corners, axis=0), intrinsics, T_cam_to_base, z_of_xy=config.table_z)[0]
    return [float(point[0]), float(point[1])]


def locate(realsense, marker_id=DRAWN_TILE_MARKER_ID, frames=LOCATE_FRAMES):
    """The marker's table (x, y), the median over `frames` RealSense frames it was seen in, or None."""
    seen = []
    for _ in range(frames):
        corners = find(realsense.snapshot(), marker_id)
        if corners is not None:
            seen.append(centre_xy(corners, realsense.intrinsics, realsense.T_cam_to_base))
    return [float(v) for v in np.median(seen, axis=0)] if seen else None


def save_zone(xy, path=None, **provenance):
    zone = {"xy": [round(float(v), 4) for v in xy], "marker_id": DRAWN_TILE_MARKER_ID, "setup": setup_fingerprint(),
            "taught": datetime.now().isoformat(timespec="seconds"), **provenance}
    (path or ZONE_FILE).write_text(json.dumps(zone, indent=2) + "\n")
    return zone


def load_zone(path=None):
    """The taught zone, or None if there's none or it was taught under another calibration."""
    path = path or ZONE_FILE
    if not path.exists():
        return None
    zone = json.loads(path.read_text())
    return zone if zone.get("setup") == setup_fingerprint() else None


def moved(zone, xy, limit_m=ZONE_MOVED_M):
    return math.dist(zone["xy"], xy) > limit_m
