"""The drop zone for drawn tiles, taught by an ArUco marker (docs/autonomy_plan.md, section 2).

The marker lies flat on the mat and the drawn tile goes on top of it, as in MahjongBot's autonomous play
(marker id 3, DICT_6X6_250). A tile on it hides it, so its position is found while it's uncovered and
saved (data/drop_zone.json) with the calibration fingerprint it was found under: a recalibrated RealSense
makes the saved position mean nothing. scripts/play.py looks again each time the zone is empty, and a
marker that moved more than ZONE_MOVED_M means the camera or the LeKiwi base moved: it stops.

Under the tile was chosen to match MahjongBot, with two things to watch that would move it beside the
zone instead: first-try pickups at the zone below the ~95% seen on the mat (no policy has seen a tile on
a black-and-white pattern), and the detector (no marker class) reading the marker's edge as a tile.

Play reaches the zone only once a hand is under way, so it's checked on the bench beforehand: the test
layout ZONE_TEST_LAYOUT (even-hand14) is the play layout's slots plus one spot on the zone, and play.py won't
start until the bench has picked a tile up there and set it back down (`zone_test`).
"""
import json
import math
from datetime import datetime

import cv2
import numpy as np

from . import config
from .hover_correction import setup_fingerprint
from .layout import pixel_to_table
from .session import read_events

DRAWN_TILE_MARKER_ID = 3
DICTIONARY = cv2.aruco.DICT_6X6_250
ZONE_FILE = config.DATA / "drop_zone.json"
ZONE_RADIUS_M = config.TILE_MERGE_RADIUS_M   # a drawn tile this close to the zone is on it (and pickable by spot)
ZONE_MOVED_M = 0.005                    # the uncovered marker read this far from where it was taught: the camera moved
LOCATE_FRAMES = 10
PLAY_LAYOUT = "even-hand13"             # the hand's slots on the standard layout (data/standard_setup.json)
ZONE_TEST_LAYOUT = "even-hand14"        # the play layout's slots + one spot on the zone, walked by the bench
SLOT_MATCH_M = 0.005                    # a test-layout spot this close to a play slot is that slot


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


def benched_at(xy, sessions_dir=None, radius_m=ZONE_RADIUS_M):
    """Bench attempts at a layout spot within radius_m of xy, over every bench session: (attempts, picked up,
    returns whose landing was scanned)."""
    attempts = picked = returned = 0
    for events_file in sorted((sessions_dir or config.SESSIONS).glob("*_bench/events.jsonl")):
        events = read_events(events_file.parent)
        here = [e for e in events if e["type"] == "attempt" and e.get("spot_xy")
                and math.dist(e["spot_xy"], xy) <= radius_m]
        held = {e.get("attempt_dir") for e in here if e.get("success")} - {None}
        attempts += len(here)
        picked += len(held)
        returned += sum(e["type"] == "return_landed" and e.get("attempt_dir") in held for e in events)
    return attempts, picked, returned


def zone_test(zone_xy, play_spots, test_spots, sessions_dir=None, play_layout=PLAY_LAYOUT,
              test_layout=ZONE_TEST_LAYOUT):
    """Is the spot a drawn tile goes on checked on the bench? The test layout should hold every play slot plus
    one spot on the zone, and the bench should have picked a tile up there and returned it. Returns
    {"ok", "spot", "attempts", "picked", "returned", "problems": [...], "fix": [commands]}."""
    problems, fix = [], []
    missing = [s["id"] for s in play_spots
               if not any(math.dist(s["xy"], t["xy"]) <= SLOT_MATCH_M for t in test_spots)]
    if missing:
        problems.append(f"{test_layout} is missing {play_layout}'s slots {', '.join(missing)} (re-taught since?)")
        fix.append(f"python scripts/layout.py --name {test_layout} --teach   # {play_layout}'s slots + a tile on the zone")
    extra = [t for t in test_spots if not any(math.dist(t["xy"], s["xy"]) <= SLOT_MATCH_M for s in play_spots)]
    spot = next((t for t in extra if math.dist(t["xy"], zone_xy) <= ZONE_RADIUS_M), None)
    if spot is None:
        closest = min(extra, key=lambda t: math.dist(t["xy"], zone_xy), default=None)
        problems.append(f"no {test_layout} spot on the drop zone at ({zone_xy[0]:.3f}, {zone_xy[1]:+.3f})"
                        + (f": {closest['id']} is {math.dist(closest['xy'], zone_xy) * 1000:.0f}mm away"
                           if closest else ""))
        spot_id = closest["id"] if closest else "Z1"
        if closest:
            fix.append(f"python scripts/layout.py --name {test_layout} --remove {spot_id}")
        fix.append(f"python scripts/layout.py --name {test_layout} --add {spot_id} {zone_xy[0]:.4f} {zone_xy[1]:.4f}")
    attempts, picked, returned = benched_at(zone_xy, sessions_dir)
    if not (picked and returned):
        problems.append(f"the bench hasn't picked up and returned a tile on the drop zone ({attempts} attempt(s), "
                        f"{picked} picked up, {returned} returned)")
        fix.append(f"python scripts/bench.py --layout {test_layout} --trials 2   # a tile on every spot, the zone's too")
    return {"ok": not problems, "spot": spot["id"] if spot else None, "attempts": attempts, "picked": picked,
            "returned": returned, "problems": problems, "fix": fix}
