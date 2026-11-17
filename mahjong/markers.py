"""The drop zone for drawn tiles, taught by an ArUco marker (docs/autonomy_plan.md, section 2).

Teaching (scripts/play.py --teach-zone, or the first run) reads the uncovered marker (id 3, DICT_6X6_250, flat
on the mat) and saves its position (data/drop_zone.json) with the calibration fingerprint it was found under: a
recalibrated RealSense makes the saved position mean nothing. Every hand after that goes on the saved position, and
nothing reads the marker: it can move or come off the mat without play noticing. A moved marker used to stop play as
if the camera had moved (20261104_220329_play), which is setup_check.py's question.

A marker left in place can have the drawn tile on top of it, with two things to watch that would mean taking it away:
first-try pickups at the zone below the ~95% seen on the mat (no policy has seen a tile on a black-and-white
pattern), and the detector (no marker class) reading the marker's edge as a tile.

Play reaches the zone only once a hand is under way, so it's checked on the bench beforehand: the test
layout ZONE_TEST_LAYOUT (even-hand14) is the play layout's slots plus one spot on the zone, and play.py won't
start until the bench has picked a tile up there and set it back down (`zone_test`).
"""
import json
import math
import statistics
from datetime import datetime

import cv2
import numpy as np

from . import config
from .hover_correction import setup_fingerprint
from .layout import pixel_to_table
from .return_correction import landings
from .session import counted_real, read_events

DRAWN_TILE_MARKER_ID = 3
DICTIONARY = cv2.aruco.DICT_6X6_250
ZONE_FILE = config.DATA / "drop_zone.json"
ZONE_RADIUS_M = config.TILE_MERGE_RADIUS_M   # a drawn tile this close to the zone is on it (and pickable by spot)
LOCATE_FRAMES = 10
PLAY_LAYOUT = "even-hand13"             # the hand's slots on the standard layout (data/standard_setup.json)
ZONE_TEST_LAYOUT = "even-hand14"        # the play layout's slots + one spot on the zone, walked by the bench
SLOT_MATCH_M = 0.005                    # a test-layout spot this close to a play slot is that slot
CARRY_MIN_LANDINGS = 2                  # carried from the zone on the bench before play starts (--trials 2 of --carry)


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


ZONE_ID = "zone"


def zone_spot():
    """The taught zone as a layout spot ({"id": "zone", "xy", "yaw_deg"}), for the scripts that pick from it or set a tile on
    it by saved position (record.py --wall-station zone, crude_run.py); None if none is taught for this calibration."""
    zone = load_zone()
    return None if zone is None else {"id": ZONE_ID, "xy": [float(v) for v in zone["xy"]], "yaw_deg": 0.0}


def benched_at(xy, sessions_dir=None, radius_m=ZONE_RADIUS_M):
    """Bench attempts at a layout spot within radius_m of xy, over every bench session: (attempts, picked up,
    returns whose landing was scanned). A simulated bench says nothing about the real zone, and a voided one (data/void_sessions.json) nothing at all."""
    attempts = picked = returned = 0
    for events_file in sorted((sessions_dir or config.SESSIONS).glob("*_bench/events.jsonl")):
        events = read_events(events_file.parent)
        if not counted_real(events_file.parent):
            continue
        here =[e for e in events if e["type"] == "attempt" and e.get("spot_xy")
                and math.dist(e["spot_xy"], xy) <= radius_m]
        held = {e.get("attempt_dir") for e in here if e.get("success")} - {None}
        attempts += len(here)
        picked += len(held)
        returned += sum(e["type"] == "return_landed" and e.get("attempt_dir") in held for e in events)
    return attempts, picked, returned


def carried_from(xy, sessions_dir=None, radius_m=ZONE_RADIUS_M):
    """How far from home each tile carried from xy landed, in real bench sessions (bench.py --carry): metres, inf where
    the scan didn't find it. Landings from before carried tiles had their own aim, or a person's hands in, don't count. One
    logged as not seen is scored again from its scan, with today's matching (a misread once lost a landing, nearest_tile)."""
    errors = []
    for events_file in sorted((sessions_dir or config.SESSIONS).glob("*_bench/events.jsonl")):
        events = read_events(events_file.parent)
        if not counted_real(events_file.parent):
            continue
        scored = {r["attempt_dir"]: r["error_m"] for r in landings(events)}
        for e in events:
            if e["type"] == "return_landed" and e.get("carried") and not e.get("by_hand") and math.dist(e["from_xy"], xy) <= radius_m:
                error = e["error_m"] if e.get("error_m") is not None else scored.get(e["attempt_dir"])
                errors.append(math.inf if error is None else error)
    return errors


def zone_test(zone_xy, play_spots, test_spots, sessions_dir=None, play_layout=PLAY_LAYOUT,
              test_layout=ZONE_TEST_LAYOUT):
    """Is the spot a drawn tile goes on checked on the bench? The test layout should hold every play slot plus
    one spot on the zone, the bench should have picked a tile up there and returned it, and carried tiles from there
    to a slot and seen them land near it: no other bench walk does what a hand's first turn does. Returns {"ok", "spot",
    "attempts", "picked", "returned", "carried", "carried_median_mm", "problems": [...], "fix": [commands]}."""
    problems, fix = [], []
    missing = [s["id"] for s in play_spots
               if not any(math.dist(s["xy"], t["xy"]) <= SLOT_MATCH_M for t in test_spots)]
    if missing:
        problems.append(f"{test_layout} is missing {play_layout}'s slots {', '.join(missing)} (re-taught since?)")
        fix.append(f"python scripts/layout.py --name {test_layout} --teach   # {play_layout}'s slots + a tile on the zone")
    extra = [t for t in test_spots if not any(math.dist(t["xy"], s["xy"]) <= SLOT_MATCH_M for s in play_spots)]
    spot = next((t for t in extra if math.dist(t["xy"], zone_xy) <= ZONE_RADIUS_M), None)
    closest = None
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
    landed = carried_from(zone_xy, sessions_dir)
    median_mm = statistics.median(landed) * 1000 if landed else None
    if len(landed) < CARRY_MIN_LANDINGS or median_mm > config.RETURN_DRIFT_WARN_M * 1000:
        unseen = sum(math.isinf(error) for error in landed)
        problems.append(f"tiles carried from the drop zone to a slot on the bench: {len(landed)} landed"
                        + (f", {unseen} not seen near home by the scan after" if unseen else "")
                        + (f", median {median_mm:.0f}mm from home" if landed and not math.isinf(median_mm) else "")
                        + f" (needs {CARRY_MIN_LANDINGS}+, median {config.RETURN_DRIFT_WARN_M * 1000:.0f}mm or less)")
        from_id = (spot or closest or {"id": "S13"})["id"]
        fix.append(f"python scripts/bench.py --layout {test_layout} --trials 2 --carry {from_id},{play_spots[0]['id']}"
                   f"   # a tile on {from_id}, {play_spots[0]['id']} empty")
    return {"ok": not problems, "spot": spot["id"] if spot else None, "attempts": attempts, "picked": picked,
            "returned": returned, "carried": len(landed),
            "carried_median_mm": None if median_mm is None or math.isinf(median_mm) else median_mm,
            "problems": problems, "fix": fix}
