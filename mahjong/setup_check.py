"""Is the workspace back where it was? Measurements for scripts/setup_check.py, each compared with a
reference taken once on a known-good setup (data/setup_reference/).

The policies, the hover correction and the return correction all assume the RealSense, the arm and the table
sit where they did when they were recorded and fitted (assets/T_cam_to_base.npy, config.table_z). After a
pack-up nothing says whether they do; a few mm off and every hover lands off-target without an error. So:

- camera: where the robot's base appears in the RealSense image vs the reference (features matched in the
  robot's box only: tiles, the keyboard and the person move between sessions, the arm's base doesn't).
  1px there is ~1.5mm on the mat (0.87m away, fx ~590).
- table: the mat's depth through T_cam_to_base vs config.table_z — catches a different table, a tilt or a
  thicker mat (and a moved camera).
- tile test: one tile placed to green on a few spots, the arm hovers over it as the bench does (measured
  position + hover correction) and the wrist camera looks. The tile landing where it did in the reference
  wrist frame means camera, table, arm calibration, hover correction and wrist camera mount all still
  agree — and it's the very image the policy starts from.
- parked pose: where the unpowered arm settles onto its stops vs the reference; a recalibrated or replaced
  servo shows up there (encoders alone can't tell).
- brightness: of the mat (RealSense) and the wrist view, vs the reference (lighting is part of the
  policy's input).
"""
import json
import math
from datetime import datetime

import cv2
import numpy as np

from . import config
from .hover_correction import setup_fingerprint
from .layout import FACE_MAX_S, FACE_MIN_V, table_to_pixel, wrap_yaw

REFERENCE_DIR = config.DATA / "setup_reference"

# The robot's box in the base frame (the LeKiwi base and the folded arm at rest): its projection is where
# the camera check looks for features.
ROBOT_BOX_X, ROBOT_BOX_Y, ROBOT_BOX_HEIGHT_M = (-0.12, 0.05), (-0.15, 0.15), 0.12
MIN_INLIERS = 12                    # matched features agreeing on one shift, for the camera check to count
WORKSPACE_CENTRE = (0.20, 0.0)      # where pixel shifts are turned into mm

# Table samples: a grid over the mat the policies work on.
TABLE_GRID_X, TABLE_GRID_Y, TABLE_GRID_STEP_M = (0.12, 0.28), (-0.20, 0.20), 0.02
TABLE_OUTLIER_MM = 8.0              # samples this far from the median are tiles or clutter, left out

# Wrist view: the tile's white face, as layout.find_tile finds it from the RealSense.
WRIST_AREA_RANGE = (1500, 150000)   # px; at the 40mm hover the face is ~140 x 150px
WRIST_MIN_FILL = 0.7                # contour area / its rotated box: a tile, not glare

# (warn, fail) thresholds. Jaws tolerate about ±6mm, so 5-6mm is where a pickup starts to miss.
LIMITS = {
    "camera_mm": (2.0, 5.0),        # robot's image shift turned into mm on the mat
    "camera_deg": (0.5, 1.5),       # image rotation
    "table_mm": (3.0, 6.0),         # mat height change at the workspace centre
    "table_tilt_mm": (2.0, 4.0),    # change in height difference per 10cm
    "wrist_mm": (3.0, 6.0),         # tile's position in the wrist view at the hover
    "wrist_size_pct": (10.0, 20.0), # its size there: the hover's height (closer = bigger)
    "wrist_deg": (5.0, 10.0),       # its angle there: the wrist roll
    "parked_deg": (3.0, 6.0),       # unpowered settle, worst joint
}
BRIGHTNESS_RANGE = (0.7, 1.4)       # live / reference; outside it's a warning (lighting changed)
STATUSES = ("pass", "warn", "fail", "skipped")


def grade(value, limit):
    """pass / warn / fail for |value| against LIMITS[limit]."""
    warn, fail = LIMITS[limit]
    return "fail" if abs(value) >= fail else "warn" if abs(value) >= warn else "pass"


def worst(statuses):
    """The worst of pass / warn / fail ("skipped" counts as nothing; all skipped is "skipped")."""
    ranked = [s for s in statuses if s in ("pass", "warn", "fail")]
    return max(ranked, key=("pass", "warn", "fail").index) if ranked else "skipped"


# --- camera: the robot's place in the RealSense image ---

def robot_region(intrinsics, T_cam_to_base, shape):
    """(u0, v0, u1, v1): the robot box's projection, clipped to the image."""
    z0 = config.table_z(0.0, 0.0)
    corners = [(x, y, z) for x in ROBOT_BOX_X for y in ROBOT_BOX_Y for z in (z0, z0 + ROBOT_BOX_HEIGHT_M)]
    px = table_to_pixel(corners, intrinsics, T_cam_to_base)
    h, w = shape[:2]
    u0, v0 = np.clip(np.floor(px.min(axis=0)).astype(int), 0, [w, h])
    u1, v1 = np.clip(np.ceil(px.max(axis=0)).astype(int), 0, [w, h])
    return int(u0), int(v0), int(u1), int(v1)


def mm_per_px(intrinsics, T_cam_to_base):
    """Metres on the mat per image pixel at the workspace centre (its depth / fx), times 1000."""
    x, y = WORKSPACE_CENTRE
    depth = (np.linalg.inv(T_cam_to_base) @ [x, y, config.table_z(x, y), 1.0])[2]
    return float(1000 * depth / intrinsics["fx"])


def _grey(image_bgr):
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)       # evens out lighting changes


def image_shift(reference_bgr, live_bgr, region):
    """How the robot's region moved from the reference to the live frame: {"dx_px", "dy_px", "shift_px",
    "rotation_deg", "inliers"}, or {"inliers": n} if too few features agree (MIN_INLIERS)."""
    u0, v0, u1, v1 = region
    mask = np.zeros(reference_bgr.shape[:2], np.uint8)
    mask[v0:v1, u0:u1] = 255
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
    kp_ref, des_ref = orb.detectAndCompute(_grey(reference_bgr), mask)
    kp_live, des_live = orb.detectAndCompute(_grey(live_bgr), mask)
    if des_ref is None or des_live is None or len(kp_ref) < MIN_INLIERS or len(kp_live) < MIN_INLIERS:
        return {"inliers": 0}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_ref, des_live)
    if len(matches) < MIN_INLIERS:
        return {"inliers": len(matches)}
    src = np.float32([kp_ref[m.queryIdx].pt for m in matches])
    dst = np.float32([kp_live[m.trainIdx].pt for m in matches])
    affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    n = int(inliers.sum()) if inliers is not None else 0
    if affine is None or n < MIN_INLIERS:
        return {"inliers": n}
    centre = np.array([(u0 + u1) / 2, (v0 + v1) / 2])
    dx, dy = affine[:, :2] @ centre + affine[:, 2] - centre
    return {"dx_px": float(dx), "dy_px": float(dy), "shift_px": float(math.hypot(dx, dy)),
            "rotation_deg": float(math.degrees(math.atan2(affine[1, 0], affine[0, 0]))), "inliers": n}


def camera_result(shift, scale_mm_per_px):
    """The camera check from image_shift(): mm on the mat, and its status."""
    if "shift_px" not in shift:
        return {**shift, "status": "warn", "why": f"only {shift['inliers']} features matched in the robot's "
                                                  f"region (need {MIN_INLIERS}): is the arm at rest and in view?"}
    shift_mm = shift["shift_px"] * scale_mm_per_px
    status = worst([grade(shift_mm, "camera_mm"), grade(shift["rotation_deg"], "camera_deg")])
    return {**shift, "shift_mm": shift_mm, "status": status}


def camera_text(result):
    if "shift_mm" not in result:
        return result["why"]
    return (f"robot moved {result['dx_px']:+.1f}px, {result['dy_px']:+.1f}px in the image (~{result['shift_mm']:.1f}mm "
            f"on the mat), turned {result['rotation_deg']:+.2f}° ({result['inliers']} features)")


def alignment_view(reference_bgr, live_bgr, region, result):
    """Live frame with the reference blended in, the robot's region boxed in the status colour and the shift
    as text: nudge the tripod until it's green."""
    colour = {"pass": (0, 200, 0), "warn": (0, 190, 255), "fail": (0, 0, 255)}[result["status"]]
    out = cv2.addWeighted(live_bgr, 0.5, reference_bgr, 0.5, 0)
    u0, v0, u1, v1 = region
    cv2.rectangle(out, (u0, v0), (u1 - 1, v1 - 1), colour, 2)
    if "shift_px" in result:
        centre = np.array([(u0 + u1) / 2, (v0 + v1) / 2])
        tip = centre - 8 * np.array([result["dx_px"], result["dy_px"]])        # the way to move it back, x8
        cv2.arrowedLine(out, tuple(int(v) for v in centre), tuple(int(v) for v in tip), colour, 2, tipLength=0.3)
    cv2.putText(out, camera_text(result)[:90], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    cv2.putText(out, "arrow: the way to move the view back (x8). Enter: done, q: quit", (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


# --- table: the mat's depth vs config.table_z ---

def table_samples(depth_m, intrinsics, T_cam_to_base):
    """(N x 3) base-frame points where the grid over the mat meets the depth image, and each one's height
    above config.table_z in mm (N)."""
    xs = np.arange(TABLE_GRID_X[0], TABLE_GRID_X[1] + 1e-9, TABLE_GRID_STEP_M)
    ys = np.arange(TABLE_GRID_Y[0], TABLE_GRID_Y[1] + 1e-9, TABLE_GRID_STEP_M)
    grid = [(x, y, config.table_z(x, y)) for x in xs for y in ys]
    pixels = np.round(table_to_pixel(grid, intrinsics, T_cam_to_base)).astype(int)
    h, w = depth_m.shape
    points = []
    for u, v in pixels:
        if not (2 <= u < w - 2 and 2 <= v < h - 2):
            continue
        window = depth_m[v - 2:v + 3, u - 2:u + 3]
        window = window[window > 0]
        if not len(window):
            continue
        d = float(np.median(window))
        cam = [(u - intrinsics["ppx"]) * d / intrinsics["fx"], (v - intrinsics["ppy"]) * d / intrinsics["fy"], d, 1.0]
        points.append((T_cam_to_base @ cam)[:3])
    points = np.array(points).reshape(-1, 3)
    heights = np.array([1000 * (z - config.table_z(x, y)) for x, y, z in points])
    return points, heights


def table_fit(depth_m, intrinsics, T_cam_to_base):
    """The mat's height above config.table_z: {"offset_mm" at the workspace centre, "slope_x_mm" and
    "slope_y_mm" per 10cm, "samples", "outliers"}, or None if under 10 samples."""
    points, heights = table_samples(depth_m, intrinsics, T_cam_to_base)
    if len(points) < 10:
        return None
    keep = np.abs(heights - np.median(heights)) <= TABLE_OUTLIER_MM
    cx, cy = WORKSPACE_CENTRE
    A = np.c_[np.ones(keep.sum()), points[keep, 0] - cx, points[keep, 1] - cy]
    offset, sx, sy = np.linalg.lstsq(A, heights[keep], rcond=None)[0]
    return {"offset_mm": float(offset), "slope_x_mm": float(sx * 0.1), "slope_y_mm": float(sy * 0.1),
            "samples": int(keep.sum()), "outliers": int((~keep).sum())}


def table_result(live, reference):
    if live is None:
        return {"status": "warn", "why": "too few depth samples on the mat"}
    if reference is None:
        return {**live, "status": "skipped", "why": "no table in the reference"}
    change = {k: live[k] - reference[k] for k in ("offset_mm", "slope_x_mm", "slope_y_mm")}
    status = worst([grade(change["offset_mm"], "table_mm"), grade(change["slope_x_mm"], "table_tilt_mm"),
                    grade(change["slope_y_mm"], "table_tilt_mm")])
    return {**live, "change": change, "status": status}


def table_text(result):
    if "offset_mm" not in result:
        return result["why"]
    text = (f"mat {result['offset_mm']:+.1f}mm vs config.table_z, tilt {result['slope_x_mm']:+.1f} (x) / "
            f"{result['slope_y_mm']:+.1f} (y) mm per 10cm")
    if "change" in result:
        c = result["change"]
        text += f"; vs reference {c['offset_mm']:+.1f}mm, tilt {c['slope_x_mm']:+.1f} / {c['slope_y_mm']:+.1f}"
    return text


# --- brightness ---

def brightness(image_bgr, region=None):
    """Median brightness (HSV V, 0-255) of an image or a (u0, v0, u1, v1) region of it."""
    if region is not None:
        u0, v0, u1, v1 = region
        image_bgr = image_bgr[v0:v1, u0:u1]
    return float(np.median(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]))


def mat_region(intrinsics, T_cam_to_base, shape):
    """(u0, v0, u1, v1) around the table grid, for the mat's brightness."""
    corners = [(x, y, config.table_z(x, y)) for x in TABLE_GRID_X for y in TABLE_GRID_Y]
    px = table_to_pixel(corners, intrinsics, T_cam_to_base)
    h, w = shape[:2]
    u0, v0 = np.clip(np.floor(px.min(axis=0)).astype(int), 0, [w, h])
    u1, v1 = np.clip(np.ceil(px.max(axis=0)).astype(int), 0, [w, h])
    return int(u0), int(v0), int(u1), int(v1)


def brightness_result(live, reference):
    if not reference:
        return {"value": live, "status": "skipped"}
    ratio = live / reference
    status = "pass" if BRIGHTNESS_RANGE[0] <= ratio <= BRIGHTNESS_RANGE[1] else "warn"
    return {"value": live, "reference": reference, "ratio": ratio, "status": status}


# --- tile test: the tile in the wrist view at the hover ---

def wrist_tile(image_bgr):
    """The tile's face in a wrist frame: {"centre_px", "area_px", "angle_deg", "size_px"} for the white,
    tile-shaped blob nearest the image centre, or None."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= FACE_MIN_V) & (hsv[:, :, 1] <= FACE_MAX_S)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # the printed symbols
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_bgr.shape[:2]
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if not WRIST_AREA_RANGE[0] <= area <= WRIST_AREA_RANGE[1]:
            continue
        (cu, cv_), (bw, bh), angle = cv2.minAreaRect(contour)
        if area < WRIST_MIN_FILL * bw * bh:
            continue
        distance = math.hypot(cu - w / 2, cv_ - h / 2)
        if best is not None and distance >= best[0]:
            continue
        if bw < bh:                                    # angle of the long side
            angle += 90.0
        best = (distance, {"centre_px": [float(cu), float(cv_)], "area_px": float(area),
                           "angle_deg": wrap_yaw(float(angle)), "size_px": sorted([float(bw), float(bh)], reverse=True)})
    return best[1] if best else None


def wrist_result(live, reference):
    """Compare the tile in the wrist view with the reference's, at the same spot."""
    if live is None:
        return {"status": "fail", "why": "no tile found in the wrist view"}
    if reference is None:
        return {**live, "status": "skipped", "why": "no reference for this spot"}
    face_mm2 = config.TILE_FACE_M[0] * config.TILE_FACE_M[1] * 1e6
    scale = math.sqrt(face_mm2 / reference["area_px"])            # mm per wrist pixel at the tile's height
    du, dv = (live["centre_px"][0] - reference["centre_px"][0], live["centre_px"][1] - reference["centre_px"][1])
    shift_mm = math.hypot(du, dv) * scale
    size_pct = 100 * (math.sqrt(live["area_px"] / reference["area_px"]) - 1)
    angle = wrap_yaw(live["angle_deg"] - reference["angle_deg"])
    status = worst([grade(shift_mm, "wrist_mm"), grade(size_pct, "wrist_size_pct"), grade(angle, "wrist_deg")])
    return {**live, "du_px": du, "dv_px": dv, "shift_mm": shift_mm, "size_pct": size_pct, "angle_change_deg": angle,
            "status": status}


def wrist_text(result):
    if "shift_mm" not in result:
        return result["why"]
    return (f"tile {result['du_px']:+.0f}px, {result['dv_px']:+.0f}px from where it was (~{result['shift_mm']:.1f}mm), "
            f"size {result['size_pct']:+.0f}% (height), turned {result['angle_change_deg']:+.1f}°")


def side_by_side(reference_bgr, live_bgr, live_tile, reference_tile, label):
    """Reference and live wrist frames next to each other, each tile's centre marked."""
    panels = []
    for name, image, tile in (("reference", reference_bgr, reference_tile), ("now", live_bgr, live_tile)):
        image = image.copy() if image is not None else np.zeros_like(live_bgr)
        if tile is not None:
            cv2.drawMarker(image, tuple(int(v) for v in tile["centre_px"]), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(image, f"{label} {name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        panels.append(image)
    return np.hstack(panels)


# --- parked pose ---

def parked_result(live, reference):
    """Joint by joint, how far the unpowered arm settled from the reference's parked pose."""
    if live is None or reference is None:
        return {"status": "skipped", "why": "no parked pose " + ("measured" if live is None else "in the reference")}
    off = {j: live[j] - reference[j] for j in config.ARM_JOINTS}
    joint = max(off, key=lambda j: abs(off[j]))
    return {"off_deg": off, "worst_joint": joint, "status": grade(off[joint], "parked_deg")}


def parked_text(result):
    if "off_deg" not in result:
        return result["why"]
    return f"unpowered arm settled {result['off_deg'][result['worst_joint']]:+.1f}° off at {result['worst_joint']} (worst joint)"


# --- the reference ---

def load_reference(folder=None):
    folder = folder or REFERENCE_DIR
    path = folder / "reference.json"
    if not path.exists():
        return None
    reference = json.loads(path.read_text())
    reference["realsense_image"] = cv2.imread(str(folder / reference["realsense"]))
    for spot in reference.get("spots", []):
        spot["wrist_image"] = cv2.imread(str(folder / spot["wrist"])) if spot.get("wrist") else None
    return reference


def save_reference(measured, realsense_bgr, wrist_frames, folder=None):
    """Write the reference: measured (reference.json's fields), the RealSense frame, and a wrist frame per
    spot id ({id: image}). Replaces what's there."""
    folder = folder or REFERENCE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("wrist_*.jpg"):
        old.unlink()
    cv2.imwrite(str(folder / "realsense.png"), realsense_bgr)
    record = {**measured, "time": datetime.now().isoformat(timespec="seconds"), "realsense": "realsense.png",
              "calibration": setup_fingerprint()}     # the hand-eye transform and arm it belongs to
    for spot in record.get("spots", []):
        if spot["id"] in wrist_frames:
            spot["wrist"] = f"wrist_{spot['id']}.jpg"
            cv2.imwrite(str(folder / spot["wrist"]), wrist_frames[spot["id"]], [cv2.IMWRITE_JPEG_QUALITY, 95])
    (folder / "reference.json").write_text(json.dumps(record, indent=2, default=_jsonable) + "\n")
    return record


def _jsonable(value):
    return value.tolist() if hasattr(value, "tolist") else value.item() if hasattr(value, "item") else str(value)


# --- what to do about it ---

ADVICE = {
    "camera": "The RealSense moved relative to the robot. Nudge the tripod (or the LeKiwi) back until the "
              "alignment view turns green, rather than recalibrating: a new T_cam_to_base sets aside the hover "
              "and return corrections.",
    "table": "The mat's height or tilt changed (another table, something under a wheel or the tripod, a "
             "different mat). If the camera check passed, fix the table; if not, fix the camera first.",
    "tiles": "The hover lands the tile somewhere else in the wrist view. If camera and table pass, look at the "
             "wrist camera's mount, the arm's calibration (parked pose) and whether the hover correction changed.",
    "parked": "The unpowered arm settles differently: a servo recalibrated or replaced, a loose horn, or "
              "something touching the arm. Joint angles may no longer mean what they did in training.",
    "brightness": "The light changed: compare the setup answers with the training sessions "
                  "(session_report.py --all); a dim mat froze v2 once.",
}


def summary(results):
    """Lines of the final table, and the overall status."""
    lines = [f"  {'check':<12} {'status':<8} what"]
    for name, result in results.items():
        lines.append(f"  {name:<12} {result['status']:<8} {result.get('text', '')}")
    overall = worst(r["status"] for r in results.values())
    for name, result in results.items():
        if result["status"] in ("warn", "fail") and name in ADVICE:
            lines.append(f"\n{name}: {ADVICE[name]}")
    return lines, overall
