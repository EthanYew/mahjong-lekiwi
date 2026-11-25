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
- tile test: the same tile placed to green on a few spots, the arm hovers over it as the bench does (measured
  position + hover correction) and the wrist camera looks. The tile landing where it did in the reference
  wrist frame means camera, table, arm calibration, hover correction and wrist camera mount all still
  agree — and it's the very image the policy starts from. It's found by matching the reference's own picture of
  it, not by colour: on a pale mat the face has no edge a threshold can find (see MATCH_MIN_SCORE).
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
from .layout import FACE_MAX_S, table_to_pixel, wrap_yaw

REFERENCE_DIR = config.DATA / "setup_reference"

# The robot's box in the base frame (the LeKiwi base and the folded arm at rest): its projection is where
# the camera check looks for features.
ROBOT_BOX_X, ROBOT_BOX_Y, ROBOT_BOX_HEIGHT_M = (-0.12, 0.05), (-0.15, 0.15), 0.12
MIN_INLIERS = 12                    # matched features agreeing on one shift, for the camera check to count
WORKSPACE_CENTRE = (0.20, 0.0)      # where pixel shifts are turned into mm

# Table samples: a grid over the mat the policies work on.
TABLE_GRID_X, TABLE_GRID_Y, TABLE_GRID_STEP_M = (0.12, 0.28), (-0.20, 0.20), 0.02
TABLE_OUTLIER_MM = 8.0              # samples this far from the median are tiles or clutter, left out

# Wrist view: the tile's white face, as layout.find_tile finds it from the RealSense. Neighbouring tiles and
# glare are in view too (the 2026-10-28 reference took a glare patch at F3's image edge for the tile).
WRIST_AREA_PX = 15000               # a face at the 40mm hover: 14-18k px on the 2026-10-28/29 frames
WRIST_AREA_RATIO = (0.5, 2.0)       # of WRIST_AREA_PX, to count as a tile
WRIST_ASPECT = (1.0, 1.7)           # long / short side of its rotated box (a face is 35 x 26mm, seen in perspective)
WRIST_MIN_FILL = 0.7                # contour area / its rotated box: a tile, not glare
WRIST_BORDER_PX = 2                 # a blob this close to the image edge is cut off (or glare), not the tile
WRIST_SEARCH_PX = 120               # how far from where it's expected the tile may be and still count as found
WRIST_MIN_V = 110                   # the face's brightness threshold comes from the frame (Otsu), never below this
WRIST_OPEN_PX = (31, 31)            # bridges thinner than this between white blobs are cut (a face is ~120px wide)

# The tile test finds the tile by template matching: the reference's picture of it, slid over the frame within
# WRIST_SEARCH_PX. Face and white stripe overlap in brightness and colour on the standard mat, so there's nothing for a
# threshold to cut (20261103_120259_setup). A template that's mostly face: a face is ~140 x 120px at the hover.
WRIST_PATCH_HALF_PX = 60
# On the striped mat, 20261103_121112_bench (v6 and v7 hovering at the same spot minutes apart): same tile 0.82-0.96 on 14
# pairs, no tile at that spot 0.68 (one frame). So this asks "is a tile there", and leaves room for a change of light. It
# can't tell which tile: another one reached 0.84 (median 0.58, 182 pairs), which is why the prompt names the tile.
MATCH_MIN_SCORE = 0.78
# Between this and MATCH_MIN_SCORE the tile test shows the match and asks a person rather than call it "no tile". Every
# "no tile" on the standard mat from 2026-11-11 to 11-23 (12 frames, the tile plainly in view) matched 0.67-0.77 within
# 15px of the reference's place: a 3-week-old reference in other light, not a moved tile. Below it: 0.47-0.56, or a
# match at the wrong place. A person's yes is graded like a found tile, and logged as theirs (`confirmed_by`).
MATCH_ASK_SCORE = 0.6
MATCH_ANGLES_DEG = tuple(range(-12, 13, 3))                 # green lets a tile sit ±6° differently; by eye, a little more
MATCH_SCALES = tuple(0.75 + 0.025 * k for k in range(21))   # ±25%: past wrist_size_pct's fail line, so that's graded
ANGLE_UNKNOWN_DEG = 0.0

# (warn, fail) thresholds. Jaws tolerate about ±6mm, so 5-6mm is where a pickup starts to miss.
LIMITS = {
    "camera_mm": (2.0, 5.0),        # robot's image shift turned into mm on the mat
    "camera_deg": (0.5, 1.5),       # image rotation
    "table_mm": (3.0, 6.0),         # mat height change at the workspace centre
    "table_tilt_mm": (2.0, 4.0),    # change in height difference per 10cm
    # tile's position in the wrist view at the hover. The chain's own noise (RealSense finding ~1.5mm, hover
    # repeatability, the face's outline) is ~3mm, and 4-5.5mm showed with nothing moved (2026-10-29), so the
    # tile test grades the median over the spots: a moved camera or arm shifts them all, noise doesn't
    "wrist_mm": (4.0, 7.0),
    "wrist_size_pct": (10.0, 20.0), # its size there: the hover's height (closer = bigger)
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


def matched_points(reference_bgr, live_bgr, region):
    """Features matched between the reference and the live frame inside `region`: (reference px, live px) as N x 2
    arrays, or the number matched if under MIN_INLIERS."""
    u0, v0, u1, v1 = region
    mask = np.zeros(reference_bgr.shape[:2], np.uint8)
    mask[v0:v1, u0:u1] = 255
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
    kp_ref, des_ref = orb.detectAndCompute(_grey(reference_bgr), mask)
    kp_live, des_live = orb.detectAndCompute(_grey(live_bgr), mask)
    if des_ref is None or des_live is None or len(kp_ref) < MIN_INLIERS or len(kp_live) < MIN_INLIERS:
        return 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_ref, des_live)
    if len(matches) < MIN_INLIERS:
        return len(matches)
    return (np.float32([kp_ref[m.queryIdx].pt for m in matches]), np.float32([kp_live[m.trainIdx].pt for m in matches]))


def image_shift(reference_bgr, live_bgr, region):
    """How the robot's region moved from the reference to the live frame: {"dx_px", "dy_px", "shift_px",
    "rotation_deg", "inliers"}, or {"inliers": n} if too few features agree (MIN_INLIERS)."""
    u0, v0, u1, v1 = region
    points = matched_points(reference_bgr, live_bgr, region)
    if isinstance(points, int):
        return {"inliers": points}
    src, dst = points
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


# --- the alignment guide: what to do to the robot, in its own terms ---
# The old guide drew the image shift x8 as one arrow, frame by frame. At 1-2px of real shift that direction is mostly
# feature noise, so it spun (2026-11-25), and an image shift mixes a turn with a slide. So the guide fits the robot's turn
# and slide on the table, steadies them over the last GUIDE_FRAMES frames, says nothing about a direction inside the
# pass line, and asks for one thing at a time: the turn first, since turning about the middle moves the slide reading.

ROBOT_FEATURE_HEIGHT_M = 0.06       # the features on the base and folded arm sit 0-12cm up (ROBOT_BOX_HEIGHT_M): halfway
GUIDE_FRAMES = 7                    # frames the guide's numbers are the median of
COLOURS = {"pass": (0, 200, 0), "warn": (0, 190, 255), "fail": (0, 0, 255), "skipped": (200, 200, 200)}
OUTLINE_COLOUR = (255, 0, 255)      # the reference robot's outline: magenta shows on the teal arm, the striped mat and grey


def robot_height(x, y):
    return config.table_z(x, y) + ROBOT_FEATURE_HEIGHT_M


def robot_motion(reference_bgr, live_bgr, region, intrinsics, T_cam_to_base):
    """How the robot sits now against the reference, on the table: {"turn_deg" (+ = turned anticlockwise seen from
    above), "dx_mm" (+ = its base origin is further forward), "dy_mm" (+ = further to its left), "inliers"}, or
    {"inliers": n} if too few features agree. Moving the camera instead reads as the robot moving the other way."""
    points = matched_points(reference_bgr, live_bgr, region)
    if isinstance(points, int):
        return {"inliers": points}
    was, now = (table_to_table_xy(p, intrinsics, T_cam_to_base) for p in points)
    fit, inliers = cv2.estimateAffinePartial2D(was, now, method=cv2.RANSAC, ransacReprojThreshold=0.002)
    n = int(inliers.sum()) if inliers is not None else 0
    if fit is None or n < MIN_INLIERS:
        return {"inliers": n}
    return {"turn_deg": float(math.degrees(math.atan2(fit[1, 0], fit[0, 0]))), "dx_mm": float(1000 * fit[0, 2]),
            "dy_mm": float(1000 * fit[1, 2]), "inliers": n}


def table_to_table_xy(pixels, intrinsics, T_cam_to_base):
    """Pixels on the robot -> base-frame (x, y) at the robot's feature height, float32 N x 2."""
    from .layout import pixel_to_table
    return np.float32(pixel_to_table(pixels, intrinsics, T_cam_to_base, robot_height)[:, :2])


def steady(readings, keys):
    """The median of each key over the readings that have them all, or None if under half of them do (the view is
    too unsettled to say)."""
    have = [r for r in readings if all(k in r for k in keys)]
    if not have or len(have) * 2 < len(readings):
        return None
    return {k: float(np.median([r[k] for r in have])) for k in keys}


def guide_steps(motion):
    """What to do about robot_motion (steadied): [{"kind": "turn" / "move", "status", "text"}], turn first. A step
    inside its pass line says so rather than give a direction (that direction would be noise)."""
    if motion is None or "turn_deg" not in motion:
        return [{"kind": "turn", "status": "warn", "text": "can't see the robot well enough: is the arm at rest, in view?"}]
    turn = motion["turn_deg"]
    turn_status = grade(turn, "camera_deg")
    turn_text = ("square: leave the turn" if turn_status == "pass" else
                 f"turn it {abs(turn):.1f} deg {'clockwise' if turn > 0 else 'anticlockwise'} (seen from above)")
    forward, left = -motion["dx_mm"], -motion["dy_mm"]              # the way back
    move_status = grade(math.hypot(forward, left), "camera_mm")
    parts = [f"{abs(forward):.0f}mm {'forward' if forward > 0 else 'back'}" if abs(forward) >= 1 else "",
             f"{abs(left):.0f}mm to its {'left' if left > 0 else 'right'}" if abs(left) >= 1 else ""]
    move_text = ("in place: leave it" if move_status == "pass" else
                 "slide it " + " and ".join(p for p in parts if p))
    return [{"kind": "turn", "status": turn_status, "text": turn_text, "deg": -turn},
            {"kind": "move", "status": move_status, "text": move_text, "forward_mm": forward, "left_mm": left}]


def step_now(steps):
    """The step to do now: the first not passing, or None when all pass."""
    return next((s for s in steps if s["status"] != "pass"), None)


OUTLINE_MIN_PX = 80                 # edges shorter than this are stripe and texture, not the robot's shape
ZOOM_MARGIN_PX = 24


def zoom_box(region, shape, margin=ZOOM_MARGIN_PX):
    """(u0, v0, u1, v1): the robot's region with a margin, clipped to the image: what the guide zooms in on."""
    u0, v0, u1, v1 = region
    h, w = shape[:2]
    return max(u0 - margin, 0), max(v0 - margin, 0), min(u1 + margin, w), min(v1 + margin, h)


def reference_outline(reference_bgr, region):
    """The reference robot's shape around `region`: its long, strong edges as contours in image pixels."""
    u0, v0, u1, v1 = zoom_box(region, reference_bgr.shape)
    grey = cv2.GaussianBlur(cv2.cvtColor(reference_bgr[v0:v1, u0:u1], cv2.COLOR_BGR2GRAY), (7, 7), 0)
    contours, _ = cv2.findContours(cv2.Canny(grey, 80, 200), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [c + np.int32([u0, v0]) for c in contours if cv2.arcLength(c, False) > OUTLINE_MIN_PX]


def _project(points_xy, intrinsics, T_cam_to_base):
    return table_to_pixel([(x, y, robot_height(x, y)) for x, y in points_xy], intrinsics, T_cam_to_base)


def screen_clockwise(anchor_xy, deg, intrinsics, T_cam_to_base):
    """Whether turning the robot by `deg` (+ = anticlockwise seen from above) looks clockwise on screen."""
    radius = 0.07
    a, b = (np.array(p) for p in _project([(anchor_xy[0] + radius, anchor_xy[1]),
                                           (anchor_xy[0] + radius * math.cos(math.radians(math.copysign(10, deg))),
                                            anchor_xy[1] + radius * math.sin(math.radians(math.copysign(10, deg))))],
                                          intrinsics, T_cam_to_base))
    c = _project([anchor_xy], intrinsics, T_cam_to_base)[0]
    cross = (a - c)[0] * (b - c)[1] - (a - c)[1] * (b - c)[0]
    return cross > 0                    # image y points down, so a positive cross product is clockwise on screen


def draw_turn(image, centre, clockwise, colour):
    """A large curved arrow around `centre` (view pixels): the way to turn."""
    radius = int(0.32 * min(image.shape[:2]))
    start, stop = (200, 340) if clockwise else (340, 200)
    angles = np.linspace(math.radians(start), math.radians(stop), 24)
    px = np.int32([(centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a)) for a in angles])
    cv2.polylines(image, [px], False, (0, 0, 0), 11, cv2.LINE_AA)
    cv2.polylines(image, [px], False, colour, 6, cv2.LINE_AA)
    cv2.arrowedLine(image, tuple(int(v) for v in px[-4]), tuple(int(v) for v in px[-1]), colour, 6, cv2.LINE_AA,
                    tipLength=1.5)


def draw_move(image, centre, direction, colour, length_px=110):
    """A large straight arrow from `centre` (view pixels) along `direction`: the way to slide."""
    tip = np.array(centre) + length_px * np.array(direction)
    for thickness, c in ((12, (0, 0, 0)), (6, colour)):
        cv2.arrowedLine(image, tuple(int(v) for v in centre), tuple(int(v) for v in tip), c, thickness, cv2.LINE_AA,
                        tipLength=0.3)


def screen_direction(anchor_xy, forward_mm, left_mm, intrinsics, T_cam_to_base):
    """The unit vector on screen of sliding the robot forward_mm / left_mm from anchor_xy."""
    norm = math.hypot(forward_mm, left_mm) or 1.0
    tip = (anchor_xy[0] + 0.03 * forward_mm / norm, anchor_xy[1] + 0.03 * left_mm / norm)
    a, b = (np.array(p) for p in _project([anchor_xy, tip], intrinsics, T_cam_to_base))
    return (b - a) / (np.linalg.norm(b - a) or 1.0)


def guide_view(reference_bgr, live_bgr, region, result, steps, intrinsics, T_cam_to_base, width=900):
    """The alignment window: the robot zoomed in, live, with the reference robot's outline in magenta to line it up
    with and a large arrow for the one step to do now (curved: turn; straight: slide); under it the two steps in
    words, each in its status colour; under that the whole view, small, for context."""
    h, w = live_bgr.shape[:2]
    z0, y0, z1, y1 = zoom_box(region, live_bgr.shape)
    scale = width / max(z1 - z0, 1)
    zoom = cv2.resize(live_bgr[y0:y1, z0:z1], (width, int((y1 - y0) * scale)), interpolation=cv2.INTER_LINEAR)
    outline = reference_outline(reference_bgr, region)
    cv2.polylines(zoom, [np.int32((c - [z0, y0]) * scale) for c in outline], False, OUTLINE_COLOUR, 3, cv2.LINE_AA)
    now = step_now(steps)
    u0, v0, u1, v1 = region
    anchor = table_to_table_xy([((u0 + u1) / 2, (v0 + v1) / 2)], intrinsics, T_cam_to_base)[0]
    middle = (zoom.shape[1] // 2, zoom.shape[0] // 2)
    if now is not None and now["kind"] == "turn" and "deg" in now:
        draw_turn(zoom, middle, screen_clockwise(anchor, now["deg"], intrinsics, T_cam_to_base), COLOURS[now["status"]])
    elif now is not None and now["kind"] == "move":
        draw_move(zoom, middle, screen_direction(anchor, now["forward_mm"], now["left_mm"], intrinsics, T_cam_to_base),
                  COLOURS[now["status"]])
    cv2.rectangle(zoom, (0, 0), (zoom.shape[1] - 1, zoom.shape[0] - 1), COLOURS[result["status"]], 4)

    lined_up = now is None and result["status"] == "pass"
    panel = np.full((150, width, 3), 30, np.uint8)
    head = "Lined up: press Enter" if lined_up else "Line the robot up with the magenta outline, one step at a time:"
    cv2.putText(panel, head, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS["pass"] if lined_up else (255, 255, 255),
                2, cv2.LINE_AA)
    for i, step in enumerate(steps):
        label = {"turn": "1 TURN ", "move": "2 SLIDE"}[step["kind"]]
        cv2.putText(panel, f"{'>' if step is now else ' '} {label}  {step['text']}", (12, 28 + 34 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75 if step is now else 0.6, COLOURS[step["status"]],
                    2 if step is now else 1, cv2.LINE_AA)
    cv2.putText(panel, "forward = toward the tiles.   Enter: done   q: leave it as it is", (12, 28 + 34 * 3 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    context = live_bgr.copy()
    cv2.polylines(context, outline, False, OUTLINE_COLOUR, 1, cv2.LINE_AA)
    cv2.rectangle(context, (z0, y0), (z1 - 1, y1 - 1), COLOURS[result["status"]], 2)
    small = cv2.resize(context, (width // 2, int(h * (width // 2) / w)))
    strip = np.full((small.shape[0], width, 3), 30, np.uint8)
    strip[:, width // 4: width // 4 + small.shape[1]] = small
    return np.vstack([zoom, panel, strip])


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

def wrist_tile(image_bgr, near_px=None):
    """The tile's face in a wrist frame: {"centre_px", "area_px", "angle_deg", "size_px"} for the white,
    tile-shaped, tile-sized blob nearest near_px (where it's expected: the reference's tile) and within
    WRIST_SEARCH_PX of it, or without near_px (saving a reference, confirmed on screen) the one nearest the image
    centre; None if there's none. Blobs cut by the image edge don't count."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pale = hsv[:, :, 1] <= FACE_MAX_S
    if not pale.any():
        return None
    otsu, _ = cv2.threshold(hsv[:, :, 2][pale].reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (pale & (hsv[:, :, 2] >= max(otsu, WRIST_MIN_V))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # the printed symbols
    # cut thin bridges: the pale jaw tip touching the face joined it to the tiles below (2026-10-28 F3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, WRIST_OPEN_PX))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_bgr.shape[:2]
    target = near_px if near_px is not None else (w / 2, h / 2)
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if not WRIST_AREA_RATIO[0] <= area / WRIST_AREA_PX <= WRIST_AREA_RATIO[1]:
            continue
        x, y, bw_, bh_ = cv2.boundingRect(contour)
        if x <= WRIST_BORDER_PX or y <= WRIST_BORDER_PX or x + bw_ >= w - WRIST_BORDER_PX or y + bh_ >= h - WRIST_BORDER_PX:
            continue
        (cu, cv_), (bw, bh), angle = cv2.minAreaRect(contour)
        long_side, short_side = max(bw, bh), max(min(bw, bh), 1e-6)
        if area < WRIST_MIN_FILL * bw * bh or not WRIST_ASPECT[0] <= long_side / short_side <= WRIST_ASPECT[1]:
            continue
        distance = math.hypot(cu - target[0], cv_ - target[1])
        if (near_px is not None and distance > WRIST_SEARCH_PX) or (best is not None and distance >= best[0]):
            continue
        if bw < bh:                                    # angle of the long side
            angle += 90.0
        best = (distance, {"centre_px": [float(cu), float(cv_)], "area_px": float(area),
                           "angle_deg": wrap_yaw(float(angle)), "size_px": [float(long_side), float(short_side)]})
    return best[1] if best else None


# --- is the wrist camera on the arm? ---
# The wrist and base cameras are the same model with the same ID and serial: only the USB port tells them apart, so swapped
# cables make /dev/wristcam the base camera with nothing failing. On 20261125_201151_setup the tile test hovered over F2 five
# times and "found no tile" in what was the base camera's view of the room. A camera on the arm sees a new picture when the
# arm moves: real wrist frames at different spots were 0.00-0.43 alike (view_similarity), the base camera's frames across
# those five hovers 0.94-1.00 (a person moving in view included).
CAMERA_STILL_SIMILARITY = 0.75
FLAT_GREY_STD = 3.0                 # a frame this featureless (a dark or blank view) can't say whether it moved


def view_similarity(a_bgr, b_bgr):
    """How alike two frames' overall pictures are (-1..1: the correlation of their 32 x 24 greys), or None if either is
    featureless."""
    greys = [cv2.cvtColor(cv2.resize(f, (32, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(float)
             for f in (a_bgr, b_bgr)]
    if any(g.std() < FLAT_GREY_STD for g in greys):
        return None
    a, b = ((g - g.mean()) / g.std() for g in greys)
    return float((a * b).mean())


def camera_stayed(rest_bgr, hover_bgr):
    """Whether the wrist camera saw the same picture at rest and at the hover: it isn't on the arm."""
    similarity = view_similarity(rest_bgr, hover_bgr)
    return similarity is not None and similarity >= CAMERA_STILL_SIMILARITY


def patch_centre(centre, shape):
    """`centre` moved just enough that the template around it fits inside an image of `shape`."""
    h, w = shape[:2]
    return [float(np.clip(centre[0], WRIST_PATCH_HALF_PX, w - WRIST_PATCH_HALF_PX)),
            float(np.clip(centre[1], WRIST_PATCH_HALF_PX, h - WRIST_PATCH_HALF_PX))]


def wrist_template(image_bgr, centre):
    """The tile's picture for match_tile: a grey square around `centre` (which fits, see patch_centre)."""
    u, v = int(round(centre[0])), int(round(centre[1]))
    half = WRIST_PATCH_HALF_PX
    return cv2.cvtColor(image_bgr[v - half:v + half, u - half:u + half], cv2.COLOR_BGR2GRAY)


def marked_tile(centre, found, shape):
    """The reference's reading for a tile a person marked at `centre`: wrist_tile's find for size and angle if it agrees
    with the mark, else a face of the usual size (its colour can't be measured on a pale mat)."""
    centre = patch_centre(centre, shape)
    if found is not None and math.dist(found["centre_px"], centre) <= WRIST_PATCH_HALF_PX / 2:
        return {**found, "centre_px": centre}
    long_side = math.sqrt(WRIST_AREA_PX * config.TILE_FACE_M[0] / config.TILE_FACE_M[1])
    return {"centre_px": centre, "area_px": float(WRIST_AREA_PX), "angle_deg": ANGLE_UNKNOWN_DEG,
            "size_px": [long_side, WRIST_AREA_PX / long_side]}


def match_tile(image_bgr, template, reference, min_score=None):
    """The reference's tile in a wrist frame, by its picture: {"centre_px", "area_px", "angle_deg", "size_px", "score"}
    as wrist_tile reads them (size and angle relative to the reference's), or None if nothing within WRIST_SEARCH_PX of
    where it was matches min_score (default MATCH_MIN_SCORE)."""
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    best = None
    for scale in MATCH_SCALES:
        scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        th, tw = scaled.shape
        u, v = reference["centre_px"]
        x0, y0 = max(int(u - WRIST_SEARCH_PX - tw / 2), 0), max(int(v - WRIST_SEARCH_PX - th / 2), 0)
        x1, y1 = min(int(u + WRIST_SEARCH_PX + tw / 2) + 1, w), min(int(v + WRIST_SEARCH_PX + th / 2) + 1, h)
        if x1 - x0 < tw or y1 - y0 < th:
            continue
        region = grey[y0:y1, x0:x1]
        for angle in MATCH_ANGLES_DEG:
            turned = cv2.warpAffine(scaled, cv2.getRotationMatrix2D((tw / 2, th / 2), angle, 1.0), (tw, th),
                                    borderMode=cv2.BORDER_REPLICATE)
            scores = np.nan_to_num(cv2.matchTemplate(region, turned, cv2.TM_CCOEFF_NORMED), nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, score, _, (lx, ly) = cv2.minMaxLoc(scores)
            if best is None or score > best[0]:
                best = (score, x0 + lx + tw / 2, y0 + ly + th / 2, scale, angle)
    if best is None or best[0] < (MATCH_MIN_SCORE if min_score is None else min_score):
        return None
    score, cu, cv_, scale, angle = best
    # getRotationMatrix2D turns anticlockwise in the image, minAreaRect's angle clockwise
    return {"centre_px": [float(cu), float(cv_)], "area_px": float(reference["area_px"] * scale ** 2),
            "angle_deg": wrap_yaw(reference["angle_deg"] - angle),
            "size_px": [float(v * scale) for v in reference["size_px"]], "score": float(score)}


def median_tile(finds):
    """One reading from several frames' wrist_tile() finds (None where a frame found nothing): the median of
    each, or None if fewer than half the frames found the tile."""
    seen = [f for f in finds if f is not None]
    if not seen or len(seen) * 2 < len(finds):
        return None
    first = seen[0]["angle_deg"]
    return {"centre_px": [float(v) for v in np.median([f["centre_px"] for f in seen], axis=0)],
            "area_px": float(np.median([f["area_px"] for f in seen])),
            "angle_deg": wrap_yaw(first + float(np.median([wrap_yaw(f["angle_deg"] - first) for f in seen]))),
            "size_px": [float(v) for v in np.median([f["size_px"] for f in seen], axis=0)], "frames": len(seen),
            **({"score": float(np.median([f["score"] for f in seen]))} if "score" in seen[0] else {})}


def wrist_result(live, reference, placed_turn_deg=None):
    """Compare the tile in the wrist view with the reference's, at the same spot. Position and size are graded;
    the angle isn't: green lets a tile sit ±6° differently, and placed_turn_deg (how differently the RealSense
    saw it placed) is shown next to it."""
    if live is None:
        return {"status": "fail", "why": "no tile found in the wrist view"}
    if reference is None:
        return {**live, "status": "skipped", "why": "no reference for this spot"}
    face_mm2 = config.TILE_FACE_M[0] * config.TILE_FACE_M[1] * 1e6
    scale = math.sqrt(face_mm2 / reference["area_px"])            # mm per wrist pixel at the tile's height
    du, dv = (live["centre_px"][0] - reference["centre_px"][0], live["centre_px"][1] - reference["centre_px"][1])
    shift_mm = math.hypot(du, dv) * scale
    size_pct = 100 * (math.sqrt(live["area_px"] / reference["area_px"]) - 1)
    status = worst([grade(shift_mm, "wrist_mm"), grade(size_pct, "wrist_size_pct")])
    return {**live, "du_px": du, "dv_px": dv, "shift_mm": shift_mm, "size_pct": size_pct,
            "angle_change_deg": wrap_yaw(live["angle_deg"] - reference["angle_deg"]),
            "placed_turn_deg": placed_turn_deg, "status": status}


def wrist_text(result):
    if "shift_mm" not in result:
        return result["why"]
    placed = (f"; placed {result['placed_turn_deg']:+.1f}° differently" if result.get("placed_turn_deg") is not None
              else "")
    match = f"; match {result['score']:.2f}" if "score" in result else ""
    match += " (a person said it's the tile)" if result.get("confirmed_by") == "person" else ""
    return (f"tile {result['du_px']:+.0f}px, {result['dv_px']:+.0f}px from where it was (~{result['shift_mm']:.1f}mm), "
            f"size {result['size_pct']:+.0f}% (height); turned {result['angle_change_deg']:+.1f}° in the wrist view"
            f"{placed} (not graded){match}")


def tiles_result(spots):
    """The tile test over its spots ({id: wrist_result}): graded on the median shift and size, since a moved
    camera or arm moves every spot while noise moves one. A single spot past the fail line makes it a warning
    at least. (The wrist image turns with the wrist roll, which differs per spot, so the shifts' directions
    don't add up across spots; their sizes do.)"""
    measured = [r for r in spots.values() if "shift_mm" in r]
    failed = [i for i, r in spots.items() if r["status"] == "fail"]
    shown = ", ".join(f"{i} {r['shift_mm']:.1f}mm" if "shift_mm" in r else f"{i} {r['status']}" for i, r in spots.items())
    if not measured:
        return {"status": "fail" if failed else "skipped", "spots": spots, "text": shown or "no spots done"}
    shift = float(np.median([r["shift_mm"] for r in measured]))
    size = float(np.median([r["size_pct"] for r in measured]))
    status = worst([grade(shift, "wrist_mm"), grade(size, "wrist_size_pct")] + (["warn"] if failed else []))
    return {"status": status, "median_shift_mm": shift, "median_size_pct": size, "spots": spots,
            "text": f"median {shift:.1f}mm, size {size:+.0f}% ({shown})"}


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
        spot["template_image"] = (cv2.imread(str(folder / spot["template"]), cv2.IMREAD_GRAYSCALE)
                                  if spot.get("template") else None)
    return reference


def save_reference(measured, realsense_bgr, wrist_frames, folder=None):
    """Write the reference: measured (reference.json's fields), the RealSense frame, and a wrist frame per
    spot id ({id: image}). Replaces what's there."""
    folder = folder or REFERENCE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for old in [*folder.glob("wrist_*.jpg"), *folder.glob("template_*.png")]:
        old.unlink()
    cv2.imwrite(str(folder / "realsense.png"), realsense_bgr)
    record = {**measured, "time": datetime.now().isoformat(timespec="seconds"), "realsense": "realsense.png",
              "calibration": setup_fingerprint()}     # the hand-eye transform and arm it belongs to
    for spot in record.get("spots", []):
        if spot["id"] in wrist_frames:
            frame = wrist_frames[spot["id"]]
            spot["wrist"] = f"wrist_{spot['id']}.jpg"
            cv2.imwrite(str(folder / spot["wrist"]), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            spot["tile"] = {**spot["tile"], "centre_px": patch_centre(spot["tile"]["centre_px"], frame.shape)}
            spot["template"] = f"template_{spot['id']}.png"
            cv2.imwrite(str(folder / spot["template"]), wrist_template(frame, spot["tile"]["centre_px"]))
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
             "wrist camera's mount, the arm's calibration (parked pose) and whether the hover correction changed. "
             "\"No tile\" with the tile in view: is it the same tile as the reference's, and is the light the same? "
             "A \"wrist camera\" that saw the room, or didn't change when the arm moved, is the base camera: swap the two "
             "camera cables between their USB ports.",
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
