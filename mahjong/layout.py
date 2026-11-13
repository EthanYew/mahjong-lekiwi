"""Tile layouts: target spots on the mat, drawn on the RealSense view so tiles can be placed the same way
every recording.

A layout is a named list of spots in the arm's base frame (x, y in metres, yaw in degrees; 0 = square
to the mat, long side along x, as in every dataset so far), stored in data/layouts.json. Each spot is
drawn as the outline of a tile's top face, projected through the hand-eye transform — the reverse of
detection. The tile actually there is found by the tile detector's box, and where its white face stands out from
the mat (brightness, low saturation) that refines the centre and gives the yaw, measured on the table plane, so each
outline turns amber/green as the tile gets close (smoothed over a few frames). On a pale or patterned mat the face
doesn't stand out, so there's no yaw reading: green is position only, and the outline is lined up by eye. What matters
most is repeatability: as long as the RealSense doesn't move, a spot lands on the same tiles-worth of mat every session.

Positions from pixels use the table plane at the tile's top face, not depth, so the finder needs only a
colour frame (and works on a synthetic one in tests and --sim, which have no detector).
"""
import json
import math
from datetime import datetime

import cv2
import numpy as np

from . import config

POSITION_TOLERANCE_MM = 5.0        # green within 5mm / 6° (4mm / 5° felt tight on hardware: the yaw reading
YAW_TOLERANCE_DEG = 6.0            # alone can be ~2.5° off, see find_tile)
HYSTERESIS = 1.5                   # once green, stays green up to 1.5x the tolerance
SMOOTH_FRAMES = 10                 # live view: median over the last 10 frames (~1/3s)
CAPTURE_FRAMES = 30                # capture: readings averaged over 30 frames
TREND_MIN_SPAN_M = 0.05            # fit a yaw drift along an axis only if the spots span this much
SEARCH_RADIUS_M = 0.035            # around a spot's centre, where the finder looks for its tile
FACE_MIN_V, FACE_MAX_S = 150, 80   # HSV: a tile's white face; the mat and the tiles' green backs fail one
FACE_AREA_RANGE = (0.4, 2.0)       # of the expected face area, for a blob to count as a tile
# A face is 35 x 26mm (1.35; median read 1.31). Area alone let slices of white stripe through (70 x 10-15mm, cut off by
# the search window; 231 of 950 reads at real tiles on the striped mat, 20261103_121112_bench).
FACE_MAX_ASPECT = 1.9
SEARCH_MARGIN_M = 0.025            # the window reaches this past SEARCH_RADIUS_M, so a tile at the radius isn't cut off
# The detector's box includes the tile's near side: its centre sits 2.5mm short of the face's in x (scatter under 1mm,
# ~12k black-top boxes, 2026-10-28 to 30). Added to a box's centre.
BOX_TO_FACE_M = (0.0025, 0.0)
COLOUR_AGREES_M = 0.006            # a colour read this close to the box is the same tile, and refines it
GHOST_ALPHA = 0.5
COLOURS = {"ok": (0, 200, 0), "close": (0, 190, 255), "off": (0, 0, 255), "missing": (0, 0, 255),
           "dim": (150, 150, 150)}


# --- camera geometry ---

def face_z(x, y):
    """Height of a tile's top face in the base frame."""
    return config.table_z(x, y) + config.TILE_THICKNESS_M


def table_to_pixel(points, intrinsics, T_cam_to_base):
    """Base-frame points (N x 3) -> pixel coordinates (N x 2). intrinsics: {fx, fy, ppx, ppy}."""
    points = np.atleast_2d(np.asarray(points, float))
    homogeneous = np.c_[points, np.ones(len(points))]
    cam = (np.linalg.inv(T_cam_to_base) @ homogeneous.T).T[:, :3]
    return np.c_[intrinsics["fx"] * cam[:, 0] / cam[:, 2] + intrinsics["ppx"],
                 intrinsics["fy"] * cam[:, 1] / cam[:, 2] + intrinsics["ppy"]]


def pixel_to_table(pixels, intrinsics, T_cam_to_base, z_of_xy=face_z):
    """Pixels (N x 2) -> base-frame points (N x 3) where each pixel's ray meets the surface z_of_xy (the
    tile top face by default; the plane is near-flat, so two fixed-point passes are exact to <0.1mm)."""
    pixels = np.atleast_2d(np.asarray(pixels, float))
    rays_cam = np.c_[(pixels[:, 0] - intrinsics["ppx"]) / intrinsics["fx"],
                     (pixels[:, 1] - intrinsics["ppy"]) / intrinsics["fy"], np.ones(len(pixels))]
    R, origin = T_cam_to_base[:3, :3], T_cam_to_base[:3, 3]
    rays = rays_cam @ R.T
    z = np.full(len(pixels), face_z(0.2, 0.0))
    for _ in range(3):
        t = (z - origin[2]) / rays[:, 2]
        points = origin + rays * t[:, None]
        z = np.array([z_of_xy(px, py) for px, py, _ in points])
    return points


def tile_corners(x, y, yaw_deg, size=None):
    """The 4 corners (4 x 3, base frame) of a tile's top face centred at (x, y)."""
    length, width = size or config.TILE_FACE_M
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    corners = []
    for dx, dy in ((length / 2, width / 2), (-length / 2, width / 2), (-length / 2, -width / 2), (length / 2, -width / 2)):
        cx, cy = x + c * dx - s * dy, y + s * dx + c * dy
        corners.append((cx, cy, face_z(cx, cy)))
    return np.array(corners)


def wrap_yaw(deg):
    """A rectangle looks the same turned 180°: yaw in (-90, 90]."""
    deg = (deg + 90.0) % 180.0 - 90.0
    return 90.0 if deg == -90.0 else deg


# --- finding the tile at a spot ---

def find_tile(image_bgr, spot, intrinsics, T_cam_to_base, size=None, boxes=None):
    """The tile near `spot` ({"xy", "yaw_deg"}), measured on the table plane: {"xy", "yaw_deg", "size_mm"}, or None.

    boxes(image) -> [(x1, y1, x2, y2)]: the tile detector's boxes (perception.RealSense.tile_boxes). With it, the
    box says whether a tile is there, because a pale mat makes tile-shaped blobs of its own (5 of 950 reads on the
    striped mat, ~9mm off); a colour read that agrees with the box refines it and gives the yaw, otherwise yaw_deg is
    None. Without it (the sim's rendered frames), the colour read alone."""
    colour = find_face(image_bgr, spot, intrinsics, T_cam_to_base, size)
    if boxes is None:
        return colour
    box = find_box(boxes(image_bgr), spot, intrinsics, T_cam_to_base)
    if box is None:
        return None
    return colour if colour is not None and math.dist(colour["xy"], box["xy"]) <= COLOUR_AGREES_M else box


def find_box(boxes, spot, intrinsics, T_cam_to_base):
    """The detector's box nearest `spot` and within SEARCH_RADIUS_M, as a find with no yaw (a box has no angle, and at
    ~24px colour inside it doesn't recover one on a pale mat); None if there's none."""
    best = None
    for x1, y1, x2, y2 in boxes:
        corners = pixel_to_table([(x1, y1), (x2, y1), (x2, y2)], intrinsics, T_cam_to_base)[:, :2]
        centre = (corners[0] + corners[2]) / 2 + BOX_TO_FACE_M
        distance = math.dist(centre, spot["xy"])
        if distance <= SEARCH_RADIUS_M and (best is None or distance < best[0]):
            sides = (round(float(np.linalg.norm(corners[1] - corners[0])) * 1000, 1),
                     round(float(np.linalg.norm(corners[2] - corners[1])) * 1000, 1))
            best = (distance, {"xy": (float(centre[0]), float(centre[1])), "yaw_deg": None,
                               "size_mm": tuple(sorted(sides, reverse=True))})
    return best[1] if best else None


def find_face(image_bgr, spot, intrinsics, T_cam_to_base, size=None):
    """The tile's white face near `spot`: {"xy", "yaw_deg", "size_mm"}, or None if no tile-shaped, tile-sized white
    blob is within SEARCH_RADIUS_M. Blobs the search window cuts off are longer than it can see: a stripe, not a tile."""
    length, width = size or config.TILE_FACE_M
    x, y = spot["xy"]
    reach = SEARCH_RADIUS_M + SEARCH_MARGIN_M
    ring = [(x + reach * math.cos(a), y + reach * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16)]
    ring_px = table_to_pixel([(px, py, face_z(px, py)) for px, py in ring], intrinsics, T_cam_to_base)
    h, w = image_bgr.shape[:2]
    u0, v0 = np.clip(np.floor(ring_px.min(axis=0)).astype(int), 0, [w, h])
    u1, v1 = np.clip(np.ceil(ring_px.max(axis=0)).astype(int), 0, [w, h])
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None
    hsv = cv2.cvtColor(image_bgr[v0:v1, u0:u1], cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= FACE_MIN_V) & (hsv[:, :, 1] <= FACE_MAX_S)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))      # the face's printed symbols
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    crop_h, crop_w = mask.shape
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if ((bx == 0 and u0 > 0) or (by == 0 and v0 > 0) or (bx + bw == crop_w and u1 < w)
                or (by + bh == crop_h and v1 < h)):
            continue
        box_px = cv2.boxPoints(cv2.minAreaRect(contour)) + [u0, v0]
        corners = pixel_to_table(box_px, intrinsics, T_cam_to_base)[:, :2]
        centre = corners.mean(axis=0)
        e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
        area = np.linalg.norm(e1) * np.linalg.norm(e2)
        if not FACE_AREA_RANGE[0] <= area / (length * width) <= FACE_AREA_RANGE[1]:
            continue
        sides = sorted((np.linalg.norm(e1), np.linalg.norm(e2)))
        if sides[1] > FACE_MAX_ASPECT * max(sides[0], 1e-6):
            continue
        distance = math.hypot(centre[0] - x, centre[1] - y)
        if distance > SEARCH_RADIUS_M or (best is not None and distance >= best[0]):
            continue
        # A tile is only ~24x18px, so this reads a perfectly square tile up to ~2.5° off (median ~1°;
        # fitting the face's principal axis or its edges did no better on rendered tiles). The error is
        # fixed per spot, so placements still repeat; it's why the yaw tolerance isn't tighter.
        long_edge = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        yaw = wrap_yaw(math.degrees(math.atan2(long_edge[1], long_edge[0])))
        best = (distance, {"xy": (float(centre[0]), float(centre[1])), "yaw_deg": float(yaw),
                           "size_mm": tuple(sorted((round(float(np.linalg.norm(e1)) * 1000, 1),
                                                    round(float(np.linalg.norm(e2)) * 1000, 1)), reverse=True))})
    return best[1] if best else None


def placement_error(spot, found):
    """(dx_mm, dy_mm, dyaw_deg) of a found tile from its spot, or None; dyaw_deg is None if its yaw wasn't read."""
    if found is None:
        return None
    return ((found["xy"][0] - spot["xy"][0]) * 1000, (found["xy"][1] - spot["xy"][1]) * 1000,
            None if found["yaw_deg"] is None else wrap_yaw(found["yaw_deg"] - spot.get("yaw_deg", 0.0)))


def status(spot, found, previous=None):
    """ok (within tolerance) / close (within twice it) / off / missing. A spot that was ok stays ok up to
    HYSTERESIS x the tolerance, so a tile sitting at the edge doesn't flicker green/amber. A yaw that wasn't read
    isn't graded."""
    error = placement_error(spot, found)
    if error is None:
        return "missing"
    worst = math.hypot(error[0], error[1]) / POSITION_TOLERANCE_MM
    if error[2] is not None:
        worst = max(worst, abs(error[2]) / YAW_TOLERANCE_DEG)
    return "ok" if worst <= (HYSTERESIS if previous == "ok" else 1) else "close" if worst <= 2 else "off"


class Smoother:
    """Per-spot median of the last SMOOTH_FRAMES finds, and status with hysteresis. A tile is ~24x18px,
    so a single frame's centre/yaw jitters by about a pixel (~1.5mm, ~3°) — the size of the tolerance."""

    def __init__(self, frames=None):
        self.frames = frames or SMOOTH_FRAMES
        self.history, self.states = {}, {}

    def update(self, spot, found):
        """Add this frame's find for `spot`; returns (smoothed find or None, status)."""
        history = self.history.setdefault(spot["id"], [])
        history.append(found)
        del history[:-self.frames]
        seen = [f for f in history if f is not None]
        if len(seen) * 2 <= len(history):          # missing in most recent frames
            smoothed = None
        else:
            read = [f["yaw_deg"] for f in seen if f["yaw_deg"] is not None]
            yaws = [read[-1] + wrap_yaw(y - read[-1]) for y in read]                                   # across ±90
            smoothed = {"xy": tuple(np.median([f["xy"] for f in seen], axis=0)),
                        "yaw_deg": wrap_yaw(float(np.median(yaws))) if yaws else None, "size_mm": seen[-1]["size_mm"]}
        self.states[spot["id"]] = status(spot, smoothed, self.states.get(spot["id"]))
        return smoothed, self.states[spot["id"]]


def capture(snapshot, spots, intrinsics, T_cam_to_base, frames=CAPTURE_FRAMES, boxes=None):
    """Every spot's reading averaged over `frames` frames: one row per spot with the mean offset (mm), yaw (deg; None
    if no frame read it), their frame-to-frame spread, and how many frames found the tile."""
    finds = {s["id"]: [] for s in spots}
    for _ in range(frames):
        image = snapshot()
        for s in spots:
            f = find_tile(image, s, intrinsics, T_cam_to_base, boxes=boxes)
            if f is not None:
                finds[s["id"]].append(f)
    rows = []
    for s in spots:
        fs = finds[s["id"]]
        row = {"id": s["id"], "xy": [float(v) for v in s["xy"]], "yaw_target": float(s.get("yaw_deg", 0.0)),
               "found_frames": len(fs), "frames": frames}
        if fs:
            errors = [placement_error(s, f) for f in fs]
            xy = np.array([e[:2] for e in errors])
            turns = [e[2] for e in errors if e[2] is not None]
            yaw = s.get("yaw_deg", 0.0) + float(np.mean(turns)) if turns else None
            row.update(dx_mm=float(xy[:, 0].mean()), dy_mm=float(xy[:, 1].mean()), yaw_deg=yaw,
                       dx_sd=float(xy[:, 0].std()), dy_sd=float(xy[:, 1].std()), yaw_sd=float(np.std(turns)) if turns else None,
                       size_mm=[float(v) for v in np.mean([f["size_mm"] for f in fs], axis=0)],
                       status=status(s, {"xy": (s["xy"][0] + xy[:, 0].mean() / 1000, s["xy"][1] + xy[:, 1].mean() / 1000),
                                         "yaw_deg": yaw}))
        rows.append(row)
    return rows


def yaw_trend(rows):
    """Does the measured yaw error drift across the mat? Mean and spread, and the drift in deg per 10cm of
    y and of x from a plane fit — each only if the found spots span TREND_MIN_SPAN_M that way (else None).
    None with fewer than 4 spots found."""
    found = [r for r in rows if r.get("yaw_deg") is not None]
    if len(found) < 4:
        return None
    errors = np.array([wrap_yaw(r["yaw_deg"] - r["yaw_target"]) for r in found])
    axes = [k for k in (1, 0) if np.ptp([r["xy"][k] for r in found]) >= TREND_MIN_SPAN_M]
    A = np.c_[np.ones(len(found)), *[[r["xy"][k] for r in found] for k in axes]]
    slopes = dict(zip(axes, np.linalg.lstsq(A, errors, rcond=None)[0][1:]))
    return {"mean_deg": float(errors.mean()), "spread_deg": float(errors.std()),
            "per_10cm_y": float(slopes[1] * 0.1) if 1 in slopes else None,
            "per_10cm_x": float(slopes[0] * 0.1) if 0 in slopes else None}


def capture_report(rows, trend):
    lines = [f"  {'spot':<5} {'found':>6} {'dx mm':>7} {'dy mm':>7} {'yaw':>6} {'±':>4} {'size mm':>12}  status"]
    for r in rows:
        if "dx_mm" not in r:
            lines.append(f"  {r['id']:<5} {r['found_frames']:>3}/{r['frames']:<2}  no tile found")
            continue
        yaw = f"{r['yaw_deg']:+6.1f} {r['yaw_sd']:4.1f}" if r["yaw_deg"] is not None else f"{'-':>6} {'-':>4}"
        lines.append(f"  {r['id']:<5} {r['found_frames']:>3}/{r['frames']:<2} {r['dx_mm']:+7.1f} {r['dy_mm']:+7.1f} {yaw} "
                     f"{r['size_mm'][0]:5.1f}x{r['size_mm'][1]:<5.1f}  {r['status']}")
    if trend:
        drift = [f"{trend[k]:+.1f}° per 10cm of {axis} ({side} +)" for k, axis, side in
                 (("per_10cm_y", "y", "left"), ("per_10cm_x", "x", "away")) if trend[k] is not None]
        lines.append(f"  yaw error: mean {trend['mean_deg']:+.1f}°, spread {trend['spread_deg']:.1f}°"
                     + (f"; drift {', '.join(drift)}" if drift else ""))
    return "\n".join(lines)


# --- drawing ---

def draw(image_bgr, spots, intrinsics, T_cam_to_base, found=None, active=None, ghost=None, states=None):
    """A copy of the frame with each spot's outline (coloured by status — `states`, e.g. a Smoother's, or
    computed from `found`) and, for the active spot, the error as text. ghost: a reference frame blended in."""
    out = image_bgr.copy()
    if ghost is not None:
        out = cv2.addWeighted(out, 1 - GHOST_ALPHA, ghost, GHOST_ALPHA, 0)
    found = found or {}
    for spot in spots:
        corners = table_to_pixel(tile_corners(*spot["xy"], spot.get("yaw_deg", 0.0)), intrinsics, T_cam_to_base)
        state = (states or {}).get(spot["id"]) or status(spot, found.get(spot["id"]))
        colour = COLOURS[state] if active in (None, spot["id"]) else COLOURS["dim"]
        cv2.polylines(out, [np.round(corners).astype(np.int32)], True, colour, 2 if active == spot["id"] else 1)
        u, v = corners.mean(axis=0)
        cv2.putText(out, spot["id"], (int(u) - 10, int(corners[:, 1].min()) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        if active in (None, spot["id"]):
            error = placement_error(spot, found.get(spot["id"]))
            if active == spot["id"]:
                yaw = None if error is None else found[spot["id"]]["yaw_deg"]
                text = (f"{spot['id']}: no tile found" if error is None else
                        f"{spot['id']}: x {error[0]:+.0f}mm  y {error[1]:+.0f}mm  "
                        + ("line up the outline by eye" if yaw is None else
                           f"yaw {yaw:+.0f} (want {spot.get('yaw_deg', 0.0):+.0f})"))
                cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


# --- where the arm can comfortably work ---

REACH_GRID_M = 0.01                # the mask's cell size
REACH_AREA_M = ((0.05, 0.40), (-0.35, 0.35))    # x, y range searched (base frame)
REACH_ALPHA = 0.3
REACH_COLOURS = {"comfortable": (0, 160, 0), "edge": (0, 170, 255)}


def reach_band(x, y, correction=None, offset_m=config.HOVER_OFFSET_M):
    """How the arm reaches a tile the RealSense sees at (x, y): "comfortable", "edge" (the hover reaches it,
    but at grasp height the wrist is within gamepad.REACH_WARNING_M of the gamepad's reach or past the IK:
    1D and WW, where corrections and pickups struggled, 2026-10-28/29) or None (no hover). The arm goes to
    the hover correction's point, ~40mm from the camera's (correction: a HoverCorrection)."""
    from . import gamepad, kinematics
    hx, hy = correction.apply(x, y, log=lambda *a: None) if correction else (x, y)
    if kinematics.hover_command(hx, hy, offset_m, config.HOVER_GRIPPER)[0] is None:
        return None
    try:
        left = gamepad.reach_left_m(hx, hy)
    except RuntimeError:            # no XLeRobot checkout: only whether the IK reaches grasp height
        left = math.inf if kinematics.hover_command(hx, hy, 0.0, config.HOVER_GRIPPER)[0] is not None else None
    return "comfortable" if left is not None and left >= gamepad.REACH_WARNING_M else "edge"


def reach_mask(shape, intrinsics, T_cam_to_base, correction=None, grid_m=REACH_GRID_M):
    """{band: uint8 mask} over an image of `shape`: the table cells in each reach_band, as the camera sees
    them. Takes a few seconds (one IK per cell), so callers compute it once."""
    h, w = shape[:2]
    masks = {band: np.zeros((h, w), np.uint8) for band in REACH_COLOURS}
    (x0, x1), (y0, y1) = REACH_AREA_M
    for x in np.arange(x0, x1, grid_m):
        for y in np.arange(y0, y1, grid_m):
            corners = [(x + dx, y + dy) for dx, dy in ((0, 0), (grid_m, 0), (grid_m, grid_m), (0, grid_m))]
            pixels = table_to_pixel([(cx, cy, config.table_z(cx, cy)) for cx, cy in corners], intrinsics, T_cam_to_base)
            if not ((pixels[:, 0] > -w) & (pixels[:, 0] < 2 * w) & (pixels[:, 1] > -h) & (pixels[:, 1] < 2 * h)).all():
                continue
            u, v = pixels.mean(axis=0)
            if not (0 <= u < w and 0 <= v < h):
                continue
            band = reach_band(x + grid_m / 2, y + grid_m / 2, correction)
            if band:
                cv2.fillPoly(masks[band], [np.round(pixels).astype(np.int32)], 255)
    return masks


def draw_reach(image_bgr, masks):
    """The reach masks tinted over the frame, each band's edge outlined, and a legend."""
    out = image_bgr.copy()
    for band, mask in masks.items():
        tint = out.copy()
        tint[mask > 0] = REACH_COLOURS[band]
        out = np.where(mask[..., None] > 0, cv2.addWeighted(out, 1 - REACH_ALPHA, tint, REACH_ALPHA, 0), out)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, REACH_COLOURS[band], 1)
    cv2.putText(out, "reach: green comfortable, amber edge of reach, untinted out of reach", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


PLACE_HELP ="place the tile until its outline is green — Space/Enter: done, g: ghost on/off, s: skip spot, q: quit"


def place(snapshot, spots, spot_id, intrinsics, T_cam_to_base, ghost=None, show=None, wait_key=None, window="layout",
          boxes=None):
    """Live view until the tile at `spot_id` is placed. Returns {"action": accept / skip / quit, "found",
    "status"}. Accepting a tile that isn't green is allowed (its status is logged)."""
    show = show or (lambda image: cv2.imshow(window, image))
    wait_key = wait_key or (lambda: cv2.waitKey(30) & 0xFF)
    spot = next(s for s in spots if s["id"] == spot_id)
    use_ghost, smoother = False, Smoother()
    while True:
        image = snapshot()
        found, state = smoother.update(spot, find_tile(image, spot, intrinsics, T_cam_to_base, boxes=boxes))
        out = draw(image, spots, intrinsics, T_cam_to_base, found={spot_id: found}, active=spot_id,
                   ghost=ghost if use_ghost else None, states={spot_id: state})
        cv2.putText(out, PLACE_HELP, (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        show(out)
        key = wait_key()
        if key in (13, 10, 32):
            return {"action": "accept", "found": found, "status": state}
        if key == ord("g") and ghost is not None:
            use_ghost = not use_ghost
        elif key == ord("s"):
            return {"action": "skip", "found": found, "status": state}
        elif key in (ord("q"), 27):
            return {"action": "quit", "found": found, "status": state}


# --- layouts file ---

def load_layouts(path=None):
    path = path or config.LAYOUTS
    return json.loads(path.read_text()) if path.exists() else {}


def load(name, path=None):
    layouts = load_layouts(path)
    if name not in layouts:
        raise SystemExit(f"no layout {name!r} in {path or config.LAYOUTS} (have: {sorted(layouts) or 'none'}); "
                         f"teach one with scripts/layout.py --name {name} --teach")
    return layouts[name]


def save(name, spots, path=None, ghost=None, note=""):
    """Write layout `name` (replacing it). ghost: the file name of its reference photo, if any."""
    path = path or config.LAYOUTS
    layouts = load_layouts(path)
    layouts[name] = {"created": datetime.now().isoformat(timespec="seconds"), "note": note, "ghost": ghost,
                     "spots": [{"id": s["id"], "xy": [round(float(v), 4) for v in s["xy"]],
                                "yaw_deg": round(float(s.get("yaw_deg", 0.0)), 1),
                                **({"label": s["label"]} if s.get("label") else {})} for s in spots]}
    path.write_text(json.dumps(layouts, indent=2) + "\n")
    return layouts[name]


def ghost_path(name):
    """A layout's reference photo. Only valid while the RealSense stays where it was when taken."""
    return config.DATA / "layouts" / f"{name}.jpg"


def load_ghost(name):
    path = ghost_path(name)
    return cv2.imread(str(path)) if path.exists() else None


def nearest_tile(tiles, xy, radius_m=config.TILE_MERGE_RADIUS_M, name=None):
    """The detected tile (perception.detect_tiles) closest to xy, if within radius_m. With `name`: the closest of that
    name within radius_m, unless there's none (then a misread: the closest of any). Within radius_m first: another copy of
    the name across the mat isn't this tile (a landed 2C read 3C, 20261104_195901_bench, was logged as not seen)."""
    def distance(tile):
        return math.hypot(tile["xy"][0] - xy[0], tile["xy"][1] - xy[1])
    near = [t for t in tiles if distance(t) <= radius_m]
    return min([t for t in near if t["tile_name"] == name] or near, key=distance, default=None)


FACE_DOWN = "face_down"     # the name a face-down tile goes by: nothing can read a back


def saved_target(spot, name=FACE_DOWN):
    """A tile at `spot`'s saved position, standing in for a detection: the detector can't see a green back, so a face-down
    pick aims where the spot was taught (record.py --face-down / --wall-station, bench.py --face-down)."""
    return {"tile_name": name, "xy": [float(v) for v in spot["xy"]], "saved_position": True}


def schedule(spots, repeats=1, start=None):
    """Spot ids in recording order, each `repeats` times in a row, starting at spot id `start`."""
    ids = [s["id"] for s in spots]
    if start is not None:
        if start not in ids:
            raise SystemExit(f"no spot {start!r} in the layout (have: {ids})")
        ids = ids[ids.index(start):]
    return [i for i in ids for _ in range(repeats)]


def spots_from_tiles(tiles, image_bgr, intrinsics, T_cam_to_base, prefix="S", boxes=None):
    """Teach: one spot per detected tile (left to right), yaw and position refined by the finder where
    it sees the tile's face."""
    spots = []
    for i, tile in enumerate(sorted(tiles, key=lambda t: -t["xy"][1]), start=1):
        spot = {"id": f"{prefix}{i}", "xy": tile["xy"], "yaw_deg": 0.0, "label": tile.get("tile_name", "")}
        found = find_tile(image_bgr, spot, intrinsics, T_cam_to_base, boxes=boxes)
        if found is not None:
            spot.update(xy=found["xy"], yaw_deg=found["yaw_deg"] or 0.0)
        spots.append(spot)
    return spots


def render(tiles, intrinsics, T_cam_to_base, shape=(480, 848), size=None):
    """A synthetic RealSense frame: dark mat, a white tile face per (x, y, yaw_deg) — for tests and --sim."""
    image = np.full((*shape, 3), (40, 50, 35), np.uint8)
    for x, y, yaw in tiles:
        corners = table_to_pixel(tile_corners(x, y, yaw, size), intrinsics, T_cam_to_base)
        cv2.fillPoly(image, [np.round(corners).astype(np.int32)], (225, 235, 240))
        u, v = corners.mean(axis=0)
        cv2.circle(image, (int(u), int(v)), 3, (40, 40, 200), -1)          # a printed symbol
    return image
