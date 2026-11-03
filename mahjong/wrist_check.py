"""Is a tile still where the scans last saw it? A close look with the wrist camera, for when a RealSense scan
says a tile went: YOLO reads a ~24px tile from 0.8m and a read can drop out for a scan (7D read as 6D, 2026-10-29),
which the bench used to ask a person about, or skip the spot over.

At a policy's hover the tile under the jaws sits between the jaw tips, at nearly the same place in the wrist image
whatever the spot: its face's centre was at 325-391 x 201-290px on the first frame of every attempt in the hand14 /
close-hand14 bench sessions (20261029_094543, 20261029_165537 on the black top, 20261030_094904 on brown paper), and
the setup reference's F2 / F6 agree. Neighbours show too (the layouts are tight), but their faces start >= ~70px
from that point, and touching faces merge into one blob (setup_check.wrist_tile then finds nothing). So the check is
how much of a disc at that point is tile face — pale and bright, the printed symbols closed over: 0.73-1.0 with the
tile there on the 103 black-top frames (median 1.0), ~0 over the bare black top. It sees *a* face, not *this* tile:
when the scan has another tile within CROWDED_M of the spot (a knock pushed one there), it doesn't look.

A pale surface fills the disc without any tile: bare white stripe on the standard mat reads 0.72 (20261103_120259_setup).
So with no dark surface in view it can only say the view hasn't changed since an earlier look at this place
(SAME_VIEW_SCORE), or that it can't tell. It never reads the name: the table
model on a close-up isn't a second read (docs/autonomy_plan.md, 8), and a template chosen by the lost tile's name wouldn't
do either, since that name can be the misread that lost it (all 11 looks in 20261103_121112_bench).
"""
import math

import cv2
import numpy as np

from .camera import grab_frames
from .layout import FACE_MAX_S
from .setup_check import WRIST_MIN_V, match_tile, wrist_template

EXPECTED_PX = (365, 255)    # the hovered tile's face centre in the wrist image (640x480)
RADIUS_PX = 35              # well inside a face (~140 x 120px at the hover) even 25px off
PRESENT_FILL = 0.5          # share of the disc that's face: >= 0.73 with the tile there, ~0 without
FRAMES = 3
# Share of the view (above the jaws) that's dark: at least 0.36 on the black top and 0.29 on the green mat, with tiles and
# jaws in view; at most 0.12 on the striped mat and 0.09 on paper. A dim room darkens a pale mat too; not measured.
SURFACE_ROWS, DARK_V, MIN_DARK_SHARE = 300, 90, 0.2
# The view at a place vs an earlier look there, on the striped mat (20261103_121112_bench): 0.96-1.00 on 8 looks, at most
# 0.81 for another place and 0.70 for bare stripe. Only an unmoved tile counts: ~7mm at the hover (4px/mm) is a knock.
SAME_VIEW_SCORE, SAME_VIEW_PX = 0.9, 30
_LOOK_TILE = {"centre_px": [float(v) for v in EXPECTED_PX], "area_px": 15000.0, "angle_deg": 0.0, "size_px": [142.0, 105.0]}
# Another tile now this close to the spot (centres) can fill the disc as well: the look can't tell them apart. On
# 2026-10-30 (20261030_115706_bench) a miss at S4 knocked 3D to 20mm from 1B's spot and pushed 1B 36mm away; the
# look at 1B's spot saw 3D's face and called 1B's absence a detection glitch.
CROWDED_M = 0.03


def crowded(xy, tiles):
    """A detected tile (from the scan the tile went missing from) within CROWDED_M of xy: it, else None."""
    return next((t for t in tiles if math.dist(t["xy"], xy) <= CROWDED_M), None)


def face_fill(image_bgr, centre=EXPECTED_PX, radius_px=RADIUS_PX):
    """Share of the disc at `centre` that's tile face (pale, bright; symbols closed over)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] <= FACE_MAX_S) & (hsv[:, :, 2] >= WRIST_MIN_V)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    disc = np.zeros_like(mask)
    cv2.circle(disc, tuple(int(v) for v in centre), radius_px, 1, -1)
    return float(mask[disc > 0].mean())


def dark_share(image_bgr):
    """Share of the view above the jaws that's dark: how much surface a pale tile face can stand out from."""
    return float((cv2.cvtColor(image_bgr[:SURFACE_ROWS], cv2.COLOR_BGR2HSV)[:, :, 2] < DARK_V).mean())


def view_template(image_bgr):
    """A look's frame, as what check() compares a later look at the same place with."""
    return wrist_template(image_bgr, EXPECTED_PX)


def same_view(image_bgr, template):
    """Whether a frame shows what `template` (view_template of an earlier look here) showed, where it showed it."""
    found = match_tile(image_bgr, template, _LOOK_TILE, min_score=SAME_VIEW_SCORE)
    return found is not None and math.dist(found["centre_px"], EXPECTED_PX) <= SAME_VIEW_PX


def look(arm, camera, x, y, hover, n=FRAMES):
    """Lift if low, hover over closed-loop target (x, y) with `hover` ({offset_m, clearance_m, gripper}) and grab n
    wrist frames: (frames — fewer if the camera is lost —, the hover's result or None if out of reach)."""
    arm.lift()
    result = arm.hover(x, y, hover["offset_m"], hover["clearance_m"], hover["gripper"], log=lambda *_: None)
    if result is None:
        return [], None
    return grab_frames(camera, n), result


def check(arm, camera, x, y, hover, template=None):
    """Look at the tile expected at hover target (x, y): {"present": True / False, or None if it couldn't look (out
    of reach, camera lost) or can't tell (a pale surface: "pale" True), "fill": the median face share, "hover_xy"}, and
    "frame" (the last one) to save. `template`: what an earlier look saw here, which is all a pale surface can go on."""
    frames, result = look(arm, camera, x, y, hover)
    if not frames:
        return {"present": None, "fill": None, "why": "out of reach" if result is None else "no wrist frames",
                "frame": None}
    fill = float(np.median([face_fill(f) for f in frames]))
    if float(np.median([dark_share(f) for f in frames])) < MIN_DARK_SHARE:
        if template is not None and sum(same_view(f, template) for f in frames) * 2 > len(frames):
            return {"present": True, "fill": fill, "pale": True, "by": "same view", "hover_xy": result["hover_xy"],
                    "frame": frames[-1]}
        return {"present": None, "fill": fill, "pale": True, "why": "the surface is too pale to tell a tile from it",
                "hover_xy": result["hover_xy"], "frame": frames[-1]}
    return {"present": fill >= PRESENT_FILL, "fill": fill, "hover_xy": result["hover_xy"], "frame": frames[-1]}
