"""The arm collects the base camera's training photos (docs/face_down_plan.md, tests 1b and 2): it picks up a face-down
tile a person laid at a layout spot, holds it up to the base camera at a taught pose, saves frames, and sets it back down.

The logic that needs no hardware lives here, so it can be tested: which tiles still need photos (`wanted`, from the holds
already in the hold-up sessions), the poses a hold visits (`pose_names`) and how each is nudged (`jitter`), and the
`holdup_shot` events the other hold-up scripts already read. scripts/base_cam_collect.py moves the arm.

A hold is one pickup: the arm visits every chosen pose with the same tile in the jaws, so the photos of a hold differ in
pose but share the grip, and `base_cam_train.py` keeps a hold whole on one side of its train / test split. Holds count, not
frames: the frames of one pose are near-copies.

Which way up the base camera sees a tile is decided by how a person laid it (the arm can't turn it) and by the pose, so
nobody can say beforehand. A person lays a tile one of two ways, `as_layout` (where the layout's face-up tile would be, the
top of its picture at the same end, flipped sideways about its long edge) or `rotated` (that, turned 180 degrees on the mat),
and says once per session what the first photo shows. `upright` and `turned` then mean what the camera sees (the picture reads
right way up, or upside down in the frame), which is what base_cam_train.py and the reader mean by them.
"""
import math
import random
from collections import defaultdict

from . import base_cam, camera_poses, config, detector
from .session import read_events

POSE_PREFIX = "base_view_"
DEFAULT_TARGET_HOLDS = 6        # holds per tile and way up the shortfall is measured against
DEFAULT_TILES = tuple(detector.TILE_NAMES)      # every tile, flowers too (the owner's choice, 2026-11-11): the short ones first
JITTER_JOINTS = ("shoulder_pan", "wrist_flex", "wrist_roll")     # nudged between holds; lift and elbow set the distance
DEFAULT_JITTER_DEG = 2.0


def interleaved(session_dir):
    """Whether a hold-up session was collected the interleaved way (several tiles, one per spot, moved between arrangements):
    the only way whose holds a reader can be trained on. Earlier ones (by hand, or the arm a tile at a time, 2026-11-05/06) let
    the scene name the tile, so they aren't counted, and the reader was never trained on them."""
    return any(e["type"] == "holdup_conditions" and e.get("order") == "interleaved" for e in read_events(session_dir))


def counted_sessions(sessions_dir=None):
    """(the real, interleaved hold-up sessions, the real ones left out): only the first count (`interleaved`)."""
    folder = sessions_dir or config.SESSIONS
    real = [s for s in sorted(folder.glob("*_holdup")) if not base_cam.is_sim(s)]
    counted = [s for s in real if interleaved(s)]
    return counted, [s for s in real if s not in counted]


def hold_counts(sessions_dir=None, review=None):
    """{(tile, kind): distinct holds} over every real, interleaved hold-up session (`interleaved`); simulated ones don't count.
    By a person's review (base_cam.rows): a hold renamed counts under its right name, one whose photos are all wrong not at all."""
    return {key: len(v["holds"]) for key, v in _tallies(sessions_dir, review).items()}


def _tallies(sessions_dir, review):
    review = base_cam.load_review() if review is None else review
    seen = defaultdict(lambda: {"holds": set(), "photos": 0, "reviewed": 0, "sessions": set(), "daylight": defaultdict(set)})
    for session in counted_sessions(sessions_dir)[0]:
        for row in base_cam.rows(session, review):
            if row["hold"] is None:         # a fixed-distance session from before holds: nothing to count
                continue
            t = seen[(row["truth"], row["kind"])]
            hold = (row["session"], row["shot"])        # by shot: a renamed hold's number can be one the right name has too
            t["holds"].add(hold)
            t["photos"] += 1
            t["reviewed"] += row["reviewed"]
            t["sessions"].add(row["session"])
            t["daylight"][row["daylight"] or "unknown"].add(hold)
    return seen


def tile_counts(sessions_dir=None, review=None, tiles=DEFAULT_TILES, kinds=base_cam.KINDS):
    """{tile: {"holds": {kind: n}, "photos", "reviewed" (photos a person has looked at), "sessions", "daylight": {bright / dim / none: holds}}} for
    every tile in `tiles`, 0s for one never collected: how well each is represented in what the reader trains on."""
    tallies = _tallies(sessions_dir, review)
    out = {}
    for tile in tiles:
        parts = [tallies[(tile, k)] for k in kinds if (tile, k) in tallies]
        out[tile] = {"holds": {k: len(tallies[(tile, k)]["holds"]) if (tile, k) in tallies else 0 for k in kinds},
                     "photos": sum(p["photos"] for p in parts), "reviewed": sum(p["reviewed"] for p in parts),
                     "sessions": len(set().union(*(p["sessions"] for p in parts))),
                     "daylight": {light: sum(len(p["daylight"].get(light, ())) for p in parts)
                                  for light in sorted({light for p in parts for light in p["daylight"]})}}
    return out


def wanted(tiles, counts, target=DEFAULT_TARGET_HOLDS, kinds=base_cam.KINDS):
    """What to collect next: [{"tile", "kind", "missing"}], the tile furthest short first (both ways up of a tile together,
    so a person lays one tile and turns it round). Tiles at `target` holds in every way up are left out."""
    short = {t: {k: max(0, target - counts.get((t, k), 0)) for k in kinds} for t in tiles}
    order = sorted((t for t in tiles if sum(short[t].values())), key=lambda t: -sum(short[t].values()))
    return [{"tile": t, "kind": k, "missing": short[t][k]} for t in order for k in kinds if short[t][k]]


LAYINGS = ("as_layout", "rotated")      # the two ways a person lays a tile face down; which way up each shows is learned


def tile_groups(wanted_rows, tiles, size):
    """The tiles laid together, `size` at a time (one per spot), furthest short first. A group short of `size` is filled with
    other tiles from `tiles` (ones already at target first), photographed like the rest, so no tile is collected alone."""
    short = list(dict.fromkeys(w["tile"] for w in wanted_rows))
    groups = [short[i:i + size] for i in range(0, len(short), size)]
    if groups and len(groups[-1]) < size:
        last = groups[-1]
        fill = [t for t in tiles if t not in short] + [t for t in short if t not in last]
        last += fill[:size - len(last)]
    return groups


def arrangements(wanted_rows, tiles, n_spots, max_rounds):
    """What to lay, in order: [{"tiles": one per spot, "laying", "rounds"}]. Collected one tile at a time, a tile's holds came in a
    row from one spot, and the light, the spot and whoever was in view named it better than its face did (2026-11-06: 99% with
    the tile blacked out). So a group of tiles is laid one per spot, and each round picks every spot once: holds of different
    tiles alternate. Between arrangements every tile moves one spot along and the laying alternates, so each tile is held from
    every spot, both ways up, through the whole session. `rounds` is what the group's furthest-short tile still needs, at most
    `max_rounds` (each pickup sets the tile down by memory, so its error compounds; a next run carries on from what's short)."""
    size = min(n_spots, len(tiles))
    missing = {}
    for w in wanted_rows:
        missing[w["tile"]] = max(missing.get(w["tile"], 0), w["missing"])
    out = []
    for group in tile_groups(wanted_rows, tiles, size):
        need = max(missing.get(t, 0) for t in group)
        rounds = min(max(1, math.ceil(need / size)), max_rounds)
        for shift in range(size):
            for laying in LAYINGS:
                out.append({"tiles": group[shift:] + group[:shift], "laying": laying, "rounds": rounds})
    return out


def pose_names(prefix=POSE_PREFIX, poses=None):
    """The saved poses a hold visits, in order: `base_view_1`, `base_view_2`, ... (numbered, not alphabetical)."""
    poses = camera_poses.load() if poses is None else poses
    def number(name):
        tail = name[len(prefix):]
        return (0, int(tail)) if tail.isdigit() else (1, 0)
    return sorted((n for n in poses if n.startswith(prefix)), key=lambda n: (number(n), n))


def jitter(pose, rng, deg=DEFAULT_JITTER_DEG):
    """A copy of the five-joint `pose` with pan, wrist flex and wrist roll each moved by up to ±`deg` degrees."""
    out = {j: float(pose[j]) for j in config.ARM_JOINTS}
    if deg:
        for joint in JITTER_JOINTS:
            out[joint] += rng.uniform(-deg, deg)
    return out


def learn_view(laying, seen):
    """{laying: way up} from one answer: this laying showed `seen` ("upright" or "turned"); the other laying shows the other."""
    other = base_cam.KINDS[1 - base_cam.KINDS.index(seen)]
    return {laying: seen, LAYINGS[1 - LAYINGS.index(laying)]: other}


def top_end_sign(laying, layout_top="far"):
    """+1 if the top of the tile's picture goes at the spot's far (+x, away from the robot) end, -1 at the near end:
    `as_layout` puts it where the layout's face-up tile has it (`layout_top`), `rotated` at the other end."""
    far = layout_top == "far"
    return 1 if far == (laying == "as_layout") else -1


def guide_frame(image, spot, intrinsics, T_cam_to_base, text, sign):
    """The RealSense frame with the target spot's outline (green), the short end where the top of the picture goes drawn
    thick and labelled TOP (sign 1 or -1; None when which way up doesn't matter: no mark), and `text` along the top, one
    line per newline: what a person lines the tile up with. The detector can't see a face-down
    tile, so nothing here says whether it is placed: it is by eye, as the layout script says when it has no yaw."""
    import cv2
    import numpy as np
    from . import layout
    out = layout.draw(image, [spot], intrinsics, T_cam_to_base, states={spot["id"]: "ok"})
    if sign is not None:
        corners = layout.table_to_pixel(layout.tile_corners(*spot["xy"], spot.get("yaw_deg", 0.0)), intrinsics, T_cam_to_base)
        a, b = (corners[0], corners[3]) if sign > 0 else (corners[1], corners[2])       # tile_corners: +x end is 0 -> 3
        cv2.line(out, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (0, 220, 255), 4)
        mid = (a + b) / 2
        away = mid - corners.mean(axis=0)
        label = mid + away * 0.9
        cv2.putText(out, "TOP", (int(label[0]) - 14, int(label[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
    lines = text.split("\n")
    cv2.rectangle(out, (0, 0), (out.shape[1], 10 + 26 * len(lines)), (0, 0, 0), -1)         # the mat is pale and striped: white text needs a ground
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 25 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out, "Space/Enter: laid, go    s: skip    q: quit", (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    return out


def lay_text(tile, laying, spot):
    """What a person does: the tile face down on `spot`, laid one of the two ways."""
    turn = "" if laying == "as_layout" else ", then turned 180 degrees on the mat (its picture's top at the other end)"
    return (f"Lay {tile} FACE DOWN on {spot['id']} ({spot['xy'][0]:.3f}, {spot['xy'][1]:+.3f}): where the layout's face-up tile "
            f"would be, the top of its picture at the same end, flipped sideways about its long edge so its long side stays "
            f"along the robot's x axis{turn}")


def shot_event(tile, kind, hold, files, frame_poses, **extra):
    """The fields of a `holdup_shot` event (what base_cam.rows reads) plus where it came from: by the arm, and each frame's pose."""
    return {"tile": tile, "kind": kind, "hold": hold, "files": files, "frame_poses": frame_poses, "source": "arm", **extra}


def rng_for(seed):
    return random.Random(seed)
