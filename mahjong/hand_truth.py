"""A person's word on a session's tiles, turned into labelled detector frames: the same scan frames mine_frames.py exports, but
named by what a person said was at each placement, not by what the detector or a reader of crops thought.

A hand scans the mat ~200 times and a person names every tile (the hand check, recounts, the audit), so every frame of it is a
frame with known names. That's labelling the detector can't talk itself into: v7x3 read a 5C as 9C at 0.91 and a 3C as 2C at
0.83 through a whole hand (20261104_183142_play), and pre-labels would have started from those names.

How a frame gets its labels:
- Each placement is followed from scan to scan: names by the person (a tile that moved keeps what was said of it, whichever of
  its placements it was said of), position by the nearest read, since a knock moves a tile without ending its placement.
- A box takes the name of the placement it sits on. A tile the detector missed in this frame borrows its box from the nearest
  scan that saw it in the same place: a miss is what's worth teaching.
- A frame is refused rather than guessed at if a confident box is on the mat with no placement under it, or a placement has
  no box anywhere near. Boxes away from every tile (the discard bin, the bag) get no label.

Order in the log, which each cost a wrong label here: a drawn tile's scans come before its `placement`; a discard's
`placement_ended` comes after the scan that no longer shows it; a tile being returned is scanned before its next placement.
Scans during a pickup (the tile's in the jaws) aren't used, as in name_ledger.SCORED_SCANS.
"""
import csv
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import config, detector
from .session import read_events

USED_SCANS = ("register", "drawn", "choose", "after_discard", "after_return", "name_check", "final")
FOLLOW_M = 0.018            # a read this close to a tile's last place is that tile (rows are ~50mm apart)
ON_MAT_M = 0.06             # a box this close to a tile, on no tile, is something on the mat nobody named
SAME_PLACE_M = 0.002        # a tile's box may be borrowed from another scan only if the tile hasn't moved more than this


def stated_names(events, audited=None):
    """{placement id: the name a person gave it}, spread over each moved tile's placements (`from_placement`): the audit names
    a kept tile by the placement it ended on, and its scans at the drop zone are under the one before. A tile given two
    names isn't named. The last answer on a placement wins, as in the ledger."""
    placements = {e["id"]: e for e in events if e["type"] == "placement"}
    said = dict(audited or {})
    for e in events:
        if e["type"] in ("hand_confirmed", "hand_audit"):
            said.update(e["confirmed"])
    for e in events:
        if e["type"] == "intervention" and e.get("placement"):
            if e.get("kind") in ("name_asked", "recount"):
                said[e["placement"]] = e["answer"]
            elif e.get("kind") == "silent_error":
                said[e["placement"]] = e["table"]
    for p in placements.values():
        if p.get("truth") and p.get("truth_source") == "person":
            said.setdefault(p["id"], p["truth"])

    def root(pid):
        seen = set()
        while pid in placements and placements[pid].get("from_placement") in placements and pid not in seen:
            seen.add(pid)
            pid = placements[pid]["from_placement"]
        return pid

    groups = defaultdict(set)
    for pid in placements:
        if pid in said:
            groups[root(pid)].add(said[pid])
    return {pid: next(iter(groups[root(pid)])) for pid in placements if len(groups[root(pid)]) == 1}


def timeline(events):
    """{placement id: (first event index it's on the mat, the event index it's gone)}. See the module's note on the order."""
    spans, started = {}, {}
    last_attempt = {}
    for i, e in enumerate(events):
        if e["type"] == "attempt" and e.get("placement"):
            last_attempt[e["placement"]] = i
        elif e["type"] == "placement":
            started[e["id"]] = _first_scan_of(events, i)
            spans[e["id"]] = (started[e["id"]], len(events))
        elif e["type"] == "placement_ended" and e.get("id") in started:
            spans[e["id"]] = (started[e["id"]], last_attempt.get(e["id"], i))
    return spans


def _first_scan_of(events, i):
    """The first of the scans just before event i with one label, if they aren't cut off by an attempt: they saw this tile."""
    first, label = i, None
    for k in range(i - 1, -1, -1):
        e = events[k]
        if e["type"] in ("attempt", "discard", "turn_done", "return"):
            break
        if e["type"] == "detections":
            if label not in (None, e.get("label")):
                break
            label, first = e.get("label"), k
    return first


def pixel_to_mat(events):
    """The homography from a scan's pixels to the base frame, fitted on the session's own tiles (box centre -> xy)."""
    import cv2
    pixels, points = [], []
    for e in events:
        if e["type"] == "detections":
            for tile in e.get("tiles") or []:
                if tile.get("pixel") is not None:
                    pixels.append(tile["pixel"])
                    points.append(tile["xy"])
    if len(pixels) < 8:
        return None
    H, _ = cv2.findHomography(np.array(pixels, float), np.array(points, float), cv2.RANSAC, 0.01)
    return H


def _to_mat(H, xy):
    import cv2
    return tuple(float(v) for v in cv2.perspectiveTransform(np.array([[xy]], float), H)[0][0])


def scans(session_dir):
    """Every usable scan's tiles and boxes: [{"scan", "label", "folder", "size", "frames": [{"index", "tiles": {placement:
    box or None}, "unknown": [box]}], "names": {placement: name or None}}]. Boxes are as saved in reads.json."""
    session_dir = Path(session_dir)
    events = read_events(session_dir)
    names, spans, H = stated_names(events), timeline(events), pixel_to_mat(events)
    if H is None:
        return []
    first_xy = {e["id"]: e["xy"] for e in events if e["type"] == "placement"}
    where, out = {}, []
    for i, e in enumerate(events):
        if e["type"] != "detections" or not e.get("frames"):
            continue
        folder = session_dir / e["frames"]
        if not (folder / "reads.json").exists():
            continue
        live = {pid: where.get(pid) or first_xy[pid] for pid, (lo, hi) in spans.items() if lo <= i < hi}
        reads = detector.load_reads(folder)
        boxes_by_frame = [[{**b, "mat": _to_mat(H, detector._centre(b["xyxy"]))} for b in detector.one_box_per_tile(boxes)]
                          for boxes in reads["frames"]]
        _follow(live, boxes_by_frame)
        where.update(live)
        if e.get("label") not in USED_SCANS and not e.get("released"):     # released: an after-pickup scan with the tile let go
            continue
        frames = []
        for index, boxes in enumerate(boxes_by_frame):
            tiles, used = {}, set()
            for pid, xy in live.items():
                near = [b for b in boxes if math.dist(b["mat"], xy) <= FOLLOW_M and id(b) not in used]
                best = max(near, key=lambda b: b["confidence"], default=None)
                tiles[pid] = best
                if best is not None:
                    used.add(id(best))
            unknown = [b for b in boxes if id(b) not in used and b["confidence"] >= config.DETECTION_CONFIDENCE
                       and any(math.dist(b["mat"], xy) <= ON_MAT_M for xy in live.values())]
            frames.append({"index": index, "tiles": tiles, "unknown": unknown})
        out.append({"scan": folder.name, "label": e.get("label"), "folder": folder, "size": reads["size"], "frames": frames,
                    "names": {pid: names.get(pid) for pid in live}, "where": dict(live), "reads_names": reads["names"]})
    return out


def _follow(live, boxes_by_frame):
    """Move each live tile to where this scan's confident reads put it. Nearest pairs first, so two tiles can't take one read."""
    points = {}
    for boxes in boxes_by_frame:
        for b in boxes:
            if b["confidence"] >= config.DETECTION_CONFIDENCE:
                points.setdefault((round(b["mat"][0], 2), round(b["mat"][1], 2)), []).append(b["mat"])
    centres = [tuple(np.median(v, axis=0)) for v in points.values()]
    pairs = sorted((math.dist(xy, c), pid, k) for pid, xy in live.items() for k, c in enumerate(centres))
    moved, taken = set(), set()
    for distance, pid, k in pairs:
        if distance > FOLLOW_M:
            break
        if pid not in moved and k not in taken:
            live[pid] = centres[k]
            moved.add(pid)
            taken.add(k)


def _box_of(scan_list, n, pid):
    """A box for placement `pid` from the scan nearest scan_list[n] that saw it (this one's other frames first), with its
    centre, or None. Only from where the tile was in the same place."""
    here = scan_list[n]["where"][pid]
    for k in sorted(range(len(scan_list)), key=lambda k: abs(k - n)):
        other = scan_list[k]
        if pid not in other["where"] or math.dist(other["where"][pid], here) > SAME_PLACE_M:
            continue
        seen = [f["tiles"][pid] for f in other["frames"] if f["tiles"].get(pid)]
        if seen:
            return max(seen, key=lambda b: b["confidence"])
    return None


def labelled(session_dir):
    """Every frame all of whose tiles have a person's name and a box: [{"session", "scan", "label", "folder", "index", "size",
    "truth": [{"tile_name", "xyxy", "placement"}], "reads": {placement: (the detector's name above the cut, or None)},
    "wrong", "missed"}], and a Counter of why the others were refused."""
    session_dir = Path(session_dir)
    scan_list, out, refused = scans(session_dir), [], Counter()
    for n, scan in enumerate(scan_list):
        if not scan["names"]:
            refused["no tiles"] += len(scan["frames"])
            continue
        if not all(scan["names"].values()):
            refused["a tile nobody named"] += len(scan["frames"])
            continue
        for frame in scan["frames"]:
            if frame["unknown"]:
                refused["a box on the mat on no tile"] += 1
                continue
            truth, reads, wrong, missed, ok = [], {}, [], [], True
            for pid, name in scan["names"].items():
                box = frame["tiles"][pid] or _box_of(scan_list, n, pid)
                if box is None:
                    ok = False
                    break
                truth.append({"tile_name": name, "xyxy": box["xyxy"], "placement": pid})
                seen = frame["tiles"][pid]
                said = seen["tile_name"] if seen and seen["confidence"] >= config.DETECTION_CONFIDENCE else None
                reads[pid] = said
                if said is None:
                    missed.append(name)
                elif said != name:
                    wrong.append((name, said))
            if not ok:
                refused["a tile with no box"] += 1
                continue
            out.append({"session": session_dir.name, "scan": scan["scan"], "label": scan["label"], "folder": scan["folder"],
                        "index": frame["index"], "size": scan["size"], "truth": truth, "reads": reads, "wrong": wrong,
                        "missed": missed, "names": scan["reads_names"]})
    return out, refused


def arrangement(frame):
    """Which tiles are on the mat: frames with the same one differ only by light and a few mm."""
    return tuple(sorted((t["placement"], t["tile_name"]) for t in frame["truth"]))


def pick(frames, per_arrangement=2, per_hard_name=40):
    """The frames worth training on. A scan's 5 frames are near copies and most of a hand is one arrangement, so: the shakiest
    `per_arrangement` frames of each arrangement (different scans first), then, for each name read wrong or missed, the
    shakiest frames holding that tile, up to `per_hard_name` frames of it counting those already taken."""
    def shake(f):
        return len(f["wrong"]) * 2 + len(f["missed"])

    picked, seen = [], set()

    def take(f):
        key = (f["scan"], f["index"])
        if key not in seen:
            seen.add(key)
            picked.append(f)

    by_arrangement = defaultdict(list)
    for f in frames:
        by_arrangement[arrangement(f)].append(f)
    for group in by_arrangement.values():
        scans_used = set()
        for f in sorted(group, key=lambda f: (-shake(f), f["scan"], f["index"])):
            if sum(1 for g in group if (g["scan"], g["index"]) in seen) >= per_arrangement:
                break
            if f["scan"] not in scans_used:
                scans_used.add(f["scan"])
                take(f)
    hard = Counter(name for f in frames for name, _ in f["wrong"]) + Counter(name for f in frames for name in f["missed"])
    for name, _ in hard.most_common():
        hits = [f for f in frames if name in {w for w, _ in f["wrong"]} | set(f["missed"])]
        scans_used = {f["scan"] for f in picked if any(t["tile_name"] == name for t in f["truth"])}
        held = len(scans_used)
        # one frame per scan: spread across the hand rather than copies of one moment
        for f in sorted(hits, key=lambda f: (f["scan"] in scans_used, -shake(f), f["scan"], f["index"])):
            if held >= per_hard_name:
                break
            if f["scan"] not in scans_used:
                scans_used.add(f["scan"])
                take(f)
                held += 1
    return sorted(picked, key=lambda f: (f["scan"], f["index"]))


def export(frames, out_dir):
    """A YOLO export of these frames under out_dir/train/, with a manifest beside it (detector.manifest_path). Refuses an
    exam session: the exam is never trained on."""
    out_dir = Path(out_dir)
    exam = detector.load_exam()["sessions"]
    images, labels = out_dir / "train" / "images", out_dir / "train" / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    with open(detector.manifest_path(out_dir), "w", newline="") as f:
        manifest = csv.writer(f)
        manifest.writerow(["file", "session", "scan", "frame", "tiles", "read_wrong", "missed"])
        for frame in frames:
            if frame["session"] in exam:
                raise SystemExit(f"{frame['session']} is an exam session: never trained on")
            stem = detector.frame_stem(frame["session"], frame["scan"], frame["index"])
            shutil.copy(Path(frame["folder"]) / f"f{frame['index']}.jpg", images / f"{stem}.jpg")
            lines = detector.yolo_lines(frame["truth"], detector.TILE_NAMES, frame["size"])
            (labels / f"{stem}.txt").write_text("".join(line + "\n" for line in lines))
            manifest.writerow([f"{stem}.jpg", frame["session"], frame["scan"], frame["index"], len(frame["truth"]),
                               " ".join(f"{t}->{s}" for t, s in frame["wrong"]), " ".join(frame["missed"])])
    (out_dir / "data.yaml").write_text(detector.data_yaml(detector.TILE_NAMES))
    return len(frames)


def score(frames, predict=None, confident=config.DETECTION_CONFIDENCE):
    """detector.score over these frames: `predict` (BGR image -> boxes) re-reads them with another detector, or None scores the
    reads the session saved. Returns (tally of all frames, {true name: Counter(right, wrong, missed)})."""
    import cv2
    images, per_name = [], defaultdict(Counter)
    for frame in frames:
        if predict is None:
            said = detector.load_reads(frame["folder"])["frames"][frame["index"]]
        else:
            said = predict(cv2.imread(str(Path(frame["folder"]) / f"f{frame['index']}.jpg")))
        said = [b for b in said if b["confidence"] >= confident]
        truth = [{"tile_name": t["tile_name"], "xyxy": t["xyxy"]} for t in frame["truth"]]
        # only boxes on a named tile count: a bin or bag behind the mat has no labels here
        on_mat = [b for b in said if any(detector.iou(b["xyxy"], t["xyxy"]) > 0.1 for t in truth)]
        images.append({"truth": truth, "said": on_mat, "group": "all"})
        pairs, missed, _ = detector.match(truth, on_mat)
        for t, s in pairs:
            per_name[t["tile_name"]]["right" if t["tile_name"] == s["tile_name"] else "wrong"] += 1
        for t in missed:
            per_name[t["tile_name"]]["missed"] += 1
    return detector.score(images)["all"] if images else None, per_name
