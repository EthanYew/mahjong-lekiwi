"""The tile detector's data and scores: the frames each RealSense scan looked at, which of them it
struggled with, pre-labelled exports for Roboflow, and scoring a detector on the exam.

A scan's folder (sessions/<session>/scans/<n>_<label>/, save_scan) holds f0.jpg ... f4.jpg (the colour
frames YOLO saw), depth.npz (the last frame's filtered depth: raw z16 and depth_scale_m) and reads.json
(the class names, the image size, and every box YOLO found in each frame down to
config.DETECTION_SAVE_CONFIDENCE; boxes below config.DETECTION_CONFIDENCE never became tiles).

The exam is labelled frames of the current scene that are never trained on: whole capture sessions
listed in data/detector_exam.json (a scan's 5 frames are near copies, so a session is split as one).
Exported frames are named <session>__<scan>__f<i>.jpg; Roboflow keeps that stem (as
<stem>_jpg.rf.<hash>.jpg), so original_name() finds a labelled frame's session again, and that is how
exam frames are kept out of training. Scores are per frame, while the live scan votes over 5 frames
(perception.merge_tiles), so the exam is harsher than the bench; and Roboflow re-encodes its copies of
the frames, which alone cost the July weights ~8% of their confident boxes (2026-10-28).
"""
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .session import read_events

# the deployed detector's classes (assets/tile_detector.pt), for the sim and tests
TILE_NAMES = ["1B", "1C", "1D", "1F", "2B", "2C", "2D", "2F", "3B", "3C", "3D", "3F", "4B", "4C", "4D", "4F",
              "5B", "5C", "5D", "6B", "6C", "6D", "7B", "7C", "7D", "8B", "8C", "8D", "9B", "9C", "9D",
              "EW", "GD", "NW", "RD", "SW", "WD", "WW"]
LOW_CONFIDENCE = 0.7        # a tile whose best read is below this is worth labelling
CLUSTER_PX = 12             # boxes this close (centres), across a scan's frames, are one tile (~24x18px)
MATCH_IOU = 0.5             # a detection and a labelled tile overlapping this much are the same tile
SEP = "__"                  # <session>__<scan>__f<i>


# --- scans ---

def save_scan(folder, frames, names):
    """Write a scan's frames ([{"color", "boxes", "depth"?, "depth_scale_m"?}]) to `folder`."""
    import cv2
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(folder / f"f{i}.jpg"), frame["color"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    depths = [f for f in frames if f.get("depth") is not None]
    if depths:
        np.savez_compressed(folder / "depth.npz", depth=depths[-1]["depth"], depth_scale_m=depths[-1]["depth_scale_m"])
    h, w = frames[0]["color"].shape[:2]
    (folder / "reads.json").write_text(json.dumps({"names": list(names), "size": [w, h],
                                                   "frames": [f["boxes"] for f in frames]}) + "\n")


def save_session_scan(realsense, session, scan_n, label):
    """The frames `realsense` just scanned into session/scans/<n>_<label>/. Returns {"frames": that folder,
    relative to the session} for the scan's `detections` event, or {} if it kept none."""
    if not getattr(realsense, "last_frames", None):
        return {}
    folder = session.dir / "scans" / f"{scan_n:03d}_{label}"
    save_scan(folder, realsense.last_frames, realsense.names)
    return {"frames": str(folder.relative_to(session.dir))}


def log_scan(realsense, session, scan_n, label, **fields):
    """Detect tiles, save the scan's frames (save_session_scan) and log it as a `detections` event. Returns
    (tiles, the event)."""
    tiles = realsense.detect_tiles()
    frames = save_session_scan(realsense, session, scan_n, label)
    event = session.event("detections", label=label, scan=scan_n, **fields, **frames,
                          tiles=[{k: t[k] for k in ("tile_name", "xy", "confidence", "reads", "box_px") if k in t}
                                 for t in tiles], faint=realsense.faint)
    return tiles, event


def load_reads(folder):
    return json.loads((Path(folder) / "reads.json").read_text())


def _centre(xyxy):
    return ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)


def cluster_tiles(frames, radius_px=CLUSTER_PX):
    """One entry per physical tile across a scan's frames: [{"centre", "reads": [(frame, box)]}]. Boxes
    join by centre, whatever their name: one tile read as two names is still one tile."""
    tiles = []
    for i, boxes in enumerate(frames):
        for box in boxes:
            c = _centre(box["xyxy"])
            near = min(tiles, key=lambda t: np.hypot(t["centre"][0] - c[0], t["centre"][1] - c[1]), default=None)
            if near is not None and np.hypot(near["centre"][0] - c[0], near["centre"][1] - c[1]) <= radius_px:
                near["reads"].append((i, box))
            else:
                tiles.append({"centre": c, "reads": [(i, box)]})
    return tiles


def scan_issues(reads, confident=config.DETECTION_CONFIDENCE, low=LOW_CONFIDENCE):
    """What a scan's frames show the detector struggling with: (Counter of issues, frame to label).
    flicker: a tile given more than one name above the cut. dropout: a tile above the cut in some frames
    but not others. low: a tile whose best read is below `low` (below the cut it was never reported).
    The frame to label is where those tiles read lowest together (unseen counts as 0)."""
    frames = reads["frames"]
    issues, totals = Counter(), np.zeros(len(frames))
    for tile in cluster_tiles(frames):
        sure = [(i, b) for i, b in tile["reads"] if b["confidence"] >= confident]
        best = max(b["confidence"] for _, b in tile["reads"])
        found = []
        if len({b["tile_name"] for _, b in sure}) > 1:
            found.append("flicker")
        if sure and len({i for i, _ in sure}) < len(frames):
            found.append("dropout")
        if best < low:
            found.append("low")
        if found:
            issues.update(found)
            totals += [max((b["confidence"] for i, b in tile["reads"] if i == k), default=0.0) for k in range(len(frames))]
    return issues, int(np.argmin(totals)) if issues else 0


# --- names: an exported frame keeps its session through a Roboflow round trip ---

def frame_stem(session, scan, frame):
    return SEP.join([session, scan, f"f{frame}"])


def original_name(filename):
    """The exported stem of a frame, before or after Roboflow ("<stem>_jpg.rf.<hash>.jpg" -> "<stem>")."""
    name = Path(filename).name
    if ".rf." not in name:
        return Path(name).stem
    stem = name.split(".rf.")[0]
    for ext in ("_jpg", "_jpeg", "_png"):
        if stem.endswith(ext):
            return stem[:-len(ext)]
    return stem


def session_of(filename):
    """The session an exported frame came from, or None (e.g. the July frames of the old dataset)."""
    stem = original_name(filename)
    return stem.split(SEP)[0] if SEP in stem else None


def shared_tiles(a, b):
    """The share of labelled frame a's tiles that b has under the same name, centres within half a tile."""
    if not a:
        return 0.0
    hits = 0
    for x in a:
        (cx, cy), r = _centre(x["xyxy"]), (x["xyxy"][2] - x["xyxy"][0]) / 2
        hits += any(y["tile_name"] == x["tile_name"] and abs(_centre(y["xyxy"])[0] - cx) < r
                    and abs(_centre(y["xyxy"])[1] - cy) < r for y in b)
    return hits / len(a)


def scenes(frames, share=0.6):
    """A scene number for each labelled frame ([[box]]): frames sharing `share` of their tiles both ways are
    one scene, and so on transitively. The July frames were each shot at 4 exposures, so a split by image
    put the same scene in train and valid (134 of v4's 182 valid frames had a train twin)."""
    parent = list(range(len(frames)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(frames)):
        for j in range(i):
            if root(i) != root(j) and min(shared_tiles(frames[i], frames[j]), shared_tiles(frames[j], frames[i])) >= share:
                parent[root(i)] = root(j)
    return [root(i) for i in range(len(frames))]


# --- the exam ---

def load_exam(path=None):
    path = path or config.DETECTOR_EXAM
    return json.loads(path.read_text()) if path.exists() else {"sessions": {}}


def add_exam_session(session_name, conditions, note="", path=None):
    """List a session's frames as exam frames (never trained on)."""
    path = path or config.DETECTOR_EXAM
    exam = load_exam(path)
    exam["sessions"][session_name] = {"added": datetime.now().isoformat(timespec="seconds"),
                                      "conditions": conditions, "note": note}
    path.write_text(json.dumps(exam, indent=2) + "\n")
    return exam


def exam_frames(paths, exam=None):
    """The paths (images or labels) that are exam frames."""
    sessions = (exam or load_exam())["sessions"]
    return [p for p in paths if session_of(p) in sessions]


# --- mining: which frames of past sessions to label ---

def mine(session_dir, everything=False):
    """A session's frames worth labelling: [{"session", "scan", "folder", "frame", "reasons", "conditions"}].
    A scan with issues (scan_issues), or one a person called a detection glitch, gives its shakiest
    frame. everything=True also takes frame 0 of every other scan: plain frames of the current scene."""
    session_dir = Path(session_dir)
    events = read_events(session_dir)
    glitches = {e.get("scan") for e in events
                if e["type"] == "scan_review" and "scan_glitch" in (e.get("kinds") or [])}
    rows, conditions = [], {}
    for e in events:
        if e["type"] in ("session_start", "conditions_changed"):
            conditions = e.get("conditions") or {}
        if e["type"] != "detections" or not e.get("frames"):
            continue
        folder = session_dir / e["frames"]
        if not (folder / "reads.json").exists():
            continue
        issues, frame = scan_issues(load_reads(folder))
        reasons = [f"{k}:{n}" for k, n in sorted(issues.items())] + (["glitch"] if e.get("scan") in glitches else [])
        if reasons or everything:
            rows.append({"session": session_dir.name, "scan": folder.name, "folder": folder,
                         "frame": frame if reasons else 0, "reasons": reasons, "conditions": conditions})
    return rows


def is_sim(session_dir):
    start = next((e for e in read_events(Path(session_dir)) if e["type"] == "session_start"), {})
    return bool(start.get("sim"))


def yolo_lines(boxes, names, size):
    """Boxes as YOLO label lines: "<class> <cx> <cy> <w> <h>", normalised to the image size."""
    w, h = size
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        lines.append(f"{names.index(b['tile_name'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def read_yolo_labels(path, names, size):
    """[{"tile_name", "xyxy"}] from a YOLO label file: boxes, or polygons (scored by their bounding box)."""
    w, h = size
    boxes = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, values = int(parts[0]), [float(v) for v in parts[1:]]
        if len(values) == 4:
            cx, cy, bw, bh = values
            xyxy = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
        else:
            xs, ys = np.array(values[0::2]) * w, np.array(values[1::2]) * h
            xyxy = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        boxes.append({"tile_name": names[cls], "xyxy": xyxy})
    return boxes


def manifest_path(out_dir):
    """Why each exported frame was picked: beside the export, so the upload folder is only the dataset."""
    return Path(out_dir).parent / f"{Path(out_dir).name}_manifest.csv"


def export(rows, out_dir, split="train"):
    """A pre-labelled batch for Roboflow, laid out as Roboflow's own YOLO export (which it imports back
    with its class names; a bare images/ + labels/ folder was refused for want of a label map):
    out_dir/<split>/images/<session>__<scan>__f<i>.jpg, the detector's own boxes above the cut in
    out_dir/<split>/labels/, and data.yaml (split paths, nc, names). The manifest goes beside the folder
    (manifest_path). Returns the number of frames."""
    out_dir = Path(out_dir)
    images, labels = out_dir / split / "images", out_dir / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    names = None
    with open(manifest_path(out_dir), "w", newline="") as f:
        manifest = csv.writer(f)
        manifest.writerow(["file", "session", "scan", "frame", "reasons", "lighting"])
        for row in rows:
            reads = load_reads(row["folder"])
            if names is None:
                names = reads["names"]
            elif reads["names"] != names:
                raise SystemExit(f"{row['folder']}: a detector with other class names than the rest — export it apart")
            stem = frame_stem(row["session"], row["scan"], row["frame"])
            shutil.copy(Path(row["folder"]) / f"f{row['frame']}.jpg", images / f"{stem}.jpg")
            boxes = one_box_per_tile([b for b in reads["frames"][row["frame"]]
                                      if b["confidence"] >= config.DETECTION_CONFIDENCE])
            lines = yolo_lines(boxes, names, reads["size"])
            (labels / f"{stem}.txt").write_text("".join(line + "\n" for line in lines))
            manifest.writerow([f"{stem}.jpg", row["session"], row["scan"], row["frame"], " ".join(row["reasons"]),
                               lighting(row["conditions"])])
    (out_dir / "data.yaml").write_text(data_yaml(names or []))
    return len(rows)


def one_box_per_tile(boxes, min_iou=0.5):
    """The most confident of overlapping boxes. The detector suppresses overlaps only within a name, so one
    tile can come out as 5B and 7B at once; both went into the pre-labels, and correcting one in Roboflow
    left the other behind (5 tiles of 2026-10-27's training and exam frames)."""
    kept = []
    for box in sorted(boxes, key=lambda b: -b["confidence"]):
        if not any(iou(k["xyxy"], box["xyxy"]) >= min_iou for k in kept):
            kept.append(box)
    return kept


def data_yaml(names):
    """data.yaml as Roboflow writes it in its YOLO exports."""
    return ("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
            f"nc: {len(names)}\nnames: [{', '.join(repr(str(n)) for n in names)}]\n")


def lighting(conditions):
    from .offline_eval import lighting as label
    return label(conditions)


# --- scoring ---

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(truth, said, min_iou=MATCH_IOU):
    """Pair the detector's boxes (`said`) with the labelled tiles (`truth`) by overlap, most confident
    first, names aside: (pairs [(truth box, said box)], missed truth boxes, extra said boxes)."""
    free, pairs, extra = list(truth), [], []
    for box in sorted(said, key=lambda b: -b.get("confidence", 1.0)):
        best = max(free, key=lambda t: iou(t["xyxy"], box["xyxy"]), default=None)
        if best is not None and iou(best["xyxy"], box["xyxy"]) >= min_iou:
            free.remove(best)
            pairs.append((best, box))
        else:
            extra.append(box)
    return pairs, free, extra


def score(images, ignore=()):
    """images: [{"truth": boxes, "said": boxes, "group": label}] -> {group: tally}, with "all" too. A tally:
    frames, tiles (labelled), found, named_right, missed, extra (boxes on no tile), mixups {(true, said): n},
    missed_names {true: n}. Tiles named in `ignore` (e.g. the flowers) are matched like any other, then
    left out: their boxes count for nothing, while another tile called by an ignored name is still a mix-up."""
    tallies = defaultdict(lambda: {"frames": 0, "tiles": 0, "found": 0, "named_right": 0, "missed": 0, "extra": 0,
                                   "mixups": Counter(), "missed_names": Counter()})
    for image in images:
        pairs, missed, extra = match(image["truth"], image["said"])
        if ignore:
            ignored = [t for t in image["truth"] if t["tile_name"] in ignore]
            pairs = [(t, s) for t, s in pairs if t["tile_name"] not in ignore]
            missed = [t for t in missed if t["tile_name"] not in ignore]
            extra = [s for s in extra if not any(iou(t["xyxy"], s["xyxy"]) > 0.1 for t in ignored)]
        truth = [t for t in image["truth"] if t["tile_name"] not in ignore]
        for group in dict.fromkeys(["all", image.get("group", "all")]):
            t = tallies[group]
            t["frames"] += 1
            t["tiles"] += len(truth)
            t["found"] += len(pairs)
            t["missed"] += len(missed)
            t["extra"] += len(extra)
            for true, said in pairs:
                if true["tile_name"] == said["tile_name"]:
                    t["named_right"] += 1
                else:
                    t["mixups"][(true["tile_name"], said["tile_name"])] += 1
            t["missed_names"].update(b["tile_name"] for b in missed)
    return dict(tallies)


def exam_set(exam_dir):
    """(image path, label path, names) for every labelled image of a YOLO-format export (any split folders:
    Roboflow's train/valid/test, or images/ and labels/)."""
    import yaml
    exam_dir = Path(exam_dir)
    names = yaml.safe_load((exam_dir / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    out = []
    for image in sorted(p for p in exam_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        if image.parent.name == "images" and label.exists():
            out.append((image, label, names))
    return out


def dedupe(boxes, min_iou=0.8):
    """Labels without duplicates (a second box of the same name on the same tile, as a slip in Roboflow
    leaves): (kept, dropped)."""
    kept, dropped = [], []
    for box in boxes:
        twin = any(k["tile_name"] == box["tile_name"] and iou(k["xyxy"], box["xyxy"]) >= min_iou for k in kept)
        (dropped if twin else kept).append(box)
    return kept, dropped


def evaluate(exam_dir, predict, exam=None, by=None, confident=config.DETECTION_CONFIDENCE, ignore=()):
    """Score `predict` (BGR image -> boxes, as perception.run_detector) on the exam frames under exam_dir.
    by: None, "lighting" or "session"; ignore: tile names left out of the score. Returns (score(...),
    frames left out because they aren't exam frames, [(frame, duplicate label)] merged away)."""
    import cv2
    exam = exam or load_exam()
    images, outside, duplicates = [], [], []
    for image_path, label_path, names in exam_set(exam_dir):
        session = session_of(image_path)
        if session not in exam["sessions"]:
            outside.append(image_path)
            continue
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        group = {"lighting": lambda: lighting(exam["sessions"][session].get("conditions")),
                 "session": lambda: session}.get(by, lambda: "all")()
        truth, dropped = dedupe(read_yolo_labels(label_path, names, (w, h)))
        duplicates += [(image_path, b) for b in dropped]
        images.append({"truth": truth, "group": group,
                       "said": [b for b in predict(image) if b["confidence"] >= confident]})
    return score(images, ignore), outside, duplicates


def summary(tally):
    """A tally as plain numbers and rates (for a scores file)."""
    tiles, found = tally["tiles"], tally["found"]
    return {"frames": tally["frames"], "tiles": tiles, "found": found, "named_right": tally["named_right"],
            "missed": tally["missed"], "extra": tally["extra"],
            "found_rate": found / tiles if tiles else None,
            "named_right_rate": tally["named_right"] / found if found else None,
            "right_rate": tally["named_right"] / tiles if tiles else None,     # found and named right
            "mixups": [[t, s, n] for (t, s), n in tally["mixups"].most_common()],
            "missed_names": dict(tally["missed_names"].most_common())}
