"""Test 1 of docs/face_down_plan.md: tiles held up by hand to the LeKiwi base camera, and how well the two readers we
already have name them. No arm. A person holds a tile up wherever the arm can reach (close, and at an angle: the camera's view of the arm's reach decides, so
there is no fixed distance), right way up and turned 180 degrees, several holds each, and the
frames are saved under the tile's name; scoring then asks the classifier (name_check) and the detector what they see.

Both readers were trained on other views (the wrist camera at the hover; the RealSense looking down at the mat), so a low
score here says "these views are new", not that the idea fails. What stops the plan is *nothing reads anywhere the arm can hold a tile*
(`readable_anywhere`). A held tile's position in the frame isn't known the way a hovered tile's is, so the classifier
gets three framings and each is scored apart: the wrist camera's fixed crop, a square around the detector's best box
(only where the detector boxes something), and the whole frame. Each photo is also read upside down, since a face-down tile
can be held either way up; `Both ways` takes the more confident of the two.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from . import name_check
from .session import read_events

KINDS = ("upright", "turned")            # a tile held right way up, and turned 180 degrees
DEFAULT_TILES = ("5C", "3C", "2C", "9C", "7C", "1B", "EW")      # the four the plan names, then a spread of suits
DEFAULT_HOLDS = 4                        # holds per tile and way up, each in a different place and angle
FRAMINGS = ("wrist crop", "detector box", "whole frame")
BOX_PAD = 1.3                            # the square around a detector box is this much wider than its longer side
LOOK_CONFIDENCE = 0.1                    # the detector's floor for "boxed something" (the live cut is 0.5)


def plan(tiles, holds, kinds=KINDS):
    """The shots in the order a person does them: a tile at a time, each way up, `holds` holds of each. Where the tile
    is held is the person's choice (what the arm can reach); it isn't recorded."""
    return [{"tile": t, "kind": k, "hold": h + 1} for t in tiles for k in kinds for h in range(holds)]


def shot_name(shot, n):
    return f"{shot['kind']}_h{shot['hold']}_{n}.jpg"


def save_shot(folder, shot, frames):
    """Writes `frames` under photos/<TILE>/ in `folder`; returns the paths relative to it."""
    out = Path(folder) / "photos" / shot["tile"]
    out.mkdir(parents=True, exist_ok=True)
    files = []
    start = len(list(out.glob(f"{shot['kind']}_h{shot['hold']}_*.jpg")))
    for i, frame in enumerate(frames):
        path = out / shot_name(shot, start + i)
        cv2.imwrite(str(path), frame)
        files.append(str(path.relative_to(folder)))
    return files


def rows(session_dir):
    """One row per saved photo: truth, kind, hold, conditions, path."""
    session_dir = Path(session_dir)
    events = read_events(session_dir)
    conditions = next((e["conditions"] for e in events if e["type"] == "holdup_conditions"), {})
    out = []
    for e in events:
        if e["type"] != "holdup_shot":
            continue
        for f in e["files"]:
            out.append({"truth": e["tile"], "kind": e["kind"], "hold": e["hold"], "path": session_dir / f,
                        "daylight": conditions.get("daylight"), "shot": e["shot"]})
    return out


# --- scoring ---

def square_around(image, xyxy, pad=BOX_PAD, size=name_check.INPUT_PX):
    """The square of `image` centred on a box, `pad` times its longer side, clamped inside the frame, resized."""
    h, w = image.shape[:2]
    cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
    side = int(min(max(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]) * pad, h, w))
    x0 = int(min(max(cx - side / 2, 0), w - side))
    y0 = int(min(max(cy - side / 2, 0), h - side))
    return cv2.resize(image[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)


def whole_frame(image, size=name_check.INPUT_PX):
    h, w = image.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return cv2.resize(image[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)


def framings(image, boxes):
    """{framing: image or None} for one frame. `detector box` is None when the detector found nothing."""
    best = max(boxes, key=lambda b: b["confidence"], default=None)
    return {"wrist crop": name_check.crop(image),
            "detector box": square_around(image, best["xyxy"]) if best else None,
            "whole frame": whole_frame(image)}


def read_frame(image, boxes, reader):
    """For each framing, what the classifier says reading it as it is, reading it upside down, and the more confident of
    the two: {framing: {"as_is": read, "turned": read, "both": read}} (None for a framing with no picture)."""
    out = {}
    for name, crop in framings(image, boxes).items():
        if crop is None:
            out[name] = None
            continue
        as_is, turned = reader.read_many([crop, cv2.rotate(crop, cv2.ROTATE_180)])
        out[name] = {"as_is": as_is, "turned": turned, "both": max((as_is, turned), key=lambda r: r["confidence"])}
    return out


def score_photo(row, image, predict, reader):
    """Adds to a row what the detector and the classifier said about its photo (the guess, kept next to the photo)."""
    boxes = predict(image)
    best = max(boxes, key=lambda b: b["confidence"], default=None)
    row["boxes"] = len(boxes)
    row["detector"] = best["tile_name"] if best else None
    row["detector_confidence"] = best["confidence"] if best else 0.0
    row["reads"] = read_frame(image, boxes, reader)
    return row


def _rate(n, of):
    return n / of if of else 0.0


def score(rows):
    """Rates per slice from scored rows. `readable_anywhere` is the plan's stop: some reader named some tile at some
    place. Photos of one hold aren't independent, so counts say how many holds (`shots`) stand behind them."""
    def right(r, framing, way):
        read = (r["reads"].get(framing) or {}).get(way)
        return bool(read) and read["name"] == r["truth"]

    def slice_of(items):
        n = len(items)
        out = {"photos": n, "shots": len({(r["truth"], r["kind"], r["shot"]) for r in items}),
               "boxed": _rate(sum(r["boxes"] > 0 for r in items), n),
               "detector_right": _rate(sum(r["detector"] == r["truth"] for r in items), n),
               "detector_right_at_0.5": _rate(sum(r["detector"] == r["truth"] and r["detector_confidence"] >= 0.5
                                                  for r in items), n)}
        for framing in FRAMINGS:
            for way in ("as_is", "both"):
                out[f"{framing}/{way}"] = _rate(sum(right(r, framing, way) for r in items), n)
        return out

    by = {"all": slice_of(rows)}
    for key in ("kind", "daylight"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for value, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
            by[f"{key}={value}"] = slice_of(items)
    per_tile = defaultdict(list)
    for r in rows:
        per_tile[r["truth"]].append(r)
    wrong = Counter()
    for r in rows:
        for framing in FRAMINGS:
            read = (r["reads"].get(framing) or {}).get("both")
            if read and read["name"] != r["truth"]:
                wrong[f"{r['truth']}->{read['name']}"] += 1
    return {"by": by, "per_tile": {t: slice_of(v) for t, v in sorted(per_tile.items())},
            "readable_anywhere": any(r["detector"] == r["truth"] or any(right(r, f, w) for f in FRAMINGS for w in ("as_is", "turned"))
                                     for r in rows),
            "misreads": wrong.most_common()}


def contact_sheet(rows, path, per_row=6, thumb=160):
    """A photo sheet of every scored photo captioned truth / the classifier's guess (whole-frame, both ways) / the detector's,
    for a person to check the exam in aggregate. Wrong reads are drawn in red."""
    import numpy as np
    cells = []
    for r in rows:
        image = cv2.imread(str(r["path"]))
        if image is None:
            continue
        cell = cv2.resize(whole_frame(image), (thumb, thumb))
        read = (r["reads"].get("whole frame") or {}).get("both")
        said = read["name"] if read else "-"
        ok = said == r["truth"]
        cv2.rectangle(cell, (0, 0), (thumb, 34), (0, 0, 0), -1)
        cv2.putText(cell, f"{r['truth']} {r['kind']} h{r['hold']}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        cv2.putText(cell, f"c:{said} d:{r['detector'] or '-'}", (3, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 200, 0) if ok else (0, 0, 255), 1)
        cells.append(cell)
    if not cells:
        return None
    while len(cells) % per_row:
        cells.append(np.zeros_like(cells[0]))
    sheet = np.vstack([np.hstack(cells[i:i + per_row]) for i in range(0, len(cells), per_row)])
    cv2.imwrite(str(path), sheet)
    return path


def summary_lines(s):
    """The report's text: one line per slice, framings as percentages, then the stop check and the misreads."""
    lines = []
    head = ["detector right", "wrist crop", "det box", "whole frame"]
    lines.append(f"{'slice':<22}{'photos':>7}  boxed  " + "  ".join(f"{h:>14}" for h in head))
    for name, v in s["by"].items():
        lines.append(f"{name:<22}{v['photos']:>7}  {v['boxed']:>4.0%}   "
                     f"{v['detector_right']:>13.0%}  {v['wrist crop/both']:>13.0%}  {v['detector box/both']:>13.0%}  "
                     f"{v['whole frame/both']:>13.0%}")
    lines.append("(classifier columns read each photo both ways up, the more confident read; the detector's is its best "
                 "box at any confidence)")
    lines.append("per tile (best framing, both ways): " + ", ".join(
        f"{t} {max(v[f'{f}/both'] for f in FRAMINGS):.0%}" for t, v in s["per_tile"].items()))
    lines.append("misreads: " + (", ".join(f"{k} x{n}" for k, n in s["misreads"][:15]) or "none"))
    lines.append("some reader named some tile anywhere: " + ("yes" if s["readable_anywhere"] else
                 "NO. The plan's stop: the base camera view needs its own reader, or another way."))
    return lines


def save_scores(session_dir, s):
    path = Path(session_dir) / "holdup_scores.json"
    path.write_text(json.dumps(s, indent=2, default=str) + "\n")
    return path
