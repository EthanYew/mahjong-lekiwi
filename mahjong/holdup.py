"""Test 1 of docs/face_down_plan.md: tiles held up by hand to the LeKiwi base camera, and how well the two readers we
already have name them. No arm. A person holds a tile up wherever the arm can reach (close, and at an angle: what the arm
can reach decides, so there is no fixed distance), right way up and turned 180 degrees, several holds each, and the
frames are saved under the tile's name; scoring then asks the classifier (name_check) and the detector what they see.

Both readers were trained on other views (the wrist camera at the hover; the RealSense looking down at the mat), so a low
score here says "these views are new", not that the idea fails. What stops the plan is *nothing reads anywhere the arm can
hold a tile* (`readable_anywhere`). A held tile's position in the frame isn't known the way a hovered tile's is, so the
classifier gets three framings and each is scored apart: the wrist camera's fixed crop, a square around the detector's best
box (only where the detector boxes something), and the whole frame. Each photo is read as it is and turned upside down, since
a face-down tile can be held either way up; the two reads are reported apart, never merged (the more confident of the two was
the wrong one more often than not on the first session, so confidence told nothing there).

A reader only counts as reading if it is right on at least two different tiles (`reads_on_two_tiles`): one that answers "3C"
to everything is right on every 3C photo and nothing else, and scored 24% on the first session while reading nothing.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from . import config, name_check, privacy
from .session import ask_conditions, previous_conditions, read_events

KINDS = ("upright", "turned")            # a tile held right way up, and turned 180 degrees
DEFAULT_TILES = ("5C", "3C", "2C", "9C", "7C", "1B", "EW")      # the four the plan names, then a spread of suits
DEFAULT_HOLDS = 4                        # holds per tile and way up, each in a different place and angle
FRAMINGS = ("wrist crop", "detector box", "whole frame")
WAYS = ("as_is", "turned")
BOX_PAD = 1.3                            # the square around a detector box is this much wider than its longer side
MIN_TILE_SHARE = 1 / 3                   # right on at least this share of a tile's photos for that tile to count as read


LIGHT_KEYS = ("daylight", "lamp", "ceiling_lights")


def ask_lighting(ask=input):
    """The lighting setup questions only, answered before a hold-up session (saved as its `holdup_conditions`)."""
    questions = [q for q in config.SETUP_QUESTIONS if q["key"] in LIGHT_KEYS]
    return ask_conditions(previous_conditions(), questions=questions, ask=ask)


def plan(tiles, holds, kinds=KINDS):
    """The shots in the order a person does them: a tile at a time, each way up, `holds` holds of each. Where the tile
    is held is the person's choice (what the arm can reach); it isn't recorded."""
    return [{"tile": t, "kind": k, "hold": h + 1} for t in tiles for k in kinds for h in range(holds)]


def shot_name(shot, n):
    return f"{shot['kind']}_h{shot['hold']}_{n}.jpg"


def save_shot(folder, shot, frames):
    """Writes `frames` under photos/<TILE>/ in `folder`, with any face blurred first (privacy.py: the base camera sees the
    room); returns the paths relative to it."""
    out = Path(folder) / "photos" / shot["tile"]
    out.mkdir(parents=True, exist_ok=True)
    files = []
    start = len(list(out.glob(f"{shot['kind']}_h{shot['hold']}_*.jpg")))
    for i, frame in enumerate(frames):
        path = out / shot_name(shot, start + i)
        cv2.imwrite(str(path), privacy.blur_faces(frame)[0])
        files.append(str(path.relative_to(folder)))
    return files


def is_sim(session_dir):
    start = next((e for e in read_events(Path(session_dir)) if e["type"] == "session_start"), {})
    return bool(start.get("sim"))


def rows(session_dir):
    """One row per saved photo (a shot redone with `b` has no files left): truth, kind, hold, daylight, path, session, pose.
    The first two sessions held tiles at fixed distances and have no holds: their `hold` is None. `pose` is the saved camera
    pose the arm held the tile at (holdup_arm.py); None for a tile held by hand."""
    session_dir = Path(session_dir)
    events = read_events(session_dir)
    conditions = next((e["conditions"] for e in events if e["type"] == "holdup_conditions"), {})
    discarded = {e["shot"] for e in events if e["type"] == "holdup_shot_discarded"}     # a retake reuses the file names
    out = []
    for e in events:
        if e["type"] != "holdup_shot" or e["shot"] in discarded:
            continue
        poses = e.get("frame_poses") or [None] * len(e["files"])        # by the arm: which saved pose each frame is from
        for f, pose in zip(e["files"], poses):
            if (session_dir / f).exists():
                out.append({"truth": e["tile"], "kind": e["kind"], "hold": e.get("hold"), "path": session_dir / f,
                            "daylight": conditions.get("daylight"), "shot": e["shot"], "session": session_dir.name,
                            "pose": pose})
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
    """The centre square of the frame (the full height), resized."""
    h, w = image.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return cv2.resize(image[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)


# --- where the held tile is (a box per saved pose) ---
# The arm holds a tile at a saved pose, so it shows up in about the same part of the base camera's frame each time: not exactly
# (each pose is nudged a couple of degrees, and the jaws close at a different place on the tile each pickup), so a pose's
# `tile_box` in data/camera_poses.json is where the tile CAN be, drawn generously by a person (holdup_check.py --mark-boxes).
# Two uses: framing the classifier on the tile instead of the room, and the control that greys the tile out to see whether
# the rest of the frame gives the tile away (the first arm sessions did, 2026-11-06: 99% with the tile blacked out).
BLANK = 128                              # the grey a tile box is filled with for the control


def tile_boxes(poses=None):
    """{pose name: [x0, y0, x1, y1]} for every saved camera pose with a tile box (base camera pixels)."""
    from . import camera_poses
    poses = camera_poses.load() if poses is None else poses
    return {name: [int(v) for v in p["tile_box"]] for name, p in poses.items() if p.get("tile_box")}


def tile_frame(image, box, size=name_check.INPUT_PX):
    """The square of `image` around a pose's tile box (its longer side), resized: the tile and as little room as possible."""
    return square_around(image, box, pad=1.0, size=size)


def without_tile(image, box):
    """A copy of `image` with the tile box filled grey: what's left is everything the photo shows except the tile."""
    out = image.copy()
    x0, y0, x1, y1 = (int(v) for v in box)
    out[max(y0, 0):y1, max(x0, 0):x1] = BLANK
    return out


def as_input(image, size=name_check.INPUT_PX):
    """An already-framed square, at the classifier's input size."""
    return image if image.shape[:2] == (size, size) else cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def framings(image, boxes):
    """{framing: image or None} for one frame. `detector box` is None when the detector found nothing."""
    best = max(boxes, key=lambda b: b["confidence"], default=None)
    return {"wrist crop": name_check.crop(image),
            "detector box": square_around(image, best["xyxy"]) if best else None,
            "whole frame": whole_frame(image)}


def read_frame(image, boxes, reader):
    """For each framing, what the classifier says reading it as it is and reading it upside down:
    {framing: {"as_is": read, "turned": read}} (None for a framing with no picture)."""
    out = {}
    for name, crop in framings(image, boxes).items():
        if crop is None:
            out[name] = None
            continue
        as_is, turned = reader.read_many([crop, cv2.rotate(crop, cv2.ROTATE_180)])
        out[name] = {"as_is": as_is, "turned": turned}
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


def _said(r, framing, way):
    """What a reader said about a scored row: the classifier on a framing read one way up, or the detector (framing None)."""
    if framing is None:
        return r["detector"]
    read = (r["reads"].get(framing) or {}).get(way)
    return read["name"] if read else None


def reads_on_two_tiles(rows, framing, way, min_tiles=2):
    """A reader (`framing` and `way`, or the detector with framing None) reads only if it is right on at least `min_tiles`
    tiles, each on MIN_TILE_SHARE or more of that tile's photos. Returns {"reads", "tiles_read", "answers", "top_answer",
    "top_share"}: the last three show a reader that answers the same name to everything. A session with one tile can't
    tell a reader from a constant answer, and only needs that one tile."""
    per_tile = defaultdict(list)
    for r in rows:
        per_tile[r["truth"]].append(_said(r, framing, way) == r["truth"])
    tiles_read = sorted(t for t, hits in per_tile.items() if sum(hits) / len(hits) >= MIN_TILE_SHARE)
    answers = Counter(_said(r, framing, way) for r in rows)
    top, top_n = answers.most_common(1)[0] if answers else (None, 0)
    return {"reads": len(tiles_read) >= min(min_tiles, len(per_tile)) and bool(tiles_read), "tiles_read": tiles_read,
            "answers": len(answers), "top_answer": top, "top_share": _rate(top_n, len(rows))}


def score(rows):
    """Rates per slice from scored rows. `readable_anywhere` is the plan's stop: some reader is right on two tiles
    (`reads_on_two_tiles`). Photos of one hold aren't independent, so `shots` says how many holds stand behind them."""
    def slice_of(items):
        n = len(items)
        out = {"photos": n, "shots": len({(r["truth"], r["kind"], r["shot"]) for r in items}),
               "boxed": _rate(sum(r["boxes"] > 0 for r in items), n),
               "detector_right": _rate(sum(r["detector"] == r["truth"] for r in items), n),
               "detector_right_at_0.5": _rate(sum(r["detector"] == r["truth"] and r["detector_confidence"] >= 0.5
                                                  for r in items), n)}
        for framing in FRAMINGS:
            for way in WAYS:
                out[f"{framing}/{way}"] = _rate(sum(_said(r, framing, way) == r["truth"] for r in items), n)
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
    readers = {"detector": reads_on_two_tiles(rows, None, None)}
    for framing in FRAMINGS:
        for way in WAYS:
            readers[f"{framing}/{way}"] = reads_on_two_tiles(rows, framing, way)
    misreads = Counter(f"{r['truth']}->{_said(r, 'whole frame', 'as_is')}" for r in rows
                       if _said(r, "whole frame", "as_is") != r["truth"])
    return {"by": by, "per_tile": {t: slice_of(v) for t, v in sorted(per_tile.items())}, "readers": readers,
            "readable_anywhere": any(v["reads"] for v in readers.values()), "misreads": misreads.most_common()}


def contact_sheet(rows, path, per_row=6, thumb=160):
    """A photo sheet of every scored photo captioned truth / the classifier's whole-frame guess as it is and turned / the
    detector's, for a person to check the exam in aggregate. Green where the as-is guess is right, red where not."""
    import numpy as np
    cells = []
    for r in rows:
        image = cv2.imread(str(r["path"]))
        if image is None:
            continue
        cell = cv2.resize(whole_frame(image), (thumb, thumb))
        a, t = _said(r, "whole frame", "as_is"), _said(r, "whole frame", "turned")
        cv2.rectangle(cell, (0, 0), (thumb, 34), (0, 0, 0), -1)
        cv2.putText(cell, f"{r['truth']} {r['kind']} h{r['hold']}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        cv2.putText(cell, f"c:{a}/{t} d:{r['detector'] or '-'}", (3, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 200, 0) if a == r["truth"] else (0, 0, 255), 1)
        cells.append(cell)
    if not cells:
        return None
    while len(cells) % per_row:
        cells.append(np.zeros_like(cells[0]))
    sheet = np.vstack([np.hstack(cells[i:i + per_row]) for i in range(0, len(cells), per_row)])
    cv2.imwrite(str(path), sheet)
    return path


def summary_lines(s):
    """The report's text: a line per slice (each framing as 'as it is / turned'), the readers' answers, the stop check."""
    pair = lambda v, f: f"{v[f'{f}/as_is']:.0%}/{v[f'{f}/turned']:.0%}"
    lines = [f"{'slice':<18}{'photos':>7}  boxed  detector  {'wrist crop':>11}  {'detector box':>12}  {'whole frame':>11}"]
    for name, v in s["by"].items():
        lines.append(f"{name:<18}{v['photos']:>7}  {v['boxed']:>4.0%}  {v['detector_right']:>8.0%}  "
                     f"{pair(v, 'wrist crop'):>11}  {pair(v, 'detector box'):>12}  {pair(v, 'whole frame'):>11}")
    lines.append("(classifier columns: right when read as it is / right when read upside down. The detector is its best box)")
    lines.append("per tile, whole frame as it is / turned: " + ", ".join(f"{t} {pair(v, 'whole frame')}"
                                                                       for t, v in s["per_tile"].items()))
    whole = s["readers"]["whole frame/as_is"]
    lines.append(f"whole frame, as it is: {whole['answers']} different answers; most common {whole['top_answer']} "
                 f"({whole['top_share']:.0%} of photos). One answer for everything is a reader that sees nothing")
    lines.append("misreads (whole frame, as it is): " + (", ".join(f"{k} x{n}" for k, n in s["misreads"][:12]) or "none"))
    reading = [k for k, v in s["readers"].items() if v["reads"]]
    lines.append("readers right on two or more tiles: " + (", ".join(reading) if reading else
                 "NONE. The plan's stop: this view needs its own reader (see holdup_check.py), or another way."))
    return lines


def save_scores(session_dir, s):
    path = Path(session_dir) / "holdup_scores.json"
    path.write_text(json.dumps(s, indent=2, default=str) + "\n")
    return path


# --- is training a reader for this view paying off? every holdup_check.py result (scripts/holdup_score.py) ---

CHECK_FILES = ("holdup_check.json", "holdup_check_held_out.json")


def check_results(folders):
    """Every holdup_check.py result saved in these session folders, oldest first. A result written before the check recorded
    its time is dated by its file."""
    from datetime import datetime
    out = []
    for folder in folders:
        for name in CHECK_FILES:
            path = Path(folder) / name
            if path.exists():
                d = json.loads(path.read_text())
                d.setdefault("written", datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes"))
                d["file"] = f"{Path(folder).name}/{name}"
                out.append(d)
    return sorted(out, key=lambda d: d["written"])


def verdict_word(verdict):
    head = verdict.split(":")[0]
    return head if head in ("PASS", "FAIL", "IN BETWEEN", "BACKGROUND", "UNCONTROLLED") else "TOO FEW TILES"


def check_lines(results):
    """The report: one block per check (what it trained and tested on, the reader against the tile-greyed control and the best
    constant answer), then what the results say together. Training pays off when the reader is well above the constant answer,
    the control is near it, and that holds on a session never trained on."""
    if not results:
        return ["No holdup_check.py results yet: python scripts/holdup_check.py (it trains a reader for this view and scores it)"]
    lines = [f"{'written':<17} {'verdict':<14} {'reader':>6} {'control':>8} {'constant':>9}  test",
             "(reader: the fresh classifier right on photos it never trained on; control: the same with the tile greyed out, "
             "which should be near constant, the best a reader that ignores the photo can do)"]
    for d in results:
        first = d["runs"][0]
        control = "-" if d.get("control_right") is None else f"{d['control_right']:.0%}"
        held_out = "held out" in d["split"] and "holds" not in d["split"]
        tiles = ", ".join(f"{t} {v:.0%}" for t, v in first["per_tile"].items())
        word = verdict_word(d["verdict"])
        if word == "PASS" and d.get("control_right") is None:       # before the control existed: nothing checked the tile was read
            word = "UNCONTROLLED"
        lines.append(f"{d['written'].replace('T', ' '):<17} {word:<14} {d['mean_right']:>6.0%} "
                     f"{control:>8} {first['best_constant_answer']:>9.0%}  {first['tiles']} tile(s), "
                     f"{'another session' if held_out else 'same session (optimistic)'}")
        sizes = (f"train {d['train_photos']} photos / {d['train_holds']} holds, test {d['test_photos']} / {d['test_holds']}"
                 if "train_photos" in d else f"test {first['photos']} photos")
        lines.append(f"    {d['split']}; {sizes}")
        lines.append(f"    per tile: {tiles}; confusions: "
                     + (", ".join(f"{k} x{n}" for k, n in first["confusions"]) or "none"))
        if d.get("caveat") and not d["caveat"].startswith("OPTIMISTIC"):
            lines.append(f"    {d['caveat']}")
    controlled = [d for d in results if d.get("control_right") is not None]
    clean = [d for d in controlled if d["control_right"] < d["runs"][0]["best_constant_answer"] + 0.15]
    across = [d for d in clean if "held out" in d["split"] and "holds" not in d["split"]]
    lines.append("")
    if not clean:
        lines.append("So far: no result with a clean control, so no score yet says the reader reads the tile")
    else:
        best = max(clean, key=lambda d: d["mean_right"] - d["runs"][0]["best_constant_answer"])
        lines.append(f"So far: {len(clean)} of {len(results)} result(s) with a clean control; the best {best['mean_right']:.0%} "
                     f"against a constant {best['runs'][0]['best_constant_answer']:.0%} on {best['runs'][0]['tiles']} tile(s) "
                     f"({best['file']})")
        lines.append(f"  on a session never trained on: " + ("; ".join(
            f"{d['mean_right']:.0%} on {', '.join(d['runs'][0]['per_tile'])} ({d['file'].split('/')[0]})" for d in across)
            if across else "none yet (holdup_check.py --test-session), the test that says it carries to new light or rooms"))
    return lines


# --- can the base camera's view be learned? (scripts/holdup_check.py) ---

def split(rows, by="auto", test_holds=2, test_session=None):
    """(train rows, test rows, how it was split, caveat). `auto` holds out the newest session if there are two or more with
    the same tiles, else the last `test_holds` holds of each tile and way up. The second is optimistic: the same session's
    hands, light and background are in both halves. Whole shots stay together (a shot's frames are near copies).
    `test_session` (a folder name) holds out that session whatever `by` says, trains on every other, and tests only the tiles
    some other session has (a tile never trained on can't be named): e.g. train on an evening, test on a day."""
    sessions = sorted({r["session"] for r in rows})
    if test_session is not None:
        if test_session not in sessions:
            raise ValueError(f"{test_session} isn't among the sessions read: {', '.join(sessions)}")
        if len(sessions) < 2:
            raise ValueError("holding out a session needs another to train on")
        train = [r for r in rows if r["session"] != test_session]
        trained = {r["truth"] for r in train}
        held = [r for r in rows if r["session"] == test_session]
        test = [r for r in held if r["truth"] in trained]
        if not test:
            raise ValueError(f"no tile in {test_session} is in another session, so there is nothing to test")
        left = sorted({r["truth"] for r in held} - trained)
        caveat = (f"left out {len(held) - len(test)} photo(s) of {', '.join(left)}: in no other session, never trained on"
                  if left else None)
        return train, test, f"held out {test_session}, trained on {', '.join(s for s in sessions if s != test_session)}", caveat
    if by == "auto":
        newest = {r["truth"] for r in rows if r["session"] == sessions[-1]}
        older = {r["truth"] for r in rows if r["session"] != sessions[-1]}
        by = "session" if len(sessions) > 1 and newest <= older else "hold"     # every tile tested must have been trained on
    if by == "session":
        if len(sessions) < 2:
            raise ValueError("a split by session needs two sessions")
        test = [r for r in rows if r["session"] == sessions[-1]]
        train = [r for r in rows if r["session"] != sessions[-1]]
        return train, test, f"held out the newest session, {sessions[-1]}", None
    top = defaultdict(int)
    for r in rows:
        key = (r["session"], r["truth"], r["kind"])
        top[key] = max(top[key], r["hold"])
    held = [r["hold"] > top[(r["session"], r["truth"], r["kind"])] - test_holds for r in rows]
    test = [r for r, h in zip(rows, held) if h]
    train = [r for r, h in zip(rows, held) if not h]
    return train, test, f"held out the last {test_holds} holds of every tile and way up", \
        "OPTIMISTIC: the same session's hands, light and background are in train and test. Treat a pass as 'worth a real capture'."


def check_summary(test_rows, said):
    """Right rates for a model's answers on held-out rows: overall, per tile, per way up, against chance and the best
    constant answer (what a model that ignored the photo could get)."""
    n = len(test_rows)
    right = [s == r["truth"] for r, s in zip(test_rows, said)]
    tiles = sorted({r["truth"] for r in test_rows})
    constant = max(Counter(r["truth"] for r in test_rows).values()) / n if n else 0.0
    return {"photos": n, "tiles": len(tiles), "right": _rate(sum(right), n), "chance": 1 / len(tiles) if tiles else 0.0,
            "best_constant_answer": constant,
            "per_tile": {t: _rate(sum(ok for ok, r in zip(right, test_rows) if r["truth"] == t),
                                  sum(r["truth"] == t for r in test_rows)) for t in tiles},
            "per_kind": {k: _rate(sum(ok for ok, r in zip(right, test_rows) if r["kind"] == k),
                                  sum(r["kind"] == k for r in test_rows)) for k in KINDS},
            "confusions": Counter(f"{r['truth']}->{s}" for r, s in zip(test_rows, said) if s != r["truth"]).most_common(10)}
