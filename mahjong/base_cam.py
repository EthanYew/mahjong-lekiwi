"""The base camera's tile reader: what the hold-up photos are, how a held tile is framed and read, and how a reader trained on
them is scored. A face-down tile drawn from the wall is held up to the LeKiwi base's front camera at a taught pose
(data/camera_poses.json, `base_view_*`), and this reader names it from those photos (docs/face_down_plan.md).

Not the tile detector (the RealSense looking down at face-up tiles) nor the name check (the wrist camera at the hover): both
read ~0% on this view (test 1, 2026-11-05/06), so the base camera has a reader of its own, `assets/holdup_reader.pt`.

The photos come from scripts/base_cam_collect.py (the arm holds each tile up; sessions/<time>_holdup/photos/<NAME>/ with
`holdup_shot` events), and scripts/base_cam_train.py trains and scores readers on them. Each photo is cropped to its pose's tile
box (`tile_frame`), and a control greys the tile out (`without_tile`) to catch a reader that names the tile from the scene.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from . import config, name_check, privacy
from .session import ask_conditions, is_sim, previous_conditions, read_events  # noqa: F401 (is_sim: callers use base_cam.is_sim)

KINDS = ("upright", "turned")            # a tile held right way up, and turned 180 degrees
BOX_PAD = 1.3                            # the square around a detector box is this much wider than its longer side
LIGHT_KEYS = ("daylight", "lamp", "ceiling_lights")


def ask_lighting(ask=input):
    """The lighting setup questions only, answered before a hold-up session (saved as its `holdup_conditions`)."""
    questions = [q for q in config.SETUP_QUESTIONS if q["key"] in LIGHT_KEYS]
    return ask_conditions(previous_conditions(), questions=questions, ask=ask)


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


REVIEW = config.DATA / "holdup_review.jsonl"     # a person's verdicts on the photos (scripts/base_cam_review.py)


def _shots(session_dir):
    """Every photo of a session as filed, before review: one row per file (see `rows`)."""
    session_dir = Path(session_dir)
    events = read_events(session_dir)
    conditions = next((e["conditions"] for e in events if e["type"] == "holdup_conditions"), {})
    discarded = {e["shot"] for e in events if e["type"] == "holdup_shot_discarded"}     # a retake reuses the file names
    out, arrangement = [], None
    for e in events:
        if e["type"] == "holdup_arrangement":
            arrangement = e.get("n")
        if e["type"] != "holdup_shot" or e["shot"] in discarded:
            continue
        poses = e.get("frame_poses") or [None] * len(e["files"])        # by the arm: which saved pose each frame is from
        first_at = {}                       # a pose's first frame stands for its others in a review, as a wrist view's f0 does
        for f, pose in zip(e["files"], poses):
            key = f"{session_dir.name}/{f}"
            if pose is not None:
                key = first_at.setdefault(pose, key)
            if (session_dir / f).exists():
                out.append({"truth": e["tile"], "kind": e["kind"], "hold": e.get("hold"), "path": session_dir / f,
                            "daylight": conditions.get("daylight"), "shot": e["shot"], "session": session_dir.name,
                            "pose": pose, "spot": e.get("spot"), "arrangement": arrangement, "key": key})
    return out


def rows(session_dir, review=None):
    """One row per saved photo (a shot redone with `b` has no files left), as a person reviewed it (data/holdup_review.jsonl,
    scripts/base_cam_review.py): a photo marked wrong is left out, a renamed one gets its right name in `truth` (`filed_as` is
    what it was saved under). Fields: truth, kind, hold, daylight, path, session, shot, pose, spot, arrangement, key (the
    review's: the first frame at the photo's pose in its shot), reviewed. The first two sessions held tiles at fixed
    distances and have no holds: their `hold` is None. `pose` is the saved camera pose the arm held the tile at
    (base_cam_collect.py); None for a tile held by hand."""
    from . import photo_review
    review = load_review() if review is None else review
    out = []
    for row in _shots(session_dir):
        name = photo_review.name_of({"key": row["key"], "name": row["truth"]}, review)
        if name is not None:
            out.append({**row, "truth": name, "filed_as": row["truth"], "reviewed": row["key"] in review})
    return out


def load_review(path=None):
    from . import photo_review
    return photo_review.load_review(path or REVIEW)


def review_photos(folders):
    """The photos a person reviews, before review: one per pose of each hold (its first frame; the pose's other frames share
    its verdict), as mahjong.photo_review wants them: key, path, name (as filed), session, group, frames, and the shot's
    kind, hold, pose, spot and arrangement. A group is one tile laid on one spot in one arrangement (every hold of it there),
    what a person gets wrong together by laying the wrong tile; a session from before arrangements groups by shot."""
    out = {}
    for folder in folders:
        for row in _shots(folder):
            photo = out.get(row["key"])
            if photo is None:
                where = (f"a{row['arrangement']}/{row['spot']}" if row["arrangement"] is not None
                         else f"shot{row['shot']}")
                photo = out[row["key"]] = {
                    "key": row["key"], "path": row["path"], "name": row["truth"], "session": row["session"],
                    "group": f"{row['session']}/{where}/{row['truth']}", "frames": [],
                    **{k: row[k] for k in ("kind", "hold", "shot", "pose", "spot", "arrangement")}}
            photo["frames"].append(row["path"])
    return list(out.values())


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
# `tile_box` in data/camera_poses.json is where the tile CAN be, drawn generously by a person (base_cam_train.py --mark-boxes).
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


READER = config.TILE_DETECTOR.parent / "holdup_reader.pt"       # trained by base_cam_train.py --save-reader


def load_reader(path=None):
    """The reader for held tiles (photos framed by as_input, so read_held's tile squares go in unchanged), or None if none is trained."""
    path = Path(path or READER)
    return name_check.Reader(path, crop_fn=as_input) if path.exists() else None


def read_held(reader, frames, from_pose, boxes):
    """The reader's name for one held tile from its photos at the saved poses, by summed confidence: {"name", "confidence" (its mean share), "votes"}.
    None without a reader, or with no photo at a pose with a tile box."""
    crops = [tile_frame(f, boxes[p]) for f, p in zip(frames, from_pose) if p in boxes]
    if reader is None or not crops:
        return None
    votes = {}
    for read in reader.read_many(crops):
        votes[read["name"]] = votes.get(read["name"], 0.0) + read["confidence"]
    name = max(votes, key=votes.get)
    return {"name": name, "confidence": round(votes[name] / len(crops), 4), "votes": {k: round(v, 4) for k, v in votes.items()}}


def without_tile(image, box):
    """A copy of `image` with the tile box filled grey: what's left is everything the photo shows except the tile."""
    out = image.copy()
    x0, y0, x1, y1 = (int(v) for v in box)
    out[max(y0, 0):y1, max(x0, 0):x1] = BLANK
    return out


def as_input(image, size=name_check.INPUT_PX):
    """An already-framed square, at the classifier's input size."""
    return image if image.shape[:2] == (size, size) else cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


# --- is training a reader for this view paying off? every base_cam_train.py result (scripts/base_cam_train.py --history) ---

CHECK_FILES = ("holdup_check.json", "holdup_check_held_out.json")


def check_results(folders):
    """Every base_cam_train.py result saved in these session folders, oldest first. A result written before the check recorded
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
        return ["No base_cam_train.py results yet: python scripts/base_cam_train.py (it trains a reader for this view and scores it)"]
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
        lines.append("  on a session never trained on: " + ("; ".join(
            f"{d['mean_right']:.0%} on {', '.join(d['runs'][0]['per_tile'])} ({d['file'].split('/')[0]})" for d in across)
            if across else "none yet (base_cam_train.py --test-session), the test that says it carries to new light or rooms"))
    return lines


# --- can the base camera's view be learned? (scripts/base_cam_train.py) ---

def _rate(n, of):
    return n / of if of else 0.0


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
