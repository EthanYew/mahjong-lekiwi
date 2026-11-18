"""Is a face-down tile still between the jaw tips? A trained look for the wrist camera at the hover: back or empty.

The hand-written look (wrist_check.back_blob: a tile-sized patch of the back's green) found the back in 859 of 1,092
face-down first frames at the hover (79%) over every face-down session to 2026-11-18, and in 4 of 5 looks on
20261118_221723_bench it found no back's shape and asked a person, who said "there" each time: the jaw tips and their
shadow cover part of the back, so its patch falls under the size floor, or the shadow takes it out of the colour range.
A colour rule tuned on one day's light breaks on the next, so this is a small classifier instead, the name check's
(mahjong/name_check.py: YOLO11n-cls, the same crop around where a hovered tile sits), with two classes:

- back: the policy's first frame (`<attempt>/wrist_0000.jpg`, at the hover) of every pickup in a face-down session (base
  camera collection, wall, crude run, face-down play, and a face-down bench's back attempts), and a wrist look's photo
  where a person, or the back's shape, said a back was there. A pickup starts over a tile, so a rare first frame with the
  tile already knocked away is a wrong label; there are too few to matter, and no person has checked each one.
- empty: hovers over bare layout spots (`wrist_capture.py --empty`, saved under `<session>_wrist/wrist_empty/`), and a
  wrist look's photo where a person said nothing was there. The arm otherwise only hovers where a tile is, so before the
  --empty captures there were ~12 of these, too few to train on.

Whole sessions listed in data/back_check_exam.json never train it. wrist_check.check asks it first when its weights are
installed (assets/back_check.pt, the owner's call after reading the exam score), and acts only when every frame agrees
with confidence; anything else falls back to the back's shape, then a person.
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import config, name_check
from .session import read_jsonl

BACK, EMPTY = config.BACK, "empty"
CLASSES = (BACK, EMPTY)
WEIGHTS = config.TILE_DETECTOR.parent / "back_check.pt"
EXAM = config.DATA / "back_check_exam.json"
EVAL = config.DATA / "back_check_eval"
EMPTY_FOLDER = "wrist_empty"            # under a _wrist session: kept apart from wrist/<NAME>/, which the name check reads
# A read acts only at this confidence, on every frame of a look; below it the look falls back to the back's shape.
CONFIDENT = 0.9
# The exam's gate (the plan of 2026-11-18): an empty spot read as a back sends the arm to grab nothing, so none of those;
# and backs read as backs often enough to take most of the questions away.
GATE_FALSE_BACKS, GATE_BACKS_SEEN = 0, 0.97
# 0 false backs means little on a handful of empties: a smoke run passed on 2 that it read unsure. ~100 is one --empty session.
GATE_MIN_EMPTIES = 100
# Sessions whose every pickup is of a face-down tile. A bench is face down by its session_start (its back attempts' folders
# end in _back); other kinds have face-up pickups too.
FACE_DOWN_KINDS = ("_holdup", "_wall", "_crude", "_play_fd")


def _face_down_bench(folder):
    for event in read_jsonl(folder / "events.jsonl") if (folder / "events.jsonl").exists() else ():
        if "face_down" in event:
            return bool(event["face_down"])
    return False


def _photo(path, name, source, folder, sessions_dir):
    return {"key": str(path.relative_to(sessions_dir)), "path": path, "name": name, "source": source,
            "session": folder.name}


def _look_photos(folder, sessions_dir, face_down):
    """A bench's wrist looks with a verdict worth training on: a person's either way, or the back's shape finding one. On a
    face-down bench every look is at a back, whatever a scan named the tile (bench.wrist_look)."""
    out = []
    for event in read_jsonl(folder / "events.jsonl") if (folder / "events.jsonl").exists() else ():
        if event.get("type") != "wrist_check" or not event.get("image") or event.get("present") is None:
            continue
        path = folder / event["image"]
        if not path.exists():
            continue
        at_back = face_down or event.get("tile") == BACK
        if event.get("by") == "person":
            name = BACK if event["present"] and at_back else EMPTY if not event["present"] else None
        else:
            name = BACK if event.get("by") == "back shape" and event["present"] else None
        if name:
            out.append(_photo(path, name, "wrist_check", folder, sessions_dir))
    return out


def photos(sessions_dir=None):
    """Every back and empty photo under sessions_dir, simulated sessions left out: [{"key", "path", "name" (back / empty),
    "source" (pickup / wrist_check / empty_capture), "session"}], in session order."""
    from .detector import is_sim
    sessions_dir = Path(sessions_dir or config.SESSIONS)
    out = []
    for folder in sorted(sessions_dir.iterdir()) if sessions_dir.is_dir() else ():
        if not folder.is_dir() or is_sim(folder):
            continue
        if folder.name.endswith(FACE_DOWN_KINDS):
            out += [_photo(p, BACK, "pickup", folder, sessions_dir) for p in sorted(folder.glob("*/wrist_0000.jpg"))]
        elif folder.name.endswith("_bench"):
            face_down = _face_down_bench(folder)
            if face_down:
                out += [_photo(p, BACK, "pickup", folder, sessions_dir)
                        for p in sorted(folder.glob(f"*_{BACK}/wrist_0000.jpg"))]
            out += _look_photos(folder, sessions_dir, face_down)
        elif folder.name.endswith("_wrist"):
            out += [_photo(p, EMPTY, "empty_capture", folder, sessions_dir)
                    for p in sorted((folder / EMPTY_FOLDER / EMPTY).glob("*.jpg"))]
    return out


def load_exam(path=None):
    path = Path(path or EXAM)
    return json.loads(path.read_text()) if path.exists() else {"sessions": {}}


def split(all_photos, exam=None):
    """(training photos, exam photos): the exam's whole sessions never train it."""
    sessions = (load_exam() if exam is None else exam)["sessions"]
    return [p for p in all_photos if p["session"] not in sessions], [p for p in all_photos if p["session"] in sessions]


def summary(some_photos):
    """"N back, M empty from S sessions (by source ...)"."""
    names, sources = Counter(p["name"] for p in some_photos), Counter(p["source"] for p in some_photos)
    return (f"{names[BACK]} back, {names[EMPTY]} empty from {len({p['session'] for p in some_photos})} sessions ("
            + ", ".join(f"{n} {s}" for s, n in sorted(sources.items())) + ")")


def train(training_photos, seed=0, epochs=name_check.EPOCHS, log=print):
    """A checkpoint for Reader: the name check's training on these two classes (each drawn as often as the other)."""
    missing = [c for c in CLASSES if not any(p["name"] == c for p in training_photos)]
    if missing:
        raise ValueError(f"no {' or '.join(missing)} photos to train on"
                         + (": take some with wrist_capture.py --empty" if EMPTY in missing else ""))
    return name_check.train(training_photos, seed=seed, epochs=epochs, log=log)


save = name_check.save


def decide(reads, confident=CONFIDENT):
    """A look's verdict from its frames' reads ([{"name", "confidence"}]): back / empty when every frame says it at
    `confident` or more, else None (the look falls back to the back's shape, then a person)."""
    if not reads:
        return None
    names = {r["name"] for r in reads}
    return reads[0]["name"] if len(names) == 1 and min(r["confidence"] for r in reads) >= confident else None


def score(reads, labels, confident=CONFIDENT):
    """The exam's numbers from one read per photo and its label: backs seen, false backs (an empty spot read as a back,
    confidently), the rest unsure, and whether the gate passes."""
    out = {name: Counter() for name in CLASSES}
    for read, label in zip(reads, labels):
        verdict = decide([read], confident)
        out[label]["right" if verdict == label else "unsure" if verdict is None else "wrong"] += 1
    backs, empties = sum(out[BACK].values()), sum(out[EMPTY].values())
    backs_seen = out[BACK]["right"] / backs if backs else None
    return {"backs": backs, "backs_seen": backs_seen, "backs_called_empty": out[BACK]["wrong"],
            "empties": empties, "empties_seen": out[EMPTY]["right"] / empties if empties else None,
            "false_backs": out[EMPTY]["wrong"], "confident": confident,
            "passes": bool(backs and empties >= GATE_MIN_EMPTIES and out[EMPTY]["wrong"] <= GATE_FALSE_BACKS
                           and backs_seen >= GATE_BACKS_SEEN)}


def evaluate(weights, exam_photos, log=print):
    """Score weights on the exam's photos and save the result in data/back_check_eval/: the score, with what it was on."""
    import cv2
    reader = name_check.Reader(weights)
    reads = reader.read_many([cv2.imread(str(p["path"])) for p in exam_photos])
    result = {**score(reads, [p["name"] for p in exam_photos]), "weights": str(weights),
              "sessions": sorted({p["session"] for p in exam_photos}), "scored": datetime.now().isoformat(timespec="seconds"),
              "wrong": [{"key": p["key"], "label": p["name"], **r} for p, r in zip(exam_photos, reads)
                        if r["name"] != p["name"]]}
    EVAL.mkdir(parents=True, exist_ok=True)
    path = EVAL / f"{Path(weights).stem}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    log(f"Saved: {path}")
    return result


def load_reader(path=None):
    """The installed back check, or None without weights (the look then goes by the back's shape, as before)."""
    path = Path(path or WEIGHTS)
    return name_check.Reader(path) if path.exists() else None
