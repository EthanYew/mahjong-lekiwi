"""Wrist-camera photos of tiles at the hover, gathered from the sessions, and what a person said about them.

Photos stay where they were taken (sessions are the record); nothing here moves or deletes one. Two
sources have files under sessions/:
- wrist_capture.py views: <session>_wrist/wrist/<NAME>/<set>_<tile>_v<view>_f<k>.jpg. One photo per view
  (its f0; the other frames of the view share its verdict), grouped by the physical tile (set, tile):
  if the detector misnamed it, every view of it is wrong together.
- bench attempts: <session>_bench/<n>_<time>_<NAME>/wrist_0000.jpg, the policy's first frame (at the
  hover). One photo per attempt.
(Recorded episodes start at the hover too, but their frames are inside the dataset videos; they're
only counted, by their episode logs.)

A name came from the detector (confirmed by a person in wrist_capture, not at the bench), so a few can
be wrong. scripts/wrist_review.py shows them by name; each verdict is a line in data/wrist_review.jsonl
(the last line for a photo wins): "ok", or "wrong" with the right name, or "wrong" with none (not a
usable photo of any tile). labelled() applies them; counts and training sets go through it.
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import config
from .session import read_jsonl

REVIEW = config.DATA / "wrist_review.jsonl"
SOURCES = ("wrist_capture", "bench")


def photos(sessions_dir=None):
    """Every wrist photo under sessions_dir, simulated sessions left out: [{"key" (path relative to
    sessions_dir), "path", "name" (as taken), "source", "session", "group" (one physical tile),
    "frames" (the view's other files)}], in session order."""
    from .detector import is_sim
    sessions_dir = Path(sessions_dir or config.SESSIONS)
    out = []
    for folder in sorted(sessions_dir.iterdir()) if sessions_dir.is_dir() else ():
        if not folder.is_dir() or is_sim(folder):
            continue
        if folder.name.endswith("_wrist"):
            for first in sorted((folder / "wrist").glob("*/*_f0.jpg")):
                set_tile = "_".join(first.stem.split("_")[:2])
                out.append({"key": str(first.relative_to(sessions_dir)), "path": first, "name": first.parent.name,
                            "source": "wrist_capture", "session": folder.name, "group": f"{folder.name}/{set_tile}",
                            "frames": sorted(first.parent.glob(first.name.replace("_f0.jpg", "_f*.jpg")))})
        elif folder.name.endswith("_bench"):
            for first in sorted(folder.glob("*/wrist_0000.jpg")):
                out.append({"key": str(first.relative_to(sessions_dir)), "path": first,
                            "name": first.parent.name.rsplit("_", 1)[-1], "source": "bench", "session": folder.name,
                            "group": f"{folder.name}/{first.parent.name}", "frames": [first]})
    return out


def load_review(path=None):
    """{photo key: its latest verdict record}."""
    return {record["photo"]: record for record in read_jsonl(path or REVIEW)}


def record(keys, verdict, name=None, path=None):
    """Append a verdict ("ok" / "wrong", with the right name if known) for each photo key. Returns the records."""
    path = Path(path or REVIEW)
    time = datetime.now().isoformat(timespec="seconds")
    records = [{"photo": key, "verdict": verdict, "name": name, "time": time} for key in keys]
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def name_of(photo, review):
    """The photo's name after review: the corrected one, its own, or None (wrong, no right name)."""
    verdict = review.get(photo["key"])
    if verdict is None or verdict["verdict"] == "ok":
        return photo["name"]
    return verdict.get("name")


def labelled(all_photos, review):
    """The usable photos, each with "name" as reviewed (photos marked wrong without a name left out)."""
    out = []
    for photo in all_photos:
        name = name_of(photo, review)
        if name is not None:
            out.append({**photo, "name": name, "reviewed": photo["key"] in review})
    return out


def counts(sessions_dir=None, review=None):
    """{source: Counter of names} over the usable photos, as reviewed."""
    review = load_review() if review is None else review
    out = {source: Counter() for source in SOURCES}
    for photo in labelled(photos(sessions_dir), review):
        out[photo["source"]][photo["name"]] += 1
    return out
