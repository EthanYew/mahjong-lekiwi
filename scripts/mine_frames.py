"""Frames from past sessions for the tile detector to learn from, exported pre-labelled for Roboflow.

Every bench and capture scan saves the frames YOLO looked at (sessions/*/scans/). This picks the ones
it struggled with: a tile given two names across a scan's 5 frames (flicker), a tile seen in some
frames and not others (dropout), a tile read below 0.7 confidence (low), or a scan a person called a
detection glitch. --all adds one plain frame from every other scan, since the detector has never seen
the current scene and plain frames teach it too.

    python scripts/mine_frames.py --list                 # what it would take, nothing written
    python scripts/mine_frames.py                        # the frames it struggled with, every session
    python scripts/mine_frames.py --all --per-session 10 # plus plain frames, up to 10 a session
    python scripts/mine_frames.py --since 20261027       # sessions from that date (name prefix) on
    python scripts/mine_frames.py --exam                 # the exam sessions' frames, for labelling the exam

The export (data/detector/<name>/) is laid out as Roboflow's own YOLO export: data.yaml (the label
map) and <split>/images/ + <split>/labels/ with the detector's own boxes as a start (split train, or
test for the exam). Drag the whole folder into Roboflow's upload: it comes in pre-labelled; correct
the boxes there. Why each frame was picked is in <name>_manifest.csv beside it (kept out of the
upload). Training frames go to the training project; exam frames to a project of
their own, so a dataset version can't pick them up. Exam sessions (data/detector_exam.json) are never
in a training export, and simulated sessions never in any.
"""
import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, detector


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sessions", nargs="*", type=Path, help="session folders (default: every session)")
    p.add_argument("--all", action="store_true", help="also one plain frame from every scan without issues")
    p.add_argument("--per-session", type=int, default=15,
                   help="with --all: at most this many plain frames a session, spread over it (default 15)")
    p.add_argument("--since", default=None, help="only sessions whose name sorts at or after this, e.g. 20261027")
    p.add_argument("--exam", action="store_true",
                   help="export the exam sessions' frames (one a scan) instead of training frames")
    p.add_argument("--list", action="store_true", help="print what would be exported, write nothing")
    p.add_argument("--out", type=Path, default=None, help="export folder (default data/detector/<kind>_<time>)")
    return p.parse_args(argv)


def spread(rows, n):
    """At most n of rows, evenly spaced (consecutive scans of a session are near copies)."""
    if len(rows) <= n:
        return rows
    return [rows[round(i * (len(rows) - 1) / (n - 1))] for i in range(n)] if n > 1 else rows[:n]


def pick(session_dirs, exam, everything=False, per_session=15, exam_only=False):
    """(rows to export, {session: why it was left out})."""
    rows, left_out = [], {}
    for folder in session_dirs:
        if detector.is_sim(folder):
            left_out[folder.name] = "simulated"
            continue
        in_exam = folder.name in exam["sessions"]
        if in_exam != exam_only:
            left_out[folder.name] = "exam session" if in_exam else "not an exam session"
            continue
        found = detector.mine(folder, everything=everything or exam_only)
        shaky = [r for r in found if r["reasons"]]
        plain = [r for r in found if not r["reasons"]]
        rows += shaky + (plain if exam_only else spread(plain, per_session))
    return rows, left_out


def main(argv=None):
    args = parse_args(argv)
    folders = args.sessions or sorted(p for p in config.SESSIONS.glob("*") if (p / "scans").is_dir())
    if args.since:
        folders = [f for f in folders if f.name >= args.since]
    exam = detector.load_exam()
    rows, left_out = pick(folders, exam, args.all, args.per_session, args.exam)
    reasons = Counter(r.split(":")[0] for row in rows for r in row["reasons"])
    per_session = Counter(r["session"] for r in rows)
    for session, n in per_session.items():
        print(f"  {session}: {n} frames")
    for session, why in left_out.items():
        print(f"  {session}: left out ({why})")
    print(f"{len(rows)} frames: {sum(bool(r['reasons']) for r in rows)} it struggled with "
          f"({', '.join(f'{k} {n}' for k, n in reasons.most_common()) or 'none'}), "
          f"{sum(not r['reasons'] for r in rows)} plain")
    if args.list or not rows:
        return
    out = args.out or config.DETECTOR_DATA / f"{'exam' if args.exam else 'mined'}_{datetime.now():%Y%m%d_%H%M%S}"
    detector.export(rows, out, split="test" if args.exam else "train")
    print(f"Exported to {out} (why each frame: {detector.manifest_path(out).name}).\nDrag that whole folder "
          f"(data.yaml included) into Roboflow's "
          f"{'exam project (kept apart from training)' if args.exam else 'training project'} and correct the labels.")


if __name__ == "__main__":
    main()
