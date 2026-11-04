"""Score the wrist-camera name check on its exam: photos a person named, in sessions it never trained on (GPU, seconds).

    python scripts/name_check_eval.py                                  # the installed weights
    python scripts/name_check_eval.py --weights outputs/name_check/name_check.pt
    python scripts/name_check_eval.py --list                           # what the exam holds

The exam's sessions are in data/name_check_exam.json. The gate is names accepted and wrong: the reader and the detector agree
on a wrong name, so nothing catches it. It must be 0, towards ~1000 reads (docs/name_check_plan.md). Until then play only uses
a disagreement to ask a person. Scores go in data/name_check_eval/.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import name_check


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--weights", type=Path, default=name_check.WEIGHTS, help="the reader to score (default: the installed one)")
    p.add_argument("--sessions-dir", type=Path, default=None, help="read another checkout's sessions/ (default: this one's)")
    p.add_argument("--list", action="store_true", help="print the exam's photos per session; score nothing")
    return p.parse_args(argv)


def weights_name(path):
    return f"{Path(path).stem}@{hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]}"


def main(argv=None):
    args = parse_args(argv)
    rows = name_check.exam_rows(args.sessions_dir)
    if not rows:
        raise SystemExit(f"No exam photos: list a session with a person's names in {name_check.EXAM}")
    for session in sorted({r["session"] for r in rows}):
        of = [r for r in rows if r["session"] == session]
        print(f"  {session}: {len(of)} photos, {len({r['group'] for r in of})} tiles, {len({r['truth'] for r in of})} names")
    if args.list:
        return None
    if not args.weights.exists():
        raise SystemExit(f"No weights at {args.weights}: python scripts/train_name_check.py")
    reader = name_check.Reader(args.weights)
    trained_on = set(reader_sessions(args.weights)) & {r["session"] for r in rows}
    if trained_on:
        raise SystemExit(f"These weights trained on exam sessions: {', '.join(sorted(trained_on))}")
    for row, read in zip(rows, reader.read_many([cv2.imread(str(r["path"])) for r in rows])):
        row["said"], row["confidence"] = read["name"], read["confidence"]
    s = name_check.score(rows)
    name = weights_name(args.weights)
    print(f"\n{name}: {s['photos']} photos of {s['tiles']} tiles, {s['names']} names")
    print(f"  reader right       {s['right']} ({s['right_rate']:.1%})")
    print(f"  detector right     {s['yolo_right']} ({s['yolo_right'] / s['photos']:.1%}); wrong {s['yolo_wrong']}, "
          f"caught by a disagreement {s['yolo_wrong_caught']}")
    print(f"  accepted and wrong {s['accepted_wrong']} of {s['agreed']} agreements; the true rate could still be up to "
          f"{s['accepted_wrong_bound']:.1%} (95%)")
    print(f"  would ask          {s['asks']} ({s['ask_rate']:.1%})")
    print("  misreads: " + (", ".join(f"{k} x{n}" for k, n in s["misreads"]) or "none"))
    name_check.EVAL.mkdir(parents=True, exist_ok=True)
    path = name_check.EVAL / f"{name}.json"
    path.write_text(json.dumps({"weights": name, "time": datetime.now().isoformat(timespec="seconds"),
                                "exam_sessions": sorted({r["session"] for r in rows}), "scores": s}, indent=2) + "\n")
    print(f"  saved {path}")
    return s


def reader_sessions(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False).get("sessions", [])


if __name__ == "__main__":
    main()
