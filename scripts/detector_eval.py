"""Score the tile detector on the exam: labelled frames of the current scene it is never trained on.

    python scripts/detector_eval.py --exam-dir data/detector/exam_labelled
    python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --weights outputs/detector/v2/weights/best.pt --against assets/tile_detector.pt
    python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --by lighting

--exam-dir is the exam's labelled frames as a YOLO export from Roboflow (data.yaml, then images/ and
labels/, in split folders or not). Only frames of the exam sessions (data/detector_exam.json) count:
anything else in the folder is left out and listed, since scoring frames a model trained on proves
nothing. Per detector: found (labelled tiles it put a box on, matched by overlap), named right (of
those found), missed, extra boxes on no tile, and the names it mixes up. A detection counts at the
live cut (config.DETECTION_CONFIDENCE), one frame at a time. The live scan merges 5 frames, so the
bench does a little better than this. Each score is saved to data/detector_eval/<weights>@<hash>.json.

Ship new weights only if they beat the current ones on the same exam: copy best.pt to
assets/tile_detector.pt and commit it, with these numbers in the message.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, detector


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exam-dir", type=Path, required=True, help="the exam's labelled frames (a YOLO export with data.yaml)")
    p.add_argument("--weights", type=Path, default=config.TILE_DETECTOR, help="detector to score (default: the live one)")
    p.add_argument("--against", type=Path, default=None, help="a second detector, scored on the same frames")
    p.add_argument("--by", choices=["lighting", "session"], default=None, help="also split the scores by this")
    p.add_argument("--mixups", type=int, default=8, help="how many of the commonest mix-ups to print (default 8)")
    return p.parse_args(argv)


def weights_id(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]
    return f"{Path(path).stem}@{digest}"


def predictor(weights):
    from ultralytics import YOLO
    from mahjong.perception import run_detector
    model = YOLO(str(weights))
    return lambda image: run_detector(model, image)


def pct(value):
    return "  -  " if value is None else f"{value * 100:5.1f}%"


def report(name, scores, mixups):
    print(f"\n{name}")
    print(f"  {'':<40} {'frames':>6} {'tiles':>6} {'found':>7} {'named right':>11} {'missed':>6} {'extra':>6}")
    for group in ["all"] + sorted(g for g in scores if g != "all"):
        s = detector.summary(scores[group])
        print(f"  {group:<40} {s['frames']:>6} {s['tiles']:>6} {pct(s['found_rate']):>7} "
              f"{pct(s['named_right_rate']):>11} {s['missed']:>6} {s['extra']:>6}")
    s = detector.summary(scores["all"])
    if s["mixups"]:
        print("  mix-ups (true -> said): " + ", ".join(f"{t}->{w} x{n}" for t, w, n in s["mixups"][:mixups]))
    if s["missed_names"]:
        print("  missed most: " + ", ".join(f"{t} x{n}" for t, n in list(s["missed_names"].items())[:mixups]))


def save(weights, exam, scores, exam_dir):
    config.DETECTOR_EVAL.mkdir(parents=True, exist_ok=True)
    name = weights_id(weights)
    path = config.DETECTOR_EVAL / f"{name}.json"
    path.write_text(json.dumps({"weights": name, "time": datetime.now().isoformat(timespec="seconds"),
                                "exam_sessions": sorted(exam["sessions"]), "exam_dir": str(exam_dir),
                                "scores": {g: detector.summary(t) for g, t in scores.items()}}, indent=2) + "\n")
    return path


def main(argv=None):
    args = parse_args(argv)
    exam = detector.load_exam()
    if not exam["sessions"]:
        raise SystemExit(f"No exam sessions in {config.DETECTOR_EXAM}: capture some with "
                         "scripts/capture_frames.py --exam, label them, then score.")
    for weights in [args.weights] + ([args.against] if args.against else []):
        scores, outside = detector.evaluate(args.exam_dir, predictor(weights), exam, by=args.by)
        if weights == args.weights and outside:
            print(f"Left out {len(outside)} frames that aren't from an exam session, e.g. {outside[0].name}")
        if "all" not in scores:
            raise SystemExit(f"No exam frames under {args.exam_dir} (labelled, from {', '.join(exam['sessions'])})")
        report(weights_id(weights), scores, args.mixups)
        print(f"  saved {save(weights, exam, scores, args.exam_dir)}")


if __name__ == "__main__":
    main()
