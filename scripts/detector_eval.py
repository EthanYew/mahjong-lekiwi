"""Score the tile detector on the exam: labelled frames of the current scene it is never trained on.

    python scripts/detector_eval.py --exam-dir data/detector/exam_labelled
    python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --weights outputs/detector/v2/weights/best.pt --against assets/tile_detector.pt
    python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --by lighting
    python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --ignore flowers

--exam-dir is the exam's labelled frames as a YOLO export from Roboflow (data.yaml, then images/ and
labels/, in split folders or not). Only frames of the exam sessions (data/detector_exam.json) count:
anything else in the folder is left out and listed, since scoring frames a model trained on proves
nothing. Per detector: found (labelled tiles it put a box on, matched by overlap), named right (of
those found), right overall (found and named right: the tile accuracy), missed, extra boxes on no
tile, and the names it mixes up. A detection counts at the live cut (config.DETECTION_CONFIDENCE), one
frame at a time. The live scan merges 5 frames, so the bench does a little better than this.

--ignore leaves tile names out (flowers = 1F,2F,3F,4F): boxes on those tiles count for nothing, while
another tile read as one of them is still a mix-up. A label drawn twice on one tile (same name, boxes
overlapping 80%+) is merged and listed: delete the copy in Roboflow. Each score is saved to
data/detector_eval/<weights>@<hash>[_without_<names>].json.

Tile backs (the `back` class): with backs labelled or called, each group also gets the back gate's two numbers
(detector.back_numbers): backs found, the share of labelled backs given a box called back (gate: at least 95%, on the
face-down exam), and false backs, boxes called back on no tile plus face tiles read as back, per frame (gate: at most 1
per 5 frames, on the face-up exam). "faces" scores the face tiles alone, so weights without the class compare fairly on
names. And where a tile was scanned face up and then flipped in place, a scan apart, the offset of its back box's
centre from its face box's, in mm on the table (detector.back_centres): the bench keeps face-down hover and return
samples out of the fits until a detector passes it, saved to data/detector_eval/<weights>@<hash>_back_centres.json.

Ship new weights only if they beat the current ones on the same exam: copy best.pt to
assets/tile_detector.pt and commit it, with these numbers in the message.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, detector
from mahjong.detector import (BACKS_FOUND_GATE, FALSE_BACKS_PER_FRAME_GATE, back_centres, back_centres_path, weights_id,
                              wrong_rate_bound)
from mahjong.perception import predictor


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exam-dir", type=Path, required=True, help="the exam's labelled frames (a YOLO export with data.yaml)")
    p.add_argument("--weights", type=Path, default=config.TILE_DETECTOR, help="detector to score (default: the live one)")
    p.add_argument("--against", type=Path, default=None, help="a second detector, scored on the same frames")
    p.add_argument("--by", choices=["lighting", "session"], default=None, help="also split the scores by this")
    p.add_argument("--ignore", type=parse_names, default=(),
                   help="tile names to leave out of the score, e.g. flowers (= 1F,2F,3F,4F) or flowers,EW")
    p.add_argument("--sessions-dir", type=Path, default=config.SESSIONS,
                   help="where the exam's capture sessions are, for the centre check's cameras (from a worktree: the main "
                        "checkout's sessions/)")
    p.add_argument("--mixups", type=int, default=8, help="how many of the commonest mix-ups to print (default 8)")
    return p.parse_args(argv)


GROUPS = {"flowers": ["1F", "2F", "3F", "4F"]}


def parse_names(text):
    """'flowers,EW' -> ('1F', '2F', '3F', '4F', 'EW'); every name must be a tile name."""
    names = [n for part in text.split(",") if part.strip()
             for n in GROUPS.get(part.strip().lower(), [part.strip().upper()])]
    unknown = [n for n in names if n not in detector.TILE_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(f"not tile names: {', '.join(unknown)} (groups: {', '.join(GROUPS)})")
    return tuple(dict.fromkeys(names))


def pct(value):
    return "  -  " if value is None else f"{value * 100:5.1f}%"


def report(name, scores, mixups):
    print(f"\n{name}")
    print(f"  {'':<40} {'frames':>6} {'tiles':>6} {'found':>7} {'named right':>11} {'right overall':>13} "
          f"{'missed':>6} {'extra':>6}")
    for group in ["all"] + sorted(g for g in scores if g != "all"):
        s = detector.summary(scores[group])
        print(f"  {group:<40} {s['frames']:>6} {s['tiles']:>6} {pct(s['found_rate']):>7} "
              f"{pct(s['named_right_rate']):>11} {pct(s['right_rate']):>13} {s['missed']:>6} {s['extra']:>6}")
    s = detector.summary(scores["all"])
    wrong = s["found"] - s["named_right"]
    if s["found"]:
        # a wrong name is what autonomous play can't afford (a silent error, docs/autonomy_plan.md section 4):
        # what this exam can rule out, as the highest rate still consistent with what it saw
        print(f"  wrong names: {wrong} of {s['found']} found; the true rate could still be up to "
              f"{pct(wrong_rate_bound(wrong, s['found'])).strip()} (95%)")
    if s["mixups"]:
        print("  mix-ups (true -> said): " + ", ".join(f"{t}->{w} x{n}" for t, w, n in s["mixups"][:mixups]))
    if s["missed_names"]:
        print("  missed most: " + ", ".join(f"{t} x{n}" for t, n in list(s["missed_names"].items())[:mixups]))
    report_backs(scores)


def gate(ok):
    return "pass" if ok else "FAIL"


def report_backs(scores):
    """The back gate's numbers per group, and the face tiles on their own, when any back was labelled or called."""
    rows = [(g, detector.summary(scores[g])) for g in ["all"] + sorted(g for g in scores if g != "all")]
    if not any(s["backs"]["labelled"] or s["backs"]["false"] for _, s in rows):
        return
    print(f"  {'backs':<40} {'labelled':>8} {'boxed':>6} {'called back':>11} {'false':>6} {'per frame':>9}")
    for group, s in rows:
        b = s["backs"]
        found = (f"{pct(b['found_rate'])} {gate(b['found_rate'] >= BACKS_FOUND_GATE)}" if b["labelled"] else "  -")
        print(f"  {group:<40} {b['labelled']:>8} {b['boxed']:>6} {found:>11} {b['false']:>6} "
              f"{b['false_per_frame']:>5.2f} {gate(b['false_per_frame'] <= FALSE_BACKS_PER_FRAME_GATE)}")
    print(f"    gates: backs called back >= {BACKS_FOUND_GATE:.0%} on face-down frames; false backs (boxes called back on "
          f"no tile + faces read as back) <= {FALSE_BACKS_PER_FRAME_GATE} per frame on face-up frames")
    f = rows[0][1]["faces"]
    print(f"  faces only: {f['tiles']} tiles, found {pct(f['found_rate']).strip()}, named right "
          f"{pct(f['named_right_rate']).strip()}, right overall {pct(f['right_rate']).strip()}")


def report_centres(result):
    print(f"  back centres vs face centres: {result['flips']} tiles flipped in place a scan apart, {result['measured']} "
          f"boxed both ways" + (f", {result['no_camera']} without a saved camera" if result["no_camera"] else ""))
    if result["measured"]:
        print(f"    back minus face, median: x {result['median_dx_mm']:+.1f}mm, y {result['median_dy_mm']:+.1f}mm "
              f"(distance median {result['median_dist_mm']:.1f}, 90th {result['p90_dist_mm']:.1f}mm); the hand moved them "
              f"x {result['label_median_dx_mm']:+.1f}, y {result['label_median_dy_mm']:+.1f}mm by the labels; less the hand's "
              f"move, 90% of tiles within {result['p90_spread_mm']:.1f}mm")
    print(f"    {gate(result['passes'])}: median within {result['gate_mm']:.0f}mm along x and y, 90% within "
          f"{result['spread_gate_mm']:.0f}mm less the hand's move, over at least {result['min_pairs']} tiles")


def save_centres(weights, result, exam_dir):
    """Save the centre check, unless one on more tiles is already saved for these weights: the bench reads one file, and a
    smaller exam (or a different one) shouldn't quietly replace it. None if it was kept."""
    path = back_centres_path(weights)
    if path.exists() and json.loads(path.read_text()).get("measured", 0) > result["measured"]:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"weights": weights_id(weights), "time": datetime.now().isoformat(timespec="seconds"),
                                "exam_dir": str(exam_dir), **result}, indent=2) + "\n")
    return path


def save(weights, exam, scores, exam_dir, ignore=()):
    config.DETECTOR_EVAL.mkdir(parents=True, exist_ok=True)
    name = weights_id(weights)
    path = config.DETECTOR_EVAL / f"{name}{'_without_' + '-'.join(ignore) if ignore else ''}.json"
    path.write_text(json.dumps({"weights": name, "time": datetime.now().isoformat(timespec="seconds"),
                                "exam_sessions": sorted(exam["sessions"]), "exam_dir": str(exam_dir),
                                "ignored_names": list(ignore),
                                "scores": {g: detector.summary(t) for g, t in scores.items()}}, indent=2) + "\n")
    return path


def main(argv=None):
    args = parse_args(argv)
    exam = detector.load_exam()
    if not exam["sessions"]:
        raise SystemExit(f"No exam sessions in {config.DETECTOR_EXAM}: capture some with "
                         "scripts/capture_frames.py --exam, label them, then score.")
    faces = {}
    for weights in [args.weights] + ([args.against] if args.against else []):
        images, outside, duplicates = detector.exam_images(args.exam_dir, predictor(weights), exam, by=args.by)
        scores = detector.score(images, args.ignore)
        if weights == args.weights and outside:
            print(f"Left out {len(outside)} frames that aren't from an exam session, e.g. {outside[0].name}")
        if weights == args.weights and duplicates:
            print(f"Merged {len(duplicates)} labels drawn twice on one tile (delete the copies in Roboflow):")
            for frame, box in duplicates:
                print(f"  {detector.original_name(frame)}: {box['tile_name']}")
        if weights == args.weights and args.ignore:
            print(f"Left out of the score: {', '.join(args.ignore)}")
        if "all" not in scores:
            raise SystemExit(f"No exam frames under {args.exam_dir} (labelled, from {', '.join(exam['sessions'])})")
        report(weights_id(weights), scores, args.mixups)
        print(f"  saved {save(weights, exam, scores, args.exam_dir, args.ignore)}")
        faces[weights] = detector.summary(scores["all"])["faces"]
        centres = back_centres(images, args.sessions_dir)
        if centres["flips"]:
            report_centres(centres)
            if centres["measured"]:
                saved = save_centres(weights, centres, args.exam_dir)
                print(f"  saved {saved}" if saved else f"  kept the saved check, on more tiles: {back_centres_path(weights)}")
    if args.against:
        new, old = faces[args.weights], faces[args.against]
        ok = (new["right_rate"] or 0) >= (old["right_rate"] or 0)
        print(f"\nFace tiles, right overall: {pct(new['right_rate']).strip()} against {pct(old['right_rate']).strip()} "
              f"({gate(ok)}: no worse than {weights_id(args.against)})")


if __name__ == "__main__":
    main()
