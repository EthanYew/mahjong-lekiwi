"""Score a holdup_capture.py session: what the name classifier and the tile detector make of tiles held up to the base camera.

    python scripts/holdup_score.py                       # the latest holdup session
    python scripts/holdup_score.py sessions/20261105_150000_holdup
    python scripts/holdup_score.py --weights outputs/name_check/name_check.pt

Prints rates per slice (all, each distance, each way up, each daylight setting), the right-name rate per tile, and every
misread, and writes `holdup_scores.json` and `holdup_sheet.jpg` (every photo with the truth, the classifier's guess and the
detector's, wrong reads in red) into the session folder: look through the sheet, it is the check of the exam. The plan's stop
is "nothing reads at any distance": the last line says whether anything did. The detector (assets/tile_detector.pt) was
trained on the RealSense looking down at the mat, and the classifier (assets/name_check.pt) on the wrist camera at the hover:
a low score says the view is new to them. Needs the GPU only for speed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import config, holdup, name_check, perception


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session", nargs="?", type=Path, help="a holdup session folder (default: the latest)")
    p.add_argument("--weights", type=Path, default=name_check.WEIGHTS, help="the classifier (default: the installed one)")
    p.add_argument("--detector", type=Path, default=config.TILE_DETECTOR, help="the detector (default: the installed one)")
    return p.parse_args(argv)


def latest_session():
    found = sorted(config.SESSIONS.glob("*_holdup"))
    if not found:
        raise SystemExit(f"No holdup session in {config.SESSIONS}: python scripts/holdup_capture.py")
    return found[-1]


def main(argv=None, reader=None, predict=None):
    args = parse_args(argv)
    folder = args.session or latest_session()
    rows = holdup.rows(folder)
    if not rows:
        raise SystemExit(f"No photos in {folder}")
    if reader is None:
        if not args.weights.exists():
            raise SystemExit(f"No classifier weights at {args.weights}: python scripts/train_name_check.py")
        reader = name_check.Reader(args.weights)
    predict = predict or perception.predictor(args.detector)
    scored = []
    for row in rows:
        image = cv2.imread(str(row["path"]))
        if image is not None:
            scored.append(holdup.score_photo(row, image, predict, reader))
    s = holdup.score(scored)
    print(f"{folder.name}: {len(scored)} photos, {len({r['truth'] for r in scored})} tiles\n")
    print("\n".join(holdup.summary_lines(s)))
    print(f"\nsaved {holdup.save_scores(folder, s)}")
    sheet = holdup.contact_sheet(scored, folder / "holdup_sheet.jpg")
    if sheet:
        print(f"saved {sheet}")
    return s


if __name__ == "__main__":
    main()
