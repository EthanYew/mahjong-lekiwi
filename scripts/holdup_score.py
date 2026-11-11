"""Is training a reader for the base camera's view paying off? Every base_cam_train.py result in one table; --installed scores
the installed readers instead.

    python scripts/holdup_score.py                       # every base_cam_train.py result, oldest first, and what they say together
    python scripts/holdup_score.py sessions/20261106_195636_holdup        # only the results saved in that session
    python scripts/holdup_score.py --installed           # the installed classifier and detector on the latest holdup session
    python scripts/holdup_score.py --installed sessions/20261105_150000_holdup --weights outputs/name_check/name_check.pt

The default reads the holdup_check.json / holdup_check_held_out.json files that base_cam_train.py writes (no GPU, nothing
trained). For each: what it trained and tested on, the fresh classifier's right rate on photos it never saw, the control (the
same with the tile greyed out) and the best constant answer. Training pays off when the reader is well above the constant
answer, the control is near it (else the background names the tile), and that holds on a session never trained on
(`base_cam_train.py --test-session`): the last lines say which results meet that.

--installed is face-down plan test 1: the name classifier (assets/name_check.pt, trained on the wrist camera at the hover) and the
detector (assets/tile_detector.pt, the RealSense looking down) on these photos. They read ~0% here because the view is new to
them, and neither is what the plan uses to read a held tile. It prints rates per slice and per tile, every misread, and writes
`holdup_scores.json` and `holdup_sheet.jpg` into the session (every photo with its guesses, wrong reads in red). Needs the GPU
only for speed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import config, base_cam, name_check, perception


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session", nargs="?", type=Path,
                   help="a holdup session folder (default: every real one; with --installed, the latest)")
    p.add_argument("--installed", action="store_true",
                   help="score the installed classifier and detector on one session's photos (test 1) instead")
    p.add_argument("--weights", type=Path, default=name_check.WEIGHTS, help="with --installed: the classifier (default: the installed one)")
    p.add_argument("--detector", type=Path, default=config.TILE_DETECTOR, help="with --installed: the detector (default: the installed one)")
    return p.parse_args(argv)


def latest_session():
    found = sorted(config.SESSIONS.glob("*_holdup"))
    if not found:
        raise SystemExit(f"No holdup session in {config.SESSIONS}: python scripts/base_cam_collect.py")
    return found[-1]


def training_report(session=None):
    """Every base_cam_train.py result (in `session`, or every real holdup session), and the lines that say what they show."""
    folders = [session] if session else [f for f in sorted(config.SESSIONS.glob("*_holdup")) if not base_cam.is_sim(f)]
    results = base_cam.check_results(folders)
    print("\n".join(base_cam.check_lines(results)))
    return results


def installed(folder, weights, detector, reader=None, predict=None):
    rows = base_cam.rows(folder)
    if not rows:
        raise SystemExit(f"No photos in {folder}")
    if reader is None:
        if not weights.exists():
            raise SystemExit(f"No classifier weights at {weights}: python scripts/train_name_check.py")
        reader = name_check.Reader(weights)
    predict = predict or perception.predictor(detector)
    scored = []
    for row in rows:
        image = cv2.imread(str(row["path"]))
        if image is not None:
            scored.append(base_cam.score_photo(row, image, predict, reader))
    s = base_cam.score(scored)
    print(f"{folder.name}: {len(scored)} photos, {len({r['truth'] for r in scored})} tiles; the INSTALLED readers, trained on "
          f"other views (whether a reader trained on this one pays off: python scripts/holdup_score.py)\n")
    print("\n".join(base_cam.summary_lines(s)))
    print(f"\nsaved {base_cam.save_scores(folder, s)}")
    sheet = base_cam.contact_sheet(scored, folder / "holdup_sheet.jpg")
    if sheet:
        print(f"saved {sheet}")
    return s


def main(argv=None, reader=None, predict=None):
    args = parse_args(argv)
    if args.installed:
        return installed(args.session or latest_session(), args.weights, args.detector, reader, predict)
    return training_report(args.session)


if __name__ == "__main__":
    main()
