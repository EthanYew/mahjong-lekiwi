"""Can the base camera's view be learned at all? Train a small classifier on some hold-up photos, score it on others it never saw.

    python scripts/holdup_check.py                        # every real holdup session; holds out the newest session if
                                                          # there are two or more, else the last 2 holds of each tile/way up
    python scripts/holdup_check.py --by hold              # always split by hold (optimistic: same session's light and hands)
    python scripts/holdup_check.py --sessions sessions/A_holdup,sessions/B_holdup --epochs 200

Why: the installed classifier and detector read nothing here (holdup_score.py), because neither was trained on this view.
This answers the question before a full capture is paid for. It is NOT a fine-tune of the installed weights: it starts a
fresh YOLO11n-cls from its ImageNet weights, with a head for just the tiles held in these sessions, framed as the whole
frame's centre square (the wrist camera's fixed crop doesn't apply: a held tile isn't where a hovered one is). Holds are
split whole, never a shot's near-identical frames across train and test. It runs --seeds times (small sets vary), and
prints the held-out right rate against chance and against the best constant answer.

The bar (docs/face_down_plan.md, test 1b, set before the first run): 70% or more, on 3+ tiles and well above the best
constant answer -> worth a real capture at the arm's pose. 40% or less -> the view or the hold is the problem, look at the
photos before capturing more. In between: capture more and rerun. Writes holdup_check.json next to the newest session.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import config, holdup, name_check

PASS_RATE, FAIL_RATE = 0.70, 0.40


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sessions", default=None, help="holdup session folders, comma separated (default: every real one)")
    p.add_argument("--by", choices=("auto", "session", "hold"), default="auto", help="how to hold photos out (default auto)")
    p.add_argument("--test-holds", type=int, default=2, help="with --by hold: holds of each tile and way up held out (default 2)")
    p.add_argument("--epochs", type=int, default=150, help="default 150 (a small set needs many passes to train)")
    p.add_argument("--seeds", type=int, default=3, help="runs, seeds 0..N-1 (default 3)")
    return p.parse_args(argv)


def find_sessions(arg):
    if arg:
        folders = [Path(s) for s in arg.split(",")]
    else:
        folders = sorted(config.SESSIONS.glob("*_holdup"))
    return [f for f in folders if f.is_dir() and not holdup.is_sim(f)]


def verdict(rate, tiles, constant):
    if tiles < 3:
        return f"only {tiles} tile(s) held out: too few to judge. Capture 3 or more"
    if rate >= PASS_RATE and rate >= constant + 0.2:
        return "PASS: worth a real capture at the arm's pose"
    if rate <= FAIL_RATE:
        return "FAIL: look at the photos (hands, tile out of frame?) before capturing more"
    return "IN BETWEEN: capture more holds and rerun"


def main(argv=None, train_fn=None, predict_fn=None):
    args = parse_args(argv)
    folders = find_sessions(args.sessions)
    rows = [r for f in folders for r in holdup.rows(f)]
    if not rows:
        raise SystemExit(f"No holdup photos in {[str(f) for f in folders] or config.SESSIONS}: python scripts/holdup_capture.py")
    train_rows, test_rows, how, caveat = holdup.split(rows, args.by, args.test_holds)
    print(f"{len(rows)} photos from {len(folders)} session(s); {how}")
    print(f"  train {len(train_rows)} photos of {len({r['truth'] for r in train_rows})} tiles, "
          f"test {len(test_rows)} of {len({r['truth'] for r in test_rows})}")
    if caveat:
        print(f"  {caveat}")
    train_fn = train_fn or name_check.train
    predict_fn = predict_fn or name_check.predict
    photos = [{"name": r["truth"], "path": r["path"], "session": r["session"]} for r in train_rows]
    images = [cv2.imread(str(r["path"])) for r in test_rows]
    runs = []
    for seed in range(args.seeds):
        checkpoint = train_fn(photos, seed=seed, epochs=args.epochs, log=lambda *_: None, crop_fn=holdup.whole_frame)
        said = predict_fn(checkpoint, images, crop_fn=holdup.whole_frame)
        runs.append(holdup.check_summary(test_rows, said))
        print(f"  seed {seed}: right {runs[-1]['right']:.0%}")
    rate = statistics.mean(r["right"] for r in runs)
    first = runs[0]
    print(f"\nheld-out right rate {rate:.0%} (seeds {min(r['right'] for r in runs):.0%} to {max(r['right'] for r in runs):.0%}); "
          f"chance {first['chance']:.0%}, best constant answer {first['best_constant_answer']:.0%}")
    print("  per tile (seed 0): " + ", ".join(f"{t} {v:.0%}" for t, v in first["per_tile"].items()))
    print("  per way up (seed 0): " + ", ".join(f"{k} {v:.0%}" for k, v in first["per_kind"].items()))
    print("  confusions (seed 0): " + (", ".join(f"{k} x{n}" for k, n in first["confusions"]) or "none"))
    result = verdict(rate, first["tiles"], first["best_constant_answer"])
    print(f"\n{result}")
    out = folders[-1] / "holdup_check.json"
    out.write_text(json.dumps({"split": how, "caveat": caveat, "sessions": [f.name for f in folders], "epochs": args.epochs,
                               "mean_right": rate, "runs": runs, "verdict": result}, indent=2) + "\n")
    print(f"saved {out}")
    return {"mean_right": rate, "runs": runs, "verdict": result}


if __name__ == "__main__":
    main()
