"""Train the wrist camera's back check (back or empty at the hover; mahjong/back_check.py) and score it on its exam.

    python scripts/train_back_check.py --dry-run                     # the photos it would train on, and the exam's
    python scripts/train_back_check.py                               # into outputs/back_check/back_check.pt, then scored
    python scripts/train_back_check.py --score outputs/back_check/back_check.pt     # score weights on the exam only

Exam sessions (data/back_check_exam.json) never train it. Each score is saved in data/back_check_eval/. Installing the
weights (copying them to assets/back_check.pt, which the bench's wrist look then asks first) is the owner's call.
Empty photos come from `wrist_capture.py --empty LAYOUT`. sessions/ is per checkout: in a worktree, pass
--sessions-dir ~/Projects/mahjong-lekiwi/sessions.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import back_check, config, name_check


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=config.REPO / "outputs" / "back_check" / "back_check.pt",
                   help="where the weights go (default outputs/back_check/back_check.pt)")
    p.add_argument("--seed", type=int, default=0, help="default 0: pick it before any score, not after")
    p.add_argument("--epochs", type=int, default=name_check.EPOCHS, help=f"default {name_check.EPOCHS}")
    p.add_argument("--sessions-dir", type=Path, default=None, help="read another checkout's sessions/ (default: this one's)")
    p.add_argument("--score", type=Path, default=None, metavar="WEIGHTS", help="score these weights on the exam; don't train")
    p.add_argument("--dry-run", action="store_true", help="print the photos it would train and score on; don't train")
    return p.parse_args(argv)


def print_score(result):
    def share(v):
        return "n/a" if v is None else f"{v:.1%}"
    print(f"Exam: backs seen {share(result['backs_seen'])} of {result['backs']} ({result['backs_called_empty']} called "
          f"empty), empties seen {share(result['empties_seen'])} of {result['empties']}, false backs {result['false_backs']} "
          f"(at {result['confident']:.2f} confidence; the rest unsure, which falls back to the back's shape)")
    print(f"Gate (0 false backs in {back_check.GATE_MIN_EMPTIES}+ empties, backs seen >= {back_check.GATE_BACKS_SEEN:.0%}): "
          + ("PASS" if result["passes"] else "not passed"))


def main(argv=None):
    args = parse_args(argv)
    training, exam = back_check.split(back_check.photos(args.sessions_dir))
    print(f"Training: {back_check.summary(training)}")
    print(f"Exam:     {back_check.summary(exam)}")
    if args.dry_run:
        return None
    if args.score:
        weights = args.score
    else:
        checkpoint = back_check.train(training, seed=args.seed, epochs=args.epochs)
        weights = back_check.save(checkpoint, args.out)
        print(f"Weights: {weights}")
    if not exam:
        print("No exam photos: list held-out sessions in data/back_check_exam.json before trusting these weights")
        return weights
    print_score(back_check.evaluate(weights, exam))
    return weights


if __name__ == "__main__":
    main()
