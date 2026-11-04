"""Train the wrist-camera name check on the wrist photos a person named (mahjong/name_check.py, GPU, a minute or two).

    python scripts/train_name_check.py --dry-run                # what it would train on
    python scripts/train_name_check.py                          # into outputs/name_check/name_check.pt
    python scripts/train_name_check.py --out assets/name_check.pt

Exam sessions (data/name_check_exam.json) never train it. Score the result before installing it:
    python scripts/name_check_eval.py --weights outputs/name_check/name_check.pt
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, name_check


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=config.REPO / "outputs" / "name_check" / "name_check.pt",
                   help="where the weights go (default outputs/name_check/name_check.pt)")
    p.add_argument("--seed", type=int, default=0, help="default 0: pick it before any score, not after")
    p.add_argument("--epochs", type=int, default=name_check.EPOCHS, help=f"default {name_check.EPOCHS}")
    p.add_argument("--sessions-dir", type=Path, default=None, help="read another checkout's sessions/ (default: this one's)")
    p.add_argument("--dry-run", action="store_true", help="print the photos it would train on; don't train")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    photos = name_check.training_photos(args.sessions_dir)
    if not photos:
        raise SystemExit("No photos a person named: wrist_capture.py takes them, wrist_review.py reviews them")
    names = Counter(p["name"] for p in photos)
    print(f"{len(photos)} photos of {len(names)} names, from {len({p['session'] for p in photos})} sessions "
          f"({sum(p['source'] == 'wrist_capture' for p in photos)} wrist_capture, "
          f"{sum(p['source'] == 'bench' for p in photos)} reviewed bench photos)")
    print("  thinnest: " + ", ".join(f"{n} {c}" for n, c in names.most_common()[:-9:-1]))
    print(f"  left out, the exam: {', '.join(name_check.load_exam()['sessions']) or 'none'}")
    if args.dry_run:
        return None
    checkpoint = name_check.train(photos, seed=args.seed, epochs=args.epochs)
    path = name_check.save(checkpoint, args.out)
    print(f"Weights: {path}\nScore them: python scripts/name_check_eval.py --weights {path}")
    return path


if __name__ == "__main__":
    main()
