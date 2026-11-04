"""Labelled detector frames from a session a person named the tiles of (a played hand, a bench with its check before trials):
how each tile read frame by frame, an export to train on, and scoring a detector on those frames.

    python scripts/hand_frames.py 20261104_183142_play                               # per tile: right, wrong, missed
    python scripts/hand_frames.py 20261104_183142_play --out data/detector/hand_v8   # also export the frames worth training on
    python scripts/hand_frames.py 20261104_183142_play --out data/detector/train_v8 --onto data/detector/train_v7x3   # a training set
    python scripts/hand_frames.py 20261104_150542_bench --weights NEW.pt --against assets/tile_detector.pt

The names are a person's, the boxes the detector's (mahjong/hand_truth.py), so these labels don't need correcting in Roboflow.
Exam sessions are refused. A session you score a candidate on stops being a check once you've trained on it.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, detector, hand_truth
from mahjong.perception import predictor


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sessions", nargs="+", help="session names or folders")
    p.add_argument("--sessions-dir", type=Path, default=config.SESSIONS, help="where they are (default: this checkout's sessions/)")
    p.add_argument("--out", type=Path, default=None, help="export the frames worth training on to this new folder")
    p.add_argument("--onto", type=Path, default=None,
                   help="with --out: add the frames to the train split of a copy of this dataset (a split_by_scene.py folder), for train_detector.py")
    p.add_argument("--per-arrangement", type=int, default=2, help="frames kept per arrangement of tiles (default 2)")
    p.add_argument("--per-hard-name", type=int, default=40, help="frames kept of a name read wrong or missed (default 40)")
    p.add_argument("--weights", type=Path, default=None, help="score this detector on the frames (default: the reads the session saved)")
    p.add_argument("--against", type=Path, default=None, help="a second detector, scored on the same frames")
    return p.parse_args(argv)


def table(name, tally, per_name):
    s = detector.summary(tally)
    print(f"\n{name}: {s['frames']} frames, {s['tiles']} tiles, {s['right_rate'] * 100:.1f}% right overall "
          f"({s['found_rate'] * 100:.1f}% found, {s['named_right_rate'] * 100:.1f}% of those named right)")
    if s["mixups"]:
        print("  mix-ups (true -> said): " + ", ".join(f"{t}->{w} x{n}" for t, w, n in s["mixups"][:10]))
    print(f"  {'tile':<5}{'frames':>7}{'right':>7}{'wrong':>7}{'missed':>7}")
    for tile, c in sorted(per_name.items(), key=lambda kv: kv[1]["right"] / max(1, sum(kv[1].values()))):
        print(f"  {tile:<5}{sum(c.values()):>7}{100 * c['right'] / max(1, sum(c.values())):>6.1f}%{c['wrong']:>7}{c['missed']:>7}")


def add_onto(base, out):
    """Copy dataset `base`'s frames in beside the export in `out`; its data.yaml too. A hand's frames all go to train: they're
    one scene to split_by_scene, so dealing them to valid would train on none of them. Refuses other class names."""
    import yaml
    names = yaml.safe_load((base / "data.yaml").read_text())["names"]
    if list(names) != detector.TILE_NAMES:
        raise SystemExit(f"{base}: other class names than the detector's")
    for split in ("train", "valid"):
        for kind in ("images", "labels"):
            folder = out / split / kind
            folder.mkdir(parents=True, exist_ok=True)
            for path in (base / split / kind).iterdir():
                if path.suffix != ".cache":
                    shutil.copy(path, folder / path.name)
    shutil.copy(base / "data.yaml", out / "data.yaml")


def main(argv=None):
    args = parse_args(argv)
    if args.out is not None and args.out.exists():
        raise SystemExit(f"{args.out} exists: pick a new folder")
    if args.onto is not None and args.out is None:
        raise SystemExit("--onto needs --out")
    exam = detector.load_exam()["sessions"]
    frames = []
    for session in args.sessions:
        folder = Path(session) if Path(session).is_dir() else args.sessions_dir / session
        if folder.name in exam:
            raise SystemExit(f"{folder.name} is an exam session: score it with detector_eval.py")
        found, refused = hand_truth.labelled(folder)
        print(f"{folder.name}: {len(found)} frames with every tile named" +
              (f"; refused: {', '.join(f'{n} {why}' for why, n in refused.most_common())}" if refused else ""))
        frames += found
    if not frames:
        raise SystemExit("No frames with every tile named")
    if args.weights is None:
        table("as read in the session", *hand_truth.score(frames))
    else:
        for weights in [args.weights] + ([args.against] if args.against else []):
            table(detector.weights_id(weights), *hand_truth.score(frames, predictor(weights)))
    if args.out is not None:
        chosen = hand_truth.pick(frames, args.per_arrangement, args.per_hard_name)
        hand_truth.export(chosen, args.out)
        if args.onto is not None:
            add_onto(args.onto, args.out)
        boxes = sum(len(f["truth"]) for f in chosen)
        print(f"\nExported {len(chosen)} frames, {boxes} labelled tiles, to {args.out}; why each was picked: "
              f"{detector.manifest_path(args.out)}")


if __name__ == "__main__":
    main()
