"""Fine-tune the tile detector on a labelled dataset, starting from the live weights.

    python scripts/train_detector.py --data "data/detector/train_v3/data.yaml" --name v3 --dry-run
    python scripts/train_detector.py --data "data/detector/train_v3/data.yaml" --name v3

--data is a YOLO export of the training project from Roboflow (the July frames plus the labelled frames
mined from sessions, scripts/mine_frames.py), re-split by scripts/split_by_scene.py so that no scene is
in both train and valid. Before training it counts images and boxes per
split and per class, and refuses if any frame is from an exam session (data/detector_exam.json):
the exam only means something if the detector never saw it. --dry-run stops there.

The defaults are the recipe of the live detector (assets/tile_detector.pt, trained 2026-07-21):
imgsz 1280 (a tile is ~24x18px at 848x480), lr0 0.0005 from those weights, no left-right flips (a
mirrored tile face is no real tile), mild colour, scale and rotation changes. Training runs on the GPU
into outputs/detector/<name>/ (around an hour for ~1000 frames). Then score it against the live one:

    python scripts/detector_eval.py --exam-dir EXAM --weights outputs/detector/<name>/weights/best.pt --against assets/tile_detector.pt
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, detector

RECIPE = {"epochs": 40, "patience": 15, "batch": 6, "imgsz": 1280, "lr0": 0.0005, "warmup_bias_lr": 0.0,
          "hsv_h": 0.01, "hsv_s": 0.3, "hsv_v": 0.2, "degrees": 5.0, "translate": 0.05, "scale": 0.3,
          "fliplr": 0.0, "freeze": 0}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True, help="the training export's data.yaml")
    p.add_argument("--name", required=True, help="run name: weights land in outputs/detector/<name>/weights/")
    p.add_argument("--start", type=Path, default=config.TILE_DETECTOR, help="weights to start from (default: the live ones)")
    p.add_argument("--epochs", type=int, default=RECIPE["epochs"], help=f"default {RECIPE['epochs']}")
    p.add_argument("--batch", type=int, default=RECIPE["batch"], help=f"default {RECIPE['batch']} (GPU memory at imgsz 1280)")
    p.add_argument("--dry-run", action="store_true", help="check the dataset and print the settings; don't train")
    return p.parse_args(argv)


def check(data_yaml, exam=None):
    """{"splits": {split: (images, boxes)}, "classes": Counter, "names": [...], "exam_frames": [...]} for the
    export holding data_yaml (every image under its folder)."""
    import yaml
    root = Path(data_yaml).parent
    names = yaml.safe_load(Path(data_yaml).read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.parent.name == "images")
    splits, classes, train_classes = Counter(), Counter(), Counter()
    boxes_per_split = Counter()
    for image in images:
        split = image.parent.parent.name if image.parent.parent != root else "(none)"
        splits[split] += 1
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        if label.exists():
            for line in label.read_text().splitlines():
                if line.split():
                    classes[names[int(line.split()[0])]] += 1
                    train_classes[names[int(line.split()[0])]] += split in ("train", "(none)")   # no split folders: all train
                    boxes_per_split[split] += 1
    return {"splits": {s: (n, boxes_per_split[s]) for s, n in splits.items()}, "classes": classes,
            "train_classes": train_classes, "names": names,
            "exam_frames": detector.exam_frames(images, exam), "images": len(images)}


def main(argv=None):
    args = parse_args(argv)
    found = check(args.data)
    print(f"{args.data}: {found['images']} images")
    for split, (n, boxes) in sorted(found["splits"].items()):
        print(f"  {split:<8} {n:>5} images {boxes:>6} boxes")
    few = [n for n in found["names"] if found["classes"][n] < 30]
    print("  boxes per class: " + ", ".join(f"{n} {found['classes'][n]}" for n in found["names"]))
    if few:
        print(f"  under 30 boxes: {', '.join(few)} — expect these to stay weak")
    unseen = [n for n in found["names"] if found["classes"][n] and not found["train_classes"][n]]
    if unseen:
        raise SystemExit(f"No boxes in train for {', '.join(unseen)} (all in other splits): the model can't learn them. "
                         "Re-split (scripts/split_by_scene.py) so each class has training boxes")
    if found["exam_frames"]:
        raise SystemExit(f"{len(found['exam_frames'])} exam frames in the training data (e.g. "
                         f"{found['exam_frames'][0].name}): take them out of this dataset version first")
    settings = {**RECIPE, "epochs": args.epochs, "batch": args.batch}
    print(f"Start from {args.start}; " + ", ".join(f"{k} {v}" for k, v in settings.items()))
    if args.dry_run:
        return
    from ultralytics import YOLO
    model = YOLO(str(args.start))
    model.train(data=str(args.data.resolve()), project=str(config.REPO / "outputs" / "detector"), name=args.name,
                **settings)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"      # ultralytics adds a number if <name> exists
    print(f"\nWeights: {best}\nScore it: python scripts/detector_eval.py --exam-dir EXAM --weights {best} "
          f"--against {config.TILE_DETECTOR}")


if __name__ == "__main__":
    main()
