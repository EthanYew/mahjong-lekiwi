"""A Roboflow training export re-split so that no scene is in both train and valid, ready for train_detector.py.

    python scripts/split_by_scene.py --data "data/detector/v4/data.yaml" --out data/detector/train_v5 --list
    python scripts/split_by_scene.py --data "data/detector/v4/data.yaml" --out data/detector/train_v5

Roboflow splits frame by frame. The July frames were each shot at 4 exposures, so the same tiles in the
same places landed in train and valid (134 of v4's 182 valid frames had a train twin): the valid score
rose while the detector got worse on the exam, and best.pt was picked on it. This pools every split of
the export, groups frames into scenes by their labels (detector.scenes), and deals whole scenes to valid
until it holds --valid of the frames, apart for the July frames and the frames from sessions so both are
in valid. Roboflow's test split goes into the pool: the exam is the detector's test. Frames with tile backs are
a third group, one scene per capture session (frame_scenes): by labels every face-down frame on a layout was one scene.

On the way, labels are made into what train_detector expects: only the 38 tile names (aruco_marker
and other classes dropped, so the head still matches the live weights; --with-backs keeps `back`, tiles lying
green side up, as a 39th class after them), polygons as their boxes, and a
label drawn twice on one tile merged. A frame with two names on one tile is left out and listed, to fix
in Roboflow; so is any exam frame.
"""
import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, detector


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True, help="the Roboflow export's data.yaml")
    p.add_argument("--out", type=Path, required=True, help="new export folder (must not exist)")
    p.add_argument("--valid", type=float, default=0.2, help="share of frames in valid (default 0.2)")
    p.add_argument("--share", type=float, default=0.6,
                   help="frames sharing this much of their tiles both ways are one scene (default 0.6)")
    p.add_argument("--seed", type=int, default=0, help="which scenes go to valid (default 0)")
    p.add_argument("--list", action="store_true", help="print the split, write nothing")
    p.add_argument("--with-backs", action="store_true",
                   help="keep the 'back' class (tiles lying green side up) as a 39th, after the 38 tile names: the head then "
                        "has one more class than the live weights, which train_detector's start carries over all but the head")
    return p.parse_args(argv)


def class_names(with_backs=False):
    """The classes train_detector gets: the 38 tile names, and with_backs the back class (config.BACK) last, so every tile
    name keeps its index."""
    return detector.TILE_NAMES + [config.BACK] if with_backs else detector.TILE_NAMES


def load(data_yaml, with_backs=False):
    """[{"image", "boxes", "split"}] for every labelled image of the export, plus what was set aside."""
    keep = set(class_names(with_backs))
    import cv2
    root = Path(data_yaml).parent
    names = yaml.safe_load(Path(data_yaml).read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    frames, notes = [], Counter()
    for image in sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                        and p.parent.name == "images"):
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        h, w = cv2.imread(str(image)).shape[:2]
        boxes = detector.read_yolo_labels(label, names, (w, h)) if label.exists() else []
        tiles = [b for b in boxes if b["tile_name"] in keep]
        notes["other boxes dropped"] += len(boxes) - len(tiles)
        tiles, twins = detector.dedupe(tiles)
        notes["labels drawn twice merged"] += len(twins)
        frames.append({"image": image, "boxes": tiles, "size": (w, h), "split": image.parent.parent.name})
    return frames, notes


def two_names_on_a_tile(boxes):
    return [(a["tile_name"], b["tile_name"]) for i, a in enumerate(boxes) for b in boxes[:i]
            if detector.iou(a["xyxy"], b["xyxy"]) >= 0.5]


def back_sessions(frames):
    """The capture sessions with any frame labelled with a back: all their frames are face-down frames."""
    return {detector.session_of(f["image"].name) for f in frames if has_back(f)} - {None}


def has_back(frame):
    return any(b["tile_name"] == config.BACK for b in frame["boxes"])


def frame_scenes(frames, share=0.6):
    """A scene for each frame: one per capture session for the sessions with backs (back_sessions), by labels
    (detector.scenes) for the rest. Every face-down frame has a back on each of the layout's spots, and the few tiles left
    face up sat on the same spots from session to session, so by labels they all made one scene: on 2026-11-16 all 1440
    back boxes went to valid with it, and best.pt found no backs. By session, each session's light and surface sit on one
    side of the split; the same spots are still in both, so valid flatters backs a little, and the exam (sessions of
    their own) is what judges them. A back frame with no session in its name is a scene of its own."""
    scenes, sessions = detector.scenes([f["boxes"] for f in frames], share), back_sessions(frames)
    out = []
    for f, scene in zip(frames, scenes):
        session = detector.session_of(f["image"].name)
        out.append(("backs", session) if session in sessions else ("frame", f["image"].name) if has_back(f)
                   else ("labels", str(scene)))
    return out


OVERFILL = 1.25     # a scene that takes valid past this much of its share is left in train, for a smaller one


ORIGINS = ("July frames", "session frames", "face-down frames")


def origin_of(frame, sessions=()):
    """Which group a frame is dealt in (sessions: back_sessions): each gives ~`valid` of its frames to valid, so backs are
    in both splits."""
    if has_back(frame) or detector.session_of(frame["image"].name) in sessions:
        return ORIGINS[2]
    return ORIGINS[0] if detector.session_of(frame["image"].name) is None else ORIGINS[1]


def split(frames, valid=0.2, share=0.6, seed=0):
    """Sets each frame's "scene" and "to" (train/valid); whole scenes, ~`valid` of the frames of each origin."""
    for f, scene in zip(frames, frame_scenes(frames, share)):
        f["scene"], f["to"] = scene, "train"
    sessions = back_sessions(frames)
    for origin in ORIGINS:
        group = [f for f in frames if origin_of(f, sessions) == origin]
        by_scene = {}
        for f in group:
            by_scene.setdefault(f["scene"], []).append(f)
        order = sorted(by_scene)
        random.Random(seed).shuffle(order)
        taken, target = 0, valid * len(group)
        for scene in order:
            if taken >= target:
                break
            if taken + len(by_scene[scene]) > target * OVERFILL:
                continue    # a scene this big would leave train short (69 bench frames took 42% of valid, 2026-11-16);
                            # with one face-down session, every back stays in train
            for f in by_scene[scene]:
                f["to"] = "valid"
            taken += len(by_scene[scene])
        if not taken and len(by_scene) > 1:     # every scene too big (a few face-down sessions): the smallest, so valid
            for f in min(by_scene.values(), key=len):   # still has some of this group
                f["to"] = "valid"
    return frames


def write(frames, out, with_backs=False):
    names = class_names(with_backs)
    for f in frames:
        images, labels = out / f["to"] / "images", out / f["to"] / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        shutil.copy(f["image"], images / f["image"].name)
        lines = detector.yolo_lines(f["boxes"], names, f["size"])
        (labels / f"{f['image'].stem}.txt").write_text("".join(line + "\n" for line in lines))
    (out / "data.yaml").write_text(yaml.safe_dump({"train": "../train/images", "val": "../valid/images",
                                                   "nc": len(names), "names": names}, sort_keys=False))


def main(argv=None):
    args = parse_args(argv)
    if args.out.exists() and not args.list:
        raise SystemExit(f"{args.out} exists: pick a new folder")
    frames, notes = load(args.data, args.with_backs)
    exam = detector.exam_frames([f["image"] for f in frames])
    clashes = {f["image"].name: two_names_on_a_tile(f["boxes"]) for f in frames}
    kept = [f for f in frames if f["image"] not in exam and not clashes[f["image"].name]]
    split(kept, args.valid, args.share, args.seed)

    print(f"{args.data}: {len(frames)} frames, {len({f['scene'] for f in kept})} scenes")
    for note, n in notes.items():
        if n:
            print(f"  {note}: {n}")
    for f in frames:
        if f["image"] in exam:
            print(f"  left out, exam frame: {f['image'].name}")
        elif clashes[f["image"].name]:
            print(f"  left out, two names on one tile {clashes[f['image'].name]} (fix in Roboflow): {f['image'].name}")
    sessions = back_sessions(kept)
    for origin in ORIGINS:
        group = [f for f in kept if origin_of(f, sessions) == origin]
        if not group:
            continue
        counts = Counter(f["to"] for f in group)
        scenes = {to: len({f["scene"] for f in group if f["to"] == to}) for to in ("train", "valid")}
        print(f"  {origin:<16} train {counts['train']:>4} frames ({scenes['train']} scenes), "
              f"valid {counts['valid']:>4} frames ({scenes['valid']} scenes)")
    moved = Counter((f["split"], f["to"]) for f in kept)
    print("  from Roboflow's split: " + ", ".join(f"{a}->{b} {n}" for (a, b), n in sorted(moved.items())))
    if args.list:
        return
    write(kept, args.out, args.with_backs)
    print(f"Written to {args.out}. Next: python scripts/train_detector.py --data {args.out / 'data.yaml'} --name NAME --dry-run")


if __name__ == "__main__":
    main()
