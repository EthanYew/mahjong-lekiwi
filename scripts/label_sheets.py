"""One picture per tile name, of every box labelled with it, so a person can check a labelled set by eye (no GPU, no arm).

    python scripts/label_sheets.py data/detector/exam_20261104_mat_all
    python scripts/label_sheets.py data/detector/exam_20261104_mat_all --names 2C,3C
    xdg-open data/detector/exam_20261104_mat_all_sheets/2C.png

The folder is a YOLO export (data.yaml, <split>/images, <split>/labels), from mine_frames.py or a Roboflow download. A wrong
label is the odd one out on its name's sheet. Each crop is captioned with its session's time and scan, and lists.txt beside the
sheets says which file that is: fix it in Roboflow, then make a new version.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import yaml

from mahjong import detector

HALF_PX, ZOOM, COLUMNS = 20, 4, 10


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("folder", type=Path, help="a YOLO export: data.yaml, <split>/images, <split>/labels")
    p.add_argument("--names", default=None, help="only these tile names, e.g. 2C,3C (default: every name labelled)")
    p.add_argument("--out", type=Path, default=None, help="where the sheets go (default: <folder>_sheets)")
    return p.parse_args(argv)


def caption(stem):
    """<session>__<scan>__f<i> as "HHMMSS scan"; anything else (July's frames) as its start."""
    parts = detector.original_name(stem + ".jpg").split(detector.SEP)
    return f"{parts[0][9:15]} {parts[1][:3]}" if len(parts) == 3 else stem[:12]


def cell(image, box, text):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box["xyxy"]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    x0, y0 = min(max(cx - HALF_PX, 0), w - 2 * HALF_PX), min(max(cy - HALF_PX, 0), h - 2 * HALF_PX)
    out = cv2.resize(image[y0:y0 + 2 * HALF_PX, x0:x0 + 2 * HALF_PX], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    cv2.putText(out, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def gather(folder):
    """{name: [(image path, box)]} over every split of the export."""
    names = yaml.safe_load((folder / "data.yaml").read_text())["names"]
    found = {}
    for labels in sorted(folder.glob("*/labels")):
        for path in sorted(labels.glob("*.txt")):
            image = labels.parent / "images" / f"{path.stem}.jpg"
            if not image.exists():
                continue
            h, w = cv2.imread(str(image)).shape[:2]
            for box in detector.read_yolo_labels(path, names, (w, h)):
                found.setdefault(box["tile_name"], []).append((image, box))
    return found


def sheet(rows):
    cells = [cell(cv2.imread(str(image)), box, caption(image.stem)) for image, box in rows]
    cells += [np.zeros_like(cells[0])] * (-len(cells) % COLUMNS)
    return np.vstack([np.hstack(cells[i:i + COLUMNS]) for i in range(0, len(cells), COLUMNS)])


def main(argv=None):
    args = parse_args(argv)
    if not (args.folder / "data.yaml").exists():
        raise SystemExit(f"No data.yaml in {args.folder}: name a YOLO export folder")
    out = args.out or args.folder.with_name(f"{args.folder.name}_sheets")
    out.mkdir(parents=True, exist_ok=True)
    found = gather(args.folder)
    wanted = sorted(found) if not args.names else [n.strip().upper() for n in args.names.split(",")]
    lines = []
    for name in wanted:
        rows = found.get(name)
        if not rows:
            print(f"  {name}: no labels")
            continue
        cv2.imwrite(str(out / f"{name}.png"), sheet(rows))
        lines += [f"{name}\t{caption(image.stem)}\t{image.name}" for image, _ in rows]
        print(f"  {name}: {len(rows)}")
    (out / "lists.txt").write_text("\n".join(lines) + "\n")
    print(f"Sheets in {out}. Look for the odd one out on each; lists.txt names each crop's file.")
    return out


if __name__ == "__main__":
    main()
