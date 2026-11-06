"""Blur faces in the base camera's saved photos (a person near the camera ends up in them).

    python scripts/blur_faces.py                          # a dry run over every session: counts, and a sheet of what would change
    python scripts/blur_faces.py --apply                  # rewrite those photos blurred (the originals are not kept)
    python scripts/blur_faces.py sessions/<time>_holdup   # one session (or several)
    python scripts/blur_faces.py --sheet /tmp/faces.jpg   # where the dry run's sheet goes

Covers what the base camera saved: `photos/**` of hold-up sessions, `view_check.jpg`, `dry_run/*_base.jpg` and `captures/*_base.jpg`.
New photos are blurred as they are saved (mahjong/privacy.py); this is for the ones already on disk. The detector is OpenCV's Haar
cascades: it misses faces at an angle, small or cut off by the frame's edge, so look through the sheet for faces it left, and
before sharing anything. A dry run writes nothing but the sheet. Without --apply nothing is changed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import config, privacy

PATTERNS = ("photos/**/*.jpg", "view_check.jpg", "dry_run/*_base.jpg", "captures/*_base.jpg")
THUMB = 200
PER_ROW = 6


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sessions", nargs="*", help="session folders (default: every folder in sessions/)")
    p.add_argument("--apply", action="store_true", help="rewrite the photos with faces blurred (default: a dry run)")
    p.add_argument("--sheet", default=None, help="where the sheet of the photos with a face goes (default: sessions/blur_faces_sheet.jpg)")
    return p.parse_args(argv)


def base_photos(session):
    session = Path(session)
    return sorted({p for pattern in PATTERNS for p in session.glob(pattern)})


def thumb(image):
    scale = THUMB / max(image.shape[:2])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))


def sheet(images, path):
    """The images as thumbnails in a grid, so a person can look for any face that is still showing."""
    if not images:
        return None
    cells = []
    for image in images:
        cell = np.zeros((THUMB, THUMB, 3), np.uint8)
        t = thumb(image)
        cell[:t.shape[0], :t.shape[1]] = t
        cells.append(cell)
    while len(cells) % PER_ROW:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + PER_ROW]) for i in range(0, len(cells), PER_ROW)]
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def main(argv=None, finder=None):
    args = parse_args(argv)
    if finder is None:
        privacy.require()
    sessions = [Path(s) for s in args.sessions] or sorted(p for p in config.SESSIONS.iterdir() if p.is_dir())
    total = hit = 0
    previews = []
    for session in sessions:
        found = 0
        for photo in base_photos(session):
            total += 1
            image = cv2.imread(str(photo))
            if image is None:
                continue
            blurred, boxes = privacy.blur_faces(image, finder)
            if not boxes:
                continue
            found += 1
            hit += 1
            previews.append(blurred)
            if args.apply:
                cv2.imwrite(str(photo), blurred, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if found:
            print(f"{session.name}: {found} photo(s) with a face" + (" blurred" if args.apply else ""))
    print(f"{hit} of {total} photos have a face found" + (", rewritten blurred." if args.apply else ". Nothing was changed (dry run): --apply rewrites them."))
    out = sheet(previews, Path(args.sheet) if args.sheet else config.SESSIONS / "blur_faces_sheet.jpg")
    if out:
        print(f"Sheet of the {len(previews)} photo(s) as they {'are now' if args.apply else 'would be'}: {out}\n"
              "Look through it, and through a few photos it did not flag: the detector misses some faces.")
    return hit, total


if __name__ == "__main__":
    main()
