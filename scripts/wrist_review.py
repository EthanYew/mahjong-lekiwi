"""Check the wrist photos of each tile name, so a frame of a misnamed tile doesn't train the name check.

    python scripts/wrist_review.py                 # every name, photos not yet reviewed
    python scripts/wrist_review.py --name 3C       # one name
    python scripts/wrist_review.py --all           # reviewed ones too
    python scripts/wrist_review.py --source bench  # only bench attempts' hover frames

Photos come from wrist_capture.py's sessions and bench attempts' first frames (mahjong.wrist_photos).
One name at a time, one row per physical tile (all its views: a misdetected tile is wrong in all of
them). Click a photo to mark it wrong; click the row's label for the whole tile. Right-click renames
(asked in the terminal): a 2C filed as 3C becomes a 2C photo, not a lost one. Shift+click opens a photo
at full size, to see which tile is between the jaws (several are in view). Space marks the page's
unmarked photos ok and moves on; p goes back; ] and [ change name; q quits.

Nothing is moved or deleted: each verdict is a line in data/wrist_review.jsonl (push it with push_session_data.py, which the
window's end reminds you of), and the counts (wrist_capture.py) and any training set read the photos through it. Run from the
main checkout (sessions are per checkout). The window and the verdict log are mahjong/photo_review.py, shared with
base_cam_review.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, detector, photo_review, wrist_photos

WINDOW = "wrist review"
THUMB_W, THUMB_H, GAP = photo_review.Reviewer.THUMB_W, photo_review.Reviewer.THUMB_H, photo_review.Reviewer.GAP
LABEL_W, HEADER_H = photo_review.Reviewer.LABEL_W, photo_review.Reviewer.HEADER_H
COLUMNS, ROWS = photo_review.Reviewer.COLUMNS, photo_review.Reviewer.ROWS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", default=None, help="start at (or only show, with --only) this tile name")
    p.add_argument("--only", action="store_true", help="with --name: just that name")
    p.add_argument("--all", action="store_true", help="show reviewed photos too (default: groups with any unreviewed)")
    p.add_argument("--source", choices=("all",) + wrist_photos.SOURCES, default="all", help="which photos (default all)")
    return p.parse_args(argv)


class Reviewer(photo_review.Reviewer):
    """One row per physical tile (all its views: a misdetected tile is wrong in all of them). A bench attempt is one
    photo, so a bench session's attempts share rows (each still marked on its own)."""

    def __init__(self, all_photos, review, names, show_all=False, log=None):
        super().__init__(all_photos, review, names, log or wrist_photos.REVIEW, show_all)

    def group_of(self, photo):
        return photo["session"] if photo["source"] == "bench" else photo["group"]

    def caption(self, row):
        first = row[0]
        detail = f"{len(row)} attempts" if first["source"] == "bench" else f"tile {first['group'].split('/')[1]}"
        return [first["group"].split("/")[0][4:15], detail, f"{first['source']} as {first['name']}"]


def main(argv=None, ask=None):
    args = parse_args(argv)
    from wrist_capture import ask_pumping
    ask = ask_pumping(ask or input)
    all_photos = [p for p in wrist_photos.photos() if args.source in ("all", p["source"])]
    names = photo_review.name_order(detector.TILE_NAMES, args.name, args.only)
    reviewer = Reviewer(all_photos, wrist_photos.load_review(), names, show_all=args.all)
    if not reviewer.names:
        print(f"Nothing to review in {config.SESSIONS}" + ("" if args.all else " (--all shows reviewed photos too)"))
        photo_review.remind_to_push(wrist_photos.REVIEW)
        return
    print(f"{len(all_photos)} photos; reviewing {', '.join(reviewer.names)}")
    try:
        photo_review.run(reviewer, ask, WINDOW, detector.TILE_NAMES)
        print(photo_review.summary(wrist_photos.load_review(), wrist_photos.REVIEW))
    finally:
        photo_review.remind_to_push(wrist_photos.REVIEW)


if __name__ == "__main__":
    main()
