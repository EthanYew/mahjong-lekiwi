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

Nothing is moved or deleted: each verdict is a line in data/wrist_review.jsonl (commit it), and the
counts (wrist_capture.py) and any training set read the photos through it. Run from the main checkout
(sessions are per checkout).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import config, detector, wrist_photos

WINDOW = "wrist review"
THUMB_W, THUMB_H, GAP = 192, 144, 4
LABEL_W, HEADER_H = 170, 44
COLUMNS, ROWS = 6, 4
GREEN, RED, AMBER, WHITE, GREY = (0, 200, 0), (40, 40, 230), (0, 190, 255), (235, 235, 235), (150, 150, 150)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", default=None, help="start at (or only show, with --only) this tile name")
    p.add_argument("--only", action="store_true", help="with --name: just that name")
    p.add_argument("--all", action="store_true", help="show reviewed photos too (default: groups with any unreviewed)")
    p.add_argument("--source", choices=("all",) + wrist_photos.SOURCES, default="all", help="which photos (default all)")
    return p.parse_args(argv)


class Reviewer:
    """The pages of one name at a time and the verdicts given on them. Pure logic: tests drive it with
    clicks and keys; main() draws it."""

    def __init__(self, all_photos, review, names, show_all=False, log=None):
        self.photos, self.review, self.show_all = all_photos, review, show_all
        self.log = log or wrist_photos.REVIEW
        # hidden: tiles reviewed before this run. Ones reviewed now stay, so p / [ go back to them (they
        # vanished, leaving a blank page, when hidden by the live review)
        self.reviewed_before = set(review)
        self.names = [n for n in names if self._rows(n)]
        self.name_i, self.page = 0, 0
        self.rows = self._rows(self.names[0]) if self.names else []

    def _rows(self, name):
        """Rows for a name: its photos (as named now) grouped by physical tile, COLUMNS a row. A bench
        attempt is one photo, so a bench session's attempts share rows (each still marked on its own)."""
        groups = {}
        for photo in self.photos:
            if wrist_photos.name_of(photo, self.review) == name:
                key = photo["session"] if photo["source"] == "bench" else photo["group"]
                groups.setdefault(key, []).append(photo)
        rows = []
        for group, members in groups.items():
            if not self.show_all and all(m["key"] in self.reviewed_before for m in members):
                continue
            rows += [members[i:i + COLUMNS] for i in range(0, len(members), COLUMNS)]
        return rows

    @property
    def name(self):
        return self.names[self.name_i] if self.names else None

    def page_rows(self):
        return self.rows[self.page * ROWS:(self.page + 1) * ROWS]

    def pages(self):
        return max(1, -(-len(self.rows) // ROWS))

    def verdict(self, photo):
        return self.review.get(photo["key"])

    def mark(self, photos, verdict, name=None):
        if verdict == "wrong" and name == self.name:      # renamed to what it was already filed as: it's fine
            verdict, name = "ok", None
        for r in wrist_photos.record([p["key"] for p in photos], verdict, name, self.log):
            self.review[r["photo"]] = r

    def hit(self, x, y):
        """(row index on the page, column or None for the label) under a click, or None."""
        if y < HEADER_H:
            return None
        row = (y - HEADER_H) // (THUMB_H + GAP)
        if row >= len(self.page_rows()):
            return None
        if x < LABEL_W:
            return row, None
        col = (x - LABEL_W) // (THUMB_W + GAP)
        return (row, col) if col < len(self.page_rows()[row]) else None

    def photo_at(self, x, y):
        target = self.hit(x, y)
        return None if target is None or target[1] is None else self.page_rows()[target[0]][target[1]]

    def click(self, x, y):
        """Toggle wrong / ok for the photo clicked, or for the whole row when its label is clicked."""
        target = self.hit(x, y)
        if target is None:
            return
        row, col = target
        photos = self.page_rows()[row] if col is None else [self.page_rows()[row][col]]
        is_wrong = all((self.verdict(p) or {}).get("verdict") == "wrong" for p in photos)
        self.mark(photos, "ok" if is_wrong else "wrong")

    def rename(self, x, y, new_name):
        """Mark the clicked photo (or the row, at its label) as wrong, with its right name."""
        target = self.hit(x, y)
        if target is None:
            return
        row, col = target
        self.mark(self.page_rows()[row] if col is None else [self.page_rows()[row][col]], "wrong", new_name)

    def next_page(self):
        """Every unmarked photo on this page is ok; then the next page (or name). False at the very end."""
        unmarked = [p for row in self.page_rows() for p in row if p["key"] not in self.review]
        if unmarked:
            self.mark(unmarked, "ok")
        if self.page + 1 < self.pages():
            self.page += 1
            return True
        return self.change_name(+1)

    def previous_page(self):
        if self.page > 0:
            self.page -= 1
        elif self.change_name(-1):
            self.page = self.pages() - 1

    def change_name(self, step):
        """To the next (step +1) or previous (-1) name with photos to show (all of one's photos may have
        been renamed away). False, staying put, if there's none that way."""
        i = self.name_i + step
        while 0 <= i < len(self.names):
            rows = self._rows(self.names[i])
            if rows:
                self.name_i, self.rows, self.page = i, rows, 0
                return True
            i += step
        return False

    def render(self, load=lambda path: cv2.imread(str(path))):
        width = LABEL_W + COLUMNS * (THUMB_W + GAP)
        canvas = np.full((HEADER_H + ROWS * (THUMB_H + GAP), width, 3), 30, np.uint8)
        wrong = sum((self.verdict(p) or {}).get("verdict") == "wrong" for row in self.rows for p in row)
        cv2.putText(canvas, f"{self.name}  page {self.page + 1}/{self.pages()}  ({sum(map(len, self.rows))} photos, "
                    f"{wrong} wrong)  name {self.name_i + 1}/{len(self.names)}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        cv2.putText(canvas, "click: wrong   click label: whole tile   right-click: rename   shift-click: full size   space: ok + next   "
                    "p: back   ] [: name   q: quit", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)
        if not self.page_rows():
            cv2.putText(canvas, f"No {self.name} photos left to show here (renamed away?): ] or [ for another name",
                        (8, HEADER_H + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER, 1)
        for r, row in enumerate(self.page_rows()):
            y = HEADER_H + r * (THUMB_H + GAP)
            first = row[0]
            cv2.putText(canvas, first["group"].split("/")[0][4:15], (6, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
            detail = f"{len(row)} attempts" if first["source"] == "bench" else f"tile {first['group'].split('/')[1]}"
            cv2.putText(canvas, detail, (6, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
            cv2.putText(canvas, f"{first['source']} as {first['name']}", (6, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        GREY, 1)
            for c, photo in enumerate(row):
                x = LABEL_W + c * (THUMB_W + GAP)
                image = load(photo["path"])
                if image is not None:
                    canvas[y:y + THUMB_H, x:x + THUMB_W] = cv2.resize(image, (THUMB_W, THUMB_H))
                v = self.verdict(photo)
                if v and v["verdict"] == "wrong":
                    cv2.rectangle(canvas, (x, y), (x + THUMB_W - 1, y + THUMB_H - 1), RED, 3)
                    cv2.line(canvas, (x, y), (x + THUMB_W - 1, y + THUMB_H - 1), RED, 2)
                    if v.get("name"):
                        cv2.putText(canvas, f"-> {v['name']}", (x + 4, y + THUMB_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, AMBER, 2)
                elif v:
                    cv2.circle(canvas, (x + THUMB_W - 9, y + 9), 5, GREEN, -1)
        return canvas


def main(argv=None, ask=None):
    args = parse_args(argv)
    from wrist_capture import ask_pumping
    ask = ask_pumping(ask or input)
    all_photos = [p for p in wrist_photos.photos() if args.source in ("all", p["source"])]
    names = list(detector.TILE_NAMES)
    if args.name:
        name = args.name.upper()
        if name not in names:
            raise SystemExit(f"{name} isn't a tile name")
        names = [name] if args.only else names[names.index(name):] + names[:names.index(name)]
    reviewer = Reviewer(all_photos, wrist_photos.load_review(), names, show_all=args.all)
    if not reviewer.names:
        print(f"Nothing to review in {config.SESSIONS}" + ("" if args.all else " (--all shows reviewed photos too)"))
        return
    print(f"{len(all_photos)} photos; reviewing {', '.join(reviewer.names)}")
    clicks = []
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, lambda event, x, y, flags, _: clicks.append((event, x, y, flags))
                         if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) else None)
    thumbs = {}

    def thumb(path):           # small copies, read once: hundreds of full frames would be ~1 MB each
        if path not in thumbs:
            image = cv2.imread(str(path))
            thumbs[path] = None if image is None else cv2.resize(image, (THUMB_W, THUMB_H))
        return thumbs[path]
    try:
        while True:
            cv2.imshow(WINDOW, reviewer.render(thumb))
            key = cv2.waitKey(30) & 0xFF
            while clicks:
                event, x, y, flags = clicks.pop(0)
                if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY:
                    photo = reviewer.photo_at(x, y)
                    if photo is not None:      # full size, to see which tile is between the jaws
                        cv2.imshow("wrist photo", cv2.imread(str(photo["path"])))
                elif event == cv2.EVENT_LBUTTONDOWN:
                    reviewer.click(x, y)
                elif reviewer.hit(x, y) is not None:
                    answer = ask("Right name (e.g. 2C; Enter: just wrong, not a usable tile photo): ").strip().upper()
                    if answer and answer not in detector.TILE_NAMES:
                        print(f"{answer} isn't a tile name")
                        continue
                    reviewer.rename(x, y, answer or None)
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("n")) and not reviewer.next_page():
                print("Every name reviewed.")
                break
            if key == ord("p"):
                reviewer.previous_page()
            if key in (ord("]"), ord("[")):
                reviewer.change_name(+1 if key == ord("]") else -1)
    finally:
        cv2.destroyAllWindows()
    review = wrist_photos.load_review()
    wrong = [r for r in review.values() if r["verdict"] == "wrong"]
    print(f"{len(review)} photos reviewed, {len(wrong)} wrong ({sum(bool(r.get('name')) for r in wrong)} renamed). "
          f"Verdicts in {wrist_photos.REVIEW}: commit it.")


if __name__ == "__main__":
    main()
