"""A person's verdicts on photos filed under a tile name, and the window they give them in. Shared by the wrist camera's
photos (mahjong/wrist_photos.py, scripts/wrist_review.py) and the base camera's hold-up photos (mahjong/base_cam.py,
scripts/base_cam_review.py).

A photo is a dict with at least "key" (unique, stable: its path relative to the sessions folder), "path" and "name" (what it
was filed as). Nothing is moved or deleted: each verdict is a line in a JSONL log (the last line for a photo wins), "ok", or
"wrong" with the right name, or "wrong" with none (not a usable photo of any tile). Counts and training sets read the photos
through `labelled`.

A run with show_all (--all) shows the photos marked wrong with no name too, under the name they were filed as.

The window shows one name at a time, a row per group (what a person gets wrong together: one physical tile for the wrist
camera, one tile laid on one spot for the base camera). Click a photo to mark it wrong; click the row's label for the whole
row. Right-click renames (asked in the terminal). Shift+click opens a photo at full size. Space marks the page's unmarked
photos ok and moves on; p goes back; ] and [ change name; q quits.
"""
from datetime import datetime
from pathlib import Path

from . import checkout, config
from .session import append_jsonl, read_jsonl

GREEN, RED, AMBER, WHITE, GREY = (0, 200, 0), (40, 40, 230), (0, 190, 255), (235, 235, 235), (150, 150, 150)
HELP = ("click: wrong   click label: whole row   right-click: rename   shift-click: full size   space: ok + next   "
        "p: back   ] [: name   q: quit")


# --- the verdict log ---

def load_review(path):
    """{photo key: its latest verdict record}."""
    return {record["photo"]: record for record in read_jsonl(path)}


def record(keys, verdict, name, path):
    """Append a verdict ("ok" / "wrong", with the right name if known) for each photo key. Returns the records."""
    time = datetime.now().isoformat(timespec="seconds")
    records = [{"photo": key, "verdict": verdict, "name": name, "time": time} for key in keys]
    append_jsonl(path, records)
    return records


def name_of(photo, review):
    """The photo's name after review: the corrected one, its own, or None (wrong, no right name)."""
    verdict = review.get(photo["key"])
    if verdict is None or verdict["verdict"] == "ok":   # a photo no scan named waits for a person's name
        return None if photo["name"] == config.UNNAMED else photo["name"]
    return verdict.get("name")


def labelled(all_photos, review):
    """The usable photos, each with "name" as reviewed (photos marked wrong without a name left out)."""
    out = []
    for photo in all_photos:
        name = name_of(photo, review)
        if name is not None:
            out.append({**photo, "name": name, "reviewed": photo["key"] in review})
    return out


# --- the window ---

def name_order(names, start=None, only=False):
    """`names` from `start` round (or just `start`, with `only`); SystemExit for a name not among them."""
    names = list(names)
    if not start:
        return names
    start = start.upper()
    if start not in names:
        raise SystemExit(f"{start} isn't a tile name")
    return [start] if only else names[names.index(start):] + names[:names.index(start)]


class Reviewer:
    """The pages of one name at a time and the verdicts given on them. Pure logic: tests drive it with clicks and keys;
    `run` draws it. Subclasses say how photos group into rows (`group_of`), what a row's label says (`caption`) and what a
    cell shows (`image_of`)."""
    COLUMNS, ROWS = 6, 4
    THUMB_W, THUMB_H, GAP = 192, 144, 4
    LABEL_W, HEADER_H = 170, 44

    def __init__(self, all_photos, review, names, log, show_all=False):
        self.photos, self.review, self.show_all, self.log = all_photos, review, show_all, log
        # hidden: groups reviewed before this run. Ones reviewed now stay, so p / [ go back to them (they vanished,
        # leaving a blank page, when hidden by the live review)
        self.reviewed_before = set(review)
        self.names = [n for n in names if self._rows(n)]
        self.name_i, self.page = 0, 0
        self.rows = self._rows(self.names[0]) if self.names else []

    # what a subclass changes
    def group_of(self, photo):
        return photo["group"]

    def caption(self, row):
        """Lines of text for a row's label."""
        return [row[0]["group"]]

    def image_of(self, photo):
        import cv2
        return cv2.imread(str(photo["path"]))

    def shown_under(self, photo):
        """The name whose pages show the photo: its name as reviewed. With show_all, one marked wrong with no right name stays
        under the name it was filed as, or no later run could show it to be named."""
        name = name_of(photo, self.review)
        return photo["name"] if name is None and self.show_all and photo["key"] in self.review else name

    def _rows(self, name):
        """Rows for a name: its photos (as named now) by group, COLUMNS a row."""
        groups = {}
        for photo in self.photos:
            if self.shown_under(photo) == name:
                groups.setdefault(self.group_of(photo), []).append(photo)
        rows = []
        for members in groups.values():
            if not self.show_all and all(m["key"] in self.reviewed_before for m in members):
                continue
            rows += [members[i:i + self.COLUMNS] for i in range(0, len(members), self.COLUMNS)]
        return rows

    @property
    def name(self):
        return self.names[self.name_i] if self.names else None

    def page_rows(self):
        return self.rows[self.page * self.ROWS:(self.page + 1) * self.ROWS]

    def pages(self):
        return max(1, -(-len(self.rows) // self.ROWS))

    def verdict(self, photo):
        return self.review.get(photo["key"])

    def mark(self, photos, verdict, name=None):
        if verdict == "wrong" and name == self.name:      # renamed to what it was already filed as: it's fine
            verdict, name = "ok", None
        for r in record([p["key"] for p in photos], verdict, name, self.log):
            self.review[r["photo"]] = r

    def hit(self, x, y):
        """(row index on the page, column or None for the label) under a click, or None."""
        if y < self.HEADER_H:
            return None
        row = (y - self.HEADER_H) // (self.THUMB_H + self.GAP)
        if row >= len(self.page_rows()):
            return None
        if x < self.LABEL_W:
            return row, None
        col = (x - self.LABEL_W) // (self.THUMB_W + self.GAP)
        return (row, col) if col < len(self.page_rows()[row]) else None

    def photo_at(self, x, y):
        target = self.hit(x, y)
        return None if target is None or target[1] is None else self.page_rows()[target[0]][target[1]]

    def _target(self, x, y):
        target = self.hit(x, y)
        if target is None:
            return None
        row, col = target
        return self.page_rows()[row] if col is None else [self.page_rows()[row][col]]

    def click(self, x, y):
        """Toggle wrong / ok for the photo clicked, or for the whole row when its label is clicked."""
        photos = self._target(x, y)
        if photos:
            is_wrong = all((self.verdict(p) or {}).get("verdict") == "wrong" for p in photos)
            self.mark(photos, "ok" if is_wrong else "wrong")

    def rename(self, x, y, new_name):
        """Mark the clicked photo (or the row, at its label) as wrong, with its right name."""
        photos = self._target(x, y)
        if photos:
            self.mark(photos, "wrong", new_name)

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
        """To the next (step +1) or previous (-1) name with photos to show (all of one's photos may have been renamed
        away). False, staying put, if there's none that way."""
        i = self.name_i + step
        while 0 <= i < len(self.names):
            rows = self._rows(self.names[i])
            if rows:
                self.name_i, self.rows, self.page = i, rows, 0
                return True
            i += step
        return False

    def render(self, load=None):
        """The page as an image. `load(photo)` gives a cell's image (default `image_of`; `run` passes a cached one)."""
        import cv2
        import numpy as np
        load = load or self.image_of
        width = self.LABEL_W + self.COLUMNS * (self.THUMB_W + self.GAP)
        canvas = np.full((self.HEADER_H + self.ROWS * (self.THUMB_H + self.GAP), width, 3), 30, np.uint8)
        wrong = sum((self.verdict(p) or {}).get("verdict") == "wrong" for row in self.rows for p in row)
        cv2.putText(canvas, f"{self.name}  page {self.page + 1}/{self.pages()}  ({sum(map(len, self.rows))} photos, "
                    f"{wrong} wrong)  name {self.name_i + 1}/{len(self.names)}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        cv2.putText(canvas, HELP, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)
        if not self.page_rows():
            cv2.putText(canvas, f"No {self.name} photos left to show here (renamed away?): ] or [ for another name",
                        (8, self.HEADER_H + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER, 1)
        for r, row in enumerate(self.page_rows()):
            y = self.HEADER_H + r * (self.THUMB_H + self.GAP)
            for i, line in enumerate(self.caption(row)):
                cv2.putText(canvas, line, (6, y + 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE if i < 2 else GREY, 1)
            for c, photo in enumerate(row):
                x = self.LABEL_W + c * (self.THUMB_W + self.GAP)
                image = load(photo)
                if image is not None:
                    canvas[y:y + self.THUMB_H, x:x + self.THUMB_W] = cv2.resize(image, (self.THUMB_W, self.THUMB_H))
                v = self.verdict(photo)
                if v and v["verdict"] == "wrong":
                    cv2.rectangle(canvas, (x, y), (x + self.THUMB_W - 1, y + self.THUMB_H - 1), RED, 3)
                    cv2.line(canvas, (x, y), (x + self.THUMB_W - 1, y + self.THUMB_H - 1), RED, 2)
                    if v.get("name"):
                        cv2.putText(canvas, f"-> {v['name']}", (x + 4, y + self.THUMB_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, AMBER, 2)
                elif v:
                    cv2.circle(canvas, (x + self.THUMB_W - 9, y + 9), 5, GREEN, -1)
        return canvas


def run(reviewer, ask, window, names):
    """The window: draws `reviewer` and turns clicks and keys into verdicts until q, Esc or the last page. `ask` is
    already pumping the window (wrist_capture.ask_pumping); `names` are the names a rename may give."""
    import cv2
    clicks = []
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, lambda event, x, y, flags, _: clicks.append((event, x, y, flags))
                         if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) else None)
    thumbs = {}

    def thumb(photo):          # small copies, made once: hundreds of full frames would be ~1 MB each
        if photo["key"] not in thumbs:
            image = reviewer.image_of(photo)
            thumbs[photo["key"]] = None if image is None else cv2.resize(image, (reviewer.THUMB_W, reviewer.THUMB_H))
        return thumbs[photo["key"]]
    try:
        while True:
            cv2.imshow(window, reviewer.render(thumb))
            key = cv2.waitKey(30) & 0xFF
            while clicks:
                event, x, y, flags = clicks.pop(0)
                if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY:
                    photo = reviewer.photo_at(x, y)
                    if photo is not None:      # full size, to see the whole frame
                        cv2.imshow(f"{window}: full size", cv2.imread(str(photo["path"])))
                elif event == cv2.EVENT_LBUTTONDOWN:
                    reviewer.click(x, y)
                elif reviewer.hit(x, y) is not None:
                    answer = ask("Right name (e.g. 2C; Enter: just wrong, not a usable tile photo): ").strip().upper()
                    if answer and answer not in names:
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


def summary(review, path):
    wrong = [r for r in review.values() if r["verdict"] == "wrong"]
    return (f"{len(review)} photos reviewed, {len(wrong)} wrong ({sum(bool(r.get('name')) for r in wrong)} renamed). "
            f"Verdicts in {path}.")


def remind_to_push(log):
    """Say so while the review log has verdicts that aren't on GitHub: left on this machine, they're easy to forget. In a
    worktree the push script can't take them (it runs from the main checkout), and removing the worktree would lose them."""
    name = Path(log).name
    if any(Path(f).name == name for f in checkout.unsaved_samples()):
        if checkout.is_worktree():
            print(f"\nREMINDER: your verdicts are in this worktree's data/{name}, and removing the worktree loses them. Append the\n"
                  "new lines to the main checkout's copy, then run there: python scripts/push_session_data.py")
        else:
            print("\nREMINDER: your verdicts aren't on GitHub yet. From the main checkout, run:\n"
                  "  python scripts/push_session_data.py")
    elif not checkout.is_worktree() and any(Path(f).name == name for f in checkout.unpushed_files()):
        print("\nREMINDER: your verdicts are committed here but not on GitHub. From the main checkout, run:\n"
              "  git pull --rebase --autostash origin main && git push origin main")
