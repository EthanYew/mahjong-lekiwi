"""How many base camera photos each tile has, and a check that none is filed under the wrong name (no arm, no GPU).

    python scripts/base_cam_review.py --counts         # holds per tile and way up, fewest first; flowers after
    python scripts/base_cam_review.py                  # review every name's photos not yet reviewed
    python scripts/base_cam_review.py --name 3C --only # one name
    python scripts/base_cam_review.py --all            # reviewed ones too, marked-wrong-with-no-name ones among them
    python scripts/base_cam_review.py --play           # the holds face-down play saved: nothing trains on them until reviewed

Holds from play (play_face_down.py) are filed under the name a person confirmed or, unattended, the reader's read (the
caption says which), and wait out of training and the counts until a person gives them a verdict here: Space (ok) or a rename.

Why: the reader (base_cam_train.py) learns each tile from the photos filed under its name, and the name is the one a person
said they laid on the spot (base_cam_collect.py): a tile laid on the wrong spot, or a hold whose tile slipped out of view,
teaches it a wrong answer. Only the interleaved sessions are counted and shown (the earlier ones let the scene name the
tile, and nothing trains on them); --all-sessions shows those too.

--counts: per playable tile, the holds each way up (a hold is one pickup shown at every pose), photos, sessions, holds by
daylight (bright / dim / none), how many photos a person has reviewed, and holds short of --target-holds each way up. A tile
never collected is listed first. Flowers are collected but never played: listed after.

The window: one name at a time, one row per tile laid on one spot in one arrangement (every hold of it there, each pose's
photo cropped to its tile box, as the reader sees it): laying the wrong tile makes the whole row wrong. Click a photo to mark
it wrong; click the row's label for the whole row. Right-click renames (asked in the terminal): a 3D filed as 3B becomes a 3D
photo, not a lost one. Shift+click shows the whole frame. Space marks the page's unmarked photos ok and moves on; p goes
back; ] and [ change name; q quits. A verdict on a pose's photo goes for its other frames too. Way up (upright / turned) isn't
reviewed: the reader learns names only.

Nothing is moved or deleted: each verdict is a line in data/holdup_review.jsonl (push it with push_session_data.py, which the
window's end reminds you of; it merges by union), and
base_cam_train.py and base_cam_collect.py's counts read the photos through it. Run from the main checkout (sessions are per
checkout).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mahjong import base_cam, base_cam_collect, config, detector, photo_review

WINDOW = "base camera review"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--counts", action="store_true", help="print the photos per tile and stop: no window")
    p.add_argument("--target-holds", type=int, default=base_cam_collect.DEFAULT_TARGET_HOLDS,
                   help="with --counts: holds per tile and way up a tile is short of (default %(default)s, as base_cam_collect.py)")
    p.add_argument("--name", default=None, help="start at (or only show, with --only) this tile name")
    p.add_argument("--only", action="store_true", help="with --name: just that name")
    p.add_argument("--all", action="store_true", help="show reviewed photos too (default: rows with any unreviewed)")
    p.add_argument("--all-sessions", action="store_true",
                   help="review the sessions from before interleaving too (they aren't counted or trained on)")
    p.add_argument("--play", action="store_true",
                   help="only the holds from face-down play (play_face_down.py): nothing trains on them until they're reviewed here")
    return p.parse_args(argv)


class Reviewer(photo_review.Reviewer):
    """A row is one tile laid on one spot in one arrangement; a cell is one pose of one hold, cropped to the pose's tile box."""
    COLUMNS, ROWS = 8, 5
    THUMB_W = THUMB_H = 150

    def __init__(self, all_photos, review, names, show_all=False, log=None, boxes=None):
        self.boxes = base_cam.tile_boxes() if boxes is None else boxes
        super().__init__(all_photos, review, names, log or base_cam.REVIEW, show_all)

    def caption(self, row):
        first = row[0]
        where = f"{first['spot']}, laying {first['arrangement']}" if first["arrangement"] is not None else f"shot {first['shot']}"
        holds = sorted({p["hold"] for p in row if p["hold"] is not None})
        by = " (reader's read)" if first.get("filed_by") == "reader" else ""
        return [first["session"][4:15], where, f"filed {first['name']}{by} {first['kind']}",
                "holds " + ",".join(map(str, holds))]

    def image_of(self, photo):
        import cv2
        image = cv2.imread(str(photo["path"]))
        if image is None:
            return None
        box = self.boxes.get(photo["pose"])
        crop = base_cam.tile_frame(image, box) if box else base_cam.whole_frame(image)
        return cv2.resize(crop, (self.THUMB_W, self.THUMB_H))


def count_lines(counts, target, counted, left_out):
    """The --counts table: playable tiles fewest holds first (never collected at the top), then the flowers."""
    from mahjong.name_ledger import FLOWERS
    kinds = base_cam.KINDS

    def line(tile):
        c = counts[tile]
        holds = [c["holds"][k] for k in kinds]
        light = " ".join(f"{k[0]}{n}" for k, n in c["daylight"].items()) or "-"
        reviewed = f"{c['reviewed'] / c['photos']:.0%}" if c["photos"] else "-"
        short = sum(max(0, target - n) for n in holds)
        return (f"{tile:<5}{holds[0]:>8}{holds[1]:>7}{c.get('play_holds', 0) or '':>6}{c['photos']:>8}{c['sessions']:>9}  {light:<14}"
                f"{reviewed:>9}{short or '':>7}{c.get('waiting', 0) or '':>9}")
    order = sorted(counts, key=lambda t: (sum(counts[t]["holds"].values()), detector.TILE_NAMES.index(t)))
    playable = [t for t in order if t not in FLOWERS]
    head = (f"{'tile':<5}{'upright':>8}{'turned':>7}{'play':>6}{'photos':>8}{'sessions':>9}  {'daylight':<14}{'reviewed':>9}{'short':>7}"
            f"{'waiting':>9}")
    total = sum(sum(c["holds"].values()) + c.get("play_holds", 0) for c in counts.values())
    waiting = sum(c.get("waiting", 0) for c in counts.values())
    none = [t for t in playable if not counts[t]["photos"]]
    lines = [f"Base camera photos per tile, as reviewed: {total} holds in {len(counted)} interleaved session(s); "
             f"short = holds missing to {target} each way up",
             "(daylight: b bright, d dim, n none, in holds; a hold is one pickup shown at every pose; play: reviewed holds from "
             "play_face_down.py, way up unknown, not in short; waiting: play holds no one has reviewed, which nothing trains on)", "", head]
    lines += [line(t) for t in playable]
    lines += ["", "Flowers (collected, never played):", head] + [line(t) for t in order if t in FLOWERS]
    lines += ["", f"{len(playable) - len(none)} of {len(playable)} playable tiles have photos"
              + (f"; none yet: {', '.join(sorted(none, key=detector.TILE_NAMES.index))}" if none else "")]
    if waiting:
        lines.append(f"{waiting} hold(s) from play wait for review: python scripts/base_cam_review.py --play")
    left_out = [s for s in left_out if base_cam.rows(s)]          # an empty session (quit before a hold) is no news
    if left_out:
        lines.append(f"Not counted (collected before interleaving, so the scene named the tile): "
                     f"{', '.join(s.name for s in left_out)}")
    return lines


def main(argv=None, ask=None):
    args = parse_args(argv)
    counted, left_out = base_cam_collect.counted_sessions(config.SESSIONS)
    if args.counts:
        counts = base_cam_collect.tile_counts(config.SESSIONS)
        print("\n".join(count_lines(counts, args.target_holds, counted, left_out)))
        return counts
    from wrist_capture import ask_pumping
    ask = ask_pumping(ask or input)
    folders = counted + (left_out if args.all_sessions else [])
    if args.play:
        folders = [f for f in folders if f.name.endswith(f"_{base_cam_collect.PLAY_KIND}")]
    photos = base_cam.review_photos(folders)
    names = photo_review.name_order(detector.TILE_NAMES, args.name, args.only)
    reviewer = Reviewer(photos, base_cam.load_review(), names, show_all=args.all)
    if not reviewer.names:
        print(f"Nothing to review in {config.SESSIONS}" + ("" if args.all else " (--all shows reviewed photos too)"))
        photo_review.remind_to_push(base_cam.REVIEW)
        return
    print(f"{len(photos)} photos (one per pose of each hold) in {len(folders)} session(s); reviewing {', '.join(reviewer.names)}")
    photo_review.run(reviewer, ask, WINDOW, detector.TILE_NAMES)
    print(photo_review.summary(base_cam.load_review(), base_cam.REVIEW))
    photo_review.remind_to_push(base_cam.REVIEW)


if __name__ == "__main__":
    main()
