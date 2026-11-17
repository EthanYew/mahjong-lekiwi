"""Train and score the base camera's tile reader on the photos base_cam_collect.py took; --save-reader installs one.

    python scripts/base_cam_train.py --mark-boxes                 # once per taught pose: drag a box round where the tile shows
    python scripts/base_cam_train.py                              # every real holdup session; holds out the newest session if
                                                                # there are two or more, else the last 2 holds of each tile/way up
    python scripts/base_cam_train.py --by hold                    # always split by hold (optimistic: same session's light and hands)
    python scripts/base_cam_train.py --sessions sessions/EVENING_holdup,sessions/DAY_holdup --test-session DAY_holdup
                                                                # train on one light, test on another (the tiles both have)
    python scripts/base_cam_train.py --framing whole              # the whole frame's centre square, as before tile boxes
    python scripts/base_cam_train.py --sessions sessions/A_holdup,sessions/B_holdup --epochs 200
    python scripts/base_cam_train.py --save-reader --sessions sessions/A_holdup,sessions/B_holdup   # install assets/holdup_reader.pt
    python scripts/base_cam_train.py --history                    # every result this has written, oldest first (no GPU, no training)

Why: the tile detector (RealSense) and the name check (wrist camera) read ~0% on this view (test 1, 2026-11-05/06), because
neither was trained on it, so the base camera has a reader of its own. A check is NOT a fine-tune of the installed weights: it starts a
fresh YOLO11n-cls from its ImageNet weights, with a head for just the tiles held in these sessions. Holds are split whole,
never a shot's near-identical frames across train and test. It runs --seeds times (small sets vary), and prints the
held-out right rate against chance and against the best constant answer.

Framing (`--framing`, default `tile`): the square round the tile box of the pose each photo was taken at (data/camera_poses.json,
drawn with --mark-boxes), so the classifier sees the tile and little of the room. Photos without one (held by hand, or a pose
with no box) are left out. `whole` uses the whole frame's centre square for every photo.

The control: the same split trained again on the whole frame with the tile box greyed out. If that still beats the best
constant answer by CONTROL_MARGIN, something other than the tile names it (the first arm sessions scored 99% with the tile
gone, 2026-11-06: each tile's holds were taken in a row, from one spot, with one light and whoever was in view), and the
verdict is BACKGROUND whatever the main rate. Without tile boxes there is no control, and no pass either.

The bar (docs/face_down_plan.md, test 1b, set before the first run): 70% or more, on 3+ tiles and well above the best
constant answer -> worth a real capture at the arm's pose. 40% or less -> the view or the hold is the problem, look at the
photos before capturing more. In between: capture more and rerun. Writes holdup_check.json next to the newest session
(with --test-session, holdup_check_held_out.json in the session held out).

--history reads those files back (in --sessions, or every real holdup session): what each trained and tested on, the reader's
right rate, the control and the best constant answer. Training pays off when the reader is well above the constant answer,
the control is near it (else the background names the tile), and that holds on a session never trained on (--test-session).
"""
import argparse
import json
import statistics
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import camera_poses, config, base_cam, base_cam_collect, name_check

PASS_RATE, FAIL_RATE = 0.70, 0.40
CONTROL_MARGIN = 0.15           # the tile-less control this far above the best constant answer: the background names the tile
MARK_PHOTOS = 60                # photos averaged per pose for --mark-boxes


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sessions", default=None, help="holdup session folders, comma separated (default: every real one)")
    p.add_argument("--by", choices=("auto", "session", "hold"), default="auto", help="how to hold photos out (default auto)")
    p.add_argument("--test-holds", type=int, default=2, help="with --by hold: holds of each tile and way up held out (default 2)")
    p.add_argument("--test-session", default=None,
                   help="hold out this session (a folder name or path among --sessions), train on the rest, test the tiles "
                        "they share (e.g. train on an evening, test on a day); overrides --by")
    p.add_argument("--framing", choices=("tile", "whole"), default="tile",
                   help="tile: the square round each pose's tile box; whole: the frame's centre square (default tile)")
    p.add_argument("--epochs", type=int, default=150, help="default 150 (a small set needs many passes to train)")
    p.add_argument("--seeds", type=int, default=3, help="runs, seeds 0..N-1, for the check and again for the control (default 3)")
    p.add_argument("--save-reader", action="store_true",
                   help="instead of scoring: train one reader on every photo of these sessions and save it to assets/holdup_reader.pt (crude_run.py reads with it). Name the clean sessions with --sessions")
    p.add_argument("--mark-boxes", action="store_true",
                   help="draw each pose's tile box on the average of its photos (a window; Enter saves, c keeps the old one), then stop")
    p.add_argument("--history", action="store_true",
                   help="list every result this has written (in --sessions, or every real holdup session), oldest first; trains nothing")
    return p.parse_args(argv)


def find_sessions(arg):
    if arg:
        folders = [Path(s) for s in arg.split(",")]
    else:
        # face-down play's holds too: base_cam.rows leaves out every one a person hasn't reviewed (base_cam_review.py --play)
        folders = sorted([*config.SESSIONS.glob("*_holdup"), *config.SESSIONS.glob(f"*_{base_cam_collect.PLAY_KIND}")], key=lambda f: f.name)
    return [f for f in folders if f.is_dir() and not base_cam.is_sim(f)]


def _bar(rate, constant):
    if rate >= PASS_RATE and rate >= constant + 0.2:
        return "PASS: worth a real capture at the arm's pose"
    if rate <= FAIL_RATE:
        return "FAIL: look at the photos (hands, tile out of frame?) before capturing more"
    return "IN BETWEEN: capture more holds and rerun"


def verdict(rate, tiles, constant, control=None):
    """The test 1b bar, after the control: `control` is the right rate with the tile greyed out (None: no tile boxes)."""
    if tiles < 3:
        return f"only {tiles} tile(s) held out: too few to judge. Capture 3 or more"
    if control is not None and control >= constant + CONTROL_MARGIN:
        return (f"BACKGROUND: with the tile greyed out a classifier still scores {control:.0%} (best constant answer "
                f"{constant:.0%}), so the photos name the tile without it and {rate:.0%} says nothing about reading. Take the "
                f"tiles in turn from the same spots (not one tile's holds in a row) and score on a session never trained on")
    result = _bar(rate, constant)
    if control is None and result.startswith("PASS"):
        return ("UNCONTROLLED: the bar is met, but with no tile boxes nothing checks that it's the tile being read "
                "(run --mark-boxes on photos the arm took)")
    return result


def framed(row, framing, boxes, blank=False):
    """One photo as the classifier gets it. `blank`: the control, the whole frame with the tile box greyed out."""
    image = cv2.imread(str(row["path"]))
    box = boxes.get(row.get("pose"))
    if blank:
        return base_cam.whole_frame(base_cam.without_tile(image, box))
    if framing == "tile":
        return base_cam.tile_frame(image, box)
    return base_cam.whole_frame(image)


def train_and_score(train_rows, test_rows, frame, seeds, epochs, train_fn, predict_fn, label):
    """Trains `seeds` fresh classifiers on `train_rows` framed by `frame`, scores each on `test_rows`. Returns their summaries."""
    runs = []
    with tempfile.TemporaryDirectory() as folder:
        photos = []
        for i, r in enumerate(train_rows):
            path = Path(folder) / f"{i:05d}.png"         # lossless: the framed photo, not a second JPEG of it
            cv2.imwrite(str(path), frame(r))
            photos.append({"name": r["truth"], "path": path, "session": r["session"]})
        images = [frame(r) for r in test_rows]
        for seed in range(seeds):
            checkpoint = train_fn(photos, seed=seed, epochs=epochs, log=lambda *_: None, crop_fn=base_cam.as_input)
            said = predict_fn(checkpoint, images, crop_fn=base_cam.as_input)
            runs.append(base_cam.check_summary(test_rows, said))
            print(f"  {label} seed {seed}: right {runs[-1]['right']:.0%}")
    return runs


def mean_photo(rows, n=MARK_PHOTOS):
    """The average of up to `n` of `rows`' photos, spread over them: the tile's smear shows where it can be."""
    import numpy as np
    step = max(1, len(rows) // n)
    images = [cv2.imread(str(r["path"])).astype(np.float32) for r in rows[::step][:n]]
    return np.mean(images, axis=0).astype(np.uint8)


def mark_boxes(rows, boxes, select=None):
    """For each pose the photos were taken at, a person drags a box round where the tile can be on the average of its photos
    (plus a margin: each pose is nudged and each grip differs). Saves it in data/camera_poses.json. Returns {pose: box}."""
    if select is None:
        def select(name, image):
            x, y, w, h = cv2.selectROI(f"{name}: drag round where the tile can be, Enter saves, c keeps the old box", image)
            cv2.destroyAllWindows()
            return [x, y, x + w, y + h] if w and h else None
    by_pose = {}
    for r in rows:
        if r.get("pose"):
            by_pose.setdefault(r["pose"], []).append(r)
    if not by_pose:
        raise SystemExit("No photo here was taken at a saved pose (held by hand?): boxes come from base_cam_collect.py's photos")
    saved = {}
    for name, items in sorted(by_pose.items()):
        image = mean_photo(items)
        if name in boxes:
            x0, y0, x1, y1 = boxes[name]
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 220, 255), 2)
        box = select(name, image)
        if box:
            camera_poses.set_tile_box(name, box, f"drawn with base_cam_train.py --mark-boxes on the average of {len(items)} "
                                                 f"photos, {date.today().isoformat()}")
            saved[name] = box
            print(f"{name}: tile box {box}")
        else:
            print(f"{name}: kept {boxes.get(name, 'no box')}")
    return saved


def save_reader(rows, args, boxes, train_fn):
    """One reader on every photo, nothing held out: the scores come from the runs above, on sessions it never saw."""
    if args.framing != "tile":
        raise SystemExit("--save-reader reads the tile box of each pose: leave --framing at tile")
    with tempfile.TemporaryDirectory() as folder:
        photos = []
        for i, r in enumerate(rows):
            path = Path(folder) / f"{i:05d}.png"
            cv2.imwrite(str(path), framed(r, args.framing, boxes))
            photos.append({"name": r["truth"], "path": path, "session": r["session"]})
        checkpoint = train_fn(photos, seed=0, epochs=args.epochs, log=lambda *_: None, crop_fn=base_cam.as_input)
    saved = name_check.save(checkpoint, base_cam.READER)
    print(f"Reader on {len(rows)} photos of {', '.join(sorted({r['truth'] for r in rows}))}: {saved}")
    return saved


def main(argv=None, train_fn=None, predict_fn=None, select=None):
    args = parse_args(argv)
    folders = find_sessions(args.sessions)
    if args.history:
        results = base_cam.check_results(folders)
        print("\n".join(base_cam.check_lines(results)))
        return results
    rows = [r for f in folders for r in base_cam.rows(f)]
    if not rows:
        raise SystemExit(f"No holdup photos in {[str(f) for f in folders] or config.SESSIONS}: python scripts/base_cam_collect.py")
    boxes = base_cam.tile_boxes()
    if args.mark_boxes:
        return mark_boxes(rows, boxes, select)
    if args.framing == "tile":
        kept = [r for r in rows if r["pose"] in boxes]
        if not kept:
            raise SystemExit("No photo has a tile box (hand-held photos never do): run --mark-boxes on photos the arm took, "
                             "or --framing whole")
        if len(kept) < len(rows):
            print(f"left out {len(rows) - len(kept)} photo(s) with no tile box (held by hand, or a pose with none)")
        rows = kept
    if args.save_reader:
        return save_reader(rows, args, boxes, train_fn or name_check.train)
    test_session = Path(args.test_session).name if args.test_session else None
    try:
        train_rows, test_rows, how, caveat = base_cam.split(rows, args.by, args.test_holds, test_session)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"{len(rows)} photos from {len({r['session'] for r in rows})} session(s), framed: {args.framing}; {how}")
    print(f"  train {len(train_rows)} photos of {len({r['truth'] for r in train_rows})} tiles, "
          f"test {len(test_rows)} of {len({r['truth'] for r in test_rows})}")
    if caveat:
        print(f"  {caveat}")
    train_fn = train_fn or name_check.train
    predict_fn = predict_fn or name_check.predict
    runs = train_and_score(train_rows, test_rows, lambda r: framed(r, args.framing, boxes), args.seeds, args.epochs,
                           train_fn, predict_fn, "check")
    controllable = all(r["pose"] in boxes for r in rows)
    if controllable:
        control_runs = train_and_score(train_rows, test_rows, lambda r: framed(r, args.framing, boxes, blank=True),
                                       args.seeds, args.epochs, train_fn, predict_fn, "control, tile greyed out")
        control = statistics.mean(r["right"] for r in control_runs)
    else:
        control_runs, control = None, None
        print("  no control: some photos have no tile box, so the tile can't be greyed out")
    rate = statistics.mean(r["right"] for r in runs)
    first = runs[0]
    print(f"\nheld-out right rate {rate:.0%} (seeds {min(r['right'] for r in runs):.0%} to {max(r['right'] for r in runs):.0%}); "
          f"chance {first['chance']:.0%}, best constant answer {first['best_constant_answer']:.0%}")
    if control is not None:
        print(f"control, tile greyed out: {control:.0%} (should be near the best constant answer)")
    print("  per tile (seed 0): " + ", ".join(f"{t} {v:.0%}" for t, v in first["per_tile"].items()))
    print("  per way up (seed 0): " + ", ".join(f"{k} {v:.0%}" for k, v in first["per_kind"].items()))
    print("  confusions (seed 0): " + (", ".join(f"{k} x{n}" for k, n in first["confusions"]) or "none"))
    result = verdict(rate, first["tiles"], first["best_constant_answer"], control)
    print(f"\n{result}")
    if test_session:            # its own file in the session held out, so the usual check's result stays
        out = next(f for f in folders if f.name == test_session) / "holdup_check_held_out.json"
    else:
        out = folders[-1] / "holdup_check.json"
    holds = lambda rs: len({(r["session"], r["truth"], r["kind"], r["shot"]) for r in rs})
    out.write_text(json.dumps({"written": datetime.now().isoformat(timespec="minutes"),
                               "split": how, "caveat": caveat, "sessions": sorted({r["session"] for r in rows}),
                               "train_photos": len(train_rows), "train_holds": holds(train_rows),
                               "test_photos": len(test_rows), "test_holds": holds(test_rows),
                               "framing": args.framing, "epochs": args.epochs, "mean_right": rate, "runs": runs,
                               "control_right": control, "control_runs": control_runs, "verdict": result}, indent=2) + "\n")
    print(f"saved {out}")
    return {"mean_right": rate, "runs": runs, "control_right": control, "verdict": result}


if __name__ == "__main__":
    main()
