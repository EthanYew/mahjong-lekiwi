"""Photos of tiles on the mat for the tile detector, without the arm: a live RealSense view with what the
detector sees, and space takes a scan (its 5 frames, depth and every box, saved as the bench saves
its scans) into sessions/<time>_capture/scans/.

    python scripts/capture_frames.py              # frames to train on
    python scripts/capture_frames.py --exam       # this session's frames are exam frames: never trained on
    python scripts/capture_frames.py --target 5   # aim for 5 of each tile name (default 3)
    python scripts/capture_frames.py --sim        # a rendered mat, no camera

The view shows the keys (space or c scan, l the lighting changed, q quit), what to change before each
scan, and under it a count of every tile name scanned so far, as the detector read them: red none yet,
amber below --target, green there. Boxes: green at or above the detection cut, amber below it (where
it hesitated). A scan whose tiles barely moved since the last one is flagged: the same scene twice
teaches little, so move or swap 5-8 tiles first. Then `scripts/mine_frames.py` exports the frames for
labelling in Roboflow (README, "The tile detector").

For the exam (data/detector_exam.json), capture one session per lighting you use (an exam session's
lighting is its setup answers, so l is off there), with tile arrangements of their own, not a
training session's.
"""
import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import config, detector
from mahjong.offline_eval import lighting
from mahjong.session import Session, ask_conditions, ask_notes

WINDOW = "capture"
UNCHANGED_M = 0.01          # a tile within this of a same-named tile in the last scan hasn't moved
UNCHANGED_SHARE = 0.8       # ...and a scan with this share of such tiles is the same arrangement
# the count panel: one column per suit, numbers down; honours (winds, dragons) in the last column
COLUMNS = [[f"{n}{s}" for n in range(1, 10)] for s in "BCD"] + [[f"{n}F" for n in range(1, 5)],
                                                                 ["EW", "SW", "WW", "NW", "RD", "GD", "WD"]]
TIPS = "Before each scan: move or swap 5-8 tiles, mix suits, use the whole mat, turn a few, push some together"
GREEN, AMBER, RED, WHITE = (0, 200, 0), (0, 170, 255), (60, 60, 230), (235, 235, 235)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exam", action="store_true",
                   help="list this session in data/detector_exam.json: its frames score the detector, never train it")
    p.add_argument("--target", type=int, default=3, help="tiles of each name to aim for over the session (default 3)")
    p.add_argument("--sim", action="store_true", help="a rendered mat instead of the RealSense")
    return p.parse_args(argv)


def take_scan(realsense, session, scan_n):
    """One scan, saved and logged as a `detections` event. Returns (tiles, issues Counter)."""
    tiles, scan = detector.log_scan(realsense, session, scan_n, "capture")
    issues = detector.scan_issues(detector.load_reads(session.dir / scan["frames"]))[0] if "frames" in scan else {}
    return tiles, issues


def unchanged_share(previous, tiles, radius_m=UNCHANGED_M):
    """The share of `tiles` with a same-named tile within radius_m in the `previous` scan (0 with none)."""
    if not tiles or not previous:
        return 0.0
    still = sum(any(p["tile_name"] == t["tile_name"] and math.dist(p["xy"], t["xy"]) <= radius_m for p in previous)
                for t in tiles)
    return still / len(tiles)


class Capture:
    """A capture session's state: scans taken, tile names counted, what the view says."""

    def __init__(self, realsense, session, exam=False, target=3):
        self.realsense, self.session, self.exam, self.target = realsense, session, exam, target
        self.scans, self.counts, self.previous, self.boxes = 0, Counter(), None, []
        self.status, self.status_colour = "No scans yet: arrange tiles, then press space", WHITE
        self.same_arrangement = 0       # scans flagged as the last one again

    def scan(self):
        tiles, issues = take_scan(self.realsense, self.session, self.scans + 1)
        self.scans += 1
        self.boxes = self.realsense.last_frames[-1]["boxes"] if self.realsense.last_frames else []
        self.counts.update(t["tile_name"] for t in tiles)
        shaky = ", ".join(f"{k} x{n}" for k, n in sorted(issues.items())) or "none"
        self.status, self.status_colour = f"Scan {self.scans}: {len(tiles)} tiles, shaky: {shaky}", WHITE
        if len(tiles) >= 3 and unchanged_share(self.previous, tiles) >= UNCHANGED_SHARE:
            self.same_arrangement += 1
            self.status = f"Scan {self.scans}: SAME ARRANGEMENT as scan {self.scans - 1} - move or swap 5-8 tiles next"
            self.status_colour = AMBER
        self.previous = tiles
        print(f"  scan {self.scans}: {len(tiles)} tiles ({', '.join(t['tile_name'] for t in tiles)}); shaky: {shaky}"
              + ("  <- same arrangement as the last scan" if self.status_colour == AMBER else ""))
        print(f"    {self.missing_text()}")

    def below_target(self):
        return [n for column in COLUMNS for n in column if self.counts[n] < self.target]

    def missing_text(self):
        """The names still below the target: listed once there are few, else how many."""
        missing = self.below_target()
        if not missing:
            return f"every name has {self.target}+"
        if len(missing) > 12:
            return f"{len(missing)} names below {self.target} (the panel shows which)"
        return f"below {self.target}: {', '.join(missing)}"

    def header(self):
        if self.exam:
            return f"EXAM session, one lighting: {lighting(self.session.conditions)}"
        return "TRAINING session" + (f": {lighting(self.session.conditions)}" if self.session.conditions else "")

    def view(self, image):
        """The camera frame with the boxes, the guidance on top and the count panel underneath."""
        out = draw(image, self.boxes)
        keys = "space: scan   q: quit" + ("" if self.exam else "   l: lighting changed (answer in the terminal)")
        lines = [(self.header(), WHITE), (keys, WHITE), (TIPS, WHITE), (self.status, self.status_colour)]
        band = out[:16 * len(lines) + 6]
        band[:] = (band * 0.35).astype(np.uint8)          # darken behind the text
        for i, (text, colour) in enumerate(lines):
            cv2.putText(out, text, (8, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        return np.vstack([out, self.panel(out.shape[1])])

    def panel(self, width):
        rows = max(len(c) for c in COLUMNS)
        panel = np.full((28 + 18 * rows, width, 3), 30, np.uint8)
        done = sum(self.counts[n] >= self.target for c in COLUMNS for n in c)
        total = sum(len(c) for c in COLUMNS)
        cv2.putText(panel, f"Tiles scanned per name, as read (aim for {self.target}): {done}/{total} names there",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
        step = width // len(COLUMNS)
        for c, column in enumerate(COLUMNS):
            for r, name in enumerate(column):
                n = self.counts[name]
                colour = RED if n == 0 else AMBER if n < self.target else GREEN
                cv2.putText(panel, f"{name:<3}{n:>3}", (8 + c * step, 42 + 18 * r), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            colour, 1, cv2.LINE_AA)
        return panel


def draw(image, boxes):
    out = image.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(round(c)) for c in b["xyxy"])
        colour = GREEN if b["confidence"] >= config.DETECTION_CONFIDENCE else AMBER
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)
        cv2.putText(out, f"{b['tile_name']} {b['confidence']:.2f}", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1)
    return out


def print_guide(exam, target):
    print(f"""
How to capture{' the exam' if exam else ''}:
  1. Lay out 12-15 face-up tiles from several suits, across the whole mat (edges and corners too).
  2. Press space to scan (keep hands out of view for ~1s).
  3. Before the next scan, change the scene: move or swap 5-8 tiles, turn a few, push some together.
     One or two scans per arrangement is plenty; repeats of the same scene teach little.
  4. Swap in the names still red or amber in the panel, until each has {target}.
  5. {'One lighting per exam session: for another lighting, quit and start a new --exam session.'
        if exam else 'Lighting changed? press l and answer in the terminal.'}
  q quits; the frames are in this session's scans/ folder (then scripts/mine_frames.py).
""")


def main(argv=None, show=None, wait_key=None):
    """show(image) and wait_key() -> key code stand in for the window in tests. Returns the Capture."""
    args = parse_args(argv)
    if args.sim:
        from mahjong.sim import SimPerception
        realsense = SimPerception()
        session = Session("capture", ask_setup=False, sim=True, exam=args.exam)
    else:
        from mahjong import hardware
        realsense = hardware.connect(["realsense"])["realsense"]
        session = Session("capture", sim=False, exam=args.exam)
    window = show is None
    show = show or (lambda image: cv2.imshow(WINDOW, image))
    wait_key = wait_key or (lambda: cv2.waitKey(30) & 0xFF)
    capture = Capture(realsense, session, exam=args.exam, target=args.target)
    try:
        if args.exam:
            if args.sim:
                print("(--sim: not listed in the exam)")
            else:
                if any(config.DETECTOR_EVAL.glob("*.json")):
                    print("Note: the exam already has scores; with this session it's a new exam, and "
                          "detector_eval.py scores both detectors again on it.")
                detector.add_exam_session(session.dir.name, session.conditions, session.notes)
                print(f"Exam session: {session.dir.name} (data/detector_exam.json) — commit that file.")
        print_guide(args.exam, args.target)
        while True:
            show(capture.view(realsense.snapshot()))
            key = wait_key()
            if key in (ord(" "), ord("c")):
                capture.scan()
            elif key == ord("l") and args.exam:
                capture.status = "Exam sessions keep one lighting: quit, then start another --exam session"
                capture.status_colour = AMBER
                print("  an exam session is one lighting (scored per lighting): quit and start another --exam session")
            elif key == ord("l"):
                conditions = ask_conditions(session.conditions, confirm=False)
                session.conditions = conditions
                session.event("conditions_changed", conditions=conditions, notes=ask_notes())
                capture.status, capture.status_colour = f"Lighting now: {lighting(conditions)}", WHITE
            elif key in (ord("q"), 27):
                break
        print(f"{capture.scans} scans in {session.dir / 'scans'}"
              + (f"; {capture.same_arrangement} repeated the scan before" if capture.same_arrangement else "")
              + f"\n{capture.missing_text()}")
    finally:
        if window:
            cv2.destroyAllWindows()
        realsense.close()
        session.close()
    return capture


if __name__ == "__main__":
    main()
