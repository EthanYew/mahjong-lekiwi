"""Photos of tiles on the mat for the tile detector, without the arm: a live RealSense view with what the
detector sees, and space takes a scan (its 5 frames, depth and every box, saved as the bench saves
its scans) into sessions/<time>_capture/scans/.

    python scripts/capture_frames.py              # frames to train on
    python scripts/capture_frames.py --exam       # this session's frames are exam frames: never trained on
    python scripts/capture_frames.py --sim        # a rendered mat, no camera

Keys in the view: space (or c) scan, l the lighting changed (answer the setup questions again), q quit.
Boxes: green at or above the detection cut, amber below it (where it hesitated). Between scans move
tiles, turn some, push some together, change the light. Then `scripts/mine_frames.py` exports the
frames for labelling in Roboflow (mahjong/detector.py has the whole loop).

For the exam (data/detector_exam.json), capture one session per lighting you use (an exam session's
lighting is its setup answers, so l is off there), with tile arrangements of their own, not a
training session's.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import config, detector
from mahjong.session import Session, ask_conditions, ask_notes

WINDOW = "capture"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exam", action="store_true",
                   help="list this session in data/detector_exam.json: its frames score the detector, never train it")
    p.add_argument("--sim", action="store_true", help="a rendered mat instead of the RealSense")
    return p.parse_args(argv)


def take_scan(realsense, session, scan_n):
    """One scan, saved and logged as a `detections` event. Returns (tiles, issues Counter)."""
    tiles = realsense.detect_tiles()
    frames = detector.save_session_scan(realsense, session, scan_n, "capture")
    session.event("detections", label="capture", scan=scan_n, **frames,
                  tiles=[{k: t[k] for k in ("tile_name", "xy", "confidence", "reads", "box_px") if k in t} for t in tiles])
    issues = detector.scan_issues(detector.load_reads(session.dir / frames["frames"]))[0] if frames else {}
    return tiles, issues


def draw(image, boxes):
    out = image.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(round(c)) for c in b["xyxy"])
        colour = (0, 200, 0) if b["confidence"] >= config.DETECTION_CONFIDENCE else (0, 170, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)
        cv2.putText(out, f"{b['tile_name']} {b['confidence']:.2f}", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1)
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.sim:
        from mahjong.sim import SimPerception
        realsense = SimPerception()
        session = Session("capture", ask_setup=False, sim=True, exam=args.exam)
    else:
        from mahjong import hardware
        realsense = hardware.connect(["realsense"])["realsense"]
        session = Session("capture", sim=False, exam=args.exam)
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
        print("space: scan, l: lighting changed, q: quit")
        scans, boxes = 0, []
        while True:
            cv2.imshow(WINDOW, draw(realsense.snapshot(), boxes))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord(" "), ord("c")):
                tiles, issues = take_scan(realsense, session, scans + 1)
                scans += 1
                boxes = realsense.last_frames[-1]["boxes"] if realsense.last_frames else []
                shaky = ", ".join(f"{k} x{n}" for k, n in sorted(issues.items())) or "none"
                print(f"  scan {scans}: {len(tiles)} tiles ({', '.join(t['tile_name'] for t in tiles)}); shaky: {shaky}")
            elif key == ord("l") and args.exam:
                print("  an exam session is one lighting (scored per lighting): quit and start another --exam session")
            elif key == ord("l"):
                conditions = ask_conditions(session.conditions, confirm=False)
                session.conditions = conditions
                session.event("conditions_changed", conditions=conditions, notes=ask_notes())
            elif key in (ord("q"), 27):
                break
        print(f"{scans} scans in {session.dir / 'scans'}")
    finally:
        cv2.destroyAllWindows()
        realsense.close()
        session.close()


if __name__ == "__main__":
    main()
