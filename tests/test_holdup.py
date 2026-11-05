"""Test 1 of docs/face_down_plan.md: the hold-up capture and its scoring. End to end on a synthetic camera and fake readers."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import config, holdup
from mahjong.session import read_events


class FakeReader:
    """Reads every photo as 5C, more confidently the second time (the turned one)."""

    def read_many(self, images):
        return [{"name": "5C", "confidence": 0.5 + 0.1 * i} for i, _ in enumerate(images)]


def capture_module(monkeypatch, tmp_path):
    import holdup_capture
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    return holdup_capture


def test_plan_goes_tile_by_tile_near_to_far_both_ways_up():
    shots = holdup.plan(["5C", "3C"], [15, 30])
    assert len(shots) == 8
    assert [(s["tile"], s["distance_cm"], s["kind"]) for s in shots[:4]] == [
        ("5C", 15, "upright"), ("5C", 15, "turned"), ("5C", 30, "upright"), ("5C", 30, "turned")]


def test_capture_saves_frames_under_the_tile_and_skip_and_redo_work(tmp_path, monkeypatch):
    capture = capture_module(monkeypatch, tmp_path)
    # 2 tiles x 1 distance x 2 ways = 4 shots: take, take, redo the second, take it, skip the third, quit
    answers = iter(["", "", "b", "", "s", "q"])
    out = capture.main(["--sim", "--tiles", "5C,3C", "--distances-cm", "20", "--frames", "2", "--gap-s", "0"],
                       ask=lambda prompt: next(answers, "q") if "hold " in prompt else "")     # setup questions: defaults
    shots = [e for e in read_events(out) if e["type"] == "holdup_shot"]
    assert [(e["tile"], e["kind"]) for e in shots] == [("5C", "upright"), ("5C", "turned"), ("5C", "turned")]
    assert len(list((out / "photos" / "5C").glob("*.jpg"))) == 4            # the redone shot's frames were deleted
    assert not (out / "photos" / "3C").exists()
    assert any(e["type"] == "holdup_conditions" for e in read_events(out))
    assert [e for e in read_events(out) if e["type"] == "holdup_done"][0]["shots_taken"] == 3


def test_score_reads_each_photo_both_ways_and_says_if_anything_read(tmp_path, monkeypatch):
    import holdup_score
    capture = capture_module(monkeypatch, tmp_path)
    out = capture.main(["--sim", "--tiles", "5C,3C", "--distances-cm", "20", "--frames", "1", "--gap-s", "0"],
                       ask=lambda prompt: "")
    boxes = lambda image: [{"tile_name": "3C", "confidence": 0.3, "xyxy": [240.0, 160.0, 400.0, 320.0]}]
    s = holdup_score.main([str(out)], reader=FakeReader(), predict=boxes)
    assert s["by"]["all"]["photos"] == 4 and s["by"]["all"]["boxed"] == 1.0
    assert s["by"]["all"]["whole frame/both"] == 0.5                         # 5C read right, 3C read as 5C
    assert s["by"]["all"]["detector_right"] == 0.5 and s["by"]["all"]["detector_right_at_0.5"] == 0.0
    assert s["per_tile"]["5C"]["whole frame/both"] == 1.0 and s["per_tile"]["3C"]["whole frame/both"] == 0.0
    assert s["readable_anywhere"]
    assert s["misreads"][0][0] == "3C->5C"
    assert "kind=turned" in s["by"] and "distance_cm=20" in s["by"]
    assert (out / "holdup_sheet.jpg").exists() and json.loads((out / "holdup_scores.json").read_text())["by"]


def test_nothing_readable_is_the_plans_stop():
    class Blind:
        def read_many(self, images):
            return [{"name": "EW", "confidence": 0.9} for _ in images]
    image = np.zeros((480, 640, 3), np.uint8)
    row = {"truth": "5C", "distance_cm": 15, "kind": "upright", "shot": 1, "daylight": "dim"}
    scored = holdup.score_photo(row, image, lambda im: [], Blind())
    assert scored["reads"]["detector box"] is None                           # no box, no box framing
    s = holdup.score([scored])
    assert not s["readable_anywhere"]
    assert "NO." in holdup.summary_lines(s)[-1]


def test_square_around_a_box_stays_inside_the_frame():
    image = np.zeros((480, 640, 3), np.uint8)
    assert holdup.square_around(image, [600.0, 400.0, 640.0, 480.0]).shape[:2] == (224, 224)
