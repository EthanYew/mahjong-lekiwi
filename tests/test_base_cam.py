"""Test 1 of docs/face_down_plan.md: the hold-up capture and its scoring. End to end on a synthetic camera and fake readers."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import config, base_cam
from mahjong.session import read_events


class FakeReader:
    """Reads every photo as 5C, whichever way up."""

    def read_many(self, images):
        return [{"name": "5C", "confidence": 0.5 + 0.1 * i} for i, _ in enumerate(images)]


def capture_module(monkeypatch, tmp_path):
    import holdup_capture
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    return holdup_capture


def test_plan_goes_tile_by_tile_each_way_up_several_holds():
    shots = base_cam.plan(["5C", "3C"], 2)
    assert len(shots) == 8
    assert [(s["tile"], s["kind"], s["hold"]) for s in shots[:5]] == [
        ("5C", "upright", 1), ("5C", "upright", 2), ("5C", "turned", 1), ("5C", "turned", 2), ("3C", "upright", 1)]


def test_capture_saves_frames_under_the_tile_and_skip_and_redo_work(tmp_path, monkeypatch):
    capture = capture_module(monkeypatch, tmp_path)
    # 2 tiles x 2 ways x 1 hold = 4 shots: take, take, redo the second, take it, skip the third, quit
    answers = iter(["", "", "b", "", "s", "q"])
    out = capture.main(["--sim", "--tiles", "5C,3C", "--holds", "1", "--frames", "2", "--gap-s", "0"],
                       ask=lambda prompt: next(answers, "q") if "hold " in prompt else "")     # setup questions: defaults
    shots = [e for e in read_events(out) if e["type"] == "holdup_shot"]
    assert [(e["tile"], e["kind"]) for e in shots] == [("5C", "upright"), ("5C", "turned"), ("5C", "turned")]
    assert len(list((out / "photos" / "5C").glob("*.jpg"))) == 4            # the redone shot's frames were deleted
    assert not (out / "photos" / "3C").exists()
    assert any(e["type"] == "holdup_conditions" for e in read_events(out))
    assert [e for e in read_events(out) if e["type"] == "holdup_done"][0]["shots_taken"] == 3
    assert len(base_cam.rows(out)) == 4                                        # the redone shot left no row


def test_a_redone_shot_leaves_no_row_and_a_sim_session_says_so(tmp_path, monkeypatch):
    capture = capture_module(monkeypatch, tmp_path)
    answers = iter(["", "b", "q"])
    out = capture.main(["--sim", "--tiles", "5C", "--holds", "1", "--frames", "2", "--gap-s", "0"],
                       ask=lambda prompt: next(answers, "q") if "hold " in prompt else "")
    assert base_cam.rows(out) == [] and base_cam.is_sim(out)


def test_score_keeps_the_two_ways_up_apart_and_one_answer_for_everything_is_not_reading(tmp_path, monkeypatch):
    import holdup_score
    capture = capture_module(monkeypatch, tmp_path)
    out = capture.main(["--sim", "--tiles", "5C,3C", "--holds", "1", "--frames", "1", "--gap-s", "0"],
                       ask=lambda prompt: "")
    boxes = lambda image: [{"tile_name": "3C", "confidence": 0.3, "xyxy": [240.0, 160.0, 400.0, 320.0]}]
    s = holdup_score.main(["--installed", str(out)], reader=FakeReader(), predict=boxes)
    assert s["by"]["all"]["photos"] == 4 and s["by"]["all"]["boxed"] == 1.0
    assert s["by"]["all"]["whole frame/as_is"] == s["by"]["all"]["whole frame/turned"] == 0.5     # 5C right, 3C read as 5C
    assert "whole frame/both" not in s["by"]["all"]
    assert s["by"]["all"]["detector_right"] == 0.5 and s["by"]["all"]["detector_right_at_0.5"] == 0.0
    assert s["per_tile"]["5C"]["whole frame/as_is"] == 1.0 and s["per_tile"]["3C"]["whole frame/as_is"] == 0.0
    assert not s["readable_anywhere"]                                        # 5C for everything, 3C for everything: no reader
    assert s["readers"]["whole frame/as_is"]["answers"] == 1 and s["readers"]["whole frame/as_is"]["top_share"] == 1.0
    assert s["misreads"][0][0] == "3C->5C"
    assert "kind=turned" in s["by"] and "daylight=bright" in s["by"]
    assert (out / "holdup_sheet.jpg").exists() and json.loads((out / "holdup_scores.json").read_text())["by"]


def test_nothing_readable_is_the_plans_stop():
    class Blind:
        def read_many(self, images):
            return [{"name": "EW", "confidence": 0.9} for _ in images]
    image = np.zeros((480, 640, 3), np.uint8)
    row = {"truth": "5C", "kind": "upright", "hold": 1, "shot": 1, "daylight": "dim"}
    scored = base_cam.score_photo(row, image, lambda im: [], Blind())
    assert scored["reads"]["detector box"] is None                           # no box, no box framing
    s = base_cam.score([scored])
    assert not s["readable_anywhere"]
    assert "NONE" in base_cam.summary_lines(s)[-1]


def test_square_around_a_box_stays_inside_the_frame():
    image = np.zeros((480, 640, 3), np.uint8)
    assert base_cam.square_around(image, [600.0, 400.0, 640.0, 480.0]).shape[:2] == (224, 224)


def scored_rows(said):
    """Rows for tiles 5C, 3C, 2C (4 photos each) where the whole-frame classifier says `said(truth, i)`."""
    rows = []
    for truth in ("5C", "3C", "2C"):
        for i in range(4):
            read = {"name": said(truth, i), "confidence": 0.9}
            rows.append({"truth": truth, "kind": "upright", "hold": i + 1, "shot": i, "daylight": "dim", "boxes": 0,
                         "detector": None, "detector_confidence": 0.0,
                         "reads": {"whole frame": {"as_is": read, "turned": read}, "wrist crop": None, "detector box": None}})
    return rows


def test_a_reader_right_on_two_tiles_reads_and_a_constant_one_does_not():
    right_on_two = base_cam.score(scored_rows(lambda truth, i: truth if truth != "2C" else "5C"))
    assert right_on_two["readable_anywhere"] and right_on_two["readers"]["whole frame/as_is"]["tiles_read"] == ["3C", "5C"]
    constant = base_cam.score(scored_rows(lambda truth, i: "3C"))
    assert not constant["readable_anywhere"] and constant["readers"]["whole frame/as_is"]["tiles_read"] == ["3C"]
    assert abs(constant["by"]["all"]["whole frame/as_is"] - 1 / 3) < 1e-9       # the 24% of the first session, in miniature


def fake_rows(sessions):
    rows = []
    for session in sessions:
        for truth in ("5C", "3C", "2C"):
            for kind in base_cam.KINDS:
                for hold in range(1, 5):
                    rows.append({"truth": truth, "kind": kind, "hold": hold, "shot": hold, "session": session,
                                 "path": f"{session}/{truth}_{kind}_{hold}.jpg"})
    return rows


def test_split_holds_out_the_newest_session_or_the_last_holds():
    two = base_cam.split(fake_rows(["a_holdup", "b_holdup"]), "auto")
    assert {r["session"] for r in two[0]} == {"a_holdup"} and {r["session"] for r in two[1]} == {"b_holdup"} and two[3] is None
    one = base_cam.split(fake_rows(["a_holdup"]), "auto", test_holds=2)
    assert {r["hold"] for r in one[1]} == {3, 4} and {r["hold"] for r in one[0]} == {1, 2}
    assert "OPTIMISTIC" in one[3]
    assert len(one[0]) + len(one[1]) == 24


def test_check_summary_compares_with_chance_and_a_constant_answer():
    rows = fake_rows(["a_holdup"])
    s = base_cam.check_summary(rows, [r["truth"] if r["truth"] != "2C" else "5C" for r in rows])
    assert s["tiles"] == 3 and abs(s["right"] - 2 / 3) < 1e-9 and abs(s["chance"] - 1 / 3) < 1e-9
    assert s["per_tile"]["2C"] == 0.0 and s["confusions"][0][0] == "2C->5C"
