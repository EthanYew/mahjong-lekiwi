"""The tile detector's data loop without a camera or a GPU: scans saved by the bench and capture_frames,
the frames it struggled with, the pre-labelled export, the exam kept out of training, and scoring."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mahjong import config, detector
from mahjong.session import Session, read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def box(name, conf, x, y, w=24, h=18):
    return {"tile_name": name, "confidence": conf, "xyxy": [x, y, x + w, y + h]}


def reads(*frames):
    return {"names": detector.TILE_NAMES, "size": [848, 480], "frames": list(frames)}


def test_a_steady_scan_has_no_issues():
    frame = [box("8D", 0.9, 100, 100), box("2B", 0.85, 200, 100)]
    issues, k = detector.scan_issues(reads(frame, frame, frame))
    assert not issues and k == 0


def test_flicker_dropout_and_low_reads_pick_the_shakiest_frame():
    steady = box("8D", 0.9, 100, 100)
    frames = [[steady, box("2B", 0.8, 200, 100)],
              [steady, box("3B", 0.6, 202, 101)],          # same tile, another name: flicker
              [steady],                                    # 2B not seen: dropout
              [steady, box("WD", 0.4, 300, 300)],          # a tile only below the cut: low
              [steady, box("2B", 0.8, 201, 100)]]
    issues, k = detector.scan_issues(reads(*frames))
    assert issues == {"flicker": 1, "dropout": 1, "low": 1}
    assert k == 2                                          # neither shaky tile seen there


def test_names_survive_a_roboflow_round_trip():
    stem = detector.frame_stem("20261027_101500_capture", "003_capture", 2)
    assert detector.original_name(f"{stem}.jpg") == stem
    assert detector.original_name(f"{stem}_jpg.rf.AnEu4EyWSu5nSHWVT0nU.jpg") == stem
    assert detector.session_of(f"{stem}_jpg.rf.x.txt") == "20261027_101500_capture"
    assert detector.session_of("frame_0000_exp200_jpg.rf.AnEu4EyWSu5nSHWVT0nU.jpg") is None   # the July frames


def test_yolo_labels_round_trip_and_polygons_score_by_their_box(tmp_path):
    boxes = [box("8D", 0.9, 100, 100), box("WW", 0.7, 400, 240, 30, 20)]
    path = tmp_path / "a.txt"
    path.write_text("\n".join(detector.yolo_lines(boxes, detector.TILE_NAMES, (848, 480))) + "\n"
                    + f"{detector.TILE_NAMES.index('2B')} 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    back = detector.read_yolo_labels(path, detector.TILE_NAMES, (848, 480))
    assert [b["tile_name"] for b in back] == ["8D", "WW", "2B"]
    assert back[1]["xyxy"] == pytest.approx([400, 240, 430, 260], abs=0.01)
    assert back[2]["xyxy"] == pytest.approx([84.8, 48, 169.6, 96], abs=0.01)


def test_score_counts_found_named_right_missed_extra_and_mixups():
    truth = [box("8D", 1, 100, 100), box("2B", 1, 200, 100), box("WD", 1, 300, 100)]
    said = [box("8D", 0.9, 101, 100), box("3B", 0.8, 200, 101), box("RD", 0.6, 500, 400)]
    scores = detector.score([{"truth": truth, "said": said, "group": "lamp"}])
    for group in ("all", "lamp"):
        s = detector.summary(scores[group])
        assert (s["tiles"], s["found"], s["named_right"], s["missed"], s["extra"]) == (3, 2, 1, 1, 1)
        assert s["mixups"] == [["2B", "3B", 1]] and s["missed_names"] == {"WD": 1}


def bench_session(tmp_path, monkeypatch):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "1", "--tile", "8D"]))
    try:
        b.arm.rest()
        b.trials(1, "8D")
    finally:
        b.close()
    return next(tmp_path.iterdir())


def test_bench_scans_save_their_frames(tmp_path, monkeypatch):
    session = bench_session(tmp_path, monkeypatch)
    scans = [e for e in read_events(session) if e["type"] == "detections"]
    assert scans and all(e["frames"].startswith("scans/") for e in scans)
    first = session / scans[0]["frames"]
    assert (first / "f0.jpg").exists()
    saved = detector.load_reads(first)
    assert saved["names"] == detector.TILE_NAMES and saved["size"] == [848, 480]
    assert {b["tile_name"] for b in saved["frames"][0]} == {t["tile_name"] for t in scans[0]["tiles"]}


def test_capture_then_mine_exports_prelabelled_frames(tmp_path, monkeypatch):
    import capture_frames
    import mine_frames
    from mahjong.sim import SimPerception
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    mat = SimPerception()
    session = Session("capture", ask_setup=False)          # a real (not sim) session, as on hardware
    tiles, issues = capture_frames.take_scan(mat, session, 1)
    mat.nudge("8D", 0.02, 0.0)
    capture_frames.take_scan(mat, session, 2)
    session.close()
    assert tiles and not issues

    rows, left_out = mine_frames.pick([session.dir], {"sessions": {}}, everything=True)
    assert [r["scan"] for r in rows] == ["001_capture", "002_capture"] and not left_out
    out = tmp_path / "export"
    assert detector.export(rows, out) == 2
    image = out / "images" / f"{session.dir.name}__001_capture__f0.jpg"
    assert image.exists()
    labels = detector.read_yolo_labels(out / "labels" / f"{image.stem}.txt", detector.TILE_NAMES, (848, 480))
    assert sorted(b["tile_name"] for b in labels) == sorted(t["tile_name"] for t in tiles)
    assert "names:" in (out / "data.yaml").read_text() and (out / "manifest.csv").exists()

    # the same session as an exam session: out of training exports, into exam ones
    exam = {"sessions": {session.dir.name: {"conditions": {}}}}
    assert mine_frames.pick([session.dir], exam, everything=True)[0] == []
    assert len(mine_frames.pick([session.dir], exam, exam_only=True)[0]) == 2
    assert detector.exam_frames(list((out / "images").iterdir()), exam)


def test_sim_sessions_are_never_mined(tmp_path, monkeypatch):
    import mine_frames
    session = bench_session(tmp_path, monkeypatch)
    rows, left_out = mine_frames.pick([session], {"sessions": {}}, everything=True)
    assert rows == [] and left_out == {session.name: "simulated"}


def test_evaluate_scores_only_exam_frames(tmp_path, monkeypatch):
    from mahjong.sim import SimPerception
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    mat = SimPerception()
    import capture_frames
    exam_session = Session("capture", ask_setup=False)
    capture_frames.take_scan(mat, exam_session, 1)
    exam_session.close()
    rows = detector.mine(exam_session.dir, everything=True)
    out = tmp_path / "exam"
    detector.export(rows, out)
    (out / "images" / "frame_0000_exp200_jpg.rf.abc.jpg").write_bytes((out / "images" / f"{rows[0]['session']}__001_capture__f0.jpg").read_bytes())
    (out / "labels" / "frame_0000_exp200_jpg.rf.abc.txt").write_text("")
    truth = mat.last_frames[0]["boxes"]
    wrong_first = [{**truth[0], "tile_name": "WW" if truth[0]["tile_name"] != "WW" else "RD"}] + truth[1:]
    exam = {"sessions": {exam_session.dir.name: {"conditions": {"daylight": "dim", "lamp": True, "ceiling_lights": False}}}}
    scores, outside = detector.evaluate(out, lambda image: wrong_first, exam, by="lighting")
    assert [p.name for p in outside] == ["frame_0000_exp200_jpg.rf.abc.jpg"]
    s = detector.summary(scores["all"])
    assert (s["frames"], s["found"], s["named_right"]) == (1, len(truth), len(truth) - 1)
    assert "daylight dim, lamp on, ceiling off" in scores


def test_add_exam_session(tmp_path):
    path = tmp_path / "exam.json"
    detector.add_exam_session("20261027_capture", {"lamp": True}, path=path)
    assert json.loads(path.read_text())["sessions"]["20261027_capture"]["conditions"] == {"lamp": True}
    assert detector.exam_frames([Path("20261027_capture__001_capture__f0_jpg.rf.x.jpg"), Path("b.jpg")],
                                detector.load_exam(path)) == [Path("20261027_capture__001_capture__f0_jpg.rf.x.jpg")]


def test_capture_counts_names_and_flags_a_repeated_arrangement(tmp_path, monkeypatch):
    import capture_frames
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    shown, keys = [], iter([ord(" "), 0, ord(" "), ord("q")])
    capture = capture_frames.main(["--sim", "--target", "2"], show=shown.append, wait_key=lambda: next(keys))
    assert capture.scans == 2 and capture.same_arrangement == 1      # nothing moved between the scans
    sim_names = [t["tile_name"] for t in capture.realsense.tiles]
    assert all(capture.counts[n] == 2 for n in sim_names)
    assert set(capture.below_target()) == set(detector.TILE_NAMES) - set(sim_names)
    assert "SAME ARRANGEMENT" in capture.status
    frame = shown[-1]
    assert frame.shape[1] == 848 and frame.shape[0] > 480          # the count panel under the camera view
    assert len(list((next(tmp_path.iterdir()) / "scans").iterdir())) == 2


def test_unchanged_share_matches_by_name_and_position():
    import capture_frames
    before = [{"tile_name": "8D", "xy": (0.20, 0.05)}, {"tile_name": "2B", "xy": (0.15, 0.00)}]
    moved = [{"tile_name": "8D", "xy": (0.20, 0.05)}, {"tile_name": "2B", "xy": (0.15, 0.04)}]
    swapped = [{"tile_name": "3C", "xy": (0.20, 0.05)}, {"tile_name": "2B", "xy": (0.15, 0.00)}]
    assert capture_frames.unchanged_share(before, before) == 1.0
    assert capture_frames.unchanged_share(before, moved) == 0.5
    assert capture_frames.unchanged_share(before, swapped) == 0.5
    assert capture_frames.unchanged_share(None, before) == 0.0


def test_count_panel_covers_every_name_once():
    import capture_frames
    assert sorted(n for column in capture_frames.COLUMNS for n in column) == sorted(detector.TILE_NAMES)
