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


def test_ignored_names_count_for_nothing_but_still_mix_up_others():
    truth = [box("8D", 1, 100, 100), box("2F", 1, 200, 100), box("1F", 1, 300, 100), box("5C", 1, 400, 100)]
    said = [box("8D", 0.9, 100, 100), box("4F", 0.8, 200, 100),          # a flower misnamed: ignored
            box("2F", 0.7, 401, 100),                                   # 5C read as a flower: still a mix-up
            box("3F", 0.6, 305, 104, 30, 30)]                           # loose box on a flower: not "extra"
    s = detector.summary(detector.score([{"truth": truth, "said": said}], ignore=("1F", "2F", "3F", "4F"))["all"])
    assert (s["tiles"], s["found"], s["named_right"], s["missed"], s["extra"]) == (2, 2, 1, 0, 0)
    assert s["mixups"] == [["5C", "2F", 1]] and s["right_rate"] == 0.5


def test_dedupe_merges_a_label_drawn_twice():
    kept, dropped = detector.dedupe([box("4C", 1, 100, 100), box("4C", 1, 101, 100), box("5C", 1, 100, 100),
                                     box("4C", 1, 200, 100)])
    assert [b["tile_name"] for b in kept] == ["4C", "5C", "4C"] and [b["tile_name"] for b in dropped] == ["4C"]


def test_a_tile_read_as_two_names_is_prelabelled_once():
    kept = detector.one_box_per_tile([box("5B", 0.6, 100, 100), box("7B", 0.8, 101, 100), box("4B", 0.7, 200, 100)])
    assert sorted(b["tile_name"] for b in kept) == ["4B", "7B"]


def test_split_by_scene_keeps_each_scene_on_one_side(tmp_path):
    import cv2
    import split_by_scene
    src = tmp_path / "export"
    names = detector.TILE_NAMES + ["aruco_marker"]
    scene_a = [box("1B", 1, 100, 100), box("2B", 1, 200, 100), box("3B", 1, 300, 100)]
    scene_b = [box("4C", 1, 100, 300), box("5C", 1, 200, 300)]
    frames = {("train", "a_exp200"): scene_a, ("valid", "a_exp300"): scene_a, ("test", "a_exp400"): scene_a,
              ("train", "b_exp200"): scene_b + [box("aruco_marker", 1, 600, 50)], ("valid", "b_exp300"): scene_b,
              ("train", "twins"): [box("6D", 1, 400, 200), box("6D", 1, 401, 200)],
              ("train", "clash"): [box("5B", 1, 500, 200), box("7B", 1, 501, 200)]}
    for (split, stem), boxes in frames.items():
        (src / split / "images").mkdir(parents=True, exist_ok=True)
        (src / split / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(src / split / "images" / f"{stem}.jpg"), np.zeros((480, 848, 3), np.uint8))
        (src / split / "labels" / f"{stem}.txt").write_text("\n".join(detector.yolo_lines(boxes, names, (848, 480))))
    (src / "data.yaml").write_text(f"nc: {len(names)}\nnames: {names}\n")

    split_by_scene.main(["--data", str(src / "data.yaml"), "--out", str(tmp_path / "out"), "--valid", "0.4"])
    out = {p.stem: p.parent.parent.name for p in (tmp_path / "out").glob("*/images/*.jpg")}
    assert "clash" not in out and len(out) == 6                                 # two names on one tile: left out
    assert len({out[s] for s in ("a_exp200", "a_exp300", "a_exp400")}) == 1 == len({out["b_exp200"], out["b_exp300"]})
    assert set(out.values()) == {"train", "valid"}
    b = detector.read_yolo_labels(tmp_path / "out" / out["b_exp200"] / "labels" / "b_exp200.txt",
                                  detector.TILE_NAMES, (848, 480))
    assert sorted(x["tile_name"] for x in b) == ["4C", "5C"]                     # aruco_marker dropped
    assert len((tmp_path / "out" / out["twins"] / "labels" / "twins.txt").read_text().splitlines()) == 1


@pytest.mark.parametrize("with_backs", [False, True])
def test_split_by_scene_keeps_backs_only_when_asked(tmp_path, with_backs):
    import cv2
    import yaml
    import split_by_scene
    src = tmp_path / "export"
    names = detector.TILE_NAMES + [config.BACK]
    for stem, boxes in {"a": [box("1B", 1, 100, 100), box(config.BACK, 1, 200, 100)],
                        "b": [box(config.BACK, 1, 100, 300), box(config.BACK, 1, 200, 300)]}.items():
        (src / "train" / "images").mkdir(parents=True, exist_ok=True)
        (src / "train" / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(src / "train" / "images" / f"{stem}.jpg"), np.zeros((480, 848, 3), np.uint8))
        (src / "train" / "labels" / f"{stem}.txt").write_text("\n".join(detector.yolo_lines(boxes, names, (848, 480))))
    (src / "data.yaml").write_text(f"nc: {len(names)}\nnames: {names}\n")
    split_by_scene.main(["--data", str(src / "data.yaml"), "--out", str(tmp_path / "out"), "--valid", "0.5"]
                        + (["--with-backs"] if with_backs else []))
    data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
    assert data["names"] == (names if with_backs else detector.TILE_NAMES) and data["nc"] == len(data["names"])
    (a,) = (tmp_path / "out").glob("*/labels/a.txt")
    kept = sorted(x["tile_name"] for x in detector.read_yolo_labels(a, data["names"], (848, 480)))
    assert kept == (["1B", config.BACK] if with_backs else ["1B"])


def test_a_back_is_scored_like_any_class():
    truth = [box(config.BACK, 1, 100, 100), box("5C", 1, 200, 100), box(config.BACK, 1, 300, 100)]
    said = [box(config.BACK, 0.8, 101, 100), box(config.BACK, 0.7, 200, 100)]     # one missed, 5C taken for a back
    s = detector.summary(detector.score([{"truth": truth, "said": said}])["all"])
    assert (s["tiles"], s["found"], s["named_right"], s["missed"]) == (3, 2, 1, 1)


def test_ignore_option_expands_groups_and_rejects_unknown_names():
    import detector_eval
    assert detector_eval.parse_names("flowers,ew") == ("1F", "2F", "3F", "4F", "EW")
    with pytest.raises(Exception):
        detector_eval.parse_names("9F")


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
    image = out / "train" / "images" / f"{session.dir.name}__001_capture__f0.jpg"     # Roboflow's export layout
    assert image.exists()
    labels = detector.read_yolo_labels(out / "train" / "labels" / f"{image.stem}.txt", detector.TILE_NAMES, (848, 480))
    assert sorted(b["tile_name"] for b in labels) == sorted(t["tile_name"] for t in tiles)
    assert sorted(p.name for p in out.iterdir()) == ["data.yaml", "train"]       # nothing else for Roboflow to misread
    assert detector.manifest_path(out).exists()
    import yaml
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == detector.TILE_NAMES and data["nc"] == 38 and data["train"] == "../train/images"
    assert [(p.name, n) for p, _, n in detector.exam_set(out)][:1] == [(image.name, detector.TILE_NAMES)]

    # the same session as an exam session: out of training exports, into exam ones
    exam = {"sessions": {session.dir.name: {"conditions": {}}}}
    assert mine_frames.pick([session.dir], exam, everything=True)[0] == []
    assert len(mine_frames.pick([session.dir], exam, exam_only=True)[0]) == 2
    assert detector.exam_frames(list((out / "train" / "images").iterdir()), exam)

    # a session given by name alone is found under sessions/; one that isn't there stops with why
    assert mine_frames.session_folders([Path(session.dir.name)]) == [session.dir]
    with pytest.raises(SystemExit, match="No scans in"):
        mine_frames.session_folders([Path("20990101_000000_capture")])


def test_face_down_frames_export_a_back_box_at_each_spot_with_no_face_on_it(tmp_path, monkeypatch):
    import json
    import capture_frames
    from mahjong.sim import SimPerception
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    spots = [(0.160, -0.165), (0.21, 0.02)]
    # 8D face up on the first spot, a back on the second that this detector can't see (the deployed weights)
    mat = SimPerception(tiles=[("8D", *spots[0]), (config.BACK, *spots[1])], sees_backs=False)
    session = Session("capture", ask_setup=False)
    tiles, _ = capture_frames.take_scan(mat, session, 1)
    session.close()
    assert [t["tile_name"] for t in tiles] == ["8D"]
    rows, _ = __import__("mine_frames").pick([session.dir], {"sessions": {}}, everything=True)
    out = tmp_path / "export"
    detector.export(rows, out, back_spots=spots)
    import yaml
    names = yaml.safe_load((out / "data.yaml").read_text())["names"]
    assert names == detector.TILE_NAMES + [config.BACK]
    (label_file,) = (out / "train" / "labels").iterdir()
    labels = detector.read_yolo_labels(label_file, names, (848, 480))
    assert sorted(b["tile_name"] for b in labels) == sorted(["8D", config.BACK])
    # where a face-up tile's box would be on that spot: the outline is the same projection
    face = SimPerception(tiles=[("2C", *spots[1])])
    face.detect_tiles()
    (expected,) = face.last_frames[0]["boxes"]
    (back,) = [b for b in labels if b["tile_name"] == config.BACK]
    assert detector.iou(back["xyxy"], expected["xyxy"]) > 0.9

    # a scan saved before the camera went into reads.json can't be pre-labelled
    reads_path = rows[0]["folder"] / "reads.json"
    reads = json.loads(reads_path.read_text())
    reads_path.write_text(json.dumps({k: v for k, v in reads.items() if k not in ("intrinsics", "T_cam_to_base")}))
    with pytest.raises(SystemExit, match="without its camera"):
        detector.export(rows, tmp_path / "export2", back_spots=spots)


def test_mine_frames_backs_layout_needs_the_sessions_named():
    import mine_frames
    with pytest.raises(SystemExit):
        mine_frames.parse_args(["--backs-layout", "even-hand14"])


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
    detector.export(rows, out, split="test")
    test = out / "test"
    (test / "images" / "frame_0000_exp200_jpg.rf.abc.jpg").write_bytes((test / "images" / f"{rows[0]['session']}__001_capture__f0.jpg").read_bytes())
    (test / "labels" / "frame_0000_exp200_jpg.rf.abc.txt").write_text("")
    truth = mat.last_frames[0]["boxes"]
    wrong_first = [{**truth[0], "tile_name": "WW" if truth[0]["tile_name"] != "WW" else "RD"}] + truth[1:]
    exam = {"sessions": {exam_session.dir.name: {"conditions": {"daylight": "dim", "lamp": True, "ceiling_lights": False}}}}
    images, outside, duplicates = detector.exam_images(out, lambda image: wrong_first, exam, by="lighting")
    scores = detector.score(images)
    assert [p.name for p in outside] == ["frame_0000_exp200_jpg.rf.abc.jpg"] and duplicates == []
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


def test_back_gate_numbers_count_found_backs_and_false_backs():
    truth = [box(config.BACK, 1, 100, 100), box(config.BACK, 1, 200, 100), box("5C", 1, 300, 100)]
    said = [box(config.BACK, 0.9, 101, 100),       # found, called back
            box("5C", 0.8, 200, 100),              # boxed, but named as a face: not found as a back
            box(config.BACK, 0.7, 300, 101),       # a face read as back
            box(config.BACK, 0.6, 500, 400)]       # a back on no tile
    face_up = {"truth": [box("5C", 1, 300, 100)], "said": [box(config.BACK, 0.7, 300, 101)]}     # one false back
    s = detector.summary(detector.score([{"truth": truth, "said": said}, face_up, {"truth": [], "said": []}])["all"])
    assert {k: s["backs"][k] for k in ("labelled", "boxed", "called_back", "false", "extra", "faces_read_as_back")} == \
        {"labelled": 2, "boxed": 2, "called_back": 1, "false": 3, "extra": 1, "faces_read_as_back": 2}
    # the gate's rate counts only the face-up frames (no back labelled): 1 false back in 2, not 3 in 3
    assert s["backs"]["found_rate"] == 0.5 and s["backs"]["face_up_frames"] == 2 and s["backs"]["false_per_frame"] == 0.5
    assert (s["faces"]["tiles"], s["faces"]["found"], s["faces"]["named_right"]) == (2, 2, 0)


# a camera 0.5m over the table looking straight down: 1px is ~0.8mm at the tile tops
CAMERA = {"intrinsics": {"fx": 600.0, "fy": 600.0, "ppx": 424.0, "ppy": 240.0},
          "T_cam_to_base": [[0, -1, 0, 0.2], [-1, 0, 0, 0.0], [0, 0, -1, 0.5], [0, 0, 0, 1]], "size": [848, 480]}


def flipped_scans(sessions, shift_px, tiles=8, session="20261116_090000_capture"):
    """Scan 1 with `tiles` faces in a row, scan 2 with each flipped in place; the detector puts each back box shift_px along
    u from its face box. Returns exam_images-style frames."""
    images = []
    for n in (1, 2):
        (sessions / session / "scans" / f"00{n}_capture").mkdir(parents=True)
        (sessions / session / "scans" / f"00{n}_capture" / "reads.json").write_text(json.dumps({**CAMERA, "frames": []}))
        name = "5C" if n == 1 else config.BACK
        truth = [box(name, 1, 100 + 40 * i, 200) for i in range(tiles)]
        shift = shift_px if callable(shift_px) else (lambda i, k=shift_px: k)
        said = [box(name, 0.9, 100 + 40 * i + (shift(i) if n == 2 else 0), 200) for i in range(tiles)]
        images.append({"frame": Path(f"{session}__00{n}_capture__f0_jpg.rf.abc.jpg"), "size": (848, 480),
                       "truth": truth, "said": said, "group": "all"})
    return images


def test_back_centres_pass_when_backs_sit_where_faces_did(tmp_path):
    result = detector.back_centres(flipped_scans(tmp_path, 0), sessions_dir=tmp_path)
    assert result["flips"] == result["measured"] == 8 and result["passes"]
    assert abs(result["median_dx_mm"]) < 0.01 and abs(result["median_dy_mm"]) < 0.01


def test_back_centres_fail_on_a_steady_offset_or_too_few_tiles(tmp_path):
    shifted = detector.back_centres(flipped_scans(tmp_path / "a", 5), sessions_dir=tmp_path / "a")
    assert not shifted["passes"] and abs(shifted["median_dy_mm"]) > 3     # u runs along -y with this camera
    few = detector.back_centres(flipped_scans(tmp_path / "b", 0, tiles=3), sessions_dir=tmp_path / "b")
    assert few["measured"] == 3 and not few["passes"]


def test_back_centres_need_the_scans_camera(tmp_path):
    result = detector.back_centres(flipped_scans(tmp_path, 0), sessions_dir=tmp_path / "elsewhere")
    assert result["flips"] == 8 and result["measured"] == 0 and result["no_camera"] == 8 and not result["passes"]


def test_a_back_sample_joins_the_fits_once_the_weights_pass(tmp_path, monkeypatch):
    import bench
    best = tmp_path / "best.pt"
    best.write_bytes(b"weights")
    installed = tmp_path / "tile_detector.pt"
    installed.write_bytes(b"weights")                     # best.pt copied into assets: the same hash, another name
    monkeypatch.setattr(config, "DETECTOR_EVAL", tmp_path / "eval")
    (tmp_path / "eval").mkdir()
    assert not detector.backs_checked(installed)
    failing = detector.back_centres(flipped_scans(tmp_path / "a", 5), sessions_dir=tmp_path / "a")
    detector.back_centres_path(best).write_text(json.dumps(failing))
    assert not detector.backs_checked(installed)
    passing = detector.back_centres(flipped_scans(tmp_path / "b", 0), sessions_dir=tmp_path / "b")
    detector.back_centres_path(best).write_text(json.dumps(passing))
    assert detector.backs_checked(installed)
    detector.back_centres_path(best).write_text(json.dumps({**passing, "passes": True, "p90_spread_mm": 9.0}))
    assert not detector.backs_checked(installed)          # judged by today's gates, not the saved flag
    back, face = {"tile_name": config.BACK}, {"tile_name": "5C"}
    assert bench.back_excluded(back, False) == {"excluded": bench.BACK_UNCHECKED} and bench.back_excluded(face, False) == {}
    assert bench.back_excluded(back, True) == {}


def test_face_down_frames_split_by_session_into_both_splits():
    import split_by_scene
    spots = [box(config.BACK, 1, 100 + 40 * i, 200) for i in range(14)]     # every face-down frame: the same 14 spots
    frames = [{"image": Path(f"2026111{s}_090000_capture__00{n}_capture__f0.jpg"), "boxes": spots + [box("5C", 1, 100, 300)]}
              for s in range(5) for n in range(1, 4)]
    frames += [{"image": Path(f"july_{i}.jpg"), "boxes": [box("1B", 1, 50 * i, 50)]} for i in range(10)]
    split_by_scene.split(frames, valid=0.2)
    backs = [f for f in frames if split_by_scene.origin_of(f) == "face-down frames"]
    by_session = {}
    for f in backs:
        by_session.setdefault(detector.session_of(f["image"].name), set()).add(f["to"])
    assert all(len(to) == 1 for to in by_session.values())                   # a session stays on one side
    assert {f["to"] for f in backs} == {"train", "valid"}                    # ...and backs are on both


def test_train_detector_refuses_a_class_with_no_training_boxes(tmp_path):
    import cv2
    import train_detector
    names = detector.TILE_NAMES + [config.BACK]
    for split, boxes in (("train", [box("1B", 1, 100, 100)]), ("valid", [box(config.BACK, 1, 100, 100)])):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        cv2.imwrite(str(tmp_path / split / "images" / "a.jpg"), np.zeros((480, 848, 3), np.uint8))
        (tmp_path / split / "labels" / "a.txt").write_text("\n".join(detector.yolo_lines(boxes, names, (848, 480))))
    (tmp_path / "data.yaml").write_text(f"nc: {len(names)}\nnames: {names}\n")
    with pytest.raises(SystemExit, match="No boxes in train for back"):
        train_detector.main(["--data", str(tmp_path / "data.yaml"), "--name", "x", "--dry-run"])


def test_one_big_scene_does_not_overfill_valid():
    import split_by_scene
    big = [{"image": Path(f"20261028_090000_bench__00{n}_x__f0.jpg"), "boxes": [box("1B", 1, 100, 100)]} for n in range(40)]
    small = [{"image": Path(f"20261028_100000_bench__00{n}_x__f0.jpg"), "boxes": [box("2B", 1, 30 * n, 300)]}
             for n in range(60)]
    for seed in range(5):
        frames = [dict(f) for f in big + small]
        split_by_scene.split(frames, valid=0.2, seed=seed)
        valid = sum(f["to"] == "valid" for f in frames)
        assert 15 <= valid <= 25 * split_by_scene.OVERFILL


def test_back_centres_fail_when_backs_scatter_about_the_face(tmp_path):
    result = detector.back_centres(flipped_scans(tmp_path, lambda i: 8 if i % 2 else -8), sessions_dir=tmp_path)
    assert abs(result["median_dy_mm"]) < 0.5 and result["p90_spread_mm"] > 4 and not result["passes"]


def test_a_tile_counts_once_however_many_frames_of_its_scan_were_exported(tmp_path):
    images = flipped_scans(tmp_path, 0, tiles=3)
    images += [{**i, "frame": Path(str(i["frame"]).replace("__f0", "__f3"))} for i in images]
    assert len(detector.flips(images)) == 3


def test_a_smaller_centre_check_does_not_replace_a_larger_one(tmp_path, monkeypatch):
    import detector_eval
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(config, "DETECTOR_EVAL", tmp_path / "eval")
    big = detector.back_centres(flipped_scans(tmp_path / "a", 0, tiles=10), sessions_dir=tmp_path / "a")
    small = detector.back_centres(flipped_scans(tmp_path / "b", 5, tiles=9), sessions_dir=tmp_path / "b")
    assert detector_eval.save_centres(weights, big, "exam_a")
    assert detector_eval.save_centres(weights, small, "exam_b") is None
    assert detector.backs_checked(weights)
    assert detector_eval.save_centres(weights, small, "exam_a")          # the same exam again: replaced
    assert not detector.backs_checked(weights)


def split_frames(sessions, frames_each=10, backs=True):
    spots = [box(config.BACK, 1, 100 + 40 * i, 200) for i in range(14)] if backs else []
    return [{"image": Path(f"{s}__{n:03d}_capture__f0.jpg"), "boxes": spots + [box("5C", 1, 100 + 9 * n, 300)]}
            for s in sessions for n in range(1, frames_each + 1)]


def test_one_face_down_session_stays_in_train():
    import split_by_scene
    frames = split_frames(["20261114_090000_capture"])
    split_by_scene.split(frames, valid=0.2)
    assert {f["to"] for f in frames} == {"train"}


def test_a_back_session_keeps_its_face_up_frames_and_a_back_frame_needs_no_session():
    import split_by_scene
    frames = split_frames([f"2026111{s}_090000_capture" for s in range(5)], frames_each=3)
    frames += split_frames(["20261110_090000_capture"], frames_each=2, backs=False)       # face up only: by labels
    frames += [{**frames[0], "boxes": frames[1]["boxes"][-1:], "image": Path("20261111_090000_capture__009_capture__f0.jpg")},
               {**frames[0], "image": Path("renamed.jpg")}]                               # no session: a scene of its own
    split_by_scene.split(frames, valid=0.2)
    by_session = {}
    for f in frames:
        if split_by_scene.origin_of(f, split_by_scene.back_sessions(frames)) == "face-down frames":
            by_session.setdefault(detector.session_of(f["image"].name), set()).add(f["to"])
    assert by_session["20261111_090000_capture"] and all(len(v) == 1 for k, v in by_session.items() if k)


def test_a_hand_that_nudged_every_tile_does_not_hide_a_bias(tmp_path):
    images = flipped_scans(tmp_path, 0)
    for t in images[1]["truth"]:              # the labels say the hand moved every tile 3px along u; the detector didn't
        t["xyxy"] = [t["xyxy"][0] + 3, t["xyxy"][1], t["xyxy"][2] + 3, t["xyxy"][3]]
    result = detector.back_centres(images, sessions_dir=tmp_path)
    assert abs(result["median_dy_mm"]) < 0.01 and abs(result["less_hand_median_dy_mm"]) > 2 and not result["passes"]


def test_flips_take_a_scans_lowest_frame_and_skip_a_scan_not_exported(tmp_path):
    images = flipped_scans(tmp_path, 0, tiles=3)
    images[1]["frame"] = Path(str(images[1]["frame"]).replace("002_capture", "004_capture"))     # 002-003 not exported
    images.append({**images[1], "frame": Path(str(images[1]["frame"]).replace("__f0", "__f10")), "truth": []})
    pairs = detector.flips(images)
    assert len(pairs) == 3 and all(detector.scan_of(b["frame"])[2] == 0 for _, _, b, _ in pairs)


def test_valid_gets_the_smallest_session_when_every_one_is_too_big():
    import split_by_scene
    frames = split_frames(["20261114_090000_capture"], frames_each=30) + split_frames(["20261114_100000_capture"],
                                                                                      frames_each=20)
    split_by_scene.split(frames, valid=0.2)
    assert {f["to"] for f in frames if "100000" in f["image"].name} == {"valid"}
    assert {f["to"] for f in frames if "090000" in f["image"].name} == {"train"}
