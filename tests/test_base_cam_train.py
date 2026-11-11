"""base_cam_train.py (can the base camera's view be learned) and teleop_view.py (both cameras while driving), with fakes."""
import json
import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import camera_poses, config


def sim_session(monkeypatch, tmp_path, holds=4):
    import holdup_capture
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    return holdup_capture.main(["--sim", "--tiles", "5C,3C,2C", "--holds", str(holds), "--frames", "1", "--gap-s", "0"],
                               ask=lambda prompt: "")


def test_check_trains_on_the_early_holds_and_scores_the_last_ones(tmp_path, monkeypatch):
    import base_cam_train
    out = sim_session(monkeypatch, tmp_path)
    assert base_cam_train.find_sessions(None) == []                               # a sim session never counts by default
    with pytest.raises(SystemExit):
        base_cam_train.main([])
    monkeypatch.setattr(base_cam_train, "find_sessions", lambda arg: [out])
    seen = {"seeds": []}

    def train(photos, seed, epochs, log, crop_fn):
        seen["seeds"].append(seed)
        seen["train"] = photos
        seen["crop_fn"] = crop_fn
        return {"classes": ["5C", "3C", "2C"]}

    def predict(checkpoint, images, crop_fn):
        seen["tested"] = len(images)
        return ["5C"] * len(images)                                             # one answer for everything

    with pytest.raises(SystemExit):                                             # hand-held: no tile boxes to frame on
        base_cam_train.main(["--by", "hold", "--seeds", "1", "--epochs", "1"], train_fn=train, predict_fn=predict)
    result = base_cam_train.main(["--by", "hold", "--seeds", "2", "--epochs", "1", "--framing", "whole"],
                               train_fn=train, predict_fn=predict)
    assert seen["seeds"] == [0, 1] and seen["tested"] == 12 and len(seen["train"]) == 12
    assert {p["name"] for p in seen["train"]} == {"5C", "3C", "2C"}
    assert result["verdict"].startswith("FAIL") and abs(result["mean_right"] - 1 / 3) < 1e-9
    assert result["control_right"] is None                                      # no boxes, no control
    assert json.loads((out / "holdup_check.json").read_text())["split"].startswith("held out the last 2 holds")


def test_verdict_bar():
    import base_cam_train
    assert base_cam_train.verdict(0.9, 4, 0.25, control=0.3).startswith("PASS")
    assert base_cam_train.verdict(0.9, 2, 0.5).startswith("only 2")
    assert base_cam_train.verdict(0.3, 4, 0.25).startswith("FAIL")
    assert base_cam_train.verdict(0.55, 4, 0.25).startswith("IN BETWEEN")
    assert base_cam_train.verdict(0.75, 4, 0.7).startswith("IN BETWEEN")            # near the best constant answer: no pass


def test_verdict_control():
    import base_cam_train
    assert base_cam_train.verdict(1.0, 3, 0.57, control=0.99).startswith("BACKGROUND")   # 2026-11-06's arm photos
    assert base_cam_train.verdict(0.3, 3, 0.4, control=0.6).startswith("BACKGROUND")     # the background beats the tile itself
    assert base_cam_train.verdict(0.9, 3, 0.34, control=0.45).startswith("PASS")         # within the margin of the constant
    assert base_cam_train.verdict(0.9, 3, 0.34).startswith("UNCONTROLLED")               # no boxes: a pass isn't a pass


def arm_like(out, poses):
    """Turns a simulated hand-held session into one the arm took: each shot's frames from `poses` in turn."""
    lines, n = [], 0
    for line in (out / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["type"] == "holdup_shot":
            e["frame_poses"] = []
            for _ in e["files"]:
                e["frame_poses"].append(poses[n % len(poses)])
                n += 1
        lines.append(json.dumps(e))
    (out / "events.jsonl").write_text("\n".join(lines) + "\n")


def test_check_frames_on_the_tile_box_and_runs_the_control(tmp_path, monkeypatch):
    import base_cam_train
    from mahjong import base_cam
    out = sim_session(monkeypatch, tmp_path)
    arm_like(out, ["v1"])
    monkeypatch.setattr(config, "CAMERA_POSES", tmp_path / "poses.json")
    camera_poses.save("v1", {j: 0.0 for j in config.ARM_JOINTS}, "test")
    camera_poses.set_tile_box("v1", [300, 200, 100, 0], "test")                # corners in any order
    assert base_cam.tile_boxes() == {"v1": [100, 0, 300, 200]}
    assert {r["pose"] for r in base_cam.rows(out)} == {"v1"}
    monkeypatch.setattr(base_cam_train, "find_sessions", lambda arg: [out])
    _, test_rows, _, _ = base_cam.split(base_cam.rows(out), "hold", 2)
    truth = [r["truth"] for r in test_rows]
    seen = {"train": 0, "shapes": set()}

    def train(photos, seed, epochs, log, crop_fn):
        seen["train"] += 1
        seen["shapes"] |= {crop_fn(cv2.imread(str(p["path"]))).shape for p in photos}
        return {}

    result = base_cam_train.main(["--by", "hold", "--seeds", "2", "--epochs", "1"], train_fn=train,
                               predict_fn=lambda checkpoint, images, crop_fn: truth)     # right whatever it's shown
    assert seen["train"] == 4                                                   # 2 seeds, then 2 for the control
    assert seen["shapes"] == {(name_check_px(), name_check_px(), 3)}
    assert result["control_right"] == 1.0 and result["verdict"].startswith("BACKGROUND")


def name_check_px():
    from mahjong import name_check
    return name_check.INPUT_PX


def test_mark_boxes_saves_a_box_per_pose_and_keeps_one_left_alone(tmp_path, monkeypatch):
    import base_cam_train
    out = sim_session(monkeypatch, tmp_path, holds=2)
    arm_like(out, ["v1", "v2"])
    monkeypatch.setattr(config, "CAMERA_POSES", tmp_path / "poses.json")
    for name in ("v1", "v2"):
        camera_poses.save(name, {j: 0.0 for j in config.ARM_JOINTS}, "test")
    monkeypatch.setattr(base_cam_train, "find_sessions", lambda arg: [out])
    shown = []

    def select(name, image):
        shown.append((name, image.shape))
        return [10, 20, 110, 220] if name == "v1" else None
    saved = base_cam_train.main(["--mark-boxes"], select=select)
    assert saved == {"v1": [10, 20, 110, 220]} and [n for n, _ in shown] == ["v1", "v2"]
    assert camera_poses.get("v1")["tile_box"] == [10, 20, 110, 220] and "tile_box" not in camera_poses.get("v2")


def test_tile_frame_and_the_greyed_out_control():
    import numpy as np
    from mahjong import base_cam
    image = np.zeros((480, 640, 3), np.uint8)
    image[0:100, 200:300] = 255                                                 # the "tile"
    assert base_cam.tile_frame(image, [200, 0, 300, 100]).mean() > 250            # the crop is the tile
    blank = base_cam.without_tile(image, [190, 0, 310, 110])
    assert blank.max() == base_cam.BLANK and image.max() == 255                   # greyed out, the original untouched


def test_a_chosen_session_is_held_out_and_only_shared_tiles_tested():
    from mahjong import base_cam
    row = lambda s, t, h: {"session": s, "truth": t, "kind": "upright", "hold": h}
    rows = [row("day", "7C", 1), row("day", "EW", 1), row("day", "EW", 2),
            row("evening", "7C", 1), row("evening", "2C", 1), row("late", "1B", 1)]
    train, test, how, caveat = base_cam.split(rows, "hold", test_session="day")          # overrides --by
    assert {r["session"] for r in train} == {"evening", "late"} and [r["truth"] for r in test] == ["7C"]
    assert how.startswith("held out day, trained on evening, late") and "2 photo(s) of EW" in caveat
    assert "1 photo(s) of 2C" in base_cam.split(rows, "auto", test_session="evening")[3]
    assert base_cam.split(rows[:4], "auto", test_session="evening")[3] is None                  # every tile shared: nothing left out
    with pytest.raises(ValueError):
        base_cam.split(rows, "auto", test_session="nowhere")
    with pytest.raises(ValueError):                                                     # 1B is in no other session
        base_cam.split(rows, "auto", test_session="late")


def test_auto_split_by_session_only_when_the_newest_tiles_were_all_trained_on():
    from mahjong import base_cam
    row = lambda s, t, h: {"session": s, "truth": t, "kind": "upright", "hold": h}
    rows = [row("a", "7C", 1), row("a", "7C", 2), row("b", "7C", 1), row("b", "EW", 1), row("b", "EW", 2)]
    assert base_cam.split(rows, "auto")[2].startswith("held out the last")       # EW is only in the newest
    rows.append(row("a", "EW", 1))
    assert base_cam.split(rows, "auto")[2].startswith("held out the newest session")


@pytest.mark.slow
def test_the_real_classifier_learns_two_easy_classes(tmp_path):
    """The real train/predict path on tiny synthetic photos: dark vs light tiles, split into train and test."""
    import cv2
    import numpy as np
    from mahjong import base_cam, name_check
    photos, images, truth = [], [], []
    for name, value in (("5C", 40), ("3C", 215)):
        for i in range(10):
            frame = np.full((480, 640, 3), value, np.uint8)
            cv2.randn(frame, value, 8)
            path = tmp_path / f"{name}_{i}.jpg"
            cv2.imwrite(str(path), frame)
            (photos if i < 7 else images).append({"name": name, "path": path, "session": "s"} if i < 7 else frame)
            if i >= 7:
                truth.append(name)
    checkpoint = name_check.train(photos, epochs=10, log=lambda *_: None, crop_fn=base_cam.whole_frame, device="cpu")
    said = name_check.predict(checkpoint, images, crop_fn=base_cam.whole_frame, device="cpu")
    assert said == truth


def test_camera_poses_save_load_and_name(tmp_path):
    path = tmp_path / "poses.json"
    joints = {j: 1.0 for j in config.JOINTS}
    camera_poses.save("manual_1", joints, "by hand", path=path)
    assert camera_poses.next_name("manual_", path) == "manual_2"
    entry = camera_poses.get("manual_1", path)
    assert set(entry["pose"]) == set(config.ARM_JOINTS) and entry["gripper"] == 1.0
    with pytest.raises(SystemExit):
        camera_poses.get("nope", path)


def test_the_saved_manual_poses_are_there():
    poses = camera_poses.load()
    assert {"manual_1", "manual_2"} <= set(poses)
    assert all(set(p["pose"]) == set(config.ARM_JOINTS) for p in poses.values())


def test_teleop_view_sim_saves_a_pose_and_both_frames(tmp_path, monkeypatch):
    import teleop_view
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(config, "CAMERA_POSES", tmp_path / "poses.json")
    presses = iter(["c", "s", "s", "q"])
    out = teleop_view.main(["--sim", "--name", "t_"], keys=lambda: next(presses, "q"))
    assert sorted(camera_poses.load()) == ["t_1", "t_2"]
    assert sorted(p.name for p in (out / "captures").iterdir()) == ["001_base.jpg", "001_wrist.jpg"]


def test_teleop_view_go_starts_at_a_saved_pose_and_needs_it_to_exist(tmp_path, monkeypatch):
    import teleop_view
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(config, "CAMERA_POSES", tmp_path / "poses.json")
    with pytest.raises(SystemExit):
        teleop_view.main(["--sim", "--go", "missing"])
    camera_poses.save("mine", camera_poses.load(config.REPO / "data" / "camera_poses.json")["manual_1"]["pose"], "test")
    teleop_view.main(["--sim", "--go", "mine"], keys=lambda: "q")


def test_score_lists_every_check_and_says_which_have_a_clean_control(tmp_path, monkeypatch):
    import holdup_score
    from mahjong import base_cam
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    run = lambda right, constant, tiles: {"photos": 24, "tiles": tiles, "right": right, "best_constant_answer": constant,
                                          "per_tile": {"1B": right, "7C": right}, "confusions": [["1B->7C", 2]]}
    day, evening = tmp_path / "20261106_105954_holdup", tmp_path / "20261106_195636_holdup"
    for folder in (day, evening):
        folder.mkdir()
    (day / "holdup_check.json").write_text(json.dumps({          # an old file: no sizes, no time
        "split": "held out the newest session, x", "caveat": None, "mean_right": 1.0, "runs": [run(1.0, 0.44, 3)],
        "control_right": 0.99, "verdict": "BACKGROUND: the scene names it"}))
    import datetime, os
    old = datetime.datetime(2026, 11, 6, 13, 44).timestamp()
    os.utime(day / "holdup_check.json", (old, old))                           # dated by its file
    (evening / "holdup_check.json").write_text(json.dumps({
        "written": "2026-11-06T22:51", "split": "held out the last 2 holds of every tile and way up", "caveat": "OPTIMISTIC: x",
        "train_photos": 576, "train_holds": 48, "test_photos": 288, "test_holds": 24, "mean_right": 1.0,
        "runs": [run(1.0, 0.17, 6)], "control_right": 0.0, "verdict": "PASS: worth a real capture"}))
    (day / "holdup_check_held_out.json").write_text(json.dumps({
        "written": "2026-11-07T09:00", "split": "held out 20261106_105954_holdup, trained on 20261106_195636_holdup",
        "caveat": "left out 96 photo(s) of EW: in no other session, never trained on", "train_photos": 864, "train_holds": 72,
        "test_photos": 120, "test_holds": 10, "mean_right": 0.99, "runs": [run(0.99, 0.60, 2)], "control_right": 0.60,
        "verdict": "only 2 tile(s) held out: too few to judge"}))
    results = holdup_score.main([])
    assert [r["file"].split("/")[1] for r in results][-2:] == ["holdup_check.json", "holdup_check_held_out.json"]
    text = "\n".join(base_cam.check_lines(results))
    assert "BACKGROUND" in text and "PASS" in text and "TOO FEW TILES" in text
    assert "train 576 photos / 48 holds" in text and "left out 96 photo(s) of EW" in text and "OPTIMISTIC" not in text
    assert "2 of 3 result(s) with a clean control" in text
    assert "on a session never trained on: 99% on 1B, 7C (20261106_105954_holdup)" in text
    assert holdup_score.main([str(evening)])[0]["file"].startswith(evening.name)       # one session's results only
    assert "No base_cam_train.py results yet" in base_cam.check_lines([])[0]
