"""holdup_check.py (can the base camera's view be learned) and teleop_view.py (both cameras while driving), with fakes."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import camera_poses, config


def sim_session(monkeypatch, tmp_path, holds=4):
    import holdup_capture
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    return holdup_capture.main(["--sim", "--tiles", "5C,3C,2C", "--holds", str(holds), "--frames", "1", "--gap-s", "0"],
                               ask=lambda prompt: "")


def test_check_trains_on_the_early_holds_and_scores_the_last_ones(tmp_path, monkeypatch):
    import holdup_check
    out = sim_session(monkeypatch, tmp_path)
    assert holdup_check.find_sessions(None) == []                               # a sim session never counts by default
    with pytest.raises(SystemExit):
        holdup_check.main([])
    monkeypatch.setattr(holdup_check, "find_sessions", lambda arg: [out])
    seen = {"seeds": []}

    def train(photos, seed, epochs, log, crop_fn):
        seen["seeds"].append(seed)
        seen["train"] = photos
        seen["crop_fn"] = crop_fn
        return {"classes": ["5C", "3C", "2C"]}

    def predict(checkpoint, images, crop_fn):
        seen["tested"] = len(images)
        return ["5C"] * len(images)                                             # one answer for everything

    result = holdup_check.main(["--by", "hold", "--seeds", "2", "--epochs", "1"], train_fn=train, predict_fn=predict)
    assert seen["seeds"] == [0, 1] and seen["tested"] == 12 and len(seen["train"]) == 12
    assert {p["name"] for p in seen["train"]} == {"5C", "3C", "2C"}
    assert result["verdict"].startswith("FAIL") and abs(result["mean_right"] - 1 / 3) < 1e-9
    assert json.loads((out / "holdup_check.json").read_text())["split"].startswith("held out the last 2 holds")


def test_verdict_bar():
    import holdup_check
    assert holdup_check.verdict(0.9, 4, 0.25).startswith("PASS")
    assert holdup_check.verdict(0.9, 2, 0.5).startswith("only 2")
    assert holdup_check.verdict(0.3, 4, 0.25).startswith("FAIL")
    assert holdup_check.verdict(0.55, 4, 0.25).startswith("IN BETWEEN")
    assert holdup_check.verdict(0.75, 4, 0.7).startswith("IN BETWEEN")            # near the best constant answer: no pass


@pytest.mark.slow
def test_the_real_classifier_learns_two_easy_classes(tmp_path):
    """The real train/predict path on tiny synthetic photos: dark vs light tiles, split into train and test."""
    import cv2
    import numpy as np
    from mahjong import holdup, name_check
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
    checkpoint = name_check.train(photos, epochs=10, log=lambda *_: None, crop_fn=holdup.whole_frame, device="cpu")
    said = name_check.predict(checkpoint, images, crop_fn=holdup.whole_frame, device="cpu")
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
