import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import config, kinematics
from mahjong import setup_check as sc
from mahjong.session import read_events


def turned(T, deg, axis):
    """T with the camera turned `deg` about its own x / y / z axis (a knocked tripod)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    R = {0: [[1, 0, 0], [0, c, -s], [0, s, c]], 1: [[c, 0, s], [0, 1, 0], [-s, 0, c]], 2: [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
    M = np.eye(4)
    M[:3, :3] = R
    return T @ M


def sim_devices():
    from mahjong.arm import Arm
    from mahjong.sim import SimPerception, SimRobot, SimWristCamera
    realsense = SimPerception(tiles=[])
    arm = Arm(SimRobot(), sleep=lambda s: None)
    return {"arm": arm, "wrist_camera": SimWristCamera(arm, realsense), "realsense": realsense}


def run(tmp_path, argv, answers=(), devices=None):
    """One run, in a sessions folder of its own (two runs in the same second would share a folder name)."""
    import setup_check
    answers = iter(answers)
    config.SESSIONS = tmp_path / "sessions" / str(len(list((tmp_path / "sessions").glob("*"))))
    return setup_check.main(["--sim", "--reference", str(tmp_path / "reference"), *argv],
                            ask=lambda prompt: next(answers), devices=devices or sim_devices())


def last_check(tmp_path):
    session = max((tmp_path / "sessions").glob("*/*_setup"), key=lambda p: int(p.parent.name))
    return next(e for e in read_events(session) if e["type"] == "setup_check"), session


def test_an_unmoved_setup_passes_and_a_knocked_camera_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    assert run(tmp_path, ["--save-reference"], ["y"]) == "pass"
    reference = sc.load_reference(tmp_path / "reference")
    assert [s["id"] for s in reference["spots"]] == ["F2", "F3", "F6"] and reference["realsense_image"] is not None

    assert run(tmp_path, []) == "pass"
    check, session = last_check(tmp_path)
    assert check["results"]["camera"]["shift_mm"] < 0.5 and check["results"]["tiles"]["status"] == "pass"
    assert (session / "wrist_F3_vs_reference.jpg").exists() and (session / "camera_vs_reference.jpg").exists()

    devices = sim_devices()
    devices["realsense"].T_render = turned(devices["realsense"].T_cam_to_base, 0.5, 0)   # tilted ~0.5°: ~8mm on the mat
    assert run(tmp_path, [], devices=devices) == "fail"
    results = last_check(tmp_path)[0]["results"]
    assert results["camera"]["status"] == "fail" and 6 < results["camera"]["shift_mm"] < 11
    assert results["tiles"]["status"] in ("warn", "fail")        # the tile placed to green is off by as much
    assert all(r["shift_mm"] > 3 for r in results["tiles"]["spots"].values())


def test_saving_asks_before_replacing_a_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    assert run(tmp_path, ["--save-reference", "--skip-tiles"], ["y"]) == "pass"
    before = (tmp_path / "reference" / "reference.json").read_text()
    import setup_check
    prompts = []
    config.SESSIONS = tmp_path / "sessions" / "9"
    assert setup_check.main(["--sim", "--reference", str(tmp_path / "reference"), "--save-reference", "--skip-tiles"],
                            ask=lambda prompt: prompts.append(prompt) or "", devices=sim_devices()) == "skipped"
    assert "Replace the reference" in prompts[-1]
    assert (tmp_path / "reference" / "reference.json").read_text() == before


def test_a_tile_not_found_is_repositioned_and_looked_at_again(tmp_path, monkeypatch):
    import setup_check
    from mahjong.sim import SimWristCamera
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    assert run(tmp_path, ["--save-reference"], ["y"]) == "pass"

    class Blind(SimWristCamera):          # sees no tile the first time it looks at F3 (the only spot past x 0.2)
        looks = 0

        def read(self):
            ok, image = super().read()
            x, _, _ = kinematics.tool_point(self.arm.measured())
            if x > 0.2 and Blind.looks < 3:       # the 3 frames of its first look
                Blind.looks += 1
                return ok, np.full_like(image, 25)
            return ok, image

    devices = sim_devices()
    devices["wrist_camera"] = Blind(devices["arm"], devices["realsense"])
    assert run(tmp_path, [], [""], devices=devices) == "pass"        # Enter: placed again, found
    check, session = last_check(tmp_path)
    assert check["results"]["tiles"]["spots"]["F3"]["attempts"] == 2
    assert [e["spot"] for e in read_events(session) if e["type"] == "setup_tile_not_found"] == ["F3"]

    # the RealSense finds no tile (none placed): F2 retried then skipped, F3 stops the tile test
    monkeypatch.setattr(setup_check, "place_sim_tile", lambda realsense, spot: setattr(realsense, "tiles", []))
    run(tmp_path, [], ["", "s", "q"])
    tiles = last_check(tmp_path)[0]["results"]["tiles"]["spots"]
    assert tiles["F2"]["status"] == "skipped" and tiles["F2"]["attempts"] == 2
    assert tiles["F3"]["status"] == "skipped" and "F6" not in tiles


def test_image_shift_reads_a_known_shift():
    from mahjong.sim import SimPerception
    mat = SimPerception(tiles=[])
    reference = mat.snapshot()
    region = sc.robot_region(mat.intrinsics, mat.T_cam_to_base, reference.shape)
    moved = cv2.warpAffine(reference, np.float32([[1, 0, 2], [0, 1, -1]]), reference.shape[1::-1])
    shift = sc.image_shift(reference, moved, region)
    assert abs(shift["dx_px"] - 2) < 0.3 and abs(shift["dy_px"] + 1) < 0.3 and abs(shift["rotation_deg"]) < 0.2
    result = sc.camera_result(shift, sc.mm_per_px(mat.intrinsics, mat.T_cam_to_base))
    assert result["status"] == "warn" and 3 < result["shift_mm"] < 4               # 2.2px, ~1.5mm per pixel
    blank = np.zeros_like(reference)
    assert sc.camera_result(sc.image_shift(reference, blank, region), 1.5)["status"] == "warn"   # can't tell


def test_table_fit_sees_a_mat_lower_than_config():
    from mahjong.sim import SimPerception
    mat = SimPerception(tiles=[])
    _, depth = mat.depth_snapshot()
    level = sc.table_fit(depth, mat.intrinsics, mat.T_cam_to_base)
    assert abs(level["offset_mm"]) < 0.5 and level["samples"] > 100
    mat.T_render = mat.T_cam_to_base.copy()
    mat.T_render[2, 3] += 0.005                     # the camera 5mm higher than calibrated: the mat reads 5mm low
    _, depth = mat.depth_snapshot()
    lower = sc.table_result(sc.table_fit(depth, mat.intrinsics, mat.T_cam_to_base), level)
    assert abs(lower["change"]["offset_mm"] + 5) < 0.5 and lower["status"] == "warn"


def test_wrist_tile_finds_the_face_and_ignores_glare():
    image = np.full((480, 640, 3), 25, np.uint8)
    cv2.fillPoly(image, [np.int32([(270, 180), (380, 180), (380, 330), (270, 330)])], (225, 235, 240))
    cv2.ellipse(image, (80, 400), (60, 20), 30, 0, 300, (200, 200, 200), -1)   # an irregular bright patch
    tile = sc.wrist_tile(image)
    assert np.allclose(tile["centre_px"], (325, 255), atol=2)
    moved = sc.wrist_result({**tile, "centre_px": [tile["centre_px"][0] + 40, tile["centre_px"][1]]}, tile)
    assert moved["status"] in ("warn", "fail") and moved["shift_mm"] > 3
    assert sc.wrist_result(None, tile)["status"] == "fail"


def test_grading():
    assert sc.grade(1.0, "camera_mm") == "pass" and sc.grade(-2.5, "camera_mm") == "warn" and sc.grade(5, "camera_mm") == "fail"
    assert sc.worst(["pass", "skipped", "warn"]) == "warn" and sc.worst(["skipped"]) == "skipped"
    parked = {j: 0.0 for j in config.ARM_JOINTS}
    off = sc.parked_result({**parked, "elbow_flex": 4.0}, parked)
    assert off["worst_joint"] == "elbow_flex" and off["status"] == "warn"
    assert sc.parked_result(None, parked)["status"] == "skipped"
