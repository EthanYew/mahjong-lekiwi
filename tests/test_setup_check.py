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
    assert abs(lower["change"]["offset_mm"] + 5) < 0.5
    # shown, never counted: the real depth reading wanders more than this with nothing moved
    assert lower["status"] == "info" and "not graded" in sc.table_text(lower)
    results = {"camera": {"status": "pass"}, "table": {**lower, "text": sc.table_text(lower)}}
    lines, overall = sc.summary(results)
    assert overall == "pass" and not any(line.startswith("\ntable:") for line in lines)
    assert sc.table_result(None, level)["status"] == "warn"          # no depth at all is still worth a look


def test_a_refit_hover_correction_doesnt_count_against_the_setup(tmp_path, monkeypatch):
    import setup_check
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    assert run(tmp_path, ["--save-reference"], ["y"]) == "pass"

    class Refit:                          # the live correction moved 22mm since the reference
        name = "gamepad/affine/99"

        def delta_at(self, x, y):
            return 0.02, 0.01
    monkeypatch.setattr(setup_check.HoverCorrection, "load", classmethod(lambda cls, *a, **k: Refit()))
    assert run(tmp_path, []) == "pass"
    spots = last_check(tmp_path)[0]["results"]["tiles"]["spots"]
    assert all(r["shift_mm"] < 0.5 and r["live_correction_change_mm"] > 15 for r in spots.values())


def test_a_tile_placed_turned_within_green_still_passes(tmp_path, monkeypatch):
    import setup_check
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    assert run(tmp_path, ["--save-reference"], ["y"]) == "pass"
    place = setup_check.place_sim_tile

    def turned_5(realsense, spot):
        place(realsense, spot)
        realsense.tiles[0]["yaw_deg"] = 5.0
    monkeypatch.setattr(setup_check, "place_sim_tile", turned_5)
    assert run(tmp_path, []) == "pass"
    for r in last_check(tmp_path)[0]["results"]["tiles"]["spots"].values():
        assert abs(abs(r["angle_change_deg"]) - 5) < 1.5 and abs(r["placed_turn_deg"] - 5) < 2.5    # shown, not graded


def test_saving_a_reference_asks_whether_the_cross_is_on_the_tile(tmp_path, monkeypatch):
    import setup_check
    from mahjong import layout, sim
    from mahjong.session import Session
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    devices = sim.connect(tiles=[], wrist_view=True)
    session = Session("setup", ask_setup=False, sim=True)
    answers = iter(["retry", "ok"])                  # F2's first find rejected, then accepted
    spots = [s for s in layout.load("eval_familiar")["spots"] if s["id"] == "F2"]
    results, frames, _ = setup_check.tile_test(devices["arm"], devices["wrist_camera"], devices["realsense"], None,
                                               spots, None, session, True, ask=lambda p: "",
                                               confirm=lambda spot, frame, found: next(answers))
    session.close()
    assert results["F2"]["attempts"] == 2 and "F2" in frames and results["F2"]["hover_delta_mm"] == [0.0, 0.0]


def test_an_old_reference_is_asked_to_be_saved_again(tmp_path, monkeypatch):
    import json
    import pytest
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    assert run(tmp_path, ["--save-reference"], ["y"]) == "pass"
    path = tmp_path / "reference" / "reference.json"
    reference = json.loads(path.read_text())
    for spot in reference["spots"]:
        del spot["hover_delta_mm"]
    path.write_text(json.dumps(reference))
    with pytest.raises(SystemExit, match="--save-reference again"):
        run(tmp_path, [])
    assert run(tmp_path, ["--skip-tiles"]) == "pass"


def test_tiles_are_graded_on_the_median_spot():
    spot = {"shift_mm": 2.0, "size_pct": 1.0, "status": "pass"}
    one_off = sc.tiles_result({"F2": spot, "F3": {**spot, "shift_mm": 9.0, "status": "fail"}, "F6": spot})
    assert one_off["status"] == "warn" and one_off["median_shift_mm"] == 2.0          # one spot: worth a look
    all_off = sc.tiles_result({i: {**spot, "shift_mm": 8.0, "status": "fail"} for i in ("F2", "F3", "F6")})
    assert all_off["status"] == "fail"                                                 # every spot: the setup


def test_wrist_tile_finds_the_tile_among_neighbours_glare_and_the_jaw():
    image = np.full((480, 640, 3), 25, np.uint8)
    cv2.fillPoly(image, [np.int32([(270, 180), (380, 180), (380, 330), (270, 330)])], (225, 235, 240))
    cv2.ellipse(image, (80, 400), (60, 20), 30, 0, 300, (200, 200, 200), -1)   # an irregular bright patch
    tile = sc.wrist_tile(image)
    assert np.allclose(tile["centre_px"], (325, 255), atol=2)
    glare = image.copy()
    cv2.rectangle(glare, (0, 60), (120, 200), (235, 235, 235), -1)            # tile-sized glare at the image edge
    cv2.fillPoly(glare, [np.int32([(450, 200), (560, 200), (560, 350), (450, 350)])], (225, 235, 240))   # a neighbour
    cv2.line(glare, (300, 330), (300, 470), (200, 205, 205), 12)              # the pale jaw tip touching the face
    cv2.rectangle(glare, (150, 440), (500, 470), (230, 230, 230), -1)         # ... and the bright table edge
    assert np.allclose(sc.wrist_tile(glare)["centre_px"], (325, 255), atol=3)
    assert np.allclose(sc.wrist_tile(glare, near_px=(500, 280))["centre_px"], (505, 275), atol=3)
    assert sc.wrist_tile(glare, near_px=(60, 130)) is None                    # the glare never counts
    dim = (image.astype(np.float32) * 0.6).astype(np.uint8)                  # a dimmer room: still found
    assert np.allclose(sc.wrist_tile(dim)["centre_px"], (325, 255), atol=2)
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


def guide_for(warp):
    """The guide's steps and view for the sim RealSense frame warped by `warp` (2 x 3) as the live one."""
    from mahjong.sim import SimPerception
    mat = SimPerception(tiles=[])
    K, T = mat.intrinsics, mat.T_cam_to_base
    reference = mat.snapshot()
    region = sc.robot_region(K, T, reference.shape)
    live = cv2.warpAffine(reference, np.float32(warp), reference.shape[1::-1])
    steps = sc.guide_steps(sc.robot_motion(reference, live, region, K, T))
    result = sc.camera_result(sc.image_shift(reference, live, region), sc.mm_per_px(K, T))
    view = sc.guide_view(reference, live, region, result, steps, K, T)
    return {s["kind"]: s for s in steps}, view, (K, T, region)


def test_the_guide_says_turn_or_slide_in_the_robots_terms():
    steps, view, _ = guide_for([[1, 0, 0], [0, 1, 0]])
    assert sc.step_now(list(steps.values())) is None and view.shape[1] == 900                 # lined up: no arrow
    _, _, (K, T, region) = guide_for([[1, 0, 0], [0, 1, 0]])
    centre = ((region[0] + region[2]) / 2, (region[1] + region[3]) / 2)

    # the robot's middle 8mm further forward (toward the tiles): slide it back, nothing about turning
    ahead = sc.table_to_table_xy([centre], K, T)[0] + [0.008, 0.0]
    du, dv = sc._project([ahead], K, T)[0] - np.array(centre)
    steps, _, _ = guide_for([[1, 0, du], [0, 1, dv]])
    assert steps["turn"]["status"] == "pass" and steps["move"]["status"] != "pass"
    assert "back" in steps["move"]["text"] and "left" not in steps["move"]["text"] and "right" not in steps["move"]["text"]
    assert sc.step_now(list(steps.values()))["kind"] == "move"

    # turned 2° about its middle: turn it first, whichever way undoes it
    turned = cv2.getRotationMatrix2D(centre, 2.0, 1.0)
    steps, _, _ = guide_for(turned)
    assert steps["turn"]["status"] == "fail" and sc.step_now(list(steps.values()))["kind"] == "turn"
    back = cv2.getRotationMatrix2D(centre, -2.0, 1.0)
    other, _, _ = guide_for(back)
    assert {steps["turn"]["text"].split()[4], other["turn"]["text"].split()[4]} == {"clockwise", "anticlockwise"}


def test_the_guide_holds_still_on_noise():
    readings = [{"turn_deg": t, "dx_mm": x, "dy_mm": 0.0} for t, x in ((0.3, 1.0), (-0.2, -1.2), (0.1, 0.4), (5.0, 9.0),
                                                                       (0.0, 0.2))]
    steps = sc.guide_steps(sc.steady(readings, ("turn_deg", "dx_mm", "dy_mm")))
    assert all(s["status"] == "pass" for s in steps)              # one wild frame doesn't swing the arrow
    assert sc.steady([{"inliers": 3}] * 3 + readings[:2], ("turn_deg",)) is None          # mostly unseen: can't say


def test_a_wrist_camera_that_doesnt_move_with_the_arm_stops_the_tile_test(tmp_path, monkeypatch):
    from mahjong.sim import SimWristCamera
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    assert run(tmp_path, ["--save-reference"], ["y"]) == "pass"
    room = cv2.resize(np.random.default_rng(0).integers(0, 255, (12, 16, 3), dtype=np.uint8), (640, 480),
                      interpolation=cv2.INTER_NEAREST)                # blocks: a room's shapes, not pixel noise

    class BaseCamera(SimWristCamera):     # the base camera on the wrist's port: the same room wherever the arm is
        def read(self):
            return True, room.copy()

    devices = sim_devices()
    devices["wrist_camera"] = BaseCamera(devices["arm"], devices["realsense"])
    assert run(tmp_path, [], devices=devices) == "fail"            # no "reposition the tile" questions
    check, session = last_check(tmp_path)
    spots = check["results"]["tiles"]["spots"]
    assert list(spots) == ["F2"] and "isn't on the arm" in spots["F2"]["why"]
    assert [e["type"] for e in read_events(session)].count("setup_wrist_camera_still") == 1


def test_a_weak_match_is_shown_to_a_person_rather_than_looped(tmp_path, monkeypatch):
    import setup_check
    from mahjong import sim
    from mahjong.session import Session
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    assert run(tmp_path, ["--save-reference"], ["y"]) == "pass"
    reference = sc.load_reference(tmp_path / "reference")
    monkeypatch.setattr(sc, "MATCH_MIN_SCORE", 1.01)                # every match weak: a reference in other light
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "check")
    spots = setup_check.check_spots(None, reference)[:1]

    def tile_test(confirm_weak, answers=()):
        devices = sim.connect(tiles=[], wrist_view=True)
        session = Session("setup", ask_setup=False, sim=True)
        answers = iter(answers)
        results, _, _ = setup_check.tile_test(devices["arm"], devices["wrist_camera"], devices["realsense"], None, spots,
                                              reference, session, True, ask=lambda p: next(answers),
                                              confirm_weak=confirm_weak)
        session.close()
        return results[spots[0]["id"]], [e["type"] for e in read_events(session.dir)]

    said = []
    result, events = tile_test(lambda spot, frame, weak: said.append(weak["score"]) or True)
    assert result["status"] == "pass" and result["confirmed_by"] == "person" and result["attempts"] == 1
    assert len(said) == 1 and sc.MATCH_ASK_SCORE <= said[0] and "setup_tile_confirmed" in events
    result, events = tile_test(lambda spot, frame, weak: False, answers=["s"])              # "no": the usual question
    assert result["status"] == "fail" and "setup_tile_not_found" in events
