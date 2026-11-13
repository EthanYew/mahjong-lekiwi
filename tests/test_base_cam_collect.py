"""The arm's hold-up collection (docs/face_down_plan.md): what is short, the poses, and the whole script on the simulated arm."""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import camera_poses, config, base_cam, base_cam_collect
from mahjong.session import read_events

POSE = {"shoulder_pan": -6.5, "shoulder_lift": 35.8, "elbow_flex": -6.6, "wrist_flex": 91.8, "wrist_roll": -88.7}


def write_session(folder, shots, sim=False, order="interleaved"):
    """A finished hold-up session with one saved photo per (tile, kind, hold); order None is one from before interleaving."""
    folder.mkdir(parents=True)
    lines = [{"type": "session_start", "sim": sim}, {"type": "holdup_conditions", "conditions": {}, "source": "arm",
                                                      **({"order": order} if order else {})}]
    for n, (tile, kind, hold) in enumerate(shots, 1):
        photo = folder / "photos" / tile / f"{kind}_h{hold}_0.jpg"
        photo.parent.mkdir(parents=True, exist_ok=True)
        photo.write_bytes(b"x")
        lines.append({"type": "holdup_shot", "shot": n, "tile": tile, "kind": kind, "hold": hold,
                      "files": [str(photo.relative_to(folder))]})
    (folder / "events.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def test_hold_counts_count_holds_across_sessions_and_leave_out_simulated_ones(tmp_path):
    write_session(tmp_path / "20261105_100000_holdup", [("5C", "upright", 1), ("5C", "upright", 2), ("5C", "turned", 1)])
    write_session(tmp_path / "20261105_110000_holdup", [("5C", "upright", 1)])      # a new session's hold 1 is another hold
    write_session(tmp_path / "20261105_120000_holdup", [("3C", "upright", 1)], sim=True)
    assert base_cam_collect.hold_counts(tmp_path) == {("5C", "upright"): 3, ("5C", "turned"): 1}


def test_holds_from_before_interleaving_dont_count_so_their_tiles_are_collected_again(tmp_path):
    write_session(tmp_path / "20261106_105954_holdup", [("EW", "upright", h) for h in range(1, 7)], order=None)
    write_session(tmp_path / "20261106_195636_holdup", [("5C", "upright", 1)])
    assert base_cam_collect.hold_counts(tmp_path) == {("5C", "upright"): 1}
    assert base_cam_collect.wanted(["EW"], base_cam_collect.hold_counts(tmp_path))[0]["missing"] == 6


def test_the_default_is_every_tile_name_and_a_typo_fails_before_any_hardware(monkeypatch, tmp_path):
    import base_cam_collect as script
    from mahjong import detector
    assert list(base_cam_collect.DEFAULT_TILES) == detector.TILE_NAMES and "1F" in base_cam_collect.DEFAULT_TILES
    with pytest.raises(SystemExit, match="not a tile name: 5X"):
        script.main(["--sim", "--tiles", "5C,5X"])


def test_a_fixed_distance_session_from_before_holds_is_left_out_not_a_crash(tmp_path):
    old = tmp_path / "20261105_090000_holdup"
    photo = old / "photos" / "5C" / "20cm_upright_0.jpg"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"x")
    (old / "events.jsonl").write_text(json.dumps({"type": "session_start"}) + "\n" + json.dumps(
        {"type": "holdup_shot", "shot": 1, "tile": "5C", "kind": "upright", "distance_cm": 20,
         "files": ["photos/5C/20cm_upright_0.jpg"]}) + "\n")
    write_session(tmp_path / "20261105_100000_holdup", [("5C", "turned", 1)])
    assert base_cam_collect.hold_counts(tmp_path) == {("5C", "turned"): 1}


def test_wanted_puts_the_tile_furthest_short_first_and_drops_the_ones_at_target():
    counts = {("5C", "upright"): 6, ("5C", "turned"): 6, ("3C", "upright"): 5, ("2C", "upright"): 1}
    rows = base_cam_collect.wanted(["5C", "3C", "2C", "9C"], counts, target=6)
    assert [(r["tile"], r["kind"], r["missing"]) for r in rows] == [
        ("9C", "upright", 6), ("9C", "turned", 6), ("2C", "upright", 5), ("2C", "turned", 6), ("3C", "upright", 1), ("3C", "turned", 6)]


def test_arrangements_lay_a_tile_per_spot_move_each_one_along_and_alternate_the_laying():
    rows = base_cam_collect.wanted(["9C", "2C", "3C", "5C"], {("5C", k): 6 for k in base_cam.KINDS}, target=6)
    plan = base_cam_collect.arrangements(rows, ["9C", "2C", "3C", "5C"], n_spots=2, max_rounds=2)
    assert [(a["tiles"], a["laying"]) for a in plan] == [
        (["9C", "2C"], "as_layout"), (["9C", "2C"], "rotated"), (["2C", "9C"], "as_layout"), (["2C", "9C"], "rotated"),
        (["3C", "5C"], "as_layout"), (["3C", "5C"], "rotated"), (["5C", "3C"], "as_layout"), (["5C", "3C"], "rotated")]
    assert {a["rounds"] for a in plan} == {2}        # 6 missing over 2 spots = 3 rounds, capped at 2
    held = {}
    for a in plan:                                    # every tile at every spot, both layings, the same number of times
        for spot, tile in enumerate(a["tiles"]):
            held[(tile, spot, a["laying"])] = held.get((tile, spot, a["laying"]), 0) + a["rounds"]
    assert len(held) == 4 * 2 * 2 and set(held.values()) == {2}
    assert base_cam_collect.arrangements(rows, ["9C", "2C", "3C", "5C"], n_spots=2, max_rounds=9)[0]["rounds"] == 3


def test_a_short_group_is_filled_so_no_tile_is_collected_alone_and_one_hold_short_needs_one_round():
    rows = base_cam_collect.wanted(["1B", "EW", "7C"], {("1B", "upright"): 6, ("1B", "turned"): 6, ("EW", "upright"): 6,
                                                   ("EW", "turned"): 6, ("7C", "upright"): 5, ("7C", "turned"): 5}, target=6)
    assert base_cam_collect.tile_groups(rows, ["1B", "EW", "7C"], 3) == [["7C", "1B", "EW"]]    # 1B and EW are at target: fillers
    plan = base_cam_collect.arrangements(rows, ["1B", "EW", "7C"], n_spots=3, max_rounds=4)
    assert len(plan) == 6 and {a["rounds"] for a in plan} == {1}       # 7C is 1 short; 3 spots x 1 round gives it 3
    assert base_cam_collect.arrangements(rows, ["7C"], n_spots=3, max_rounds=4)[0]["tiles"] == ["7C"]     # one tile, one spot


def test_pose_names_are_numbered_not_alphabetical():
    poses = {f"base_view_{n}": {} for n in (10, 2, 1)} | {"manual_1": {}}
    assert base_cam_collect.pose_names(poses=poses) == ["base_view_1", "base_view_2", "base_view_10"]


def test_jitter_moves_only_pan_flex_and_roll_within_the_limit_and_zero_leaves_the_pose():
    rng = random.Random(1)
    for _ in range(50):
        out = base_cam_collect.jitter(POSE, rng, 2.0)
        assert out["shoulder_lift"] == POSE["shoulder_lift"] and out["elbow_flex"] == POSE["elbow_flex"]
        assert all(abs(out[j] - POSE[j]) <= 2.0 for j in base_cam_collect.JITTER_JOINTS)
    assert base_cam_collect.jitter(POSE, rng, 0) == POSE


def test_one_answer_about_the_first_photo_fixes_what_both_layings_show():
    assert base_cam_collect.learn_view("as_layout", "turned") == {"as_layout": "turned", "rotated": "upright"}
    assert base_cam_collect.learn_view("rotated", "turned") == {"rotated": "turned", "as_layout": "upright"}


def test_the_lay_text_says_where_and_how_without_naming_a_way_up():
    spot = {"id": "S7", "xy": [0.192, -0.012]}
    text = base_cam_collect.lay_text("5C", "as_layout", spot)
    assert "FACE DOWN on S7" in text and "180" not in text
    assert "180" in base_cam_collect.lay_text("5C", "rotated", spot)


def script_in(monkeypatch, tmp_path, poses=2):
    import base_cam_collect as script
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(config, "CAMERA_POSES", tmp_path / "poses.json")
    for n in range(1, poses + 1):
        camera_poses.save(f"base_view_{n}", POSE, "test")
    return script


def run_script(script, argv, answers):
    answers = iter(answers)
    return script.main(["--sim", "--layout", "even-hand14", "--spots", "S7", *argv], ask=lambda prompt: next(answers, "q"))


def test_a_simulated_run_picks_shows_and_sets_back_down_and_leaves_photos_the_other_scripts_read(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder = run_script(script, ["--tiles", "5C", "--target-holds", "2", "--rounds", "2", "--frames", "2"],
                        ["", "", "", "", "d", ""])  # three lighting answers, lay as_layout, "upside down" about its first photo, lay rotated
    events = read_events(folder)
    kinds = [e["type"] for e in events]
    assert kinds.count("holdup_pick") == 4 and kinds.count("holdup_return") == 4
    assert kinds.count("holdup_view_map") == 1          # asked once, not per hold
    assert next(e for e in events if e["type"] == "holdup_view_map")["view"] == {"as_layout": "turned", "rotated": "upright"}
    rows = base_cam.rows(folder)
    assert len(rows) == 4 * 2 * 2           # 4 holds x 2 poses x 2 frames
    assert {(r["truth"], r["kind"], r["hold"]) for r in rows} == {
        ("5C", "turned", 1), ("5C", "turned", 2), ("5C", "upright", 1), ("5C", "upright", 2)}
    shot = next(e for e in events if e["type"] == "holdup_shot")
    assert shot["source"] == "arm" and shot["frame_poses"] == ["base_view_1"] * 2 + ["base_view_2"] * 2
    assert base_cam.is_sim(folder)


def test_a_simulated_run_with_two_tiles_alternates_them_and_moves_each_to_the_other_spot(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path, poses=1)
    answers = iter(["", "", "", "", "", "d", "", "", "", "", "", ""])   # lighting x3, lay 2, "upside down", then lay 2 per arrangement
    folder = script.main(["--sim", "--layout", "even-hand14", "--spots", "S5,S9", "--tiles", "1B,EW", "--target-holds", "2",
                          "--rounds", "1", "--frames", "1"], ask=lambda prompt: next(answers, "q"))
    events = read_events(folder)
    shots = [(e["tile"], e["spot"]) for e in events if e["type"] == "holdup_shot"]
    assert shots == [("1B", "S5"), ("EW", "S9"), ("1B", "S5"), ("EW", "S9"),
                     ("EW", "S5"), ("1B", "S9"), ("EW", "S5"), ("1B", "S9")]     # never the same tile twice in a row
    layouts = [e["tiles"] for e in events if e["type"] == "holdup_arrangement"]
    assert layouts == [{"S5": "1B", "S9": "EW"}] * 2 + [{"S5": "EW", "S9": "1B"}] * 2
    assert [e["laying"] for e in events if e["type"] == "holdup_arrangement"] == ["as_layout", "rotated"] * 2
    counts = {(r["truth"], r["kind"]) for r in base_cam.rows(folder)}
    assert counts == {(t, k) for t in ("1B", "EW") for k in base_cam.KINDS}


def test_several_tiles_need_several_spots(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        run_script(script, ["--tiles", "1B,EW"], [])            # run_script lays on S7 only


def test_nothing_is_collected_when_every_tile_already_has_enough(monkeypatch, tmp_path, capsys):
    script = script_in(monkeypatch, tmp_path, poses=1)
    write_session(tmp_path / "20261105_100000_holdup", [("5C", k, h) for k in base_cam.KINDS for h in (1, 2)])
    run_script(script, ["--tiles", "5C", "--target-holds", "2"], [])
    assert "nothing to collect" in capsys.readouterr().out


def test_the_empty_pass_visits_each_pose_with_nothing_in_the_jaws_and_saves_a_sheet(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder = run_script(script, ["--empty"], ["", ""])
    events = read_events(folder)
    assert [e["pose"] for e in events if e["type"] == "holdup_empty_pose"] == ["base_view_1", "base_view_2"]
    assert (folder / "dry_run" / "base_view_1_base.jpg").exists() and (folder / "dry_run_sheet.jpg").exists()
    assert not any(e["type"] == "holdup_pick" for e in events)
    assert folder.name.endswith("_holdup_dry") and not list(tmp_path.glob("*_holdup"))      # the other scripts never see it


def test_a_pose_name_that_isnt_saved_fails_before_any_hardware(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        run_script(script, ["--poses", "nope"], [])


def test_a_miss_reopens_the_window_for_the_same_laying_and_the_answer_decides_what_happens(monkeypatch, tmp_path):
    from types import SimpleNamespace
    script = script_in(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(script, "guide", lambda rs, placement, spot, sign, text=None: seen.append((placement["laying"], sign, text)) or "s")
    monkeypatch.setattr(script, "close_windows", lambda: seen.append("closed"))
    rig = SimpleNamespace(realsense=object(), args=SimpleNamespace(layout_top="far"))
    placement, spot = {"tile": "5C", "laying": "rotated", "holds": 2}, {"id": "S7", "xy": [0.19, 0.0]}
    answer = script.put_back_by_hand(rig, placement, spot, lambda p: pytest.fail("no terminal prompt with a window"),
                                     "5C: missed", "unused")
    assert answer == "s" and seen[0][:2] == ("rotated", -1)       # the rotated laying's TOP end, as when it was laid
    assert "back FACE DOWN on S7" in seen[0][2] and seen[1] == "closed"
    rig.realsense = None                                           # without the RealSense: the terminal prompt
    assert script.put_back_by_hand(rig, placement, spot, lambda p: " Q ", "5C: missed", "prompt") == "q"


def test_a_missed_pickup_stops_for_the_tile_to_be_put_back_and_skip_moves_on(monkeypatch, tmp_path):
    from mahjong import sim
    script = script_in(monkeypatch, tmp_path)
    original = sim.SimPolicy.run

    def misses(self, arm, camera, trace_dir, abort=None, mark=None, log=print):
        result = original(self, arm, camera, trace_dir, abort, mark, log)
        arm.robot.holding = False       # the jaws closed on nothing
        arm.move_to({"gripper": 40.0}, 0.1)
        return result
    monkeypatch.setattr(sim.SimPolicy, "run", misses)
    folder = run_script(script, ["--tiles", "5C", "--target-holds", "1", "--frames", "1"],
                        ["", "", "", "", "s", "q"])     # lighting x3, lay, "skip this tile" after the miss, quit at the next
    events = read_events(folder)
    picks = [e for e in events if e["type"] == "holdup_pick"]
    assert len(picks) == 1 and picks[0]["success"] is False
    assert not [e for e in events if e["type"] == "holdup_shot"]


def test_repeated_misses_on_a_spot_give_the_tile_up_instead_of_stalling_the_run(monkeypatch, tmp_path):
    from mahjong import sim
    script = script_in(monkeypatch, tmp_path)
    original = sim.SimPolicy.run

    def misses(self, arm, camera, trace_dir, abort=None, mark=None, log=print):
        result = original(self, arm, camera, trace_dir, abort, mark, log)
        arm.robot.holding = False
        arm.move_to({"gripper": 40.0}, 0.1)
        return result
    monkeypatch.setattr(sim.SimPolicy, "run", misses)
    prompts = []
    answers = iter(["", "", "", "", "", "q"])       # lighting x3, lay, "try again" after miss 1; then miss 2 asks nothing

    def ask(prompt):
        prompts.append(prompt)
        return next(answers, "q")
    folder = script.main(["--sim", "--layout", "even-hand14", "--spots", "S7", "--tiles", "5C", "--target-holds", "1",
                          "--max-misses", "2"], ask=ask)
    events = read_events(folder)
    assert [e["success"] for e in events if e["type"] == "holdup_pick"][:2] == [False, False]
    gave_up = [e for e in events if e["type"] == "holdup_gave_up"]
    assert gave_up and gave_up[0]["spot"] == "S7" and gave_up[0]["misses"] == 2
    assert sum("missed" in p for p in prompts) == 1     # asked after the first miss only


def test_the_top_of_the_picture_goes_to_the_far_end_for_as_layout_and_the_near_end_for_rotated():
    assert base_cam_collect.top_end_sign("as_layout") == 1 and base_cam_collect.top_end_sign("rotated") == -1
    assert base_cam_collect.top_end_sign("as_layout", "near") == -1 and base_cam_collect.top_end_sign("rotated", "near") == 1


def test_the_guide_frame_outlines_the_spot_and_marks_the_end_the_top_goes():
    from mahjong.sim import SimPerception
    mat = SimPerception()
    spot = {"id": "S7", "xy": [0.192, -0.012], "yaw_deg": 0.0}
    image = mat.snapshot()
    far = base_cam_collect.guide_frame(image, spot, mat.intrinsics, mat.T_cam_to_base, "Lay 5C", 1)
    near = base_cam_collect.guide_frame(image, spot, mat.intrinsics, mat.T_cam_to_base, "Lay 5C", -1)
    assert far.shape == image.shape and not (far == image).all()
    assert not (far == near).all()          # the thick TOP edge moves to the other end


def test_the_guide_window_shows_live_frames_until_a_key_and_maps_keys_to_answers(monkeypatch, tmp_path):
    from mahjong.sim import SimPerception
    script = script_in(monkeypatch, tmp_path)
    mat = SimPerception()
    spot = {"id": "S7", "xy": [0.192, -0.012]}
    placement = {"tile": "5C", "laying": "rotated", "holds": 1}
    shown = []
    keys = iter([255, 255, ord("s")])
    assert script.guide(mat, placement, spot, -1, show=shown.append, wait_key=lambda: next(keys)) == "s"
    assert len(shown) == 3                  # redrawn every pass, so the view is live
    for key, answer in ((13, ""), (32, ""), (ord("q"), "q"), (27, "q")):
        assert script.guide(mat, placement, spot, 1, show=lambda image: None, wait_key=lambda key=key: key) == answer


def test_a_camera_that_keeps_tearing_is_rerun_once_then_counts_as_a_miss(monkeypatch, tmp_path):
    """Torn frames stop the pick as "camera", and a torn camera reopens with no prompt: the rerun has to end somewhere."""
    from mahjong import hardware, sim
    script = script_in(monkeypatch, tmp_path)
    monkeypatch.setattr(hardware, "reopen_wrist_camera", lambda cam: cam)
    original, calls = sim.SimPolicy.run, []

    def torn(self, arm, camera, trace_dir, abort=None, mark=None, log=print):
        result = original(self, arm, camera, trace_dir, abort, mark, log)
        calls.append(1)
        if len(calls) > 20:
            raise RuntimeError("the pick was rerun with no end")
        arm.robot.holding = False
        arm.move_to({"gripper": 40.0}, 0.1)
        return {**result, "stop_reason": "camera", "camera_problem": "torn"}
    monkeypatch.setattr(sim.SimPolicy, "run", torn)
    folder = script.main(["--sim", "--layout", "even-hand14", "--spots", "S7", "--tiles", "5C", "--target-holds", "1",
                          "--max-misses", "2"], ask=lambda prompt: "")
    gave_up = [e["misses"] for e in read_events(folder) if e["type"] == "holdup_gave_up"]
    assert gave_up and set(gave_up) == {2}
    assert len(calls) == 4 * len(gave_up)          # per laying: a pick and its rerun, twice, for the two misses
