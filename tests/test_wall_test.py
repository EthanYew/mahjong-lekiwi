"""The wall test: what's tried in what order, how it's scored, and the whole script on the simulated arm."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import camera_poses, config, wall_test
from mahjong.session import read_events

POSE = {j: 0.0 for j in config.ARM_JOINTS}


def test_each_round_tries_every_station_and_case_and_the_order_flips():
    assert wall_test.schedule(["F", "L"], ["flat"], 3) == [(1, "F", "flat"), (1, "L", "flat"), (2, "L", "flat"), (2, "F", "flat"),
                                                           (3, "F", "flat"), (3, "L", "flat")]


def test_a_cycle_draws_top_then_bottom_and_is_fed_every_second_draw():
    plan = wall_test.cycle_plan(5)
    assert [d["layer"] for d in plan] == ["top", "bottom", "top", "bottom", "top"]
    assert [d["feed_before"] for d in plan] == [False, False, True, False, True]
    assert [d["draws_since_feed"] for d in plan] == [0, 1, 0, 1, 0]


def test_a_top_tile_hovers_one_tile_higher_and_its_close_is_measured_from_its_own_level():
    assert wall_test.HOVER_RAISE_M["top"] == config.TILE_THICKNESS_M and wall_test.HOVER_RAISE_M["bottom"] == 0
    assert wall_test.above_own_level_m(0.033, "top") == pytest.approx(0.013)
    assert wall_test.above_own_level_m(0.013, "bottom") == pytest.approx(0.013)
    assert wall_test.above_own_level_m(None, "top") is None


@pytest.mark.parametrize("clean,tries,case,expected", [
    (8, 10, "flat", "carry on"), (7, 10, "top", "try a change, then decide"), (4, 10, "top", "try a change, then decide"), (3, 10, "bottom", "record demos"),
    (4, 5, "control", "control fine"), (3, 5, "control", "control under 80%: fix the setup before reading anything else")])
def test_the_plans_thresholds(clean, tries, case, expected):
    assert wall_test.verdict(clean, tries, case) == expected


def test_too_few_tries_get_no_verdict():
    assert "for 10" in wall_test.verdict(5, 5, "flat")
    assert wall_test.verdict(2, 3, "control") == "too few tries to judge"


def test_a_push_check_passes_on_how_far_pushes_stray_not_on_a_steady_offset():
    steady_but_off = [(0.255, 0.0), (0.2552, 0.0), (0.2548, 0.0)]
    result = wall_test.push_result(steady_but_off, [(0.255, 0.0)] * 3, (0.245, 0.0))
    assert result["passed"] and result["worst_mm"] == pytest.approx(0.2, abs=0.01) and result["offset_from_station_mm"] == pytest.approx(10.0, abs=0.1)
    assert not wall_test.push_result([(0.245, 0.0), (0.253, 0.0)], [], (0.245, 0.0))["passed"]      # 4mm either side of their mean


def test_the_summary_separates_a_cycles_top_and_bottom_draws():
    rows = [{"station": "F", "case": "cycle", "layer": layer, "outcome": outcome, "held": True, "above_own_level_m": 0.01}
            for layer, outcome in (("top", wall_test.CLEAN), ("top", "lifted two tiles"), ("bottom", wall_test.CLEAN))]
    groups = {g["layer"]: g for g in wall_test.summarize(rows)}
    assert groups["top"]["clean"] == 1 and groups["top"]["tries"] == 2 and groups["top"]["not_clean"] == {"lifted two tiles": 1}
    assert groups["bottom"]["clean"] == 1 and groups["bottom"]["close_median_mm"] == 10.0
    assert any("lifted two" in line for line in wall_test.table(wall_test.summarize(rows)))


def test_the_stations_are_in_the_layouts_file_and_every_outcome_key_is_one_character():
    layouts = json.loads(config.LAYOUTS.read_text())
    spots = {s["id"]: s["xy"] for s in layouts[wall_test.LAYOUT]["spots"]}
    assert set(spots) == {"F", "L", "F2", "L2"}
    # F2 and L2 give the hand's tiles more room: F2 further out than F, L2 further left than L, each on F's / L's line
    assert spots["F2"][0] > spots["F"][0] and spots["F2"][1] == spots["F"][1]
    assert spots["L2"][1] > spots["L"][1] and spots["L2"][0] == spots["L"][0]
    assert all(len(k) == 1 for k in wall_test.OUTCOMES) and "q" not in wall_test.OUTCOMES   # q quits


ALL_STEPS = [*(wall_test.lay_step(case, "F", f"{case}, try 10/10") for case in wall_test.CASES), wall_test.hover_step("F"), wall_test.push_step("F"),
             wall_test.wall_step("F", 7), wall_test.feed_step("F"), wall_test.put_right_step("F"), wall_test.clear_and_feed_step("F")]


def test_every_line_of_every_instruction_fits_the_camera_window():
    """A long line used to run off the edge of the 848px window, which cut the instruction short."""
    import cv2
    for step in ALL_STEPS:
        for line in step["lines"]:
            width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
            assert width <= 848 - 20, line


def test_each_instruction_says_how_many_tiles_and_which_side_up_and_matches_its_picture():
    tiles = {"control": 1, "flat": 1, "strip": 2, "top": 4, "bottom": 3}
    for case, count in tiles.items():
        step = wall_test.lay_step(case, "F", "tag")
        assert sum(cell is not None for row in step["picture"] for cell in row) == count
        assert f"{count} tile" in " ".join(step["lines"])
        assert ("WHITE" if case == "control" else "GREEN") in " ".join(step["lines"])
        assert sum("[" in (cell or "") for row in step["picture"] for cell in row) == 1     # one tile the arm goes for
    assert wall_test.CASES["bottom"]["picture"][0][0] is None                               # nothing on top of the bottom tile
    assert "NOTHING on top" in " ".join(wall_test.lay_step("bottom", "F", "tag")["lines"])
    assert wall_test.push_step("F")["picture"][0][0] == "W" and wall_test.push_step("F")["picture"][1][0] == "G"


def test_the_terminal_shows_the_instruction_the_picture_and_what_the_letters_mean():
    text = wall_test.step_text(wall_test.lay_step("top", "F", "top, try 1/10"))
    assert "[top, try 1/10] Clear station F" in text and "[G]" in text and "W = white side up" in text


def test_the_window_has_no_top_mark_when_which_way_up_doesnt_matter_and_shows_every_line():
    import numpy as np
    from mahjong import holdup_arm, sim
    mat = sim.SimPerception(tiles=[])
    image, spot = np.zeros((480, 848, 3), np.uint8), {"id": "F", "xy": [0.235, -0.012]}
    plain = holdup_arm.guide_frame(image, spot, mat.intrinsics, mat.T_cam_to_base, "one", None)
    marked = holdup_arm.guide_frame(image, spot, mat.intrinsics, mat.T_cam_to_base, "one", 1)
    assert not (plain == marked).all()
    two = holdup_arm.guide_frame(image, spot, mat.intrinsics, mat.T_cam_to_base, "one\ntwo", None)
    assert two[36:60].any() and not plain[36:60].any()      # the second line has its own row


def test_the_windows_picture_draws_every_step_inside_the_frame():
    import numpy as np
    import wall_test as script
    for step in ALL_STEPS:
        frame = np.zeros((480, 848, 3), np.uint8)
        script.draw_picture(frame, step["picture"])
        assert frame.any()


def prompts_of(script, argv, answers):
    prompts, answers = [], iter(answers)

    def ask(text):
        prompts.append(text)
        return next(answers, "q")
    script.main(["--sim", *argv], ask=ask)
    return prompts


def test_a_simulated_session_prompts_with_what_to_lay_and_no_top_mark(monkeypatch, tmp_path):
    prompts = prompts_of(script_in(monkeypatch, tmp_path), ["--cases", "bottom", "--trials", "1"], [*LIGHT, "n", "", "c"])
    assert any("3 tiles, GREEN side up" in p and "NOTHING on top" in p for p in prompts)
    assert not any("TOP" in p for p in prompts)


def test_a_shifted_hover_starts_nearer_the_robot_and_is_logged(monkeypatch, tmp_path):
    from mahjong import layout
    from mahjong.hover_correction import HoverCorrection
    script = script_in(monkeypatch, tmp_path)
    folder = run(script, ["--cases", "top", "--trials", "1", "--hover-back-mm", "10"], [*LIGHT, "n", "", "w"])
    row = attempts_of(folder)[0]
    station = next(s for s in layout.load(wall_test.LAYOUT)["spots"] if s["id"] == "F")
    x, y = HoverCorrection.load().apply(*station["xy"])
    assert row["hover_xy"] == pytest.approx([x - 0.010, y]) and row["hover_back_m"] == 0.010
    assert row["outcome"] == "took the wrong tile"


def test_a_shifted_hover_is_never_pooled_with_the_unshifted_tries():
    rows = [{"station": "F", "case": "top", "layer": "top", "outcome": "took the wrong tile", "held": True, "above_own_level_m": 0.01},       # logged before the option existed
            {"station": "F", "case": "top", "layer": "top", "outcome": wall_test.CLEAN, "held": True, "above_own_level_m": 0.01, "hover_back_m": 0.010}]
    groups = {g["hover_back_mm"]: g for g in wall_test.summarize(rows)}
    assert groups[0]["clean"] == 0 and groups[10]["clean"] == 1 and groups[0]["tries"] == groups[10]["tries"] == 1


@pytest.mark.parametrize("argv", [["--hover-back-mm", "25"], ["--hover-back-mm", "-25"], ["--hover-back-mm", "5", "--push-check"], ["--hover-back-mm", "5", "--hover-check"]])
def test_a_hover_shift_out_of_range_or_on_a_check_fails_before_any_hardware(monkeypatch, tmp_path, argv):
    script = script_in(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        script.main(["--sim", *argv], ask=lambda prompt: pytest.fail("asked a question"))


def test_a_missed_bottom_tile_is_one_step_clear_and_push_with_a_tile_to_aim_at(monkeypatch, tmp_path):
    prompts = prompts_of(script_in(monkeypatch, tmp_path), ["--cases", "cycle", "--trials", "3"], [*LIGHT, "n", "", "c", "p", "", "c"])
    both = [p for p in prompts if "Lift off anything" in p]
    assert len(both) == 1 and "Push the whole wall" in both[0] and "[G]" in both[0]
    assert not any("[cycle] Feed the wall" in p for p in prompts)       # it isn't asked for a second time


def test_a_clean_bottom_tile_still_gets_the_plain_feed(monkeypatch, tmp_path):
    prompts = prompts_of(script_in(monkeypatch, tmp_path), ["--cases", "cycle", "--trials", "3"], [*LIGHT, "n", "", "c", "c", "", "c"])
    assert sum("[cycle] Feed the wall" in p for p in prompts) == 1 and not any("Lift off anything" in p for p in prompts)


def test_only_top_tiles_get_the_top_shift(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder = run(script, ["--cases", "cycle", "--trials", "2", "--top-back-mm", "10"], [*LIGHT, "n", "", "c", "c"])
    top, bottom = attempts_of(folder)
    assert (top["layer"], top["hover_back_m"]) == ("top", 0.010) and (bottom["layer"], bottom["hover_back_m"]) == ("bottom", 0.0)
    assert bottom["hover_xy"][0] - top["hover_xy"][0] == pytest.approx(0.010, abs=1e-6)
    assert {g["hover_back_mm"]: g["layer"] for g in wall_test.summarize(wall_test.attempts(folder))} == {10: "top", 0: "bottom"}


@pytest.mark.parametrize("argv", [["--top-back-mm", "25", "--cases", "top"], ["--top-back-mm", "10", "--hover-back-mm", "10", "--cases", "top"],
                                  ["--top-back-mm", "10", "--cases", "flat"], ["--top-back-mm", "10", "--cases", "bottom"], ["--top-back-mm", "10", "--push-check"]])
def test_a_top_shift_that_wouldnt_do_what_it_says_fails_before_any_hardware(monkeypatch, tmp_path, argv):
    script = script_in(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        script.main(["--sim", *argv], ask=lambda prompt: pytest.fail("asked a question"))


def test_a_session_left_out_says_why_and_stops_counting(monkeypatch, tmp_path, capsys):
    script = script_in(monkeypatch, tmp_path)
    real = tmp_path / "20261110_100000_wall"
    real.mkdir()
    events = [{"type": "session_start", "sim": False},
              {"type": "wall_attempt", "station": "F", "case": "cycle", "layer": "top", "held": False, "above_own_level_m": 0.012, "outcome": "closed on air"}]
    (real / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    script.main(["--exclude-session", real.name, "--why", "the wall was turned 90 degrees"])
    assert wall_test.attempts(real) == [] and wall_test.excluded(real) == "the wall was turned 90 degrees"
    capsys.readouterr()
    script.main(["--report"])
    out = capsys.readouterr().out
    assert "left out (the wall was turned 90 degrees)" in out and "closed on air" not in out


@pytest.mark.parametrize("argv", [["--exclude-session", "20261110_100000_wall"], ["--exclude-session", "20261110_999999_wall", "--why", "x"], ["--exclude-session", "20261110_100000_bench", "--why", "x"]])
def test_leaving_a_session_out_needs_a_real_wall_session_and_a_reason(monkeypatch, tmp_path, argv):
    script = script_in(monkeypatch, tmp_path)
    for name in ("20261110_100000_wall", "20261110_100000_bench"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "events.jsonl").write_text("")
    with pytest.raises(SystemExit):
        script.main(argv)


def test_a_cycle_doesnt_ask_for_the_wall_put_right_after_its_last_draw(monkeypatch, tmp_path):
    prompts = prompts_of(script_in(monkeypatch, tmp_path), ["--cases", "cycle", "--trials", "1"], [*LIGHT, "n", "", "p"])
    assert any("stacks of 2" in p for p in prompts) and not any("Not clean" in p for p in prompts)


def test_a_cycle_says_what_the_wall_should_look_like_after_a_draw_that_wasnt_clean(monkeypatch, tmp_path):
    prompts = prompts_of(script_in(monkeypatch, tmp_path), ["--cases", "cycle", "--trials", "2"], [*LIGHT, "n", "", "p", "", "c"])
    assert any("Take the top tile off by hand" in p for p in prompts)


def test_every_station_is_comfortably_reached_once_the_hover_correction_has_moved_it():
    """The arm goes ~40mm from the RealSense's xy (hover correction): a spot picked from its xy alone looked fine at 0.245 and was at the edge."""
    from mahjong import layout
    from mahjong.hover_correction import HoverCorrection
    correction = HoverCorrection.load()
    for spot in layout.load(wall_test.LAYOUT)["spots"]:
        assert layout.reach_band(*spot["xy"], correction) == "comfortable", spot["id"]


# --- the script on the simulated arm ---

def script_in(monkeypatch, tmp_path):
    import wall_test as script
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(config, "CAMERA_POSES", tmp_path / "poses.json")
    camera_poses.save("base_view_1", POSE, "test")
    return script


def run(script, argv, answers):
    answers = iter(answers)
    return script.main(["--sim", *argv], ask=lambda prompt: next(answers, "q"))


LIGHT = ["", "", ""]


def attempts_of(folder):
    return [e for e in read_events(folder) if e["type"] == "wall_attempt"]


def test_a_simulated_session_alternates_stations_and_keeps_what_each_attempt_needs(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder = run(script, ["--stations", "F,L", "--cases", "flat", "--trials", "2"],
                 [*LIGHT, "n", "y", "", "c", "", "a", "", "c", "", "2"])     # lighting, whites per station, then lay + outcome per attempt
    rows = attempts_of(folder)
    assert [(r["station"], r["outcome"]) for r in rows] == [("F", wall_test.CLEAN), ("L", "closed on air"), ("L", wall_test.CLEAN), ("F", "lifted two tiles")]
    first = rows[0]
    assert first["face"] == "down" and first["layer"] == "flat" and first["white_tiles_near"] is False and rows[1]["white_tiles_near"] is True
    assert first["held"] is True and first["above_own_level_m"] == pytest.approx(first["grasp_clearance_m"])
    attempt = folder / first["attempt_dir"]
    assert (attempt / "base.jpg").exists() and (attempt / "table_before_after.jpg").exists()
    summary = json.loads((folder / "wall_summary.json").read_text())
    assert {(g["station"], g["clean"], g["tries"]) for g in summary} == {("F", 1, 2), ("L", 1, 2)}


def test_a_top_tile_is_hovered_one_tile_higher_on_the_arm(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder = run(script, ["--cases", "flat,top", "--trials", "1"], [*LIGHT, "n", "", "c", "", "c"])
    flat, top = attempts_of(folder)
    assert (flat["layer"], top["layer"]) == ("flat", "top") and top["behind"] == "stack" and top["hover_raise_m"] == config.TILE_THICKNESS_M
    assert top["hover_clearance_m"] - flat["hover_clearance_m"] == pytest.approx(config.TILE_THICKNESS_M, abs=0.004)
    assert top["above_own_level_m"] == pytest.approx(top["grasp_clearance_m"] - config.TILE_THICKNESS_M)


def test_a_cycle_asks_for_a_feed_every_second_draw_and_for_the_wall_put_right_after_a_miss(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder = run(script, ["--cases", "cycle", "--trials", "4"],
                 [*LIGHT, "n", "", "c", "p", "", "c", "c"])     # lay the wall; top clean; bottom pushed, then one step to clear and feed; top; bottom
    events = read_events(folder)
    rows = attempts_of(folder)
    assert [r["layer"] for r in rows] == ["top", "bottom", "top", "bottom"] and [r["feed_index"] for r in rows] == [0, 0, 1, 1]
    assert [e["before_draw"] for e in events if e["type"] == "wall_feed"] == [3]
    assert [e["draw"] for e in events if e["type"] == "wall_put_right"] == [2]


def test_quitting_at_the_outcome_keeps_the_attempt_but_doesnt_score_it(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder = run(script, ["--trials", "3"], [*LIGHT, "n", "", "c", "", "q"])
    assert [r["outcome"] for r in attempts_of(folder)] == [wall_test.CLEAN, None]
    assert len(wall_test.attempts(folder)) == 1


def test_the_jaws_must_have_read_a_tile_for_an_attempt_to_be_clean(monkeypatch, capsys):
    import wall_test as script
    monkeypatch.setattr(script, "drain_stdin", lambda: None)
    answers = iter(["c", "", "a"])
    assert script.ask_outcome(lambda _: next(answers), False, None, None) == "closed on air"
    assert "isn't clean" in capsys.readouterr().out


def test_a_push_check_scans_each_push_and_records_the_verdict_without_an_arm(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder = run(script, ["--push-check", "--trials", "3"], ["", "", "", ""])
    events = read_events(folder)
    assert len([e for e in events if e["type"] == "wall_push"]) == 3
    summary = next(e for e in events if e["type"] == "wall_push_summary")
    assert summary["passed"] and summary["offset_from_station_mm"] == pytest.approx(0.0, abs=0.01) and folder.name.endswith("_wall_push")


def test_a_hover_check_goes_to_each_station_and_saves_the_wrist_view(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder = run(script, ["--hover-check", "--stations", "F,L"], ["", ""])
    hovers = [e for e in read_events(folder) if e["type"] == "wall_hover"]
    assert [e["station"] for e in hovers] == ["F", "L"] and all(e["reachable"] for e in hovers)
    assert (folder / "hover_F.jpg").exists() and (folder / "hover_L.jpg").exists() and folder.name.endswith("_wall_hover")


def test_the_report_scores_real_sessions_and_leaves_out_simulated_ones(monkeypatch, tmp_path, capsys):
    script = script_in(monkeypatch, tmp_path)
    run(script, ["--trials", "1"], [*LIGHT, "n", "", "c"])          # a simulated session: not evidence
    script.main(["--report"])
    assert "No real wall-test sessions" in capsys.readouterr().out
    real = tmp_path / "20261110_100000_wall"
    real.mkdir()
    events = [{"type": "session_start", "sim": False},
              {"type": "wall_attempt", "station": "F", "case": "top", "layer": "top", "held": True, "above_own_level_m": 0.012, "outcome": wall_test.CLEAN}]
    (real / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    script.main(["--report"])
    out = capsys.readouterr().out
    assert real.name in out and "All 1 session" in out and "1/1" in out


@pytest.mark.parametrize("argv", [["--stations", "X"], ["--cases", "nope"], ["--cases", "cycle,flat"], ["--cases", "cycle", "--stations", "F,L"],
                                  ["--trials", "0"], ["--push-check", "--stations", "F,L"]])
def test_a_bad_request_fails_before_any_hardware(monkeypatch, tmp_path, argv):
    script = script_in(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        script.main(["--sim", *argv], ask=lambda prompt: pytest.fail("asked a question"))
