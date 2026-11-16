"""The bench end to end on simulated hardware: trials write a complete, reportable session."""
import sys
from pathlib import Path

import numpy as np
import pytest

from mahjong import config, detector, wrist_photos
from mahjong.arm import release_height
from mahjong.hover_correction import read_all
from mahjong.session import read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_sim_trials_write_a_complete_session(tmp_path, monkeypatch):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "2", "--tile", "8D"]))
    try:
        b.arm.rest()
        b.trials(2, "8D")
    finally:
        b.close()
    session = next(tmp_path.iterdir())
    events = read_events(session)
    attempts = [e for e in events if e["type"] == "attempt"]
    assert [a["success"] for a in attempts] == [True, True]
    assert [a["torn_frames"] for a in attempts] == [0, 0]      # without it session_report calls a sim session pre-2026-11-13
    returns = [e for e in events if e["type"] == "return"]
    assert len(returns) == 2
    for a, r in zip(attempts, returns):        # let go just above where that pickup's jaws closed
        assert r["grasp_clearance_m"] == a["grasp_clearance_m"] is not None
        assert r["release_target_m"] == release_height(r["grasp_clearance_m"])
    assert len({a["attempt_dir"] for a in attempts}) == 2
    for a in attempts:
        assert (session / a["attempt_dir"] / "overview_before.jpg").exists()
    assert (session / "session.log").read_text()
    samples = read_all(session / "hover_corrections.jsonl")     # sim samples stay out of data/
    assert [r["source"] for r in samples] == ["bench", "bench"] and not samples[0].get("excluded")


def hover_once(tmp_path, monkeypatch, camera=None):
    """The bench's hover command on the sim, with the first tile chosen: the session folder."""
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_: "1")
    b = bench.Bench(bench.parse_args(["--sim"]))
    try:
        b.camera = camera or b.camera
        b.hover()
    finally:
        b.close()
    return next(tmp_path.iterdir())


def test_hover_saves_a_wrist_photo_where_the_wrist_photos_are_gathered_from(tmp_path, monkeypatch):
    monkeypatch.setattr(detector, "is_sim", lambda _: False)     # photos() leaves simulated sessions out
    session = hover_once(tmp_path, monkeypatch)
    hover = next(e for e in read_events(session) if e["type"] == "hover")
    assert (session / hover["attempt_dir"] / "wrist_0000.jpg").exists()
    photo, = wrist_photos.photos(tmp_path)
    assert (photo["source"], photo["name"]) == ("bench", hover["tile"])
    assert not [e for e in read_events(session) if e["type"] == "attempt"]    # a hover isn't a pickup attempt


def test_hover_with_a_lost_wrist_camera_reopens_it_and_still_logs_the_hover(tmp_path, monkeypatch):
    import bench

    class Lost:
        def flush(self):
            pass

        def read(self):
            return False, None

        def release(self):
            pass

    reopened = []
    monkeypatch.setattr(bench.Bench, "reopen_camera", lambda self: reopened.append(True))
    monkeypatch.setattr("mahjong.camera.read_within", lambda camera: (False, None))
    session = hover_once(tmp_path, monkeypatch, camera=Lost())
    hover = next(e for e in read_events(session) if e["type"] == "hover")
    assert reopened == [True] and hover["attempt_dir"] is None
    assert not list(session.glob("*/wrist_0000.jpg"))


def test_perturbed_trials_offset_the_hover_and_set_their_samples_aside(tmp_path, monkeypatch):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    args = bench.parse_args(["--sim", "--trials", "2", "--tile", "8D", "--hover-perturb-mm", "0,6x"])
    assert args.hover_perturb_mm == [(0.0, 0.0), (0.006, 0.0)]
    b = bench.Bench(args)
    try:
        b.arm.rest()
        b.trials(2, "8D")
    finally:
        b.close()
    session = next(tmp_path.iterdir())
    first, second = [e for e in read_events(session) if e["type"] == "attempt"]
    assert (first["hover_perturb_xy"], second["hover_perturb_xy"]) == ([0.0, 0.0], [0.006, 0.0])
    assert second["hover_xy"][0] - first["hover_xy"][0] == pytest.approx(0.006, abs=1e-6)   # the sim mat returns the tile to ~0.1um
    assert first["correction"] == b.correction.name and len(first["hover_sag_xyz_m"]) == 3
    plain, perturbed = read_all(session / "hover_corrections.jsonl")
    assert not plain.get("excluded") and perturbed["excluded"]


def test_hover_gripper_overrides_the_policys_opening_and_is_logged(tmp_path, monkeypatch):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "1", "--tile", "8D", "--hover-gripper", "25"]))
    opened_at, run = [], b.policy.run

    def wrapped(arm, *args, **kwargs):
        opened_at.append(arm.commanded()["gripper"])
        return run(arm, *args, **kwargs)
    b.policy.run = wrapped
    try:
        b.arm.rest()
        b.trials(1, "8D")
    finally:
        b.close()
    assert opened_at == [pytest.approx(25.0)] and b.policy.hover["gripper"] != 25.0
    events = read_events(next(tmp_path.iterdir()))
    setup = next(e for e in events if e["type"] == "bench_setup")
    assert setup["hover_gripper_override"] == 25.0 and setup["hover"]["gripper"] == 25.0
    assert [e["hover_gripper"] for e in events if e["type"] == "attempt"] == [25.0]


def test_hover_lower_mm_lowers_the_hover_saves_no_samples_and_is_logged(tmp_path, monkeypatch):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "1", "--tile", "8D", "--hover-lower-mm", "6"]))
    own = b.policy.hover
    lowered = b.hover_of(b.policy)
    assert lowered["offset_m"] == pytest.approx(own["offset_m"] - 0.006) and lowered["clearance_m"] == pytest.approx(own["clearance_m"] - 0.006)
    assert lowered["gripper"] == own["gripper"]
    try:
        b.arm.rest()
        b.trials(1, "8D")
    finally:
        b.close()
    session = next(tmp_path.iterdir())
    events = read_events(session)
    assert next(e for e in events if e["type"] == "bench_setup")["hover_lower_mm"] == 6.0
    attempt = next(e for e in events if e["type"] == "attempt")
    assert attempt["success"] and attempt["hover_clearance_m"] < own["clearance_m"] - 0.002       # it hovered lower
    assert not (session / "hover_corrections.jsonl").exists()                                    # and saved no sample of it


def test_hover_lower_mm_is_limited_to_what_the_settle_can_reach():
    import bench
    with pytest.raises(SystemExit):
        bench.parse_args(["--sim", "--hover-lower-mm", "12"])
    assert bench.parse_args(["--sim", "--hover-lower-mm", "-8"]).hover_lower_mm == -8.0


def sim_bench(tmp_path, monkeypatch, trials=2):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    b = bench.Bench(bench.parse_args(["--sim", "--trials", str(trials), "--tile", "8D"]))
    b.arm.rest()
    return b


def after_each_run(b, then):
    run = b.policy.run

    def wrapped(*args, **kwargs):
        then()
        return run(*args, **kwargs)
    b.policy.run = wrapped


def test_scans_around_each_trial_and_a_clean_run_needs_no_one(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch)
    try:
        b.trials(2, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    labels = [e["label"] for e in events if e["type"] == "detections"]
    assert labels == ["choose", "after_pickup", "after_return"] * 2
    after_pickup = next(e for e in events if e.get("label") == "after_pickup")
    assert "8D" not in [t["tile_name"] for t in after_pickup["tiles"]]       # it's in the jaws
    assert interventions.derive(events) == []
    assert events[-1]["reason"] == "quit"


NEIGHBOUR = {"tile_name": "5C", "xy": [0.160, -0.220]}   # 55mm from the sim's 8D, away from its corrected hover (3B: 183mm)


def test_a_neighbour_that_moved_with_no_bump_keyed_is_asked_about_right_after_the_attempt(tmp_path, monkeypatch):
    import builtins
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    b.realsense.tiles.append(dict(NEIGHBOUR))
    after_each_run(b, lambda: b.realsense.nudge("5C", 0.008, 0.0))
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt: asked.append((prompt, b.held is not None)) or "")
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert len(asked) == 1 and "5C moved 8mm during the pickup" in asked[0][0]
    assert asked[0][1]          # still holding 8D: asked before the return, not at the end of the batch
    (knock,) = interventions.derive(events)
    assert (knock["kind"], knock["during"], knock["source"]) == ("knocked", "pickup", "confirmed")
    assert [m["tile_name"] for m in knock["changes"]["moved"]] == ["5C"]


def test_a_bump_keyed_during_the_pickup_makes_a_neighbours_move_a_knock_without_asking(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    b.realsense.tiles.append(dict(NEIGHBOUR))
    b.keys.press("b")                       # at the menu: not during a pickup, so not a mark
    after_each_run(b, lambda: (b.realsense.nudge("5C", 0.0, 0.008), b.keys.press("b")))
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert asked == []
    (pressed,) = [e for e in events if e["type"] == "keys"]
    assert pressed["during"] == "pickup" and [k["kind"] for k in pressed["presses"]] == ["collided"]
    found = interventions.derive(events)
    assert sorted((i["kind"], i["source"]) for i in found) == [("collided", "key"), ("knocked", "confirmed")]
    assert next(i for i in found if i["kind"] == "knocked")["by"] == "key"


def test_a_bump_keyed_during_the_return_is_the_returns(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    b.realsense.tiles.append(dict(NEIGHBOUR))
    place = b.arm.place

    def bumping(*args, **kwargs):
        b.realsense.nudge("5C", 0.008, 0.0)
        b.keys.press("B")
        return place(*args, **kwargs)
    b.arm.place = bumping
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    assert asked == []
    found = interventions.derive(read_events(b.session.dir))
    assert sorted((i["kind"], i["during"]) for i in found) == [("collided", "return"), ("knocked", "return")]


def test_a_far_tile_that_moved_and_stayed_is_settled_by_the_scans_without_asking(tmp_path, monkeypatch, capsys):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch)
    nudged = []

    def once():
        if not nudged:
            nudged.append(b.realsense.nudge("3B", 0.008, 0.0))
    after_each_run(b, once)
    asked = no_questions(monkeypatch)
    try:
        b.trials(2, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert asked == []
    (moved,) = interventions.derive(events)
    first = next(e for e in events if e["type"] == "attempt")["attempt_dir"]
    assert (moved["kind"], moved["class"], moved["source"], moved["attempt_dir"]) == \
        ("moved_elsewhere", "observation", "scans", first)
    assert "3B moved 8mm (scan 2), still so 3 scans later" in capsys.readouterr().out


def test_a_change_between_commands_that_goes_back_is_a_glitch_and_asks_no_one(tmp_path, monkeypatch, capsys):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch)
    asked = no_questions(monkeypatch)
    try:
        b.scan("choose")
        b.realsense.nudge("3B", 0.008, 0.0)
        b.scan("choose")
        b.realsense.nudge("3B", -0.008, 0.0)      # the detector's: back where it was
        b.trials(1, "8D")
    finally:
        b.close()
    assert asked == []
    found = interventions.derive(read_events(b.session.dir))
    assert [(i["kind"], i["source"]) for i in found] == [("scan_glitch", "scans")]
    assert "Detection glitches settled by later scans: 3B x1" in capsys.readouterr().out


def test_a_marked_trial_is_asked_about_right_after_it(tmp_path, monkeypatch):
    import builtins
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch)
    after_each_run(b, lambda: b.events.update(exit_early=True))      # → during every ACT
    answers = iter(["1", ""])        # the first: touched the tile; the second: Enter
    asked = []

    def answer(prompt):
        returns = sum(e["type"] == "return" for e in read_events(b.session.dir))
        asked.append((prompt, returns))
        return next(answers)
    monkeypatch.setattr(builtins, "input", answer)
    try:
        b.trials(2, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert all("marked at 2.0s" in p for p, _ in asked)
    assert [r for _, r in asked] == [0, 1]           # each before that attempt's return
    attempts = [e for e in events if e["type"] == "attempt"]
    assert [a.get("mark_kind") for a in attempts] == ["touched_tile", "unspecified"]
    found = interventions.derive(events)
    assert [(i["kind"], i["class"]) for i in found] == [("touched_tile", "rescue"), ("unspecified", "rescue")]
    s = interventions.summary(found, attempts)
    assert (s["clean_successes"], s["assisted_successes"]) == (0, 2)


def test_a_mark_with_a_key_asks_nothing(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    after_each_run(b, lambda: (b.events.update(exit_early=True), b.keys.press("l"), b.keys.press("h")))
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    assert asked == []
    events = read_events(b.session.dir)
    (look,) = interventions.derive(events)        # h is no key: not noted
    assert (look["kind"], look["class"], look["source"], look["during"]) == ("look", "observation", "key", "pickup")


def test_a_discard_that_missed_and_a_note(tmp_path, monkeypatch):
    import builtins
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch)
    answers = iter(["", "n", "1", "rotated 3B back"])   # picked it up: Enter; in the container: n; note: straightened
    monkeypatch.setattr(builtins, "input", lambda prompt: next(answers))
    try:
        b.pickup("8D")
        b.discard()
        b.note()
    finally:
        b.close()
    found = interventions.derive(read_events(b.session.dir))
    assert [(i["kind"], i["class"]) for i in found] == [("discard_missed", "rescue"), ("straightened", "rescue")]
    assert found[1]["note"] == "rotated 3B back" and found[1]["attempt_dir"] == found[0]["attempt_dir"]


def test_paired_policies_take_turns_from_the_same_offset_and_their_own_hover(tmp_path, monkeypatch):
    import bench
    from mahjong.sim import SimPolicy
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    b = bench.Bench(bench.parse_args(["--sim", "--policy", "a,b", "--trials", "3", "--tile", "8D",
                                      "--hover-perturb-mm", "0,6x"]))
    b.policies[1].hover = {**SimPolicy.hover, "clearance_m": SimPolicy.hover["clearance_m"] + 0.01}
    try:
        b.arm.rest()
        b.trials(3, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    attempts = [e for e in events if e["type"] == "attempt"]
    assert [a["policy"] for a in attempts] == ["a", "b", "b", "a", "a", "b"]          # a b, b a, a b
    assert [a["hover_perturb_xy"][0] for a in attempts] == [0.0, 0.0, 0.006, 0.006, 0.0, 0.0]
    clearance = {a["policy"]: a["hover_clearance_m"] for a in attempts}
    assert clearance["b"] - clearance["a"] == pytest.approx(0.01, abs=0.002)          # each from its own hover
    (done,) = [e for e in events if e["type"] == "trials_done"]
    assert done["policies"] == ["a", "b", "b", "a", "a", "b"] and all(done["results"])
    assert b.policy.id == "a"
    with pytest.raises(SystemExit):
        bench.parse_policies("a,a")


def test_a_layout_walk_visits_every_spot_and_skips_an_empty_one(tmp_path, monkeypatch):
    import builtins
    import bench
    from mahjong import interventions, layout
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt: asked.append(prompt) or "s")    # nothing to put there: skip
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(config, "LAYOUTS", tmp_path / "layouts.json")
    layout.save("walk", [{"id": "L", "xy": [0.21, 0.02]}, {"id": "empty", "xy": [0.25, 0.10]},
                         {"id": "R", "xy": [0.160, -0.165]}])
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "2", "--layout", "walk"]))
    b.realsense.tiles = [t for t in b.realsense.tiles if t["tile_name"] != "3B"]   # the sim lift can't reach it:
    b.realsense.tiles.append({"tile_name": "5C", "xy": [0.212, 0.021]})           # a tile placed near spot L instead
    try:
        b.arm.rest()
        b.trials(2)
    finally:
        b.close()
    events = read_events(b.session.dir)
    attempts = [e for e in events if e["type"] == "attempt"]
    assert [(a["spot"], a["tile"]) for a in attempts] == [("L", "5C")] * 2 + [("R", "8D")] * 2
    assert all(a["success"] for a in attempts)
    skips = [e for e in events if e["type"] == "attempt_skipped"]
    assert [(s["reason"], s["spot"]) for s in skips] == [("not_at_spot", "empty")]    # tried once, then skipped
    assert [(i["kind"], i["class"]) for i in interventions.derive(events)] == [("spot_empty", "observation")]
    (done,) = [e for e in events if e["type"] == "trials_done"]
    assert done["spots"] == ["L", "L", "R", "R"] and done["skipped_spots"] == ["empty"]
    assert len(asked) == 1 and "spot empty" in asked[0]


FACE_DOWN_SPOTS = [{"id": "L", "xy": [0.21, 0.02]}, {"id": "R", "xy": [0.160, -0.165]}]


def face_down_bench(tmp_path, monkeypatch, argv=(), sees_backs=False):
    """A face-down bench on the sim. sees_backs: detector weights with the back class (False: the older ones, which box none)."""
    import bench
    from mahjong import layout, sim
    monkeypatch.setattr(sim, "SEES_BACKS", sees_backs)
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(config, "LAYOUTS", tmp_path / "layouts.json")
    layout.save("down", FACE_DOWN_SPOTS)
    return bench.Bench(bench.parse_args(["--sim", "--layout", "down", "--face-down", *argv]))


def test_face_down_trials_aim_at_each_spots_saved_position_and_log_no_samples_or_landings(tmp_path, monkeypatch):
    from mahjong import scorecard
    b = face_down_bench(tmp_path, monkeypatch, ["--trials", "2"])
    try:
        b.arm.rest()
        b.trials(2)
    finally:
        b.close()
    events = read_events(b.session.dir)
    attempts = [e for e in events if e["type"] == "attempt"]
    spot_xy = {s["id"]: s["xy"] for s in FACE_DOWN_SPOTS}
    assert [a["spot"] for a in attempts] == ["L", "L", "R", "R"] and all(a["success"] for a in attempts)
    assert all(a["name_source"] == "saved_position" and a["detection_xy"] == spot_xy[a["spot"]] for a in attempts)
    assert len([e for e in events if e["type"] == "return"]) == 4
    assert not [e for e in events if e["type"] == "return_landed"]      # a green back can't be seen landing
    assert not (b.session.dir / "hover_corrections.jsonl").exists()      # a saved position isn't a detection
    assert not (b.session.dir / "return_corrections.jsonl").exists()
    assert [e["spots"] for e in events if e["type"] == "face_down_laid"] == [["L", "R"]]    # asked once, not per round
    assert "face-down tiles, picked by saved position" in scorecard.off_standard(events, {"surface": None, "layout": "down"})


def test_face_down_pickup_at_the_menu_asks_which_spot(tmp_path, monkeypatch):
    answers = iter(["X", "r", "", "y"])        # no such spot, R, the lay-it prompt's Enter, "picked it up?"
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    b = face_down_bench(tmp_path, monkeypatch)
    try:
        b.ask_names = input             # a real mat: the person is asked to lay it
        b.arm.rest()
        assert b.pickup() is True
    finally:
        b.close()
    (attempt,) = [e for e in read_events(b.session.dir) if e["type"] == "attempt"]
    assert attempt["spot"] == "R" and attempt["detection_xy"] == FACE_DOWN_SPOTS[1]["xy"]
    assert b.held["home_xy"] == FACE_DOWN_SPOTS[1]["xy"] and b.held["saved_position"]


@pytest.mark.parametrize("argv", [["--face-down"], ["--face-down", "--layout", "x", "--tile", "8D"]])
def test_face_down_needs_a_layout_and_no_tile_name(argv):
    import bench
    with pytest.raises(SystemExit):
        bench.parse_args(argv)


def test_face_down_unattended_needs_weights_that_see_backs(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="back"):
        face_down_bench(tmp_path, monkeypatch, ["--trials", "1", "--unattended"])


def test_a_paired_round_of_the_standard_layout_reaches_the_gates(tmp_path, monkeypatch):
    import builtins
    import bench
    from mahjong import layout, scorecard
    monkeypatch.setattr(builtins, "input", lambda prompt: "s")
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    standard = scorecard.standard_setup()
    b = bench.Bench(bench.parse_args(["--sim", "--policy", "a,b", "--trials", "1", "--layout", standard["layout"],
                                      "--hover-gripper", "21"]))
    spots = layout.load(standard["layout"])["spots"]
    b.realsense.tiles = [{"tile_name": s["label"], "xy": list(s["xy"])} for s in spots]    # the sim's own tiles aren't there
    try:
        b.arm.rest()
        b.trials(1)
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert "a simulated run (--sim)" in scorecard.off_standard(events)                     # as run, it never counts
    for e in events:
        if e["type"] == "session_start":
            e["conditions"] = {"surface": standard["surface"]}                             # --sim asks no setup questions
            e["sim"] = False                                                               # stand in for a real run
    assert [e for e in events if e["type"] == "attempt_skipped"] == []
    assert len(scorecard.pairs(events, "a", "b")) == len(spots)
    assert scorecard.off_standard(events) == []
    text = "\n".join(scorecard.report(events))
    assert "iteration check:" in text and f"of {len(spots)})" in text and "play-ready, a:" in text
    assert "hover openings differ" not in text


def test_a_tile_not_detected_in_trials_is_put_back_by_a_person_and_rescanned(tmp_path, monkeypatch):
    import builtins
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    mat = b.realsense
    tile = next(t for t in mat.tiles if t["tile_name"] == "8D")
    mat.tiles.remove(tile)                          # not seen: a flickering read, or knocked off its spot
    asked = []

    def person(prompt):
        asked.append(prompt)
        mat.tiles.append(tile)
        return ""
    monkeypatch.setattr(builtins, "input", person)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert len(asked) == 1 and "8D not detected" in asked[0] and "Enter to rescan" in asked[0]
    assert [a["success"] for a in events if a["type"] == "attempt"] == [True]
    assert not any(e["type"] == "attempt_skipped" for e in events)
    assert [(i["kind"], i["class"]) for i in interventions.derive(events)] == [("tile_reset", "reset")]


def test_every_return_in_a_trial_run_lands_at_the_tiles_home(tmp_path, monkeypatch):
    """2026-10-27: returning each round to its own grasp point walked the tile 8cm in 10 rounds."""
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "3", "--tile", "8D", "--hover-perturb-mm", "0,6x,-6y"]))
    try:
        b.arm.rest()
        b.trials(3, "8D")
    finally:
        b.close()
    events = read_events(next(tmp_path.iterdir()))
    attempts = [e for e in events if e["type"] == "attempt"]
    returns = [e for e in events if e["type"] == "return"]
    assert len({tuple(a["grasp_xy"]) for a in attempts}) == 3          # the sim grasps where it hovered: offset each round
    assert len(returns) == 3 and all(r["home_xy"] == attempts[0]["detection_xy"] for r in returns)
    assert all(e["error_m"] < 0.001 for e in events if e["type"] == "return_landed")


def test_an_attempt_the_wrist_camera_dropped_out_of_is_void_and_run_again(tmp_path, monkeypatch):
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    run = b.policy.run
    calls = []

    def camera_lost_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:         # the policy never saw a frame: stopped at the hover, empty-handed
            return {"steps": 0, "duration_s": 0.0, "stop_reason": "camera", "aborted": False, "held_at_s": None,
                    "slipped": False, "stall_warning": False, "descended": False, "floored_steps": 0,
                    "grasp_step": None, "grasp_xy": None, "marks_s": []}
        return run(*args, **kwargs)
    b.policy.run = camera_lost_once
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(next(tmp_path.iterdir()))
    void = [e for e in events if e["type"] == "attempt_void"]
    attempts = [e for e in events if e["type"] == "attempt"]
    assert len(calls) == 2 and len(void) == 1
    assert [a["success"] for a in attempts] == [True]
    assert void[0]["attempt_dir"] != attempts[0]["attempt_dir"]
    assert next(e for e in events if e["type"] == "trials_done")["results"] == [True]


def test_an_attempt_on_torn_wrist_frames_is_void_and_run_again_with_its_reason(tmp_path, monkeypatch):
    import bench
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    run = b.policy.run
    calls = []

    def torn_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:         # frames kept coming, torn, until the policy loop stopped it short of the tile
            return {"steps": 40, "duration_s": 1.3, "stop_reason": "camera", "camera_problem": "torn", "torn_frames": 5,
                    "aborted": False, "held_at_s": None, "slipped": False, "stall_warning": False, "descended": True,
                    "floored_steps": 0, "grasp_step": None, "grasp_xy": None, "marks_s": []}
        return run(*args, **kwargs)
    b.policy.run = torn_once
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(next(tmp_path.iterdir()))
    void = [e for e in events if e["type"] == "attempt_void"]
    assert len(calls) == 2 and [v["reason"] for v in void] == [bench.VOID_TORN]
    assert [a["success"] for a in events if a["type"] == "attempt"] == [True]


def test_returns_aim_at_the_home_so_a_crept_tile_goes_back(tmp_path, monkeypatch):
    import builtins
    from mahjong.return_correction import landings
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    monkeypatch.setattr(builtins, "input", lambda prompt: "")      # the creep between rounds: as the scans say
    try:
        b.trials(1, "8D")
        b.realsense.nudge("8D", 0.010, 0.0)         # it crept 10mm since (a person, a knock)
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    first, second = [e for e in events if e["type"] == "return"]
    assert first["home_xy"] == second["home_xy"]
    (landed_1, landed_2) = [e for e in events if e["type"] == "return_landed"]
    assert landed_1["error_m"] < 0.001 and landed_2["error_m"] < 0.001      # the sim mat lands it where it's set down
    assert [r["error_m"] < 0.001 for r in landings(events)] == [True, True]
    assert len(read_all(b.session.dir / "return_corrections.jsonl")) == 2     # sim samples stay out of data/


def test_a_return_landing_far_off_its_home_is_put_back_by_a_person(tmp_path, monkeypatch):
    import builtins
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    mat = b.realsense
    b.arm.robot.on_release = lambda xy, clearance: mat.release([xy[0] + 0.03, xy[1]], clearance)     # lands 30mm off
    asked = []

    def person(prompt):
        asked.append(prompt)
        if "Put 8D back" in prompt:
            mat.nudge("8D", -0.03, 0.0)
        return ""
    monkeypatch.setattr(builtins, "input", person)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    (landed,) = [e for e in events if e["type"] == "return_landed"]
    assert landed["error_m"] == pytest.approx(0.03, abs=0.001)
    assert len(asked) == 1           # the reset, and nothing about a knock
    assert [e["label"] for e in events if e["type"] == "detections"][-1] == "after_reset"
    found = interventions.derive(events)
    assert [(i["kind"], i["class"]) for i in found] == [("tile_reset", "reset")]


def test_a_return_landing_a_little_off_is_only_noted(tmp_path, monkeypatch, capsys):
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    mat = b.realsense
    b.arm.robot.on_release = lambda xy, clearance: mat.release([xy[0], xy[1] + 0.015], clearance)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    assert "8D landed 15mm from its home (+0, +15)mm" in capsys.readouterr().out
    assert not any(e.get("label") == "after_reset" for e in read_events(b.session.dir))


def test_a_mistyped_policy_fails_before_anything_connects(tmp_path, monkeypatch):
    """A policy id that isn't registered used to fail after the hardware connected and the setup questions
    made a session folder: an empty session."""
    import bench
    from mahjong import hardware, policy

    def unknown(policy_id):
        raise SystemExit(f"unknown policy {policy_id}")
    connected = []
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(policy, "Policy", unknown)
    monkeypatch.setattr(hardware, "connect", lambda needed, **kw: connected.append(needed))
    with pytest.raises(SystemExit, match="unknown policy pickup_mat_typo"):
        bench.Bench(bench.parse_args(["--policy", "pickup_mat_typo"]))
    assert connected == [] and not list(tmp_path.iterdir())


def no_questions(monkeypatch):
    import builtins
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt: asked.append(prompt) or "")
    return asked


def test_a_neighbour_the_scan_missed_while_holding_and_the_next_saw_again_asks_no_one(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    after_each_run(b, lambda: b.realsense.flicker("3B"))         # gone from the after-pickup scan only
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert asked == []
    assert not any(e["type"] == "wrist_check" for e in events)    # the after-return scan saw it: no need to look
    assert {i["kind"] for i in interventions.derive(events)} == {"scan_glitch"}


def test_a_tile_the_scans_keep_missing_is_seen_by_the_wrist_camera_and_no_one_is_asked(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=2)
    after_each_run(b, lambda: b.realsense.flicker("3B", 2))     # after the pickup and after the return
    asked = no_questions(monkeypatch)
    try:
        b.trials(2, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    looks = [e for e in events if e["type"] == "wrist_check"]
    assert asked == [] and len(looks) == 2 and all(c["present"] and c["tile"] == "3B" for c in looks)
    assert all((b.session.dir / c["image"]).exists() for c in looks)
    labels = {e["scan"]: e["label"] for e in events if e["type"] == "detections"}
    assert [labels[c["at_scan"]] for c in looks] == ["after_return"] * 2     # once the held tile was put down
    assert [a["success"] for a in events if a["type"] == "attempt"] == [True, True]
    assert {i["kind"] for i in interventions.derive(events)} == {"scan_glitch"}   # and the replay agrees


def test_a_neighbour_really_gone_is_looked_at_and_is_a_knock_without_asking(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    mat = b.realsense
    mat.tiles.append(dict(NEIGHBOUR))
    after_each_run(b, lambda: mat.tiles.remove(next(t for t in mat.tiles if t["tile_name"] == "5C")))
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    (look,) = [e for e in events if e["type"] == "wrist_check"]
    assert look["tile"] == "5C" and look["present"] is False and look["fill"] < 0.1
    assert asked == []
    (knock,) = interventions.derive(events)
    assert (knock["kind"], knock["during"], knock["by"]) == ("knocked", "pickup", "wrist camera")


def test_a_tile_gone_with_a_knocked_neighbour_now_beside_its_spot_is_not_looked_at_but_watched(tmp_path, monkeypatch):
    """2026-10-30: a miss pushed 1B away and 3D to 20mm from 1B's spot; the look saw 3D's face and called it 1B.
    Nor is 5C now beside 3B's spot 3B coming back."""
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    mat = b.realsense
    mat.tiles.append({"tile_name": "5C", "xy": [0.151, 0.048]})

    def knock():
        mat.tiles.remove(next(t for t in mat.tiles if t["tile_name"] == "3B"))
        mat.nudge("5C", 0.0, -0.012)                            # 18mm from 3B's spot
    after_each_run(b, knock)
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    (look,) = [e for e in events if e["type"] == "wrist_check"]
    assert look["tile"] == "3B" and look["present"] is None and "5C is 18mm" in look["why"]
    assert asked == []                  # both far from 8D: the later scans settle them
    found = interventions.derive(events)
    assert sorted(interventions.changes_text(i["changes"]) for i in found if i["kind"] == "moved_elsewhere") == \
        ["3B gone", "5C moved 12mm"]


def test_a_tile_not_detected_at_its_home_is_picked_up_where_the_wrist_camera_sees_it(tmp_path, monkeypatch):
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
        b.realsense.flicker("8D")
        tiles = b.scan("choose", confirmed=["tile_reset"])      # the detector missed it; no look from the scan
        tile, reason, _ = b.find(tiles, "8D")
        assert tile is None and reason == "not_detected"
        found = b.look_at_home("8D")
        assert found["wrist_seen"] and list(found["xy"]) == b.homes.get("8D") and b.arm.is_at_rest()
        b.realsense.tiles.remove(next(t for t in b.realsense.tiles if t["tile_name"] == "8D"))
        b.scan_n += 1                                           # a new scan: its own looks
        assert b.look_at_home("8D") is None                     # not there: a person is asked, as before
    finally:
        b.close()
    assert asked == []



def bump_then_put_back(b, r_during_return):
    """b during the pickup as it knocks 5C 8mm; during the return a person puts 5C back (r, or not)."""
    b.realsense.tiles.append(dict(NEIGHBOUR))
    after_each_run(b, lambda: (b.realsense.nudge("5C", 0.008, 0.0), b.keys.press("b")))
    place = b.arm.place

    def putting_back(*args, **kwargs):
        b.realsense.nudge("5C", -0.008, 0.0)
        if r_during_return:
            b.keys.press("r")
        return place(*args, **kwargs)
    b.arm.place = putting_back


def test_a_knocked_tile_put_back_during_the_return_with_r_is_straightened_not_knocked_again(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    bump_then_put_back(b, r_during_return=True)
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    assert asked == []
    found = interventions.derive(read_events(b.session.dir))
    assert sorted((i["kind"], i["during"]) for i in found) == [
        ("collided", "pickup"), ("knocked", "pickup"), ("straightened", "return")]


def test_a_knocked_tile_put_back_without_r_can_be_answered_as_by_hand(tmp_path, monkeypatch):
    import builtins
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    bump_then_put_back(b, r_during_return=False)
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt: asked.append(prompt) or "2")     # 2: I did, by hand
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    assert len(asked) == 1 and "5C moved 8mm during the return. What moved it" in asked[0]
    found = interventions.derive(read_events(b.session.dir))
    assert sorted((i["kind"], i["during"]) for i in found) == [
        ("collided", "pickup"), ("knocked", "pickup"), ("straightened", "return")]


def test_a_tile_moved_by_hand_during_a_pickup_with_r_is_a_reset_not_watched(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    after_each_run(b, lambda: (b.realsense.nudge("3B", 0.010, 0.0), b.keys.press("r")))
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    assert asked == [] and b.changes.pending == []
    events = read_events(b.session.dir)
    (reset,) = interventions.derive(events)
    assert (reset["kind"], reset["source"], reset["by"]) == ("tile_reset", "confirmed", "key")
    (pressed,) = [e for e in events if e["type"] == "keys"]
    assert [k["kind"] for k in pressed["presses"]] == ["by_hand"]        # no intervention by itself


def test_a_watched_knock_put_back_and_noted_is_a_knock_not_a_glitch(tmp_path, monkeypatch):
    import builtins
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    mat = b.realsense
    mat.tiles.append(dict(NEIGHBOUR))
    after_each_run(b, lambda: mat.hidden.update({"5C": 99}))       # knocked off where the scans see it
    answers = iter(["1", ""])        # note: straightened, no words

    def person(prompt):
        return next(answers) if "What did you do" in prompt or "few words" in prompt else ""
    monkeypatch.setattr(builtins, "input", person)
    monkeypatch.setattr(b, "wrist_look", lambda tile, scan_n: {"present": None, "why": "test"})
    try:
        b.trials(1, "8D")
        mat.hidden.clear()                                          # put back on its spot
        b.note()
        b.scan("choose")
    finally:
        b.close()
    found = interventions.derive(read_events(b.session.dir))
    assert sorted(i["kind"] for i in found) == ["knocked", "straightened"]



def test_a_grip_keyed_low_in_the_pickup_and_high_in_the_return_are_counted_and_the_success_stands(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    after_each_run(b, lambda: b.keys.press("d"))
    place = b.arm.place

    def carried(*args, **kwargs):
        b.keys.press("u")
        return place(*args, **kwargs)
    b.arm.place = carried
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    assert asked == []
    events = read_events(b.session.dir)
    found = interventions.derive(events)
    assert sorted((i["kind"], i["class"], i["during"]) for i in found) == [
        ("grip_high", "observation", "return"), ("grip_low", "observation", "pickup")]
    s = interventions.summary(found, [e for e in events if e["type"] == "attempt"])
    assert (s["low_grips"], s["high_grips"], s["clean_successes"], s["rescues"]) == (1, 1, 1, 0)


def test_a_tile_the_scans_keep_losing_is_looked_at_every_time(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=5)
    after_each_run(b, lambda: b.realsense.flicker("3B", 2))
    asked = no_questions(monkeypatch)
    try:
        b.trials(5, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert asked == [] and len([e for e in events if e["type"] == "wrist_check"]) == 5
    assert {i["kind"] for i in interventions.derive(events)} == {"scan_glitch"}


def test_a_pale_surface_asks_once_a_place_and_then_goes_by_the_view(tmp_path, monkeypatch):
    import builtins

    import numpy as np

    from mahjong import wrist_check
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    tile = {"tile_name": "3B", "xy": next(t for t in b.realsense.tiles if t["tile_name"] == "3B")["xy"]}
    frame = np.full((480, 640, 3), 200, np.uint8)
    templates = []

    def pale(arm, camera, x, y, hover, template=None, back=False):
        templates.append(template)
        if template is None:
            return {"present": None, "fill": 0.7, "pale": True, "why": "pale", "hover_xy": [x, y], "frame": frame}
        return {"present": True, "fill": 0.7, "pale": True, "by": "same view", "hover_xy": [x, y], "frame": frame}
    monkeypatch.setattr(wrist_check, "check", pale)
    prompts = []
    monkeypatch.setattr(builtins, "input", lambda prompt: prompts.append(prompt) or "")
    try:
        first = b.wrist_look(tile, 1)
        second = b.wrist_look(tile, 2)
        assert first["present"] is True and first["by"] == "person" and len(prompts) == 1
        assert second["present"] is True and second["by"] == "same view" and len(prompts) == 1
        assert templates[0] is None and templates[1] is not None
        monkeypatch.setattr(builtins, "input", lambda prompt: "n")
        b.views.forget(tile["xy"])
        assert b.wrist_look(tile, 3)["present"] is False and b.views.template(tile["xy"]) is None
    finally:
        b.close()


def test_a_tile_lost_with_a_faint_box_still_there_needs_no_look(tmp_path, monkeypatch):
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    xy = next(t for t in b.realsense.tiles if t["tile_name"] == "3B")["xy"]
    b.realsense.faint = [{"tile_name": "7C", "confidence": 0.18, "xy": (xy[0] + 0.002, xy[1])}]
    after_each_run(b, lambda: b.realsense.flicker("3B", 2))
    asked = no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    (faint,) = [e for e in events if e["type"] == "tile_faint"]
    assert faint["tile"] == "3B" and faint["guess"] == "7C" and asked == []      # during the pickup, with the tile held
    assert not any(e["type"] == "wrist_check" for e in events)
    assert {i["kind"] for i in interventions.derive(events)} == {"scan_glitch"}


def test_a_faint_box_further_off_is_not_the_lost_tile(tmp_path, monkeypatch):
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    xy = next(t for t in b.realsense.tiles if t["tile_name"] == "3B")["xy"]
    b.realsense.faint = [{"tile_name": "3B", "confidence": 0.2, "xy": (xy[0] + 0.02, xy[1])}]    # 20mm: it may have moved
    after_each_run(b, lambda: b.realsense.flicker("3B", 2))
    no_questions(monkeypatch)
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert not any(e["type"] == "tile_faint" for e in events)
    assert any(e["type"] == "wrist_check" for e in events)


def test_a_spot_the_scan_cant_name_is_picked_up_from_its_box_unnamed(tmp_path, monkeypatch):
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    spot = {"id": "S13", "xy": [0.192, -0.165], "label": "WW"}
    b.realsense.faint = [{"tile_name": "7C", "confidence": 0.18, "xy": (0.195, -0.160)}]
    looks = []
    monkeypatch.setattr(b, "wrist_look", lambda tile, scan_n: looks.append(tile) or {"present": None, "why": "test"})
    try:
        tile = b.look_at_home(spot=spot, tiles=[])
        assert tile["tile_name"] == config.UNNAMED and tile["xy"] == (0.195, -0.160) and tile["faint"] and looks == []
        b.realsense.faint = []
        assert b.look_at_home(spot=spot, tiles=[]) is None
        assert looks[0]["tile_name"] == config.UNNAMED                 # never the spot's id or the layout's label
    finally:
        b.close()


def test_a_missing_tile_is_never_asked_for_by_the_layouts_label(tmp_path, monkeypatch):
    import builtins
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    prompts = []
    monkeypatch.setattr(builtins, "input", lambda prompt: prompts.append(prompt) or "s")
    try:
        assert b.reposition("No tile", spot={"id": "S13", "xy": [0.192, -0.165], "label": "WW"}) is False
    finally:
        b.close()
    assert "WW" not in prompts[0] and "the tile" in prompts[0]


def test_trials_on_a_layout_record_whats_on_the_mat_and_tie_attempts_to_it(tmp_path, monkeypatch):
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    b.realsense.tiles = [{"tile_name": "9D", "xy": [0.21, 0.02]}, {"tile_name": "3B", "xy": [0.21, -0.04]}]
    b.spots = [{"id": f"S{i + 1}", "xy": list(t["xy"])} for i, t in enumerate(b.realsense.tiles)]
    names = [t["tile_name"] for t in b.realsense.tiles]
    answers = iter(["r", "S1=2B", ""])
    b.ask_names = lambda prompt: next(answers)
    try:
        b.trials(1)
    finally:
        b.close()
    events = read_events(b.session.dir)
    placed = {e["spot"]: e for e in events if e["type"] == "placement"}
    assert (placed["S1"]["truth"], placed["S1"]["truth_source"], placed["S1"]["read_at_start"]) == ("2B", "person", names[0])
    assert (placed["S2"]["truth"], placed["S2"]["truth_source"]) == (names[1], "person")
    attempts = [e for e in events if e["type"] == "attempt"]
    assert {a["placement"] for a in attempts} == {placed["S1"]["id"], placed["S2"]["id"]}
    assert all(a["name_source"] == "read" for a in attempts)


def test_quitting_the_names_step_leaves_them_unconfirmed(tmp_path, monkeypatch):
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    b.spots = [{"id": "S1", "xy": list(b.realsense.tiles[0]["xy"])}]
    b.ask_names = lambda prompt: "q"
    try:
        b.name_the_mat()
    finally:
        b.close()
    placed = next(e for e in read_events(b.session.dir) if e["type"] == "placement")
    assert (placed["truth"], placed["truth_source"]) == (None, "unconfirmed")


def test_a_repositioned_tile_starts_a_new_placement_with_the_name_given(tmp_path, monkeypatch):
    import builtins
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    spot = {"id": "S1", "xy": [0.2, 0.0]}
    b.placements.start("S1", spot["xy"], truth="3C", truth_source="person")
    try:
        monkeypatch.setattr(builtins, "input", lambda prompt: "2c")
        assert b.reposition("No tile", spot=spot) is True
        monkeypatch.setattr(builtins, "input", lambda prompt: "")
        assert b.reposition("No tile", spot=spot) is True
        monkeypatch.setattr(builtins, "input", lambda prompt: "s")
        assert b.reposition("No tile", spot=spot) is False
    finally:
        b.close()
    placed = [e for e in read_events(b.session.dir) if e["type"] == "placement"]
    assert [p["truth"] for p in placed] == ["3C", "2C", "2C"]


def carry_bench(tmp_path, monkeypatch, returns=None, argv=()):
    """A sim mat where the camera and the arm disagree by an amount that changes with position, as on the real mat: one
    tile on spot Z, Y empty. A carry's landing is then only right if its aim is the destination's."""
    import bench
    from mahjong import layout
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(config, "LAYOUTS", tmp_path / "layouts.json")
    layout.save("carry", [{"id": "Z", "xy": [0.192, -0.165]}, {"id": "Y", "xy": [0.142, 0.09]}])
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "1", "--layout", "carry", "--carry", "Z,Y", *argv]))
    b.realsense.tiles = [{"tile_name": "8D", "xy": [0.192, -0.165]}]
    b.realsense.arm_offset = lambda xy: (0.040, 0.014 - 0.09 * xy[1])
    if returns is not None:
        b.returns = returns
    return b


def carried_landings(b):
    try:
        b.arm.rest()
        b.trials(1)
    finally:
        b.close()
    return [e for e in read_events(b.session.dir) if e["type"] == "return_landed"]


def same_place_sim_returns():
    from mahjong.return_correction import ReturnCorrection, return_record
    records = []
    for x in (0.14, 0.17, 0.20, 0.23):
        for y in (-0.17, -0.09, 0.0, 0.09, 0.17):
            gap = np.array([0.040, 0.014 - 0.09 * y])
            records.append(return_record([x, y], [x, y] + gap, [x, y] + gap, [x, y], home_xy=[x, y]))
    return ReturnCorrection(records)


def test_a_carried_tile_lands_at_its_slot_with_the_destinations_aim(tmp_path, monkeypatch):
    landed = carried_landings(carry_bench(tmp_path, monkeypatch, same_place_sim_returns()))
    assert len(landed) == 2 and all(e["carried"] for e in landed)         # there and back
    assert all(e["error_m"] < 0.003 for e in landed)


def test_the_old_aim_misses_on_the_same_mat_which_is_what_this_sim_is_for(tmp_path, monkeypatch):
    import builtins
    from mahjong.return_correction import ReturnCorrection
    class AskedToPutItBack(Exception):
        pass

    def input_(prompt):
        raise AskedToPutItBack(prompt)

    monkeypatch.setattr(builtins, "input", input_)      # it landed so far off that a person is asked: that's enough
    b = carry_bench(tmp_path, monkeypatch, ReturnCorrection([]))
    with pytest.raises(AskedToPutItBack, match="Put 8D back"):
        carried_landings(b)
    (landed,) = [e for e in read_events(b.session.dir) if e["type"] == "return_landed"]
    assert landed["error_m"] > 0.015


@pytest.mark.parametrize("argv", [["--carry", "A"], ["--carry", "A,A"], ["--carry", "A,B"]])
def test_carry_needs_two_spots_of_a_layout(argv):
    import bench
    with pytest.raises(SystemExit):
        bench.parse_args(["--sim", *argv])


def test_a_return_with_r_pressed_is_no_sample(tmp_path, monkeypatch):
    from mahjong.return_correction import check_landing
    import types
    events = []
    session = types.SimpleNamespace(dir=tmp_path, event=lambda kind, **f: events.append((kind, f)))
    path = tmp_path / "returns.jsonl"
    held = {"tile_name": "8D", "home_xy": [0.2, 0.0], "detection_xy": [0.2, 0.0], "grasp_xy": [0.24, 0.0],
            "attempt_dir": "001", "spot": "S1", "by_hand": True}
    check_landing(held, [0.24, 0.0], [{"tile_name": "8D", "xy": (0.2, 0.0)}], session, path, None, config.DEFAULT_HOVER)
    assert not path.exists() and events[0][1]["by_hand"]


def test_a_back_the_scans_lose_is_seen_by_the_wrist_camera_and_picked_up(tmp_path, monkeypatch):
    """2026-11-16: backs flickered out of scans, and a look at a face-down spot could only say "not there" (no white face),
    so the spot was skipped. The look finds the back by its shape, and the bench picks it up."""
    b = face_down_bench(tmp_path, monkeypatch, ["--trials", "1"], sees_backs=True)
    b.realsense.flicker_at(FACE_DOWN_SPOTS[1]["xy"], 50)       # R's back unseen by every scan
    asked = no_questions(monkeypatch)
    try:
        b.arm.rest()
        b.trials(1)
    finally:
        b.close()
    events = read_events(b.session.dir)
    looks = [e for e in events if e["type"] == "wrist_check"]
    assert looks and all(e["tile"] == config.BACK and e["present"] is True and e["by"] == "back shape" for e in looks)
    assert asked == []
    attempts = [e for e in events if e["type"] == "attempt"]
    assert [a["spot"] for a in attempts] == ["L", "R"] and all(a["success"] for a in attempts)
