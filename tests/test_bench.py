"""The bench end to end on simulated hardware: trials write a complete, reportable session."""
import sys
from pathlib import Path

import pytest

from mahjong import config
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


def test_a_neighbour_nudged_during_a_trial_is_knocked(tmp_path, monkeypatch):
    import builtins
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch, trials=1)
    after_each_run(b, lambda: b.realsense.nudge("3B", 0.008, 0.0))
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt: asked.append(prompt) or "")   # Enter: as the scans say
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert len(asked) == 1 and "3B moved 8mm during the pickup" in asked[0]
    assert events.index(next(e for e in events if e["type"] == "scan_review")) > \
        events.index(next(e for e in events if e["type"] == "trials_done"))        # asked once the batch is done
    (knock,) = interventions.derive(events)
    assert (knock["kind"], knock["during"], knock["source"]) == ("knocked", "pickup", "confirmed")
    assert [m["tile_name"] for m in knock["changes"]["moved"]] == ["3B"]


def test_a_change_between_commands_is_asked_about_before_the_arm_moves(tmp_path, monkeypatch):
    import builtins
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch)
    moves = []
    asked = []

    def answer(prompt):
        asked.append((prompt, len(b.arm.robot.sent)))
        return "4" if "since the last scan" in prompt else ""     # the 3B "move": a detection glitch
    monkeypatch.setattr(builtins, "input", answer)
    try:
        b.scan("choose")
        b.realsense.nudge("3B", 0.008, 0.0)
        moves.append(len(b.arm.robot.sent))
        b.pickup("8D")
    finally:
        b.close()
    (glitch,) = interventions.derive(read_events(b.session.dir))
    assert (glitch["kind"], glitch["class"]) == ("scan_glitch", "observation")
    question = next((p, sent) for p, sent in asked if "since the last scan" in p)
    assert "3B moved 8mm" in question[0] and question[1] == moves[0]    # before the arm moved for the pickup


def test_marked_trials_are_asked_about_once_at_the_end(tmp_path, monkeypatch):
    import builtins
    from mahjong import interventions
    b = sim_bench(tmp_path, monkeypatch)
    after_each_run(b, lambda: b.events.update(exit_early=True))      # → during every ACT
    answers = iter(["1", ""])        # the first: touched the tile; the second: Enter
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt: asked.append(prompt) or next(answers))
    try:
        b.trials(2, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert len(asked) == 2 and all("marked at 2.0s" in p for p in asked)
    reviews = [e for e in events if e["type"] == "attempt_review"]
    assert [r["mark_kind"] for r in reviews] == ["touched_tile", "unspecified"]
    found = interventions.derive(events)
    assert [(i["kind"], i["class"]) for i in found] == [("touched_tile", "rescue"), ("unspecified", "rescue")]
    s = interventions.summary(found, [e for e in events if e["type"] == "attempt"])
    assert (s["clean_successes"], s["assisted_successes"]) == (0, 2)


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
    import types
    import bench
    from mahjong import hardware, policy

    def unknown(policy_id):
        raise SystemExit(f"unknown policy {policy_id}")
    connected = []
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(policy, "Policy", unknown)
    monkeypatch.setattr(hardware, "connect", lambda needed, **kw: connected.append(needed))
    monkeypatch.setitem(sys.modules, "lerobot.common.control_utils",
                        types.SimpleNamespace(init_keyboard_listener=lambda: (None, {})))
    with pytest.raises(SystemExit, match="unknown policy pickup_mat_typo"):
        bench.Bench(bench.parse_args(["--policy", "pickup_mat_typo"]))
    assert connected == [] and not list(tmp_path.iterdir())
