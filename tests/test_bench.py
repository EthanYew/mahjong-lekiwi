"""The bench end to end on simulated hardware: trials write a complete, reportable session."""
import argparse
import sys
from pathlib import Path

import pytest

from mahjong import config
from mahjong.hover_correction import read_all
from mahjong.session import read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_sim_trials_write_a_complete_session(tmp_path, monkeypatch):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    args = argparse.Namespace(policy=None, trials=2, tile="8D", sim=True, device=None, hover_perturb_mm=None)
    b = bench.Bench(args)
    try:
        b.arm.rest()
        b.trials(2, "8D")
    finally:
        b.close()
    session = next(tmp_path.iterdir())
    events = read_events(session)
    attempts = [e for e in events if e["type"] == "attempt"]
    assert [a["success"] for a in attempts] == [True, True]
    assert sum(e["type"] == "return" for e in events) == 2
    assert len({a["attempt_dir"] for a in attempts}) == 2
    for a in attempts:
        assert (session / a["attempt_dir"] / "overview_before.jpg").exists()
    assert (session / "session.log").read_text()
    samples = read_all(session / "hover_corrections.jsonl")     # sim samples stay out of data/
    assert [r["source"] for r in samples] == ["bench", "bench"] and not samples[0].get("excluded")


def test_perturbed_trials_offset_the_hover_and_set_their_samples_aside(tmp_path, monkeypatch):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    offsets = bench.parse_perturb("0,6x")
    assert offsets == [(0.0, 0.0), (0.006, 0.0)]
    args = argparse.Namespace(policy=None, trials=2, tile="8D", sim=True, device=None, hover_perturb_mm=offsets)
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
    args = argparse.Namespace(policy=None, trials=trials, tile="8D", sim=True, device=None, hover_perturb_mm=None)
    b = bench.Bench(args)
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
