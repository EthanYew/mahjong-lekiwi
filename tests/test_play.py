"""Autonomous play end to end on simulated hardware: a whole hand, and the questions that keep a wrong tile
from going unnoticed."""
import json
import sys
from pathlib import Path

import pytest

from mahjong import config
from mahjong.session import read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class Answers:
    """A person at the keyboard: answers in order (Enter once they run out), prompts kept."""
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else ""


def player(tmp_path, monkeypatch, ask, argv=("--sim",)):
    import play
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    return play.Player(play.parse_args(list(argv)), ask=ask)


def run(p):
    try:
        return p.play()
    finally:
        p.close()


def events_of(p):
    return read_events(p.session.dir)


def interventions_of(p):
    return [e for e in events_of(p) if e["type"] == "intervention"]


def test_a_sim_hand_plays_to_a_clean_win(tmp_path, monkeypatch):
    ask = Answers()
    p = player(tmp_path, monkeypatch, ask)
    assert run(p) == "clean"
    events = events_of(p)
    assert p.state["discards"] == ["GD", "RD"]          # GD straight from the zone; RD from H12, 9B moved in
    assert p.state["hand"]["H12"] == "9B" and p.state["phase"] == "won"
    end = next(e for e in events if e["type"] == "hand_end")
    assert (end["result"], end["turns"], end["grade"], end["silent_errors"]) == ("win", 2, "clean", 0)
    attempts = [e for e in events if e["type"] == "attempt"]
    assert [a["tile"] for a in attempts] == ["GD", "RD", "9B"] and all(a["success"] for a in attempts)
    landed = next(e for e in events if e["type"] == "return_landed")
    assert landed["error_m"] < 0.003                    # the drawn tile set down on H12's home
    assert sum(e["type"] == "discard" for e in events) == 2
    assert not interventions_of(p)
    assert len(ask.prompts) == 1                        # only the end-of-hand check
    assert json.loads((p.session.dir / "checkpoint.json").read_text())["phase"] == "won"
    assert p.session.dir.name.endswith("_play")


def test_an_unsure_drawn_tile_is_asked_not_guessed(tmp_path, monkeypatch):
    ask = Answers("9B")
    p = player(tmp_path, monkeypatch, ask)
    mat, detect = p.mat, p.mat.detect_tiles

    def split_read(n_frames=None):          # the 2nd draw reads 9B and 8B almost equally
        tiles = detect(n_frames)
        for t in tiles:
            if t["tile_name"] == "9B":
                t["votes"] = {"8B": 2.3, "9B": 2.0}
                t["tile_name"] = "8B"
        return tiles
    monkeypatch.setattr(mat, "detect_tiles", split_read)
    assert run(p) == "attended"
    asked = [i for i in interventions_of(p) if i["kind"] == "name_asked"]
    assert len(asked) == 1 and asked[0]["slot"] == "zone" and asked[0]["answer"] == "9B"
    assert p.state["hand"]["H12"] == "9B"


def test_a_hand_tile_that_reads_differently_goes_to_a_recount(tmp_path, monkeypatch):
    ask = Answers("NW")     # the person: H13 really is the NW the robot has
    p = player(tmp_path, monkeypatch, ask)
    mat, detect = p.mat, p.mat.detect_tiles
    calls = {"n": 0}

    def misread_after_registering(n_frames=None):
        calls["n"] += 1
        tiles = detect(n_frames)
        if calls["n"] > 3:                  # after the 3 registration scans, NW reads as WW, sure of it
            for t in tiles:
                if t["tile_name"] == "NW" and abs(t["xy"][0] - 0.21) < 0.01:
                    t["tile_name"], t["votes"] = "WW", {"WW": 4.5}
        return tiles
    monkeypatch.setattr(mat, "detect_tiles", misread_after_registering)
    p.draws = ["GD"]
    assert run(p) == "attended"
    recounts = [i for i in interventions_of(p) if i["kind"] == "recount"]
    assert recounts and recounts[0]["slot"] == "H13" and recounts[0]["expected"] == "NW"
    assert p.state["hand"]["H13"] == "NW"


def test_the_end_of_hand_check_counts_silent_errors(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers("h3=4C"))
    p.draws = ["GD"]
    assert run(p) == "silent error"
    end = next(e for e in events_of(p) if e["type"] == "hand_end")
    assert (end["result"], end["silent_errors"]) == ("end", 1)
    error = next(i for i in interventions_of(p) if i["kind"] == "silent_error")
    assert (error["slot"], error["robot"], error["table"]) == ("H3", "3C", "4C")


def test_a_hand_resumes_from_its_checkpoint(tmp_path, monkeypatch):
    first = player(tmp_path, monkeypatch, Answers())
    first.draws = ["GD"]                    # one turn, then the wall "runs out": interrupt instead
    first.bench.arm.rest()
    try:
        first.register()
        assert first.turn() is None
    finally:
        first.close()
    folder = first.session.dir
    assert json.loads((folder / "checkpoint.json").read_text())["turn"] == 1
    second = player(tmp_path, monkeypatch, Answers(), argv=("--sim", "--resume", str(folder)))
    second.draws = ["9B", "NW"]
    second.resume(folder)
    assert run(second) == "clean"
    assert second.state["turn"] == 2 and second.state["discards"] == ["GD", "RD"]
    assert any(e["type"] == "play_resumed" for e in events_of(second))


def test_layout_is_required_on_hardware():
    import play
    with pytest.raises(SystemExit):
        play.parse_args([])


def test_the_report_shows_the_hand(tmp_path, monkeypatch, capsys):
    import session_report
    from mahjong import interventions
    p = player(tmp_path, monkeypatch, Answers())
    run(p)
    events = events_of(p)
    capsys.readouterr()
    session_report.play_report(events, interventions.derive(events))
    out = capsys.readouterr().out
    assert "hand: win after 2 turns, clean" in out and "discards GD RD" in out and "turn time" in out


def test_a_pickup_marked_during_play_is_asked_about_at_the_end_of_the_step(tmp_path, monkeypatch):
    """Play's pickups don't ask as they go (bench.pickup(ask=False) queues marked or stopped ones); play
    never asked about the queue, so a → during play went unrecorded."""
    import builtins
    prompts = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": prompts.append(prompt) or "1")   # 1: touched the tile
    p = player(tmp_path, monkeypatch, Answers())
    run_policy, marked = p.bench.policy.run, []

    def mark_the_first(*a, **kw):
        if not marked:
            marked.append(True)
            p.bench.events["exit_early"] = True        # → as the policy starts
        return run_policy(*a, **kw)
    monkeypatch.setattr(p.bench.policy, "run", mark_the_first)
    run(p)
    (review,) = [e for e in events_of(p) if e["type"] == "attempt_review"]
    assert review["mark_kind"] == "touched_tile"
    assert any("what happened" in q for q in prompts)
