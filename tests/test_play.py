"""Autonomous play end to end on simulated hardware: a whole hand, and the questions that keep a wrong tile
from going unnoticed."""
import json
import sys
from pathlib import Path

import pytest

from mahjong import config, markers
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
    assert len(ask.prompts) == 2                        # only the hand check before the first draw and the one at the end
    assert "Check it against the table" in ask.prompts[0]
    assert json.loads((p.session.dir / "checkpoint.json").read_text())["phase"] == "won"
    assert p.session.dir.name.endswith("_play")


def test_every_turn_says_which_tiles_it_is_after(tmp_path, monkeypatch, capsys):
    p = player(tmp_path, monkeypatch, Answers())
    assert run(p) == "clean"
    out = capsys.readouterr().out
    assert "=== Turn 1 ===" in out and "Wins on: " in out
    chosen = [e for e in events_of(p) if e["type"] == "discard_chosen"]
    assert chosen and all(e["seeking"] for e in chosen)
    for e in chosen:                                    # the log's list and the choice's count are the same tiles
        assert (len(e["seeking"]), sum(left for _, left in e["seeking"])) == (e["ukeire_kinds"], e["ukeire_tiles"])


def test_an_unsure_drawn_tile_is_asked_not_guessed(tmp_path, monkeypatch):
    ask = Answers("", "9B")
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
    ask = Answers("", "NW")     # the person: the hand looks right; then H13 really is the NW the robot has
    p = player(tmp_path, monkeypatch, ask)
    mat, detect = p.mat, p.mat.detect_tiles
    calls = {"n": 0}

    def misread_after_registering(n_frames=None):
        calls["n"] += 1
        tiles = detect(n_frames)
        if calls["n"] > 3:                  # after the 3 registration scans, NW has moved and reads as WW, sure of it
            for t in tiles:
                if t["tile_name"] == "NW" and abs(t["xy"][0] - 0.21) < 0.01:
                    t["tile_name"], t["votes"] = "WW", {"WW": 4.5}
                    t["xy"] = [t["xy"][0], t["xy"][1] + 0.006]      # moved, or the hand check's word would hold
        return tiles
    monkeypatch.setattr(mat, "detect_tiles", misread_after_registering)
    p.draws = ["GD"]
    assert run(p) == "rescued"              # a tile that moved with the arm away: the scans settle it as a person's touch
    recounts = [i for i in interventions_of(p) if i["kind"] == "recount"]
    assert len(recounts) == 1 and recounts[0]["slot"] == "H13" and recounts[0]["expected"] == "NW"
    assert p.state["hand"]["H13"] == "NW"


def test_the_end_of_hand_check_counts_silent_errors(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers("", "h3=4C"))
    p.draws = ["GD"]
    assert run(p) == "silent error"
    end = next(e for e in events_of(p) if e["type"] == "hand_end")
    assert (end["result"], end["silent_errors"]) == ("end", 1)
    error = next(i for i in interventions_of(p) if i["kind"] == "silent_error")
    assert (error["slot"], error["robot"], error["table"]) == ("H3", "3D", "4C")


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
    # its own sessions folder: session folders are named to the second, and the first hand can end within it
    second = player(tmp_path / "second", monkeypatch, Answers(), argv=("--sim", "--resume", str(folder)))
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


def test_a_name_other_tiles_get_misread_as_is_asked_about_even_when_the_votes_are_sure(tmp_path, monkeypatch):
    import play
    monkeypatch.setattr(play, "SIM_HAND", ["7C", *play.SIM_HAND[1:]])
    p = player(tmp_path, monkeypatch, Answers("8C"))     # no reader in --sim: no look, so the person's asked
    assert run(p) == "attended"
    asked = [i for i in interventions_of(p) if i["kind"] == "name_asked"]
    assert [(a["slot"], a["answer"], a["leading_read"]) for a in asked] == [("H1", "8C", "7C")]
    assert "8C, 6C and 3C have been misread" in asked[0]["why"]


def test_a_low_confidence_read_is_asked_about(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers("1D"))
    detect = p.mat.detect_tiles

    def faint_first_tile(n_frames=None):
        return [{**t, "confidence": 0.6} if t["tile_name"] == "1D" else t for t in detect(n_frames)]
    p.mat.detect_tiles = faint_first_tile
    assert run(p) == "attended"
    asked = [i for i in interventions_of(p) if i["kind"] == "name_asked"]
    assert [a["slot"] for a in asked] == ["H1"] and "0.60 confidence" in asked[0]["why"]


# --- the hand check before the first draw, corrections, the window ---

def test_a_wrong_name_the_votes_were_sure_of_can_be_corrected_before_the_first_draw(tmp_path, monkeypatch):
    """The real case: a 5C read as 9C (S9, 20261104_164541_play), accepted on the votes, and no way to say so."""
    ask = Answers("h1=9C h2=9C", "")
    p = player(tmp_path, monkeypatch, ask)
    p.draws = []
    run(p)
    assert (p.state["hand"]["H1"], p.state["hand"]["H2"]) == ("9C", "9C")
    fixed = [i for i in interventions_of(p) if i["kind"] == "recount"]
    assert [(i["slot"], i["expected"], i["answer"]) for i in fixed] == [("H1", "1D", "9C"), ("H2", "2D", "9C")]
    confirmed = next(e for e in events_of(p) if e["type"] == "hand_confirmed")["confirmed"]
    assert confirmed[fixed[0]["placement"]] == "9C" and len(confirmed) == 13
    # asked again with the corrections showing, before the end-of-hand check
    assert sum("Check it against the table" in q for q in ask.prompts) == 3


def test_the_hand_check_comes_before_the_draw(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    order = []
    original = p.wait_for_draw
    monkeypatch.setattr(p, "wait_for_draw", lambda: order.append(p.state.get("hand_confirmed")) or original())
    run(p)
    assert order[0] is True


def test_a_clean_check_doesn_t_cost_the_hand_its_grade(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    assert run(p) == "clean"
    assert any(e["type"] == "hand_confirmed" for e in events_of(p))


@pytest.mark.parametrize("bad", ["9C", "H99=9C", "H1=XX"])
def test_a_bad_correction_is_asked_again_and_changes_nothing(tmp_path, monkeypatch, bad):
    ask = Answers(bad, "")
    p = player(tmp_path, monkeypatch, ask)
    p.draws = []
    assert run(p) == "clean"
    assert p.state["hand"]["H1"] == "1D"
    assert sum("Check it against the table" in q for q in ask.prompts) == 3


def test_a_correction_that_makes_a_fifth_copy_goes_to_a_recount(tmp_path, monkeypatch):
    ask = Answers("H1=SW H2=SW", "", "1D")      # 5 SW: H1 is really the 1D
    p = player(tmp_path, monkeypatch, ask)
    p.draws = []
    run(p)
    assert p.state["hand"]["H1"] == "1D"


def test_a_resumed_hand_that_never_had_its_check_gets_it(tmp_path, monkeypatch):
    first = player(tmp_path, monkeypatch, Answers())
    first.bench.arm.rest()
    try:
        first.register()                        # what an older session's checkpoint looks like
    finally:
        first.close()
    folder = first.session.dir
    ask = Answers("H1=9C", "")
    second = player(tmp_path / "second", monkeypatch, ask, argv=("--sim", "--resume", str(folder)))
    second.draws = []
    second.resume(folder)
    run(second)
    assert second.state["hand"]["H1"] == "9C"


def test_a_hand_tile_can_be_corrected_at_the_draw_prompt_and_the_discard_uses_it(tmp_path, monkeypatch):
    class AtDraw(Answers):
        def __call__(self, prompt):
            if "drawn tile" in prompt and "=" not in (self.answers[0] if self.answers else ""):
                p.mat.tiles.append({"tile_name": "GD", "xy": list(p.zone["xy"])})
            return super().__call__(prompt)
    ask = AtDraw("", "h13=9C", "", "end")
    p = player(tmp_path, monkeypatch, ask)
    p.args.sim = False                          # the real draw prompt
    p.bench.arm.rest()
    p.register()
    p.confirm_hand()
    chosen = []
    choose = __import__("mahjong.game").game.choose_discard
    monkeypatch.setattr("mahjong.game.choose_discard", lambda hand, drawn, seen: chosen.append(list(hand)) or choose(hand, drawn, seen))
    assert p.turn() is None
    p.close()
    assert chosen and "9C" in chosen[0] and "NW" not in chosen[0]


def test_a_stop_after_a_turn_finds_the_turn_already_saved(tmp_path, monkeypatch):
    """--resume reads the checkpoint: a turn not saved would be played again on a tile already moved."""
    p = player(tmp_path, monkeypatch, Answers())
    p.bench.arm.rest()
    p.register()
    p.state.update(phase="place", vacated="H12", discard_slot="H12", discard={"tile_name": "RD", "shanten_after": 1}, drawn=None)
    p.state["discards"].append("RD")
    p.save()

    def stopped():
        raise KeyboardInterrupt
    monkeypatch.setattr(p, "end_step", stopped)
    with pytest.raises(KeyboardInterrupt):
        p.finish_turn("RD", "H12")
    p.close()
    saved = json.loads((p.session.dir / "checkpoint.json").read_text())
    assert (saved["phase"], saved["turn"]) == ("draw", 1) and "vacated" not in saved


# --- the drop zone: a saved position, and a tile on it before the turn goes on ---

def no_marker(monkeypatch):
    """Anything that reads the marker fails the test."""
    def read(*_, **__):
        raise AssertionError("read the marker")
    monkeypatch.setattr(markers, "locate", read)
    monkeypatch.setattr(markers, "find", read)


def test_the_last_zone_carries_over_without_reading_the_marker(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    monkeypatch.setattr(markers, "ZONE_FILE", tmp_path / "drop_zone.json")
    saved = markers.save_zone([0.2, -0.17], session="last")
    no_marker(monkeypatch)
    assert p.load_zone(teach=False) == saved
    p.close()


def test_teach_zone_reads_the_marker_and_the_next_hand_gets_the_new_position(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    monkeypatch.setattr(markers, "ZONE_FILE", tmp_path / "drop_zone.json")
    markers.save_zone([0.2, -0.17], session="last")
    monkeypatch.setattr(markers, "locate", lambda realsense: [0.21, -0.15])
    assert p.load_zone(teach=True)["xy"] == [0.21, -0.15]
    no_marker(monkeypatch)
    assert p.load_zone(teach=False)["xy"] == [0.21, -0.15]
    p.close()


def at_the_real_draw_prompt(p):
    """The hand registered and checked, with a person putting each drawn tile on the zone rather than the sim."""
    p.args.sim = False
    p.bench.arm.rest()
    p.register()
    p.confirm_hand()


def test_a_turn_waits_for_a_tile_on_the_zone_and_never_reads_the_marker(tmp_path, monkeypatch):
    class PutsItOnTheSecondAsk(Answers):
        def __call__(self, prompt):
            if "No tile seen" in prompt:
                p.mat.tiles.append({"tile_name": "GD", "xy": list(p.zone["xy"])})
            return super().__call__(prompt)
    ask = PutsItOnTheSecondAsk()
    p = player(tmp_path, monkeypatch, ask)
    no_marker(monkeypatch)
    at_the_real_draw_prompt(p)
    assert p.turn() is None                     # the whole turn, finish_turn included
    p.close()
    asked = [prompt for prompt in ask.prompts if "drop zone" in prompt]
    assert "Put the drawn tile on the drop zone" in asked[0] and "No tile seen on the drop zone" in asked[1]
    assert f"{p.zone['xy'][0]:.3f}" in asked[1] and "marker" not in asked[1]
    assert p.state["discards"] == ["GD"] and sum(e.get("kind") == "zone_asked" for e in events_of(p)) == 1


def test_a_tile_too_far_from_the_zone_is_not_on_it(tmp_path, monkeypatch):
    class Misplaces(Answers):
        def __call__(self, prompt):
            x, y = p.zone["xy"]
            if "Put the drawn tile" in prompt:
                p.mat.tiles.append({"tile_name": "GD", "xy": [x, y + 0.03]})    # 30mm off
            elif "No tile seen" in prompt:
                p.mat.tiles[-1]["xy"] = [x, y]
            return super().__call__(prompt)
    ask = Misplaces()
    p = player(tmp_path, monkeypatch, ask)
    at_the_real_draw_prompt(p)
    assert p.turn() is None
    p.close()
    assert any("No tile seen" in prompt for prompt in ask.prompts) and p.state["discards"] == ["GD"]


def test_a_tile_the_detector_can_only_box_is_there_and_is_asked_about_never_named_from_the_box(tmp_path, monkeypatch):
    """3C reads at a median 0.18: "no tile seen" would never end, and the box's guess isn't a name to go on."""
    class FaintDraw(Answers):
        def __call__(self, prompt):
            assert "No tile seen" not in prompt     # asked again, it would ask for ever
            if "Put the drawn tile" in prompt:
                p.mat.faint = [{"tile_name": "9B", "confidence": 0.18, "xy": tuple(p.zone["xy"])}]
            return super().__call__(prompt)
    ask = FaintDraw("", "", "3C")               # the hand check, the draw prompt, then the question about the tile
    p = player(tmp_path, monkeypatch, ask)
    at_the_real_draw_prompt(p)
    assert p.wait_for_draw() == "3C"
    question = ask.prompts[-1]
    assert "can't name it" in question and "[Enter" not in question
    named = next(e for e in events_of(p) if e["type"] == "name_decided" and e["slot"] == "zone")
    assert (named["name"], named["how"]) == ("3C", "person")
    p.state.update(phase="place", vacated="H12", drawn="3C")    # and it can be picked up from the box
    assert p.pick("zone", "3C") is True
    p.close()
    attempt = next(e for e in events_of(p) if e["type"] == "attempt")
    assert (attempt["tile"], attempt["name_source"]) == ("3C", "faint")


def drawn_reading_differently(p, phase, drawn="GD", now_reads="9B"):
    """A settled drawn tile, on the zone, whose votes have since moved to another name."""
    p.bench.arm.rest()
    p.register()
    p.mat.tiles.append({"tile_name": now_reads, "xy": list(p.zone["xy"])})
    p.state.update(phase=phase, drawn=drawn, vacated="H12")
    p.votes.known("zone", now_reads)


def test_a_drawn_tile_corrected_before_its_discard_is_chosen_from_under_its_new_name(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers("9B"))
    drawn_reading_differently(p, "discard")
    assert p.pick("zone", "GD") is False                # it's another tile: the caller chooses again
    p.close()
    assert p.state["drawn"] == "9B"


def test_a_drawn_tile_corrected_on_its_way_to_a_slot_carries_on_under_its_new_name(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers("9B"))
    drawn_reading_differently(p, "place")
    assert p.pick("zone", "GD") is True
    p.close()
    assert p.state["drawn"] == "9B"


def over_counted(tmp_path, monkeypatch, ask, discards, drawn=None, hand=None):
    """A registered hand holding too many of a tile. A check_counts that stops asking would loop for ever: fail instead."""
    p = player(tmp_path, monkeypatch, ask)
    p.bench.arm.rest()
    p.register()
    p.state.update(discards=discards, drawn=drawn)
    p.state["hand"].update(hand or {})
    if drawn:
        p.votes.known("zone", drawn)
    recount, calls = p.recount, []

    def counted(*args, **kwargs):
        calls.append(args)
        assert len(calls) < 5, "check_counts is going round for ever"
        return recount(*args, **kwargs)
    monkeypatch.setattr(p, "recount", counted)
    return p


def test_five_of_a_tile_none_of_them_in_the_hand_asks_about_the_drawn_one(tmp_path, monkeypatch):
    p = over_counted(tmp_path, monkeypatch, Answers("9B"), ["GD"] * 4, drawn="GD")
    p.check_counts()
    p.close()
    assert p.state["drawn"] == "9B"


def test_too_many_of_a_tile_can_be_settled_by_correcting_the_drawn_one(tmp_path, monkeypatch):
    p = over_counted(tmp_path, monkeypatch, Answers("", "9B"), ["GD"] * 3, drawn="GD", hand={"H1": "GD"})
    p.check_counts()
    p.close()
    assert (p.state["hand"]["H1"], p.state["drawn"]) == ("GD", "9B")
    assert not [i for i in interventions_of(p) if i["kind"] == "count_unresolved"]


def test_a_person_who_keeps_every_name_ends_the_count_check_and_isnt_asked_again(tmp_path, monkeypatch):
    # the misread one is a discard, already in the bin: nothing a person types at the hand can fix it
    ask = Answers()
    p = over_counted(tmp_path, monkeypatch, ask, ["GD"] * 4, hand={"H1": "GD"})
    p.check_counts()
    asked = len(ask.prompts)
    p.check_counts()
    p.close()
    assert len(ask.prompts) == asked == 1
    assert [i["names"] for i in interventions_of(p) if i["kind"] == "count_unresolved"] == [["GD"]]


def test_a_turn_goes_on_the_drawn_name_the_count_check_left(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    p.bench.arm.rest()
    p.register()
    monkeypatch.setattr(p, "check_counts", lambda: p.state.update(drawn="NW"))
    chosen = []
    choose = __import__("mahjong.game").game.choose_discard
    monkeypatch.setattr("mahjong.game.choose_discard", lambda hand, drawn, seen: chosen.append(drawn) or choose(hand, drawn, seen))
    p.turn()
    p.close()
    assert chosen[0] == "NW"


# --- a person's word holds until the tile moves ---

def reads_nw_as_ww(p, monkeypatch, moved_m=0.0, after_scans=3):
    """After the registration scans, H13's NW reads as a sure WW, `moved_m` from where it was."""
    detect, calls = p.mat.detect_tiles, {"n": 0}

    def misread(n_frames=None):
        calls["n"] += 1
        tiles = [dict(t) for t in detect(n_frames)]
        if calls["n"] > after_scans:
            for t in tiles:
                if t["tile_name"] == "NW" and abs(t["xy"][0] - 0.21) < 0.01:
                    t.update(tile_name="WW", votes={"WW": 4.5}, xy=[t["xy"][0], t["xy"][1] + moved_m])
        return tiles
    monkeypatch.setattr(p.mat, "detect_tiles", misread)


def recounts_of(p):
    return [i for i in interventions_of(p) if i["kind"] == "recount"]


def test_the_hand_checks_enter_holds_against_a_steady_misread(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    reads_nw_as_ww(p, monkeypatch)
    p.draws = ["GD", "9B"]                  # both leave H13 in the hand
    run(p)
    held = [e for e in events_of(p) if e["type"] == "name_held"]
    assert not recounts_of(p) and held and all(e["slot"] == "H13" and e["name"] == "NW" for e in held)
    assert p.state["hand"]["H13"] == "NW"


def test_a_typed_correction_holds_every_turn(tmp_path, monkeypatch):
    ask = Answers("H13=8D")     # registered as NW; it's really an 8D the scans keep reading as NW
    p = player(tmp_path, monkeypatch, ask)
    p.draws = ["GD", "9B", "1D"]
    run(p)
    fixed = recounts_of(p)
    assert [(i["slot"], i["answer"]) for i in fixed] == [("H13", "8D")]        # the correction itself, and nothing after it


def test_a_misread_tile_that_moved_is_asked_about_once(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    reads_nw_as_ww(p, monkeypatch, moved_m=0.006)
    p.draws = ["GD", "9B", "1D"]
    run(p)
    assert [(i["slot"], i["expected"], i["answer"]) for i in recounts_of(p)] == [("H13", "NW", "NW")]


def test_a_tile_that_moved_under_the_knock_distance_still_holds(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    reads_nw_as_ww(p, monkeypatch, moved_m=0.004)
    p.draws = ["GD"]
    run(p)
    assert not recounts_of(p)


def test_a_tile_gone_from_its_slot_is_asked_about_even_with_a_word_on_it(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    p.bench.arm.rest()
    p.register()
    p.confirm_hand()
    p.latest.pop("H13")
    assert p.word_holds("H13", "NW") is False


def test_a_word_doesn_t_cover_a_different_name(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    p.bench.arm.rest()
    p.register()
    p.confirm_hand()
    assert p.word_holds("H13", "NW") is True and p.word_holds("H13", "WW") is False


def test_a_word_ends_with_its_placement(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    p.bench.arm.rest()
    p.register()
    p.confirm_hand()
    p.placements.end("H13", "discarded")
    p.placements.start("H13", p.home("H13"), truth_source="game", read="NW")
    assert p.word_holds("H13", "NW") is False


def test_the_window_treats_the_hand_checks_enter_as_your_word(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    p.bench.arm.rest()
    p.register()
    p.confirm_hand()
    p.votes.add("H1", {"tile_name": "2C", "confidence": 0.9, "votes": {"2C": 9.0}})
    found = p.attention("H1")
    assert "you said 1D" in found["why"] and found["could_be"] == ["2C"]
    p.close()


def test_a_word_on_the_drawn_tile_goes_with_it_to_its_slot(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    p.bench.arm.rest()
    p.register()
    p.confirm_hand()
    p.draws = ["9B"]
    p.state["drawn"] = p.wait_for_draw()
    p.give_word("zone", "9B")
    p.placements.end("H12", "discarded")
    p.mat.tiles = [t for t in p.mat.tiles if t["tile_name"] != "RD"]
    assert p.place_drawn("H12", "9B") is True
    word = p.words[p.placements.id_at("H12")]
    assert word == {"name": "9B", "xy": None}
    p.scan("after")
    assert p.word_holds("H12", "9B") is True and word["xy"] is not None
    p.close()


def test_a_recount_during_the_turn_changes_the_hand_the_discard_is_chosen_from(tmp_path, monkeypatch):
    """hand13 used to be taken before check_hand, so a person's answer there didn't reach the discard or the win check."""
    ask = Answers("", "9C")
    p = player(tmp_path, monkeypatch, ask)
    detect = p.mat.detect_tiles
    calls = {"n": 0}

    def misread_after_registering(n_frames=None):
        calls["n"] += 1
        tiles = detect(n_frames)
        if calls["n"] > 3:
            for t in tiles:
                if t["tile_name"] == "NW" and abs(t["xy"][0] - 0.21) < 0.01:
                    t["tile_name"], t["votes"] = "WW", {"WW": 4.5}
                    t["xy"] = [t["xy"][0], t["xy"][1] + 0.006]      # moved, or the hand check's word would hold
        return tiles
    monkeypatch.setattr(p.mat, "detect_tiles", misread_after_registering)
    p.draws = ["GD"]
    chosen = []
    choose = __import__("mahjong.game").game.choose_discard
    monkeypatch.setattr("mahjong.game.choose_discard", lambda hand, drawn, seen: chosen.append(list(hand)) or choose(hand, drawn, seen))
    run(p)
    assert "9C" in chosen[0] and "NW" not in chosen[0]


class Shown:
    """The window: pictures kept instead of shown."""
    def __init__(self):
        self.pictures, self.closed = [], False

    def show(self, image):
        self.pictures.append(image)

    def close(self):
        self.closed = True


def test_every_question_goes_on_screen_and_a_hand_doesn_t_need_a_window(tmp_path, monkeypatch):
    import play
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    shown, ask = Shown(), Answers()
    p = play.Player(play.parse_args(["--sim"]), ask=ask, view=shown)
    assert run(p) == "clean"
    assert len(shown.pictures) == len(ask.prompts) and shown.closed
    assert shown.pictures[0].ndim == 3


def test_the_picture_has_the_wrist_view_beside_the_mat_when_the_arm_has_looked(tmp_path, monkeypatch):
    import numpy as np
    from mahjong import hand_view, sim
    mat = sim.SimPerception()
    tiles = [{"id": "S1", "xy": [0.15, 0.0], "name": "5C"}, {"id": "zone", "xy": [0.25, 0.0], "name": None}]
    overview = mat.snapshot()
    alone = hand_view.compose(overview, tiles, "x " * 200, mat.intrinsics, mat.T_cam_to_base, focus="S1")
    wrist = np.zeros((480, 640, 3), np.uint8)
    beside = hand_view.compose(overview, tiles, "short", mat.intrinsics, mat.T_cam_to_base, focus="S1", wrist=wrist)
    assert alone.shape[1] <= hand_view.PICTURE_FIT_PX[0] and beside.shape[1] == alone.shape[1] + hand_view.PANEL_PX
    long, short = (hand_view.compose(overview, tiles, c, mat.intrinsics, mat.T_cam_to_base) for c in ("x " * 200, "short"))
    assert long.shape[0] > short.shape[0]           # a long caption wraps onto more lines


def test_the_picture_zooms_in_on_the_tiles_and_keeps_every_one_in_it():
    import numpy as np
    from mahjong import hand_view
    image = np.zeros((480, 848, 3), np.uint8)
    outlines = [np.array([[300.0, 300.0], [330.0, 330.0]]), np.array([[500.0, 340.0], [530.0, 370.0]])]
    picture, to_picture = hand_view.zoom_on(image, outlines)
    assert picture.shape[1] == 848                  # was 280 wide with its margins
    corners = to_picture(np.vstack(outlines))
    assert (corners >= 0).all() and (corners[:, 0] < picture.shape[1]).all() and (corners[:, 1] < picture.shape[0]).all()
    assert corners[1, 0] - corners[0, 0] > 90       # a 30px tile is at least 3x larger


def test_tiles_at_the_frame_s_edge_don_t_break_the_zoom():
    import numpy as np
    from mahjong import hand_view
    image = np.zeros((480, 848, 3), np.uint8)
    picture, to_picture = hand_view.zoom_on(image, [np.array([[0.0, 0.0], [847.0, 479.0]])])
    assert picture.shape[:2] == (480, 848) and to_picture(np.array([[0.0, 0.0]])).tolist() == [[0.0, 0.0]]


def test_the_window_process_gets_every_picture_and_leaves_when_the_pipe_closes():
    import io
    import struct
    import cv2
    import numpy as np
    from mahjong import hand_view
    image = np.full((20, 30, 3), 200, np.uint8)
    data = cv2.imencode(".png", image)[1].tobytes()
    got = []
    hand_view.read_pictures(io.BytesIO((struct.pack(">I", len(data)) + data) * 2), got.append)
    assert len(got) == 2 and (got[0] == image).all()


def test_a_window_that_died_doesn_t_stop_the_hand(capsys):
    import subprocess
    import sys
    import numpy as np
    from mahjong import hand_view
    window = hand_view.Window.__new__(hand_view.Window)
    window.process = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.PIPE)
    window.process.wait()
    big = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)     # more than the pipe holds
    window.show(big)
    window.show(big)
    window.close()
    assert "window is gone" in capsys.readouterr().out


def recording_compose(monkeypatch):
    """What each picture was asked to show: [(tiles, focus)]."""
    from mahjong import hand_view
    composed, compose = [], hand_view.compose

    def record(overview, tiles, *args, **kwargs):
        composed.append((tiles, kwargs.get("focus")))
        return compose(overview, tiles, *args, **kwargs)
    monkeypatch.setattr(hand_view, "compose", record)
    return composed


def test_the_hand_check_marks_a_tile_the_votes_were_sure_of_with_what_else_it_could_be(tmp_path, monkeypatch):
    """The real case: S9's 5C read as 9C at 0.91 of the votes, 20261104_164541_play. Nothing asked, and nothing pointed at it."""
    from mahjong import hand_view
    p = player(tmp_path, monkeypatch, Answers())
    p.view = Shown()
    p.bench.arm.rest()
    p.register()
    p.votes.totals["H1"]["5C"] = 1.8                    # H1's 1D has about 14 from the sim's 3 scans
    composed = recording_compose(monkeypatch)
    p.confirm_hand()
    p.close()
    tiles = {t["id"]: t for t in composed[-1][0]}
    assert tiles["H1"]["could_be"] == ["5C"] and "scans also read 5C" in tiles["H1"]["why"]
    assert not tiles["H2"].get("why")                   # the rest aren't marked
    assert hand_view.suspected(tiles["H1"]) == "1D or 5C" and hand_view.suspected(tiles["H2"]) == "2D"
    assert not [i for i in interventions_of(p) if i["kind"] in ("name_asked", "recount")]       # the marks ask nothing


def test_a_review_of_several_tiles_shows_them_all_from_the_first_question(tmp_path, monkeypatch):
    """The recount at each turn of 20261104_183142_play asked about S5 and S9 in turn, with only the one on screen."""
    p = player(tmp_path, monkeypatch, Answers("", ""))
    p.view = Shown()
    p.bench.arm.rest()
    p.register()
    p.set_name("H1", "1D", typed=False)
    p.votes.totals["H1"]["9C"] = 20.0
    p.votes.totals["H2"]["9B"] = 30.0
    composed = recording_compose(monkeypatch)
    p.recount(["H1", "H2"])
    p.close()
    (first, focus), (second, then) = composed
    assert (focus, then) == ("H1", "H2")
    marked = {t["id"]: t for t in first if t.get("why")}
    assert set(marked) == {"H1", "H2"} and marked["H1"]["could_be"] == ["9C"] and marked["H2"]["could_be"] == ["9B"]
    assert marked["H1"]["why"].startswith("you said 1D")    # a person's earlier answer is on screen
    assert {t["id"] for t in second if t.get("why")} == {"H1", "H2"}


def test_a_tile_a_person_has_checked_isn_t_marked_for_being_on_the_list(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    p.bench.arm.rest()
    p.register()
    p.state["hand"]["H1"] = "7C"
    assert "misread" in p.attention("H1")["why"]
    p.set_name("H1", "7C", typed=True)
    assert p.attention("H1") == {}
    p.close()


def test_the_wrist_camera_s_word_goes_with_the_mark(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    p.bench.arm.rest()
    p.register()
    p.state["hand"]["H1"] = "7C"
    p.wrist_seen["H1"] = {"frame": None, "read": {"name": "7C", "confidence": 0.97}}
    assert "wrist camera agrees (0.97)" in p.attention("H1")["why"]
    p.wrist_seen["H1"] = {"frame": None, "read": {"name": "8C", "confidence": 0.9}}
    found = p.attention("H1")
    assert found["why"].startswith("wrist camera reads 8C (0.90)") and found["could_be"][0] == "8C"
    p.wrist_seen["H2"] = {"frame": None, "read": {"name": "2D", "confidence": 0.97}}
    assert p.attention("H2") == {}                          # agreeing about a tile nothing's wrong with says nothing
    p.close()


def test_every_list_entry_names_the_tiles_misread_as_it():
    from mahjong import game, identity
    for read in identity.ALWAYS_ASK_READS:
        found = identity.misread_from(read)
        assert found and read not in found and all(game.is_playable(name) for name in found)
    assert identity.misread_from("2C") == ["3C"]            # "by detector v7x3 at up to 0.96" names no tile
    assert identity.misread_from("1D") == []


def test_only_a_review_dims_the_mat_and_its_list_says_why():
    import numpy as np
    from mahjong import hand_view, sim
    mat = sim.SimPerception()
    plain = [{"id": "S1", "xy": [0.15, 0.0], "name": "5C"}, {"id": "S2", "xy": [0.15, 0.06], "name": "1D"}]
    marked = [{**plain[0], "why": "scans also read 9C (9%)", "could_be": ["9C", "3C"]}, plain[1]]
    overview = np.full((480, 848, 3), 200, np.uint8)
    quiet = hand_view.compose(overview, plain, "q", mat.intrinsics, mat.T_cam_to_base)
    loud = hand_view.compose(overview, marked, "q", mat.intrinsics, mat.T_cam_to_base)
    assert quiet[-1, 0].max() == 200                        # nothing marked: no list, the mat as bright as it was
    assert loud.shape[0] > quiet.shape[0]                   # a list under the picture
    assert hand_view.attention_list(marked, focus="S1") == [("> S1: 5C or 9C, 3C. scans also read 9C (9%)", hand_view.FOCUS)]
    assert hand_view.suspected(marked[0]) == "5C or 9C"     # the picture has room for one; the list has them all
    assert hand_view.suspected(plain[0]) == "5C"


def test_a_detection_s_box_outlines_the_tile_and_a_tile_with_none_is_projected():
    from mahjong import hand_view, sim
    mat = sim.SimPerception()
    box = hand_view.outline_of({"box": (100, 200, 20, 30), "xy": [0.15, 0.0]}, mat.intrinsics, mat.T_cam_to_base)
    assert box.tolist() == [[90, 185], [110, 185], [110, 215], [90, 215]]
    assert hand_view.outline_of({"box": None, "xy": [0.15, 0.0]}, mat.intrinsics, mat.T_cam_to_base).shape == (4, 2)
