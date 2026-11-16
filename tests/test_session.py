"""Setup questions: structured answers, previous session's answers as defaults, no folder until answered."""
import builtins
from contextlib import nullcontext

import pytest

from mahjong import config
from mahjong.session import Session, ask_conditions, ask_one, previous_conditions, read_events


def answering(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_first_session_asks_each_question_and_takes_numbers_names_or_enter():
    # daylight "2" -> dim, lamp "y", ceiling Enter -> n, moved Enter -> nothing, tiles "ro" -> rows, battery "1",
    # surface "2" -> green cutting mat
    got = ask_conditions({}, ask=answering("2", "y", "", "", "ro", "1", "2"))
    assert got == {"daylight": "dim", "lamp": True, "ceiling_lights": False, "moved": "nothing",
                   "tiles": "rows", "battery": "full", "surface": "green cutting mat"}


def test_unclear_answers_are_asked_again():
    got = ask_conditions({}, ask=answering("9", "b", "maybe", "n", "", "", "", "", ""))
    assert got["daylight"] == "bright" and got["lamp"] is False


def test_enter_keeps_the_previous_setup_but_not_what_moved():
    previous = {"daylight": "none", "lamp": True, "ceiling_lights": True, "moved": "mat", "tiles": "one",
                "battery": "low", "surface": "paper"}
    got = ask_conditions(previous, ask=answering(""))
    assert got == {**previous, "moved": "nothing"}


def test_change_walks_through_with_previous_answers_as_defaults():
    previous = {"daylight": "none", "lamp": True, "ceiling_lights": True, "moved": "nothing", "tiles": "one",
                "battery": "low", "surface": "black table top"}
    got = ask_conditions(previous, ask=answering("c", "1", "", "", "3", "", "", ""))
    assert got == {**previous, "daylight": "bright", "moved": "mat"}


def test_session_stores_conditions_and_the_next_session_defaults_to_them(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(builtins, "input", answering("3", "n", "y", "", "", "2", "", "dusk"))
    s = Session("bench")
    s.close()
    start = read_events(s.dir)[0]
    assert start["conditions"] == {"daylight": "none", "lamp": False, "ceiling_lights": True, "moved": "nothing",
                                   "tiles": "one", "battery": "mid", "surface": "black table top"}
    assert start["notes"] == "dusk"
    assert s.conditions == start["conditions"] and s.notes == "dusk"        # record.py copies them per episode
    assert previous_conditions() == start["conditions"]


def test_quitting_during_the_questions_leaves_no_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS", tmp_path)

    def interrupt(prompt):
        raise KeyboardInterrupt
    monkeypatch.setattr(builtins, "input", interrupt)
    with pytest.raises(KeyboardInterrupt):
        Session("bench")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("raise_, reason", [(None, "quit"), (ZeroDivisionError, "ZeroDivisionError"),
                                             (SystemExit, "quit")])
def test_session_end_records_why_it_ended(tmp_path, monkeypatch, raise_, reason):
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    s = Session("bench", ask_setup=False)
    with pytest.raises(raise_) if raise_ else nullcontext():
        try:
            if raise_:
                raise raise_()
        finally:
            s.close()
    end = read_events(s.dir)[-1]
    assert end["type"] == "session_end" and end["reason"] == reason
    assert ("error" in end) == (reason != "quit")


def test_interventions_are_events(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    s = Session("bench", ask_setup=False)
    s.intervention("discard_missed", attempt_dir="002_3D", note="bounced off the rim")
    s.close()
    event = read_events(s.dir)[1]
    assert (event["type"], event["kind"], event["attempt_dir"], event["note"]) == \
        ("intervention", "discard_missed", "002_3D", "bounced off the rim")


def test_green_paper_is_a_surface_choice():
    surface = next(q for q in config.SETUP_QUESTIONS if q["key"] == "surface")
    assert "green paper" in surface["choices"] and surface["choices"][0] == "black table top"


def test_a_new_surface_typed_at_the_prompt_is_kept_for_later_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADDED_CHOICES", tmp_path / "added_choices.json")
    surface = next(q for q in config.SETUP_QUESTIONS if q["key"] == "surface")
    assert ask_one(surface, None, answering("blue felt", "y")) == "blue felt"
    again = ask_one(surface, None, answering("blue"))                       # a prefix of the saved one now works
    assert again == "blue felt"
    n = len(surface["choices"])
    assert ask_one(surface, None, answering(str(n + 1))) == "blue felt"     # and it has the next number


def test_declining_to_add_asks_again_and_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADDED_CHOICES", tmp_path / "added_choices.json")
    surface = next(q for q in config.SETUP_QUESTIONS if q["key"] == "surface")
    assert ask_one(surface, None, answering("blue felt", "n", "2")) == "green cutting mat"
    assert not (tmp_path / "added_choices.json").exists()


def test_a_question_that_isnt_addable_never_offers_to_add(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADDED_CHOICES", tmp_path / "added_choices.json")
    daylight = next(q for q in config.SETUP_QUESTIONS if q["key"] == "daylight")
    assert ask_one(daylight, None, answering("blue felt", "2")) == "dim"
