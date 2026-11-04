"""The wrist-camera name check: which photos it trains on and is scored on, and the rule play applies: agreement changes
nothing, disagreement asks a person, a read never names a tile by itself."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mahjong import config, identity, name_check, wrist_photos
from mahjong.session import read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_play import Answers, events_of, interventions_of, player, run


def jpg(path, value=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((480, 640, 3), value, np.uint8))


def make_sessions(root):
    """A wrist_capture session (person-named), a bench session with one reviewed photo and one not, an exam bench session."""
    wrist = root / "20261104_080000_wrist"
    for name, value in (("3C", 60), ("2C", 200)):
        for view in range(4):
            jpg(wrist / "wrist" / name / f"01_01_v{view:02d}_f0.jpg", value)
    events = [{"type": "wrist_view", "files": [f"wrist/3C/01_01_v{v:02d}_f0.jpg"], "detected_as": "2C"} for v in range(4)]
    (wrist / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    jpg(root / "20261104_090000_bench" / "001_090000_4B" / "wrist_0000.jpg")
    jpg(root / "20261104_090000_bench" / "002_090100_9B" / "wrist_0000.jpg")
    exam = root / "20261104_100000_bench"
    jpg(exam / "001_100000_2C" / "wrist_0000.jpg", 200)
    jpg(exam / "002_100100_2C" / "wrist_0000.jpg", 60)
    jpg(exam / "003_100200_5B" / "wrist_0000.jpg")
    events = [{"type": "placement", "id": "P1", "truth": "2C"}, {"type": "placement", "id": "P2", "truth": "3C"},
              {"type": "attempt", "tile": "2C", "placement": "P1", "attempt_dir": "001_100000_2C"},
              {"type": "attempt", "tile": "2C", "placement": "P2", "attempt_dir": "002_100100_2C"},
              {"type": "attempt", "tile": "5B", "placement": None, "attempt_dir": "003_100200_5B"}]
    (exam / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return {"sessions": {exam.name: {}}}


def test_the_crop_is_where_a_hovered_tile_sits_and_stays_in_frame():
    image = np.zeros((480, 640, 3), np.uint8)
    assert name_check.crop(image).shape == (name_check.INPUT_PX, name_check.INPUT_PX, 3)


def test_it_trains_only_on_names_a_person_gave_and_never_on_the_exam(tmp_path):
    exam = make_sessions(tmp_path)
    review = {"20261104_090000_bench/001_090000_4B/wrist_0000.jpg": {"verdict": "wrong", "name": "2B"}}
    photos = name_check.training_photos(tmp_path, review, exam)
    assert sorted({p["session"] for p in photos}) == ["20261104_080000_wrist", "20261104_090000_bench"]
    bench = [p for p in photos if p["source"] == "bench"]
    assert [p["name"] for p in bench] == ["2B"]              # the reviewed one, renamed; the unreviewed 9B is left out


def test_the_exam_takes_a_person_s_name_and_leaves_out_a_photo_with_none(tmp_path):
    exam = make_sessions(tmp_path)
    rows = name_check.exam_rows(tmp_path, exam, {})
    assert [(r["truth"], r["yolo"], r["group"]) for r in rows] == [("2C", "2C", "P1"), ("3C", "2C", "P2")]


def test_the_gate_is_a_wrong_name_both_agree_on():
    row = lambda truth, yolo, said: {"truth": truth, "yolo": yolo, "said": said, "group": truth + yolo + said}
    s = name_check.score([row("3C", "2C", "2C"), row("3C", "2C", "3C"), row("5B", "5B", "5B"), row("7B", "7B", "6B")])
    assert (s["accepted_wrong"], s["agreed"], s["asks"], s["yolo_wrong"], s["yolo_wrong_caught"]) == (1, 2, 2, 2, 1)
    assert s["right"] == 2 and 0.5 < s["accepted_wrong_bound"] <= 1.0


def tiny_model(n):
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, stride=4), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                               torch.nn.Linear(4, n))


def test_training_scripts_end_to_end_on_a_made_up_folder(tmp_path, monkeypatch, capsys):
    import name_check_eval
    import train_name_check
    sessions = tmp_path / "sessions"
    exam = make_sessions(sessions)
    (tmp_path / "exam.json").write_text(json.dumps(exam))
    monkeypatch.setattr(name_check, "EXAM", tmp_path / "exam.json")
    monkeypatch.setattr(name_check, "EVAL", tmp_path / "eval")
    monkeypatch.setattr(wrist_photos, "REVIEW", tmp_path / "none.jsonl")
    monkeypatch.setattr(name_check, "build_model", tiny_model)
    monkeypatch.setattr(name_check, "_device", lambda: "cpu")
    base = ["--sessions-dir", str(sessions)]
    assert train_name_check.main([*base, "--dry-run"]) is None
    assert "8 photos of 2 names" in capsys.readouterr().out
    weights = tmp_path / "w.pt"
    assert train_name_check.main([*base, "--epochs", "1", "--out", str(weights)]) == weights
    reader = name_check.Reader(weights)
    read = reader.read(np.zeros((480, 640, 3), np.uint8))
    assert read["name"] in ("2C", "3C") and 0.0 <= read["confidence"] <= 1.0
    s = name_check_eval.main([*base, "--weights", str(weights)])
    assert s["photos"] == 2 and s["yolo_wrong"] == 1
    assert list((tmp_path / "eval").glob("w@*.json"))
    assert name_check_eval.main([*base, "--list", "--weights", str(weights)]) is None


def test_weights_that_trained_on_the_exam_are_refused(tmp_path, monkeypatch):
    import name_check_eval
    sessions = tmp_path / "sessions"
    exam = make_sessions(sessions)
    (tmp_path / "exam.json").write_text(json.dumps(exam))
    monkeypatch.setattr(name_check, "EXAM", tmp_path / "exam.json")
    monkeypatch.setattr(wrist_photos, "REVIEW", tmp_path / "none.jsonl")
    weights = name_check.save({"classes": ["2C", "3C"], "model": tiny_model(2), "sessions": list(exam["sessions"])}, tmp_path / "w.pt")
    with pytest.raises(SystemExit, match="trained on exam sessions"):
        name_check_eval.main(["--sessions-dir", str(sessions), "--weights", str(weights)])


# --- play's rule, on a simulated hand ---

class Says:
    """A reader that echoes the name the robot's going on, except `wrong`: {that name: what it says instead}."""
    def __init__(self, p, wrong=None):
        self.p, self.wrong, self.reads = p, wrong or {}, 0

    def read(self, frame):
        self.reads += 1
        name = self.p.picking or self.hand_name()
        return {"name": self.wrong.get(name, name), "confidence": 0.9}

    def hand_name(self):
        return self.p.current_hand_name


def checks_of(p):
    return [e for e in events_of(p) if e["type"] == "name_check"]


def test_agreement_changes_nothing(tmp_path, monkeypatch):
    ask = Answers()
    p = player(tmp_path, monkeypatch, ask)
    p.reader = Says(p)
    assert run(p) == "clean"
    assert len(ask.prompts) == 2                         # only the hand checks, before the first draw and at the end
    checks = checks_of(p)
    assert [c["name"] for c in checks] == ["GD", "RD", "9B"] and all(c["agrees"] for c in checks)
    assert (p.session.dir / checks[0]["attempt_dir"] / "name_check.jpg").exists()


def test_without_weights_a_hand_plays_with_no_second_read(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    assert p.reader is None and run(p) == "clean" and not checks_of(p)


def test_a_disagreement_asks_and_the_person_s_word_keeps_the_pickup_going(tmp_path, monkeypatch):
    ask = Answers()                                       # Enter: the robot's name stands
    p = player(tmp_path, monkeypatch, ask)
    p.reader = Says(p, {"RD": "7B"})
    assert run(p) == "attended"
    recount = [i for i in interventions_of(p) if i["kind"] == "recount"]
    assert len(recount) == 1 and recount[0]["slot"] == "H12" and recount[0]["answer"] == "RD"
    assert p.state["discards"] == ["GD", "RD"]            # it went on, and the read named nothing
    assert any("H12" in prompt for prompt in ask.prompts)


def test_a_read_never_changes_a_name_without_a_person(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    p.reader = Says(p, {"GD": "1D", "RD": "2D", "9B": "3D"})
    run(p)
    assert "1D" not in p.state["discards"] and p.state["discards"] == ["GD", "RD"]
    assert p.state["hand"]["H12"] == "9B"


def test_a_corrected_discard_calls_off_the_pickup_and_chooses_again(tmp_path, monkeypatch):
    ask = Answers("", "GD")                               # the tile at H12 is a GD: the robot has the wrong name for it
    p = player(tmp_path, monkeypatch, ask)
    p.reader = Says(p, {"RD": "GD"})
    run(p)
    events = events_of(p)
    off = [e for e in events if e["type"] == "pickup_called_off"]
    assert len(off) == 1 and (off[0]["slot"], off[0]["name"], off[0]["now"]) == ("H12", "RD", "GD")
    attempts = [e for e in events if e["type"] == "attempt"]
    assert not any(a["attempt_dir"] == off[0]["attempt_dir"] for a in attempts)   # the policy never ran on it
    assert "RD" not in p.state["discards"]                # the discard was chosen again, from the corrected hand


def test_a_drawn_tile_corrected_on_its_way_to_the_hand_goes_in_under_the_person_s_name(tmp_path, monkeypatch):
    ask = Answers("", "8B")                               # the 9B on the zone is really an 8B
    p = player(tmp_path, monkeypatch, ask)
    p.reader = Says(p, {"9B": "8B"})
    run(p)
    asked = [i for i in interventions_of(p) if i["kind"] == "name_asked"]
    assert asked and asked[0]["slot"] == "zone" and asked[0]["answer"] == "8B"
    assert p.state["hand"]["H12"] == "8B"


def test_read_hand_hovers_over_every_tile_and_asks_about_a_disagreement(tmp_path, monkeypatch):
    ask = Answers("2D")                                   # H2 is what the robot has... the person types its real name
    p = player(tmp_path, monkeypatch, ask, argv=("--sim", "--read-hand"))
    reader = Says(p, {"2D": "9D"})
    p.reader = reader
    names = iter(["1D", "2D", "3D", "4D", "5D", "6D", "7B", "8B", "SW", "SW", "SW", "RD", "NW"])
    original = p.read_wrist

    def named(slot, name, folder, keep_as=None):
        p.current_hand_name = name
        return original(slot, name, folder, keep_as)
    monkeypatch.setattr(p, "read_wrist", named)
    p.register()
    p.close()
    checks = checks_of(p)
    assert len(checks) == 13 and [c["name"] for c in checks] == list(names)
    assert sum(not c["agrees"] for c in checks) == 1
    assert [i["slot"] for i in interventions_of(p) if i["kind"] == "recount"] == ["H2"]
    assert all((p.session.dir / c["attempt_dir"] / "wrist_0000.jpg").exists() for c in checks)     # photos for wrist_review


def test_the_ask_list_has_the_name_3c_gets_misread_as():
    assert "2C" in identity.ALWAYS_ASK_READS and "3C" in identity.ALWAYS_ASK_READS["2C"]
    votes = identity.Votes()
    votes.add("H1", {"tile_name": "2C", "confidence": 0.96, "votes": {"2C": 4.8}})
    assert "2C" in votes.doubt("H1", "2C")


# --- a read on the ask list gets the wrist camera's look before a person is asked ---

def hand_with(monkeypatch, *names):
    import play
    monkeypatch.setattr(play, "SIM_HAND", [*names, *play.SIM_HAND[len(names):]])


def reader_for(p, monkeypatch, wrong=None):
    """A reader that echoes the name being looked at, except `wrong`."""
    reader = Says(p, wrong)
    p.reader, original = reader, p.read_wrist

    def named(slot, name, folder, keep_as=None):
        p.current_hand_name = name
        return original(slot, name, folder, keep_as)
    monkeypatch.setattr(p, "read_wrist", named)
    return reader


def test_a_problem_read_the_wrist_agrees_with_waits_for_the_hand_check(tmp_path, monkeypatch):
    hand_with(monkeypatch, "7C")
    ask = Answers()
    p = player(tmp_path, monkeypatch, ask)
    reader = reader_for(p, monkeypatch)
    p.draws = []
    assert run(p) == "clean"
    assert not [i for i in interventions_of(p) if i["kind"] == "name_asked"]
    decided = next(e for e in events_of(p) if e["type"] == "name_decided" and e["slot"] == "H1")
    assert (decided["name"], decided["how"]) == ("7C", "wrist") and reader.reads == 1
    assert any(e["type"] == "hand_confirmed" for e in events_of(p))      # a person still saw it
    assert p.bench.arm.tip_clearance()[0] > 0.05                          # and the arm went back to rest


def test_a_problem_read_the_wrist_disagrees_with_asks_and_says_what_it_saw(tmp_path, monkeypatch):
    hand_with(monkeypatch, "7C")
    ask = Answers("8C")
    p = player(tmp_path, monkeypatch, ask)
    reader_for(p, monkeypatch, {"7C": "8C"})
    p.draws = []
    assert run(p) == "attended"
    asked = next(i for i in interventions_of(p) if i["kind"] == "name_asked")
    assert asked["answer"] == "8C" and "wrist camera reads 8C (0.90)" in asked["why"]
    assert p.state["hand"]["H1"] == "8C"


def test_the_arm_is_back_at_rest_once_the_looks_are_done(tmp_path, monkeypatch):
    """The next scan needs the mat in clear view, and a person's asked about the hand next."""
    hand_with(monkeypatch, "7C")
    p = player(tmp_path, monkeypatch, Answers())
    reader_for(p, monkeypatch)
    p.bench.arm.rest()
    p.register()
    commanded = p.bench.arm.commanded()
    p.close()
    commanded.pop("gripper")                              # rest leaves the jaws as they are
    assert commanded == pytest.approx({j: v for j, v in config.REST_POSE.items() if j != "gripper"})


def test_only_problem_reads_get_a_look(tmp_path, monkeypatch):
    hand_with(monkeypatch, "7C", "9C")
    p = player(tmp_path, monkeypatch, Answers())
    reader = reader_for(p, monkeypatch)
    p.bench.arm.rest()
    p.register()
    p.close()
    assert reader.reads == 2 and [c["slot"] for c in checks_of(p)] == ["H1", "H2"]


def test_without_a_reader_a_problem_read_is_asked_as_before(tmp_path, monkeypatch):
    hand_with(monkeypatch, "7C")
    p = player(tmp_path, monkeypatch, Answers("7C"))
    p.draws = []
    assert p.reader is None and run(p) == "attended"
    assert [i["slot"] for i in interventions_of(p) if i["kind"] == "name_asked"] == ["H1"]


def test_a_tile_out_of_reach_of_the_look_is_asked_about(tmp_path, monkeypatch):
    hand_with(monkeypatch, "7C")
    p = player(tmp_path, monkeypatch, Answers("7C"))
    reader = reader_for(p, monkeypatch)
    monkeypatch.setattr(p.bench, "hover_over", lambda tile: None)
    p.bench.arm.rest()
    p.register()
    p.close()
    assert reader.reads == 0 and [i["slot"] for i in interventions_of(p) if i["kind"] == "name_asked"] == ["H1"]


def test_a_drawn_problem_read_is_asked_about_even_when_the_wrist_agrees(tmp_path, monkeypatch):
    ask = Answers("", "7C")                              # the hand's fine; the drawn tile is the 7C it reads
    p = player(tmp_path, monkeypatch, ask)
    reader_for(p, monkeypatch)
    p.draws = ["7C"]
    run(p)
    asked = [i for i in interventions_of(p) if i["kind"] == "name_asked"]
    assert [a["slot"] for a in asked] == ["zone"] and "wrist camera reads 7C (0.90)" in asked[0]["why"]


def test_read_hand_doesn_t_look_twice_at_a_problem_read(tmp_path, monkeypatch):
    hand_with(monkeypatch, "7C")
    p = player(tmp_path, monkeypatch, Answers(), argv=("--sim", "--read-hand"))
    reader_for(p, monkeypatch)
    p.bench.arm.rest()
    p.register()
    p.close()
    slots = [c["slot"] for c in checks_of(p)]
    assert len(slots) == 13 and slots.count("H1") == 1


def test_a_wrong_name_the_wrist_agreed_with_counts_against_the_votes_in_the_ledger(tmp_path, monkeypatch):
    """A person corrects it at the hand check: the robot's agreed name was wrong, so it mustn't read as a question that was asked."""
    from mahjong import name_ledger
    hand_with(monkeypatch, "7C")
    p = player(tmp_path, monkeypatch, Answers("H1=8C", ""))
    reader_for(p, monkeypatch)
    p.draws = []
    run(p)
    placements, _, _ = name_ledger.load(tmp_path, review={}, include_sim=True)
    assert [row["tile"] for row in placements if row["outcome"] == name_ledger.WRONG] == ["8C"]


def test_the_ask_list_has_the_name_5c_was_misread_as():
    assert "5C" in identity.ALWAYS_ASK_READS["9C"]
