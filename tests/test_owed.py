"""mahjong/owed.py: what's owed and overdue, shown by status.py and at each hardware session's start."""
import json

from mahjong import owed, scorecard
from test_scorecard import on_standard, paired

REGISTRY = {"default": "b", "default_note": {"since": "2026-11-12"}, "policies": {"a": {}, "b": {}, "c": {}}}


def bench(sessions, name, events):
    folder = sessions / f"{name}_bench"
    folder.mkdir(parents=True)
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return folder


def test_a_default_with_no_check_since_it_was_chosen_is_owed(tmp_path):
    assert "has passed neither scorecard check" in owed.default_checks(tmp_path, REGISTRY)


def test_a_play_ready_run_on_the_standard_setup_since_then_settles_it(tmp_path):
    rounds = scorecard.PLAY_READY_ATTEMPTS
    bench(tmp_path, "20261111_100000", on_standard(paired(rounds)))                  # before it became the default
    assert owed.default_checks(tmp_path, REGISTRY)
    simulated = [{**e, "sim": True} if e["type"] == "session_start" else e for e in on_standard(paired(rounds))]
    bench(tmp_path, "20261112_100000", simulated)                                    # a sim never counts
    assert owed.default_checks(tmp_path, REGISTRY)
    folder = bench(tmp_path, "20261112_174429", on_standard(paired(rounds)))
    assert owed.default_checks(tmp_path, REGISTRY) is None
    assert (folder / "clean_counts.json").exists()                                    # the derive isn't redone next time


def test_a_session_start_never_derives_and_says_nothing_it_cant_back(tmp_path):
    bench(tmp_path, "20261112_174429", on_standard(paired(scorecard.PLAY_READY_ATTEMPTS)))
    assert owed.default_checks(tmp_path, REGISTRY, quick=True) is None               # not summarised yet: no claim
    assert not (tmp_path / "20261112_174429_bench" / "clean_counts.json").exists()
    assert owed.default_checks(tmp_path, REGISTRY) is None                           # status.py summarises it


def test_a_bench_earlier_on_the_day_of_the_decision_doesnt_count(tmp_path):
    decided = {**REGISTRY, "default_note": {"since": "2026-11-12", "at": "20261112_120000"}}
    bench(tmp_path, "20261112_090000", on_standard(paired(scorecard.PLAY_READY_ATTEMPTS)))
    assert owed.default_checks(tmp_path, decided)
    bench(tmp_path, "20261112_174429", on_standard(paired(scorecard.PLAY_READY_ATTEMPTS)))
    assert owed.default_checks(tmp_path, decided) is None


def test_a_default_chosen_face_down_still_owes_a_face_up_round(tmp_path):
    from test_scorecard import face_down
    rounds = scorecard.ITERATION_PAIRS
    bench(tmp_path, "20261112_120000", face_down(on_standard(paired(rounds, a_clean=lambda i: i > 2))))
    assert owed.default_checks(tmp_path, REGISTRY) is None                       # face down now counts: Ahead 3-0
    assert "has had no face-up round" in owed.face_up_round(tmp_path, REGISTRY)
    bench(tmp_path, "20261112_150000", on_standard(paired(rounds, b_clean=lambda i: i > 2)))
    assert "has had no face-up round" in owed.face_up_round(tmp_path, REGISTRY)  # a face-up round, but a Drop
    bench(tmp_path, "20261112_180000", on_standard(paired(rounds)))
    assert owed.face_up_round(tmp_path, REGISTRY) is None                        # level is no Drop: settled


def test_a_voided_session_settles_nothing(tmp_path, monkeypatch):
    void = tmp_path / "void.json"
    void.write_text(json.dumps({"20261112_174429_bench": "torn wrist frames"}))
    monkeypatch.setattr(scorecard, "VOID_SESSIONS", void)
    bench(tmp_path, "20261112_174429", on_standard(paired(scorecard.PLAY_READY_ATTEMPTS)))
    assert owed.default_checks(tmp_path, REGISTRY) and owed.face_up_round(tmp_path, REGISTRY)


def test_roadmap_behind_the_registry(tmp_path):
    (tmp_path / "ROADMAP.md").write_text("| Item | State (2026-11-02) | Evidence | Next |\n| Default | b |\n")
    lines = owed.roadmap_lines(tmp_path, REGISTRY)
    assert lines[0].startswith("registered but not in ROADMAP.md: c") and "dated 2026-11-02" in lines[1]
    (tmp_path / "ROADMAP.md").write_text("| Item | State (2026-11-13) | Evidence | Next |\n| Default | b, then c |\n")
    assert owed.roadmap_lines(tmp_path, REGISTRY) == []


def test_a_policy_is_named_by_its_id_or_its_version_alone():
    assert owed._named("pickup_mat_cl_v11", "v11 leads v10") and not owed._named("pickup_mat_cl_v1", "pickup_mat_cl_v10 only")


def test_a_reminder_that_breaks_says_so_and_never_stops_a_session(monkeypatch):
    monkeypatch.setattr(owed, "default_checks", lambda *a, **k: 1 / 0)
    assert any("couldn't check one owed item: ZeroDivisionError" in line for line in owed.items())
