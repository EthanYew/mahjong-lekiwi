"""scripts/status.py: the report reads the registry and the bench sessions; --set-default keeps its evidence."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import status
from mahjong import config, session


def bench_session(sessions, name, policy, results):
    folder = sessions / f"{name}_bench"
    folder.mkdir(parents=True)
    (folder / "events.jsonl").write_text("".join(
        json.dumps({"type": "attempt", "attempt_dir": f"{i:03}", "policy": policy, "success": ok}) + "\n"
        for i, ok in enumerate(results)))


def test_the_report_counts_each_policys_bench_attempts_and_marks_the_default(tmp_path, capsys):
    bench_session(tmp_path, "20261029_094543", "pickup_mat_cl_v4", [True, True, False])
    bench_session(tmp_path, "20261029_123745", "pickup_mat_cl_v4", [True])
    results = status.bench_results(tmp_path)
    assert results["pickup_mat_cl_v4"] == {"attempts": 4, "picked": 3, "sessions": 2, "last": "20261029_123745"}
    text = status.report(sessions_dir=tmp_path)
    default = json.loads(config.POLICIES.read_text())["default"]
    assert any(line.startswith(default) and line.endswith("<- default") for line in text.splitlines())
    assert "hand14" in text and "Eval sets" in text
    assert "Standard setup: standard mat, even-hand14" in text and "no bench sessions on it yet" in text
    assert any(line.startswith("pickup_mat_cl_v4") and "3/4" in line for line in text.splitlines())


def test_set_default_writes_the_evidence_with_it(tmp_path, monkeypatch):
    policies = tmp_path / "policies.json"
    shutil.copy(config.POLICIES, policies)
    monkeypatch.setattr(config, "POLICIES", policies)
    with pytest.raises(SystemExit):
        status.parse_args(["--set-default", "pickup_mat_cl_v3c"])            # no evidence: refused
    status.main(["--set-default", "pickup_mat_cl_v3c", "--evidence", "sessions/x_bench: 10/10", "--why", "test"])
    reg = json.loads(policies.read_text())
    assert reg["default"] == "pickup_mat_cl_v3c"
    assert reg["default_note"]["policy"] == "pickup_mat_cl_v3c" and reg["default_note"]["evidence"] == "sessions/x_bench: 10/10"
    with pytest.raises(SystemExit):
        status.main(["--set-default", "no_such_policy", "--evidence", "x"])


def test_set_default_says_what_docs_still_name_the_old_one_and_warns_on_main(tmp_path, monkeypatch, capsys):
    policies = tmp_path / "policies.json"
    shutil.copy(config.POLICIES, policies)
    monkeypatch.setattr(config, "POLICIES", policies)
    monkeypatch.setattr(status.checkout, "git", lambda *args, **kw: "main")
    status.main(["--set-default", "pickup_mat_cl_v3c", "--evidence", "x", "--since", "2026-11-05"])
    out = capsys.readouterr().out
    assert "WARNING: this is main" in out
    assert "docs/usage.md: the policy list" in out           # v3c isn't the docs' default (ROADMAP's DAgger row names it)
    assert json.loads(policies.read_text())["default_note"]["since"] == "2026-11-05"
    current = json.loads((config.REPO / "data" / "policies.json").read_text())["default"]    # the repo's docs agree with its own
    assert status.stale_docs(current) == []


def test_stale_docs_finds_an_old_dagger_example(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "usage.md").write_text("policies (default: `new`)\n")
    (tmp_path / "CLAUDE.md").write_text("record.py --dataset Yewy/x --policy pickup_mat_cl_v7\n")
    (tmp_path / "ROADMAP.md").write_text("# R\n\n## Status\n\n| Default | new |\n")
    assert [s.split(":")[0] for s in status.stale_docs("new", tmp_path)] == ["CLAUDE.md"]


def test_a_simulated_bench_adds_nothing_to_a_real_policys_totals(tmp_path):
    """bench.py --sim --policy <id> logs attempts under the real id: 20261112_091817_bench gave v10 a pickup that never happened."""
    bench_session(tmp_path, "20261112_091817", "pickup_mat_cl_v10", [True])
    events = tmp_path / "20261112_091817_bench" / "events.jsonl"
    events.write_text(json.dumps({"type": "session_start", "kind": "bench", "sim": True}) + "\n" + events.read_text())
    bench_session(tmp_path, "20261112_202418", "pickup_mat_cl_v10", [True, False])
    assert status.bench_results(tmp_path)["pickup_mat_cl_v10"]["attempts"] == 2
    assert status.bench_clean(tmp_path).get("pickup_mat_cl_v10", 0) <= 1


def test_a_voided_session_is_left_out_of_every_count(tmp_path, monkeypatch):
    void = tmp_path / "void.json"
    void.write_text(json.dumps({"20261113_100710_bench": "torn wrist frames"}))
    monkeypatch.setattr(session, "VOID_SESSIONS", void)
    bench_session(tmp_path / "s", "20261113_100710", "pickup_mat_cl_v10", [False, False])
    bench_session(tmp_path / "s", "20261116_213816", "pickup_mat_cl_v10", [True])
    assert status.bench_results(tmp_path / "s")["pickup_mat_cl_v10"]["attempts"] == 1
    assert "voided, left out of every count above (data/void_sessions.json): 20261113_100710_bench" in status.report(
        sessions_dir=tmp_path / "s")


def test_face_up_and_face_down_are_counted_on_separate_lines(tmp_path):
    from test_scorecard import face_down, on_standard, paired
    for name, events in (("20261117_100000", on_standard(paired(2))), ("20261117_110000", face_down(on_standard(paired(3))))):
        folder = tmp_path / f"{name}_bench"
        folder.mkdir()
        (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    on_it = status.standard_results(tmp_path)
    assert on_it[("face up", "a")]["attempts"] == 2 and on_it[("face down", "a")]["attempts"] == 3
    text = status.report(sessions_dir=tmp_path)
    assert "face down when the tiles were found by the scans' back boxes" in text
    assert any(line.strip().startswith("face down b") and "3/3" in line for line in text.splitlines())


def play_session(sessions, name, events, phase="play"):
    folder = sessions / name
    folder.mkdir(parents=True)
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (folder / "checkpoint.json").write_text(json.dumps({"phase": phase}))
    return folder


STARTED = [{"type": "session_start"},
           {"type": "fd_turn_start", "turn": 1, "slots": ["S1"], "hand": {"S1": "5C"}, "discards": []}]


def test_a_play_run_stopped_partway_can_be_voided_and_is_listed_but_not_counted(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(session, "VOID_SESSIONS", tmp_path / "void.json")
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "s")
    play_session(tmp_path / "s", "20261120_071428_play_fd", STARTED)
    assert "unfinished" in status.report()
    with pytest.raises(SystemExit):
        status.parse_args(["--void", "20261120_071428_play_fd"])        # no reason: refused
    status.main(["--void", "sessions/20261120_071428_play_fd", "--why", "power cut"])
    assert json.loads((tmp_path / "void.json").read_text()) == {"20261120_071428_play_fd": "power cut"}
    text = status.report()
    assert "voided: power cut" in text and "counted 0 hand(s)" in text
    with pytest.raises(SystemExit, match="already voided"):
        status.main(["--void", "20261120_071428_play_fd", "--why", "again"])


@pytest.mark.parametrize("name, events, phase, refusal", [
    ("20261120_100000_play_fd", [*STARTED, {"type": "hand_end", "result": "win", "end_check": True}], "won", "ran to its end"),
    ("20261120_100000_play", [{"type": "hand_end", "result": "end", "end_check": True, "silent_errors": 1}], "end", "wrong tile name"),
    ("20261120_100000_play_fd", [*STARTED, {"type": "hand_end", "result": "win", "end_check": False}], "won", "ran to its end"),
    ("20261120_100000_play_fd", STARTED, "won", "--end-check"),
    ("20261120_100000_bench", [], "play", "isn't a play run"),
])
def test_a_hand_the_robot_finished_or_misread_cant_be_voided(tmp_path, monkeypatch, name, events, phase, refusal):
    monkeypatch.setattr(session, "VOID_SESSIONS", tmp_path / "void.json")
    play_session(tmp_path, name, events, phase)
    with pytest.raises(SystemExit, match=refusal):
        status.void_play(name, "why", sessions_dir=tmp_path)
    assert not (tmp_path / "void.json").exists()


@pytest.mark.parametrize("truth", ["6D", None])
def test_a_hand_with_a_photo_marked_wrong_cant_be_voided(tmp_path, monkeypatch, truth):
    monkeypatch.setattr(session, "VOID_SESSIONS", tmp_path / "void.json")
    play_session(tmp_path, "20261120_100000_play_fd", STARTED)
    monkeypatch.setattr(status.face_down_play, "scored_reads", lambda folder: [{"filed": "5C", "truth": truth, "reviewed": True}])
    with pytest.raises(SystemExit, match="wrong tile name"):
        status.void_play("20261120_100000_play_fd", "why", sessions_dir=tmp_path)

