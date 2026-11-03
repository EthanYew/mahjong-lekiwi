"""The name report, on small sessions built from what really happened: 2B read as 4B and accepted, a knocked 3D sitting on
1B's spot, attempts logged under a spot id; and end to end on sim bench and play runs."""
import json
import sys
from pathlib import Path

import pytest

from mahjong import config, name_ledger
from mahjong.session import read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def scan(n, reads, label="choose", **fields):
    return {"type": "detections", "scan": n, "label": label, "faint": [], "tiles": [
        {"tile_name": name, "xy": xy, "confidence": 0.8} for name, xy in reads], **fields}


def attempt(folder, spot="S7", xy=(0.2, 0.0), tile="2B"):
    return {"type": "attempt", "tile": tile, "spot": spot, "spot_xy": list(xy), "detection_xy": list(xy),
            "attempt_dir": folder, "success": True}


def write_session(tmp_path, name, events, **start):
    folder = tmp_path / name
    folder.mkdir()
    start = {"type": "session_start", "conditions": {"surface": "standard mat"}, "detector": "det@1", **start}
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in [start, *events]))
    return folder


def verdict(name=None):
    return {"verdict": "wrong", "name": name} if name else {"verdict": "ok", "name": None}


XY = (0.2, 0.0)


def test_a_wrong_name_the_votes_accept_is_severe_with_what_it_was_read_as(tmp_path):
    write_session(tmp_path, "20261103_121112_bench", [
        scan(1, [("4B", XY)]), attempt("001_1_NW"), scan(2, [("4B", XY)], "after_return"), scan(3, [("4B", XY)])])
    placements, _, _ = name_ledger.load(tmp_path, review={"20261103_121112_bench/001_1_NW/wrist_0000.jpg": verdict("2B")})
    row = name_ledger.rows(placements)[0]
    assert (row["tile"], row["band"], row["accepted_wrong"], row["mixups"]) == ("2B", "severe", 1, [("4B", 3)])
    assert row["wrong_share"] == 1.0 and row["backfilled"] == 1


def test_a_knocked_neighbour_on_the_spot_isnt_a_misread(tmp_path):
    """20261030_115706_bench: a knock left 3D on 1B's spot, and 60 scans "read 1B as 3D". The review doesn't say what moved."""
    write_session(tmp_path, "20261030_115706_bench", [
        scan(1, [("1B", XY), ("3D", (0.2, 0.06))]), attempt("001_1_1B", tile="1B"),
        scan(2, [("1B", XY), ("3D", (0.2, 0.06))], "after_return"),
        scan(3, [("1B", (0.2, -0.036)), ("3D", (0.2, 0.02))], "after_return"),
        {"type": "scan_review", "scan": 3, "proposed": "knocked", "kinds": ["knocked"]},
        scan(4, [("1B", (0.2, -0.036)), ("3D", (0.2, 0.02))]), scan(5, [("1B", (0.2, -0.036)), ("3D", (0.2, 0.02))])])
    placements, _, _ = name_ledger.load(tmp_path, review={"20261030_115706_bench/001_1_1B/wrist_0000.jpg": verdict()})
    assert [(p["tile"], p["right"], p["wrong"], p["outcome"]) for p in placements] == [("1B", 2, 0, name_ledger.RIGHT)]


def test_a_spot_id_is_never_a_name_and_an_unreviewed_photo_isnt_truth(tmp_path):
    write_session(tmp_path, "20261103_214640_bench", [scan(1, [("7C", XY)]), attempt("001_1_S9", spot="S9", tile="S9"),
                                                      scan(2, [("7C", XY)]), attempt("002_1_7C", spot="S2", tile="7C")])
    placements, skipped, _ = name_ledger.load(tmp_path, review={"20261103_214640_bench/001_1_S9/wrist_0000.jpg": verdict()})
    assert placements == [] and skipped["no truth"] == 2
    placements, _, _ = name_ledger.load(tmp_path, review={"20261103_214640_bench/001_1_S9/wrist_0000.jpg": verdict("8C")})
    assert [(p["tile"], p["outcome"]) for p in placements] == [("8C", name_ledger.WRONG)]


def test_a_spot_with_two_names_in_its_photos_is_left_out(tmp_path):
    write_session(tmp_path, "20261103_000000_bench", [scan(1, [("2B", XY)]), attempt("001_1_2B"), scan(2, [("2B", XY)]),
                                                      attempt("002_1_3C", tile="3C")])
    review = {"20261103_000000_bench/001_1_2B/wrist_0000.jpg": verdict(),
              "20261103_000000_bench/002_1_3C/wrist_0000.jpg": verdict()}
    placements, skipped, _ = name_ledger.load(tmp_path, review=review)
    assert placements == [] and skipped["conflicted"] == 1


def test_a_person_said_name_is_the_truth_and_a_review_overrules_it(tmp_path):
    placement = {"type": "placement", "id": "P1", "spot": "S7", "xy": list(XY), "truth": "4B", "truth_source": "person"}
    write_session(tmp_path, "20261104_000000_bench", [placement, scan(1, [("4B", XY)]), scan(2, [("4B", XY)]),
                                                      {**attempt("001_1_4B", tile="4B"), "placement": "P1"}])
    placements, _, _ = name_ledger.load(tmp_path, review={})
    assert [(p["tile"], p["truth_source"], p["outcome"], p["overruled"], p["backfilled"]) for p in placements] == [
        ("4B", "person", name_ledger.RIGHT, False, False)]
    placements, _, _ = name_ledger.load(tmp_path, review={"20261104_000000_bench/001_1_4B/wrist_0000.jpg": verdict("2B")})
    assert [(p["tile"], p["truth_source"], p["outcome"], p["overruled"]) for p in placements] == [
        ("2B", "review", name_ledger.WRONG, True)]


def test_an_unchecked_placement_isnt_scored(tmp_path):
    placement = {"type": "placement", "id": "P1", "spot": "S7", "xy": list(XY), "truth": None, "truth_source": "unconfirmed"}
    write_session(tmp_path, "20261104_000000_bench", [placement, scan(1, [("4B", XY)])])
    placements, skipped, _ = name_ledger.load(tmp_path, review={})
    assert placements == [] and skipped["no truth"] == 1


@pytest.mark.parametrize("fields, band", [
    (dict(accepted_wrong=1, asks=0, placements=20, sessions=9, wrong_share=0.0, unnamed_share=0.0), "severe"),
    (dict(accepted_wrong=0, asks=2, placements=8, sessions=4, wrong_share=0.0, unnamed_share=0.0), "costly"),
    (dict(accepted_wrong=0, asks=1, placements=8, sessions=4, wrong_share=0.05, unnamed_share=0.06), "flaky"),
    (dict(accepted_wrong=0, asks=0, placements=5, sessions=3, wrong_share=0.0, unnamed_share=0.0), "fine"),
    (dict(accepted_wrong=0, asks=0, placements=4, sessions=3, wrong_share=0.0, unnamed_share=0.0), "not enough data"),
    (dict(accepted_wrong=0, asks=0, placements=5, sessions=2, wrong_share=0.0, unnamed_share=0.0), "not enough data"),
])
def test_bands(fields, band):
    assert name_ledger.band(fields) == band


def test_weights_fall_back_to_what_git_had_when_the_session_began(monkeypatch):
    name_ledger._weights_history.cache_clear()
    monkeypatch.setattr(name_ledger, "_weights_history", lambda: [(1000, "old"), (2000, "new")])
    from datetime import datetime
    name = datetime.fromtimestamp(1500).strftime("%Y%m%d_%H%M%S") + "_bench"
    assert name_ledger.weights_of(name, {}) == "old"
    assert name_ledger.weights_of(name, {"detector": "logged"}) == "logged"


def test_sim_sessions_are_left_out_and_a_sim_bench_run_can_be_read_with_sim(tmp_path, monkeypatch):
    import bench
    from mahjong import layout
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(layout, "LAYOUTS", tmp_path / "layouts.json", raising=False)
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "1"]))
    b.realsense.tiles = [{"tile_name": "9D", "xy": [0.21, 0.02]}, {"tile_name": "3B", "xy": [0.21, -0.04]}]
    b.spots = [{"id": "S1", "xy": [0.21, 0.02]}, {"id": "S2", "xy": [0.21, -0.04]}]
    try:
        b.trials(1)
    finally:
        b.close()
    assert name_ledger.load(tmp_path, review={})[0] == []
    placements, _, _ = name_ledger.load(tmp_path, review={}, include_sim=True)
    assert {(p["tile"], p["truth_source"], p["outcome"]) for p in placements} == {
        ("9D", "sim", name_ledger.RIGHT), ("3B", "sim", name_ledger.RIGHT)}
    assert all(p["weights"] == read_start(b)["detector"] for p in placements)


def read_start(b):
    return read_events(b.session.dir)[0]


class Answers:
    def __init__(self, *answers):
        self.answers = list(answers)

    def __call__(self, prompt):
        return self.answers.pop(0) if self.answers else ""


def play_a_hand(tmp_path, monkeypatch, *answers):
    import play
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    p = play.Player(play.parse_args(["--sim"]), ask=Answers(*answers))
    try:
        p.play()
    finally:
        p.close()
    return p


def test_a_played_hand_logs_every_name_and_the_audit_settles_them(tmp_path, monkeypatch):
    p = play_a_hand(tmp_path, monkeypatch)
    events = read_events(p.session.dir)
    decided = [e for e in events if e["type"] == "name_decided"]
    assert len(decided) >= 13 and {e["how"] for e in decided} == {"vote"} and all(e["votes"] and e["placement"] for e in decided)
    audit = next(e for e in events if e["type"] == "hand_audit")
    assert len(audit["confirmed"]) >= 13
    placements, skipped, _ = name_ledger.load(tmp_path, review={}, include_sim=True)
    assert all(p_["truth_source"] == "person" for p_ in placements)     # a tile moved into a slot keeps what was said of it
    assert sum(p_["decided_live"] for p_ in placements) >= 13
    assert {p_["outcome"] for p_ in placements} == {name_ledger.RIGHT}


def test_a_wrong_discard_found_at_the_audit_counts_as_accepted_wrong(tmp_path, monkeypatch):
    p = play_a_hand(tmp_path, monkeypatch, "d1=5D")
    events = read_events(p.session.dir)
    error = next(e for e in events if e["type"] == "intervention" and e["kind"] == "silent_error")
    assert error["slot"] == "D1" and error["placement"] and error["table"] == "5D"
    end = next(e for e in events if e["type"] == "hand_end")
    assert end["silent_errors"] == 1 and end["grade"] == "silent error"
    placements, _, _ = name_ledger.load(tmp_path, review={}, include_sim=True)
    assert [(p_["tile"], p_["outcome"]) for p_ in placements if p_["outcome"] == name_ledger.WRONG] == [("5D", name_ledger.WRONG)]


def test_a_person_answer_is_logged_with_the_votes_and_whether_they_typed(tmp_path, monkeypatch):
    import play
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    p = play.Player(play.parse_args(["--sim"]), ask=Answers("4c"))
    try:
        p.votes.add("H1", {"tile_name": "3C", "confidence": 0.5})
        assert p.ask_name("H1", "unsure") == "4C"
    finally:
        p.close()
    events = read_events(p.session.dir)
    asked = next(e for e in events if e["type"] == "intervention" and e["kind"] == "name_asked")
    assert asked["typed"] is True and asked["votes"] == {"3C": 0.5} and asked["placement"]
    assert next(e for e in events if e["type"] == "name_decided")["how"] == "person"


def test_the_report_runs_end_to_end(tmp_path, capsys):
    import name_report
    write_session(tmp_path, "20261103_121112_bench", [
        scan(1, [("4B", XY)]), attempt("001_1_NW"), scan(2, [("4B", XY)], "after_return")])
    review_file = config.DATA / "wrist_review.jsonl"
    assert review_file.exists()
    name_report.main(["--sessions-dir", str(tmp_path), "--by", "lighting", "--list"])
    out = capsys.readouterr().out
    assert "1 sessions" in out and "not scored" in out      # that photo isn't in the real review file


def test_status_says_when_nothing_is_scored(tmp_path):
    assert "no scored placements" in name_ledger.worst_line(tmp_path)
