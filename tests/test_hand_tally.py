"""mahjong/hand_tally.py: which face-down hands count toward 8 clean in 10 consecutive, 0 silent errors."""
import json

from mahjong import face_down_play, hand_tally, play_log

END_OK = {"result": "win", "turns": 15, "grade": "clean", "end_check": True, "silent_errors": 0}


def hand(end=END_OK, unattended=False):
    return {"end": end, "unattended": unattended, "win": None, "audit": {"checked": True, "wrong": {}, "covers": ["ZONE"]},
            "slots": ["S1"], "turns": []}


def read(filed="5C", truth="5C", reviewed=True, where="draw"):
    return {"filed": filed, "truth": truth, "reviewed": reviewed, "where": where, "guess": filed, "turn": 1}


TURN = [{"type": "fd_turn_start", "turn": 1}]


def judge(h, reads, events=TURN, monkeypatch=None):
    return hand_tally.judge(h, reads, events)


def verified(monkeypatch, status="verified"):
    monkeypatch.setattr(play_log, "verdict", lambda h, reads=None: {"status": status, "lines": ["no one vouched"]})


def test_a_reviewed_hand_with_every_read_right_is_graded_as_hand_end_graded_it(monkeypatch):
    verified(monkeypatch)
    assert judge(hand(), [read(), read()])["status"] == "clean"
    assert judge(hand({**END_OK, "grade": "attended"}), [read()])["status"] == "attended"


def test_an_unattended_hand_counts_once_its_holds_are_reviewed(monkeypatch):
    verified(monkeypatch)
    assert judge(hand(unattended=True), [read(), read()])["status"] == "clean"
    assert judge(hand(unattended=True), [read(), read(reviewed=False)])["status"] == "waiting for review"


def test_a_hold_marked_wrong_with_no_name_is_still_waiting(monkeypatch):
    verified(monkeypatch)
    assert judge(hand(), [read(truth=None)])["status"] == "waiting for review"


def test_a_wrong_read_the_hand_used_is_a_silent_error_even_with_holds_still_waiting(monkeypatch):
    verified(monkeypatch)
    out = judge(hand(unattended=True), [read(filed="9C", truth="5C"), read(reviewed=False)])
    assert out["status"] == "silent error" and "9C" in out["why"] and "5C" in out["why"]


def test_the_end_checks_wrong_tile_is_a_silent_error(monkeypatch):
    verified(monkeypatch)
    assert judge(hand({**END_OK, "silent_errors": 1}), [read()])["status"] == "silent error"


def test_a_win_no_person_vouched_for_is_not_verified(monkeypatch):
    verified(monkeypatch, "not verified")
    assert judge(hand(), [read()])["status"] == "not verified"


def test_a_hand_that_started_turns_and_never_ended_is_a_failed_hand_but_a_quit_at_the_window_is_not_a_hand():
    assert judge(hand(None), [])["status"] == "unfinished"
    assert judge(hand({**END_OK, "end_check": False}), [])["status"] == "unfinished"
    assert judge(hand(None), [], events=[]) is None


def test_hands_after_one_waiting_for_review_are_not_counted_as_consecutive():
    rows = [{"status": s} for s in ("clean", "waiting for review", "clean")]
    counted, behind = hand_tally.series(rows)
    assert [r["status"] for r in counted] == ["clean"] and len(behind) == 2


def test_the_target_needs_ten_hands_with_eight_clean_and_no_silent_error():
    def win(statuses):
        return hand_tally.window([{"status": s, "setup": "x"} for s in statuses])
    assert win(["clean"] * 8 + ["attended"] * 2)["met"]
    assert not win(["clean"] * 7 + ["rescued"] * 3)["met"]
    assert not win(["clean"] * 9)["met"], "nine hands is not ten"
    assert not win(["clean"] * 9 + ["silent error"])["met"]
    assert win(["rescued", "rescued"] + ["clean"] * 10)["met"], "only the last ten count"


def test_the_tally_reads_session_folders_and_leaves_out_sim_and_voided_ones(tmp_path, monkeypatch):
    def session(name, sim=False, grade="clean"):
        folder = tmp_path / f"{name}_play_fd"
        folder.mkdir()
        events = [{"type": "session_start", "kind": "play_fd", "sim": sim, "unattended": True, "policy": "p", "layout": "l",
                   "station": "L2"}, {"type": "fd_turn_start", "turn": 1, "slots": ["S1"], "hand": ["5C"], "discards": []},
                  {"type": "fd_audit", "checked": True, "wrong": {}, "hand": {"S1": "5C"}},
                  {"type": "hand_end", **END_OK, "grade": grade, "slots": ["S1"], "hand": ["5C"]}]
        (folder / "events.jsonl").write_text("".join(json.dumps({"time": "2026-11-19T10:00:00", **e}) + "\n" for e in events))
    session("20261119_100000")
    session("20261119_110000", sim=True)
    session("20261119_120000", grade="rescued")
    monkeypatch.setattr(face_down_play, "scored_reads", lambda folder, review=None: [read()])
    monkeypatch.setattr(play_log, "verdict", lambda h, reads=None: {"status": "verified", "lines": []})
    rows = hand_tally.tally(tmp_path, review={})
    assert [(r["name"], r["status"]) for r in rows] == [("20261119_100000_play_fd", "clean"),
                                                          ("20261119_120000_play_fd", "rescued")]
    text = "\n".join(hand_tally.lines(rows))
    assert "counted 2 hand(s)" in text and "not met" in text
