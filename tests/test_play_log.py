"""mahjong/play_log.py: one way to read a played hand, face up or face down, from the logs written before it and after."""
from mahjong import game, play_log

SLOTS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S14"]
# 20261119_142253_play_fd's hand at its last turn: 3D completes it (234B 456B 345D 567C 7C7C)
HAND = ["4B", "5D", "4B", "3B", "5C", "6B", "7C", "7C", "4D", "5B", "6C", "2B", "7C"]
BEFORE = dict(zip(SLOTS, ["4B", "5D", "4B", "3B", "5C", "6B", "7C", "7C", "4D", "8B", "6C", "2B", "7C"]))


def event(minute, type, **fields):
    return {"time": f"2026-11-19T15:{minute:02d}:00", "type": type, **fields}


def old_face_down(audit_wrong=None, checked=True):
    """The events of a face-down hand as play_face_down.py logged them before play_log: two turns, then a win on 3D."""
    return [
        event(0, "session_start", kind="play_fd", unattended=True),
        event(1, "crude_pick", turn=0, step="register", spot="S1", held=True, attempt_dir="001_150100_reg_S1"),
        event(1, "fd_register_read", slot="S1", guess={"name": "4B", "confidence": 0.95}),
        event(2, "fd_hand_confirmed", hand=BEFORE, fixed={}, by="reader"),
        event(3, "fd_turn_start", turn=1, slots=SLOTS, hand=BEFORE, discards=[]),
        event(4, "crude_pick", turn=1, step="draw", spot="L2", held=True, attempt_dir="002_150400_t1_draw_top"),
        event(5, "crude_read", turn=1, guess="5B", name="5B", guess_confidence=0.94, by="reader"),
        event(5, "crude_choice", turn=1, drawn="5B", discard="8B", slot="S10", shanten_after=0),
        event(6, "crude_pick", turn=1, step="pick_from_hand", spot="S10", held=True, attempt_dir="003_150600_t1_S10"),
        event(7, "crude_pick", turn=1, step="pick_from_zone", spot="zone", held=True, attempt_dir="004_150700_t1_zone"),
        event(8, "fd_turn_start", turn=2, slots=SLOTS, hand=dict(zip(SLOTS, HAND)), discards=["8B"]),
        event(9, "crude_read", turn=2, guess="WD", name="WD", guess_confidence=0.95, by="reader"),
        event(9, "crude_choice", turn=2, drawn="WD", discard="WD", slot=None, shanten_after=0),
        event(10, "fd_turn_start", turn=3, slots=SLOTS, hand=dict(zip(SLOTS, HAND)), discards=["8B", "WD"]),
        event(11, "crude_read", turn=3, guess="3D", name="3D", guess_confidence=0.9529, by="reader"),
        event(11, "crude_win", turn=3, drawn="3D"),
        event(12, "fd_audit", checked=checked, wrong=audit_wrong or {}, hand=dict(zip(SLOTS, HAND))),
        event(12, "hand_end", result="win", turns=3, grade="clean", hand=dict(zip(SLOTS, HAND)), discards=["8B", "WD"]),
    ]


def new_face_down():
    """The same hand as play_log writes it now."""
    read = {"name": "5B", "confidence": 0.94, "by": "reader"}
    return [
        event(0, "session_start", kind="play_fd", unattended=True),
        event(1, "attempt", tile=None, policy="v13", success=True, attempt_dir="001_150100_reg_S1", spot="S1", step="register", turn=0),
        event(1, "fd_register_read", slot="S1", guess={"name": "4B", "confidence": 0.95}),
        event(2, "fd_hand_confirmed", hand=BEFORE, fixed={}, by="reader"),
        event(3, "fd_turn_start", turn=1, slots=SLOTS, hand=BEFORE, discards=[]),
        event(5, "discard_chosen", turn=1, drawn="5B", slot="S10", index=9, tile_name="8B", shanten_after=0, read=read),
        event(6, "attempt", tile="8B", policy="v13", success=True, attempt_dir="003_150600_t1_S10", spot="S10",
              step="pick_from_hand", turn=1),
        event(7, "turn_done", turn=1, discarded="8B", slot="S10", slots=SLOTS, hand=HAND, discards=["8B"]),
        event(8, "fd_turn_start", turn=2, slots=SLOTS, hand=dict(zip(SLOTS, HAND)), discards=["8B"]),
        event(9, "discard_chosen", turn=2, drawn="WD", slot="zone", index=13, tile_name="WD", shanten_after=0,
              read={"name": "WD", "confidence": 0.95, "by": "reader"}),
        event(9, "turn_done", turn=2, discarded="WD", slot="zone", slots=SLOTS, hand=HAND, discards=["8B", "WD"]),
        event(10, "fd_turn_start", turn=3, slots=SLOTS, hand=dict(zip(SLOTS, HAND)), discards=["8B", "WD"]),
        event(11, "win", turn=3, drawn="3D", read={"name": "3D", "confidence": 0.9529, "by": "reader"}),
        event(12, "hand_end", result="win", turns=3, grade="clean", asks=0, rescues=0, silent_errors=0, slots=SLOTS,
              hand=HAND, discards=["8B", "WD"], winning="3D", audit={"checked": True, "wrong": {}, "covers": [*SLOTS, "zone"]}),
    ]


def test_an_older_face_down_log_reads_as_the_same_hand():
    h = play_log.hand(old_face_down())
    assert h["kind"] == "play_fd" and h["unattended"] and h["slots"] == SLOTS
    assert h["start"]["hand"] == list(BEFORE.values())
    one, two = h["turns"]
    assert (one["drawn"], one["discard"], one["slot"], one["read"]["confidence"]) == ("5B", "8B", "S10", 0.94)
    assert one["before"] == list(BEFORE.values()) and one["after"] == HAND and one["discards"] == ["8B"]
    assert one["done"] == "2026-11-19T15:08:00"          # the next turn's start: older logs had no turn_done
    assert (two["slot"], two["seen"], two["discards"]) == (None, ["8B"], ["8B", "WD"])
    assert h["win"]["drawn"] == "3D" and h["win"]["before"] == HAND and h["win"]["seen"] == ["8B", "WD"]
    assert h["audit"]["checked"] and h["audit"]["covers"] == SLOTS
    picks = play_log.pickups(old_face_down())
    assert [(p["step"], p["tile"], p["held"]) for p in picks] == [
        ("register", None, True), ("draw", None, True), ("pick_from_hand", "8B", True), ("pick_from_zone", "5B", True)]


def test_the_shared_events_read_the_same_as_the_older_ones():
    old, new = play_log.hand(old_face_down()), play_log.hand(new_face_down())
    keys = ("turn", "drawn", "discard", "slot", "before", "after", "seen", "discards")
    assert [{k: t[k] for k in keys} for t in new["turns"]] == [{k: t[k] for k in keys} for t in old["turns"]]
    assert new["win"]["drawn"] == old["win"]["drawn"] and new["win"]["before"] == old["win"]["before"]
    assert [p["step"] for p in play_log.pickups(new_face_down())] == ["register", "pick_from_hand"]


def test_a_win_with_the_13_checked_and_its_winning_tile_unreviewed_is_not_verified_yet():
    """20261119_142253_play_fd: unattended, the end check passed, and the 3D it won on was named only by the reader."""
    v = play_log.verdict(play_log.hand(old_face_down()), reads=[])
    assert v["status"] == "not verified"
    assert "End check: a person checked the 13 in hand: all matched" in v["lines"]
    assert "Winning 3D: read at 95% by the reader, photos not reviewed yet" in v["lines"]
    assert v["lines"][-1].startswith("Win not verified yet: needs a review of the winning 3D's photos")


def test_a_reviewed_winning_tile_verifies_the_win_and_a_wrong_one_unmakes_it():
    h = play_log.hand(old_face_down())
    right = [{"turn": 3, "where": "draw t3", "reviewed": True, "truth": "3D"}]
    assert play_log.verdict(h, right)["status"] == "verified"
    wrong = [{"turn": 3, "where": "draw t3", "reviewed": True, "truth": "2D"}]       # 2D doesn't complete it
    v = play_log.verdict(h, wrong)
    assert v["status"] == "not a win" and "Winning 3D: its photos were reviewed: it was 2D" in v["lines"]


def test_an_end_check_that_finds_a_wrong_tile_says_whether_the_hand_still_wins():
    still = play_log.verdict(play_log.hand(old_face_down({"S1": "4B"})), [{"turn": 3, "where": "draw t3", "reviewed": True, "truth": "3D"}])
    assert still["status"] == "verified"                # the same name: no change
    broke = play_log.verdict(play_log.hand(old_face_down({"S10": "9B"})), [])
    assert broke["status"] == "not a win" and "End check: a person checked the 13 in hand: S10 was 9B" in broke["lines"]
    skipped = play_log.verdict(play_log.hand(old_face_down(checked=False)), [])
    assert skipped["status"] == "not verified" and "the end check" in skipped["lines"][-1]


def test_a_winning_tile_on_the_drop_zone_checked_by_a_person_verifies_without_a_review():
    v = play_log.verdict(play_log.hand(new_face_down()))
    assert v["status"] == "verified" and "Winning 3D: a person checked it" in v["lines"]


def test_an_older_face_up_hand_whose_end_check_matched_every_tile_is_verified():
    """Before play_log, face-up play logged no winning tile, but its end check showed a person every tile, the drawn one too."""
    events = [event(0, "session_start", kind="play"), event(1, "turn_done", turn=1, discarded="7B"),
              event(2, "hand_audit", confirmed={}),
              event(3, "hand_end", result="win", turns=2, grade="clean", silent_errors=0, asks=0, rescues=0,
                    hand=HAND, discards=["7B"])]
    assert play_log.verdict(play_log.hand(events))["status"] == "verified"
    events.insert(2, event(2, "intervention", kind="silent_error", slot="H3", robot="4B", table="5B"))
    assert play_log.verdict(play_log.hand(events))["status"] == "not verified"


def test_a_hand_that_wasnt_won_has_no_verdict():
    events = old_face_down()
    events[-1] = {**events[-1], "result": "end"}
    assert play_log.verdict(play_log.hand(events)) is None


def test_a_win_splits_into_its_sets_and_pair_for_the_video():
    assert play_log.melds_line(HAND + ["3D"]) == "567C  345D  234B  456B  77C"
    assert play_log.hand_line(HAND) == "5C 6C 7C 7C 7C  4D 5D  2B 3B 4B 4B 5B 6B"
    pairs = ["1C", "1C", "2D", "2D", "3B", "3B", "EW", "EW", "RD", "RD", "9C", "9C", "5D", "5D"]
    assert len(game.melds(pairs)) == 7 and all(len(m) == 2 for m in game.melds(pairs))
    orphans = ["1C", "9C", "1D", "9D", "1B", "9B", "EW", "SW", "WW", "NW", "WD", "GD", "RD", "RD"]
    assert game.melds(orphans) == [sorted(orphans, key=game.INDEX.get)]
    assert game.melds(HAND + ["8D"]) is None
