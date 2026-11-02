"""mahjong/scorecard.py: a clean attempt harmed nothing, and a paired session only decides what it can show."""
import json

import pytest

from mahjong import scorecard


def attempt(n, policy, spot, success=True, **fields):
    return {"type": "attempt", "attempt_dir": f"{n:03}", "policy": policy, "spot": spot, "success": success,
            "duration_s": 10.0, **fields}


def bump(n, kind="collided"):
    return {"type": "keys", "attempt_dir": f"{n:03}", "during": "pickup", "time": "t",
            "presses": [{"kind": kind, "t_s": 1.0}]}


def paired(rounds, a_clean=lambda i: True, b_clean=lambda i: True):
    """rounds of a then b at one spot each, in the order bench.py --policy a,b runs them."""
    events = [{"type": "session_start", "conditions": {"surface": "black table top"}},
              {"type": "bench_setup", "policies": [{"id": "a"}, {"id": "b"}]}]
    for i in range(rounds):
        events += [attempt(2 * i, "a", f"S{i}"), attempt(2 * i + 1, "b", f"S{i}")]
        if not a_clean(i):
            events.append(bump(2 * i))
        if not b_clean(i):
            events.append(bump(2 * i + 1))
    return events + [{"type": "session_end", "reason": "quit"}]


def test_exact_ranges_match_the_roadmaps_figures():
    low, high = scorecard.exact_ci(28, 28)
    assert low == pytest.approx(0.877, abs=0.001) and high == 1.0
    assert scorecard.exact_ci(9, 9)[0] == pytest.approx(0.664, abs=0.001)
    assert scorecard.exact_ci(0, 0) is None


def test_mcnemar_is_exact():
    assert scorecard.mcnemar_p(5, 0) == pytest.approx(0.0625)
    assert scorecard.mcnemar_p(6, 1) == pytest.approx(0.125)
    assert scorecard.mcnemar_p(0, 0) == 1.0


@pytest.mark.parametrize("kind", ["collided", "pushed_target"])
def test_a_picked_tile_that_bumped_something_is_not_clean(kind):
    events = [attempt(0, "a", "S1"), bump(0, kind), attempt(1, "a", "S1"), attempt(2, "a", "S1", success=False)]
    assert scorecard.clean_flags(events) == {"000": False, "001": True, "002": False}
    row = scorecard.policy_rows(events)["a"]
    assert (row["attempts"], row["picked"], row["clean"]) == (3, 2, 1)


def test_a_precarious_grip_stands_and_a_voided_press_says_nothing():
    events = [attempt(0, "a", "S1"), bump(0, "grip_low"), attempt(1, "a", "S1"), bump(1),
              {"type": "keys_void", "keys": [{"attempt_dir": "001", "during": "pickup"}]}]
    assert scorecard.clean_flags(events) == {"000": True, "001": True}


def test_pairs_leave_out_an_attempt_with_no_partner():
    events = paired(2) + [attempt(9, "a", "S0")]
    assert [spot for spot, _, _ in scorecard.pairs(events, "a", "b")] == ["S0", "S1"]


def test_too_few_pairs_get_no_verdict():
    assert scorecard.verdict(paired(19), "a", "b")["verdict"] == "no verdict"


def test_a_tie_is_more_attempts_not_a_ship():
    v = scorecard.verdict(paired(20), "a", "b")
    assert v["verdict"] == "extend" and (v["wins"], v["losses"]) == (0, 0)


def test_a_clear_split_ships_or_reverts():
    ship = scorecard.verdict(paired(20, a_clean=lambda i: i > 4), "a", "b")
    assert ship["verdict"] == "ship" and (ship["wins"], ship["losses"]) == (5, 0) and "black table top" in ship["why"]
    assert scorecard.verdict(paired(20, b_clean=lambda i: i > 4), "a", "b")["verdict"] == "revert"


def test_a_candidate_worse_at_one_spot_gets_more_attempts_there():
    events = [{"type": "session_start"}, {"type": "bench_setup", "policies": [{"id": "a"}, {"id": "b"}]}]
    for n in range(22):
        spot = "S1" if n < 2 else f"S{n}"
        events += [attempt(2 * n, "a", spot), attempt(2 * n + 1, "b", spot)]
        if n < 2:
            events.append(bump(2 * n + 1))
    v = scorecard.verdict(events, "a", "b")
    assert v["verdict"] == "extend" and v["worse_spots"] == ["S1"]


def test_the_report_shows_clean_before_picked_and_says_when_it_cant_judge():
    text = "\n".join(scorecard.report(paired(20, a_clean=lambda i: i > 4)))
    assert "15/20" in text and "20/20" in text and "b against a: SHIP" in text
    assert "no verdict" in "\n".join(scorecard.report([attempt(0, "a", "S1")]))


def test_session_clean_is_saved_and_redone_when_the_events_grow(tmp_path):
    events_file = tmp_path / "events.jsonl"
    events_file.write_text("".join(json.dumps(e) + "\n" for e in [attempt(0, "a", "S1")]))
    assert scorecard.session_clean(tmp_path) == {"a": 1}
    assert (tmp_path / "clean_counts.json").exists()
    with events_file.open("a") as f:
        f.write(json.dumps(bump(0)) + "\n")
    assert scorecard.session_clean(tmp_path) == {"a": 0}
