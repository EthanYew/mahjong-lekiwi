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


def on_standard(events):
    standard = scorecard.standard_setup()
    return [{**e, "conditions": {"surface": standard["surface"]}} if e["type"] == "session_start"
            else {**e, "layout": standard["layout"]} if e["type"] == "bench_setup" else e for e in events]


def test_only_the_standard_surface_and_layout_count():
    events = paired(14)
    assert len(scorecard.off_standard(events)) == 2
    assert scorecard.off_standard(on_standard(events)) == []
    carrying = [{**e, "carry": ["S13", "S1"]} if e["type"] == "bench_setup" else e for e in on_standard(events)]
    assert ["carries between two spots, not a walk of the layout"] == scorecard.off_standard(carrying)
    simulated = [{**e, "sim": True} if e["type"] == "session_start" else e for e in on_standard(events)]
    assert scorecard.off_standard(simulated) == ["a simulated run (--sim)"]
    text = "\n".join(scorecard.report(events))
    assert "gates don't apply" in text and "iteration check" not in text
    text = "\n".join(scorecard.report(on_standard(paired(14, a_clean=lambda i: i > 2))))
    assert "iteration check: AHEAD (3-0" in text and "play-ready, b: not yet (14/14" in text


@pytest.mark.parametrize("a_misses, b_misses, rounds, expected", [
    (3, 0, 13, "too few"),
    (2, 0, 14, "ahead"),
    (0, 3, 14, "drop"),
    (0, 0, 14, "another round"),
    (0, 2, 14, "another round"),
    (0, 0, 28, "keep baseline"),
    (1, 0, 28, "ahead"),
])
def test_the_iteration_check(a_misses, b_misses, rounds, expected):
    events = paired(rounds, a_clean=lambda i: i >= a_misses, b_clean=lambda i: i >= b_misses)
    assert scorecard.iteration_check(events, "a", "b")["result"] == expected


def one_policy(rounds, spots, misses=()):
    events, n = [], 0
    for _ in range(rounds):
        for spot in spots:
            events.append(attempt(n, "a", spot))
            if n in misses:
                events.append(bump(n))
            n += 1
    return events


def test_play_ready_needs_the_count_the_attempts_and_no_repeated_spot():
    spots = [f"S{i}" for i in range(14)]
    assert scorecard.play_ready(one_policy(2, spots, misses={0, 1}), "a")["ok"]                    # 26/28, two spots
    assert not scorecard.play_ready(one_policy(2, spots, misses={0, 1, 2}), "a")["ok"]             # 25/28
    repeated = scorecard.play_ready(one_policy(2, spots, misses={0, 14}), "a")                     # 26/28, S0 twice
    assert not repeated["ok"] and repeated["repeated_spots"] == ["S0"]
    assert not scorecard.play_ready(one_policy(1, spots), "a")["ok"]                               # 14/14 is too few


def test_the_odds_match_the_docs():
    assert scorecard.ship_chance(0.8, 0.95, 28) == pytest.approx(0.33, abs=0.01)
    assert scorecard.ship_chance(0.8, 0.8, 28) == pytest.approx(0.022, abs=0.002)
    assert scorecard.play_ready_chance(0.95) == pytest.approx(0.84, abs=0.01)
    assert scorecard.play_ready_chance(0.80) == pytest.approx(0.06, abs=0.01)
    assert scorecard.iteration_chance(0.8, 0.95) > scorecard.iteration_chance(0.8, 0.8) \
        > scorecard.iteration_chance(0.8, 0.7)


def test_differing_hover_openings_are_flagged():
    events = paired(1)
    events[1]["policies"] = [{"id": "a", "hover": {"gripper": 21.0}}, {"id": "b", "hover": {"gripper": 26.7}}]
    assert "hover openings differ" in "\n".join(scorecard.report(events))
    events[1]["policies"][0]["hover"]["gripper"] = 26.6
    assert "hover openings differ" not in "\n".join(scorecard.report(events))


def test_session_clean_is_saved_and_redone_when_the_events_grow(tmp_path):
    events_file = tmp_path / "events.jsonl"
    events_file.write_text("".join(json.dumps(e) + "\n" for e in [attempt(0, "a", "S1")]))
    assert scorecard.session_clean(tmp_path) == {"a": 1}
    assert (tmp_path / "clean_counts.json").exists()
    with events_file.open("a") as f:
        f.write(json.dumps(bump(0)) + "\n")
    assert scorecard.session_clean(tmp_path) == {"a": 0}
