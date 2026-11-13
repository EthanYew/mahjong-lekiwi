"""Offline evaluation without a GPU: the per-episode scores on a synthetic pickup with stand-in policies
(an oracle that plans the demo's own future, a frozen one, one with a roll offset), the held-out
split, labels and gates."""
import numpy as np
import pytest

from mahjong import config, offline_eval
from test_dataset import episode

CHUNK = 100


def built_episode():
    """(states, actions, actions_raw) as a built dataset stores them: deltas from the first state."""
    states_raw, actions_raw, _ = episode(n=240)
    return states_raw - states_raw[0], actions_raw - states_raw[0], actions_raw


def oracle(actions, roll_offset=0.0):
    def plan(i, state):
        chunk = actions[i:i + CHUNK]
        chunk = np.vstack([chunk, np.repeat(chunk[-1:], CHUNK - len(chunk), axis=0)])
        return chunk + np.array([0, 0, 0, 0, roll_offset, 0])
    return plan


def frozen(i, state):
    return np.repeat(np.asarray(state)[None], CHUNK, axis=0)


def score(plan, ep):
    states, actions, actions_raw = ep
    return offline_eval.score_episode(plan, lambda i: i, states, actions, actions_raw)


def test_an_oracle_scores_zero():
    ep = built_episode()
    s = score(oracle(ep[1]), ep)
    assert s["arm_mae"] < 1e-6 and s["grip_mae"] < 1e-6
    assert s["close_err_s"] == 0 and not s["never_closes"]
    assert not s["freeze"] and s["onset"] == s["demo_onset"] is not None
    assert s["grasp_mm"] < 1e-6 and abs(s["roll_err_deg"]) < 1e-6 and s["plan_closes"]


def test_a_frozen_policy_is_flagged():
    ep = built_episode()
    s = score(frozen, ep)
    assert s["freeze"] and s["onset"] is None and s["never_closes"] and s["grasp_mm"] > 5.0
    # Teacher-forced error barely sees it: fed the demo's state, holding it is only the command-state lag
    # off. That's why the open-loop probes (descent onset, close, grasp point) exist.
    assert s["arm_mae"] < 0.5


def test_a_roll_offset_shows_in_the_roll_error_not_the_grasp_point():
    ep = built_episode()
    s = score(oracle(ep[1], roll_offset=3.0), ep)
    assert abs(s["roll_err_deg"] - 3.0) < 1e-6
    assert s["grasp_mm"] < 3.0                   # the tool point is only slightly off the roll axis
    assert abs(s["arm_mae"] - 3.0 / 5) < 1e-6


def test_descent_onset():
    joints = np.zeros((50, 6))
    joints[20:, 1] = np.linspace(0, 10, 30)
    assert offline_eval.descent_onset(joints) == 26
    assert offline_eval.descent_onset(np.zeros((50, 6))) is None


def records_at(spots):
    """Log records: {episode: {detection_xy}} with n episodes at each (x, y), within 3mm of it."""
    rng = np.random.default_rng(0)
    records, e = {}, 0
    for (x, y), n in spots:
        for _ in range(n):
            records[e] = {"detection_xy": [x + rng.uniform(-0.003, 0.003), y + rng.uniform(-0.003, 0.003)]}
            e += 1
    return records


def test_heldout_covers_every_spot_with_enough_episodes_and_is_repeatable():
    spots = [((0.15, 0.01), 16), ((0.15, 0.07), 9), ((0.22, -0.10), 6), ((0.22, 0.02), 5), ((0.15, -0.13), 2)]
    records = records_at(spots)
    chosen = offline_eval.choose_heldout(records)
    assert chosen == offline_eval.choose_heldout(records)                   # same seed, same split
    assert chosen != offline_eval.choose_heldout(records, seed=1)
    labels, _ = offline_eval.position_clusters([records[e]["detection_xy"] for e in sorted(records)])
    clusters = {labels[e] for e in chosen}
    assert clusters == {0, 1, 2, 3}                                         # the 2-episode spot keeps both
    assert len(chosen) == round(0.10 * len(records))


def test_eval_sets_are_never_trained_on():
    split = {"heldout": {"Yewy/train": [3, 1]}, "eval_sets": ["Yewy/eval_v1"]}
    assert offline_eval.excluded_episodes("Yewy/train", split) == [1, 3]
    assert offline_eval.excluded_episodes("Yewy/other", split) == []
    with pytest.raises(SystemExit):
        offline_eval.excluded_episodes("Yewy/eval_v1", split)


def test_labels_from_log_records():
    record = {"slice": "dim", "spot": "F2", "tile_name": "3B", "detection_xy": [0.151, 0.014],
              "conditions": {"daylight": "none", "lamp": False, "ceiling_lights": False}}
    assert offline_eval.labels(record, "eval") == {"slice": "dim", "spot": "F2", "tile": "3B",
                                                   "lighting": "daylight none, lamp off, ceiling off",
                                                   "surface": "black table top"}
    old = {"tile_name": "3B", "detection_xy": [0.151, 0.014]}          # recorded before slices and conditions
    assert offline_eval.labels(old, "heldout") == {"slice": "heldout", "spot": "15,+1", "tile": "3B",
                                                   "lighting": "unrecorded", "surface": "black table top"}
    assert offline_eval.labels({**old, "slice": "train"}, "heldout")["slice"] == "heldout"


def row(slice_, arm, freeze=False, grasp=2.0):
    return {"slice": slice_, "spot": "s", "lighting": "l", "arm_mae": arm, "grip_mae": 1.0, "close_err_s": 0.1,
            "never_closes": False, "freeze": freeze, "onset": 30, "demo_onset": 28, "grasp_mm": grasp, "roll_err_deg": 1.0}


def test_gates_catch_freezes_and_regressions_but_not_on_tilt():
    reference = [row("familiar", 0.5), row("new", 0.8), row("tilt", 1.0)]
    good = [row("familiar", 0.55), row("new", 0.9), row("tilt", 5.0, freeze=True)]
    assert all(ok for ok, _ in offline_eval.gates(good, reference))
    worse = [row("familiar", 0.55), row("new", 1.5), row("tilt", 1.0)]
    failed = [m for ok, m in offline_eval.gates(worse, reference) if not ok]
    assert len(failed) == 1 and failed[0].startswith("new: arm error")
    frozen_rows = [row("familiar", 0.5, freeze=True)]
    assert [m for ok, m in offline_eval.gates(frozen_rows) if not ok] == ["familiar: 1 freezes"]
    far = [row("familiar", 0.5, grasp=9.0)]
    assert not all(ok for ok, _ in offline_eval.gates(far, [row("familiar", 0.5)]))


def test_gates_catch_a_late_descent_or_close_without_a_reference():
    late = [{**row("familiar", 0.5), "onset": 50, "demo_onset": 28}]           # v1 lagged ~21 steps
    assert [m for ok, m in offline_eval.gates(late) if not ok] == [
        "familiar: descent starts +22 steps from the demo's (limit +10)"]
    slow_close = [{**row("familiar", 0.5), "close_err_s": 0.4}]
    assert [m for ok, m in offline_eval.gates(slow_close) if not ok][0].startswith("familiar: closes 0.40s")


def test_summary_table_prints_every_slice():
    rows = [row("familiar", 0.5), row("familiar", 0.7), row("dim", 1.2, freeze=True)]
    summary = offline_eval.summarise(rows)
    assert summary["familiar"]["episodes"] == 2 and abs(summary["familiar"]["arm_mae"] - 0.6) < 1e-9
    assert summary["dim"]["freezes"] == 1 and summary["familiar"]["onset_lag"] == 2
    text = offline_eval.table(summary, "v2")
    assert "familiar" in text and "dim" in text
    blank = offline_eval.summarise([{**row("dim", 1.0, freeze=True), "onset": None}])
    lines = offline_eval.table(blank, "t").splitlines()
    assert len(lines[1]) == len(lines[2])                                   # a missing value keeps the columns


def test_episodes_are_labelled_by_the_surface_they_were_recorded_on():
    green = offline_eval.labels({"conditions": {"surface": "green cutting mat"}}, "heldout")
    assert green["surface"] == "green cutting mat"
    assert offline_eval.labels({"conditions": {"daylight": "bright"}}, "heldout")["surface"] == config.SURFACE_BEFORE_ASKED
    assert offline_eval.labels({}, "heldout")["surface"] == config.SURFACE_BEFORE_ASKED
