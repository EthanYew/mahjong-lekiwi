"""The offline gates on real policies: v2 (picked up 9/9 on hardware) passes them, v1 (froze 3 of 5) fails.
Slow: builds the held-out episodes of data/eval_split.json once (cached), then runs both on the GPU.

    pytest -m slow tests/test_offline_eval_slow.py
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

from mahjong import offline_eval

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def split():
    split = offline_eval.load_split()
    held = split.get("heldout", {}).get("Yewy/mahjong_mat_pickup_lekiwi_cl_hover")
    if not held:
        pytest.skip("no held-out split for v2's recording")
    return {"heldout": {"Yewy/mahjong_mat_pickup_lekiwi_cl_hover": held}, "eval_sets": []}


def rows(policy_id, split):
    from mahjong.policy import Policy
    return offline_eval.evaluate(Policy(policy_id), split, log=lambda *_: None)


def test_v2_passes_and_v1_fails_the_gates(split):
    v2, v1 = rows("pickup_mat_cl", split), rows("pickup_mat_cl_v1", split)
    assert all(ok for ok, _ in offline_eval.gates(v2, v1)), offline_eval.gates(v2, v1)
    failed = [m for ok, m in offline_eval.gates(v1) if not ok]
    assert any("freezes" in m for m in failed)
