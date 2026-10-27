"""The return model: aiming a return so the tile lands at its home, and reading returns from session events."""
import numpy as np
import pytest

from mahjong.return_correction import (MAX_CORRECTION_M, ReturnCorrection, landed_tile, landings, return_record)

FRAME = np.array([0.035, 0.015])      # the arm's tool point sits this far from the RealSense's view of the tile
KEPT = np.array([0.6, 0.6])           # of the tile's offset in the jaws, what stays with it when set down
LEAN = np.array([0.0, 0.003])         # every landing leans this way


def lands(target, det, grasp):
    """A made-up arm: the tile lands at the set-down target, back in the RealSense frame, carrying KEPT of
    its offset in the jaws (det - grasp, minus the frame offset), plus LEAN."""
    offset_in_jaws = np.subtract(det, grasp) + FRAME
    return np.asarray(target) - FRAME + KEPT * offset_in_jaws + LEAN


def returns(n, seed=0, noise=0.001):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        det = np.array([rng.uniform(0.14, 0.27), rng.uniform(-0.18, 0.18)])
        grasp = det + FRAME + rng.normal(0, 0.008, 2)          # the policy closes off-centre
        target = grasp                                         # returned where the jaws closed
        out.append(return_record(det, grasp, target, lands(target, det, grasp) + rng.normal(0, noise, 2)))
    return out


def test_no_returns_aims_at_home_plus_the_grasp_offset():
    model = ReturnCorrection([])
    assert model.kind == "none"
    assert model.target([0.20, 0.05], [0.19, 0.04], [0.23, 0.06], log=lambda *a: None) == pytest.approx((0.24, 0.07))


def test_grip_dependent_landings_fit_and_the_target_lands_at_home():
    model = ReturnCorrection(returns(40))
    assert model.kind in ("grip", "grip+position")
    assert model.loo_rms[model.kind] < 0.5 * model.loo_rms["none"]
    rng = np.random.default_rng(7)
    for _ in range(10):
        det = np.array([rng.uniform(0.15, 0.25), rng.uniform(-0.15, 0.15)])
        home = det + rng.normal(0, 0.01, 2)                   # it crept a little since it was first seen
        grasp = det + FRAME + rng.normal(0, 0.008, 2)
        target = model.target(home, det, grasp, log=lambda *a: None)
        assert np.linalg.norm(lands(target, det, grasp) - home) < 0.002
        baseline = ReturnCorrection([]).target(home, det, grasp)
        assert np.linalg.norm(lands(baseline, det, grasp) - home) > np.linalg.norm(lands(target, det, grasp) - home)


def test_the_correction_is_capped():
    model = ReturnCorrection(returns(40))
    far_grasp = [0.20 + FRAME[0] + 0.5, FRAME[1]]
    assert np.hypot(*model.correction_at([0.20, 0.0], far_grasp)) == pytest.approx(MAX_CORRECTION_M)


def test_one_position_is_not_enough_to_fit():
    one_spot = [{**r, "detection_xy": [0.2, 0.0]} for r in returns(20)]
    model = ReturnCorrection(one_spot)
    assert model.kind == "none" and "too few positions" in model.summary()


def test_landed_tile_prefers_the_same_name_nearest_home():
    tiles = [{"tile_name": "8D", "xy": (0.30, 0.0)}, {"tile_name": "8D", "xy": (0.205, 0.0)},
             {"tile_name": "3B", "xy": (0.201, 0.0)}]
    assert landed_tile(tiles, "8D", [0.2, 0.0])["xy"] == (0.205, 0.0)
    assert landed_tile(tiles, "9D", [0.2, 0.0])["tile_name"] == "3B"         # a misread name: nearest of any
    assert landed_tile(tiles, "8D", [0.2, 0.2]) is None


def test_landings_pair_each_return_with_its_after_return_scan():
    attempt = {"type": "attempt", "tile": "8D", "attempt_dir": "001", "detection_xy": [0.2, 0.0],
               "hover_xy": [0.23, 0.01], "grasp_xy": [0.235, 0.012], "policy": "p"}
    events = [attempt,
              {"type": "return", "xy": [0.236, 0.013]},
              {"type": "detections", "label": "after_return", "tiles": [{"tile_name": "8D", "xy": [0.203, 0.004]}]},
              {**attempt, "attempt_dir": "002"},
              {"type": "return", "xy": [0.236, 0.013], "home_xy": [0.199, 0.0]},
              {"type": "detections", "label": "after_return", "tiles": []}]
    first, second = landings(events)
    assert first["home_xy"] == [0.2, 0.0] and first["landed_xy"] == [0.203, 0.004]
    assert first["error_m"] == pytest.approx(0.005)
    assert second["home_xy"] == [0.199, 0.0] and second["landed_xy"] is None
