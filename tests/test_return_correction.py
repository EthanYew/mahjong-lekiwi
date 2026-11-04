"""The return model: aiming a return so the tile lands at its home, and reading returns from session events."""
import numpy as np
import pytest

from mahjong.return_correction import (HAND_MOVED, MAX_CORRECTION_M, ReturnCorrection, landed_tile, landings, load_records,
                                       return_record)

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


def test_a_misread_landing_isnt_lost_to_another_copy_of_the_name_across_the_mat():
    # 20261104_195901_bench: a 2C carried to S1 read 3C 3mm from home, and a real 2C 104mm away took the name match
    tiles = [{"tile_name": "3C", "xy": (0.189, 0.139)}, {"tile_name": "2C", "xy": (0.187, 0.037)},
             {"tile_name": "2C", "xy": (0.428, -0.151)}]
    assert landed_tile(tiles, "2C", [0.192, 0.141])["tile_name"] == "3C"


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


# The real mat: what the RealSense sees and where the arm must go differ by an amount that changes with position.
def gap(xy):
    return np.array([0.040, 0.014 - 0.09 * xy[1]])


STAYS = np.array([0.7, 0.5])          # of how far a close strayed from the usual, what stays with the tile
MARGIN = np.array([0.0, 0.004])       # every close leaves this margin, on purpose


def lands_on_mat(target, det, grasp):
    stray = np.subtract(grasp, det) - gap(det) - MARGIN
    landed = np.asarray(target, dtype=float) - gap(target)      # near enough: the gap barely changes over a few mm
    return landed - STAYS * stray


def same_place_returns(n=200, seed=1, noise=0.0005):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        det = np.array([rng.uniform(0.14, 0.27), rng.uniform(-0.18, 0.18)])
        grasp = det + gap(det) + MARGIN + rng.normal(0, 0.006, 2)
        target = det + gap(det)
        out.append(return_record(det, grasp, target, lands_on_mat(target, det, grasp) + rng.normal(0, noise, 2),
                                 home_xy=list(det)))
    return out


ZONE, S4 = np.array([0.203, -0.172]), np.array([0.140, 0.091])


def test_a_tile_carried_across_the_mat_lands_at_its_new_home_where_the_old_aim_did_not():
    model = ReturnCorrection(same_place_returns())
    assert model.across is not None
    grasp = ZONE + gap(ZONE) + MARGIN
    assert np.linalg.norm(lands_on_mat(model.target(S4, ZONE, grasp, log=lambda *a: None), ZONE, grasp) - S4) < 0.003
    old = ReturnCorrection([]).target(S4, ZONE, grasp, log=lambda *a: None)      # the pickup spot's gap, carried along
    assert np.linalg.norm(lands_on_mat(old, ZONE, grasp) - S4) > 0.015


def test_a_steady_margin_at_the_close_changes_nothing_and_a_stray_moves_the_aim_by_the_share_that_stays():
    model = ReturnCorrection(same_place_returns())
    usual = ZONE + gap(ZONE) + MARGIN
    at_usual = np.array(model.target(S4, ZONE, usual, log=lambda *a: None))
    assert np.linalg.norm(at_usual - (S4 + gap(S4))) < 0.002        # the margin isn't aimed for
    strayed = np.array(model.target(S4, ZONE, usual + [0.0, 0.010], log=lambda *a: None))
    assert (strayed - at_usual)[1] == pytest.approx(0.010 * STAYS[1], abs=0.001)


def test_a_return_to_the_same_place_is_aimed_as_before():
    model = ReturnCorrection(same_place_returns())
    det, grasp = np.array([0.2, 0.05]), np.array([0.245, 0.06])
    bx, by = det + grasp - det
    dx, dy = model.correction_at(det, grasp)
    assert model.target(det, det, grasp, log=lambda *a: None) == pytest.approx((bx - dx, by - dy))


def test_too_few_returns_to_aim_a_carried_tile_says_so(capsys):
    model = ReturnCorrection(same_place_returns(n=5))
    assert model.across is None
    model.target(S4, ZONE, ZONE + gap(ZONE))
    assert "too few returns" in capsys.readouterr().out


def test_carried_returns_and_a_persons_hands_stay_out_of_the_fit(tmp_path, monkeypatch):
    import json
    from mahjong import return_correction
    fingerprint = {"follower_id": "arm", "t_cam_to_base_sha": "abc"}
    monkeypatch.setattr(return_correction, "setup_fingerprint", lambda: fingerprint)
    keep = {**return_record([0.2, 0.0], [0.24, 0.01], [0.24, 0.01], [0.2, 0.0], home_xy=[0.2, 0.0]), **fingerprint}
    across = {**keep, "home_xy": [0.14, 0.09]}
    (session, attempt) = sorted(HAND_MOVED)[0]
    by_hand = {**keep, "session": session, "attempt_dir": attempt}
    path = tmp_path / "returns.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in (keep, across, by_hand)))
    fitted, _, excluded = load_records(path)
    assert fitted == [keep] and excluded == [across, by_hand]


def test_landings_say_when_r_was_pressed_during_the_return():
    attempt = {"type": "attempt", "tile": "8D", "attempt_dir": "001", "detection_xy": [0.2, 0.0], "hover_xy": [0.23, 0.01]}
    scan = {"type": "detections", "label": "after_return", "tiles": [{"tile_name": "8D", "xy": [0.2, 0.0]}]}
    keys = {"type": "keys", "during": "return", "presses": [{"key": "r", "kind": "by_hand", "t_s": 1.0}]}
    pressed, clean = landings([attempt, keys, {"type": "return", "xy": [0.23, 0.01]}, scan,
                               attempt, {"type": "return", "xy": [0.23, 0.01]}, scan])
    assert pressed["by_hand"] and not clean["by_hand"]
