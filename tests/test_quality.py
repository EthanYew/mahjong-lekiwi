"""Pickup phases and the hover→grasp check (mahjong/quality.py) on synthetic demos built with the IK."""
import numpy as np
import pytest

from mahjong import config, kinematics, quality

FPS = 30
X, Y = 0.20, 0.0


def pose(x, y, offset, gripper):
    command, _ = kinematics.hover_command(x, y, offset, gripper)
    return np.array([command[j] for j in config.JOINTS])


def ramp(a, b, n):
    """n frames from pose a to pose b (joint space: straight enough over a few cm), b included."""
    return [a + (b - a) * k / n for k in range(1, n + 1)]


def demo(steer_y_m=0.010, pause_s=1.0, jitter_m=0.0):
    """Narrow at the hover, hold still `pause_s`, steer `steer_y_m` in y, descend, close, hold."""
    hx, hy = X, Y + jitter_m
    wide, hover = pose(hx, hy, 0.045, 90), pose(hx, hy, 0.045, 27)
    steered, low = pose(hx, hy + steer_y_m, 0.045, 27), pose(hx, hy + steer_y_m, 0.010, 27)
    shut = low.copy(); shut[5] = 0
    frames = [wide] + ramp(wide, hover, 29) + [hover] * int(pause_s * FPS)
    frames += ramp(hover, steered, 30) + ramp(steered, low, 45) + [shut] * 60
    return np.array(frames)


def test_phases_time_the_wait_the_approach_and_its_detour():
    p = quality.phases(demo(pause_s=1.0), FPS)
    assert p["narrow_s"] == pytest.approx(1.0, abs=0.1)
    assert p["wait_s"] == pytest.approx(1.0, abs=0.1)          # still at the hover before the first move
    assert p["approach_s"] == pytest.approx(2.5, abs=0.15)     # steer 1 s + descend 1.5 s
    assert p["pause_s"] == 0 and p["approach_rev"] == 0
    assert 1.05 < p["path_ratio"] < 1.5                        # steer then descend: an L, not the diagonal
    assert p["after_s"] == pytest.approx(2.0, abs=0.1) and p["wobble_mm"] < 0.2


def test_hover_move_is_the_steer_with_a_deliberate_offset_added_back():
    assert quality.hover_move(demo(steer_y_m=0.010)) == pytest.approx([0, 10], abs=0.5)
    # hover moved +6 mm on purpose, operator steered 6 mm less to the same tile: same miss of the hover
    assert quality.hover_move(demo(steer_y_m=0.004, jitter_m=0.006), (0.0, 0.006)) == pytest.approx([0, 10], abs=0.5)
    assert quality.hover_move(demo()[:50]) is None                                  # never closed


def test_drift_compares_only_demos_that_steered():
    """2026-10-26: a third of v2's demos went straight down, so medians that counted them read as a drift."""
    record = {"detection_xy": [X, Y], "spot": "F3"}
    today = {0: (demo(0.010), record), 1: (demo(0.012), record)}
    reference = {i: (demo(0.0), {"detection_xy": [X, Y]}) for i in range(3)}
    reference.update({i: (demo(0.011), {"detection_xy": [X + 0.005, Y]}) for i in range(3, 6)})
    row = quality.drift_table(today, reference).loc["F3"]
    assert row["reference"] == 6 and row["reference_straight_down"] == pytest.approx(0.5)
    assert row["shift_y"] == pytest.approx(0.0, abs=1.0)                   # the steered ones agree
    lonely = quality.drift_table(today, {0: (demo(0.0), {"detection_xy": [0.15, 0.15]})}).loc["F3"]
    assert np.isnan(lonely["shift_y"]) and lonely["reference"] == 0
