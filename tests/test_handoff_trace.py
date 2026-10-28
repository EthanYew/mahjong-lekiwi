"""handoff_trace tells the three handoff movements apart: the servos sinking under a still command, the command
moving with nothing pressed, and pan / reach changing the commanded height."""
import pytest

from mahjong import config, kinematics
from mahjong.handoff_trace import describe, summarize

STILL = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}


def hover(dz=0.0, dx=0.0):
    pose, _ = kinematics.hover_command(0.20 + dx, 0.0, 0.040 + dz, 30.0)
    return {j: float(pose[j]) for j in config.JOINTS}


def row(t, command, measured, sticks=None, buttons=()):
    return {"t": t, "command": command, "measured": measured, "sticks": {**STILL, **(sticks or {})}, "buttons": list(buttons)}


def test_a_still_command_with_a_sinking_arm_is_drift_not_a_command_move():
    rows = [row(0.0, hover(), hover(-0.005))] + [row(k / 30, hover(), hover(-0.005 - 0.001 * k)) for k in range(1, 11)]
    rows.append(row(11 / 30, hover(), hover(-0.016), {"ly": 0.5}))
    s = summarize(rows)
    assert s["idle_command_moved_mm"] == 0 and s["command_moved_untouched_mm"] == 0
    assert s["idle_drift_mm"][2] == pytest.approx(-10, abs=1.0)
    assert s["sag_at_handoff_mm"][2] == pytest.approx(-5, abs=0.5)
    assert s["first_input_s"] == pytest.approx(11 / 30, abs=0.01)
    assert any("drifted" in line for line in describe(s)) and not any("COMMAND" in line for line in describe(s))


def test_the_step_that_first_sees_input_is_not_counted_as_moving_before_input():
    """2026-10-28: 7 of 27 tiles said "the COMMAND moved ~1mm before any input" while nothing-pressed steps
    showed 0: the first input's own step was counted as idle."""
    rows = [row(0.0, hover(), hover()), row(0.1, hover(), hover()),
            row(0.2, hover(0.0, 0.0015), hover(), {"ly": 0.3})]
    s = summarize(rows)
    assert s["idle_command_moved_mm"] == 0 and s["command_moved_untouched_mm"] == 0
    assert not any("COMMAND" in line for line in describe(s))


def test_a_command_moving_with_nothing_pressed_is_flagged():
    rows = [row(0.0, hover(), hover()), row(0.1, hover(-0.004), hover(-0.004)), row(0.2, hover(-0.004), hover(-0.004))]
    s = summarize(rows)
    assert s["idle_command_moved_mm"] == pytest.approx(4, abs=0.3)
    assert s["command_moved_untouched_mm"] == pytest.approx(4, abs=0.3)
    assert any("COMMAND moved" in line for line in describe(s))


def test_height_changed_by_a_reach_push_is_counted_and_a_height_push_is_not():
    rows = [row(0.0, hover(), hover()),
            row(0.1, hover(-0.003, 0.005), hover(-0.003, 0.005), {"ly": 1.0}),       # reach out, and it dropped 3mm
            row(0.2, hover(-0.008, 0.005), hover(-0.008, 0.005), {"ry": -1.0})]      # asked to go down
    s = summarize(rows)
    dropped = (kinematics.tool_point(rows[1]["command"])[2] - kinematics.tool_point(rows[0]["command"])[2]) * 1000
    assert dropped < -2 and s["height_change_on_pan_reach_mm"] == pytest.approx(dropped, abs=0.1)
    assert s["command_moved_untouched_mm"] == 0


def test_without_input_readings_the_input_numbers_are_unknown():
    rows = [{"t": 0.0, "command": hover(), "measured": hover()}, {"t": 0.1, "command": hover(), "measured": hover(-0.002)}]
    s = summarize(rows)
    assert s["first_input_s"] is None and s["command_moved_untouched_mm"] is None
    assert s["idle_drift_mm"][2] == pytest.approx(-2, abs=0.3)
