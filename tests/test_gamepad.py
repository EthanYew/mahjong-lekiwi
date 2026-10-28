"""Gamepad takeover and stick gating, with a scripted pad driving XLeRobot's real EETeleop."""
import pytest

from mahjong import config, kinematics
from mahjong.arm import Arm
from mahjong.gamepad import EXCLUSIVE_AXES, CommittedSticks, GamepadTeleop, _load_xlerobot
from mahjong.sim import SimRobot

DT = 1 / 30


class ScriptedPad:
    def __init__(self):
        self.values, self.keys, self.hat = {}, [], (0, 0)
        self.device = type("Device", (), {"active_keys": lambda _: list(self.keys)})()

    def dpad(self):
        return self.hat

    def stick(self, name):
        return self.values.get(name, 0.0)

    def pressed(self):
        return set()

    def just_pressed(self, buttons):
        return set()


def teleop_with(pad, commit_s=0.1):
    teleop = GamepadTeleop.__new__(GamepadTeleop)
    teleop.utils, teleop.ee = _load_xlerobot()
    teleop.pad = pad
    teleop.sticks = CommittedSticks(pad, commit_s, teleop.utils.STICK_CODES, exclusive=EXCLUSIVE_AXES)
    teleop.locked_buttons, teleop.speed_index = set(), teleop.ee.DEFAULT_SPEED_INDEX
    return teleop


def hovering_arm():
    command, _ = kinematics.hover_command(0.20, -0.14, 0.05, gripper=90.0)
    return Arm(SimRobot(start=command), sleep=lambda s: None), command


def test_takeover_with_idle_sticks_keeps_the_command():
    """Seeding from the measured (sagged) pose re-commanded the arm lower at every takeover."""
    arm, command = hovering_arm()
    teleop = teleop_with(ScriptedPad())
    teleop.start(arm)
    for _ in range(10):
        sent = teleop.step(arm, DT)
    for joint in config.JOINTS:
        assert sent[joint] == pytest.approx(command[joint], abs=1e-6)


def test_a_flick_shorter_than_the_commit_time_does_nothing():
    pad = ScriptedPad()
    sticks = CommittedSticks(pad, 0.1, ["ly"])
    pad.values["ly"] = 1.0
    for _ in range(3):              # 100ms
        sticks.tick(DT)
        assert sticks.stick("ly") == 0.0
    for _ in range(4):
        sticks.tick(DT)
    assert sticks.stick("ly") > 0.5
    pad.values["ly"] = 0.0
    sticks.tick(DT)
    assert sticks.stick("ly") == 0.0   # release stops at once


def test_pulling_the_right_stick_down_never_rolls_the_wrist():
    """2026-10-27: a descent's pull leaked into rx and turned the wrist mid-episode."""
    arm, command = hovering_arm()
    pad = ScriptedPad()
    teleop = teleop_with(pad)
    teleop.start(arm)
    pad.values.update(ry=-1.0)
    for _ in range(5):
        sent = teleop.step(arm, DT)
    pad.values.update(rx=0.6, ry=-0.7)         # thumb drifts sideways mid-descent
    for _ in range(20):
        sent = teleop.step(arm, DT)
    assert sent["wrist_roll"] == pytest.approx(command["wrist_roll"], abs=1e-6)
    assert kinematics.tool_point(sent)[2] < kinematics.tool_point(command)[2] - 0.005    # still descending


def test_the_other_axis_takes_the_stick_only_after_it_returns_to_centre():
    pad = ScriptedPad()
    sticks = CommittedSticks(pad, 0.0, ["rx", "ry"], exclusive=[("rx", "ry")])
    pad.values.update(rx=0.8, ry=-0.3)         # starts mostly sideways: roll owns it
    sticks.tick(DT)
    assert (sticks.stick("rx"), sticks.stick("ry")) == (0.8, 0.0)
    pad.values.update(rx=0.2, ry=-0.9)         # still held off-centre: roll keeps it
    sticks.tick(DT)
    assert (sticks.stick("rx"), sticks.stick("ry")) == (0.2, 0.0)
    pad.values.update(rx=0.0, ry=0.0)
    sticks.tick(DT)
    pad.values.update(rx=0.1, ry=-0.9)         # released and pushed down: height owns it
    sticks.tick(DT)
    assert (sticks.stick("rx"), sticks.stick("ry")) == (0.0, -0.9)


def test_touched_on_any_stick_or_button_at_once_but_not_a_locked_one():
    pad = ScriptedPad()
    teleop = teleop_with(pad)
    assert not teleop.touched()
    pad.values["ly"] = 0.3                       # no commit delay: a DAgger takeover starts on the first push
    assert teleop.touched()
    pad.values.clear()
    pad.pressed = lambda: {"zl"}
    assert teleop.touched()
    teleop.locked_buttons = {"zl"}
    assert not teleop.touched()
    teleop.sticks.disabled = {"ly"}
    pad.values["ly"] = 0.3
    assert not teleop.touched()


def test_touched_on_a_face_button_or_the_dpad_too():
    """A/B/X/Y and the d-pad aren't teleop controls, but pressing one is still someone taking over."""
    from evdev import ecodes
    pad = ScriptedPad()
    teleop = teleop_with(pad)
    pad.keys = [ecodes.BTN_SOUTH]
    assert teleop.touched()
    pad.keys, pad.hat = [], (0, -1)
    assert teleop.touched()
    pad.hat, pad.keys = (0, 0), [ecodes.BTN_TL2]
    teleop.locked_buttons = {"zl"}
    assert not teleop.touched()


def test_the_far_row_is_at_the_edge_of_the_gamepad_s_reach():
    """2026-10-28: WW (base r 0.291) is past it at the tile, 1D (0.287) 3mm short; the middle has room."""
    from mahjong.gamepad import REACH_WARNING_M, reach_left_m, warn_if_at_reach
    assert reach_left_m(0.255, -0.14) < 0 < reach_left_m(0.241, 0.155) < REACH_WARNING_M
    assert reach_left_m(0.21, 0.0) > 0.05
    lines = []
    warn_if_at_reach(0.241, 0.155, log=lines.append)
    warn_if_at_reach(0.21, 0.0, log=lines.append)
    assert len(lines) == 1 and "tilt the jaws forward" in lines[0]
