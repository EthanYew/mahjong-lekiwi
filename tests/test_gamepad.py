"""Gamepad takeover and stick gating, with a scripted pad driving XLeRobot's real EETeleop."""
import pytest

from mahjong import config, kinematics
from mahjong.arm import Arm
from mahjong.gamepad import CommittedSticks, GamepadTeleop, _load_xlerobot
from mahjong.sim import SimRobot

DT = 1 / 30


class ScriptedPad:
    def __init__(self):
        self.values = {}

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
    teleop.sticks = CommittedSticks(pad, commit_s, teleop.utils.STICK_CODES)
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
