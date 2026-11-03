"""Gamepad takeover and stick gating, with a scripted pad driving XLeRobot's real EETeleop."""
import math

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


def test_only_a_fresh_press_of_a_takes_over():
    """Sticks, other buttons, and an A held since before the episode started don't (2026-10-28: drift and
    a key reading held took over at t=0)."""
    from mahjong.gamepad import TAKEOVER_KEY
    pad = ScriptedPad()
    teleop = teleop_with(pad)
    pad.values["ly"], pad.keys, pad.hat = 0.6, [304, 310], (0, 1)    # a stick, B and L, the d-pad
    teleop.arm_takeover()
    assert not teleop.takeover_pressed()
    pad.keys = [TAKEOVER_KEY]
    assert teleop.takeover_pressed()
    assert not teleop.takeover_pressed()             # still held: one press, one takeover
    teleop.arm_takeover()                            # held when the next episode starts
    assert not teleop.takeover_pressed()
    pad.keys = []
    assert not teleop.takeover_pressed()
    pad.keys = [TAKEOVER_KEY]
    assert teleop.takeover_pressed()


def test_the_far_row_is_at_the_edge_of_the_gamepad_s_reach():
    """2026-10-28: WW (base r 0.291) is past it at the tile, 1D (0.287) 3mm short; the middle has room."""
    from mahjong.gamepad import REACH_WARNING_M, reach_left_m, warn_if_at_reach
    assert reach_left_m(0.255, -0.14) < 0 < reach_left_m(0.241, 0.155) < REACH_WARNING_M
    assert reach_left_m(0.21, 0.0) > 0.05
    lines = []
    warn_if_at_reach(0.241, 0.155, log=lines.append)
    warn_if_at_reach(0.21, 0.0, log=lines.append)
    assert len(lines) == 1 and "tilt the jaws forward" in lines[0]


class RawPad(ScriptedPad):
    """A pad with raw axes, as XLeRobot's Gamepad has (before its deadzone)."""
    def _raw_axis(self, name):
        return self.values.get(name, 0.0)


def test_a_wider_deadzone_ignores_a_resting_thumb_that_xlerobots_lets_drive():
    """2026-10-28: a thumb resting on the left stick at ~0.37 of its travel pulled the arm 87mm toward the base."""
    from mahjong.gamepad import DeadzonePad
    utils, _ = _load_xlerobot()
    pad = RawPad()
    pad.values["ly"] = -0.30
    assert utils.shape_stick(0.0, -0.30)[1] < 0                      # XLeRobot's 0.15 moves the arm
    assert DeadzonePad(pad, utils.shape_stick, 0.35).stick("ly") == 0.0
    pad.values["ly"] = -1.0
    assert DeadzonePad(pad, utils.shape_stick, 0.35).stick("ly") == pytest.approx(-1.0)   # full push: full speed


def peak_jaw_speed_mm_s(values, level=None, deadzone=None, xy=(0.24, 0.0), steps=15):
    """Hold the sticks past the commit time; the fastest the commanded jaws then move."""
    from mahjong.gamepad import DeadzonePad
    utils, _ = _load_xlerobot()
    pad = RawPad()
    pad.values.update(values)
    teleop = teleop_with(DeadzonePad(pad, utils.shape_stick, deadzone) if deadzone else pad)
    if level is not None:
        teleop.speed_index = level
    command, _ = kinematics.hover_command(*xy, 0.05, gripper=90.0)
    arm = Arm(SimRobot(start=command), sleep=lambda s: None)
    teleop.start(arm)
    point, peak = kinematics.tool_point(command), 0.0
    for _ in range(steps):
        previous, point = point, kinematics.tool_point(teleop.step(arm, DT))
        peak = max(peak, math.dist(point, previous) / DT * 1000)
    return peak


@pytest.mark.parametrize("level", [0, 1, 2])
def test_no_stick_moves_the_jaws_past_its_cap_at_any_speed_setting(level):
    """Pan at 0.5x full stick is ~130mm/s at the jaws; it peaked at 102 lining up (2026-10-28)."""
    from mahjong.gamepad import HEIGHT_MAX_MM_S, LEFT_STICK_MAX_MM_S
    assert peak_jaw_speed_mm_s({"lx": 1.0}, level) <= LEFT_STICK_MAX_MM_S * 1.02
    assert peak_jaw_speed_mm_s({"ly": -1.0}, level) <= LEFT_STICK_MAX_MM_S * 1.02
    assert peak_jaw_speed_mm_s({"lx": 1.0, "ly": -1.0}, level) <= LEFT_STICK_MAX_MM_S * 1.02    # a diagonal doesn't add up
    assert peak_jaw_speed_mm_s({"ry": -1.0}, level) <= HEIGHT_MAX_MM_S * 1.02


def test_the_cap_leaves_the_descent_demos_use():
    """Demos descend at full stick at 0.5x, 50mm/s (half of all moving frames); capping that would cost time."""
    assert peak_jaw_speed_mm_s({"ry": -1.0}) == pytest.approx(50, abs=1)
    assert peak_jaw_speed_mm_s({"lx": 1.0}) == pytest.approx(30, abs=1)


def test_a_push_just_past_the_deadzone_moves_the_jaws_slowly_and_a_bigger_one_faster():
    """The point of scaling the whole travel: a clip at 30mm/s would keep pan's steep climb off the deadzone."""
    def speed(lx):
        return peak_jaw_speed_mm_s({"lx": lx}, deadzone=0.35)
    assert speed(0.40) < 3                       # 0.05 past the deadzone: under 3mm/s (~9 without the cap)
    assert speed(0.40) < speed(0.70) < speed(1.0)   # the push still sets the speed


def test_a_left_cap_of_zero_means_no_left_cap_and_the_input_line_shows_the_scaled_stick():
    from mahjong.gamepad import cap_sticks
    _, ee = _load_xlerobot()
    assert cap_sticks({"lx": 1.0, "ly": 0.0}, 0.5, 0.25, ee, left_mm_s=0)["lx"] == 1.0
    assert cap_sticks({"rx": 1.0}, 1.0, 0.25, ee)["rx"] == 1.0           # roll isn't a jaw speed
    pad = RawPad()
    teleop = teleop_with(pad)
    arm, _ = hovering_arm()
    teleop.start(arm)
    pad.values["lx"] = 1.0
    for _ in range(10):
        teleop.step(arm, DT)
    assert 0 < teleop.last["sticks"]["lx"] < 1 and teleop.last["raw"]["lx"] == 1.0


def test_a_stick_resting_off_centre_is_said_at_the_handoff_and_every_step_keeps_the_raw_reading():
    from mahjong.gamepad import DeadzonePad, input_text, moving
    utils, _ = _load_xlerobot()
    pad = RawPad()
    teleop = teleop_with(DeadzonePad(pad, utils.shape_stick, 0.35))
    teleop.deadzone = 0.35
    assert teleop.off_centre() == {}
    pad.values["ly"] = -0.30
    assert teleop.off_centre() == {"left": (0.0, -0.3)}
    arm, _ = hovering_arm()
    teleop.start(arm)
    teleop.step(arm, DT)
    assert teleop.last["raw"]["ly"] == pytest.approx(-0.30) and teleop.last["sticks"]["ly"] == 0.0
    assert not moving(teleop.last) and "raw +0.00,-0.30" in input_text(teleop.last)
    pad.values["ly"] = -0.9
    for _ in range(10):                                              # past the commit time
        teleop.step(arm, DT)
    assert moving(teleop.last) and teleop.last["sticks"]["ly"] < 0
