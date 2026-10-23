"""policy.run's stopping rules on the simulated arm, with a scripted stand-in for the policy."""
import threading

import numpy as np
import pytest

from mahjong import config, kinematics
from mahjong.arm import Arm
from mahjong.policy import HOLD_CONFIRM_S, run
from mahjong.sim import SimCamera, SimRobot

X, Y = 0.20, -0.14


class ScriptedPickup:
    """Descend (2s), close on the tile (0.5s), lift (2s), hold — as absolute commands, returned as
    deltas from the anchor run() measures. Optionally lets the tile slip at `slip_step`."""
    id = "scripted"

    def __init__(self, arm, slip_step=None):
        self.arm, self.slip_step = arm, slip_step
        hover = arm.commanded()
        down, _ = kinematics.hover_command(X, Y, 0.007, hover["gripper"])
        up, _ = kinematics.hover_command(X, Y, 0.15, 0.0)
        closed = {**down, "gripper": 0.0}
        self.path = (self._ramp(hover, down, 60) + self._ramp(down, closed, 15) + self._ramp(closed, up, 60))
        self.final = up

    @staticmethod
    def _ramp(a, b, n):
        return [{j: a[j] + (k / n) * (b[j] - a[j]) for j in config.JOINTS} for k in range(1, n + 1)]

    def reset(self):
        self.k = 0
        self.anchor = np.array([self.arm.measured()[j] for j in config.JOINTS])

    def step(self, frame, state_delta):
        target = self.path[self.k] if self.k < len(self.path) else self.final
        if target["gripper"] < 11:
            self.arm.robot.holding = True          # the jaws have closed on the tile
        if self.slip_step is not None and self.k >= self.slip_step:
            self.arm.robot.holding = False         # ...and it slips out
        self.k += 1
        return np.array([target[j] for j in config.JOINTS]) - self.anchor


def hovering_arm():
    hover, _ = kinematics.hover_command(X, Y, 0.045, gripper=90.0)
    return Arm(SimRobot(start=hover), sleep=lambda s: None)


def test_stops_once_the_tile_is_held_and_lifted(tmp_path):
    arm = hovering_arm()
    summary = run(ScriptedPickup(arm), arm, SimCamera(), tmp_path, log=lambda *a: None)
    assert summary["stop_reason"] == "done"
    assert not summary["slipped"]
    # lifted past 60mm partway through the 2s lift, then the 1.5s hold check
    assert 3.5 < summary["duration_s"] < 6.0 < config.ACT_DURATION_S
    assert summary["duration_s"] - summary["held_at_s"] >= HOLD_CONFIRM_S - 0.05
    assert np.allclose(summary["grasp_xy"], (X, Y), atol=0.002)
    assert (tmp_path / "act_trace.npz").exists()


def test_a_slip_during_the_hold_check_is_not_done(tmp_path):
    arm = hovering_arm()
    summary = run(ScriptedPickup(arm, slip_step=125), arm, SimCamera(), tmp_path, log=lambda *a: None)
    assert summary["slipped"]
    assert summary["stop_reason"] == "cap"
    assert summary["duration_s"] == config.ACT_DURATION_S


def test_esc_still_aborts(tmp_path):
    arm = hovering_arm()
    abort = threading.Event()
    policy = ScriptedPickup(arm)
    step = policy.step

    def step_then_abort(frame, state_delta):
        if policy.k == 30:
            abort.set()
        return step(frame, state_delta)

    policy.step = step_then_abort
    summary = run(policy, arm, SimCamera(), tmp_path, abort=abort, log=lambda *a: None)
    assert summary["stop_reason"] == "aborted" and summary["aborted"]
    assert summary["duration_s"] < 1.2


def test_a_mark_is_timed_and_the_policy_carries_on(tmp_path):
    arm = hovering_arm()
    mark = threading.Event()
    policy = ScriptedPickup(arm)
    step = policy.step

    def step_and_mark(frame, state_delta):
        if policy.k in (60, 90):          # pressed during the descent, and again while closing
            mark.set()
        return step(frame, state_delta)

    policy.step = step_and_mark
    summary = run(policy, arm, SimCamera(), tmp_path, mark=mark, log=lambda *a: None)
    assert summary["stop_reason"] == "done"
    assert summary["marks_s"] == [pytest.approx(61 / 30, abs=0.01), pytest.approx(91 / 30, abs=0.01)]
    assert not mark.is_set()


def test_unmarked_runs_report_no_marks(tmp_path):
    arm = hovering_arm()
    assert run(ScriptedPickup(arm), arm, SimCamera(), tmp_path, log=lambda *a: None)["marks_s"] == []


def test_every_frame_the_policy_saw_is_saved_as_video(tmp_path):
    import cv2
    arm = hovering_arm()
    summary = run(ScriptedPickup(arm), arm, SimCamera(), tmp_path, log=lambda *a: None)
    video = cv2.VideoCapture(str(tmp_path / "wrist.mp4"))
    frames = 0
    while video.read()[0]:
        frames += 1
    assert frames == summary["steps"] + 1          # + the frame of the done check it stopped on
    assert (tmp_path / "wrist_0000.jpg").exists()
