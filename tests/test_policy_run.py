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



class FailingCamera(SimCamera):
    """A wrist camera whose reads number n (1-based) fail where `fails(n)` (it dropped out, 2026-10-28)."""

    def __init__(self, fails):
        self.reads, self.fails = 0, fails

    def read(self):
        self.reads += 1
        return (False, None) if self.fails(self.reads) else super().read()


def test_a_camera_that_stops_giving_frames_stops_the_attempt(tmp_path):
    arm = hovering_arm()
    camera = FailingCamera(lambda n: n > 40)
    summary = run(ScriptedPickup(arm), arm, camera, tmp_path, log=lambda *a: None)
    assert summary["stop_reason"] == "camera" and not summary["aborted"]
    assert summary["steps"] == 40
    assert camera.reads == 40 + config.FPS             # a second of failed reads, not the 16s cap


def test_a_dropped_frame_or_two_carries_on(tmp_path):
    arm = hovering_arm()
    summary = run(ScriptedPickup(arm), arm, FailingCamera(lambda n: n in (50, 51)), tmp_path, log=lambda *a: None)
    assert summary["stop_reason"] == "done"


# --- a person takes over (mahjong/takeover.py) ---

class Stuck:
    """A policy that never leaves the hover: the failure a person steps in on."""
    id = "stuck"

    def __init__(self):
        self.calls = 0

    def reset(self):
        pass

    def step(self, frame, state_delta):
        self.calls += 1
        return np.zeros(6)


class Pad:
    """A gamepad whose A reads pressed at its `presses` (calls to takeover_pressed, 1-based), and whose sticks drive the
    scripted pickup from wherever the takeover found the arm."""

    def __init__(self, arm, presses=(10,)):
        self.arm, self.presses, self.calls, self.starts, self.path = arm, set(presses), 0, 0, None

    def arm_takeover(self):
        pass

    def takeover_pressed(self):
        self.calls += 1
        return self.calls in self.presses

    def start(self, arm):
        self.starts += 1
        self.pickup = ScriptedPickup(arm)
        self.pickup.reset()

    def step(self, arm, dt):
        delta = self.pickup.step(None, None)
        command = {j: float(self.pickup.anchor[i] + delta[i]) for i, j in enumerate(config.JOINTS)}
        arm.send(command)
        return command


def take(arm, presses=(10,)):
    from test_record import FakeDataset

    from mahjong.takeover import Takeover
    return Takeover(Pad(arm, presses), "Yewy/test_takeover", dataset=FakeDataset("/nonexistent"), log=lambda *a: None)


def test_a_takeover_hands_the_gamepad_the_arm_and_keeps_every_frame(tmp_path):
    arm = hovering_arm()
    policy, takeover = Stuck(), take(arm)
    summary = run(policy, arm, SimCamera(), tmp_path, takeover=takeover, log=lambda *a: None)
    assert summary["takeover_step"] == 9 and summary["takeover_end"] == "done"     # the 10th check is step 9
    assert summary["stop_reason"] == "done" and takeover.gamepad.starts == 1
    assert policy.calls == 9                       # the policy drove until the takeover, never after
    assert len(takeover.frames) == summary["steps"]
    assert summary["policy_grasp_xy"] is None      # the stuck policy never closed: the grasp is the person's
    assert np.allclose(summary["grasp_xy"], (X, Y), atol=0.002)
    assert int(np.load(tmp_path / "act_trace.npz")["takeover_step"]) == 9


def test_a_takeover_outlasts_the_policys_cap(tmp_path):
    arm = hovering_arm()
    late = int(config.ACT_DURATION_S * config.FPS) - 5          # A just before the 16s cap
    summary = run(Stuck(), arm, SimCamera(), tmp_path, takeover=take(arm, presses=(late,)), log=lambda *a: None)
    assert summary["takeover_end"] == "done" and summary["duration_s"] > config.ACT_DURATION_S


def test_a_again_ends_the_recovery_and_esc_throws_it_away(tmp_path):
    arm = hovering_arm()
    summary = run(Stuck(), arm, SimCamera(), tmp_path, takeover=take(arm, presses=(10, 40)), log=lambda *a: None)
    assert summary["stop_reason"] == summary["takeover_end"] == "pressed" and summary["steps"] == 39

    arm = hovering_arm()
    abort = threading.Event()
    takeover = take(arm)
    step = takeover.step

    def step_then_esc(arm_, dt):
        if len(takeover.frames) == 30:
            abort.set()
        return step(arm_, dt)

    takeover.step = step_then_esc
    summary = run(Stuck(), arm, SimCamera(), tmp_path, abort=abort, takeover=takeover, log=lambda *a: None)
    assert summary["aborted"] and summary["takeover_end"] == "discarded"


def test_no_press_runs_the_policy_as_before(tmp_path):
    arm = hovering_arm()
    summary = run(ScriptedPickup(arm), arm, SimCamera(), tmp_path, takeover=take(arm, presses=()), log=lambda *a: None)
    assert summary["stop_reason"] == "done" and summary["takeover_step"] is None
    assert np.allclose(summary["grasp_xy"], (X, Y), atol=0.002)
