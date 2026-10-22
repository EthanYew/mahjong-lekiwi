"""Motion invariants on the simulated arm (mahjong/sim.py) — the failure modes found on hardware this
week, each as a test."""
import numpy as np
import pytest

from mahjong import config, kinematics
from mahjong.arm import Arm, PLACE_TIP_CLEARANCE_M, CLEARANCE_TOLERANCE_M
from mahjong.sim import SimRobot

TILES = {"8D row 1 right": (0.200, -0.140), "3B row 1 centre": (0.193, 0.027), "9D row 1 left": (0.172, 0.194),
         "9B row 2 right": (0.272, -0.075), "8D row 2 left": (0.252, 0.140)}
TILE_HANG_M = 0.012           # a held tile's underside below the pincer tip


def make_arm(start=None, holding=False):
    robot = SimRobot(start=start, holding=holding)
    return Arm(robot, sleep=lambda s: None), robot


def grasp_pose(x, y, holding=True):
    """Where an ACT pickup ends: lifted over the tile, gripper commanded shut."""
    command, _ = kinematics.hover_command(x, y, 0.13, gripper=0.0)
    return command


def assert_no_jumps(robot, start, max_step_deg=2.0):
    previous = start
    for command in robot.sent:
        step = max(abs(command[j] - previous[j]) for j in config.JOINTS)
        assert step <= max_step_deg, f"command jumped {step:.1f} deg"
        previous = command


def test_rest_at_rest_sends_nothing():
    arm, robot = make_arm()
    arm.rest()
    assert robot.sent == []


@pytest.mark.parametrize("tile", TILES)
def test_hover_settles_from_above_without_jumps(tile):
    x, y = TILES[tile]
    arm, robot = make_arm()
    result = arm.hover(x, y, config.HOVER_OFFSET_M, config.HOVER_CLEARANCE_M, gripper=90.0, log=lambda *a: None)
    assert result is not None
    assert_no_jumps(robot, config.REST_POSE)
    assert abs(result["clearance_m"] - config.HOVER_CLEARANCE_M) <= CLEARANCE_TOLERANCE_M
    # once the first (higher) hover has arrived, the jaws never go below the target height
    first_arrival = int(1.5 * 50) + int(3.0 * 50)      # raise from rest + move to the first hover
    assert min(robot.clearances[first_arrival:]) >= (config.HOVER_CLEARANCE_M - CLEARANCE_TOLERANCE_M) * 1000


@pytest.mark.parametrize("tile", TILES)
def test_rest_after_pickup_keeps_the_grip_and_clears_the_base(tile):
    x, y = TILES[tile]
    start = grasp_pose(x, y)
    arm, robot = make_arm(start=start, holding=True)
    arm.rest()
    assert_no_jumps(robot, start)
    assert all(c["gripper"] == 0.0 for c in robot.sent)
    assert robot.holding
    assert arm.is_at_rest()
    assert lowest_tile_near_base(robot) > 0.040


@pytest.mark.parametrize("tile", TILES)
def test_return_sets_down_at_grasp_height(tile):
    x, y = TILES[tile]
    start = {**config.REST_POSE, "gripper": 0.0}
    arm, robot = make_arm(start=start, holding=True)
    released_at = arm.place(x, y, approach_offset_m=config.HOVER_OFFSET_M, log=lambda *a: None)
    assert released_at == pytest.approx(PLACE_TIP_CLEARANCE_M, abs=CLEARANCE_TOLERANCE_M)
    assert_no_jumps(robot, start)
    grips = [c["gripper"] for c in robot.sent]
    release = next(i for i, g in enumerate(grips) if g > 0)
    assert all(g == 0.0 for g in grips[:release])
    assert max(grips) == pytest.approx(config.GRIPPER_RELEASE)
    # nothing lower than the set-down height before the release
    assert min(robot.clearances[:release]) >= (PLACE_TIP_CLEARANCE_M - CLEARANCE_TOLERANCE_M) * 1000
    assert lowest_tile_near_base(robot) > 0.040
    assert arm.is_at_rest()


def test_rest_does_not_reconfigure_at_the_folded_pose():
    """The startup/exit swing: at rest the tip is 'low' over the extrapolated table, and the IK's lift
    there is a different arm shape. A pose just off rest must go straight back, without a big swing."""
    start = {**config.REST_POSE, "shoulder_pan": config.REST_POSE["shoulder_pan"] + 5}
    arm, robot = make_arm(start=start)
    arm.rest()
    assert_no_jumps(robot, start)
    lifts = [c["shoulder_lift"] for c in robot.sent]
    assert max(lifts) <= config.RAISED_REST["shoulder_lift"] + 1


def lowest_tile_near_base(robot):
    """Lowest held-tile underside (base frame z, m) while the arm swings within 140mm of the pan axis —
    in front of the LeKiwi base, inside the nearest tiles (~150mm+ out)."""
    lowest = np.inf
    for command in robot.sent:
        tip = kinematics.pincer_tip(command)
        swinging = abs(command["shoulder_pan"] - config.REST_POSE["shoulder_pan"]) > 10
        if swinging and np.hypot(tip[0] - kinematics.SHOULDER_PAN_X, tip[1]) < 0.14:
            lowest = min(lowest, tip[2] - TILE_HANG_M)
    return lowest
