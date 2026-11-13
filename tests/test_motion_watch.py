"""mahjong/motion_watch.py: the checks every simulated arm command goes through, on trajectories built to break each one."""
import pytest

from mahjong import config, kinematics, motion_watch
from mahjong.arm import Arm
from mahjong.sim import SimPerception, SimRobot

pytestmark = pytest.mark.motion_unwatched        # these send bad commands on purpose; each asserts what was caught


def kinds(robot):
    return {kind for kind, _, _ in robot.violations}


def over(x, y, height_m, gripper=30.0):
    return kinematics.hover_command(x, y, height_m, gripper)[0]


def test_a_good_hover_and_pickup_breaks_nothing():
    mat = SimPerception([("8D", 0.160, -0.165), ("5C", 0.160, -0.220)])
    robot = SimRobot(on_release=mat.release)
    arm = Arm(robot, sleep=lambda s: None)
    arm.hover(0.160, -0.165, 0.04, 0.04, gripper=30.0)
    arm.move_to(over(0.160, -0.165, 0.007), 1.0)        # down onto the target: it may
    assert robot.mat is mat and robot.violations == []


def test_a_jump_is_a_move_started_from_somewhere_else():
    robot = SimRobot()
    robot.send_action({f"{j}.pos": v for j, v in config.RAISED_REST.items()})    # REST to RAISED_REST in one command
    assert "jump" in kinds(robot)


def test_a_joint_past_its_limit_and_a_roll_past_170():
    robot = SimRobot()
    robot.command.update(wrist_roll=-169.0, wrist_flex=99.0)
    robot.send_action({"wrist_roll.pos": -173.0, "wrist_flex.pos": 103.0})
    assert {"roll_limit", "joint_limit"} <= kinds(robot)


def test_the_rest_pose_and_a_taught_camera_pose_are_within_the_limits():
    """REST_POSE sits past the MuJoCo model's shoulder_lift and elbow_flex (the arm settles on its stops); poses taught on the arm
    command wrist_flex up to 100."""
    assert not motion_watch.check(None, dict(config.REST_POSE))
    assert not motion_watch.check(None, {**config.REST_POSE, "wrist_flex": 100.0})


def test_jaws_into_the_table_or_onto_a_neighbour():
    low = over(0.160, -0.165, -0.010)                   # 10mm under the grasp point: 5mm into the table
    assert "below_table" in {k for k, _ in motion_watch.check(None, low)}
    down_on_neighbour = over(0.160, -0.220, 0.002)
    found = motion_watch.check(None, down_on_neighbour, tiles=[[0.160, -0.165], [0.160, -0.220]], target=[0.160, -0.165])
    assert any(k == "onto_tile" for k, _ in found)
    assert not motion_watch.check(None, down_on_neighbour, tiles=[[0.160, -0.220]], target=[0.160, -0.220])  # its own target


def test_a_held_tile_set_down_on_another():
    onto = over(0.160, -0.220, 0.010, gripper=0.0)
    found = motion_watch.check(None, onto, holding=True, tiles=[[0.160, -0.220]])
    assert any(k == "onto_tile" and "held tile" in d for k, d in found)
    beside = over(0.160, -0.165, 0.010, gripper=0.0)    # 55mm along: the neighbour is clear
    assert not motion_watch.check(None, beside, holding=True, tiles=[[0.160, -0.220]])


def test_all_motion_goes_through_arm_py():
    """CLAUDE.md: all motion goes through mahjong/arm.py, which is what the invariants and this watchdog guard. A send_action
    anywhere else is motion they never see."""
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    calls = [f"{p.relative_to(repo)}:{n}" for d in ("mahjong", "scripts") for p in (repo / d).glob("*.py")
             if p.name not in ("arm.py", "sim.py")
             for n, line in enumerate(p.read_text().splitlines(), 1) if re.search(r"\.send_action\(", line)]
    assert not calls, f"motion outside mahjong/arm.py: {calls}"
