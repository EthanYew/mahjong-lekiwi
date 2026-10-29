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
    placed = arm.place(x, y, approach_offset_m=config.HOVER_OFFSET_M, log=lambda *a: None)
    assert placed["released_at_m"] == pytest.approx(PLACE_TIP_CLEARANCE_M, abs=CLEARANCE_TOLERANCE_M)
    assert placed["shift_m"] == 0 and placed["xy"] == pytest.approx([x, y])
    assert_no_jumps(robot, start)
    grips = [c["gripper"] for c in robot.sent]
    release = next(i for i, g in enumerate(grips) if g > 0)
    assert all(g == 0.0 for g in grips[:release])
    # opened just past the jaws' reading on the tile, not to the wide GRIPPER_RELEASE
    assert max(grips) == pytest.approx(placed["release_gripper"])
    assert placed["release_gripper"] <= config.GRIPPER_RELEASE
    # nothing lower than the set-down height before the release
    assert min(robot.clearances[:release]) >= (PLACE_TIP_CLEARANCE_M - CLEARANCE_TOLERANCE_M) * 1000
    assert lowest_tile_near_base(robot) > 0.040
    assert arm.is_at_rest()


@pytest.mark.parametrize("grasp_xy", [(0.249, 0.196), (0.253, 0.200)])
def test_a_grasp_beyond_reach_is_returned_a_little_closer(grasp_xy):
    """2026-10-23: the policy closed on 4D at these points, past the IK's reach; both returns refused."""
    x, y = grasp_xy
    start = {**config.REST_POSE, "gripper": 0.0}
    arm, robot = make_arm(start=start, holding=True)
    placed = arm.place(x, y, approach_offset_m=0.050, log=lambda *a: None)
    assert 0 < placed["shift_m"] <= 0.015
    px, py = placed["xy"]
    assert np.hypot(px - x, py - y) == pytest.approx(placed["shift_m"], abs=1e-6)
    assert np.hypot(px - kinematics.SHOULDER_PAN_X, py) < np.hypot(x - kinematics.SHOULDER_PAN_X, y)
    assert placed["released_at_m"] == pytest.approx(PLACE_TIP_CLEARANCE_M, abs=CLEARANCE_TOLERANCE_M)
    assert_no_jumps(robot, start)
    assert arm.is_at_rest()


def test_a_return_far_out_of_reach_does_not_move():
    start = {**config.REST_POSE, "gripper": 0.0}
    arm, robot = make_arm(start=start, holding=True)
    assert arm.place(0.40, 0.25, approach_offset_m=0.050, log=lambda *a: None) is None
    assert robot.sent == []


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


class ParkingBus:
    """The sim bus plus disable_torque, like the real one, after which the arm sags off REST_POSE."""
    def __init__(self, robot, sag):
        self.robot, self.sag, self.torque = robot, sag, True
        self.sync_read = lambda register: ({**robot.bus_read(register), **self.sag} if not self.torque else robot.bus_read(register))

    def disable_torque(self):
        self.torque = False


def test_moved_while_off_compares_with_where_the_arm_settled_not_rest(tmp_path, monkeypatch):
    """2026-10-26: every session logged 'arm moved while off' at 8°, because the unpowered arm sags off
    REST_POSE onto the same stops each time."""
    from mahjong import arm as arm_module
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    arm, robot = make_arm(start=config.REST_POSE)
    sagged = {"shoulder_lift": config.REST_POSE["shoulder_lift"] + 8.0}
    robot.bus_read = robot.bus.sync_read
    robot.bus = ParkingBus(robot, sagged)
    robot.disconnect = lambda: None
    assert arm_module.connect_state(dict(config.REST_POSE, **sagged))["arm_at_rest"] is False   # no record yet: vs rest
    arm.disconnect()
    parked = arm_module.load_parked()
    assert parked["shoulder_lift"] == pytest.approx(sagged["shoulder_lift"])
    found = arm_module.connect_state(dict(config.REST_POSE, **sagged), parked)
    assert found["arm_at_rest"] is True and found["off_parked_deg"] < 1.0 and found["off_rest_deg"] == pytest.approx(8.0)
    moved = arm_module.connect_state(dict(config.REST_POSE, **sagged, shoulder_pan=config.REST_POSE["shoulder_pan"] + 20), parked)
    assert moved["arm_at_rest"] is False and moved["off_parked_deg"] == pytest.approx(20.0, abs=1.0)


def test_disconnect_without_a_torque_bus_records_nothing(tmp_path, monkeypatch):
    from mahjong import arm as arm_module
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    arm, robot = make_arm()
    robot.disconnect = lambda: None
    arm.disconnect()
    assert arm_module.load_parked() is None


def test_an_arm_found_where_it_settled_unpowered_doesnt_move_at_startup():
    """2026-10-27: connect held the sagged pose, so the first rest() lifted the arm via RAISED_REST and
    folded it back down. REST_POSE is now where the unpowered arm settles (sessions/parked_pose.json)."""
    settled = {"shoulder_pan": -90.64, "shoulder_lift": -101.89, "elbow_flex": 97.63, "wrist_flex": 67.43,
               "wrist_roll": -99.12, "gripper": 26.76}
    arm, robot = make_arm(start=settled)
    arm.hold_measured()
    robot.sent.clear()
    arm.rest()
    assert robot.sent == []


@pytest.mark.parametrize("tile", TILES)
def test_after_a_release_the_open_jaws_rise_straight_up_clear_of_the_tile(tile):
    """2026-10-27: one joint move from the release to the approach swung the open jaws through the tile."""
    from mahjong.arm import RELEASE_RISE_M
    x, y = TILES[tile]
    arm, robot = make_arm(start={**config.REST_POSE, "gripper": 0.0}, holding=True)
    arm.place(x, y, approach_offset_m=config.HOVER_OFFSET_M, log=lambda *a: None)
    release = next(i for i, c in enumerate(robot.sent) if c["gripper"] > 0)
    tips = [kinematics.pincer_tip(c) for c in robot.sent[release:]]
    clear = next(i for i, t in enumerate(tips) if t[2] - tips[0][2] >= RELEASE_RISE_M - 0.001)
    drift = max(np.hypot(*(t[:2] - tips[0][:2])) for t in tips[:clear + 1])
    assert drift < 0.003, f"the jaws moved {drift * 1000:.1f}mm sideways before clearing the tile"


def test_lift_never_lowers_an_arm_already_commanded_higher():
    """After a return's approach (commanded ~95mm) a sagging far-reach arm measures under LIFT_CLEARANCE_M;
    lift() re-commanded 80mm, lowering it before the swing to rest."""
    from mahjong.arm import LIFT_CLEARANCE_M
    q, _ = kinematics.hover_command(0.272, -0.075, 0.095, 17.0)
    arm, robot = make_arm(start=q)
    robot._measured_offset = {j: hi for j, (lo, hi) in robot._band().items()}      # arrived from below
    assert arm.tip_clearance()[0] < LIFT_CLEARANCE_M
    arm.lift()
    assert robot.sent == []


@pytest.mark.parametrize("tile", ["3B row 1 centre", "9B row 2 right"])
def test_a_return_logs_its_rise_and_goes_higher_over_the_tile_before_swinging_if_low(tile, monkeypatch):
    from mahjong import arm as arm_module
    x, y = TILES[tile]
    arm, robot = make_arm(start={**config.REST_POSE, "gripper": 0.0}, holding=True)
    rise = arm.place(x, y, approach_offset_m=config.HOVER_OFFSET_M, log=lambda *a: None)["rise"]
    assert 0.015 < rise["up_m"] <= arm_module.RELEASE_RISE_M + 0.001 and rise["sideways_m"] < 0.01
    assert rise["swing_from_m"] >= arm_module.SWING_CLEARANCE_M

    monkeypatch.setattr(arm_module, "SWING_CLEARANCE_M", rise["swing_from_m"] + 0.01)     # say it's too low
    arm, robot = make_arm(start={**config.REST_POSE, "gripper": 0.0}, holding=True)
    logs = []
    placed = arm.place(x, y, approach_offset_m=config.HOVER_OFFSET_M, log=logs.append)
    assert any("raising before the swing" in line for line in logs)
    assert placed["rise"]["swing_from_m"] >= arm_module.SWING_CLEARANCE_M - CLEARANCE_TOLERANCE_M
    release = next(i for i, c in enumerate(robot.sent) if c["gripper"] > 0)
    swing = next(i for i, c in enumerate(robot.sent) if i > release and
                 np.hypot(*(kinematics.tool_point(c)[:2] - np.array([x, y]))) > 0.003)
    assert robot.clearances[swing - 1] >= (arm_module.SWING_CLEARANCE_M - CLEARANCE_TOLERANCE_M) * 1000


def test_the_release_height_is_just_above_where_the_jaws_closed():
    from mahjong.arm import RELEASE_ABOVE_GRASP_M, RELEASE_MAX_M, RELEASE_MIN_M, release_height
    assert release_height(None) == PLACE_TIP_CLEARANCE_M
    assert release_height(0.004) == pytest.approx(0.004 + RELEASE_ABOVE_GRASP_M)
    assert release_height(-0.01) == RELEASE_MIN_M and release_height(0.05) == RELEASE_MAX_M


def test_grasp_clearance_is_the_measured_tip_as_the_jaws_were_commanded_shut():
    hover, _ = kinematics.hover_command(0.2, 0.0, 0.040, 30.0)
    low, _ = kinematics.hover_command(0.2, 0.0, 0.004, 30.0)
    states = [[hover[j] for j in config.JOINTS], [low[j] for j in config.JOINTS], [low[j] for j in config.JOINTS]]
    actions = [s[:5] + [30.0] for s in states[:2]] + [states[2][:5] + [0.0]]
    assert kinematics.grasp_clearance(states, actions) == pytest.approx(kinematics.tip_clearance(low))
    assert kinematics.grasp_clearance(states, [a[:5] + [30.0] for a in actions]) is None


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("closed_at", [-0.002, 0.001, 0.006, 0.013])
def test_a_tile_is_let_go_just_above_where_its_jaws_closed_never_below(tile, closed_at):
    """2026-10-29: a fixed 10mm release let go of v4's tiles 3-9mm above where they had been held; they dropped
    and tilted. The set-down comes down to the grasp's own height and doesn't overshoot below it."""
    from mahjong.arm import PLACE_TOLERANCE_M, release_height
    x, y = TILES[tile]
    arm, robot = make_arm(start={**config.REST_POSE, "gripper": 0.0}, holding=True)
    placed = arm.place(x, y, approach_offset_m=config.HOVER_OFFSET_M, log=lambda *a: None, grasp_clearance_m=closed_at)
    target = release_height(closed_at)
    assert placed["release_target_m"] == target
    assert target - 0.0002 <= placed["released_at_m"] <= target + PLACE_TOLERANCE_M
    release = next(i for i, c in enumerate(robot.sent) if c["gripper"] > 0)
    assert min(robot.clearances[:release]) >= (target - 0.0002) * 1000          # never under it on the way down
    assert_no_jumps(robot, {**config.REST_POSE, "gripper": 0.0})
