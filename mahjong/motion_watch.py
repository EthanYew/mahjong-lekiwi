"""The motion watchdog: every command the simulated arm is sent, checked against what must stay true of anything that moves the
real one (.claude/rules/motion.md, returns.md).

`SimRobot.send_action` calls `check` with the command before and after, and keeps what it returns; `tests/conftest.py` fails any
test whose sim arm broke one. So every end-to-end sim test (bench, record, play, wall_test, crude_run, base_cam_collect...) is also
a trajectory test, and a change to motion code that merges unattended has been checked by more than its own unit tests. A test that
sends a bad command on purpose marks itself `@pytest.mark.motion_unwatched`.

What the sim can't show it doesn't claim: sag beyond its model, flex at full reach (far spots ~10-15mm lower than FK), pushes,
jaw wear. Those are the arm's to confirm.
"""
import math

from . import config, kinematics

ROLL_LIMIT_DEG = 170.0          # the wrist roll's command limit (motion.md: "the −170° command limit")
GRIPPER_RANGE = (0.0, 100.0)
# The largest step between two commands on any arm joint. Arm.move_to interpolates in small steps; a bigger one is a move started
# from somewhere else, e.g. the measured pose (motion.md, the first invariant). Calibrated on the suite as of 2026-11-13.
MAX_STEP_DEG = 6.0
TABLE_MARGIN_M = 0.001          # commanded jaw tips this far under the table plane: driving the jaws into the mat
TILE_CLEAR_M = 0.002            # a tile's top less than this under the tips (or the held tile's underside) with the tips over it
HELD_TILE_HANG_M = 0.012        # a held tile's underside below the pincer tip (tests/test_arm.py's TILE_HANG_M)
OVERLAP_SLACK_M = 0.002         # a held tile's footprint may overlap a neighbour's by this much in x or y before it counts
# Not checked here: clearing the LeKiwi base. The sim has no model of the base, and a radius rule fires on hovers over the near row;
# tests/test_arm.py checks the moves from rest (lowest_tile_near_base) and the RAISED_REST route.
# The joints may go where the unpowered arm settles on its stops (REST_POSE is past the MuJoCo model's shoulder_lift and elbow_flex).
# And the real wrist goes past the model's 95°: poses taught on the arm by gamepad command up to 100.0 (data/camera_poses.json, manual_2).
LIMITS_DEG = {j: (min(lo, config.REST_POSE[j] - 1.0), max(hi, config.REST_POSE[j] + 1.0))
              for j, (lo, hi) in kinematics.JOINT_LIMITS_DEG.items()}
LIMITS_DEG["wrist_flex"] = (LIMITS_DEG["wrist_flex"][0], 101.0)


def _inside(point_xy, tile_xy):
    """The point is over the tile's top face (its long side along x, as tiles lie on the mat)."""
    length, width = config.TILE_FACE_M
    return abs(point_xy[0] - tile_xy[0]) < length / 2 and abs(point_xy[1] - tile_xy[1]) < width / 2


def _overlaps(centre_xy, tile_xy):
    """A tile centred at centre_xy (lying as tiles do) overlaps the one at tile_xy, past the slack."""
    length, width = config.TILE_FACE_M
    return abs(centre_xy[0] - tile_xy[0]) < length - OVERLAP_SLACK_M and abs(centre_xy[1] - tile_xy[1]) < width - OVERLAP_SLACK_M


def check(previous, command, holding=False, tiles=(), target=None):
    """[(kind, detail)] for one command, [] if it's fine. previous: the command before (None for the first one); tiles: the
    [x, y] of every tile on the mat; target: the tile a descent is going for (the nearest to the tool point), which the jaws may reach.
    Open jaws are checked at the fixed jaw's tip (the lowest point); a held tile hangs under the tool point, between the jaws."""
    out = []
    for joint, (lo, hi) in LIMITS_DEG.items():
        if not lo - 0.5 <= command[joint] <= hi + 0.5:
            out.append(("joint_limit", f"{joint} {command[joint]:.1f} outside {lo:.0f}..{hi:.0f}"))
    if abs(command["wrist_roll"]) > ROLL_LIMIT_DEG + 0.5:
        out.append(("roll_limit", f"wrist_roll {command['wrist_roll']:.1f} past ±{ROLL_LIMIT_DEG:.0f}"))
    if not GRIPPER_RANGE[0] - 0.5 <= command["gripper"] <= GRIPPER_RANGE[1] + 0.5:
        out.append(("gripper_range", f"gripper {command['gripper']:.1f}"))
    if previous is not None:
        step = max(abs(command[j] - previous[j]) for j in config.JOINTS if j != "gripper")
        if step > MAX_STEP_DEG:
            joint = max((j for j in config.JOINTS if j != "gripper"), key=lambda j: abs(command[j] - previous[j]))
            out.append(("jump", f"{joint} moved {step:.1f} deg in one command"))
    tip = kinematics.pincer_tip(command)
    floor = config.table_z(tip[0], tip[1])
    if tip[2] < floor - TABLE_MARGIN_M:
        out.append(("below_table", f"jaw tips {1000 * (floor - tip[2]):.1f}mm under the table at ({tip[0]:.3f}, {tip[1]:+.3f})"))
    tool = kinematics.tool_point(command)
    for xy in tiles:
        if target is not None and math.dist(xy, target) < 1e-6:
            continue
        top = config.table_z(*xy) + config.TILE_THICKNESS_M
        if holding and _overlaps(tool[:2], xy) and tip[2] - HELD_TILE_HANG_M < top - TILE_CLEAR_M:
            out.append(("onto_tile", f"the held tile {1000 * (top - tip[2] + HELD_TILE_HANG_M):.1f}mm into the tile at "
                                     f"({xy[0]:.3f}, {xy[1]:+.3f})"))
        elif (_inside(tool[:2], xy) or _inside(tip[:2], xy)) and tip[2] < top - TILE_CLEAR_M:      # the jaws' centre or the fixed tip
            out.append(("onto_tile", f"the jaws {1000 * (top - tip[2]):.1f}mm into the tile at ({xy[0]:.3f}, {xy[1]:+.3f})"))
    return out
