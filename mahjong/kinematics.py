"""SO101 kinematics: closed-form forward kinematics and a geometric inverse kinematics for top-down
(optionally slightly tilted) approaches.

From ~/Projects/so101_mahjong_pick_and_place (so101_forward_kinematics.py, so101_inverse_kinematics.py),
minus MuJoCo (joint limits are constants below), the residual-correction hook (fitted on the tabletop
mount, off on LeKiwi) and the servo-bus helpers. This is the URDF chain in closed form: MahjongBot's
placo FK of the same joints agrees to within 0.004mm once the pincer tip is expressed as a fixed
offset in this tool frame (PINCER_TOOL_OFFSET), so there is one kinematics model, not two.

Angles are lerobot degrees. Joint configs are dicts keyed by config.ARM_JOINTS (+ "gripper").
"""
import math

import numpy as np

from . import config

# Joint ranges from the arm's MuJoCo model (so101_mahjong_pick_and_place/model/scene.xml).
JOINT_LIMITS_DEG = {"shoulder_pan": (-110.0, 110.0), "shoulder_lift": (-100.0, 100.0),
                    "elbow_flex": (-96.8299, 96.8299), "wrist_flex": (-95.0, 95.0)}
SHOULDER_PAN_X = 0.0388353     # shoulder_pan's axis in the base frame
# MahjongBot's pincer tip (its PINCER_LOCAL_OFFSET, the point ACT hovers and the table floor are
# defined on) in this module's tool frame. Fitted on 400 random poses; max error 0.0036mm.
PINCER_TOOL_OFFSET = np.array([-0.016412, 0.008233, 0.026474])


# --- forward kinematics ---

def _rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _quat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _pose(rotation, displacement):
    # Filled in place, not np.block: the IK refines through ~200 FK calls per solve, and np.block was
    # 8us of each joint's 9 (a simulated hand spent 12s of 14s here).
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = displacement
    return T


# Each joint: fixed mounting rotation and translation from the URDF, then Rz(angle).
_JOINT_FRAMES = {
    "shoulder_pan": ((3.56167e-16, 1.22818e-15, -1.0, -4.14635e-16), (0.0388353, -8.97657e-09, 0.0624)),
    "shoulder_lift": ((0.5, -0.5, -0.5, -0.5), (-0.0303992, -0.0182778, -0.0542)),
    "elbow_flex": ((7.07106781e-01, -5.98612815e-17, -2.58050920e-17, 7.07106781e-01), (-0.11257, -0.028, 1.73763e-16)),
    "wrist_flex": ((7.07106781e-01, 9.58721703e-16, -7.51312768e-16, -7.07106781e-01), (-0.13490, 0.0052, 3.62355e-17)),
    "wrist_roll": ((0.01720911, -0.01720911, 0.70689734, 0.70689734), (5.55112e-17, -0.0611, 0.0181)),
}
_TOOL = _pose(_quat(7.07106781e-01, -0.0, 7.07106781e-01, -2.37787926e-17), (0.00332633, 0.00484881, -0.11503887))


_MOUNTS = {joint: _quat(*quat) for joint, (quat, _) in _JOINT_FRAMES.items()}


def joint_frame(joint, deg):
    return _pose(_MOUNTS[joint] @ _rot("z", deg), _JOINT_FRAMES[joint][1])


def forward(q):
    """Tool-frame pose (position, rotation) for joint config q."""
    T = np.eye(4)
    for joint in config.ARM_JOINTS:
        T = T @ joint_frame(joint, q[joint])
    T = T @ _TOOL
    return T[:3, 3], T[:3, :3]


def tool_point(q):
    """The IK's target point — hover targets and hover corrections are in this frame."""
    return forward(q)[0]


def grasp_frame(actions):
    """The first step commanding the gripper shut (below GRIP_CLOSED_BELOW), or None. actions: steps x
    config.JOINTS."""
    if not len(actions):
        return None
    shut = np.asarray(actions)[:, config.GRIPPER] < config.GRIP_CLOSED_BELOW
    return int(np.argmax(shut)) if shut.any() else None


def grasp_point(actions):
    """(step, tool-point xy) of the first command with the gripper shut — where the jaws closed on the
    tile — or (None, None). actions: steps x config.JOINTS, absolute."""
    grasp = grasp_frame(actions)
    if grasp is None:
        return None, None
    return grasp, [float(v) for v in tool_point(dict(zip(config.JOINTS, actions[grasp])))[:2]]


def tip_clearance(q):
    """The pincer tip's height above the grasp point under it (config.grasp_z), m."""
    tip = pincer_tip(q)
    return float(tip[2] - config.grasp_z(tip[0], tip[1]))


def grasp_clearance(states, actions):
    """tip_clearance where the servos were (states, measured) at the first step commanding the gripper shut:
    how high the tip was as the jaws closed on the tile, or None. Rows of config.JOINTS, absolute."""
    k = grasp_frame(actions)
    return None if k is None or k >= len(states) else tip_clearance(dict(zip(config.JOINTS, states[k])))


def pincer_tip(q):
    """The fixed jaw's tip: what clearances, set-down heights and the table floor are measured on."""
    position, rotation = forward(q)
    return position + rotation @ PINCER_TOOL_OFFSET


# --- inverse kinematics ---

L1 = float(np.linalg.norm(joint_frame("elbow_flex", 0)[:3, 3]))    # shoulder_lift pivot -> elbow pivot
L2 = float(np.linalg.norm(joint_frame("wrist_flex", 0)[:3, 3]))    # elbow pivot -> wrist_flex pivot


def _elbow_zero_reference():
    """Interior shoulder-elbow-wrist angle at elbow_flex = 0 (not 180: the joint's mounting is rotated)."""
    v1 = np.linalg.inv(joint_frame("elbow_flex", 0))[:3, 3]
    v2 = joint_frame("wrist_flex", 0)[:3, 3]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def _shoulder_zero_reference():
    """shoulder_lift value that makes the upper arm horizontal (not 90: same mounting quirk)."""
    base = joint_frame("shoulder_pan", 0) @ joint_frame("shoulder_lift", 90)
    v = (base @ joint_frame("elbow_flex", 0))[:3, 3] - base[:3, 3]
    return 90 + math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))


ELBOW_ZERO_REFERENCE = _elbow_zero_reference()        # ~106.18
SHOULDER_ZERO_REFERENCE = _shoulder_zero_reference()  # ~76.03


def _interior_angle(a, b, c):
    """Law of cosines: the angle opposite side a."""
    cos = (b * b + c * c - a * a) / (2 * b * c)
    return math.degrees(math.atan2(math.sqrt(1 - cos * cos), cos))


def _wrist_flex_target(target, tilt_deg, wrist_roll_deg):
    """Pose of the wrist_flex frame that puts the tool point at `target` pointing down, tilted by
    tilt_deg. wrist_roll matters at the mm level: the tool point is slightly off the roll axis."""
    tool = _pose(_rot("y", 90 + tilt_deg), target)
    wrist_to_tool = joint_frame("wrist_roll", wrist_roll_deg) @ _TOOL
    wrist = tool @ np.linalg.inv(wrist_to_tool)
    return wrist[:3, 3], wrist[:3, :3]


def _closed_form(target, tilt_deg):
    pan = -math.degrees(math.atan2(target[1], target[0] - SHOULDER_PAN_X))   # negated: mounting flip
    roll = pan - 90
    wrist_position, wrist_rotation = _wrist_flex_target(target, tilt_deg, roll)
    pivot = (joint_frame("shoulder_pan", pan) @ joint_frame("shoulder_lift", 0))[:3, 3]
    d = math.hypot(wrist_position[0] - pivot[0], wrist_position[1] - pivot[1])
    h = wrist_position[2] - pivot[2]
    r = math.hypot(d, h)
    lift = SHOULDER_ZERO_REFERENCE - _interior_angle(L2, L1, r) - math.degrees(math.atan2(h, d))
    elbow = ELBOW_ZERO_REFERENCE - _interior_angle(r, L1, L2)
    # wrist_flex: rotate the wrist's z axis from where it points with wrist_flex = 0 to the target's.
    current = (joint_frame("shoulder_pan", pan) @ joint_frame("shoulder_lift", lift)
               @ joint_frame("elbow_flex", elbow) @ joint_frame("wrist_flex", 0))[:3, 1]
    wanted = wrist_rotation[:, 1]
    flex = (math.degrees(math.atan2(wanted[2], math.hypot(wanted[0], wanted[1])))
            - math.degrees(math.atan2(current[2], math.hypot(current[0], current[1]))))
    return np.array([pan, lift, elbow, flex])


def _tool_point_4(theta):
    pan, lift, elbow, flex = theta
    return tool_point({"shoulder_pan": pan, "shoulder_lift": lift, "elbow_flex": elbow,
                       "wrist_flex": flex, "wrist_roll": pan - 90})


def _refine(target, theta, iterations=20, tol=1e-6, damping=1e-3, eps=1e-4):
    """Damped least-squares on the true tool point. The closed form assumes the shoulder/elbow sweep
    a clean vertical plane, which the URDF's small lateral offsets break: alone it lands 17-27mm off."""
    target = np.asarray(target, dtype=float)
    for _ in range(iterations):
        current = _tool_point_4(theta)
        error = target - current
        if np.linalg.norm(error) < tol:
            break
        jacobian = np.zeros((3, 4))
        for i in range(4):
            step = theta.copy()
            step[i] += eps
            jacobian[:, i] = (_tool_point_4(step) - current) / eps
        theta = theta + jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(3), error)
    return theta


def inverse(target, tilt_deg=0.0):
    """Joint config putting the tool point at `target`, pointing down (tilted by tilt_deg), with
    wrist_roll = shoulder_pan - 90 (jaws keep the same world heading everywhere)."""
    pan, lift, elbow, flex = _refine(target, _closed_form(target, tilt_deg))
    return {"shoulder_pan": pan, "shoulder_lift": lift, "elbow_flex": elbow, "wrist_flex": flex,
            "wrist_roll": pan - 90}


def joint_margin(q):
    return min(min(q[j] - lo, hi - q[j]) for j, (lo, hi) in JOINT_LIMITS_DEG.items())


def _reachable(target, tilt_deg, tolerance):
    try:
        q = inverse(target, tilt_deg)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
    wanted, _ = _wrist_flex_target(target, tilt_deg, q["wrist_roll"])
    T = np.eye(4)
    for joint in config.ARM_JOINTS[:4]:
        T = T @ joint_frame(joint, q[joint])
    if np.linalg.norm(T[:3, 3] - wanted) >= tolerance or joint_margin(q) < 0:
        return None
    return q


def find_reachable_pose(target, max_tilt_deg=15, min_margin_deg=10, tolerance=0.03):
    """Smallest tilt (0, then +-1, +-2 ... deg) that reaches `target` with min_margin_deg to every joint
    limit. Returns (q, tilt) or (None, None). Lowering the margin re-tilts every pose, not just the
    marginal ones (tried 2026-09-03, more collisions) — pass a smaller margin per call if needed."""
    for tilt in [0] + [s * m for m in range(1, max_tilt_deg + 1) for s in (-1, 1)]:
        q = _reachable(target, tilt, tolerance)
        if q is not None and joint_margin(q) >= min_margin_deg:
            return q, tilt
    return None, None


STRAIGHT_AXIS_WEIGHT_M = 0.05     # straight_up: an axis error of 1 deg weighs like ~0.9mm of position


def straight_up(q, dz, keep_axis=True, iterations=20):
    """Joint config moving the tool point dz straight up from where q puts it, the jaws pointing as they do
    in q (wrist roll and gripper kept as they are), or None outside the joint limits or if it doesn't
    converge. hover_command picks its own tilt, which after a teleop can be 20-30 deg from the wrist's.
    Jaws tilted forward at full reach run wrist_flex into its limit within ~30mm: keep_axis=False holds
    wrist_flex instead, and the jaws tilt a little with the arm."""
    position, rotation = forward(q)
    target, axis = position + np.array([0.0, 0.0, dz]), rotation[:, 0]
    joints = config.ARM_JOINTS[:4] if keep_axis else config.ARM_JOINTS[:3]

    def residual(theta):
        p, r = forward({**q, **dict(zip(joints, theta))})
        return np.concatenate([target - p, STRAIGHT_AXIS_WEIGHT_M * (axis - r[:, 0])]) if keep_axis else target - p

    theta, eps = np.array([q[j] for j in joints], dtype=float), 1e-4
    for _ in range(iterations):
        error = residual(theta)
        if np.linalg.norm(error) < 1e-6:
            break
        jacobian = np.zeros((len(error), len(joints)))
        for i in range(len(joints)):
            step = theta.copy()
            step[i] += eps
            jacobian[:, i] = (error - residual(step)) / eps
        theta = theta + np.linalg.solve(jacobian.T @ jacobian + 1e-6 * np.eye(len(joints)), jacobian.T @ error)
    moved = {**q, **dict(zip(joints, (float(t) for t in theta)))}
    if np.linalg.norm(residual(theta)[:3]) > 0.0005 or joint_margin(moved) < 0:
        return None
    return moved


def hover_command(x, y, offset_m, gripper):
    """Joint command hovering the tool point over closed-loop target (x, y) with the pincer tip
    offset_m above the grasp point there, gripper at `gripper`. The tool point and the pincer tip sit
    at different heights, so the tool target's z is iterated until the pincer tip lands. Returns
    (command, tilt) or (None, None) if out of reach."""
    tip_target = config.grasp_z(x, y) + offset_m
    z = tip_target
    q = tilt = None
    for _ in range(3):
        q, tilt = find_reachable_pose([x, y, z])
        if q is None:
            return None, None
        tip_z = float(pincer_tip(q)[2])
        if abs(tip_z - tip_target) < 0.001:
            break
        z += tip_target - tip_z
    return {**q, "gripper": float(gripper)}, tilt
