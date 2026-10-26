"""The follower arm, moved only by interpolating from its last command.

The servos run lerobot's default gains (I=0) and sit below their command — 5-13mm at the hover, more
when extended. Three bugs this week came from restarting motion at the *measured* pose: that
re-commands the arm lower and it sags again (the takeover drop, the 10-17mm dips while a hover
settled, return_to_rest). So: every move here starts from Goal_Position, the gripper keeps its
commanded value until a release, and arriving at a hover is always from above (a P-only servo holding
against gravity stops further short arriving from below).

Heights are pincer-tip heights above the grasp point (config.grasp_z, table + 5mm); a tile's top face
is ~15mm above that.
"""
import time

import numpy as np

from . import config, kinematics

MOVE_TIME_S = 3.0           # rest <-> table
ADJUST_TIME_S = 1.0
RAISE_TIME_S = 1.5
SETTLE_S = 0.5
COMMAND_HZ = 50

SETTLE_FROM_ABOVE_M = 0.020   # first hover this far above, measure the sag there, then come down
CLEARANCE_TOLERANCE_M = 0.003
MAX_EXTRA_RAISE_M = 0.040
MAX_LOWER_M = 0.010
MAX_ADJUSTS = 3
APPROACH_M = 0.010            # an upward adjustment overshoots this much, then comes down

LIFT_CLEARANCE_M = 0.080      # before swinging back to rest from low over the table
LIFT_MIN_RADIUS_M = 0.13      # ...but only out over the table, not at the folded rest pose
LIFT_MAX_JOINT_CHANGE_DEG = 30.0
AT_REST_TOLERANCE_DEG = 3.0

PLACE_TIP_CLEARANCE_M = 0.010     # recorded grasps closed with the tip at 7mm (median)
RETURN_EXTRA_CLEARANCE_M = 0.030
PLACE_MAX_SHIFT_M = 0.015         # a return may move this far toward the shoulder to fit the reach (place_point)
PLACE_DESCEND_TIME_S = 2.0
RELEASE_TIME_S = 0.6

JOINT_LIMIT_MARGIN_DEG = 3.0


def holds_tile(command_gripper, measured_gripper):
    """Something between the jaws: the gripper commanded shut but measured config.HOLDING_MARGIN wider."""
    return command_gripper < config.GRIP_CLOSED_BELOW and measured_gripper - command_gripper > config.HOLDING_MARGIN


def place_point(x, y, approach_offset_m, gripper):
    """(x, y), or the nearest point toward the shoulder (at most PLACE_MAX_SHIFT_M in), from which the
    whole set-down is reachable: every height from the release up to the approach. (x, y, shift_m), or
    None. 2026-10-23: the policy closed on 4D 11-12mm beyond its hover, ~0.29m from the shoulder and
    outside the IK's envelope (15 deg tilt, 10 deg joint margins), though the policy got there; 7 and
    13mm in fit. Tiles sit 57mm+ apart, so 15mm still clears the neighbours. Reaching the approach
    height alone isn't enough: 4mm in reached 50mm up but not the release, and would drop the tile."""
    r = np.hypot(x - kinematics.SHOULDER_PAN_X, y)
    heights = np.append(np.arange(PLACE_TIP_CLEARANCE_M, approach_offset_m, 0.005), approach_offset_m)
    for shift in np.arange(0.0, PLACE_MAX_SHIFT_M + 1e-9, 0.001):
        k = 1 - shift / r
        px, py = kinematics.SHOULDER_PAN_X + (x - kinematics.SHOULDER_PAN_X) * k, y * k
        if all(kinematics.hover_command(px, py, h, gripper)[0] is not None for h in heights):
            return float(px), float(py), float(shift)
    return None


class Arm:
    at_connect = None       # set by connect(): {"arm_at_rest", "off_rest_deg"}

    def __init__(self, robot, sleep=time.sleep):
        self.robot = robot
        self.sleep = sleep

    @classmethod
    def connect(cls):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        robot = SO101Follower(SO101FollowerConfig(port=config.FOLLOWER_PORT, id=config.FOLLOWER_ID))
        try:
            robot.connect(calibrate=False)
        except Exception:
            # lerobot leaves the port open when the motors don't answer; close it so a retry can reopen it
            try:
                robot.bus.port_handler.closePort()
            except Exception:
                pass
            raise
        arm = cls(robot)
        arm.hold_measured()
        # where it was found: not at rest means it fell or was moved while off (the scripts log it)
        off = arm.off_rest_deg()
        arm.at_connect = {"arm_at_rest": off < AT_REST_TOLERANCE_DEG, "off_rest_deg": off}
        return arm

    def disconnect(self):
        self.robot.disconnect()

    # --- state ---

    def commanded(self):
        goal = self.robot.bus.sync_read("Goal_Position")
        return {j: float(goal[j]) for j in config.JOINTS}

    def measured(self):
        present = self.robot.bus.sync_read("Present_Position")
        return {j: float(present[j]) for j in config.JOINTS}

    def hold_measured(self):
        """Once, at connect: make the command match where the arm is, before any move."""
        self.send(self.measured())

    def send(self, pose):
        self.robot.send_action({f"{j}.pos": float(v) for j, v in pose.items()})

    def tip_clearance(self):
        """(pincer tip above the grasp point, above the tile's top face) under the measured tip, m."""
        tip = kinematics.pincer_tip(self.measured())
        above_grasp = float(tip[2] - config.grasp_z(tip[0], tip[1]))
        return above_grasp, above_grasp - (config.TILE_THICKNESS_M - config.GRASP_POINT_ABOVE_TABLE_M)

    def tool_sag(self):
        """Measured minus commanded tool point (x, y, z), m: how far the servos sit from their command."""
        return [float(v) for v in kinematics.tool_point(self.measured()) - kinematics.tool_point(self.commanded())]

    def holding(self):
        """(something between the jaws?, commanded gripper, measured gripper)."""
        command, measured = self.commanded()["gripper"], self.measured()["gripper"]
        return holds_tile(command, measured), command, measured

    def joint_limits(self):
        """Per-joint command limits from the servos' calibrated ranges (less a margin, so a command
        never drives into a mechanical stop); generic ranges when no calibration is available."""
        limits = {j: (-100.0, 100.0) for j in config.ARM_JOINTS}
        limits.update(wrist_roll=(-175.0, 175.0), gripper=(0.0, 100.0))
        bus = getattr(self.robot, "bus", None)
        calibration = getattr(bus, "calibration", None)
        for joint in config.ARM_JOINTS if calibration else ():
            cal = calibration[joint]
            max_res = bus.model_resolution_table[bus.motors[joint].model] - 1
            half = (cal.range_max - cal.range_min) / 2 * 360 / max_res
            limits[joint] = (-half + JOINT_LIMIT_MARGIN_DEG, half - JOINT_LIMIT_MARGIN_DEG)
        return limits

    def off_rest_deg(self):
        """The largest commanded arm-joint distance from the rest pose, degrees."""
        command = self.commanded()
        return max(abs(command[j] - config.REST_POSE[j]) for j in config.ARM_JOINTS)

    def is_at_rest(self):
        return self.off_rest_deg() < AT_REST_TOLERANCE_DEG

    # --- motion ---

    def move_to(self, target, duration):
        """Interpolate the command from the current command to `target` (joints not in target keep
        their commanded value). Returns the final command."""
        start = self.commanded()
        end = {**start, **target}
        steps = max(1, int(duration * COMMAND_HZ))
        for i in range(1, steps + 1):
            a = i / steps
            self.send({j: start[j] + a * (end[j] - start[j]) for j in config.JOINTS})
            self.sleep(duration / steps)
        return end

    def set_gripper(self, value, duration=RELEASE_TIME_S):
        return self.move_to({"gripper": float(value)}, duration)

    def raise_from_rest(self):
        """At rest, lift the folded arm in place before it swings out over the LeKiwi base."""
        if self.is_at_rest():
            self.move_to({j: config.RAISED_REST[j] for j in config.ARM_JOINTS}, RAISE_TIME_S)

    def lift(self):
        """If low out over the table, rise straight up to LIFT_CLEARANCE_M (wrist roll and gripper as they
        are), so the next move doesn't sweep across the tiles."""
        command = self.commanded()
        x, y, _ = kinematics.tool_point(command)
        clearance, _ = self.tip_clearance()
        if clearance < LIFT_CLEARANCE_M and np.hypot(x - kinematics.SHOULDER_PAN_X, y) >= LIFT_MIN_RADIUS_M:
            lifted, _ = kinematics.hover_command(x, y, LIFT_CLEARANCE_M, command["gripper"])
            if lifted is not None:
                lifted["wrist_roll"] = command["wrist_roll"]          # a lift, not a re-grip
                if max(abs(lifted[j] - command[j]) for j in config.ARM_JOINTS) <= LIFT_MAX_JOINT_CHANGE_DEG:
                    self.move_to({j: lifted[j] for j in config.ARM_JOINTS}, ADJUST_TIME_S)

    def rest(self):
        """Lift if low over the table, swing in via the raised rest, lower to rest. Never touches the
        gripper (a held tile stays held). Does nothing if already at rest."""
        if self.is_at_rest():
            return
        self.lift()
        self.move_to({j: config.RAISED_REST[j] for j in config.ARM_JOINTS}, MOVE_TIME_S)
        self.move_to({j: config.REST_POSE[j] for j in config.ARM_JOINTS}, RAISE_TIME_S)

    def hover(self, x, y, offset_m, clearance_m, gripper, log=print):
        """Hover the tool point over closed-loop target (x, y) and settle until the measured pincer tip
        is clearance_m above the grasp point. Goes SETTLE_FROM_ABOVE_M high first, measures the sag
        there, then comes down (an upward correction overshoots and comes back down), so the jaws never
        dip below the target while settling. Commanded offsets stay within
        [offset_m - MAX_LOWER_M, offset_m + MAX_EXTRA_RAISE_M]. Returns a dict, or None if unreachable."""
        first = None
        for extra in (SETTLE_FROM_ABOVE_M, SETTLE_FROM_ABOVE_M / 2, 0.0):
            first, tilt = kinematics.hover_command(x, y, offset_m + extra, gripper)
            if first is not None:
                break
        if first is None:
            return None
        self.raise_from_rest()
        self.move_to(first, MOVE_TIME_S)
        self.sleep(SETTLE_S)
        settled = self._settle(x, y, offset_m + extra, clearance_m, gripper,
                               offset_m - MAX_LOWER_M, offset_m + MAX_EXTRA_RAISE_M, log)
        log(f"  hover settled: tip {settled['clearance_m'] * 1000:.0f}mm above the grasp point "
            f"(~{settled['tile_top_gap_m'] * 1000:.0f}mm above the tile top), commanded offset "
            f"{settled['offset_m'] * 1000:.0f}mm")
        return {"hover_xy": [x, y], **settled, "tilt_deg": settled["tilt_deg"] or tilt,
                "command": self.commanded()}

    def _settle(self, x, y, offset, target_m, gripper, min_offset, max_offset, log):
        """From a hover at commanded `offset` over (x, y), correct the commanded offset from the
        measured clearance until it is target_m (within CLEARANCE_TOLERANCE_M), at most MAX_ADJUSTS
        times, within [min_offset, max_offset]. Going up overshoots by APPROACH_M and comes back
        down, so every arrival is from above; near full reach a raise takes as much as fits. The sag
        (Arm.tool_sag) is logged before and after each adjust, to track it over battery level and time."""
        clearance, gap = self.tip_clearance()
        adjusts, tilt, first_sag = [], None, self.tool_sag()
        for _ in range(MAX_ADJUSTS):
            error = target_m - clearance              # > 0: too low
            if abs(error) <= CLEARANCE_TOLERANCE_M:
                break
            goal = min(max(offset + error, min_offset), max_offset)
            if abs(goal - offset) < 0.0005:
                log(f"  clearance {clearance * 1000:.0f}mm: at the limit of the allowed heights")
                break
            command = None
            for fraction in (1.0, 0.5, 0.25):
                new_offset = offset + fraction * (goal - offset)
                command, tilt = kinematics.hover_command(x, y, new_offset, gripper)
                if command is not None:
                    break
            if command is None:
                log(f"  can't go higher here (out of reach); clearance stays {clearance * 1000:.0f}mm")
                break
            if new_offset > offset:
                above, _ = kinematics.hover_command(x, y, new_offset + APPROACH_M, gripper)
                if above is not None:
                    self.move_to(above, ADJUST_TIME_S)
            self.move_to(command, ADJUST_TIME_S if abs(new_offset - offset) < 0.03 else PLACE_DESCEND_TIME_S)
            self.sleep(SETTLE_S)
            offset = new_offset
            clearance, gap = self.tip_clearance()
            adjusts.append({"offset_m": offset, "clearance_m": clearance, "sag_xyz_m": self.tool_sag()})
        return {"offset_m": offset, "clearance_m": clearance, "tile_top_gap_m": gap, "adjusts": adjusts,
                "tilt_deg": tilt, "first_sag_xyz_m": first_sag, "sag_xyz_m": self.tool_sag()}

    def place(self, x, y, approach_offset_m, log=print):
        """Set a held tile down with the tool point over (x, y): approach from above (extra height for
        the swing), measure the sag, descend until the measured tip is PLACE_TIP_CLEARANCE_M above the
        grasp point, release, lift, rest. Beyond reach, (x, y) moves toward the shoulder (place_point).
        Returns {released_at_m (the measured clearance at release), xy, shift_m}, or None if out of reach
        even so (nothing moved)."""
        held = self.commanded()["gripper"]
        point = place_point(x, y, approach_offset_m, held)
        if point is None:
            log(f"  ({x:.3f}, {y:+.3f}) is out of reach, even {PLACE_MAX_SHIFT_M * 1000:.0f}mm closer — not moving")
            return None
        if point[2]:
            log(f"  ({x:.3f}, {y:+.3f}) is out of reach: setting down {point[2] * 1000:.0f}mm closer, "
                f"at ({point[0]:.3f}, {point[1]:+.3f})")
        x, y, shift = point
        for extra in (SETTLE_FROM_ABOVE_M + RETURN_EXTRA_CLEARANCE_M, SETTLE_FROM_ABOVE_M, 0.0):
            above_offset = approach_offset_m + extra
            above, _ = kinematics.hover_command(x, y, above_offset, held)
            if above is not None:       # at the latest with extra 0: place_point checked that height
                break
        self.raise_from_rest()
        self.move_to(above, MOVE_TIME_S)
        self.sleep(SETTLE_S)
        settled = self._settle(x, y, above_offset, PLACE_TIP_CLEARANCE_M, held, 0.0, above_offset, log)
        released_at = settled["clearance_m"]
        log(f"  setting down: tip {released_at * 1000:.0f}mm above the grasp point "
            f"(commanded {settled['offset_m'] * 1000:.0f}mm)")
        release = min(config.GRIPPER_RELEASE, self.measured()["gripper"] + config.GRIPPER_RELEASE_PAST_HELD)
        self.set_gripper(release)
        self.sleep(0.3)
        self.move_to({j: above[j] for j in config.ARM_JOINTS}, ADJUST_TIME_S)
        self.rest()
        return {"released_at_m": released_at, "xy": [x, y], "shift_m": shift, "release_gripper": release}

    def go_to_pose(self, pose, duration=MOVE_TIME_S):
        """A taught arm pose (e.g. over the discard container), via the raised rest; gripper untouched."""
        self.raise_from_rest()
        self.move_to({j: pose[j] for j in config.ARM_JOINTS}, duration)
