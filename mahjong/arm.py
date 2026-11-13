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
import json
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
LIFT_MIN_RADIUS_M = 0.14      # ...but only out over the table (tiles from ~0.15m), not at the folded rest (0.132m)
LIFT_MAX_JOINT_CHANGE_DEG = 30.0      # per RISE_STEP_M step
AT_REST_TOLERANCE_DEG = 3.0
# "Moved while off" compares the arm at connect with where it settled at the last disconnect. REST_POSE is
# that settled pose now (it used to sag ~8° from rest onto the stops, 2026-10-26), but a recording beats a constant.
PARK_SETTLE_S = 1.0
PARKED_TOLERANCE_DEG = 3.0

PLACE_TIP_CLEARANCE_M = 0.010     # release height when the grasp's isn't known (recorded grasps closed at 7mm, median)
# A return lets go RELEASE_ABOVE_GRASP_M above where this tile's own jaws closed (measured tip), so it lands as it
# was held. The fixed 10mm let go of v4's tiles a median 3.3mm above that (6-9mm at 7D/4D on 2026-10-29), and
# landings got worse with the drop: 3.4mm from home under 3mm, 6.2mm at 6mm or more (137 returns). Grasps close
# at 6-8mm near the base and 12-14mm at full reach (flex the FK doesn't see), and v3c and v4 differ by 5mm.
RELEASE_ABOVE_GRASP_M = 0.001
RELEASE_MIN_M, RELEASE_MAX_M = -0.003, 0.020     # the grasp point is 5mm above the table: from 2mm above it
# A set-down only ever goes down, SET_DOWN_GAIN of the measured height still to go at a time, and stops within
# PLACE_TOLERANCE_M above the release height. _settle's raises (overshoot up, come back down) can end lower than they
# started at full reach, and with a tighter tolerance they ended 4-10mm under the target in the sim; the full step
# overshoots down 2mm, 0.8 lands 0.1-1.3mm above it (9 spots x 5 heights).
SET_DOWN_GAIN = 0.8
SET_DOWN_ADJUSTS = 4
PLACE_TOLERANCE_M = 0.0015
PLACE_MIN_OFFSET_M = -0.003       # the lowest commanded tip at a set-down: 2mm above the table
RETURN_EXTRA_CLEARANCE_M = 0.030
PLACE_MAX_SHIFT_M = 0.015         # a return may move this far toward the shoulder to fit the reach (place_point)
PLACE_DESCEND_TIME_S = 2.0
# After a release, rise straight up this far above the release height before moving on: one joint move to the
# approach swung the open jaws sideways through the tile (and dragged it) on the way to rest (2026-10-27)
RELEASE_RISE_M = 0.030
# The rise goes through IK waypoints RISE_STEP_M apart (a straight commanded line), at lift()'s ~70mm/s. At 0.15s a
# step (33mm/s) the open jaws were seen drifting sideways off the tile on the way up while lift()'s single fast
# move looked straight (2026-10-29): P-only servos rising against gravity lag, and unequally per joint.
RISE_STEP_M, RISE_STEP_S = 0.005, 0.075
RISE_MAX_DEG_S = 60.0     # ...but no joint faster than this (near full reach 5mm up turns a joint several degrees)
# Before a return swings sideways to rest, the measured tip must be this far above the grasp point (~45mm over
# the tile top); the commanded approach (~95mm) sits 3-26mm higher than measured (574 hovers, 2026-10-27/28).
SWING_CLEARANCE_M = 0.060
RELEASE_TIME_S = 0.6
SQUEEZE_SETTLE_S = 0.3
# A direct move with a tile in the jaws, from over the mat to where it's going without going to rest (play). Timed by
# distance, at the speed of the slowest of today's 3s moves with a tile held (raised rest to the discard pose: 99 deg).
TRAVEL_DEG_S = 33.0
TRAVEL_MIN_S = 0.5
# Held, the tile hangs ~12mm under the tip and a neighbour's top is 15mm over the grasp point: 45mm leaves ~18mm spare.
# lift() commands 80mm, measured 54mm+ (sag), so this only stops a lift that gave up early.
TRAVEL_CLEARANCE_M = 0.045
# Returns were measured arriving from rest, the pan always turning up. A P-only servo with backlash stops somewhere else
# coming the other way (as with heights, arriving from below), so a direct move turns past and comes back.
TRAVEL_PAN_BACK_DEG = 2.0

JOINT_LIMIT_MARGIN_DEG = 3.0


def holds_tile(command_gripper, measured_gripper):
    """Something between the jaws: the gripper commanded shut but measured config.HOLDING_MARGIN wider."""
    return command_gripper < config.GRIP_CLOSED_BELOW and measured_gripper - command_gripper > config.HOLDING_MARGIN


def release_height(grasp_clearance_m=None):
    """The measured tip height to let a held tile go at: just above where its grasp closed, or
    PLACE_TIP_CLEARANCE_M if that isn't known."""
    if grasp_clearance_m is None:
        return PLACE_TIP_CLEARANCE_M
    return float(min(max(grasp_clearance_m + RELEASE_ABOVE_GRASP_M, RELEASE_MIN_M), RELEASE_MAX_M))


def place_point(x, y, approach_offset_m, gripper, release_m=PLACE_TIP_CLEARANCE_M):
    """(x, y), or the nearest point toward the shoulder (at most PLACE_MAX_SHIFT_M in), from which the
    whole set-down is reachable: every height from the release up to the approach. (x, y, shift_m), or
    None. 2026-10-23: the policy closed on 4D 11-12mm beyond its hover, ~0.29m from the shoulder and
    outside the IK's envelope (15 deg tilt, 10 deg joint margins), though the policy got there; 7 and
    13mm in fit. Tiles sit 57mm+ apart, so 15mm still clears the neighbours. Reaching the approach
    height alone isn't enough: 4mm in reached 50mm up but not the release, and would drop the tile."""
    r = np.hypot(x - kinematics.SHOULDER_PAN_X, y)
    heights = np.append(np.arange(release_m, approach_offset_m, 0.005), approach_offset_m)
    for shift in np.arange(0.0, PLACE_MAX_SHIFT_M + 1e-9, 0.001):
        k = 1 - shift / r
        px, py = kinematics.SHOULDER_PAN_X + (x - kinematics.SHOULDER_PAN_X) * k, y * k
        if all(kinematics.hover_command(px, py, h, gripper)[0] is not None for h in heights):
            return float(px), float(py), float(shift)
    return None


def parked_path():
    return config.SESSIONS / "parked_pose.json"          # per machine, like sessions/


def load_parked():
    """The arm's joints where it settled unpowered at the last disconnect, or None."""
    try:
        return json.loads(parked_path().read_text())["pose"]
    except (OSError, ValueError, KeyError):
        return None


def save_parked(pose):
    parked_path().parent.mkdir(parents=True, exist_ok=True)
    parked_path().write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "pose": pose}) + "\n")


def connect_state(pose, parked=None):
    """{"arm_at_rest", "off_rest_deg", "off_parked_deg"} for the arm found at `pose`: at rest means within
    PARKED_TOLERANCE_DEG of where it settled at the last disconnect (without one, of REST_POSE)."""
    off_rest = max(abs(pose[j] - config.REST_POSE[j]) for j in config.ARM_JOINTS)
    if parked is None:
        return {"arm_at_rest": off_rest < AT_REST_TOLERANCE_DEG, "off_rest_deg": off_rest, "off_parked_deg": None}
    off_parked = max(abs(pose[j] - parked[j]) for j in config.ARM_JOINTS)
    return {"arm_at_rest": off_parked < PARKED_TOLERANCE_DEG, "off_rest_deg": off_rest, "off_parked_deg": off_parked}


class Arm:
    at_connect = None       # set by connect(): connect_state()

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
        # where it was found: not where it settled at the last disconnect means it was moved while off
        arm.at_connect = connect_state(arm.commanded(), load_parked())
        return arm

    def disconnect(self):
        """Torque off, then note where the unpowered arm settles, for the next connect's check."""
        bus = getattr(self.robot, "bus", None)
        if hasattr(bus, "disable_torque"):          # the real bus (the sim has none)
            try:
                bus.disable_torque()
                self.sleep(PARK_SETTLE_S)
                save_parked(self.measured())
            except Exception as error:             # e.g. an overloaded servo: still disconnect
                print(f"  (couldn't record where the arm settled: {error})")
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
        above_grasp = kinematics.tip_clearance(self.measured())
        return above_grasp, above_grasp - (config.TILE_THICKNESS_M - config.GRASP_POINT_ABOVE_TABLE_M)

    def measured_tip(self):
        """The pincer tip (x, y, z) where the servos are, m."""
        return kinematics.pincer_tip(self.measured())

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
        """If low out over the table, rise straight up to LIFT_CLEARANCE_M, RISE_STEP_M at a time, the jaws
        pointing as they are (kinematics.straight_up; wrist roll and gripper kept), so the next move doesn't
        sweep across the tiles. Every move off the table starts here (rest, hover, place), whatever came
        before: a pickup, a teleop, a discarded episode. One joint-space move to the IK's lift pose instead
        curved through the tiles, and was skipped when the wrist had been tilted (L/R) more than
        LIFT_MAX_JOINT_CHANGE_DEG from the IK's pose. Where keeping the tilt would run the wrist into its
        limit (jaws tilted forward at full reach), wrist_flex holds and the jaws tilt a little with the arm."""
        command = self.commanded()
        tip = kinematics.pincer_tip(command)
        commanded_clearance = kinematics.tip_clearance(command)
        # measured low but already commanded higher (a sagging arm after a return's approach): re-commanding
        # LIFT_CLEARANCE_M would lower it before the swing to rest
        if self.is_at_rest() or self.tip_clearance()[0] >= LIFT_CLEARANCE_M or \
                commanded_clearance >= LIFT_CLEARANCE_M or np.hypot(tip[0] - kinematics.SHOULDER_PAN_X, tip[1]) < LIFT_MIN_RADIUS_M:
            return
        for _ in range(int(LIFT_CLEARANCE_M / RISE_STEP_M) + 2):
            if commanded_clearance >= LIFT_CLEARANCE_M - 0.0005:
                return
            current = self.commanded()
            dz = min(RISE_STEP_M, LIFT_CLEARANCE_M - commanded_clearance)
            step = kinematics.straight_up(current, dz) or kinematics.straight_up(current, dz, keep_axis=False)
            if step is None:
                return
            turn = max(abs(step[j] - current[j]) for j in config.ARM_JOINTS)
            if turn > LIFT_MAX_JOINT_CHANGE_DEG:
                return
            self.move_to({j: step[j] for j in config.ARM_JOINTS}, max(RISE_STEP_S, turn / RISE_MAX_DEG_S))
            commanded_clearance = kinematics.tip_clearance(self.commanded())

    def lift_clear(self):
        """lift(), then whether the measured tip is high enough to carry a tile over the others."""
        self.lift()
        return self.tip_clearance()[0] >= TRAVEL_CLEARANCE_M

    def travel_time(self, target):
        """Seconds for a direct move to `target` from the current command (TRAVEL_DEG_S, at most MOVE_TIME_S)."""
        command = self.commanded()
        turn = max(abs(target[j] - command[j]) for j in config.ARM_JOINTS)
        return min(max(turn / TRAVEL_DEG_S, TRAVEL_MIN_S), MOVE_TIME_S)

    def travel(self, target):
        """A direct move from over the mat to `target` over a tile, the pan arriving the way it does from rest
        (TRAVEL_PAN_BACK_DEG). The caller has lifted first."""
        if target["shoulder_pan"] < self.commanded()["shoulder_pan"]:
            past = {**target, "shoulder_pan": target["shoulder_pan"] - TRAVEL_PAN_BACK_DEG}
            self.move_to(past, self.travel_time(past))
        self.move_to(target, self.travel_time(target))

    def squeeze(self):
        """Close jaws that stopped nearly shut the rest of the way, as the policies do. Returns the command
        they were at, or None if nothing was done. A person stops closing once the jaws stall on a tile
        (commanded 5-12, reading 11-13: 20 of 35 episodes on 2026-10-30), which holds_tile can't tell from
        empty jaws, so the tile wasn't returned. On a tile they stall at ~11 and grip tighter; on nothing
        they shut, a few mm."""
        command, measured = self.commanded()["gripper"], self.measured()["gripper"]
        if command < config.GRIP_CLOSED_BELOW or measured >= config.GRIP_SQUEEZE_BELOW:
            return None
        self.set_gripper(0.0)
        self.sleep(SQUEEZE_SETTLE_S)
        return command

    def rise(self, x, y, from_offset, to_offset):
        """Straight up over tool point (x, y), RISE_STEP_M at a time (wrist roll and gripper as they are), so
        nothing near the jaws gets swept: each step is a short move over the same point."""
        command = self.commanded()
        for offset in np.append(np.arange(from_offset + RISE_STEP_M, to_offset, RISE_STEP_M), to_offset):
            step, _ = kinematics.hover_command(x, y, float(offset), command["gripper"])
            if step is None:
                return
            step["wrist_roll"] = command["wrist_roll"]
            current = self.commanded()
            turn = max(abs(step[j] - current[j]) for j in config.ARM_JOINTS)
            self.move_to({j: step[j] for j in config.ARM_JOINTS}, max(RISE_STEP_S, turn / RISE_MAX_DEG_S))

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
        self.lift()
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

    def _set_down(self, x, y, offset, target_m, gripper):
        """From commanded `offset` over (x, y), lower until the measured tip is within PLACE_TOLERANCE_M above
        target_m: never up, SET_DOWN_GAIN of what's left each time. Returns {offset_m, clearance_m, adjusts}."""
        clearance, _ = self.tip_clearance()
        adjusts = []
        for _ in range(SET_DOWN_ADJUSTS):
            left = clearance - target_m
            if left <= PLACE_TOLERANCE_M:
                break
            new_offset = max(offset - SET_DOWN_GAIN * left, PLACE_MIN_OFFSET_M)
            command, _ = kinematics.hover_command(x, y, new_offset, gripper)
            if command is None or new_offset >= offset:
                break
            self.move_to(command, ADJUST_TIME_S if offset - new_offset < 0.03 else PLACE_DESCEND_TIME_S)
            self.sleep(SETTLE_S)
            offset = new_offset
            clearance, _ = self.tip_clearance()
            adjusts.append({"offset_m": offset, "clearance_m": clearance})
        return {"offset_m": offset, "clearance_m": clearance, "adjusts": adjusts}

    def place(self, x, y, approach_offset_m, log=print, grasp_clearance_m=None, direct=False):
        """Set a held tile down with the tool point over (x, y): approach from above (extra height for
        the swing), measure the sag, descend until the measured tip is at release_height(grasp_clearance_m)
        (just above where the jaws closed on this tile, kinematics.grasp_clearance), release, lift, rest.
        Beyond reach, (x, y) moves toward the shoulder (place_point). `direct`: from over the mat straight to the
        approach (travel), not in MOVE_TIME_S. Returns {released_at_m (the measured
        clearance at release), release_target_m, xy, shift_m, ...}, or None if out of reach even so (nothing
        moved)."""
        held = self.commanded()["gripper"]
        target = release_height(grasp_clearance_m)
        point = place_point(x, y, approach_offset_m, held, target)
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
        self.lift()                     # from where a recording ended, low over the table
        if direct and not self.is_at_rest():
            self.travel(above)
        else:
            self.raise_from_rest()
            self.move_to(above, MOVE_TIME_S)
        self.sleep(SETTLE_S)
        settled = self._set_down(x, y, above_offset, target, held)
        released_at = settled["clearance_m"]
        log(f"  setting down: tip {released_at * 1000:.0f}mm above the grasp point "
            f"(commanded {settled['offset_m'] * 1000:.0f}mm; "
            + (f"the jaws closed at {grasp_clearance_m * 1000:.0f}mm)" if grasp_clearance_m is not None
               else "where they closed isn't known)"))
        release = min(config.GRIPPER_RELEASE, self.measured()["gripper"] + config.GRIPPER_RELEASE_PAST_HELD)
        self.set_gripper(release)
        self.sleep(0.3)
        at_release = self.measured_tip()
        self.rise(x, y, settled["offset_m"], settled["offset_m"] + RELEASE_RISE_M)
        after_rise = self.measured_tip()
        self.move_to({j: above[j] for j in config.ARM_JOINTS}, ADJUST_TIME_S)
        self.sleep(SETTLE_S)
        swing_from, _ = self.tip_clearance()
        if swing_from < SWING_CLEARANCE_M:      # still low over the tile: higher, over the same point, first
            log(f"  {swing_from * 1000:.0f}mm over the grasp point after the rise: raising before the swing to rest")
            raised = self._settle(x, y, above_offset, SWING_CLEARANCE_M, self.commanded()["gripper"], above_offset,
                                  above_offset + MAX_EXTRA_RAISE_M, log)
            swing_from = raised["clearance_m"]
        self.rest()
        rise = {"up_m": float(after_rise[2] - at_release[2]),
                "sideways_m": float(np.hypot(*(after_rise[:2] - at_release[:2]))), "swing_from_m": swing_from}
        return {"released_at_m": released_at, "release_target_m": target, "xy": [x, y], "shift_m": shift,
                "release_gripper": release, "rise": rise}

    def go_to_pose(self, pose, duration=MOVE_TIME_S, direct=False):
        """A taught arm pose (e.g. over the discard container), via the raised rest; gripper untouched. `direct`: a
        move from over the mat is timed by its distance."""
        if direct and not self.is_at_rest():
            duration = self.travel_time(pose)
        self.raise_from_rest()
        self.move_to({j: pose[j] for j in config.ARM_JOINTS}, duration)

    def discard(self, pose, direct=False):
        """Let the held tile go over the discard container (`pose`: config.discard_pose()), then rest. Goes via rest
        first (go_to_pose doesn't lift from over the mat) unless `direct`, where the caller has lifted clear (play).
        Returns whether the jaws still read as holding a tile on arrival. Bench, record, wall_test and crude_run share it."""
        if not direct:
            self.rest()
        self.go_to_pose(pose, direct=direct)
        held = self.holding()[0]
        self.set_gripper(config.GRIPPER_RELEASE)
        self.rest()
        return held
