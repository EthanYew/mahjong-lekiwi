"""What the arm does once the gamepad has it: every teleop step's command, measured joints and input.

2026-10-28: lining up hand corrections, the arm dropped and moved after the hover with nothing pressed,
and corrections came out larger than they looked. Three things look the same from outside, and a trace
tells them apart:
- the servos sinking under an unchanged command (creep / load): measured moves, command doesn't;
- the command moving with nothing pressed: a bug in the teleop path (it should hold the seed exactly);
- XLeRobot's teleop re-solving the pose with its own 2-link IK on the first stick push (it eases
  shoulder_lift / elbow_flex onto its solution if the seed isn't one): the command moves in a direction
  that wasn't asked for, e.g. height while only reach was pushed.
A correction records the *commanded* tool point, while the jaws are lined up where the arm really is, so
the measured-minus-commanded tool point at → is error carried straight into the correction.

Points are kinematics.tool_point (what corrections record), mm, relative to the command at the handoff.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from . import config, kinematics

NOT_HEIGHT_STICKS = ("lx", "ly")        # pan, reach: neither should change the tool point's height


def _tool(pose):
    return np.asarray(kinematics.tool_point(pose), dtype=float)


def seed_ik_mismatch_deg(teleop, seed):
    """How far XLeRobot's own IK puts shoulder_lift / elbow_flex from the seed (deg): what its first x/y
    input eases onto. None for a teleop without XLeRobot's internals."""
    kin = getattr(teleop, "kin", None)
    if kin is None or not hasattr(teleop, "x"):
        return None
    lift, elbow = kin.inverse_kinematics(teleop.x, teleop.y)
    return float(max(abs(lift - seed["shoulder_lift"]), abs(elbow - seed["elbow_flex"])))


class HandoffTrace:
    """Started as the gamepad takes over (after gamepad.start); add() after each gamepad.step."""

    def __init__(self, arm, gamepad, clock=time.monotonic):
        self.clock, self.t0 = clock, clock()
        self.seed = arm.commanded()
        self.ik_mismatch_deg = seed_ik_mismatch_deg(getattr(gamepad, "teleop", None), self.seed)
        self.rows = [self._row(self.seed, arm.measured(), None, getattr(gamepad, "teleop", None))]

    def _row(self, command, measured, last, teleop):
        row = {"t": round(self.clock() - self.t0, 4), "command": {j: float(command[j]) for j in config.JOINTS},
               "measured": {j: float(measured[j]) for j in config.JOINTS}}
        if last is not None:
            row["sticks"] = {k: float(v) for k, v in last["sticks"].items()}
            row["buttons"] = list(last["buttons"])
        if teleop is not None and hasattr(teleop, "x"):
            row["ee"] = [float(teleop.x), float(teleop.y), float(teleop.pitch)]
        return row

    def add(self, arm, command, gamepad):
        """One teleop step: `command` is what gamepad.step sent; the measured pose and the input come from
        gamepad.last when it has them (no second bus read)."""
        last = getattr(gamepad, "last", None)
        measured = last["measured"] if last else arm.measured()
        self.rows.append(self._row(command, measured, last, getattr(gamepad, "teleop", None)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")
        return path

    def summary(self):
        return summarize(self.rows, self.ik_mismatch_deg)


def touched(row):
    """Any input in this step (None: the gamepad didn't say)."""
    if "sticks" not in row:
        return None
    return any(row["sticks"].values()) or bool(row["buttons"])


def summarize(rows, ik_mismatch_deg=None):
    """Numbers to tell sag from a moving command (mm, s); see the module docstring."""
    cmd = np.array([_tool(r["command"]) for r in rows]) * 1000
    meas = np.array([_tool(r["measured"]) for r in rows]) * 1000
    t = np.array([r["t"] for r in rows])
    inputs = [touched(r) for r in rows]
    known = all(i is not None for i in inputs[1:])
    first = next((k for k, i in enumerate(inputs) if i), None)
    idle_end = first if first is not None else len(rows) - 1
    step = np.diff(cmd, axis=0)
    idle_steps = [k for k in range(1, len(rows)) if inputs[k] is False]
    no_height_input = [k for k in range(1, len(rows)) if inputs[k] and not rows[k]["sticks"].get("ry")
                       and not {"l", "r"} & set(rows[k]["buttons"])
                       and any(rows[k]["sticks"].get(s) for s in NOT_HEIGHT_STICKS)]
    r = lambda v: [round(float(x), 1) for x in v]
    return {
        "steps": len(rows) - 1, "duration_s": round(float(t[-1]), 2),
        "ik_mismatch_deg": None if ik_mismatch_deg is None else round(ik_mismatch_deg, 1),
        "sag_at_handoff_mm": r(meas[0] - cmd[0]),
        "first_input_s": None if first is None or not known else round(float(t[first]), 2),
        # before anything was pressed: the command should not move at all; the arm may
        "idle_command_moved_mm": round(float(np.max(np.linalg.norm(cmd[:idle_end + 1] - cmd[0], axis=1))), 1),
        "idle_drift_mm": r(meas[idle_end] - meas[0]),
        # whenever nothing was pressed (after the first input too)
        "command_moved_untouched_mm": round(float(sum(np.linalg.norm(step[k - 1]) for k in idle_steps)), 1) if known else None,
        # the command's height change while only pan / reach were pushed
        "height_change_on_pan_reach_mm": round(float(sum(step[k - 1][2] for k in no_height_input)), 1) if known else None,
        "sag_at_end_mm": r(meas[-1] - cmd[-1]),
        "worst_sag_mm": round(float(np.max(np.linalg.norm(meas - cmd, axis=1))), 1),
    }


def describe(s):
    """One line per finding, for the terminal."""
    sag = lambda v: f"({v[0]:+.1f}, {v[1]:+.1f}, {v[2]:+.1f})"
    lines = [f"sag at handoff {sag(s['sag_at_handoff_mm'])}mm, at → {sag(s['sag_at_end_mm'])}mm "
             f"(x, y, z: measured - commanded tool point; the correction records the commanded one)"]
    if s["idle_command_moved_mm"] > 0.5:
        lines.append(f"the COMMAND moved {s['idle_command_moved_mm']:.1f}mm before any input: a teleop bug")
    drift = math.hypot(*s["idle_drift_mm"])
    if drift > 1.0:
        lines.append(f"the arm drifted {sag(s['idle_drift_mm'])}mm under an unchanged command before the first input")
    if s["ik_mismatch_deg"] is not None and s["ik_mismatch_deg"] > 2.0:
        lines.append(f"the hover is {s['ik_mismatch_deg']:.0f}° from XLeRobot's IK solution: the first stick push eases onto it")
    if s["command_moved_untouched_mm"] and s["command_moved_untouched_mm"] > 0.5:
        lines.append(f"the command moved {s['command_moved_untouched_mm']:.1f}mm in steps with nothing pressed")
    if s["height_change_on_pan_reach_mm"] and abs(s["height_change_on_pan_reach_mm"]) > 1.0:
        lines.append(f"pan / reach alone changed the commanded height by {s['height_change_on_pan_reach_mm']:+.1f}mm")
    return lines
