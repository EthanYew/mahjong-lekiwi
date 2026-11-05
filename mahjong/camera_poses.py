"""Arm poses that hold a tile up to the base camera (docs/face_down_plan.md, test 2), saved by name in data/camera_poses.json
the way the discard pose is: the five arm joints (what `Arm.go_to_pose` takes) and the gripper, with where each came from.

A pose placed by hand with the torque off is *measured*; commanding it makes the servos sag a few mm below it (CLAUDE.md,
"never start motion from the measured pose"), so a taught pose (scripts/teleop_view.py saves the *commanded* one) is the
better source once the camera view is checked. Going to a pose here is always `Arm.go_to_pose`: via the raised rest, from above.
"""
import json
from datetime import datetime

from . import config


def load(path=None):
    path = path or config.CAMERA_POSES
    return json.loads(path.read_text())["poses"] if path.exists() else {}


def save(name, joints, source, session=None, path=None):
    """Adds or replaces pose `name`. `joints` has the five arm joints and may have the gripper. Returns the entry."""
    path = path or config.CAMERA_POSES
    poses = load(path)
    entry = {"pose": {j: float(joints[j]) for j in config.ARM_JOINTS}, "source": source,
             "saved": datetime.now().isoformat(timespec="seconds")}
    if "gripper" in joints:
        entry["gripper"] = float(joints["gripper"])
    if session:
        entry["session"] = session
    poses[name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"poses": poses}, indent=2) + "\n")
    return entry


def next_name(prefix, path=None):
    """`prefix` + the next unused number: base_view_1, base_view_2, ..."""
    taken = load(path)
    n = 1
    while f"{prefix}{n}" in taken:
        n += 1
    return f"{prefix}{n}"


def get(name, path=None):
    poses = load(path)
    if name not in poses:
        raise SystemExit(f"No camera pose '{name}' in {path or config.CAMERA_POSES} (has: {', '.join(poses) or 'none'})")
    return poses[name]
