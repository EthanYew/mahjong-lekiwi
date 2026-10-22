"""Episode quality: how hesitant or erratic each demonstration was, from its commanded actions.

Per episode, on the pincer tip's path (kinematics.pincer_tip of action_raw): direction reversals of
the tip (x/y/z, counted once the tip comes back REVERSAL_MM from its furthest point, so jitter
doesn't count but "went the wrong way, corrected" does) and of the gripper, RMS jerk, duration, and
the idle fraction (reported, not scored — a pause is fine, a reversal is what makes a policy unsure).
Scored metrics become robust z-scores against the dataset's own median and are summed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, kinematics

REVERSAL_MM = 5.0
REVERSAL_GRIPPER = 5.0
IDLE_SPEED_MM_S = 3.0
DEFAULT_FLAG_SCORE = 4.0
MIN_EPISODES_FOR_SCORES = 8
SCORED_METRICS = {"tip_rev": 1.0, "grip_rev": 1.0, "jerk": None, "duration": None}   # spread floors
RELATIVE_SPREAD_FLOOR = 0.1


def turning_points(signal, threshold):
    """(index, size of the run that ended there) each time `signal` comes back `threshold` from its
    furthest point."""
    points, direction, extreme, extreme_i, run_start = [], 0, signal[0], 0, signal[0]
    for i, v in enumerate(signal):
        if direction == 0:
            if abs(v - run_start) >= threshold:
                direction, extreme, extreme_i = (1 if v > run_start else -1), v, i
        elif (v - extreme) * direction > 0:
            extreme, extreme_i = v, i
        elif (extreme - v) * direction >= threshold:
            points.append((extreme_i, abs(extreme - run_start)))
            direction, run_start, extreme, extreme_i = -direction, extreme, v, i
    return points


def dataset_root(repo_id):
    from lerobot.utils.constants import HF_LEROBOT_HOME
    return Path(HF_LEROBOT_HOME) / repo_id


def episode_metrics(joints, fps):
    tips = np.array([kinematics.pincer_tip(dict(zip(config.JOINTS, q))) for q in joints]) * 1000
    dt = 1.0 / fps
    events = [(i, f"tip {axis}", size, "mm") for k, axis in enumerate("xyz") for i, size in turning_points(tips[:, k], REVERSAL_MM)]
    tip_events = len(events)
    events += [(i, "gripper", size, "%") for i, size in turning_points(joints[:, 5], REVERSAL_GRIPPER)]
    speed = np.linalg.norm(np.diff(tips, axis=0), axis=1) / dt
    jerk = np.diff(tips / 1000, n=3, axis=0) / dt ** 3 if len(tips) >= 4 else np.zeros((1, 3))
    return {"frames": len(joints), "duration": len(joints) / fps, "path_mm": float(speed.sum() * dt),
            "tip_rev": tip_events, "grip_rev": len(events) - tip_events,
            "jerk": float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1)))),
            "idle": float(np.mean(speed < IDLE_SPEED_MM_S)) if len(speed) else 0.0}, sorted(events)


def quality_table(repo_id):
    """(table indexed by episode, {episode: reversal events}, fps)."""
    root = dataset_root(repo_id)
    df = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").glob("*/*.parquet"))], ignore_index=True)
    fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
    log = {}
    for name in ("episode_log.jsonl", "episode_teleop_log.jsonl"):        # new and pre-rebuild sidecars
        path = root / name
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    log[record["episode_index"]] = record
    rows, events = [], {}
    for episode, frames in df.groupby("episode_index"):
        joints = np.stack(frames.sort_values("frame_index")["action_raw"].to_numpy())
        metrics, events[episode] = episode_metrics(joints, fps)
        rows.append({"episode": episode, **metrics, "tile": log.get(episode, {}).get("tile_name", "")})
    table = pd.DataFrame(rows).set_index("episode")
    if len(table) >= MIN_EPISODES_FOR_SCORES:
        for metric, floor in SCORED_METRICS.items():
            values = table[metric].astype(float)
            median = values.median()
            spread = max(1.4826 * (values - median).abs().median(),
                         floor if floor is not None else RELATIVE_SPREAD_FLOOR * abs(median), 1e-9)
            table[f"z_{metric}"] = ((values - median) / spread).clip(lower=0)
        z = [f"z_{m}" for m in SCORED_METRICS]
        table["score"] = table[z].sum(axis=1)
        table["worst"] = table[z].idxmax(axis=1).str[2:].where(table["score"] > 0, "")
        table = table.sort_values("score", ascending=False)
    return table, events, fps
