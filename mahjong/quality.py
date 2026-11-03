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
from .dataset import EPISODE_LOG, build, episode_log

DROPPED_FILE = config.DATA / "dropped_episodes.json"     # {repo: {episode: reason}}: reviewed, never trained on
GRIPPER_FIXES_FILE = config.DATA / "episode_fixes.json"   # {repo: {episode: reason}}: gripper closed to 0 when built
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
    log = episode_log(repo_id)
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


# --- phases of a pickup, and the hover→grasp move (2026-10-26 golden set review) ---
#
# A demo: the gripper narrows at the hover, the operator waits, steers and descends (the approach), closes,
# then lifts and holds. Hesitation shows as pauses and reversals within the approach and as a path much
# longer than the straight line; "jitter" as wobble (the tip path less a 7-frame moving average), which
# was ~0.3 mm RMS in every recording so far: the stop-go of corrections, not noise a filter would remove.

PAUSE_MIN_S = 0.4              # a still stretch at least this long within the approach counts as a pause
WOBBLE_WINDOW = 7
STRAIGHT_DOWN_MM = 1.5         # a hover→grasp move under this: lowered straight down, didn't steer
DRIFT_RADIUS_M = 0.015         # reference grasps this close to a tile count as the same place
DRIFT_MIN_REFERENCE = 2        # ...and a spot needs this many that steered
JAW_TOLERANCE_MM = 6.0         # the jaws still close on a tile this far off; a larger shift matters
PHASE_COLUMNS = ["narrow_s", "wait_s", "approach_s", "pause_s", "approach_rev", "path_ratio", "after_s", "wobble_mm"]


def phases(joints, fps):
    """Seconds per phase of one pickup (narrowing, wait at the hover, approach to the close, after it), the
    pauses and reversals within the approach, its path over the straight line, and the tip's wobble (mm RMS).
    None if the gripper never shuts."""
    from .dataset import narrowing_end
    close = kinematics.grasp_frame(joints)
    if close is None:
        return None
    tips = np.array([kinematics.pincer_tip(dict(zip(config.JOINTS, q))) for q in joints]) * 1000
    speed = np.r_[0.0, np.linalg.norm(np.diff(tips, axis=0), axis=1) * fps]
    start = narrowing_end(joints)
    moving = np.flatnonzero(speed[start:close] >= IDLE_SPEED_MM_S)
    first = start + int(moving[0]) if len(moving) else close
    pause, run = 0, 0
    for still in speed[first:close] < IDLE_SPEED_MM_S:
        run = run + 1 if still else 0
        pause += (run == round(PAUSE_MIN_S * fps)) * run + (run > round(PAUSE_MIN_S * fps))
    approach = tips[first:close + 1]
    reversals = sum(len(turning_points(approach[:, k], REVERSAL_MM)) for k in range(3)) if len(approach) > 1 else 0
    path = float(np.linalg.norm(np.diff(approach, axis=0), axis=1).sum()) if len(approach) > 1 else 0.0
    straight = float(np.linalg.norm(approach[-1] - approach[0]))
    k = WOBBLE_WINDOW
    smooth = np.array([np.convolve(tips[:, i], np.ones(k) / k, mode="same") for i in range(3)]).T
    wobble = float(np.sqrt(np.mean(np.sum((tips - smooth)[k:-k] ** 2, axis=1)))) if len(tips) > 2 * k else 0.0
    return {"narrow_s": start / fps, "wait_s": (first - start) / fps, "approach_s": (close - first) / fps,
            "pause_s": pause / fps, "approach_rev": reversals, "path_ratio": path / max(straight, 1.0),
            "after_s": (len(joints) - close) / fps, "wobble_mm": wobble}


def hover_move(joints, jitter_xy=(0.0, 0.0)):
    """The pincer tip's x/y move from the hover (first frame) to the close, mm, with a deliberate hover
    offset added back — what the hover itself missed by. None if the gripper never shuts."""
    close = kinematics.grasp_frame(joints)
    if close is None:
        return None
    start, end = (kinematics.pincer_tip(dict(zip(config.JOINTS, joints[i]))) for i in (0, close))
    return (end[:2] - start[:2]) * 1000 + np.asarray(jitter_xy, float) * 1000


def load_episodes(repo_id):
    """{episode: (action_raw joints, its episode-log record)} and fps."""
    root = dataset_root(repo_id)
    df = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").glob("*/*.parquet"))], ignore_index=True)
    fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
    log = episode_log(repo_id)
    return {int(ep): (np.stack(f.sort_values("frame_index")["action_raw"].to_numpy()), log.get(int(ep), {}))
            for ep, f in df.groupby("episode_index")}, fps


def phase_table(repo_id):
    """Per-episode phases (see phases()) with each episode's slice and spot."""
    episodes, fps = load_episodes(repo_id)
    rows = []
    for ep, (joints, record) in episodes.items():
        p = phases(joints, fps)
        if p is not None:
            rows.append({"episode": ep, "slice": record.get("slice", "train"), "spot": spot_label(record), **p})
    return pd.DataFrame(rows).set_index("episode")


def spot_label(record):
    xy = record.get("detection_xy")
    return record.get("spot") or (f"{xy[0] * 100:.0f},{xy[1] * 100:+.0f}" if xy else "?")


def moves(episodes):
    """[(episode, record, move mm)] for episodes whose gripper shuts and whose log has the detection."""
    out = []
    for ep, (joints, record) in episodes.items():
        move = hover_move(joints, record.get("hover_jitter_xy", (0.0, 0.0)))
        if move is not None and record.get("detection_xy"):
            out.append((ep, record, move))
    return out


def drift_table(episodes, reference):
    """Per spot of `episodes`: its median hover→grasp move against the reference's at the same place (within
    DRIFT_RADIUS_M). Only demos that steered are compared: a demo lowered straight down (move under
    STRAIGHT_DOWN_MM) says "close enough for the jaws", not where the tile was — v2's demos did that a third
    of the time, which made a mat-wide comparison read as a 10-15 mm drift that wasn't there."""
    ref = moves(reference)
    ref_xy = np.array([r["detection_xy"] for _, r, _ in ref]) if ref else np.zeros((0, 2))
    ref_moves = np.array([m for _, _, m in ref]) if ref else np.zeros((0, 2))
    ref_steered = np.hypot(*ref_moves.T) >= STRAIGHT_DOWN_MM if ref else np.zeros(0, bool)
    by_spot = {}
    for ep, record, move in moves(episodes):
        by_spot.setdefault(spot_label(record), []).append((record["detection_xy"], move))
    rows = []
    for spot, items in by_spot.items():
        xy = np.median([i[0] for i in items], axis=0)
        mv = np.array([i[1] for i in items])
        steered = mv[np.hypot(*mv.T) >= STRAIGHT_DOWN_MM]
        near = np.hypot(*(ref_xy - xy).T) <= DRIFT_RADIUS_M if len(ref_xy) else np.zeros(0, bool)
        row = {"spot": spot, "episodes": len(mv), "straight_down": 1 - len(steered) / len(mv),
               "reference": int(near.sum()), "reference_straight_down": float(1 - ref_steered[near].mean()) if near.any() else np.nan}
        ok = len(steered) and (near & ref_steered).sum() >= DRIFT_MIN_REFERENCE
        mine = np.median(steered, axis=0) if len(steered) else (np.nan, np.nan)
        theirs = np.median(ref_moves[near & ref_steered], axis=0) if ok else (np.nan, np.nan)
        rows.append({**row, "move_x": mine[0], "move_y": mine[1], "ref_x": theirs[0], "ref_y": theirs[1],
                     "shift_x": mine[0] - theirs[0], "shift_y": mine[1] - theirs[1]})
    return pd.DataFrame(rows).set_index("spot").sort_index()


# --- a cleaned copy: the recording itself stays as it was ---

def clean_copy(repo_id, dst_repo, log=print):
    """Write dst_repo: repo_id at its own image size, without its dropped episodes and with every stop cut
    (dataset.pause_mask). Its episode log says which source episode each came from. Returns dataset.build's summary."""
    root = dataset_root(repo_id)
    info = json.loads((root / "meta" / "info.json").read_text())
    dropped = dropped_episodes(repo_id)
    episodes = [e for e in range(info["total_episodes"]) if e not in dropped]
    size = tuple(info["features"]["observation.images.wrist"]["shape"][:2])
    summary = build(repo_id, dst_repo, trim_idle=False, trim_pauses=True, size=size, episodes=episodes, log=log)
    records = episode_log(repo_id)
    rows = [{**records.get(source, {}), "episode_index": built, "source_repo": repo_id, "source_episode": source,
             "source_frames": records.get(source, {}).get("frames")}
            for built, source in enumerate(summary["source_episodes"])]
    (dataset_root(dst_repo) / EPISODE_LOG).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (dataset_root(dst_repo) / "clean_summary.json").write_text(
        json.dumps({"source": repo_id, "dropped": {str(e): r for e, r in dropped.items()}, **summary}, indent=2) + "\n")
    return summary


# --- episodes dropped after review: listed, not deleted, so a dataset still being recorded into keeps its
# episode numbers (and its episode log's); make_policy.py leaves them out of training ---

def dropped_episodes(repo_id, path=None):
    """{episode: reason} dropped from repo_id after review."""
    return _listed(repo_id, Path(path or DROPPED_FILE))


def drop_episodes(repo_id, episodes, reason, path=None):
    """Add episodes to repo_id's dropped list (reason kept with each). Returns the whole list."""
    return _list(repo_id, episodes, reason, Path(path or DROPPED_FILE))


def gripper_fixes(repo_id, path=None):
    """{episode: reason} whose gripper command dataset.build carries on to 0 (dataset.finish_close)."""
    return _listed(repo_id, Path(path or GRIPPER_FIXES_FILE))


def fix_gripper(repo_id, episodes, reason, path=None):
    """Mark episodes whose operator stopped short of closing the gripper fully (and nothing else was wrong)."""
    return _list(repo_id, episodes, reason, Path(path or GRIPPER_FIXES_FILE))


def _listed(repo_id, path):
    if not path.exists():
        return {}
    return {int(ep): reason for ep, reason in json.loads(path.read_text()).get(repo_id, {}).items()}


def _list(repo_id, episodes, reason, path):
    data = json.loads(path.read_text()) if path.exists() else {}
    entry = data.setdefault(repo_id, {})
    for ep in episodes:
        entry[str(int(ep))] = reason
    data[repo_id] = dict(sorted(entry.items(), key=lambda kv: int(kv[0])))
    path.write_text(json.dumps(data, indent=2) + "\n")
    return _listed(repo_id, path)


def undrop_episodes(repo_id, episodes, path=None):
    """Take episodes off repo_id's dropped list (e.g. once a fix makes them usable)."""
    path = Path(path or DROPPED_FILE)
    data = json.loads(path.read_text()) if path.exists() else {}
    for ep in episodes:
        data.get(repo_id, {}).pop(str(int(ep)), None)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return _listed(repo_id, path)
