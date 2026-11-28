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


# --- a tile turned sideways: the wrist rolls to meet it (2026-11-28) ---
#
# Face-down backs have no detected angle (record.py logs placed_yaw_deg None for them), so which demos picked a tile turned
# on purpose is read from the wrist roll at the close, less the episode's first frame. Upright tiles: under 20 deg in every
# set checked; partly turned ones 31-41 deg (mahjong_facedown_dagger_v14 38-40); sideways ones 85-103 deg (daytime 29, 31,
# 37, 39, 43, 45, 49, 55; dagger_v14 41-44). A turn takes longer and lands the close elsewhere, so the hesitation scores
# and the grip reach, both fitted on upright tiles, would read a deliberate turn as a bad demo: sideways episodes are marked,
# shown, and never flagged or proposed on those numbers. "turned" (20-60 deg) is only shown: jitter_v3 and dagger_v9/v10 have
# 37 of them, both ways (-45 to +55 deg), not checked by eye, so they're still judged as before. Training is unchanged.
TURNED_ROLL_DEG = 20.0
SIDEWAYS_ROLL_DEG = 60.0
SIDEWAYS = "sideways"


def roll_at_close(actions, record=None):
    """Degrees the wrist rolled from the episode's first frame to the close, or None without one. A DAgger episode's close
    is the person's, after the takeover (dataset.takeover_frame); its roll still counts from the first frame, since the
    tile's turn is what's measured, whoever started rolling toward it."""
    from .dataset import takeover_frame
    actions = np.asarray(actions)
    start = takeover_frame(record or {})[1] or 0
    close = kinematics.grasp_frame(actions[start:])
    if close is None:
        return None
    roll = config.JOINTS.index("wrist_roll")
    return float(actions[start + close, roll] - actions[0, roll])


def tile_turn(roll_deg):
    """"sideways", "turned" or "" (upright) for a roll at the close (roll_at_close); "" without one."""
    if roll_deg is None or np.isnan(roll_deg):
        return ""
    return SIDEWAYS if abs(roll_deg) >= SIDEWAYS_ROLL_DEG else "turned" if abs(roll_deg) >= TURNED_ROLL_DEG else ""


def video_starts(repo_id):
    """{episode: (video file under the dataset's folder, start in seconds)} for its wrist camera (else its first), from
    meta/episodes; {} if the dataset doesn't say."""
    root = dataset_root(repo_id)
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        return {}
    meta = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cameras = sorted(c[len("videos/"):-len("/file_index")] for c in meta.columns
                     if c.startswith("videos/") and c.endswith("/file_index"))
    if not cameras:
        return {}
    cam = next((c for c in cameras if "wrist" in c), cameras[0])
    col = lambda name: meta[f"videos/{cam}/{name}"]
    return {int(ep): (f"videos/{cam}/chunk-{int(chunk):03d}/file-{int(file):03d}.mp4", float(start))
            for ep, chunk, file, start in zip(meta["episode_index"], col("chunk_index"), col("file_index"),
                                              col("from_timestamp"))}


def clock(seconds):
    """m:ss.s, a video player's time."""
    return "-" if seconds != seconds else f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def turn_table(repo_id):
    """Per episode: spot, roll at the close (deg), its mark (tile_turn) and where its video starts."""
    root = dataset_root(repo_id)
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"no recording at {root}")
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action_raw"]) for f in files],
                   ignore_index=True)
    log, starts = episode_log(repo_id), video_starts(repo_id)
    rows = []
    for ep, frames in df.groupby("episode_index"):
        record = log.get(int(ep), {})
        roll = roll_at_close(np.stack(frames.sort_values("frame_index")["action_raw"].to_numpy()), record)
        video, start = starts.get(int(ep), ("", float("nan")))
        rows.append({"episode": int(ep), "spot": record.get("spot") or "", "roll_deg": roll, "turn": tile_turn(roll),
                     "video": video, "start_s": start})
    return pd.DataFrame(rows).set_index("episode")


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
        record = log.get(episode, {})
        rows.append({"episode": episode, **metrics, "tile": record.get("tile_name", ""),
                     "turn": tile_turn(roll_at_close(joints, record))})
    table = pd.DataFrame(rows).set_index("episode")
    upright = table["turn"] != SIDEWAYS    # a sideways tile's longer pickup isn't hesitation: the baseline leaves it out
    if upright.sum() >= MIN_EPISODES_FOR_SCORES:
        for metric, floor in SCORED_METRICS.items():
            values = table[metric].astype(float)
            median = values[upright].median()
            spread = max(1.4826 * (values[upright] - median).abs().median(),
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
    starts = video_starts(repo_id)
    rows = []
    for ep, (joints, record) in episodes.items():
        p = phases(joints, fps)
        if p is not None:
            video, start = starts.get(int(ep), ("", float("nan")))
            rows.append({"episode": ep, "slice": record.get("slice", "train"), "spot": spot_label(record), **p,
                         "turn": tile_turn(roll_at_close(joints, record)), "video": video, "start_s": start})
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


# --- grip reach: did a demo close short of the tile's middle, toward the robot ("gripped low", 2026-11-23) ---
#
# The owner's worry: v13 grips near the tile's robot-side end, which works on a tile where it was put but may not on one that
# drifted outward in a game (docs/retrospectives/2026-11-23_v13_grip_and_wrist_calibration.md). The measure is where the jaws
# closed in robot x (kinematics.grasp_point: the commanded tool point) less the tile's detected x: the bench logs the same two
# numbers (`grasp_xy`, `detection_xy`). It sits around +40 mm because the tool point isn't the jaws' middle, and it depends on
# where the tile is, so an episode is judged against the value expected at its tile's position. That expectation is fitted on
# the face-up bench attempts the owner didn't mark gripped low (`d`, interventions "grip_low"): a close is judged by how far it
# strayed from the usual place, not by its distance from the detection (demos leave a margin on purpose). Fitted 2026-11-23 on
# 399 attempts: unmarked median shortfall -0.1 mm, marked-low -5.5 mm; at 4 mm short, 68% of marked-low and 16% of unmarked.

REACH_SHORT_MM = 4.0            # this far short of the reference: proposed for dropping
REACH_SPOT_ONLY_SHORT_MM = 6.0  # ...for an episode whose log has the spot's saved position, not a detection (v8-v10 face down)
REACH_MARGIN_M = 0.010          # a tile further than this outside the bench attempts' extent has no reference
REACH_SINCE = "20261118"        # bench sessions from here on: the owner marked low grips closely from 11-18 (the retro)
REACH_THIN_SPOT = 3             # warn when a spot would be left with fewer episodes than this...
REACH_THIN_SHARE = 1 / 3        # ...or lose more than this share of them
# Lone tiles grip ~10 mm further out and ~12 mm higher than in a full layout (the retro's finding 3): not the usual place.
# The setup answers can't tell them apart (the runs with neighbours that evening also said tiles "one").
LONE_TILE_SESSIONS = {"20261123_203643_bench", "20261123_204256_bench", "20261123_210705_bench"}


def reach_features(xy):
    """The fit's terms for tile positions (n x 2, m): 1, x, y, y² (the reach grows toward both ends of a row)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    return np.c_[np.ones(len(xy)), xy[:, 0], xy[:, 1], xy[:, 1] ** 2]


def bench_reach_labels(sessions_dir=None, since=REACH_SINCE):
    """Face-up bench attempts with a policy close and a detection, from sessions on or after `since` where the owner pressed
    `d` (gripped low) at least once, so an unmarked attempt means "not low" rather than "not marked". Sim, voided and lone-tile
    sessions are left out, and so are attempts a person took over (their grasp_xy is the policy's, the person's grip is what
    was marked). [{session, attempt_dir, policy, spot, detection_xy, reach_mm, low}]."""
    from .session import read_jsonl, void_reason
    root = Path(sessions_dir or config.SESSIONS)
    rows = []
    for folder in sorted(root.glob("*_bench")):
        if folder.name[:8] < since or folder.name in LONE_TILE_SESSIONS or void_reason(folder):
            continue
        events = read_jsonl(folder / "events.jsonl")
        if any(e.get("type") == "session_start" and e.get("sim") for e in events):
            continue
        low = {e["attempt_dir"] for e in events if e.get("type") == "keys" and e.get("attempt_dir")
               and any(p.get("kind") == "grip_low" for p in e.get("presses", []))}
        if not low:
            continue
        for e in events:
            if (e.get("type") != "attempt" or not e.get("attempt_dir") or e.get("face_down") or not e.get("grasp_xy")
                    or not e.get("detection_xy") or e.get("person_grasp_xy")):
                continue
            rows.append({"session": folder.name, "attempt_dir": e.get("attempt_dir"), "policy": e.get("policy"),
                         "spot": e.get("spot"), "detection_xy": e["detection_xy"],
                         "reach_mm": (e["grasp_xy"][0] - e["detection_xy"][0]) * 1000, "low": e.get("attempt_dir") in low})
    return rows


class ReachReference:
    """The reach expected at a tile position, fitted (least squares on reach_features) on bench attempts not marked low."""

    def __init__(self, labels):
        ok = [r for r in labels if not r["low"]]
        if len(ok) < 2 * reach_features([[0, 0]]).shape[1]:
            raise ValueError(f"{len(ok)} unmarked bench attempts: too few to fit a reference")
        xy = np.array([r["detection_xy"] for r in ok], float)
        self.coef, *_ = np.linalg.lstsq(reach_features(xy), np.array([r["reach_mm"] for r in ok]), rcond=None)
        self.lo, self.hi = xy.min(axis=0) - REACH_MARGIN_M, xy.max(axis=0) + REACH_MARGIN_M
        self.attempts, self.marked_low = len(labels), sum(r["low"] for r in labels)
        self.sessions = len({r["session"] for r in labels})

    def expected(self, xy):
        return float((reach_features(xy) @ self.coef)[0])

    def covers(self, xy):
        xy = np.asarray(xy, float)
        return bool(np.all(xy >= self.lo) and np.all(xy <= self.hi))

    def shortfall(self, xy, reach_mm):
        """reach less the expected: negative is short of the usual place, toward the robot."""
        return reach_mm - self.expected(xy)


def reach_calibration(labels, reference, thresholds=(2, 3, 4, 5, 6)):
    """[(mm short, share of marked-low attempts caught, share of unmarked flagged)] on the bench labels: the cutoff's trade-off.
    In-sample for the unmarked ones (the fit saw them), so their share is a little low."""
    short = np.array([reference.shortfall(r["detection_xy"], r["reach_mm"]) for r in labels])
    low = np.array([r["low"] for r in labels])
    return [(t, float((short[low] <= -t).mean()) if low.any() else np.nan,
             float((short[~low] <= -t).mean()) if (~low).any() else np.nan) for t in thresholds]


def episode_reach(actions, states, record):
    """(reach mm, close height mm) of one recorded episode: tool point x at the close less the tile's x, and the pincer tip
    over the grasp point as the jaws closed. A DAgger episode is measured from the takeover on, the frames it's trained on
    (dataset.takeover_frame); before that the policy may have closed on its own. (None, None) without a close, a detection,
    or (DAgger) a takeover."""
    from .dataset import takeover_frame
    dagger, takeover = takeover_frame(record)
    if (dagger and takeover is None) or not record.get("detection_xy"):
        return None, None
    start = takeover or 0
    _, xy = kinematics.grasp_point(actions[start:])
    if xy is None:
        return None, None
    clearance = kinematics.grasp_clearance(states[start:], actions[start:])
    return (xy[0] - record["detection_xy"][0]) * 1000, None if clearance is None else clearance * 1000


def reach_row(episode, record, reach_mm, height_mm, reference, excluded=(), dropped=(), turn=""):
    """One episode's verdict: its shortfall, how far the reference can be trusted there (`basis`), and whether it's proposed.
    A sideways tile (`turn`, tile_turn) closes elsewhere than the upright tiles the reference is fitted on: its basis is
    "sideways", and it's never proposed. A partly turned one is judged as before."""
    sideways = turn == SIDEWAYS
    row = {"episode": int(episode), "spot": record.get("spot") or "", "slice": record.get("slice", "train"),
           "tile": record.get("tile_name", ""), "reach_mm": reach_mm, "close_mm": height_mm, "shortfall_mm": np.nan,
           "heldout": int(episode) in excluded, "dropped": int(episode) in dropped, "propose": False}
    xy = record.get("detection_xy")
    if reach_mm is None:
        row["basis"] = "no close"
    elif not reference.covers(xy):
        row["basis"] = "outside"
    else:
        row["shortfall_mm"] = reference.shortfall(xy, reach_mm)
        spot_only = record.get("placement_status") == "saved_position"
        row["basis"] = SIDEWAYS if sideways else "spot-only" if spot_only else "measured"
    limit = REACH_SPOT_ONLY_SHORT_MM if row["basis"] == "spot-only" else REACH_SHORT_MM
    row["propose"] = bool(row["shortfall_mm"] <= -limit and row["slice"] == "train" and not row["heldout"]
                          and not row["dropped"] and not sideways)
    return row


def grip_reach_table(repo_id, reference):
    """Per episode of a recording (see reach_row), from action_raw / observation.state_raw and its episode log, with its
    held-out episodes (offline_eval.excluded_episodes) and earlier drops marked, never proposed."""
    from .offline_eval import excluded_episodes
    root = dataset_root(repo_id)
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"no recording at {root}")
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action_raw", "observation.state_raw"])
                    for f in files], ignore_index=True)
    log, excluded, dropped = episode_log(repo_id), set(excluded_episodes(repo_id)), set(dropped_episodes(repo_id))
    rows = []
    for ep, frames in df.groupby("episode_index"):
        frames = frames.sort_values("frame_index")
        record = log.get(int(ep), {})
        actions = np.stack(frames["action_raw"].to_numpy())
        reach, height = episode_reach(actions, np.stack(frames["observation.state_raw"].to_numpy()), record)
        rows.append(reach_row(ep, record, reach, height, reference, excluded, dropped,
                              turn=tile_turn(roll_at_close(actions, record))))
    return pd.DataFrame(rows).set_index("episode")


def thin_spots(table):
    """{spot: (episodes trained on now, left after the proposed drops)} for named spots the drops would leave thin."""
    named = table[(table["spot"] != "") & (table["slice"] == "train") & ~table["dropped"] & ~table["heldout"]]
    out = {}
    for spot, group in named.groupby("spot"):
        now, left = len(group), int((~group["propose"]).sum())
        if left < now and (left < REACH_THIN_SPOT or (now - left) > REACH_THIN_SHARE * now):
            out[spot] = (now, left)
    return out


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


def all_dropped(path=None):
    """{repo: {episode: reason}} for every dataset with an episode dropped after review."""
    path = Path(path or DROPPED_FILE)
    repos = json.loads(path.read_text()) if path.exists() else {}
    return {repo: _listed(repo, path) for repo, entry in repos.items() if entry}     # undropping leaves empty entries


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
