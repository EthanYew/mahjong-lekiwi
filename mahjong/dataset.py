"""Turning a recorded dataset into a training dataset: trims, re-anchoring and the 224 resize.

Trims (each a per-episode frame mask, applied in order):
- idle: frames where no action dimension (gripper included) changes by IDLE_DELTA within
  +-IDLE_WINDOW frames. A policy has no clock, so still frames all look alike, and the v1 policy
  learned to wait at the hover indefinitely from them (30% of the first dataset's frames).
- takeover sag: in episodes recorded before the 2026-10-22 takeover fix (first action == first state),
  the opening frames until the pincer tip stops sagging.
- gripper narrowing (optional): the opening phase where the jaws are narrowed from the hover's 90% to
  ~21% before descending. Trimming it makes episodes start where that ends, so they can be mixed with
  episodes recorded from a ~23% hover.
- DAgger (record.py --policy): frames before the takeover are the policy's own, not a demo; they're
  dropped, and an episode with no takeover is left out.
Deltas (observation.state, action) are re-anchored to each episode's first kept frame, so every
episode starts from a zero state delta, as deployment does at the hover. A DAgger episode stays
anchored at its first frame, the hover: deployed, the policy is at that point in the pickup with a state
delta from the hover too.

`episodes` picks which source episodes go in: make_policy.py leaves out the held-out ones
(data/eval_split.json), offline_eval.py builds only those. `extra` adds episodes of other recorded
datasets (make_policy.py --add, e.g. DAgger corrections) after them.
"""
from pathlib import Path

import numpy as np

from . import config, kinematics
from .session import read_jsonl

IDLE_WINDOW = 7
IDLE_DELTA = 0.5
SAG_SETTLE_MM, SAG_SETTLE_FRAMES, SAG_MAX_FRAMES = 0.3, 9, 60
NARROWING_TOLERANCE = 2.0      # gripper % of its pre-grasp opening that counts as "narrowed"
EPISODE_LOG = "episode_log.jsonl"                          # record.py's sidecar log: one row per saved episode
EPISODE_LOGS = (EPISODE_LOG, "episode_teleop_log.jsonl")   # ...and the pre-rebuild recorder's


def idle_mask(actions):
    n = len(actions)
    return np.array([np.abs(actions[max(0, i - IDLE_WINDOW):i + IDLE_WINDOW + 1] - actions[i]).max() < IDLE_DELTA
                     for i in range(n)])


def is_pre_takeover_fix(states_raw, actions_raw):
    return bool(np.abs(actions_raw[0] - states_raw[0]).max() < 1e-3)


def takeover_sag_end(states_raw):
    """First frame from which the pincer tip holds within SAG_SETTLE_MM for SAG_SETTLE_FRAMES."""
    z = np.array([kinematics.pincer_tip(dict(zip(config.JOINTS, s)))[2] * 1000
                  for s in states_raw[:SAG_MAX_FRAMES + SAG_SETTLE_FRAMES]])
    for k in range(min(SAG_MAX_FRAMES, len(z) - SAG_SETTLE_FRAMES)):
        if np.abs(z[k:k + SAG_SETTLE_FRAMES] - z[k]).max() < SAG_SETTLE_MM:
            return k
    return SAG_MAX_FRAMES


def narrowing_end(actions_raw):
    """First frame where the commanded gripper is within NARROWING_TOLERANCE of the opening it held
    just before closing on the tile (0 if the episode never narrows from a wider start)."""
    grip = actions_raw[:, 5]
    close = int(np.argmax(grip < config.GRIP_CLOSED_BELOW))
    if close == 0:
        return 0
    plateau = float(np.median(grip[max(0, close - 30):max(1, close - 15)]))
    if grip[0] <= plateau + NARROWING_TOLERANCE:
        return 0
    return int(np.argmax(grip <= plateau + NARROWING_TOLERANCE))


def finish_close(grip, default_rate=1.67):
    """A commanded gripper series that stopped short of 0 on a tile, carried on to 0 at the rate it was
    closing (default: ZL at the default speed, %/frame) and held there until it opens again. The jaws stall
    on the tile at ~11 either way, so the frames still show what happened; a policy learns to command 0
    (CLAUDE.md: policies grip by commanding 0). Unchanged if it already reaches 0."""
    grip = np.asarray(grip, dtype=float).copy()
    low = int(np.argmin(grip))
    if grip[low] <= 0.5:
        return grip
    falling = -np.diff(grip[max(0, low - 10):low + 1])
    rate = float(np.median(falling[falling > 0])) if (falling > 0).any() else default_rate
    for i in range(low + 1, len(grip)):
        if grip[i] > grip[low] + 0.5:          # opened again: leave the rest alone
            break
        grip[i] = max(0.0, grip[low] - rate * (i - low))
    return grip


def keep_mask(states_raw, actions_raw, actions, trim_idle=True, trim_sag=True, trim_narrowing=False, start=0):
    """start: the first frame that can be kept (a DAgger episode's takeover)."""
    drop = idle_mask(actions) if trim_idle else np.zeros(len(actions), bool)
    drop[:start] = True
    if trim_sag and is_pre_takeover_fix(states_raw, actions_raw):
        drop[:takeover_sag_end(states_raw)] = True
    if trim_narrowing:
        drop[:narrowing_end(actions_raw)] = True
    return ~drop


def takeover_frame(record):
    """(is a DAgger episode, frame the person took over at or None) from its episode-log record."""
    return "policy" in record, record.get("takeover_frame")


def build(src_repo, dst_repo, trim_idle=True, trim_sag=True, trim_narrowing=False, size=(224, 224), episodes=None,
          close_gripper=(), extra=(), log=print):
    """Write dst_repo: src_repo's wrist camera at `size`, trimmed and re-anchored. episodes: the source
    episode indices to include (default all). extra: [(repo, episodes, close_gripper)] of other recorded
    datasets to add after them. Returns a summary; its source_episodes maps each built episode of src_repo
    back to its source index (and so to the source's episode log), sources every built one to
    [repo, index], and dagger_episodes lists the built DAgger ones (they start at a takeover, not at a
    zero state delta)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    src = LeRobotDataset(src_repo)
    if (HF_LEROBOT_HOME / dst_repo).exists():
        raise FileExistsError(f"{HF_LEROBOT_HOME / dst_repo} exists — choose a new name or delete it")
    features = {k: dict(v) for k, v in src.meta.features.items()
                if k in ("observation.state", "action", "observation.state_raw", "action_raw", "observation.images.wrist")}
    features["observation.images.wrist"]["shape"] = (size[0], size[1], 3)
    dst = LeRobotDataset.create(repo_id=dst_repo, fps=src.fps, features=features, robot_type=src.meta.robot_type,
                                use_videos=True)
    trims = dict(trim_idle=trim_idle, trim_sag=trim_sag, trim_narrowing=trim_narrowing)
    sources, dagger, source_frames, start_grippers = [], [], 0, []
    for repo, eps, closing in [(src_repo, episodes, close_gripper), *extra]:
        ds = src if repo == src_repo else LeRobotDataset(repo)
        if ds.fps != src.fps:
            raise ValueError(f"{repo} is {ds.fps} fps, {src_repo} {src.fps}")
        built, takeovers, frames, grippers = _add_episodes(dst, ds, repo, eps, set(closing), size, trims, log,
                                                 label="" if repo == src_repo else f"{repo} ")
        dagger += [len(sources) + k for k, t in enumerate(takeovers) if t is not None]
        sources += [[repo, ep] for ep in built]
        source_frames += frames
        start_grippers += grippers
    dst.finalize()
    return {"episodes": len(sources), "source_episodes": [ep for repo, ep in sources if repo == src_repo],
            "sources": sources, "dagger_episodes": dagger, "frames": int(dst.meta.total_frames), "source_frames": source_frames,
            "start_gripper_median": float(np.median(start_grippers)), "root": str(dst.root)}


def _add_episodes(dst, src, repo, episodes, close_gripper, size, trims, log, label=""):
    """Add src's `episodes` (default all) to dst. Returns (episodes built, each one's takeover frame or
    None, their source frames, each one's start gripper)."""
    import torchvision.transforms.functional as F
    h = src.hf_dataset
    actions, actions_raw, states_raw = (np.stack(h[k]) for k in ("action", "action_raw", "observation.state_raw"))
    actions, actions_raw = actions.copy(), actions_raw.copy()
    records = episode_log(repo)
    episodes = list(range(src.num_episodes)) if episodes is None else sorted(int(e) for e in episodes)
    built, takeovers, source_frames, start_grippers = [], [], 0, []
    for n, ep in enumerate(episodes, start=1):
        meta = src.meta.episodes[ep]
        a, b = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        dagger, takeover = takeover_frame(records.get(ep, {}))
        if dagger and takeover is None:
            log(f"  {label}episode {ep} ({n}/{len(episodes)}): left out, the policy's alone (no takeover)")
            continue
        if ep in close_gripper:                # stopped short of 0 on the tile (quality.gripper_fixes)
            fixed = finish_close(actions_raw[a:b, 5])
            actions[a:b, 5] += fixed - actions_raw[a:b, 5]
            actions_raw[a:b, 5] = fixed
        keep = np.flatnonzero(keep_mask(states_raw[a:b], actions_raw[a:b], actions[a:b], trims["trim_idle"],
                                        trims["trim_sag"] and not dagger, trims["trim_narrowing"], start=takeover or 0))
        if not len(keep):
            log(f"  {label}episode {ep} ({n}/{len(episodes)}): left out, nothing kept")
            continue
        anchor = states_raw[a] if dagger else states_raw[a + keep[0]]
        start_grippers.append(float(anchor[5]))
        for i in keep:
            item = src[a + int(i)]
            image = F.resize(item["observation.images.wrist"], list(size), antialias=True)
            dst.add_frame({
                "observation.state": (states_raw[a + i] - anchor).astype("float32"),
                "action": (actions_raw[a + i] - anchor).astype("float32"),
                "observation.state_raw": states_raw[a + i].astype("float32"),
                "action_raw": actions_raw[a + i].astype("float32"),
                "observation.images.wrist": (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8"),
                "task": item["task"]})
        dst.save_episode()
        built.append(ep)
        takeovers.append(takeover)
        source_frames += b - a
        log(f"  {label}episode {ep} ({n}/{len(episodes)}): kept {len(keep)}/{b - a}"
            + (f" (from the takeover at frame {takeover})" if dagger else ""))
    return built, takeovers, source_frames, start_grippers


def episode_log(repo_id):
    """{episode index: record} from a recorded dataset's sidecar log (EPISODE_LOGS: record.py's, and the
    pre-rebuild recorder's), or {}."""
    from lerobot.utils.constants import HF_LEROBOT_HOME
    return {record["episode_index"]: record for name in EPISODE_LOGS
            for record in read_jsonl(Path(HF_LEROBOT_HOME) / repo_id / name)}


def hover_settings(src_repo, built_summary):
    """The hover a policy trained on the built dataset should be deployed from: offset and clearance
    as recorded (episode_log.jsonl; config defaults for datasets recorded before it existed), gripper
    opening as at the trimmed episodes' first frame."""
    offset, clearance = config.HOVER_OFFSET_M, config.HOVER_CLEARANCE_M
    hovers = [r["hover"] for r in episode_log(src_repo).values() if r.get("hover")]
    if hovers:
        offset = float(np.median([h["offset_m"] for h in hovers]))
        clearance = float(np.median([h["clearance_m"] for h in hovers]))
    return {"offset_m": offset, "clearance_m": clearance, "gripper": round(built_summary["start_gripper_median"], 1)}
