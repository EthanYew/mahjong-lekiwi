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
- pauses (optional, quality.clean_copy): every stop cut to its last frame, however short. The idle trim keeps
  IDLE_WINDOW frames on each side of a stop and misses any under ~0.5s, so half the still frames of a stop-go demo
  survive it (mahjong_surfaces_standard_mat_eval_v1, 2026-11-03). Only frames where nothing is commanded go, so
  the commands stay continuous; a move backwards stays, since cutting it would make the arm jump.
- DAgger (record.py --policy): frames before the takeover are the policy's own, not a demo; they're
  dropped, and an episode with no takeover is left out.
Deltas (observation.state, action) are re-anchored to each episode's first kept frame, so every
episode starts from a zero state delta, as deployment does at the hover. A DAgger episode stays
anchored at its first frame, the hover: deployed, the policy is at that point in the pickup with a state
delta from the hover too.

`episodes` picks which source episodes go in: make_policy.py leaves out the held-out ones
(data/eval_split.json), offline_eval.py builds only those. `extra` adds episodes of other recorded
datasets (make_policy.py --add, e.g. DAgger corrections) after them.

replace_mat (make_policy.py --replace-mat): every recording so far is on the black table top, and v5 closed its
jaws in mid-air over a green grid mat (2026-10-29, 20261029_180643_bench). In that fraction of the episodes
recorded on the black table top (others are already another surface, and the mask is tuned to the black top:
a green mat would pass for tile backs) the mat in every wrist frame is swapped for another surface (surface_keep, random_surface), one surface per
episode, shaded by the original frame so tile shadows and the lamp's glare stay.
"""
import zlib
from pathlib import Path

import numpy as np

from . import config, kinematics
from .session import read_jsonl

IDLE_WINDOW = 7
IDLE_DELTA = 0.5
SAG_SETTLE_MM, SAG_SETTLE_FRAMES, SAG_MAX_FRAMES = 0.3, 9, 60
PAUSE_DELTA = 0.02             # a joint's command changing less than this between frames isn't moving
PAUSE_MIN_FRAMES = 4           # a stop at least this long (~0.13s) is cut: shorter is the stick's own stepping
NARROWING_TOLERANCE = 2.0      # gripper % of its pre-grasp opening that counts as "narrowed"
EPISODE_LOG = "episode_log.jsonl"                          # record.py's sidecar log: one row per saved episode
EPISODE_LOGS = (EPISODE_LOG, "episode_teleop_log.jsonl")   # ...and the pre-rebuild recorder's


def idle_mask(actions):
    n = len(actions)
    return np.array([np.abs(actions[max(0, i - IDLE_WINDOW):i + IDLE_WINDOW + 1] - actions[i]).max() < IDLE_DELTA
                     for i in range(n)])


def pause_mask(actions_raw):
    """True on frames to cut: the frames of a stop of at least PAUSE_MIN_FRAMES where no joint's command moves,
    all but its last, which shows what the arm settled to as it moves on. Frame 0 stays: deltas are anchored there."""
    actions_raw = np.asarray(actions_raw)
    still = np.r_[False, np.abs(np.diff(actions_raw, axis=0)).max(axis=1) < PAUSE_DELTA]
    cut = np.zeros(len(actions_raw), bool)
    i = 0
    while i < len(still):
        if not still[i]:
            i += 1
            continue
        end = i
        while end + 1 < len(still) and still[end + 1]:
            end += 1
        if end - i + 1 >= PAUSE_MIN_FRAMES:
            cut[i:end] = True
        i = end + 1
    return cut


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
    grip = actions_raw[:, config.GRIPPER]
    close = kinematics.grasp_frame(actions_raw) or 0
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


def keep_mask(states_raw, actions_raw, actions, trim_idle=True, trim_sag=True, trim_narrowing=False, start=0,
              trim_pauses=False):
    """start: the first frame that can be kept (a DAgger episode's takeover). Idle is checked only from
    there on, so the window doesn't reach into the dropped pre-takeover frames the policy drove (a
    different enough regime to mask real idling right after the takeover). The same goes for a trimmed
    narrowing: a still frame just after it would count as moving only because of frames that are dropped."""
    drop = np.zeros(len(actions), bool)
    first = max(start, narrowing_end(actions_raw)) if trim_narrowing else start
    if trim_idle:
        drop[first:] = idle_mask(actions[first:])
    if trim_pauses:
        drop |= pause_mask(actions_raw)
    drop[:first] = True
    if trim_sag and is_pre_takeover_fix(states_raw, actions_raw):
        drop[:takeover_sag_end(states_raw)] = True
    return ~drop


def takeover_frame(record):
    """(is a DAgger episode, frame the person took over at or None) from its episode-log record."""
    return "policy" in record, record.get("takeover_frame")


def build(src_repo, dst_repo, trim_idle=True, trim_sag=True, trim_narrowing=False, size=(224, 224), episodes=None,
          close_gripper=(), extra=(), replace_mat=0.0, seed=0, log=print, trim_pauses=False):
    """Write dst_repo: src_repo's wrist camera at `size`, trimmed and re-anchored. episodes: the source
    episode indices to include (default all). extra: [(repo, episodes, close_gripper)] of other recorded
    datasets to add after them. Returns a summary; its source_episodes maps each built episode of src_repo
    back to its source index (and so to the source's episode log), sources every built one to
    [repo, index], and dagger_episodes lists the built DAgger ones (they start at a takeover, not at a
    zero state delta). replace_mat: the fraction of episodes whose mat is swapped (module docstring); the
    summary's surfaces names each built episode's surface (None: the black table top as recorded)."""
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
    trims = dict(trim_idle=trim_idle, trim_sag=trim_sag, trim_narrowing=trim_narrowing, trim_pauses=trim_pauses)
    surfacing = dict(fraction=replace_mat, seed=seed)
    sources, dagger, source_frames, start_grippers, surfaces = [], [], 0, [], []
    for repo, eps, closing in [(src_repo, episodes, close_gripper), *extra]:
        ds = src if repo == src_repo else LeRobotDataset(repo)
        if ds.fps != src.fps:
            raise ValueError(f"{repo} is {ds.fps} fps, {src_repo} {src.fps}")
        built, takeovers, frames, grippers, kinds = _add_episodes(dst, ds, repo, eps, set(closing), size, trims, log,
                                                         label="" if repo == src_repo else f"{repo} ",
                                                         surfacing=surfacing)
        dagger += [len(sources) + k for k, t in enumerate(takeovers) if t is not None]
        sources += [[repo, ep] for ep in built]
        source_frames += frames
        if repo == src_repo:        # the hover it deploys from is the source's, however many episodes --add brings
            start_grippers += grippers
        surfaces += kinds
    dst.finalize()
    return {"episodes": len(sources), "source_episodes": [ep for repo, ep in sources if repo == src_repo],
            "sources": sources, "dagger_episodes": dagger, "frames": int(dst.meta.total_frames), "source_frames": source_frames,
            "start_gripper_median": float(np.median(start_grippers)), "root": str(dst.root),
            "replace_mat": replace_mat, "surfaces": surfaces}


def _add_episodes(dst, src, repo, episodes, close_gripper, size, trims, log, label="", surfacing=None):
    """Add src's `episodes` (default all) to dst. Returns (episodes built, each one's takeover frame or
    None, their source frames, each one's start gripper, each one's surface or None)."""
    import torchvision.transforms.functional as F
    h = src.hf_dataset
    actions, actions_raw, states_raw = (np.stack(h[k]) for k in ("action", "action_raw", "observation.state_raw"))
    actions, actions_raw = actions.copy(), actions_raw.copy()
    records = episode_log(repo)
    episodes = list(range(src.num_episodes)) if episodes is None else sorted(int(e) for e in episodes)
    built, takeovers, source_frames, start_grippers, kinds = [], [], 0, [], []
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
                                        trims["trim_sag"] and not dagger, trims["trim_narrowing"], start=takeover or 0,
                                        trim_pauses=trims["trim_pauses"]))
        if not len(keep):
            log(f"  {label}episode {ep} ({n}/{len(episodes)}): left out, nothing kept")
            continue
        anchor = states_raw[a] if dagger else states_raw[a + keep[0]]
        start_grippers.append(float(anchor[5]))
        recorded_on = (records.get(ep, {}).get("conditions") or {}).get("surface", config.SURFACE_BEFORE_ASKED)
        kind, texture = (episode_surface(repo, ep, size, **(surfacing or {}))     # surface_keep knows only the black top
                         if recorded_on == config.SURFACE_BEFORE_ASKED else (None, None))
        for i in keep:
            item = src[a + int(i)]
            image = F.resize(item["observation.images.wrist"], list(size), antialias=True)
            image = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            dst.add_frame({
                "observation.state": (states_raw[a + i] - anchor).astype("float32"),
                "action": (actions_raw[a + i] - anchor).astype("float32"),
                "observation.state_raw": states_raw[a + i].astype("float32"),
                "action_raw": actions_raw[a + i].astype("float32"),
                "observation.images.wrist": image if texture is None else replace_surface(image, texture),
                "task": item["task"]})
        dst.save_episode()
        built.append(ep)
        takeovers.append(takeover)
        kinds.append(kind)
        source_frames += b - a
        log(f"  {label}episode {ep} ({n}/{len(episodes)}): kept {len(keep)}/{b - a}"
            + (f" (from the takeover at frame {takeover})" if dagger else "") + (f", on {kind}" if kind else ""))
    return built, takeovers, source_frames, start_grippers, kinds


# --- another surface under the tiles (build(replace_mat=...)) ---

KEEP_WHITE = (80, 0.45, 90)     # HSV (OpenCV 0-180 hue): saturation below, value above this share of the frame's 99th
                                # percentile and above 90: tile faces, the jaws' pale parts
KEEP_FACE_SHADE = (0.25, 55)    # ...and the greyer pixels (value above this share, and 55) joined to those: a face's
                                # shaded side (in the dimmer recordings a face falls to ~60-100)
KEEP_GREEN = (35, 85, 60)       # hue from, to, saturation above: the tiles' green backs
KEEP_CYAN = (80, 105, 80)       # the jaws
SURFACES = ["grid", "solid", "wood", "cloth", "paper"]
SURFACE_WEIGHTS = [0.35, 0.2, 0.2, 0.15, 0.1]   # grids first (the mat that failed), paper last (as bright as a face)


def surface_keep(rgb):
    """What isn't the mat in a wrist frame (H x W x 3 uint8 RGB): tile faces with their prints filled in, the
    tiles' green backs and the cyan jaws. The rest is mat, including the lamp's brown glare on it."""
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = (hsv[..., k].astype(int) for k in range(3))
    top = np.percentile(v, 99)
    pale = (s < KEEP_WHITE[0]) & (v > max(KEEP_FACE_SHADE[1], KEEP_FACE_SHADE[0] * top))
    _, labels = cv2.connectedComponents(pale.astype(np.uint8))
    lit = pale & (v > max(KEEP_WHITE[2], KEEP_WHITE[1] * top))
    faces = np.isin(labels, np.unique(labels[lit])) & pale
    keep = (faces
            | (h >= KEEP_GREEN[0]) & (h <= KEEP_GREEN[1]) & (s >= KEEP_GREEN[2]) & (v >= 60)
            | (h >= KEEP_CYAN[0]) & (h <= KEEP_CYAN[1]) & (s >= KEEP_CYAN[2]) & (v >= 60)).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(keep, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:            # one at a time: drawn together, a face's print counts as a hole
        if cv2.contourArea(contour) > 30:
            cv2.drawContours(keep, [contour], -1, 1, -1)
    return cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)


def random_surface(rng, size=(224, 224), kind=None):
    """(kind, H x W x 3 uint8 RGB texture): a grid mat, a plain colour, wood, cloth or paper, its colour,
    scale and angle drawn from rng."""
    import cv2
    h, w = size
    kind = kind or rng.choice(SURFACES, p=SURFACE_WEIGHTS)
    big = int(np.ceil(np.hypot(h, w))) + 2

    def colour(hue=None, sat=(0.0, 0.8), val=(0.15, 0.85)):
        hsv = np.uint8([[[rng.uniform(0, 180) if hue is None else rng.uniform(*hue), 255 * rng.uniform(*sat),
                          255 * rng.uniform(*val)]]])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(float)

    canvas = np.ones((big, big, 3)) * colour()
    if kind == "grid":
        line = colour(sat=(0.0, 0.3), val=(0.6, 1.0)) if rng.random() < 0.7 else colour(val=(0.0, 0.3))
        for spacing, width in ((int(rng.uniform(10, 40)), int(rng.integers(1, 3))), (int(rng.uniform(50, 120)), 2)):
            for k in range(0, big, spacing):
                canvas[k:k + width, :] = line
                canvas[:, k:k + width] = line
    elif kind == "wood":
        canvas = np.ones((big, big, 3)) * colour(hue=(5, 25), sat=(0.3, 0.8), val=(0.2, 0.7))
        y = np.arange(big)[:, None] + 6 * np.sin(np.arange(big)[None, :] / rng.uniform(15, 40))
        canvas *= (0.85 + 0.15 * np.sin(y / rng.uniform(2, 6)))[..., None]
    elif kind == "cloth":
        noise = cv2.GaussianBlur(rng.normal(0, 1, (big, big)), (0, 0), rng.uniform(0.6, 2.0))
        canvas *= (1 + 0.12 * noise / max(noise.std(), 1e-6))[..., None]
    elif kind == "paper":
        canvas = np.ones((big, big, 3)) * colour(sat=(0.0, 0.08), val=(0.75, 0.95))
        canvas *= (1 + 0.03 * rng.normal(0, 1, (big, big)))[..., None]
    turn = cv2.getRotationMatrix2D((big / 2, big / 2), rng.uniform(0, 90), 1.0)
    canvas = cv2.warpAffine(canvas.clip(0, 255).astype(np.uint8), turn, (big, big))
    top, left = (big - h) // 2, (big - w) // 2
    return str(kind), np.ascontiguousarray(canvas[top:top + h, left:left + w])


def replace_surface(rgb, texture):
    """The wrist frame with its mat (surface_keep) swapped for `texture`, shaded by the frame's own
    brightness relative to the mat's median, so shadows beside the tiles and the lamp's glare carry over."""
    keep = surface_keep(rgb)
    if keep.all():
        return rgb
    value = rgb.max(axis=2).astype(float)
    shade = np.clip(0.6 + 0.4 * value / max(np.median(value[~keep]), 1.0), 0.5, 1.6)
    swapped = (texture.astype(float) * shade[..., None]).clip(0, 255).astype(np.uint8)
    return np.where(keep[..., None], rgb, swapped)


def episode_surface(repo, episode, size=(224, 224), fraction=0.0, seed=0):
    """(kind, texture) for one episode, or (None, None) to keep the black table top: the same every build of
    the same dataset, episode and seed."""
    rng = np.random.default_rng(zlib.crc32(f"{repo}:{episode}:{seed}".encode()))
    if rng.random() >= fraction:
        return None, None
    return random_surface(rng, size)


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
