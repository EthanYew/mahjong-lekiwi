"""Offline evaluation: score a policy on recorded episodes it never trained on, without the arm.

The eval episodes (data/eval_split.json) are held-out episodes of training recordings plus whole
eval-only recordings (record.py --slice familiar/new/dim/tilt/jitter). Each is built like a training
dataset (dataset.build: 224px, trims, re-anchored deltas), with the narrowing trimmed when the policy's
hover is the narrowed one, so one 90%-hover recording scores both kinds of policy.

Teacher-forced: at every EVAL_STRIDE-th frame the policy gets the demo's real image and state and
plans a chunk (no temporal ensembling); its first action is compared with the demo's. Per episode:
- arm_mae (joints 0-4, deg) and grip_mae (gripper, %): mean absolute error of the first action
- close_err_s: from the plan made GRASP_PROBE_S before the demo closed, when that plan shuts the gripper
  vs when the demo did (+ = later); never_closes = it doesn't within its chunk. (Not teacher-forced: fed
  the demo's state, a policy that just holds its state would "close" one frame behind the demo.)
- onset / demo_onset: from the frame where the demo's narrowing ends (its hover, ready to descend),
  plan steps until the plan's lift/elbow move DESCENT_DEG vs demo frames until the demo's did;
  freeze = the plan never descends within its chunk while the demo did
- grasp_mm / roll_err_deg: from that same plan, where the jaws are when it closes (or at its end) vs where the demo closed (xy, mm) and the wrist roll there vs the demo's — the
  roll is what a tilted tile needs. demo_roll_deg is the demo's own roll at close (a tilt measure).

Limits: teacher forcing can't show errors compounding (a policy 2mm off sees different images from
then on). The offline gates catch freezes and gross regressions; the bench decides what ships.
"""
import hashlib
import json
import math
import random
from datetime import datetime

import numpy as np

from . import config, kinematics
from .dataset import GRIP_CLOSED_BELOW, episode_log, narrowing_end

EVAL_STRIDE = 3
DESCENT_DEG = 2.0              # lift/elbow movement that counts as "descending" (as the freeze probe)
GRASP_PROBE_S = 1.0
HELDOUT_FRACTION = 0.10
HELDOUT_MIN_CLUSTER = 3        # a spot needs this many episodes before one is held out
NARROWED_HOVER_BELOW = 50.0    # hover gripper (%) below which a policy starts from the narrowed opening
NOT_GATED = ("tilt",)          # slices reported but not gated (no dataset has trained on tilt yet)
# Gates per gated slice. Calibrated 2026-10-23 on v2's 15 held-out episodes (both policies trained on
# them): v1, which froze on hardware, had 4 freezes, a mean descent lag of 21 steps and 0.37s close
# error; v2, which didn't, had 0, 1 step and 0.06s. Their arm error (0.22 vs 0.21 deg) and grasp point
# (1.4 vs 0.9mm) did NOT separate them — teacher-forced error can't see a policy that waits.
GATE_ONSET_LAG_STEPS = 10      # fail if the plan's descent starts on average this much later than the demo's
GATE_CLOSE_ERR_S = 0.25        # fail if the open-loop close timing is off by more (mean, absolute)
# Against a reference policy (e.g. the current default): no gated slice much worse.
GATE_ARM_MAE_RATIO, GATE_ARM_MAE_SLACK = 1.25, 0.1     # fail if arm_mae > ratio * reference + slack
GATE_GRASP_MM_SLACK = 3.0                              # fail if median grasp_mm > reference + slack


# --- metrics on arrays ---

def first_below(values, threshold):
    hits = np.flatnonzero(np.asarray(values) < threshold)
    return int(hits[0]) if len(hits) else None


def descent_onset(joints, threshold=DESCENT_DEG):
    """First step where lift or elbow is `threshold` from step 0 (rows of joint values), or None."""
    joints = np.asarray(joints)
    moved = np.abs(joints[:, 1:3] - joints[0, 1:3]).max(axis=1)
    return next((k for k in range(len(joints)) if moved[k] > threshold), None)


def score_episode(plan, frame, states, actions, actions_raw, fps=config.FPS, stride=EVAL_STRIDE):
    """Score one built episode. plan(image, state_delta) -> chunk of deltas (chunk x 6); frame(i) -> the
    episode's i-th wrist image; states/actions: deltas (n x 6); actions_raw: absolute commands."""
    n = len(actions)
    anchor = actions_raw[0] - actions[0]
    chunks = {}

    def chunk_at(i):
        if i not in chunks:
            chunks[i] = np.asarray(plan(frame(i), states[i]))
        return chunks[i]

    evaluated = list(range(0, n, stride))
    first = np.array([chunk_at(i)[0] for i in evaluated])
    target = actions[evaluated]
    arm_mae = float(np.abs(first[:, :5] - target[:, :5]).mean())
    grip_mae = float(np.abs(first[:, 5] - target[:, 5]).mean())

    demo_close = first_below(actions_raw[:, 5], GRIP_CLOSED_BELOW)

    start = narrowing_end(actions_raw)
    probe = chunk_at(start)
    onset, demo_onset = descent_onset(probe), descent_onset(actions_raw[start:])
    freeze = onset is None and demo_onset is not None and demo_onset < len(probe)

    grasp_mm = roll_err = demo_roll = close_err_s = None
    plan_closes = False
    if demo_close is not None:
        p = max(0, demo_close - int(GRASP_PROBE_S * fps))
        grasp_plan = anchor + chunk_at(p)
        c = first_below(grasp_plan[:, 5], GRIP_CLOSED_BELOW)
        plan_closes = c is not None
        close_err_s = (p + c - demo_close) / fps if plan_closes else None
        at = grasp_plan[c if plan_closes else -1]
        predicted_tool = kinematics.tool_point(dict(zip(config.JOINTS, at)))
        demo_tool = kinematics.tool_point(dict(zip(config.JOINTS, actions_raw[demo_close])))
        grasp_mm = float(np.hypot(*(np.asarray(predicted_tool[:2]) - demo_tool[:2])) * 1000)
        roll_err = float(at[4] - actions_raw[demo_close][4])
        demo_roll = float(actions[demo_close][4])
    return {"frames": n, "arm_mae": arm_mae, "grip_mae": grip_mae, "demo_close_s": None if demo_close is None else demo_close / fps,
            "close_err_s": close_err_s, "never_closes": demo_close is not None and not plan_closes,
            "onset": onset, "demo_onset": demo_onset, "freeze": bool(freeze), "grasp_mm": grasp_mm,
            "plan_closes": plan_closes, "roll_err_deg": roll_err, "demo_roll_deg": demo_roll}


# --- which episodes ---

def position_clusters(xys, radius_m=config.TILE_MERGE_RADIUS_M):
    """A cluster index per position: positions within radius_m of a cluster's first member share it."""
    seeds, labels = [], []
    for x, y in xys:
        k = next((i for i, (sx, sy) in enumerate(seeds) if math.hypot(x - sx, y - sy) <= radius_m), None)
        if k is None:
            seeds.append((x, y))
            k = len(seeds) - 1
        labels.append(k)
    return labels, seeds


def spot_name(x, y):
    """A readable spot label from a base-frame position: x and y in cm."""
    return f"{x * 100:.0f},{y * 100:+.0f}"


def choose_heldout(records, fraction=HELDOUT_FRACTION, seed=0):
    """Episodes of a training recording to hold out: one at random from every spot with at least
    HELDOUT_MIN_CLUSTER episodes, topped up at random from the biggest spots to `fraction`. Seeded, so the
    same records always give the same split. records: {episode: log record with detection_xy}."""
    episodes = sorted(records)
    labels, _ = position_clusters([records[e]["detection_xy"] for e in episodes])
    by_cluster = {}
    for e, c in zip(episodes, labels):
        by_cluster.setdefault(c, []).append(e)
    rng = random.Random(seed)
    chosen = {c: [rng.choice(eps)] for c, eps in sorted(by_cluster.items()) if len(eps) >= HELDOUT_MIN_CLUSTER}
    target = round(fraction * len(episodes))
    while sum(map(len, chosen.values())) < target:
        c = max((c for c in by_cluster if len(by_cluster[c]) - len(chosen.get(c, [])) >= HELDOUT_MIN_CLUSTER),
                key=lambda c: len(by_cluster[c]) - len(chosen.get(c, [])), default=None)
        if c is None:
            break
        chosen.setdefault(c, []).append(rng.choice([e for e in by_cluster[c] if e not in chosen.get(c, [])]))
    return sorted(e for eps in chosen.values() for e in eps)


def load_split(path=None):
    path = path or config.EVAL_SPLIT
    return json.loads(path.read_text()) if path.exists() else {"heldout": {}, "eval_sets": []}


def save_split(split, path=None):
    (path or config.EVAL_SPLIT).write_text(json.dumps(split, indent=2) + "\n")


def excluded_episodes(src_repo, split=None):
    """Episodes of src_repo that must not be trained on: its held-out ones, or all if it's an eval set."""
    split = split or load_split()
    if src_repo in split.get("eval_sets", []):
        raise SystemExit(f"{src_repo} is an eval set (data/eval_split.json) — it is never trained on")
    return sorted(split.get("heldout", {}).get(src_repo, []))


def lighting(conditions):
    """A short label from a session's setup answers, or 'unrecorded'."""
    if not conditions:
        return "unrecorded"
    yn = {True: "on", False: "off"}
    return (f"daylight {conditions.get('daylight', '?')}, lamp {yn.get(conditions.get('lamp'), '?')}, "
            f"ceiling {yn.get(conditions.get('ceiling_lights'), '?')}")


def labels(record, default_slice):
    xy = record.get("detection_xy")
    return {"slice": record.get("slice", default_slice) if record.get("slice") != "train" else default_slice,
            "spot": record.get("spot") or (spot_name(*xy) if xy else "?"),
            "lighting": lighting(record.get("conditions")), "tile": record.get("tile_name", "")}


# --- building and running ---

def eval_dataset_name(src_repo, episodes, narrowed):
    digest = hashlib.sha1(json.dumps(sorted(episodes)).encode()).hexdigest()[:8]
    return f"{config.HF_USER}/{src_repo.split('/')[-1]}_eval_{'narrowed' if narrowed else 'full'}_{digest}_224"


def eval_dataset(src_repo, episodes, narrowed, log=print):
    """The built (224, trimmed) version of src_repo's `episodes`, built once and cached locally under a
    name derived from the episodes. Returns (LeRobotDataset, build summary)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from .dataset import build
    name = eval_dataset_name(src_repo, episodes, narrowed)
    summary_path = HF_LEROBOT_HOME / name / "build_summary.json"
    if not summary_path.exists():
        log(f"  building {name} ({len(episodes)} episodes of {src_repo})")
        summary = build(src_repo, name, trim_narrowing=narrowed, episodes=episodes, log=lambda *_: None)
        summary_path.write_text(json.dumps(summary, indent=2))
    return LeRobotDataset(name), json.loads(summary_path.read_text())


def eval_items(split=None):
    """(source repo, episodes, default slice) for everything in the split."""
    split = split or load_split()
    items = [(src, sorted(eps), "heldout") for src, eps in split.get("heldout", {}).items() if eps]
    for src in split.get("eval_sets", []):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        items.append((src, list(range(LeRobotDataset(src).num_episodes)), "eval"))
    return items


def bgr_frame(ds, index):
    item = ds[index]
    return (item["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)[:, :, ::-1]


def evaluate(policy, split=None, max_episodes=None, stride=EVAL_STRIDE, log=print):
    """Score `policy` (policy.Policy) on every eval episode. Returns a list of rows (labels + metrics)."""
    narrowed = policy.hover["gripper"] < NARROWED_HOVER_BELOW
    rows = []
    for src, episodes, default_slice in eval_items(split):
        records = episode_log(src)
        ds, summary = eval_dataset(src, episodes, narrowed, log=log)
        h = ds.hf_dataset
        for built_ep, src_ep in enumerate(summary["source_episodes"][:max_episodes]):
            meta = ds.meta.episodes[built_ep]
            a, b = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
            states, actions, actions_raw = (np.stack(h[k][a:b]) for k in ("observation.state", "action", "action_raw"))
            metrics = score_episode(policy.plan, lambda i: bgr_frame(ds, a + i), states, actions, actions_raw,
                                    fps=ds.fps, stride=stride)
            rows.append({"source": src, "episode": src_ep, **labels(records.get(src_ep, {}), default_slice), **metrics})
            log(f"  {src.split('/')[-1]} ep {src_ep:>3} [{rows[-1]['slice']}/{rows[-1]['spot']}]: arm {metrics['arm_mae']:.2f}° "
                f"grip {metrics['grip_mae']:.1f}%  onset {metrics['onset']} (demo {metrics['demo_onset']})"
                + ("  FREEZE" if metrics["freeze"] else ""))
    return rows


# --- summarising ---

def summarise(rows, by="slice"):
    """{group: aggregate metrics} over rows grouped by a label (slice / spot / lighting)."""
    groups = {}
    for row in rows:
        groups.setdefault(row[by], []).append(row)

    def mean(values, absolute=False):
        values = [abs(v) if absolute else v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    def median(values):
        values = [v for v in values if v is not None]
        return float(np.median(values)) if values else None
    return {g: {"episodes": len(rs), "arm_mae": mean([r["arm_mae"] for r in rs]),
                "grip_mae": mean([r["grip_mae"] for r in rs]),
                "close_err_s": mean([r["close_err_s"] for r in rs], absolute=True),
                "never_closes": sum(r["never_closes"] for r in rs), "freezes": sum(r["freeze"] for r in rs),
                "onset_lag": mean([r["onset"] - r["demo_onset"] for r in rs if r["onset"] is not None and r["demo_onset"] is not None]),
                "grasp_mm": median([r["grasp_mm"] for r in rs]),
                "roll_err_deg": mean([r["roll_err_deg"] for r in rs], absolute=True)}
            for g, rs in sorted(groups.items())}


def gates(candidate_rows, reference_rows=None):
    """[(passed, message)] per gated slice: no freezes, descent on time, close on time; with a reference,
    arm error and grasp point not much worse than its."""
    results = []
    candidate = summarise(candidate_rows)
    reference = summarise(reference_rows) if reference_rows else {}
    for name, c in candidate.items():
        if name in NOT_GATED:
            continue
        results.append((c["freezes"] == 0, f"{name}: {c['freezes']} freezes"))
        if c["onset_lag"] is not None:
            results.append((c["onset_lag"] <= GATE_ONSET_LAG_STEPS,
                            f"{name}: descent starts {c['onset_lag']:+.0f} steps from the demo's (limit +{GATE_ONSET_LAG_STEPS})"))
        if c["close_err_s"] is not None:
            results.append((c["close_err_s"] <= GATE_CLOSE_ERR_S,
                            f"{name}: closes {c['close_err_s']:.2f}s off the demo (limit {GATE_CLOSE_ERR_S:.2f}s)"))
        r = reference.get(name)
        if r:
            limit = GATE_ARM_MAE_RATIO * r["arm_mae"] + GATE_ARM_MAE_SLACK
            results.append((c["arm_mae"] <= limit, f"{name}: arm error {c['arm_mae']:.2f}° (limit {limit:.2f}°)"))
            if c["grasp_mm"] is not None and r["grasp_mm"] is not None:
                limit = r["grasp_mm"] + GATE_GRASP_MM_SLACK
                results.append((c["grasp_mm"] <= limit, f"{name}: grasp point {c['grasp_mm']:.1f}mm (limit {limit:.1f}mm)"))
    return results


def table(summary, title):
    def f(v, spec):
        return format(v, spec) if v is not None else format("-", f">{spec.split('.')[0]}")
    lines = [title, f"  {'':<28} {'n':>3} {'arm °':>6} {'grip %':>6} {'close s':>7} {'freeze':>6} {'lag':>5} "
                    f"{'grasp mm':>8} {'roll °':>6}"]
    for g, s in summary.items():
        lines.append(f"  {g:<28} {s['episodes']:>3} {f(s['arm_mae'], '6.2f')} {f(s['grip_mae'], '6.1f')} "
                     f"{f(s['close_err_s'], '7.2f')} {s['freezes']:>6} {f(s['onset_lag'], '5.0f')} {f(s['grasp_mm'], '8.1f')} "
                     f"{f(s['roll_err_deg'], '6.1f')}")
    return "\n".join(lines)


def save_results(policy_id, rows, split=None):
    """data/offline_eval/<policy>.json: the rows and per-slice summary, with what they were scored on."""
    config.OFFLINE_EVAL.mkdir(parents=True, exist_ok=True)
    from .session import git_commit
    path = config.OFFLINE_EVAL / f"{str(policy_id).strip('/').replace('/', '__')}.json"
    path.write_text(json.dumps({"policy": str(policy_id), "time": datetime.now().isoformat(timespec="seconds"),
                                "commit": git_commit(), "split": split or load_split(),
                                "slices": summarise(rows), "rows": rows}, indent=2, default=float) + "\n")
    return path
