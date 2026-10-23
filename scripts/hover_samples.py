"""Import hover-correction samples from where the jaws actually closed on a tile: recorded demos (the
operator's grasp) and successful bench attempts (the policy's grasp).

Each sample is the commanded tool point at the first gripper-shut command minus the detection — what
collect_corrections.py measures at the hover, measured at the grasp instead. They're appended to
data/hover_corrections.jsonl as source "demo" / "bench" (already-imported ones are skipped), and
fitted only once data/hover_model.json lists their source. bench.py records new bench samples itself;
this backfills older sessions. Demos from a jittered hover (record.py --hover-jitter-mm) stay in: a
demo sample is the grasp minus the detection, whatever the hover, and the operator had to steer to it.

Recordings and old sessions don't store the RealSense calibration, so only samples from after the
current setup's first gamepad correction are imported (older ones may predate a recalibration).

    python scripts/hover_samples.py --dataset Yewy/mahjong_mat_pickup_lekiwi_cl_hover
    python scripts/hover_samples.py --sessions                    # every bench session
    python -m mahjong.hover_correction --sources gamepad,demo     # what the hover would become
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mahjong import config
from mahjong.hover_correction import append_record, grasp_record, load_records, read_all, source_of
from mahjong.policy import grasp_point
from mahjong.session import read_events

EPISODE_LOGS = ("episode_log.jsonl", "episode_teleop_log.jsonl")   # record.py's; the older recorder's


def episode_log(root):
    """{episode_index: the episode's log row} from whichever log the dataset has."""
    for name in EPISODE_LOGS:
        path = Path(root) / name
        if path.exists():
            return {r["episode_index"]: r for r in map(json.loads, filter(str.strip, path.read_text().splitlines()))}
    raise SystemExit(f"{root} has no episode log ({' / '.join(EPISODE_LOGS)}): no detections to compare against")


def row_time(row):
    """When an episode was recorded: its timestamp, or its session folder's (record.py before timestamps)."""
    if row.get("timestamp"):
        return row["timestamp"]
    return datetime.strptime(row["session"][:15], "%Y%m%d_%H%M%S").isoformat(timespec="seconds")


def demo_records(repo, actions_by_episode, log):
    """Samples from recorded episodes: {episode_index: absolute actions (steps x JOINTS)} + their log rows."""
    records = []
    for ep, actions in sorted(actions_by_episode.items()):
        row = log.get(ep)
        _, grasp_xy = grasp_point(actions)
        if row is None or grasp_xy is None:
            continue
        records.append(grasp_record("demo", row["detection_xy"], grasp_xy, timestamp=row_time(row),
                                    tile_name=row.get("tile_name"), hover_xy=row.get("hover_xy"),
                                    hover_jitter_xy=row.get("hover_jitter_xy"),
                                    hover_correction=row.get("hover_correction"), dataset=repo, episode_index=ep))
    return records


def load_demos(repo):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    log = episode_log(HF_LEROBOT_HOME / repo)
    src = LeRobotDataset(repo)
    actions = np.stack(src.hf_dataset["action_raw"])
    by_episode = {}
    for ep in range(src.num_episodes):
        meta = src.meta.episodes[ep]
        by_episode[ep] = actions[int(meta["dataset_from_index"]):int(meta["dataset_to_index"])]
    return demo_records(repo, by_episode, log)


def bench_records(session_dir):
    """Samples from a bench session's successful, real (not --sim) attempts."""
    events = read_events(session_dir)
    if any(e["type"] == "session_start" and e.get("sim") for e in events):
        return []
    return [grasp_record("bench", e["detection_xy"], e["grasp_xy"], timestamp=e["time"], tile_name=e["tile"],
                         hover_xy=e["hover_xy"], hover_perturb_xy=e.get("hover_perturb_xy"), policy=e["policy"],
                         correction=e.get("correction"), session=session_dir.name, attempt_dir=e["attempt_dir"])
            for e in events if e["type"] == "attempt" and e.get("success") and e.get("grasp_xy")]


def key(record):
    if source_of(record) == "demo":
        return ("demo", record.get("dataset"), record.get("episode_index"))
    if source_of(record) == "bench":
        return ("bench", record.get("session"), record.get("attempt_dir"))
    return None


def setup_since(path=None):
    """When the current setup's first gamepad correction was recorded (None: no corrections yet)."""
    fitted, _, excluded, unused = load_records(path, sources={"gamepad": 1.0})
    times = [r["timestamp"] for r in fitted + excluded if source_of(r) == "gamepad"]
    return min(times) if times else None


def import_records(records, path=None, since=None):
    """Append the records not already in the file and not older than `since`. Returns (added, already, too_old)."""
    have = {key(r) for r in read_all(path)}
    added = already = too_old = 0
    for record in records:
        if key(record) in have:
            already += 1
        elif since is None or record["timestamp"] < since:
            too_old += 1
        else:
            append_record(record, path)
            have.add(key(record))
            added += 1
    return added, already, too_old


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", action="append", default=[], help="recorded (not trimmed) dataset repo id; repeatable")
    p.add_argument("--sessions", nargs="*", type=Path, default=None, help="bench session folders (none given: all)")
    args = p.parse_args()
    if not args.dataset and args.sessions is None:
        p.error("give --dataset and/or --sessions")
    since = setup_since()
    if since is None:
        raise SystemExit("no gamepad corrections for this setup — collect_corrections.py first")
    batches = [(repo, load_demos(repo)) for repo in args.dataset]
    if args.sessions is not None:
        folders = args.sessions or sorted(config.SESSIONS.glob("*_bench"))
        batches += [(folder.name, bench_records(folder)) for folder in folders]
    for name, records in batches:
        added, already, too_old = import_records(records, since=since)
        print(f"{name}: {added} added, {already} already there"
              + (f", {too_old} from before this setup's first correction ({since}) skipped" if too_old else ""))
    print("\n`python -m mahjong.hover_correction --sources gamepad,demo,bench` compares a fit using them with the live one")


if __name__ == "__main__":
    main()
