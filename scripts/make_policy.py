"""One command from a recorded dataset to a registered, deployable policy.

  1. quality summary of the recorded episodes
  2. build the training dataset: trims (idle, pre-fix takeover sag, optionally the gripper narrowing),
     re-anchored deltas, wrist camera at 224 — leaving out the held-out episodes (data/eval_split.json)
  3. verify it: every episode kept, zero start deltas, no idle frames left
  4. upload it to the Hub
  5. train ACT with the recipe of every policy so far (resnet18, chunk 100, batch 96, lr 1e-5), steps
     scaled to ~131 passes over the data; the trained policy is pushed to the Hub
  6. check each checkpoint offline: scores per slice on the eval episodes it never trained on, with the
     gates against the current default policy (scripts/offline_eval.py); replay error on a few
     training episodes, and the freeze probe on the fixture frames
  7. register it in data/policies.json with the hover its episodes started from

    python scripts/make_policy.py --source Yewy/mahjong_mat_pickup_v3 --version v3 --dry-run
    python scripts/make_policy.py --source Yewy/mahjong_mat_pickup_v3 --version v3 --default    # pickup_mat_cl_v3

Settle the options before starting: training takes hours, and the dataset name is fixed once uploaded.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mahjong import config

EPOCHS = 131                # what v1 and v2 were trained for
BATCH = 96
FREEZE_FRAMES = ["180956_8D_step0300.jpg", "181050_8D_step0300.jpg", "181145_8D_step0300.jpg"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", required=True, help="recorded dataset repo id")
    p.add_argument("--version", required=True, help="e.g. v3; used in the output names")
    p.add_argument("--policy-id", default=None, help="registry id (default: pickup_mat_cl_<version>, like the "
                                                     "registered ones)")
    p.add_argument("--trim-narrowing", action="store_true",
                   help="also drop the opening gripper narrowing (start episodes at the pre-grasp opening)")
    p.add_argument("--steps", type=int, default=None, help="training steps (default: ~131 passes over the data)")
    p.add_argument("--private", action="store_true", help="upload the dataset and policy as private")
    p.add_argument("--default", action="store_true", help="make it the registry's default policy")
    p.add_argument("--skip-build", action="store_true", help="the training dataset already exists locally")
    p.add_argument("--skip-upload", action="store_true",
                   help="don't push the training dataset (training still pushes the policy)")
    p.add_argument("--skip-train", action="store_true", help="evaluate/register an existing training run")
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    return p.parse_args()


def names(args):
    base = args.source.split("/")[-1]
    return {"dataset": f"{config.HF_USER}/{base}_{args.version}_224",
            "policy_repo": f"{config.HF_USER}/{base}_{args.version}_policy",
            "job": f"act_{base}_{args.version}",
            "output_dir": config.LEROBOT_REPO / "outputs" / "train" / f"{base}_{args.version}",
            "policy_id": args.policy_id or f"pickup_mat_cl_{args.version}"}


def train_command(n, steps, private):
    return [str(Path(sys.executable).parent / "lerobot-train"),
            f"--dataset.repo_id={n['dataset']}", "--policy.type=act", "--policy.device=cuda",
            f"--policy.repo_id={n['policy_repo']}", "--policy.push_to_hub=true", f"--policy.private={str(private).lower()}",
            "--policy.tags=[mahjong,so101,lekiwi,act]", f"--output_dir={n['output_dir']}", f"--job_name={n['job']}",
            f"--steps={steps}", f"--batch_size={BATCH}", "--num_workers=10", f"--save_freq={max(int(steps) // 7, 1000)}",
            "--log_freq=200", "--seed=1000", "--wandb.enable=false"]


def verify(dataset_repo, trimmed_idle):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from mahjong.dataset import idle_mask
    ds = LeRobotDataset(dataset_repo)
    h = ds.hf_dataset
    ep = np.array(h["episode_index"])
    state, action = np.stack(h["observation.state"]), np.stack(h["action"])
    starts = [np.flatnonzero(ep == e)[0] for e in range(ds.num_episodes)]
    assert np.abs(state[starts]).max() < 1e-4, "an episode doesn't start at a zero state delta"
    if trimmed_idle:
        idle = sum(idle_mask(action[ep == e]).sum() for e in range(ds.num_episodes))
        assert idle == 0, f"{idle} idle frames left"
    print(f"  verified: {ds.num_episodes} episodes, {len(ep)} frames, zero start deltas"
          + (", no idle frames" if trimmed_idle else ""))
    return ds.num_episodes


def evaluate(policy, dataset_repo, hover):
    """(replay MAE on training episodes — optimistic, mean freeze-probe descent onset in plan steps)."""
    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from mahjong.offline_eval import descent_onset, to_bgr
    ds = LeRobotDataset(dataset_repo)
    errors = []
    for e in np.linspace(0, ds.num_episodes - 1, 3).astype(int):
        meta = ds.meta.episodes[int(e)]
        # the first action of a fresh plan at every 3rd frame (stepping the ensembler with gaps would
        # misalign its averaging)
        for i in range(int(meta["dataset_from_index"]), int(meta["dataset_to_index"]), 3):
            item = ds[i]
            first = policy.plan(to_bgr(item["observation.images.wrist"]), item["observation.state"].numpy())[0]
            errors.append(np.abs(first - item["action"].numpy()).mean())
    # The freeze frames were captured after the jaws had been narrowed from a 90% hover (gripper delta
    # -31); a policy recorded from a narrowed hover starts at that opening, i.e. delta 0.
    probe_state = np.array([0, 0, 0, 0, 0, -31.0 if hover["gripper"] > 50 else 0.0], np.float32)
    onsets = []
    for name in FREEZE_FRAMES:
        plan = policy.plan(cv2.imread(str(config.DATA / "fixtures" / "freeze_frames" / name)), probe_state)
        onset = descent_onset(plan)
        onsets.append(len(plan) if onset is None else onset)
    return float(np.mean(errors)), float(np.mean(onsets))


def main():
    args = parse_args()
    n = names(args)
    from mahjong.quality import dropped_episodes, quality_table
    from mahjong.dataset import build, hover_settings
    from mahjong import offline_eval
    table, _, _ = quality_table(args.source)
    frames = int(table["frames"].sum())
    print(f"1. quality: {args.source}: {len(table)} episodes, {frames} frames"
          + (f", {int((table['score'] >= 4).sum())} flagged (scripts/quality_report.py to review)" if "score" in table else ""))
    heldout = offline_eval.excluded_episodes(args.source)
    dropped = dropped_episodes(args.source)
    train_episodes = sorted(int(e) for e in set(table.index) - set(heldout) - set(dropped))
    frames = int(table.loc[train_episodes, "frames"].sum())
    print(f"   training on {len(train_episodes)} episodes; {len(heldout)} held out for evaluation"
          + (f", {len(dropped)} dropped after review {sorted(dropped)}" if dropped else "")
          + ("" if heldout else f" — none: scripts/offline_eval.py --make-split {args.source} first, or every "
                                 f"episode is trained on and offline scores are optimistic"))
    print(f"   training dataset {n['dataset']}, policy {n['policy_repo']} -> registry id {n['policy_id']}")
    if args.dry_run:
        est = int(frames * (0.56 if args.trim_narrowing else 0.70))   # measured on the first dataset
        steps = args.steps or max(10_000, int(round(EPOCHS * est / BATCH, -3)))
        print(f"2. build: trims idle + pre-fix sag{' + gripper narrowing' if args.trim_narrowing else ''}, "
              f"~{est} frames kept (estimate)\n3. verify\n4. upload ({'private' if args.private else 'public'})\n"
              f"5. train {steps} steps (~{steps / 4.6 / 3600:.1f}h at 4.6 steps/s):\n   {' '.join(train_command(n, steps, args.private))}\n"
              f"6. evaluate each checkpoint in {n['output_dir']}/checkpoints\n7. register {n['policy_id']}"
              f"{' as default' if args.default else ''}")
        return

    from lerobot.utils.constants import HF_LEROBOT_HOME
    print("2. build")
    if not args.skip_build:
        summary = build(args.source, n["dataset"], trim_narrowing=args.trim_narrowing, episodes=train_episodes)
        (HF_LEROBOT_HOME / n["dataset"] / "build_summary.json").write_text(json.dumps(summary, indent=2))
    summary = json.loads((HF_LEROBOT_HOME / n["dataset"] / "build_summary.json").read_text())
    print(f"   kept {summary['frames']}/{summary['source_frames']} frames")
    print("3. verify")
    verify(n["dataset"], trimmed_idle=True)
    if not args.skip_upload:
        print("4. upload")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        LeRobotDataset(n["dataset"]).push_to_hub(
            private=args.private, tags=["mahjong", "so101", "lekiwi", "act"],
            dataset_description=f"{args.source}, trimmed (idle frames, pre-fix takeover sag"
                                f"{', gripper narrowing' if args.trim_narrowing else ''}), deltas re-anchored to each "
                                f"episode's first kept frame, wrist camera at 224x224. {summary['frames']} of "
                                f"{summary['source_frames']} frames kept. Built by mahjong-lekiwi scripts/make_policy.py.")
    steps = args.steps or max(10_000, int(round(EPOCHS * summary["frames"] / BATCH, -3)))
    if not args.skip_train:
        print(f"5. train {steps} steps")
        subprocess.run(train_command(n, steps, args.private), cwd=config.LEROBOT_REPO, check=True)
    hover = hover_settings(args.source, summary)
    print(f"6. evaluate checkpoints (hover {hover})")
    from mahjong.policy import Policy, spec
    split = offline_eval.load_split()
    has_eval = bool(offline_eval.eval_items(split))
    reference_rows = None
    if has_eval:
        reference_id, _ = spec()
        print(f"   reference: the current default, {reference_id}")
        reference_rows = offline_eval.evaluate(Policy(reference_id), split, log=lambda *_: None)
        print(offline_eval.table(offline_eval.summarise(reference_rows), f"   {reference_id}"))
    results = []
    for checkpoint in sorted((n["output_dir"] / "checkpoints").glob("[0-9]*")):
        policy = Policy(source=checkpoint / "pretrained_model", hover=hover)
        mae, onset = evaluate(policy, n["dataset"], hover)
        print(f"   {checkpoint.name}: replay error {mae:.2f} deg (training episodes), freeze-probe descent at plan step {onset:.0f}")
        slices = {}
        if has_eval:
            rows = offline_eval.evaluate(policy, split, log=lambda *_: None)
            offline_eval.save_results(f"{n['policy_id']}@{checkpoint.name}", rows, split)
            slices = {k: {"arm_mae": round(v["arm_mae"], 2), "freezes": v["freezes"]}
                      for k, v in offline_eval.summarise(rows).items()}
            print(offline_eval.table(offline_eval.summarise(rows), f"   {checkpoint.name} on the eval episodes"))
            for ok, message in offline_eval.gates(rows, reference_rows):
                print(f"     {'pass' if ok else 'FAIL'}  {message}")
        results.append((checkpoint.name, mae, onset, slices))
    print(f"7. register {n['policy_id']}")
    from mahjong.policy import register
    register(n["policy_id"], {"repo": n["policy_repo"], "dataset": n["dataset"], "hover": hover,
                              "notes": f"make_policy.py from {args.source}; {summary['frames']} frames, {steps} steps; "
                                       f"held out: {len(heldout)} episodes; checkpoints (replay, onset, eval slices): "
                                       f"{[(c, round(m, 2), o, sl) for c, m, o, sl in results]}"},
             make_default=args.default)
    print(f"Done. Try it: python scripts/bench.py --policy {n['policy_id']} --trials 5 --tile 8D")


if __name__ == "__main__":
    main()
