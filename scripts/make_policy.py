"""One command from a recorded dataset to a registered, deployable policy.

  1. quality summary of the recorded episodes
  2. build the training dataset: trims (idle, pre-fix takeover sag, optionally the gripper narrowing),
     re-anchored deltas, wrist camera at 224 — leaving out the held-out episodes (data/eval_split.json);
     with --add, other recorded datasets' episodes too (DAgger: from the takeover on, see mahjong/dataset.py)
  3. verify it: every episode kept, zero start deltas (DAgger episodes start at their takeover), no idle
     frames left
  4. upload it to the Hub
  5. train ACT with the recipe of every policy so far (resnet18, chunk 100, batch 96, lr 1e-5), steps
     scaled to ~131 passes over the data; the trained policy is pushed to the Hub
  6. check each checkpoint offline: scores per slice on the eval episodes it never trained on, with the
     gates against the current default policy (scripts/offline_eval.py); replay error on a few
     training episodes, and the freeze probe on the fixture frames
  7. register it in data/policies.json with the hover its episodes started from

    python scripts/make_policy.py --source Yewy/mahjong_mat_pickup_v3 --version v3 --dry-run
    python scripts/make_policy.py --source Yewy/mahjong_mat_pickup_v3 --version v3 --default    # pickup_mat_cl_v3
    python scripts/make_policy.py --source Yewy/mahjong_mat_pickup_jitter_v3 --add Yewy/mahjong_mat_dagger_v3c \
        --version v4 --dry-run

Settle the options before starting: training takes hours, and the dataset name is fixed once uploaded.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mahjong import config, recipe

EPOCHS = 131                # what v1 and v2 were trained for
BATCH = 96
FREEZE_FRAMES = ["180956_8D_step0300.jpg", "181050_8D_step0300.jpg", "181145_8D_step0300.jpg"]
# --augment: LeRobot's colour jitter, wider than its defaults (lighting has already caught a policy out, v2),
# without its RandomAffine: the jaws sit at a fixed place in the wrist image and the task is lining the tile
# up against them. At most 3 of these per frame.
AUGMENT = {"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.6, 1.4]}},
           "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.6, 1.4]}},
           "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
           "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.08, 0.08]}},
           "sharpness": {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}}}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", required=True, help="recorded dataset repo id")
    p.add_argument("--version", required=True, help="e.g. v3; used in the output names")
    p.add_argument("--policy-id", default=None, help="registry id (default: pickup_mat_cl_<version>, like the "
                                                     "registered ones)")
    p.add_argument("--add", action="append", default=[], metavar="REPO",
                   help="another recorded dataset to train on as well (repeatable), e.g. DAgger corrections from "
                        "record.py --policy: only their frames from the takeover on, episodes without one left out")
    p.add_argument("--leave-out", action="append", default=[], metavar="REPO",
                   help="a dataset the default policy trained on that this run leaves out on purpose (repeatable); "
                        "without it, leaving one out stops the run (mahjong/recipe.py)")
    p.add_argument("--trim-narrowing", action="store_true",
                   help="also drop the opening gripper narrowing (start episodes at the pre-grasp opening)")
    p.add_argument("--augment", action="store_true",
                   help="colour jitter during training (AUGMENT: brightness, contrast, saturation, hue, sharpness)")
    p.add_argument("--replace-mat", type=float, default=0.0, metavar="FRACTION",
                   help="swap the black table top for another surface (grid, plain, wood, cloth, paper) in this fraction of "
                        "the episodes, e.g. 0.7 (mahjong/dataset.py); --dry-run saves a preview")
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


def run_options(args):
    """This run's options, as mahjong/recipe.py compares and registers them."""
    return recipe.options(args.augment, args.replace_mat, args.trim_narrowing)


def train_command(n, steps, private, augment=False):
    return [str(Path(sys.executable).parent / "lerobot-train"),
            f"--dataset.repo_id={n['dataset']}", "--policy.type=act", "--policy.device=cuda",
            f"--policy.repo_id={n['policy_repo']}", "--policy.push_to_hub=true", f"--policy.private={str(private).lower()}",
            "--policy.tags=[mahjong,so101,lekiwi,act]", f"--output_dir={n['output_dir']}", f"--job_name={n['job']}",
            f"--steps={steps}", f"--batch_size={BATCH}", "--num_workers=10", f"--save_freq={max(int(steps) // 7, 1000)}",
            "--log_freq=200", "--seed=1000", "--wandb.enable=false"] + (
        ["--dataset.image_transforms.enable=true", f"--dataset.image_transforms.tfs={json.dumps(AUGMENT)}"]
        if augment else [])


def preview_surfaces(repo, path, fraction, count=12):
    """A contact sheet of `count` frames from recorded `repo`, each with the surface its episode would get
    (black table top where it keeps it), to check the swap before hours of training."""
    import cv2
    import torchvision.transforms.functional as F
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from mahjong.dataset import episode_surface, replace_surface
    ds = LeRobotDataset(repo)
    tiles = []
    for ep in np.linspace(0, ds.num_episodes - 1, count).astype(int):
        meta = ds.meta.episodes[int(ep)]
        a, b = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        image = F.resize(ds[a + (b - a) // 2]["observation.images.wrist"], [224, 224], antialias=True)
        image = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        kind, texture = episode_surface(repo, int(ep), fraction=fraction)
        out = image if texture is None else replace_surface(image, texture)
        tiles.append(cv2.putText(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), f"{ep} {kind or 'black top'}", (4, 14),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))


def verify(dataset_repo, trimmed_idle, dagger=()):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from mahjong.dataset import idle_mask
    ds = LeRobotDataset(dataset_repo)
    h = ds.hf_dataset
    ep = np.array(h["episode_index"])
    state, action = np.stack(h["observation.state"]), np.stack(h["action"])
    starts = [np.flatnonzero(ep == e)[0] for e in range(ds.num_episodes) if e not in set(dagger)]
    assert not starts or np.abs(state[starts]).max() < 1e-4, "an episode doesn't start at a zero state delta"
    if trimmed_idle:
        idle = sum(idle_mask(action[ep == e]).sum() for e in range(ds.num_episodes))
        assert idle == 0, f"{idle} idle frames left"
    print(f"  verified: {ds.num_episodes} episodes ({len(dagger)} DAgger), {len(ep)} frames, zero start deltas"
          + (", no idle frames" if trimmed_idle else ""))
    return ds.num_episodes


def evaluate(policy, dataset_repo, hover):
    """(replay MAE on training episodes — optimistic, mean freeze-probe descent onset in plan steps)."""
    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from mahjong.offline_eval import NARROWED_HOVER_BELOW, descent_onset, to_bgr
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
    probe_state = np.array([0, 0, 0, 0, 0, 0.0 if hover["gripper"] < NARROWED_HOVER_BELOW else -31.0], np.float32)
    onsets = []
    for name in FREEZE_FRAMES:
        plan = policy.plan(cv2.imread(str(config.DATA / "fixtures" / "freeze_frames" / name)), probe_state)
        onset = descent_onset(plan)
        onsets.append(len(plan) if onset is None else onset)
    return float(np.mean(errors)), float(np.mean(onsets))


def main():
    args = parse_args()
    n = names(args)
    reg = recipe.registry()
    if not args.skip_train:     # --skip-train registers a run already trained: its recipe was settled then
        recipe.check(reg, args.source, args.add, args.leave_out, run_options(args))
    from mahjong.quality import dropped_episodes, flagged, gripper_fixes, quality_table
    from mahjong.dataset import build, episode_log, hover_settings
    from mahjong import offline_eval
    table, _, _ = quality_table(args.source)
    frames = int(table["frames"].sum())
    print(f"1. quality: {args.source}: {len(table)} episodes, {frames} frames"
          + (f", {len(flagged(table)[0])} flagged (scripts/quality_report.py to review)" if "score" in table else ""))
    heldout = offline_eval.excluded_episodes(args.source)
    dropped = dropped_episodes(args.source)
    train_episodes = sorted(int(e) for e in set(table.index) - set(heldout) - set(dropped))
    frames = int(table.loc[train_episodes, "frames"].sum())
    print(f"   training on {len(train_episodes)} episodes; {len(heldout)} held out for evaluation"
          + (f", {len(dropped)} dropped after review {sorted(dropped)}" if dropped else "")
          + (f"; gripper closed to 0 in {sorted(fixes)}" if (fixes := set(gripper_fixes(args.source)) & set(train_episodes)) else "")
          + ("" if heldout else f" — none: scripts/offline_eval.py --make-split {args.source} first, or every "
                                 f"episode is trained on and offline scores are optimistic"))
    extra = []
    for repo in args.add:
        added = quality_table(repo)[0]
        left_out = set(offline_eval.excluded_episodes(repo)) | set(dropped_episodes(repo))
        episodes = sorted(int(e) for e in set(added.index) - left_out)
        records = episode_log(repo)
        if not records:     # without it DAgger episodes can't be told from demos: the policy's frames would be trained on
            raise SystemExit(f"{repo} has no episode_log.jsonl: copy the recorded dataset folder itself "
                             f"(a Hub download leaves it out)")
        takeovers = [e for e in episodes if records.get(e, {}).get("takeover_frame") is not None]
        dagger = [e for e in episodes if "policy" in records.get(e, {})]
        extra.append((repo, episodes, gripper_fixes(repo)))
        frames += int(sum(added.loc[e, "frames"] - (records.get(e, {}).get("takeover_frame") or 0)
                          for e in episodes if e in takeovers or e not in dagger))
        print(f"   + {repo}: {len(episodes)} episodes"
              + (f" ({len(dagger)} DAgger: {len(takeovers)} with a takeover, from which they're trained on)" if dagger else "")
              + (f"; {len(left_out & set(added.index))} held out or dropped" if left_out & set(added.index) else ""))
    print(f"   training dataset {n['dataset']}, policy {n['policy_repo']} -> registry id {n['policy_id']}")
    if args.dry_run:
        est = int(frames * (0.56 if args.trim_narrowing else 0.70))   # measured on the first dataset
        steps = args.steps or max(10_000, int(round(EPOCHS * est / BATCH, -3)))
        print(f"2. build: trims idle + pre-fix sag{' + gripper narrowing' if args.trim_narrowing else ''}, "
              f"~{est} frames kept (estimate)"
              + (f"; the mat swapped in ~{args.replace_mat:.0%} of episodes" if args.replace_mat else "")
              + f"\n   for scale, the default, {recipe.scale(reg)}: a new version should be no smaller"
              + f"\n3. verify\n4. upload ({'private' if args.private else 'public'})\n"
              f"5. train {steps} steps (~{steps / 4.6 / 3600:.1f}h at 4.6 steps/s){', colour jitter on' if args.augment else ''}:\n"
              f"   {' '.join(train_command(n, steps, args.private, args.augment))}\n"
              f"6. evaluate each checkpoint in {n['output_dir']}/checkpoints\n7. register {n['policy_id']}"
              f"{' as default' if args.default else ''}")
        if args.replace_mat:
            path = n["output_dir"].parent / f"{n['output_dir'].name}_surfaces.jpg"
            preview_surfaces(args.source, path, args.replace_mat)
            print(f"   surface preview (one frame per sampled episode, as the build will swap it): {path}")
        return

    from lerobot.utils.constants import HF_LEROBOT_HOME
    print("2. build")
    if not args.skip_build:
        summary = build(args.source, n["dataset"], trim_narrowing=args.trim_narrowing, episodes=train_episodes,
                        close_gripper=set(gripper_fixes(args.source)), extra=extra, replace_mat=args.replace_mat)
        (HF_LEROBOT_HOME / n["dataset"] / "build_summary.json").write_text(json.dumps(summary, indent=2))
    summary = json.loads((HF_LEROBOT_HOME / n["dataset"] / "build_summary.json").read_text())
    print(f"   kept {summary['frames']}/{summary['source_frames']} frames")
    print("3. verify")
    verify(n["dataset"], trimmed_idle=True, dagger=summary.get("dagger_episodes", ()))
    if not args.skip_upload:
        print("4. upload")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        LeRobotDataset(n["dataset"]).push_to_hub(
            private=args.private, tags=["mahjong", "so101", "lekiwi", "act"],
            dataset_description=f"{' + '.join([args.source, *args.add])}, trimmed (idle frames, pre-fix takeover sag"
                                f"{', gripper narrowing' if args.trim_narrowing else ''}), "
                                f"{f'the black table top swapped for other surfaces in {args.replace_mat:.0%} of episodes, ' if args.replace_mat else ''}"
                                f"deltas re-anchored to each "
                                f"episode's first kept frame, wrist camera at 224x224. {summary['frames']} of "
                                f"{summary['source_frames']} frames kept. Built by mahjong-lekiwi scripts/make_policy.py.")
    steps = args.steps or max(10_000, int(round(EPOCHS * summary["frames"] / BATCH, -3)))
    if not args.skip_train:
        print(f"5. train {steps} steps")
        subprocess.run(train_command(n, steps, args.private, args.augment), cwd=config.LEROBOT_REPO, check=True)
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
                              "datasets": [args.source, *args.add],     # mahjong/recipe.py checks the next version against these
                              "options": run_options(args), "frames": summary["frames"], "steps": steps,
                              **({"left_out": args.leave_out} if args.leave_out else {}),     # for the record
                              "notes": f"make_policy.py from {' + '.join([args.source, *args.add])}; {summary['frames']} frames, {steps} steps; "
                                       + (f"mat swapped in {summary.get('replace_mat', 0):.0%} of episodes; " if summary.get("replace_mat") else "")
                                       + ("colour jitter (AUGMENT); " if args.augment else "")
                                       + f"held out: {len(heldout)} episodes; checkpoints (replay, onset, eval slices): "
                                       f"{[(c, round(m, 2), o, sl) for c, m, o, sl in results]}"},
             make_default=args.default)
    print(f"Done. Try it: python scripts/bench.py --policy {n['policy_id']} --trials 5 --tile 8D")


if __name__ == "__main__":
    main()
