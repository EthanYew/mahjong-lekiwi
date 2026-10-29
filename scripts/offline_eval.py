"""Score policies on recorded episodes they never trained on, without the arm (mahjong/offline_eval.py).

data/eval_split.json lists the eval episodes: held-out episodes of training recordings (make_policy.py
leaves them out of training) and whole eval-only recordings (record.py --slice familiar/new/dim/tilt/jitter).
Scores are per slice — never averaged into one number — and optionally per spot or lighting.

    python scripts/offline_eval.py --show
    python scripts/offline_eval.py --make-split Yewy/mahjong_mat_pickup_lekiwi_cl_hover   # 1+ per spot, >= 10%
    python scripts/offline_eval.py --add-eval-set Yewy/mahjong_mat_eval_v1
    python scripts/offline_eval.py --policy pickup_mat_cl
    python scripts/offline_eval.py --policy pickup_mat_cl --against pickup_mat_cl_v1 --by spot

With --against, the gates compare the two: no freezes, and no gated slice (all but tilt) much worse than
the reference. Offline scores screen out bad policies; the bench decides what ships.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import offline_eval
from mahjong.dataset import episode_log


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--show", action="store_true", help="print the eval split")
    p.add_argument("--make-split", metavar="REPO", help="hold out episodes of this training recording")
    p.add_argument("--fraction", type=float, default=offline_eval.HELDOUT_FRACTION,
                   help="with --make-split: hold out one episode per spot with 3+ episodes, topped up to this share "
                        "(default %(default)s)")
    p.add_argument("--seed", type=int, default=0, help="with --make-split: random seed, so the split can be remade")
    p.add_argument("--force", action="store_true", help="with --make-split: replace an existing split for REPO")
    p.add_argument("--add-eval-set", metavar="REPO", help="add an eval-only recording (never trained on)")
    p.add_argument("--policy", help="registered policy id, or a checkpoint's pretrained_model folder")
    p.add_argument("--against", help="reference policy for the gates (e.g. the current default)")
    p.add_argument("--by", choices=["slice", "spot", "lighting", "surface"], default="slice", help="extra breakdown")
    p.add_argument("--max-episodes", type=int, default=None, help="per source, for a quick look")
    p.add_argument("--stride", type=int, default=offline_eval.EVAL_STRIDE,
                   help="score every Nth frame of each episode (default %(default)s)")
    return p.parse_args()


def show(split):
    held = split.get("heldout", {})
    if not held and not split.get("eval_sets"):
        print("The eval split is empty: --make-split REPO and/or --add-eval-set REPO.")
    for src, episodes in held.items():
        records = episode_log(src)
        print(f"{src}: {len(episodes)} held out — " + ", ".join(
            f"{e} ({offline_eval.labels(records.get(e, {}), 'heldout')['spot']})" for e in episodes))
    for src in split.get("eval_sets", []):
        print(f"{src}: eval-only, every episode")


def make_split(args, split):
    if args.make_split in split.get("eval_sets", []):
        raise SystemExit(f"{args.make_split} is an eval set — all of it is already held out")
    if split.get("heldout", {}).get(args.make_split) and not args.force:
        raise SystemExit(f"{args.make_split} already has a split; changing it makes earlier scores incomparable "
                         f"(--force to replace it)")
    records = episode_log(args.make_split)
    if not records:
        raise SystemExit(f"no episode log for {args.make_split} — can't tell its episodes' positions apart")
    chosen = offline_eval.choose_heldout(records, args.fraction, args.seed)
    split.setdefault("heldout", {})[args.make_split] = chosen
    offline_eval.save_split(split)
    print(f"Held out {len(chosen)} of {len(records)} episodes (seed {args.seed}).")
    show(split)


def load_policy(name):
    from mahjong.policy import Policy, registry
    if name in registry()["policies"]:
        return Policy(name)
    if Path(name).is_dir():
        return Policy(source=name)
    raise SystemExit(f"{name!r} is neither a registered policy nor a checkpoint folder")


def score(name, split, args):
    policy = load_policy(name)
    print(f"Scoring {policy.id} (hover gripper {policy.hover['gripper']:.0f}%)")
    rows = offline_eval.evaluate(policy, split, max_episodes=args.max_episodes, stride=args.stride)
    print(f"  saved {offline_eval.save_results(policy.id, rows, split)}")
    return policy.id, rows


def main():
    args = parse_args()
    split = offline_eval.load_split()
    if args.make_split:
        return make_split(args, split)
    if args.add_eval_set:
        if args.add_eval_set in split.get("heldout", {}):
            raise SystemExit(f"{args.add_eval_set} is a training recording with held-out episodes")
        split.setdefault("eval_sets", [])
        if args.add_eval_set not in split["eval_sets"]:
            split["eval_sets"].append(args.add_eval_set)
        offline_eval.save_split(split)
        return show(split)
    if args.show or not args.policy:
        return show(split)
    if not offline_eval.eval_items(split):
        raise SystemExit("The eval split is empty: --make-split REPO and/or --add-eval-set REPO first.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    results = [score(args.policy, split, args)]
    if args.against:
        results.append(score(args.against, split, args))
    for policy_id, rows in results:
        print("\n" + offline_eval.table(offline_eval.summarise(rows), f"{policy_id} by slice"))
        if args.by != "slice":
            print(offline_eval.table(offline_eval.summarise(rows, by=args.by), f"{policy_id} by {args.by}"))
    print("\n  arm/grip: first-action error, teacher-forced; close: open-loop close timing error (s, abs);"
          "\n  freeze: plan never descends from the hover; lag: plan descent onset - demo's (steps);"
          "\n  grasp: where the jaws close vs the demo (median mm); roll: wrist roll at the close (deg, abs);"
          "\n  high: the jaw tip's height when the plan shuts, above the demo's (median mm; + closes high on the tile)")
    reference = results[1][1] if len(results) > 1 else None
    checks = offline_eval.gates(results[0][1], reference)
    print(f"\nGates for {results[0][0]}" + (f" against {results[1][0]}" if reference else " (freezes only)") + ":")
    for ok, message in checks:
        print(f"  {'pass' if ok else 'FAIL'}  {message}")
    print("All gates pass." if all(ok for ok, _ in checks) else "Some gates fail — not worth bench time yet.")


if __name__ == "__main__":
    main()
