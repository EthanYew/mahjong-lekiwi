"""Rank a dataset's episodes by hesitation / erratic motion (mahjong/quality.py), to review and drop the
worst before training.

    python scripts/quality_report.py --dataset Yewy/mahjong_mat_pickup_lekiwi_cl_hover
    python scripts/quality_report.py --dataset ... --episode 37     # when each reversal happened
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong.quality import DEFAULT_FLAG_SCORE, dataset_root, quality_table


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--worst", type=int, default=15)
    p.add_argument("--flag", type=float, default=DEFAULT_FLAG_SCORE)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    table, events, fps = quality_table(args.dataset)
    if args.episode is not None:
        row = table.loc[args.episode]
        print(f"Episode {args.episode}: {row['duration']:.1f}s, tip path {row['path_mm']:.0f}mm, "
              f"{row['tip_rev']} tip / {row['grip_rev']} gripper reversals, jerk {row['jerk']:.1f}")
        for frame, what, size, unit in events[args.episode]:
            print(f"  {frame / fps:6.2f}s  {what:<8} turned back after a {size:.0f}{unit} run")
        return
    medians = table[["duration", "path_mm", "tip_rev", "grip_rev", "jerk", "idle"]].median()
    print(f"{args.dataset}: {len(table)} episodes. Median {medians['duration']:.1f}s, {medians['tip_rev']:.0f} tip / "
          f"{medians['grip_rev']:.0f} gripper reversals, jerk {medians['jerk']:.1f}, idle {medians['idle']:.0%}\n")
    scored = "score" in table
    print(f"{'ep':>4} {'score':>6} {'worst':<9} {'dur s':>6} {'tip_rev':>7} {'grip_rev':>8} {'jerk':>7} {'idle':>5}  tile")
    for ep, r in (table.head(args.worst) if scored else table).iterrows():
        print(f"{ep:>4} {r.get('score', float('nan')):>6.1f} {r.get('worst', ''):<9} {r['duration']:>6.1f} "
              f"{r['tip_rev']:>7} {r['grip_rev']:>8} {r['jerk']:>7.1f} {r['idle']:>5.0%}  {r['tile']}")
    if args.csv:
        table.to_csv(args.csv)
    if scored:
        flagged = sorted(int(e) for e in table.index[table["score"] >= args.flag])
        print(f"\n{len(flagged)} scored >= {args.flag}: {flagged}")
        if flagged:
            print(f"Review them (--episode N), then to write a filtered copy:\n  lerobot-edit-dataset --repo_id {args.dataset} "
                  f"--root {dataset_root(args.dataset)} --new_repo_id {args.dataset}_filtered "
                  f"--operation.type delete_episodes --operation.episode_indices \"{flagged}\"")


if __name__ == "__main__":
    main()
