"""Rank a dataset's episodes by hesitation / erratic motion (mahjong/quality.py), to review and drop the
worst before training; with each slice's pickup phases (wait at the hover, approach, pauses, reversals,
detours, wobble), and optionally the hover→grasp move against another recording at the same places.

    python scripts/quality_report.py --dataset Yewy/mahjong_mat_pickup_lekiwi_cl_hover
    python scripts/quality_report.py --dataset ... --episode 37     # when each reversal happened
    python scripts/quality_report.py --dataset Yewy/mahjong_mat_eval_v1 --phases     # phases per episode
    python scripts/quality_report.py --dataset ... --drop 8,39 --reason "searching"   # never trained on
    python scripts/quality_report.py --dataset ... --fix-gripper 117 --reason "stopped at 5.5%"   # closed to 0 when built
    python scripts/quality_report.py --dataset Yewy/mahjong_mat_eval_v1 --against Yewy/mahjong_mat_pickup_lekiwi_cl_hover

The hover→grasp check compares only demos that steered: one lowered straight down says "close enough for the
jaws", not where the tile was. A shift beyond the jaws' ~6 mm at most spots, in one direction, is worth a
collect_corrections.py session; a difference at a few spots is as likely to be how the demos were done.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong.quality import (DEFAULT_FLAG_SCORE, JAW_TOLERANCE_MM, PHASE_COLUMNS, drift_table, drop_episodes,
                             dropped_episodes, fix_gripper, load_episodes, phase_table, quality_table)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, help="recorded dataset repo id, e.g. Yewy/mahjong_mat_pickup_v3")
    p.add_argument("--worst", type=int, default=15, help="rows of the ranking shown (default %(default)s)")
    p.add_argument("--flag", type=float, default=DEFAULT_FLAG_SCORE,
                   help="score at which an episode is flagged (default %(default)s)")
    p.add_argument("--episode", type=int, default=None, help="this episode's events: when each reversal happened")
    p.add_argument("--csv", default=None, help="save the table to this path")
    p.add_argument("--phases", action="store_true", help="print each episode's pickup phases")
    p.add_argument("--against", metavar="REPO", help="compare hover→grasp moves with this recording, spot by spot")
    p.add_argument("--drop", metavar="EPS", help="comma-separated episodes to leave out of training (data/dropped_episodes.json)")
    p.add_argument("--fix-gripper", metavar="EPS",
                   help="comma-separated episodes whose gripper stopped short of 0 on the tile: built with it closed "
                        "to 0 (data/episode_fixes.json)")
    p.add_argument("--reason", default="reviewed", help="with --drop / --fix-gripper: why, kept with each episode")
    args = p.parse_args()
    if args.fix_gripper:
        fixed = fix_gripper(args.dataset, [int(e) for e in args.fix_gripper.split(",") if e.strip()], args.reason)
        print(f"Gripper closed to 0 when {args.dataset} is built: {sorted(fixed)}\n")
    if args.drop:
        dropped = drop_episodes(args.dataset, [int(e) for e in args.drop.split(",") if e.strip()], args.reason)
        print(f"Dropped from {args.dataset} (make_policy.py leaves them out): {sorted(dropped)}\n")
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
    scored, dropped = "score" in table, dropped_episodes(args.dataset)
    if dropped:
        print(f"Dropped after review, not trained on (x below): {sorted(dropped)}\n")
    print(f"{'ep':>4} {'score':>6} {'worst':<9} {'dur s':>6} {'tip_rev':>7} {'grip_rev':>8} {'jerk':>7} {'idle':>5}  tile")
    for ep, r in (table.head(args.worst) if scored else table).iterrows():
        print(f"{ep:>4} {r.get('score', float('nan')):>6.1f} {r.get('worst', ''):<9} {r['duration']:>6.1f} "
              f"{r['tip_rev']:>7} {r['grip_rev']:>8} {r['jerk']:>7.1f} {r['idle']:>5.0%}  {r['tile']}"
              + ("  x" if ep in dropped else ""))
    if args.csv:
        table.to_csv(args.csv)
    if scored:
        flagged = sorted(int(e) for e in table.index[table["score"] >= args.flag] if int(e) not in dropped)
        print(f"\n{len(flagged)} scored >= {args.flag}{' (not yet dropped)' if dropped else ''}: {flagged}")
        if flagged:
            print(f"Review them (--episode N), then leave the bad ones out of training:\n  "
                  f"python scripts/quality_report.py --dataset {args.dataset} --drop {','.join(map(str, flagged))} --reason ...")
    print_phases(phase_table(args.dataset), args.phases)
    if args.against:
        print_drift(args.dataset, args.against)


def print_phases(phases, per_episode):
    """Medians per slice (and per episode with --phases): where the time and the hesitation went."""
    header = (f"{'':>9} {'narrow':>7} {'wait':>5} {'approach':>8} {'pauses':>6} {'revs':>4} {'detour':>6} "
              f"{'after':>5} {'wobble':>6}")
    def line(label, r):
        return (f"{label:>9} {r['narrow_s']:>6.1f}s {r['wait_s']:>4.1f}s {r['approach_s']:>7.1f}s {r['pause_s']:>5.1f}s "
                f"{r['approach_rev']:>4.0f} {r['path_ratio']:>5.2f}x {r['after_s']:>4.1f}s {r['wobble_mm']:>4.2f}mm")
    print("\nPickup phases (median): narrowing, wait at the hover, approach to the close and its pauses, reversals,"
          "\npath over the straight line, after the close, wobble (the tip less a 7-frame average)")
    print(header)
    for name, group in phases.groupby("slice"):
        print(line(name, group[PHASE_COLUMNS].median()))
    if per_episode:
        print(f"\n{'ep':>4} {'slice':<9} {'spot':<7}" + header[9:])
        for ep, r in phases.iterrows():
            print(f"{ep:>4} {r['slice']:<9} {r['spot']:<7}" + line("", r)[9:])


def print_drift(repo, reference):
    episodes, _ = load_episodes(repo)
    episodes = {e: v for e, v in episodes.items() if v[1].get("slice") != "tilt"}     # turned tiles grasp elsewhere
    ref, _ = load_episodes(reference)
    table = drift_table(episodes, ref)
    print(f"\nHover→grasp move (mm, pincer tip, deliberate hover offsets removed) vs {reference}, demos that steered:")
    print(f"{'spot':<7} {'eps':>3} {'straight':>8} {'move':>14} {'ref eps':>7} {'ref straight':>12} {'ref move':>14} {'shift':>14}")
    for spot, r in table.iterrows():
        ref_part = (f"{r['ref_x']:+6.1f},{r['ref_y']:+6.1f} {r['shift_x']:+6.1f},{r['shift_y']:+6.1f}"
                    if r["ref_x"] == r["ref_x"] else f"{'(none near)':>14} {'':>14}")
        ref_straight = f"{r['reference_straight_down']:.0%}" if r["reference"] else "-"
        print(f"{spot:<7} {r['episodes']:>3.0f} {r['straight_down']:>8.0%} {r['move_x']:+6.1f},{r['move_y']:+6.1f} "
              f"{r['reference']:>7.0f} {ref_straight:>12} {ref_part}")
    compared = table.dropna(subset=["shift_x"])
    if len(compared):
        big = (compared[["shift_x", "shift_y"]].abs().max(axis=1) > JAW_TOLERANCE_MM)
        print(f"\n{big.sum()} of {len(compared)} compared spots shifted beyond the jaws' {JAW_TOLERANCE_MM:.0f} mm; "
              f"median shift {compared.shift_x.median():+.1f},{compared.shift_y.median():+.1f} mm. Straight down: "
              f"{table.straight_down.mean():.0%} here vs {compared.reference_straight_down.mean():.0%} in the reference near these spots.")


if __name__ == "__main__":
    main()
