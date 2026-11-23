"""Rank a dataset's episodes by hesitation / erratic motion (mahjong/quality.py), to review and drop the
worst before training; with each slice's pickup phases (wait at the hover, approach, pauses, reversals,
detours, wobble), and optionally the hover→grasp move against another recording at the same places.

    python scripts/quality_report.py --dataset Yewy/mahjong_mat_pickup_lekiwi_cl_hover
    python scripts/quality_report.py --dataset ... --episode 37     # when each reversal happened
    python scripts/quality_report.py --dataset Yewy/mahjong_mat_eval_v1 --phases     # phases per episode
    python scripts/quality_report.py --dataset ... --drop 8,39 --reason "searching"   # never trained on
    python scripts/quality_report.py --dataset ... --fix-gripper 117 --reason "stopped at 5.5%"   # closed to 0 when built
    python scripts/quality_report.py --dataset Yewy/mahjong_mat_eval_v1 --against Yewy/mahjong_mat_pickup_lekiwi_cl_hover
    python scripts/quality_report.py --dataset ... --clean Yewy/<name>_clean     # a copy without pauses or dropped episodes
    python scripts/quality_report.py --grip-reach --recipe pickup_mat_cl_v13 --calibration   # demos that gripped short

The hover→grasp check compares only demos that steered: one lowered straight down says "close enough for the
jaws", not where the tile was. A shift beyond the jaws' ~6 mm at most spots, in one direction, is worth a
collect_corrections.py session; a difference at a few spots is as likely to be how the demos were done.

--grip-reach finds demos whose jaws closed short of the tile's middle, toward the robot (what the bench's `d`, "gripped
low", marks): where the close landed in x, less the tile's x, against what the bench attempts not marked low did at that
position (mahjong/quality.py, "grip reach"). It reads the bench sessions, so run it from the main checkout or pass
--sessions. It only proposes: look at each one's close in the wrist video, then --drop the ones that look low.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, recipe
from mahjong.quality import (DEFAULT_FLAG_SCORE, JAW_TOLERANCE_MM, PHASE_COLUMNS, REACH_SHORT_MM, REACH_SPOT_ONLY_SHORT_MM,
                             ReachReference, bench_reach_labels, clean_copy, drift_table, drop_episodes, dropped_episodes,
                             fix_gripper, grip_reach_table, load_episodes, phase_table, quality_table, reach_calibration,
                             thin_spots)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", help="recorded dataset repo id, e.g. Yewy/mahjong_mat_pickup_v3")
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
    p.add_argument("--clean", metavar="REPO",
                   help="write a copy under this new name: every stop in the commands cut to one frame, dropped episodes "
                        "left out; the recording itself isn't changed")
    p.add_argument("--reason", default="reviewed", help="with --drop / --fix-gripper: why, kept with each episode")
    p.add_argument("--grip-reach", action="store_true",
                   help="demos that closed short of the tile's middle, against the bench attempts not marked gripped low")
    p.add_argument("--recipe", metavar="POLICY",
                   help="with --grip-reach: every dataset this registered policy trained on, instead of --dataset")
    p.add_argument("--calibration", action="store_true",
                   help="with --grip-reach: how many marked-low and unmarked bench attempts each cutoff would catch")
    p.add_argument("--sessions", default=str(config.SESSIONS),
                   help="with --grip-reach: the bench sessions folder (default %(default)s; a worktree has none)")
    args = p.parse_args()
    if args.grip_reach:
        others = [name for name in ("drop", "fix_gripper", "clean", "against", "episode") if getattr(args, name) is not None]
        others += ["phases"] * args.phases
        if others:
            p.error("--grip-reach only reports; it can't go with " + ", ".join("--" + o.replace("_", "-") for o in others)
                    + " (run --drop after looking at the proposals)")
        return grip_reach(args)
    if args.recipe or args.calibration:
        p.error("--recipe and --calibration go with --grip-reach")
    if not args.dataset:
        p.error("--dataset is required (--recipe goes with --grip-reach)")
    if args.fix_gripper:
        fixed = fix_gripper(args.dataset, [int(e) for e in args.fix_gripper.split(",") if e.strip()], args.reason)
        print(f"Gripper closed to 0 when {args.dataset} is built: {sorted(fixed)}\n")
    if args.drop:
        dropped = drop_episodes(args.dataset, [int(e) for e in args.drop.split(",") if e.strip()], args.reason)
        print(f"Dropped from {args.dataset} (make_policy.py leaves them out): {sorted(dropped)}\n")
    if args.clean:
        summary = clean_copy(args.dataset, args.clean)
        print(f"\nWrote {args.clean}: {summary['episodes']} episodes, {summary['frames']} of {summary['source_frames']} "
              f"frames ({summary['root']})\n")
        return
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


def grip_reach(args):
    if bool(args.dataset) == bool(args.recipe):
        raise SystemExit("--grip-reach takes --dataset REPO or --recipe POLICY")
    if args.recipe:
        entry = recipe.registry()["policies"].get(args.recipe)
        repos = recipe.registered_datasets(entry) if entry else None
        if not repos:
            raise SystemExit(f"can't tell which datasets {args.recipe} trained on (data/policies.json)")
    else:
        repos = [args.dataset]
    labels = bench_reach_labels(args.sessions)
    if not labels:
        raise SystemExit(f"no face-up bench attempts with a gripped-low mark under {args.sessions}: run from the main "
                         "checkout, or pass --sessions")
    reference = ReachReference(labels)
    print(f"Reference: {reference.attempts} face-up bench attempts in {reference.sessions} sessions, "
          f"{reference.marked_low} marked gripped low; fitted on the {reference.attempts - reference.marked_low} unmarked. "
          f"Covers tiles at x {reference.lo[0]:.3f}-{reference.hi[0]:.3f}, y {reference.lo[1]:+.3f} to {reference.hi[1]:+.3f}.")
    if args.calibration:
        print("\nOn the bench attempts: a cutoff this far short catches this share of the marked-low ones, and flags this "
              "share of the unmarked")
        for mm, caught, flagged in reach_calibration(labels, reference):
            print(f"  {mm:>2} mm short: {caught:4.0%} of marked low, {flagged:4.0%} of unmarked")
    print(f"\nProposed: {REACH_SHORT_MM:.0f}+ mm short of the reference ({REACH_SPOT_ONLY_SHORT_MM:.0f}+ where the log has "
          "only the spot's saved position, 'spot-only'). 'outside': no bench attempts near that tile, not judged.")
    tables = {}
    for repo in repos:
        try:
            tables[repo] = grip_reach_table(repo, reference)
        except SystemExit as e:
            print(f"\n{repo}: skipped ({e})")
    print(f"\n{'dataset':<40} {'eps':>4} {'judged':>6} {'outside':>7} {'median':>7} {'proposed':>8}")
    for repo, t in tables.items():
        kept = t[~t["dropped"]]
        judged = kept[kept["basis"].isin(["measured", "spot-only"])]
        median = f"{judged['shortfall_mm'].median():+.1f}" if len(judged) else "-"
        print(f"{repo:<40} {len(kept):>4} {len(judged):>6} {int((kept['basis'] == 'outside').sum()):>7} {median:>7} "
              f"{int(t['propose'].sum()):>8}")
    print("  (episodes not already dropped; median shortfall in mm, negative is short)")
    for repo, t in tables.items():
        proposed = t[t["propose"]].sort_values("shortfall_mm")
        if not len(proposed):
            continue
        print(f"\n{repo}: {len(proposed)} proposed")
        print(f"{'ep':>5} {'spot':<5} {'tile':<9} {'short':>6} {'close':>6} {'basis':<9}")
        for ep, r in proposed.iterrows():
            close = "-" if pd.isna(r["close_mm"]) else f"{r['close_mm']:.0f}mm"
            print(f"{ep:>5} {r['spot']:<5} {r['tile']:<9} {r['shortfall_mm']:>+6.1f} {close:>6} {r['basis']:<9}")
        thin = thin_spots(t)
        if thin:
            print("  would leave spots thin (episodes now -> left): " + ", ".join(f"{s} {n}->{k}" for s, (n, k) in thin.items())
                  + ". Some short grips may be keeping clear of the next tile out (the 2026-11-23 retro's hypothesis)")
        print(f"  python scripts/quality_report.py --dataset {repo} --drop {','.join(map(str, proposed.index))} "
              f"--reason \"gripped short of the tile's middle (--grip-reach)\"")
    if args.csv and tables:
        pd.concat(tables, names=["dataset"]).to_csv(args.csv)
    print("\nLook at each one's close in the wrist video before dropping it: a number says the close was short, not why.")


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
