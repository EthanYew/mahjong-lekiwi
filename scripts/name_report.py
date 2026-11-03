"""Which tiles get misread, and how much it matters, over every bench session and played hand (mahjong/name_ledger.py).

    python scripts/name_report.py                      # every tile, per detector and surface, worst first
    python scripts/name_report.py --by lighting        # also split by lighting (or spot, or session)
    python scripts/name_report.py --tile 8C --list     # one tile, with each placement behind it
    python scripts/name_report.py --since 20261103     # sessions from that date (name prefix) on
    python scripts/name_report.py --json out.json      # the same rows and placements, as a file

Bands: severe (a wrong name was accepted without asking), costly (a person asked on 25%+ of placements), flaky (votes right,
10%+ of scans wrong or unnamed), fine (none of those over 5+ placements and 3+ sessions), not enough data. The thresholds are
first guesses. "(up to)" is the highest accepted-wrong rate the placements so far can't rule out (95%).

Reads the sessions and data/wrist_review.jsonl and writes nothing unless --json is given; moves nothing. Run it from the main
checkout: sessions are per checkout (--sessions-dir reads another's).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import name_ledger


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--by", choices=["lighting", "spot", "session"], action="append", default=[],
                   help="also split each row by this (repeatable)")
    p.add_argument("--tile", default=None, help="only this tile name")
    p.add_argument("--since", default=None, help="only sessions from this date (name prefix, e.g. 20261103) on")
    p.add_argument("--list", action="store_true", help="also list each placement behind the rows")
    p.add_argument("--json", type=Path, default=None, help="also write the rows and placements to this file")
    p.add_argument("--sessions-dir", type=Path, default=None, help="read another checkout's sessions/ (default: this one's)")
    p.add_argument("--sim", action="store_true", help="include simulated sessions (left out by default: they prove nothing)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    placements, skipped, sessions = name_ledger.load(args.sessions_dir, include_sim=args.sim, since=args.since)
    every = name_ledger.rows(placements)
    if args.tile:
        placements = [p for p in placements if p["tile"] == args.tile.upper()]
    all_rows = name_ledger.rows(placements, by=args.by)
    outcomes = Counter(p["outcome"] for p in placements)
    print(f"{sessions} sessions, {len(placements)} scored placements: {outcomes[name_ledger.RIGHT]} accepted right, "
          f"{outcomes[name_ledger.ASKS]} ask, {outcomes[name_ledger.WRONG]} accepted wrong")
    print(f"  {sum(p['backfilled'] for p in placements)} backfilled from reviewed photos (votes replayed over the whole "
          f"placement, not as play would see it); {sum(p['decided_live'] for p in placements)} decided live in a hand")
    print(f"  a review overruled the name given at the time: {sum(p['overruled'] for p in placements)}")
    if skipped:
        print("  not scored: " + ", ".join(f"{why} {n}" for why, n in skipped.most_common()))
    for weights, surface in dict.fromkeys((r["weights"], r["surface"]) for r in all_rows):
        print(f"\n{surface}, detector {weights}")
        print("\n".join(name_ledger.table([r for r in all_rows if (r["weights"], r["surface"]) == (weights, surface)])))
        missing = name_ledger.untested(every, weights, surface)
        if missing and not args.tile:
            print(f"  not tested here ({len(missing)}): {' '.join(missing)}")
    if args.list:
        print("\nPlacements:")
        for p in placements:
            print(f"  {p['session']} {p['spot']:<5} {p['tile']:<3} {p['outcome']:<14} truth: {p['truth_source']:<7} "
                  f"{p['right']} right, {p['wrong']} wrong, {p['unnamed']} unnamed of {p['scans']}  "
                  + " ".join(f"{name} x{n}" for name, n in p["mixups"].most_common(3)))
    if args.json:
        args.json.write_text(json.dumps({"rows": all_rows, "placements": placements, "skipped": skipped}, indent=1) + "\n")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
