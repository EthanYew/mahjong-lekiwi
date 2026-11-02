"""Where things stand, from the files that decide them (not from the docs, which go stale).

    python scripts/status.py                       # checkout, default policy, policies on the bench, eval sets, layouts...
    python scripts/status.py --set-default pickup_mat_cl_v5 --evidence sessions/20261101_100000_bench --why "..."

The default policy, the registered policies and how each did on the bench (every bench session's attempts:
clean, then picked, docs/scorecard.md; the first run works out and saves each session's clean count),
the eval sets, the detector exam, the layouts, the drop zone and whether the bench checked a pickup there
(markers.zone_test), and this checkout against origin/main. ROADMAP.md's status table points here.

--set-default is how the default changes: it writes the decision's evidence next to it in
data/policies.json (default_note), which tests/test_docs.py checks agrees with the default.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import checkout, config, layout, markers, scorecard
from mahjong.session import read_events


def bench_results(sessions_dir=None):
    """policy -> {"attempts", "picked", "sessions", "last"} over every bench session."""
    out = defaultdict(lambda: {"attempts": 0, "picked": 0, "sessions": 0, "last": ""})
    for events_file in sorted((sessions_dir or config.SESSIONS).glob("*_bench/events.jsonl")):
        seen = set()
        for e in read_events(events_file.parent):
            if e["type"] == "attempt" and e.get("policy"):
                row = out[e["policy"]]
                row["attempts"] += 1
                row["picked"] += bool(e.get("success"))
                row["last"] = events_file.parent.name[:15]
                seen.add(e["policy"])
        for policy in seen:
            out[policy]["sessions"] += 1
    return dict(out)


def bench_clean(sessions_dir=None):
    """policy -> clean attempts over every bench session (docs/scorecard.md). Slow once per new session."""
    out = defaultdict(int)
    for events_file in sorted((sessions_dir or config.SESSIONS).glob("*_bench/events.jsonl")):
        for policy, clean in scorecard.session_clean(events_file.parent).items():
            out[policy] += clean
    return dict(out)


def report(sessions_dir=None):
    reg = json.loads(config.POLICIES.read_text())
    lines = []
    branch = checkout.git("rev-parse", "--abbrev-ref", "HEAD")
    lines.append(f"Checkout: {branch} at {checkout.git('rev-parse', '--short', 'HEAD')}"
                 + "".join(f"\n  WARNING: {w}" for w in checkout.warnings()))

    note = reg.get("default_note", {})
    lines.append(f"\nDefault policy: {reg['default']} (since {note.get('since', '?')})"
                 f"\n  evidence: {note.get('evidence', 'none recorded')}" + (f"\n  why: {note['why']}" if note.get("why") else ""))
    results = bench_results(sessions_dir)
    clean = bench_clean(sessions_dir)
    lines.append(f"\n{'policy':<22} {'bench clean':>12} {'picked':>9} {'sessions':>9}  last bench")
    for pid in reg["policies"]:
        r = results.get(pid, {"attempts": 0, "picked": 0, "sessions": 0, "last": "never"})
        picked = f"{r['picked']}/{r['attempts']}" if r["attempts"] else "-"
        tidy = f"{clean.get(pid, 0)}/{r['attempts']}" if r["attempts"] else "-"
        lines.append(f"{pid:<22} {tidy:>12} {picked:>9} {r['sessions']:>9}  {r['last']}"
                     + ("   <- default" if pid == reg["default"] else ""))
    lines.append("  clean: picked up, nothing bumped, no one helped (docs/scorecard.md). Totals mix layouts and surfaces, and\n"
                 "  bumps were marked more closely as the sessions went on: compare policies only within a paired session")

    split = json.loads((config.DATA / "eval_split.json").read_text())
    lines.append(f"\nEval sets (never trained on): {', '.join(split.get('eval_sets', [])) or 'none'}; "
                 f"held-out episodes: {sum(len(v) for v in split.get('heldout', {}).values())}")
    exam = json.loads((config.DATA / "detector_exam.json").read_text()).get("sessions", {})
    lines.append(f"Detector exam sessions: {len(exam)} ({', '.join(exam) or 'none'})")

    lines.append("\nLayouts:")
    for name, entry in layout.load_layouts().items():
        lines.append(f"  {name:<14} {len(entry['spots']):>2} spots  {entry.get('note') or '(no note)'}")

    zone = markers.load_zone()
    if zone is None:
        lines.append("\nDrop zone: not taught (play.py teaches it on its first run)")
    else:
        test = markers.zone_test(zone["xy"], layout.load("hand13")["spots"],
                                 layout.load(markers.ZONE_TEST_LAYOUT)["spots"], sessions_dir=sessions_dir)
        lines.append(f"\nDrop zone at ({zone['xy'][0]:.3f}, {zone['xy'][1]:+.3f}): "
                     + (f"benched at {markers.ZONE_TEST_LAYOUT} {test['spot']}, {test['picked']} picked up, "
                        f"{test['returned']} returned" if test["ok"] else "; ".join(test["problems"])))

    left = checkout.leftovers()
    lines.append(f"\nMerged branches not closed: {len(left)}" + (" (python scripts/merge_pr.py --close)" if left else ""))
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set-default", metavar="ID", help="make this registered policy the default (with --evidence)")
    p.add_argument("--evidence", help="what the switch rests on, e.g. the bench session folder(s) and their scores")
    p.add_argument("--why", default="", help="the reason, in a sentence")
    args = p.parse_args(argv)
    if args.set_default and not args.evidence:
        p.error("--set-default needs --evidence (e.g. sessions/<time>_bench and what it showed)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.set_default:
        from mahjong.policy import set_default
        note = set_default(args.set_default, args.evidence, args.why)
        print(f"Default is now {args.set_default}: {note}\nCommit data/policies.json, and update ROADMAP.md's status.")
        return
    print(report())


if __name__ == "__main__":
    main()
