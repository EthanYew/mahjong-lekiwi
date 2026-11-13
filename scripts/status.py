"""Where things stand, from the files that decide them (not from the docs, which go stale).

    python scripts/status.py                       # checkout, default policy, policies on the bench, eval sets, layouts...
    python scripts/status.py --set-default pickup_mat_cl_v5 --evidence sessions/20261101_100000_bench --why "..."

The default policy, the registered policies and how each did on the bench (every bench session's attempts:
clean, then picked, docs/scorecard.md; the first run works out and saves each session's clean count),
the standard setup and each policy's clean count on it, the eval sets, the detector exam, the layouts, the drop zone and whether the bench checked a pickup there
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

from mahjong import checkout, config, layout, markers, name_ledger, scorecard
from mahjong.session import is_sim, read_events


def real_benches(sessions_dir=None):
    """Every bench session's events file but simulated ones: `bench.py --sim --policy <id>` logs attempts under a real policy's
    id (20261112_091817_bench gave v10 a clean pickup that never happened)."""
    return [f for f in sorted((sessions_dir or config.SESSIONS).glob("*_bench/events.jsonl")) if not is_sim(f.parent)]


def bench_results(sessions_dir=None):
    """policy -> {"attempts", "picked", "sessions", "last"} over every bench session."""
    out = defaultdict(lambda: {"attempts": 0, "picked": 0, "sessions": 0, "last": ""})
    for events_file in real_benches(sessions_dir):
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
    for events_file in real_benches(sessions_dir):
        for policy, clean in scorecard.session_clean(events_file.parent).items():
            out[policy] += clean
    return dict(out)


def standard_results(sessions_dir=None):
    """policy -> {"attempts", "clean", "sessions"} over the bench sessions on the standard setup only."""
    out = defaultdict(lambda: {"attempts": 0, "clean": 0, "sessions": 0})
    for events_file in sorted((sessions_dir or config.SESSIONS).glob("*_bench/events.jsonl")):
        events = read_events(events_file.parent)
        if scorecard.off_standard(events):
            continue
        clean = scorecard.session_clean(events_file.parent)
        attempts = defaultdict(int)
        for e in events:
            if e["type"] == "attempt" and e.get("policy"):
                attempts[e["policy"]] += 1
        for policy, count in attempts.items():
            row = out[policy]
            row["attempts"] += count
            row["clean"] += clean.get(policy, 0)
            row["sessions"] += 1
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
    standard = scorecard.standard_setup()
    lines.append(f"\nStandard setup: {standard['surface']}, {standard['layout']} (data/standard_setup.json, since {standard['since']})")
    on_it = standard_results(sessions_dir)
    lines += [f"  {pid:<20} {r['clean']}/{r['attempts']} clean over {r['sessions']} session(s), for orientation: gates read one session"
              for pid, r in on_it.items()] or ["  no bench sessions on it yet"]

    split = json.loads((config.DATA / "eval_split.json").read_text())
    lines.append(f"\nEval sets (never trained on): {', '.join(split.get('eval_sets', [])) or 'none'}; "
                 f"held-out episodes: {sum(len(v) for v in split.get('heldout', {}).values())}")
    exam = json.loads((config.DATA / "detector_exam.json").read_text()).get("sessions", {})
    lines.append(f"Detector exam sessions: {len(exam)} ({', '.join(exam) or 'none'})")
    lines.append(name_ledger.worst_line(sessions_dir))

    lines.append("\nLayouts:")
    for name, entry in layout.load_layouts().items():
        lines.append(f"  {name:<14} {len(entry['spots']):>2} spots  {entry.get('note') or '(no note)'}")

    zone = markers.load_zone()
    if zone is None:
        lines.append("\nDrop zone: not taught (play.py teaches it on its first run)")
    else:
        test = markers.zone_test(zone["xy"], layout.load(markers.PLAY_LAYOUT)["spots"],
                                 layout.load(markers.ZONE_TEST_LAYOUT)["spots"], sessions_dir=sessions_dir)
        lines.append(f"\nDrop zone at ({zone['xy'][0]:.3f}, {zone['xy'][1]:+.3f}): "
                     + (f"benched at {markers.ZONE_TEST_LAYOUT} {test['spot']}, {test['picked']} picked up, "
                        f"{test['returned']} returned, {test['carried']} carried to a slot (median "
                           f"{test['carried_median_mm']:.0f}mm from home)" if test["ok"] else "; ".join(test["problems"])))

    left = checkout.leftovers()
    lines.append(f"\nMerged branches not closed: {len(left)}" + (" (python scripts/merge_pr.py --close)" if left else ""))
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set-default", metavar="ID", help="make this registered policy the default (with --evidence)")
    p.add_argument("--evidence", help="what the switch rests on, e.g. the bench session folder(s) and their scores")
    p.add_argument("--why", default="", help="the reason, in a sentence")
    p.add_argument("--since", default=None, help="with --set-default: the day it was decided, YYYY-MM-DD (default today)")
    args = p.parse_args(argv)
    if args.set_default and not args.evidence:
        p.error("--set-default needs --evidence (e.g. sessions/<time>_bench and what it showed)")
    return args


def stale_docs(policy_id, repo=None):
    """Where the docs still disagree with a new default: what tests/test_docs.py checks, so the pre-commit hook would
    refuse the commit until each is fixed (2026-11-06: a default changed alone on main could be neither committed nor merged)."""
    import re
    repo = Path(repo) if repo else Path(__file__).resolve().parents[1]
    readme, claude, roadmap = ((repo / name).read_text() for name in ("README.md", "CLAUDE.md", "ROADMAP.md"))
    out = []
    if f"(default: `{policy_id}`" not in readme:
        out.append(f"README.md: the policy list's \"(default: `...`\" names another policy; make it `{policy_id}`")
    for name, text in (("README.md", readme), ("CLAUDE.md", claude)):
        for line in text.splitlines():
            if "record.py" in line and (m := re.search(r"--policy (pickup_mat_cl\w*)", line)) and m.group(1) != policy_id:
                out.append(f"{name}: a DAgger example drives {m.group(1)}: {line.strip()[:90]}")
    status = roadmap.split("\n## Status\n", 1)[-1].split("\n## ", 1)[0]
    if policy_id not in status:
        out.append(f"ROADMAP.md: the status table's Default policy row doesn't name {policy_id}")
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.set_default:
        from mahjong.policy import set_default
        note = set_default(args.set_default, args.evidence, args.why, args.since)
        print(f"Default is now {args.set_default}: {note}")
        if checkout.git("rev-parse", "--abbrev-ref", "HEAD") == "main":
            print("WARNING: this is main. A default is a decision: make it on a branch and merge it with merge_pr.py "
                  "(undo here with: git restore data/policies.json)")
        stale = stale_docs(args.set_default)
        print("Still to change before the commit passes the tests:\n  " + "\n  ".join(stale) if stale else
              "README.md, CLAUDE.md and ROADMAP.md already name it.")
        print("Then commit data/policies.json with the docs, open a PR, and merge it with scripts/merge_pr.py.")
        return
    print(report())


if __name__ == "__main__":
    main()
