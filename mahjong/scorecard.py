"""The scorecard (docs/scorecard.md, ROADMAP 4): clean attempts per policy, and what a paired session says.

A clean attempt picked the tile up, harmed nothing and needed no one. "Picked" alone hid the damage: on
2026-10-30 (20261030_142320_bench) v6 picked 27/28 in a session with 14 confirmed knocks.
"""
import json
from collections import defaultdict
from math import comb

from scipy.stats import beta

from . import interventions
from .session import read_events

# What makes an attempt not clean, when attributed to it. Left out on purpose: `straightened` (a person fixing a knock
# that already counted), grip_low / grip_high (precarious, but the attempt stands), and rescan, interrupted, crashed and
# killed, which derive() pins on the most recent attempt whatever caused them.
NOT_CLEAN = {"knocked", "collided", "pushed_target", "aborted", "touched_tile", "helped_arm", "caught_drop", "other",
             "unspecified", "discard_missed"}
MIN_PAIRS = 20          # fewer can't show a difference worth acting on (ROADMAP 4's power figures)
ALPHA = 0.1             # exact McNemar: 5-0 pairs is p 0.06, 6-1 is 0.125, 7-1 is 0.07
SPOT_WORSE_BY = 2       # a candidate this many clean attempts worse at one spot gets more attempts there


def exact_ci(k, n, confidence=0.95):
    """Exact (Clopper-Pearson) range for a true rate, given k of n. None with no attempts."""
    if not n:
        return None
    tail = (1 - confidence) / 2
    low = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return low, high


def mcnemar_p(wins, losses):
    """Exact two-sided McNemar: how often a split this lopsided happens by chance among the pairs that differ."""
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n)


def clean_flags(events, found=None):
    """attempt_dir -> is it clean, for every attempt in a session's events. found: derive(events), if already done
    (it takes seconds on a real session: on_mat runs the IK per tile per scan)."""
    found = interventions.derive(events) if found is None else found
    harmed = {i["attempt_dir"] for i in found if i["kind"] in NOT_CLEAN}
    return {e["attempt_dir"]: bool(e.get("success")) and e["attempt_dir"] not in harmed
            for e in events if e["type"] == "attempt"}


def spot_of(attempt):
    return attempt.get("spot") or attempt.get("tile")


def policy_rows(events, found=None):
    """policy -> attempts, picked, clean, its range, person steps (rescues, resets and questions a person answered)
    per 10 attempts, and the ACT seconds, in the session's order."""
    found = interventions.derive(events) if found is None else found
    clean = clean_flags(events, found)
    by_dir = defaultdict(int)
    for i in found:
        if i["class"] in (interventions.RESCUE, interventions.RESET, interventions.ASK) or i["source"] == "confirmed":
            by_dir[i["attempt_dir"]] += 1
    rows = {}
    for e in (e for e in events if e["type"] == "attempt"):
        row = rows.setdefault(e.get("policy"), {"attempts": 0, "picked": 0, "clean": 0, "steps": 0, "seconds": []})
        row["attempts"] += 1
        row["picked"] += bool(e.get("success"))
        row["clean"] += clean[e["attempt_dir"]]
        row["steps"] += by_dir[e["attempt_dir"]]
        row["seconds"].append(e.get("duration_s", 0.0))
    for row in rows.values():
        row["clean_ci"] = exact_ci(row["clean"], row["attempts"])
        row["steps_per_10"] = 10 * row["steps"] / row["attempts"]
    return rows


def pairs(events, baseline, candidate, found=None):
    """Attempts paired by spot and order: the nth at a spot by one policy with the nth there by the other. So
    --policy a,b's rounds pair up, and an attempt with no partner (a session stopped mid-round) is left out."""
    attempts = defaultdict(lambda: defaultdict(list))
    clean = clean_flags(events, found)
    for e in (e for e in events if e["type"] == "attempt"):
        attempts[spot_of(e)][e.get("policy")].append(clean[e["attempt_dir"]])
    return [(spot, a, b) for spot, by_policy in attempts.items()
            for a, b in zip(by_policy.get(baseline, []), by_policy.get(candidate, []))]


def verdict(events, baseline, candidate, min_pairs=MIN_PAIRS, found=None):
    """What a paired session says about a candidate: "no verdict", "revert", "extend" or "ship", and why.
    Only clean attempts decide here. A tie isn't a ship: a person can still ship one for a benefit they name
    (speed, a surface it alone works on), in the --evidence of status.py --set-default."""
    paired = pairs(events, baseline, candidate, found)
    wins = sum(b and not a for _, a, b in paired)
    losses = sum(a and not b for _, a, b in paired)
    out = {"pairs": len(paired), "wins": wins, "losses": losses, "p": mcnemar_p(wins, losses), "worse_spots": []}
    surfaces = {e.get("conditions", {}).get("surface") for e in events if e["type"] == "session_start"}
    if len(paired) < min_pairs:
        return {**out, "verdict": "no verdict", "why": f"{len(paired)} pairs, {min_pairs} needed"}
    if out["p"] <= ALPHA and losses > wins:
        return {**out, "verdict": "revert", "why": f"{losses}-{wins} clean pairs against it, p {out['p']:.2f}"}
    if out["p"] <= ALPHA:
        return {**out, "verdict": "ship", "why": f"{wins}-{losses} clean pairs for it, p {out['p']:.2f}"
                + (f" (only on {surfaces.pop()})" if len(surfaces) == 1 and None not in surfaces else "")}
    by_spot = defaultdict(int)
    for spot, a, b in paired:
        by_spot[spot] += int(a) - int(b)
    out["worse_spots"] = sorted(s for s, d in by_spot.items() if d >= SPOT_WORSE_BY)
    return {**out, "verdict": "extend", "why": f"{wins}-{losses} clean pairs, p {out['p']:.2f}: no difference shown"
            + (f"; worse at {', '.join(out['worse_spots'])}, run more there" if out["worse_spots"] else "")}


def session_clean(session_dir):
    """policy -> clean attempts in a session folder, saved beside its events: status.py reads every session, and
    deriving them all takes minutes. Redone if the events grow (a keys_void) or NOT_CLEAN changes."""
    events_file, saved = session_dir / "events.jsonl", session_dir / "clean_counts.json"
    key = {"size": events_file.stat().st_size, "not_clean": sorted(NOT_CLEAN)}
    if saved.exists():
        old = json.loads(saved.read_text())
        if old["key"] == key:
            return old["clean"]
    clean = {policy: row["clean"] for policy, row in policy_rows(read_events(session_dir)).items() if policy}
    saved.write_text(json.dumps({"key": key, "clean": clean}))
    return clean


def report(events, found=None):
    """The scorecard for one session, as text lines. A verdict only if exactly two policies ran, the second the
    candidate (bench.py --policy baseline,candidate)."""
    found = interventions.derive(events) if found is None else found
    rows = policy_rows(events, found)
    voided = sum(e["type"] == "attempt_void" for e in events)
    skipped = sum(e["type"] == "attempt_skipped" for e in events)
    lines = ["scorecard (docs/scorecard.md)",
             f"  {'policy':<20} {'clean':>7} {'95% range':>11} {'picked':>7} {'person steps/10':>16} {'median s':>9}"]
    for policy, r in rows.items():
        low, high = r["clean_ci"]
        seconds = sorted(r["seconds"])[len(r["seconds"]) // 2]
        lines.append(f"  {policy or '?':<20} {r['clean']:>3}/{r['attempts']:<3} {low:>5.0%}-{high:<5.0%} "
                     f"{r['picked']:>3}/{r['attempts']:<3} {r['steps_per_10']:>16.1f} {seconds:>9.1f}")
    lines.append("  clean: picked up, nothing knocked or bumped, no one helped; steps: rescues, resets, answered questions")
    if voided or skipped:
        lines.append(f"  check: {voided} attempt(s) voided, {skipped} skipped; these aren't in the counts above")
    policies = [p for p in rows if p]
    start = next((e for e in events if e["type"] == "bench_setup"), {})
    order = [p["id"] for p in start.get("policies", [])] or policies
    if len(order) == 2 and set(order) == set(policies):
        v = verdict(events, order[0], order[1], found=found)
        lines.append(f"  {order[1]} against {order[0]}: {v['verdict'].upper()} ({v['why']})")
    else:
        lines.append("  no verdict: one session must pair exactly two policies (bench.py --policy baseline,candidate)")
    return lines
