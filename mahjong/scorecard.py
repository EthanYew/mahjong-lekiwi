"""The scorecard (docs/scorecard.md, ROADMAP 4): clean attempts per policy, and what a paired session says.

A clean attempt picked the tile up, harmed nothing and needed no one. "Picked" alone hid the damage: on
2026-10-30 (20261030_142320_bench) v6 picked 27/28 in a session with 14 confirmed knocks.
"""
import json
from collections import defaultdict
from math import comb

from scipy.stats import beta

from . import config, interventions
from .session import read_events, void_reason

# What makes an attempt not clean, when attributed to it. Left out on purpose: `straightened` (a person fixing a knock
# that already counted), grip_low / grip_high / grip_shallow (precarious, but the attempt stands), and rescan, interrupted, crashed and
# killed, which derive() pins on the most recent attempt whatever caused them.
NOT_CLEAN = {"knocked", "collided", "pushed_target", "aborted", "touched_tile", "helped_arm", "caught_drop", "other",
             "unspecified", "discard_missed", "takeover"}
MIN_PAIRS = 20          # inherited from the 2026-10-23 draft's 4 spots x 5, not calculated (docs/scorecard.md, which has the odds)
ALPHA = 0.1             # exact McNemar: 5-0 pairs is p 0.06, 6-1 is 0.125, 7-1 is 0.07
SPOT_WORSE_BY = 2       # a candidate this many clean attempts worse at one spot gets more attempts there
ITERATION_PAIRS = 14    # the iteration check: one round of even-hand14, a second if the first is unclear
ITERATION_DROP_BEHIND = 3   # this many clean pairs behind the default: drop the candidate
PLAY_READY_ATTEMPTS = 28    # the play-ready check: two rounds of even-hand14 in one session...
PLAY_READY_MISSES = 2       # ...at most this many not clean (26/28)...
PLAY_READY_SPOT_MISSES = 2  # ...and no spot not clean this many times
HOVER_GRIPPER_SPREAD = 1.0  # openings further apart than this (%) aren't the same hover: .claude/rules/policy_data.md, "Datasets store..."
STANDARD_SETUP = config.DATA / "standard_setup.json"
SEES_BACKS = "found by the scans' back boxes"          # face_down_aim: the only face-down sessions that count


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


def standard_setup(path=None):
    """What a counted result runs on: {"surface", "layout", ...} (data/standard_setup.json)."""
    return json.loads((path or STANDARD_SETUP).read_text())


def off_standard(events, standard=None, folder=None):
    """Why a session's attempts don't count towards the gates: a list of reasons, empty if they do. Face down counts as its
    own setup when the standard names one and the scans found the backs (docs/scorecard.md). A setup check isn't logged on
    the session, so passing it stays a person's step. folder: also checks data/void_sessions.json."""
    standard = standard or standard_setup()
    start = next((e for e in events if e["type"] == "session_start"), {})
    setup = next((e for e in events if e["type"] == "bench_setup"), {})
    surface, name = start.get("conditions", {}).get("surface"), setup.get("layout")
    void = void_reason(folder) if folder is not None else None
    sees_backs = next((e for e in events if e["type"] == "face_down_setup"), {}).get("sees_backs", False)
    face_down_counts = bool(standard.get("face_down")) and (sees_backs or not standard["face_down"].get("sees_backs", True))
    return ([f"voided: {void}"] if void else []) \
        + (["a simulated run (--sim)"] if start.get("sim") else []) \
        + ([f"surface {surface or 'not recorded'}, not {standard['surface']}"] if surface != standard["surface"] else []) \
        + ([f"layout {name or 'none'}, not {standard['layout']}"] if name != standard["layout"] else []) \
        + (["carries between two spots, not a walk of the layout"] if setup.get("carry") else []) \
        + ([f"hover {setup['hover_lower_mm']:g}mm lower than the policy's own (--hover-lower-mm): an experiment"]
           if setup.get("hover_lower_mm") else []) \
        + ([f"face-down tiles, {face_down_aim(events)}"] if setup.get("face_down") and not face_down_counts else [])


def face_down_aim(events):
    """How a bench --face-down session found its tiles: by the scan's back boxes (weights with the class), or by saved
    position (older weights, and sessions from before the class: no face_down_setup event)."""
    setup = next((e for e in events if e["type"] == "face_down_setup"), {})
    return SEES_BACKS if setup.get("sees_backs") else "picked by saved position"


def setup_label(events):
    """Which counted setup a session ran: "face down" or "face up". Each is compared only with itself."""
    return "face down" if next((e for e in events if e["type"] == "bench_setup"), {}).get("face_down") else "face up"


def iteration_check(events, baseline, candidate, found=None):
    """Is the candidate a better base to keep building on? The clean count only: whether the failure it was meant to
    fix improved, with no new failure type or freeze, is a person's call. Results: "too few", "drop", "ahead",
    "another round" (the next round is its own session: add the two sessions' wins and losses), "keep baseline"."""
    paired = pairs(events, baseline, candidate, found)
    wins = sum(b and not a for _, a, b in paired)
    losses = sum(a and not b for _, a, b in paired)
    if len(paired) < ITERATION_PAIRS:
        result = "too few"
    elif losses - wins >= ITERATION_DROP_BEHIND:
        result = "drop"
    elif wins > losses:
        result = "ahead"
    else:
        result = "keep baseline" if len(paired) >= 2 * ITERATION_PAIRS else "another round"
    return {"pairs": len(paired), "wins": wins, "losses": losses, "result": result}


def play_ready(events, policy, found=None):
    """Is a policy worth spending scored hands on? A fixed bar in one session, no comparison. A pass isn't proof of
    clean hands; those are measured on hands (docs/scorecard.md)."""
    clean = clean_flags(events, found)
    mine = [e for e in events if e["type"] == "attempt" and e.get("policy") == policy]
    misses = defaultdict(int)
    for e in mine:
        misses[spot_of(e)] += not clean[e["attempt_dir"]]
    total = sum(misses.values())
    repeated = sorted(s for s, n in misses.items() if n >= PLAY_READY_SPOT_MISSES)
    return {"attempts": len(mine), "clean": len(mine) - total, "repeated_spots": repeated,
            "ok": len(mine) >= PLAY_READY_ATTEMPTS and total <= PLAY_READY_MISSES and not repeated}


def _pair_outcomes(base, candidate, n):
    """(wins, losses, chance) for n pairs, taking the two attempts in a pair as independent: an approximation, since
    both are at one spot."""
    win, loss = candidate * (1 - base), base * (1 - candidate)
    for w in range(n + 1):
        for l in range(n - w + 1):
            yield w, l, comb(n, w) * comb(n - w, l) * win ** w * loss ** l * (1 - win - loss) ** (n - w - l)


def ship_chance(base, candidate, n):
    """The chance n pairs reach Ship, given true clean rates. What docs/scorecard.md's power table is made of."""
    return sum(c for w, l, c in _pair_outcomes(base, candidate, n) if w > l and mcnemar_p(w, l) <= ALPHA)


def iteration_chance(base, candidate):
    """The chance the iteration check's clean count keeps the candidate, across both its rounds."""
    keep = 0.0
    for w, l, c in _pair_outcomes(base, candidate, ITERATION_PAIRS):
        if w > l:
            keep += c
        elif l - w < ITERATION_DROP_BEHIND:
            keep += c * sum(c2 for w2, l2, c2 in _pair_outcomes(base, candidate, ITERATION_PAIRS)
                            if w + w2 > l + l2)
    return keep


def play_ready_chance(rate):
    """The chance a true clean rate reaches the play-ready count, leaving out the repeated-spot condition."""
    n, most = PLAY_READY_ATTEMPTS, PLAY_READY_MISSES
    return sum(comb(n, k) * (1 - rate) ** k * rate ** (n - k) for k in range(most + 1))


SUMMARY_VERSION = 3       # 2: the gates too (mahjong/owed.py), so a session's interventions are derived once for both;
                          # 3: face up or down, and who each policy had a full round with and dropped against


def _summary_key(session_dir):
    return {"size": (session_dir / "events.jsonl").stat().st_size, "not_clean": sorted(NOT_CLEAN), "v": SUMMARY_VERSION}


def cached_summary(session_dir):
    """session_summary's saved result if it's still good, else None, without deriving anything."""
    saved = session_dir / "clean_counts.json"
    if saved.exists():
        old = json.loads(saved.read_text())
        if old.get("key") == _summary_key(session_dir):
            return old
    return None


def session_summary(session_dir):
    """{"clean": policy -> clean attempts, "face_down": bool, "gates": policy -> {"play_ready", "ahead_of", "paired_with",
    "dropped_against": [policies]}} for a session folder, saved beside its events: status.py reads every session, and deriving
    them all takes minutes. paired_with: a full iteration round (ITERATION_PAIRS) with that policy as baseline. Redone if the
    events grow (a keys_void) or NOT_CLEAN changes."""
    if (old := cached_summary(session_dir)):
        return old
    events = read_events(session_dir)
    found = interventions.derive(events)
    clean = {policy: row["clean"] for policy, row in policy_rows(events, found).items() if policy}
    policies = sorted(clean)
    checks = {(q, p): iteration_check(events, q, p, found) for p in policies for q in policies if q != p}
    gates = {p: {"play_ready": play_ready(events, p, found)["ok"],
                 "ahead_of": [q for q in policies if q != p and checks[(q, p)]["result"] == "ahead"],
                 "paired_with": [q for q in policies if q != p and checks[(q, p)]["pairs"] >= ITERATION_PAIRS],
                 "dropped_against": [q for q in policies if q != p and checks[(q, p)]["result"] == "drop"]}
             for p in policies}
    summary = {"key": _summary_key(session_dir), "clean": clean, "face_down": setup_label(events) == "face down",
               "gates": gates}
    (session_dir / "clean_counts.json").write_text(json.dumps(summary))
    return summary


def session_clean(session_dir):
    """policy -> clean attempts in a session folder (session_summary, saved beside its events)."""
    return session_summary(session_dir)["clean"]


def report(events, found=None, folder=None):
    """The scorecard for one session, as text lines. A verdict only if exactly two policies ran, the second the
    candidate (bench.py --policy baseline,candidate). folder: the session's, so a voided one says so."""
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
    openings = [p["hover"]["gripper"] for p in start.get("policies", []) if p.get("hover")]
    if openings and max(openings) - min(openings) > HOVER_GRIPPER_SPREAD:
        lines.append(f"  check: hover openings differ ({min(openings)}-{max(openings)}%), so this isn't one change")
    return lines + gate_lines(events, order if len(order) == 2 and set(order) == set(policies) else None, policies, found,
                              folder)


def gate_lines(events, pair, policies, found, folder=None):
    """The iteration and play-ready checks, for a session on a counted setup; otherwise why they don't apply."""
    away = off_standard(events, folder=folder)
    if away:
        return [f"  gates don't apply: {'; '.join(away)}"]
    label = setup_label(events)
    lines = [f"  counted setup: {label} (compare only with other {label} sessions)"]
    if pair:
        c = iteration_check(events, pair[0], pair[1], found)
        lines.append(f"  iteration check: {c['result'].upper()} ({c['wins']}-{c['losses']} clean pairs of {c['pairs']}); "
                     "keeping also needs the named failure improved, no new failure type, no freezes"
                     + ("; and a face-up round with no Drop" if label == "face down" else ""))
    for policy in policies:
        r = play_ready(events, policy, found)
        lines.append(f"  play-ready ({label}), {policy}: {'PASS' if r['ok'] else 'not yet'} ({r['clean']}/{r['attempts']} clean"
                     + (f", repeated misses at {', '.join(r['repeated_spots'])}" if r["repeated_spots"] else "")
                     + f"; needs {PLAY_READY_ATTEMPTS - PLAY_READY_MISSES}/{PLAY_READY_ATTEMPTS}); "
                       "also needs a passing setup check and no silent wrong tile name")
    return lines
