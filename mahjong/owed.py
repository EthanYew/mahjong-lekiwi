"""What's owed and overdue: decisions and loose ends nothing else reminds anyone of, one line each.

The 2026-11-13 retrospective (docs/retrospectives/2026-11-13_time_and_automation_review.md) found the advice about the robot was
acted on and the advice about the workflow wasn't: v10 became the default with its scorecard checks "still owed" in its own note,
ROADMAP's status table went stale, 14 merged branches stayed open. Each was written down once and shown again by nothing. These
lines are printed by status.py and at the start of every hardware session, beside checkout.warnings().
"""
import json
import re
from pathlib import Path

from . import checkout, config, scorecard
from .session import is_sim, read_events


def _counted_since_default(sessions_dir, reg, quick):
    """(default, since, standard, [summary of each counted bench session since it became the default]), or None in quick mode
    when one isn't summarised yet: nothing is said rather than guessed (status.py summarises them)."""
    default, note = reg["default"], reg.get("default_note", {})
    since = note.get("since", "")
    cutoff = note.get("at") or since.replace("-", "")       # "at" (YYYYMMDD_HHMMSS) since 2026-11-13; older notes have the day only
    standard = scorecard.standard_setup()
    summaries = []
    for events_file in sorted((Path(sessions_dir or config.SESSIONS)).glob("*_bench/events.jsonl")):
        folder = events_file.parent
        if folder.name[:len(cutoff)] < cutoff or is_sim(folder):
            continue
        if scorecard.off_standard(read_events(folder), standard, folder=folder):
            continue
        summary = scorecard.cached_summary(folder) if quick else scorecard.session_summary(folder)
        if summary is None:
            return None
        summaries.append(summary)
    return default, since, standard, summaries


_UNSET = object()


def default_checks(sessions_dir=None, registry=None, quick=False, found=_UNSET):
    """A line if the default policy has passed neither scorecard check (docs/scorecard.md) on a counted setup since it became
    the default, else None. Simulated, voided and uncounted sessions don't count. quick (a hardware session's start): only
    sessions already summarised count. found: _counted_since_default's result, when the caller already has it."""
    if found is _UNSET:
        found = _counted_since_default(sessions_dir, registry or json.loads(config.POLICIES.read_text()), quick)
    if found is None:
        return None
    default, since, standard, summaries = found
    if any((mine := s["gates"].get(default)) and (mine["play_ready"] or mine["ahead_of"]) for s in summaries):
        return None
    return (f"the default, {default} (since {since or '?'}), has passed neither scorecard check on a counted setup since: "
            f"pair it with the policy it replaced, bench.py --policy <that>,{default} --layout {standard['layout']}, face up or "
            "with --face-down (docs/scorecard.md)")


def face_up_round(sessions_dir=None, registry=None, quick=False, found=_UNSET):
    """A line if the default's latest face-up round on the standard setup since it became the default is missing or went
    badly. A round is a full iteration round against another policy (in either order on the bench) or a play-ready run (its
    PLAY_READY_ATTEMPTS). It's bad if another policy left the default 3 pairs behind (Drop), or if it was a play-ready run
    alone that failed; a full round that didn't Drop is good even short of play-ready. The latest decides, so a later bad
    round brings the line back. A face-down gain mustn't hide a face-up loss (docs/scorecard.md); v10, v11 and v13 were all
    chosen face down, and only v10 had a valid face-up run (2026-11-17)."""
    if found is _UNSET:
        found = _counted_since_default(sessions_dir, registry or json.loads(config.POLICIES.read_text()), quick)
    if found is None:
        return None
    default, since, standard, summaries = found
    latest = None                                                   # None: no face-up round; else whether the latest was good
    for s in summaries:                                             # in session order
        mine = s["gates"].get(default)
        if not mine or s.get("face_down"):
            continue
        if mine["paired_with"]:
            latest = not mine["dropped_against"]
        elif mine["attempts"] >= scorecard.PLAY_READY_ATTEMPTS:
            latest = mine["play_ready"]
    if latest:
        return None
    what = "had no face-up round" if latest is None else "fallen behind in its latest face-up round"
    return (f"the default, {default} (since {since or '?'}), has {what} on the standard setup since: "
            f"bench.py --policy <the face-up best>,{default} --layout {standard['layout']} --trials 1, face up (docs/scorecard.md)")


def _named(policy, text):
    short = re.search(r"_(v\d+\w*)$", policy)
    return bool(re.search(re.escape(policy) + r"(?!\w)", text) or (short and re.search(rf"\b{short.group(1)}\b", text)))


def roadmap_lines(repo=None, registry=None):
    """ROADMAP.md's status behind the registry: a policy registered after the default that it never names, or a table dated
    before the default changed."""
    repo = Path(repo or config.REPO)
    reg = registry or json.loads(config.POLICIES.read_text())
    text = (repo / "ROADMAP.md").read_text()
    ids = list(reg["policies"])
    newer = ids[ids.index(reg["default"]) + 1:] if reg["default"] in ids else []
    out = [f"registered but not in ROADMAP.md: {', '.join(p for p in newer if not _named(p, text))} (a row says what it's for)"] \
        if any(not _named(p, text) for p in newer) else []
    dated = re.search(r"\| Item \| State \((\d{4}-\d{2}-\d{2})\)", text)
    since = reg.get("default_note", {}).get("since", "")
    if dated and since and dated.group(1) < since:
        out.append(f"ROADMAP.md's status table is dated {dated.group(1)}, and the default changed on {since}: bring its rows and date up to date")
    return out


def unconfirmed(repo=None):
    """The invariants still marked unconfirmed on the arm, in CLAUDE.md and .claude/rules/."""
    repo = Path(repo or config.REPO)
    files = [repo / "CLAUDE.md", *sorted((repo / ".claude" / "rules").glob("*.md"))]
    return sum(f.read_text().count("Unconfirmed") for f in files if f.exists())


def items(repo=None, sessions_dir=None, quick=False):
    """Every owed line, most consequential first. Each part fails quietly: a reminder must never stop a hardware session.
    quick: nothing that derives a session (a hardware session's start, with the arm connected)."""
    out = []
    def default_lines():        # the sessions are read and summarised once, for both
        found = _counted_since_default(sessions_dir, json.loads(config.POLICIES.read_text()), quick)
        return [default_checks(found=found), face_up_round(found=found)]
    for part in (default_lines, lambda: roadmap_lines(repo),
                 lambda: [f"{n} merged branch(es) not closed: python scripts/merge_pr.py --close"]
                 if (n := len(checkout.leftovers(repo))) else [],
                 lambda: [] if quick else _held_prs(),
                 lambda: [f"samples not saved to main: {', '.join(files)} (python scripts/save_samples.py)"]
                 if not checkout.is_worktree(repo) and (files := checkout.unsaved_samples(repo)) else [],
                 lambda: [f"{n} invariants still say Unconfirmed: /review-bench lists the ones a session settles"]
                 if (n := unconfirmed(repo)) else []):
        try:
            out += [line for line in part() if line]
        except Exception as e:                           # a reminder, not a check: say so and go on
            out.append(f"(couldn't check one owed item: {type(e).__name__}: {e})")
    return out


def _held_prs():
    """PRs the merge queue left for the owner (merge_pr.py --auto labels them needs-owner). Asks GitHub, so status.py only."""
    import subprocess
    out = subprocess.run(["gh", "pr", "list", "--label", "needs-owner", "--json", "number,title"], capture_output=True, text=True,
                         cwd=config.REPO, timeout=20)
    held = json.loads(out.stdout or "[]") if out.returncode == 0 else []
    return [f"{len(held)} PR(s) held for you, decisions the queue won't merge: "
            + ", ".join(f"#{p['number']} {p['title'][:50]}" for p in held[:5]) + " (python scripts/merge_pr.py N)"] if held else []
