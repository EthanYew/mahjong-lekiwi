"""Is the post-merge reviewer any good? Run it on past merges whose real bugs are known, and count what it finds.

    python scripts/review_backtest.py run evening-11-04             # one run on one case (data/review_backtest/cases.json)
    python scripts/review_backtest.py score                         # score the runs that aren't scored yet
    python scripts/review_backtest.py table                         # every scored run

A run gets a scratch copy of the repo as it stood then: history up to that commit and nothing after, session logs (no images or
video) only from before the original review, no GitHub remote, no old transcripts, and the arm scripts denied outright (--sim
excepted). The reviewer and its brief are added from today, so it's the current method on the old code, and that code's own CLAUDE.md.

Whoever edits the reviewer has seen these answers, so a good score on a case the method was written around proves less than one on
a case it wasn't: cases.json says which. A new real bug found by a later review goes in as a new case, not a change to an old one.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import checkout, config

import review_brief

HOME = config.DATA / "review_backtest"
CASES = HOME / "cases.json"
RESULTS = HOME / "results.jsonl"
SESSION_MEDIA = ("*.mp4", "*.avi", "*.jpg", "*.jpeg", "*.png", "*.npy", "*.npz")     # 5GB of them; a reviewer reads the logs
GATE = ".claude/hooks/arm_run_gate.py"
# The gate asks a person, and a backtest has none: turn its "ask" into a refusal, so nothing can start the arm
DENY_GATE = f"python3 -I {GATE} | sed 's/\"permissionDecision\": \"ask\"/\"permissionDecision\": \"deny\"/'"
NOT_ALLOWED = ["Bash(gh *)", "Bash(*/.claude/projects*)", "Read(~/.claude/projects/**)", "Read(~/Projects/mahjong-lekiwi/**)",
               "Bash(*Projects/mahjong-lekiwi*)", "Agent", "WebFetch", "WebSearch"]
FINDINGS_SCHEMA = {"type": "object", "required": ["findings"], "properties": {
    "summary": {"type": "string"},
    "decisions": {"type": "array", "items": {"type": "string"}},
    "findings": {"type": "array", "items": {"type": "object", "required": ["file", "summary", "scenario", "evidence"], "properties": {
        "file": {"type": "string"}, "line": {"type": "integer"}, "kind": {"type": "string"}, "severity": {"type": "string"},
        "summary": {"type": "string"}, "scenario": {"type": "string"}, "evidence": {"type": "string"}, "source": {"type": "string"}}}}}}
SCORE_SCHEMA = {"type": "object", "required": ["known", "extras"], "properties": {
    "known": {"type": "array", "items": {"type": "object", "required": ["id", "grade", "finding", "why"], "properties": {
        "id": {"type": "string"}, "grade": {"enum": ["hit", "partial", "miss"]}, "finding": {"type": "integer"}, "why": {"type": "string"}}}},
    "extras": {"type": "array", "items": {"type": "integer"}}}}


def load_cases():
    return json.loads(CASES.read_text())


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=checkout.ENV).stdout.strip()


def scratch_copy(head, target, source=None, sessions_before=None, sessions_from=None):
    """target becomes a repo with head's history only and no remote, plus links to the sessions from before that moment."""
    source = Path(source or config.REPO)
    target = Path(target)
    target.mkdir(parents=True)
    git("init", "-q", cwd=target)
    full = git("rev-parse", head, cwd=source)
    git("fetch", "-q", str(source), full, cwd=target)
    git("checkout", "-q", "-B", "main", full, cwd=target)
    git("config", "core.hooksPath", "/dev/null", cwd=target)
    (target / ".git" / "FETCH_HEAD").unlink(missing_ok=True)
    (target / "sessions").mkdir(exist_ok=True)
    if sessions_before and sessions_from and Path(sessions_from).exists():
        for folder in sorted(Path(sessions_from).iterdir()):
            if folder.is_dir() and folder.name < sessions_before:
                shutil.copytree(folder, target / "sessions" / folder.name, ignore=shutil.ignore_patterns(*SESSION_MEDIA))
    return target


def add_reviewer(target):
    """Today's reviewer in the old code, and the arm gate whether or not the old code had one."""
    target = Path(target)
    repo = config.REPO
    for relative in (".claude/agents/merge-reviewer.md", GATE, "ruff.toml"):
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / relative, target / relative)
    hook = {"type": "command", "command": DENY_GATE, "timeout": 10}
    settings = {"permissions": {"deny": NOT_ALLOWED}, "hooks": {"PreToolUse": [{"matcher": "Bash|Monitor", "hooks": [hook]}]}}
    (target / ".claude" / "settings.json").write_text(json.dumps(settings, indent=2))


def prompt_for(case, brief_path):
    return (f"Review the range {case['base']}..{case['head']}. Findings only: change nothing in the code. The brief is at {brief_path}. "
            f"Treat this checkout as all there is: there's no GitHub here, and don't look outside this folder except for the "
            f"read-only links under sessions/. Return the findings as the structured output.")


def run_case(name, runs_dir, source=None, sessions_from=None, budget=12, subscription=False):
    case = load_cases()["cases"][name]
    sessions_from = sessions_from or _main_checkout_sessions()
    work = Path(tempfile.mkdtemp(prefix=f"bt-{name}-", dir=runs_dir))
    target = scratch_copy(case["head"], work / "repo", source, case.get("sessions_before"), sessions_from)
    add_reviewer(target)
    bodies = {str(n): review_brief.pr_body(n) for n in case.get("prs", [])}     # fetched here: the scratch copy has no GitHub
    started = time.time()
    text = review_brief.brief(case["base"], case["head"], target, bodies, run=True)
    brief_path = target / ".review_brief.md"
    brief_path.write_text(text)
    print(f"brief: {len(text)} characters, {time.time() - started:.0f}s", flush=True)
    command = ["claude", "-p", prompt_for(case, ".review_brief.md"), "--agent", "merge-reviewer", "--model", "opus", "--effort", "high",
               "--permission-mode", "auto", "--strict-mcp-config", "--output-format", "json", "--json-schema", json.dumps(FINDINGS_SCHEMA),
               "--max-budget-usd", str(budget)]
    started = time.time()
    done = subprocess.run(command, cwd=target, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=account_env(subscription))
    raw = work / "output.json"
    raw.write_text(done.stdout or done.stderr)
    if done.returncode:
        raise SystemExit(f"the reviewer stopped with exit {done.returncode}: {stopped_because(done.stdout or done.stderr)} (see {raw})")
    return save_run(name, json.loads(done.stdout), time.time() - started, str(work))


def account_env(subscription=False):
    """Which account pays. An ANTHROPIC_API_KEY in the shell wins over a claude.ai login, so a run bills it unless told otherwise:
    on 2026-11-05 three runs used up a key's credit five minutes in, while the session that started them ran on regardless."""
    if not subscription:
        return dict(os.environ)
    return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}


def stopped_because(output):
    """The reason in a failed run's output, so "Credit balance is too low" doesn't read as a bad review."""
    try:
        return str(json.loads(output).get("result"))[:200]
    except ValueError:
        return output[:200]


def save_run(name, output, seconds, work):
    usage = output.get("usage") or {}
    row = {"case": name, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "minutes": round(seconds / 60, 1), "work": work,
           "calls": output.get("num_turns"), "cost_usd": output.get("total_cost_usd"),
           "tokens": {k: usage.get(k) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")},
           "report": output.get("structured_output") or {"findings": [], "summary": output.get("result", "")}}
    append_result(row)
    print(f"{name}: {len(row['report'].get('findings', []))} findings, {row['minutes']} min, ${row['cost_usd']}")
    return row


def append_result(row):
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


def read_results():
    return [json.loads(line) for line in RESULTS.read_text().splitlines()] if RESULTS.exists() else []


def _main_checkout_sessions():
    """Sessions live in the main checkout, wherever this runs from."""
    return (config.REPO / git("rev-parse", "--git-common-dir", cwd=config.REPO)).resolve().parent / "sessions"


def score_prompt(case, report, rules):
    return ("You are grading a code review against a list of real bugs known from a later fix. Grade strictly, by this rule:\n" + rules +
            "\n\nKNOWN ITEMS:\n" + json.dumps(case["known"], indent=1) + "\n\nFINDINGS, numbered from 0:\n" +
            json.dumps(list(enumerate(report.get("findings", []))), indent=1) +
            "\n\nFor every known item give its grade and the number of the best finding (-1 for a miss) and a short reason. "
            "Under extras, list the numbers of findings that match no known item. Return the structured output.")


def score_run(row, cases):
    """A separate model with no tools and no sight of how the reviewer was written grades the report."""
    case = cases["cases"][row["case"]]
    done = subprocess.run(["claude", "-p", score_prompt(case, row["report"], cases["scoring"]), "--model", "sonnet", "--tools", "",
                           "--strict-mcp-config", "--output-format", "json", "--json-schema", json.dumps(SCORE_SCHEMA)],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tempfile.gettempdir())
    if done.returncode:
        raise SystemExit(f"the scorer stopped with exit {done.returncode}: {done.stdout[:400]}")
    return json.loads(done.stdout)["structured_output"]


def tally(known):
    """Hits, partials and the size of the list a score is out of."""
    grades = [k["grade"] for k in known]
    return {"hit": grades.count("hit"), "partial": grades.count("partial"), "of": len(grades)}


def score_all():
    cases, rows = load_cases(), read_results()
    for row in rows:
        if "score" not in row:
            row["score"] = score_run(row, cases)
            row["tally"] = tally(row["score"]["known"])
    RESULTS.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print_table(rows)


def print_table(rows):
    for row in rows:
        t = row.get("tally")
        found = f"{t['hit']} hit, {t['partial']} partial, of {t['of']}" if t else "not scored"
        extras = len(row["score"]["extras"]) if "score" in row else "?"
        print(f"{row['case']:<20} {row['at']}  {found:<28} extras {extras}  findings {len(row['report'].get('findings', []))}"
              f"  {row['minutes']} min  ${row.get('cost_usd')}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="review one case in a scratch copy")
    run.add_argument("case")
    run.add_argument("--runs-dir", default=tempfile.gettempdir(), help="where the scratch copies go")
    run.add_argument("--budget", type=float, default=12, help="stop the run past this many US dollars (a stopped run scores nothing)")
    run.add_argument("--subscription", action="store_true", help="use the claude.ai login, not an ANTHROPIC_API_KEY set in the shell")
    sub.add_parser("score", help="grade the runs not graded yet")
    sub.add_parser("table", help="every run so far")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        run_case(args.case, args.runs_dir, budget=args.budget, subscription=args.subscription)
    elif args.command == "score":
        score_all()
    else:
        print_table(read_results())


if __name__ == "__main__":
    main()
