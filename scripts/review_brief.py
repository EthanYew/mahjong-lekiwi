"""What a post-merge review starts from, so it spends its attention on judgement rather than on working this out.

    python scripts/review_brief.py                        # the merges since the last review, tests and sim runs included
    python scripts/review_brief.py --range abc123..def456 # a range of your own
    python scripts/review_brief.py --no-run               # skip the tests and sim runs

Prints: each merge with its files and what its PR said it overlaps and couldn't verify, the files several merges changed,
ruff, functions copied between files, the fast tests, and simulated runs of the bench and of play. Only ever --sim.
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import checkout, code_health, config

# What the sim runs need to reach the end with no one at the keyboard (docs/usage.md, "Testing on the arm")
SIM_RUNS = {"bench": ["--sim", "--trials", "3", "--tile", "8D", "--unattended"],
            "play": ["--sim", "--layout", "even-hand13"]}
SIM_TIMEOUT_S = 240
ENTER_PRESSES = "\n" * 500      # the answer a person gives when all looks right (play's hand check needs one)
TESTS_TIMEOUT_S = 400
WANTED_SECTIONS = ("Overlaps", "Unverified on the arm")
PR_NUMBER = re.compile(r"Merge pull request #(\d+)")
LIST_AT_MOST = 80


def sections(body, wanted=WANTED_SECTIONS):
    """The named `## ` sections of a PR description, {name: text}."""
    found = {}
    for chunk in re.split(r"^#+\s+", body or "", flags=re.M)[1:]:
        title, _, text = chunk.partition("\n")
        for name in wanted:
            if title.strip().lower().startswith(name.lower()):
                found[name] = text.strip()
    return found


def merges_in(start, end, repo=None):
    """[{"sha", "subject", "pr", "files"}] for each commit on the main line of start..end: a merge's files are what it brought in."""
    out = []
    for line in checkout.git("log", "--first-parent", "--reverse", "--format=%h %s", f"{start}..{end}", repo=repo).splitlines():
        sha, _, subject = line.partition(" ")
        files = checkout.git("diff", "--name-only", f"{sha}^1", sha, repo=repo).splitlines()
        number = PR_NUMBER.search(subject)
        out.append({"sha": sha, "subject": subject, "pr": int(number.group(1)) if number else None, "files": files})
    return out


def pr_body(number, bodies=None, repo=None):
    """A PR's description: from a file of {number: body} if given (a backtest has no GitHub), else from gh. "" if neither."""
    if bodies is not None:
        return bodies.get(str(number), "")
    result = subprocess.run(["gh", "pr", "view", str(number), "--json", "body", "--jq", ".body"], cwd=repo or config.REPO,
                            capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def run_quietly(command, cwd, timeout, answers=""):
    """(exit code, the last lines of its output). No one is there to answer, so past `answers` a question ends in EOFError."""
    try:
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, input=answers)
    except subprocess.TimeoutExpired:
        return "timed out", f"(no end after {timeout}s: a hang, or waiting for an answer)"
    return done.returncode, "\n".join((done.stdout + done.stderr).strip().splitlines()[-12:])


def checks_section(repo):
    """ruff, copied functions, the fast tests, the sim runs."""
    lines = ["## Checks already run: don't redo these", ""]
    found = code_health.ruff(repo)
    lines += [f"ruff: {len(found)} finding(s)"] + [f"- {f['file']}:{f['line']} {f['code']} {f['message']}" for f in found[:30]]
    copies = code_health.copied_functions(repo)
    lines += [f"\nFunctions copied between files: {len(copies)}"] + [f"- {' = '.join(group)}" for group in copies]
    code, tail = run_quietly([sys.executable, "-m", "pytest", "-m", "not slow", "-q", "-p", "no:cacheprovider", "-n", "8"], repo, TESTS_TIMEOUT_S)
    lines += [f"\nFast tests: exit {code}", "```", tail, "```"]
    for script, options in SIM_RUNS.items():
        if (Path(repo) / "scripts" / f"{script}.py").exists():
            command = [sys.executable, f"scripts/{script}.py", *options]
            code, tail = run_quietly(command, repo, SIM_TIMEOUT_S, answers=ENTER_PRESSES)
            lines += [f"\n`{' '.join(command[1:])}`: exit {code}", "```", tail, "```"]
    return lines


def brief(start, end, repo=None, bodies=None, run=True):
    repo = Path(repo or config.REPO)
    merges = merges_in(start, end, repo)
    lines = [f"# Review brief: {start}..{end}", "",
             f"{len(merges)} commit(s) on the main line, {sum(m['pr'] is not None for m in merges)} of them merged PRs.", ""]
    lines += ["## What merged", ""]
    for merge in merges[:LIST_AT_MOST]:
        code = [f for f in merge["files"] if not f.startswith(("data/", "sessions/"))]
        lines.append(f"### {merge['sha']} {merge['subject']}\n{len(code)} file(s): {', '.join(code[:25])}{' ...' if len(code) > 25 else ''}")
        if merge["pr"]:
            for name, text in sections(pr_body(merge["pr"], bodies, repo)).items():
                lines.append(f"**{name}:** {text}")
        lines.append("")
    if len(merges) > LIST_AT_MOST:
        lines.append(f"... and {len(merges) - LIST_AT_MOST} more: git log --first-parent {start}..{end}\n")
    counts = Counter(f for m in merges for f in m["files"] if f.endswith(".py") or f == "ruff.toml")
    shared = [(f, n) for f, n in counts.most_common() if n > 1]
    lines += ["## Code files several of these changed: look here for assumptions that clash", ""]
    lines += [f"- {f}: {n}" for f, n in shared] or ["- none"]
    lines.append("")
    if run:
        lines += checks_section(repo)
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--range", dest="span", metavar="A..B", help="the commits to review (default: since the last review, to origin/main)")
    p.add_argument("--repo", help="review a checkout other than this one (a backtest's scratch copy)")
    p.add_argument("--pr-bodies", help="a JSON file of {number: description}, instead of asking GitHub")
    p.add_argument("--no-run", action="store_true", help="skip the tests and sim runs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.span:
        start, _, end = args.span.partition("..")
    else:
        start, end = checkout.last_reviewed(args.repo), checkout.BASE
        if not start:
            raise SystemExit(f"no earlier review in {checkout.REVIEW_LOG}: give --range A..B")
    bodies = json.loads(Path(args.pr_bodies).read_text()) if args.pr_bodies else None
    print(brief(start, end, args.repo, bodies, run=not args.no_run))


if __name__ == "__main__":
    main()
