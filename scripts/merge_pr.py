"""Approve and merge a branch's PR, then close the branch (CLAUDE.md, "Worktrees and merging").

    python scripts/merge_pr.py               # open PRs, and merged branches/worktrees not closed yet (changes nothing)
    python scripts/merge_pr.py 34            # check PR 34, ask, merge it, pull main, test main, close its branch
    python scripts/merge_pr.py 34 --slow     # ... with the slow (GPU) tests on the merged tree too
    python scripts/merge_pr.py --close       # close every merged branch: worktree, local and GitHub branch (asks)
    python scripts/merge_pr.py --review      # start the post-merge review of what's merged since the last one
    python scripts/merge_pr.py --auto        # the merge queue: every open, ready PR, oldest first, no questions (below)
    python scripts/merge_pr.py --auto 34     # just PR 34 (what /hand-back runs once its own checks pass)

The queue (--auto) asks nothing: every check the person used to judge is a refusal. A PR is merged only if it isn't a decision
(HOLD_FILES, the default policy, installed weights, or labelled `owner`: those get `needs-owner` and wait), its description has
every section filled (and ## Review citing /code-review for REVIEW_FILES), nothing was added on both sides, and the merged tree
passes the fast tests (slow too for SLOW_FILES) and the sim runs of the bench and play. It merges only the head it tested, and
retests if main moved meanwhile. One run at a time (a lock). A PR that fails gets the output as a comment and goes back to draft.
After a merge it doesn't pull this checkout while an arm script is running; when the batch leaves nothing ready, it starts the
post-merge review (on the claude.ai login, never an API key).

Run it from the main checkout. It stops, with nothing changed, if the pull after the merge would fail: main has commits
GitHub doesn't, or the PR changes a data file with uncommitted changes here. Before asking, it shows what you're approving:
1. the PR's description, and which of its sections are missing (what changed / files / overlaps /
   unverified on the arm);
2. what the branch and main both changed since the branch point (same files; the same function, constant or
   option added on both sides: the duplicates that merged without a conflict on 2026-10-23);
3. the fast tests on the merged tree: main + the branch in a scratch worktree. A PR merged in GitHub's page
   never runs the pre-merge-commit hook, and 29 of the first 40 merges were done that way.
Then it merges on GitHub, pulls main here (hardware runs this checkout), runs the fast tests on it, and does
CLAUDE.md's closing step: the branch's worktree's sessions/ moved into this checkout's, the worktree removed,
the local and GitHub branches deleted. Nothing is removed that isn't merged (by content, `git cherry`).
"""
import argparse
import contextlib
import fcntl
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import checkout, config, session

SECTIONS = ["What changed", "Files", "Overlaps", "Unverified on the arm"]
# A PR changing these also needs "## Review": what /code-review found and what was done about each, or why it wasn't run. The bugs
# the 11-04 review found were in play, bench and the arm, and passed 91% line coverage; all 103 PRs to 2026-11-13 merged unreviewed.
REVIEW_FILES = ("scripts/play.py", "scripts/bench.py", "mahjong/arm.py", "mahjong/kinematics.py", "mahjong/policy.py",
                "mahjong/dataset.py", "mahjong/game.py", "mahjong/identity.py")
SLOW_FILES = ("mahjong/policy.py", "mahjong/dataset.py")      # CLAUDE.md: run the slow tests too
# Decisions: what counts as a result or which policy runs. The queue holds a PR changing one for the owner (`needs-owner`).
HOLD_FILES = ("data/eval_split.json", "data/detector_exam.json", "data/name_check_exam.json", "data/standard_setup.json",
              "docs/scorecard.md", "mahjong/scorecard.py")
HOLD_LABELS = ("owner", "needs-owner")
QUEUE_RETRIES = 3               # main moved between the tests and the merge: test the new main + branch again, this many times
DEF = re.compile(r"^\+\s*def (\w+)\(|^\+([A-Z][A-Z0-9_]+) = |add_argument\(\s*\"(--[\w-]+)\"")
REVIEW_PROMPT = ("Review the merges since the last post-merge review. Start with scripts/review_brief.py. Fix what's clearly broken "
                 "in a draft PR, and list what needs a person's judgement.")


def run(cmd, cwd=None, check=True):
    print("$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd or config.REPO, env=checkout.ENV)
    if check and result.returncode:
        raise SystemExit(f"stopped: {cmd[0]} exited {result.returncode}")
    return result.returncode


def gh(*args):
    result = subprocess.run(["gh", *args], cwd=config.REPO, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"gh {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def gh_try(*args):
    """(it worked, stderr): gh where failing is an answer, not an error (a merge refused because the head moved)."""
    result = subprocess.run(["gh", *args], cwd=config.REPO, capture_output=True, text=True)
    return result.returncode == 0, result.stderr.strip()


def confirm(question, ask=input):
    return ask(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def missing_sections(body, touched=()):
    """The PR description's sections that aren't there, or have nothing under them but the template's comment (a markdown heading
    each). "Review" only for a PR that changes REVIEW_FILES."""
    text = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    filled = {m.group(1).strip().lower() for m in re.finditer(r"^#+\s*(.+)$\n((?:(?!^#).*\n?)*)", text, re.M) if m.group(2).strip()}
    wanted = SECTIONS + (["Review"] if needs_review(touched) else [])
    return [s for s in wanted if not any(h.startswith(s.lower()) for h in filled)]


def needs_review(touched):
    return sorted(f for f in touched if f in REVIEW_FILES)


def added_names(diff):
    """Functions, module constants and command-line options a diff adds."""
    return {next(g for g in m.groups() if g) for line in diff.splitlines() if (m := DEF.search(line))}


def overlaps(branch_ref, repo=None, base=checkout.BASE):
    """What base and the branch both changed since they split: (files, names added on both sides)."""
    point = checkout.git("merge-base", base, branch_ref, repo=repo, check=True)
    files = lambda ref: set(checkout.git("diff", "--name-only", point, ref, repo=repo).splitlines())
    names = lambda ref: added_names(checkout.git("diff", point, ref, repo=repo))
    return sorted(files(base) & files(branch_ref)), sorted(names(base) & names(branch_ref))


def test_command(slow=False):
    """pytest with this Python (the lerobot env's), on 8 workers if pytest-xdist is installed."""
    xdist = subprocess.run([sys.executable, "-c", "import xdist"], capture_output=True).returncode == 0
    return [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *([] if slow else ["-m", "not slow"]),
            *(["-n", "8"] if xdist else [])]


def test_merged(branch_ref, slow=False, repo=None, base=checkout.BASE):
    """The tests on base + the branch, merged in a scratch worktree. True if they pass."""
    repo = Path(repo or config.REPO)
    scratch = repo / ".claude" / "worktrees" / "_merge_check"
    if scratch.exists():
        run(["git", "worktree", "remove", "--force", str(scratch)], cwd=repo, check=False)
    run(["git", "worktree", "add", "--detach", str(scratch), base], cwd=repo)
    try:
        if run(["git", "merge", "--no-ff", "--no-commit", branch_ref], cwd=scratch, check=False):
            print(f"\n{branch_ref} doesn't merge cleanly into {base}: merge main into the branch, resolve, push, "
                  "and run this again.")
            return False
        return run(test_command(slow), cwd=scratch, check=False) == 0
    finally:
        run(["git", "worktree", "remove", "--force", str(scratch)], cwd=repo, check=False)


def move_sessions(tree, repo=None):
    """A worktree's sessions/ (gitignored, so removing the worktree would delete them) into the main checkout's.
    Returns the names it couldn't move (a folder of that name is already there)."""
    target = Path(repo or config.REPO) / "sessions"
    left = []
    for folder in sorted((Path(tree) / "sessions").glob("*")):
        if folder.name.startswith("sim_") or session.is_sim(folder):
            continue                    # a simulated run, named like a real one: 40 had reached sessions/ by 2026-11-13
        if (target / folder.name).exists():
            left.append(folder.name)
            continue
        target.mkdir(exist_ok=True)
        shutil.move(str(folder), str(target / folder.name))
        print(f"  moved sessions/{folder.name} here")
    return left


def close_branch(item, repo=None, push=True):
    """CLAUDE.md's closing step for one merged branch: its sessions moved, its worktree removed, its local and
    remote branches deleted. Returns False (and does nothing more) if its worktree has uncommitted code."""
    repo = Path(repo or config.REPO)
    name, tree = item["branch"], item.get("worktree")
    print(f"closing {name}")
    if tree:
        if (files := checkout.changed_code(tree)):
            print(f"  kept: {tree} has uncommitted changes ({', '.join(files[:5])})")
            return False
        if (left := move_sessions(tree, repo)):
            print(f"  kept: sessions already here under the same name: {', '.join(left)}")
            return False
        run(["git", "worktree", "remove", "--force", str(tree)], cwd=repo)
    if item.get("local"):
        run(["git", "branch", "-D", name], cwd=repo)    # merged by content (checkout.leftovers), or abandoned on a yes
    if push and item.get("remote"):
        retarget_stacked(name)
        run(["git", "push", "origin", "--delete", name], cwd=repo, check=False)
    return True


def retarget_stacked(branch):
    """Open PRs built on branch, moved onto main before it's deleted: GitHub closes a PR whose base branch is deleted by hand (#109,
    2026-11-13). REST, because `gh pr edit` fails on this repo (classic projects)."""
    for pr in json.loads(gh("pr", "list", "--base", branch, "--json", "number")):
        gh("api", "-X", "PATCH", f"repos/{{owner}}/{{repo}}/pulls/{pr['number']}", "-f", "base=main")
        print(f"  #{pr['number']} was built on {branch}: moved onto main")


def must_be_main_checkout():
    if checkout.is_worktree():
        raise SystemExit("run this from the main checkout (hardware runs there, and it pulls main)")
    if checkout.git("rev-parse", "--abbrev-ref", "HEAD") != "main":
        raise SystemExit("the main checkout isn't on main: switch back first")
    if (files := checkout.changed_code()):
        raise SystemExit(f"uncommitted code in the main checkout: {', '.join(files)} (commit or move it first)")


def pull_blockers(ref, touched, repo=None):
    """What would stop `git pull --ff-only` after the merge, which by then has already happened on GitHub, leaving main
    merged there but not here, untested and its branch not closed (PR #65, 2026-11-04). One message per cause. Call after a
    fetch. Uncommitted data that the PR doesn't touch doesn't block a pull, and neither do the same lines already staged."""
    problems = []
    if (commits := checkout.unpushed(repo)):
        problems.append(f"this checkout's main has {len(commits)} commit(s) GitHub doesn't: "
                        + "; ".join(commits[:5]) + (" ..." if len(commits) > 5 else "")
                        + "\n  Put them on GitHub first: git pull --rebase --autostash origin main && git push origin main")
    both = [path for path in checkout.changed_data(repo) if path in touched
            and (checkout.git("diff", "--name-only", ref, "--", path, repo=repo)
                 or checkout.git("diff", "--name-only", "--", path, repo=repo))]
    if both:
        problems.append(f"uncommitted changes to files this PR also changes: {', '.join(both)}"
                        "\n  Commit and push them first: python scripts/save_samples.py (for samples), or git add <files> && "
                        "git commit -m '<why>' && git pull --rebase --autostash origin main && git push origin main")
    return problems


def abandoned():
    """Unmerged branches whose PR was closed without merging (e.g. superseded): closing them loses their commits,
    so they're listed apart, with the commit to recover them from."""
    closed = {pr["headRefName"]: pr["number"] for pr in
              json.loads(gh("pr", "list", "--state", "closed", "--limit", "200", "--json", "headRefName,number,mergedAt"))
              if not pr["mergedAt"]}
    trees = {w["branch"]: w["path"] for w in checkout.worktrees() if w["branch"]}
    local, remote = set(checkout.local_branches()), set(checkout.remote_branches())
    merged = {i["branch"] for i in checkout.leftovers()}
    return [{"branch": b, "pr": closed[b], "worktree": trees.get(b), "local": b in local, "remote": b in remote,
             "tip": checkout.git("rev-parse", "--short", b if b in local else f"origin/{b}")}
            for b in sorted((set(trees) | local | remote) & set(closed)) if b not in merged]


def where(item):
    return ", ".join(w for w, on in (("worktree", item["worktree"]), ("local", item["local"]),
                                     ("GitHub", item["remote"])) if on)


def show_status():
    prs = json.loads(gh("pr", "list", "--json", "number,title,isDraft,headRefName"))
    print("Open PRs:" if prs else "No open PRs.")
    for pr in prs:
        print(f"  #{pr['number']:<4} {'draft ' if pr['isDraft'] else ''}{pr['title']}  ({pr['headRefName']})")
    left, gone = checkout.leftovers(), abandoned()
    if left:
        print(f"\nMerged but not closed ({len(left)}): python scripts/merge_pr.py --close")
        for item in left:
            print(f"  {item['branch']:<36} {where(item)}")
    if gone:
        print("\nPR closed without merging (--close asks about these separately):")
        for item in gone:
            print(f"  {item['branch']:<36} {where(item)}  (#{item['pr']}, last commit {item['tip']})")
    print("\nTo approve one: python scripts/merge_pr.py <number>")


def close_all(ask=input):
    left, gone = checkout.leftovers(), abandoned()
    if not left and not gone:
        print("Nothing to close: every merged branch is gone.")
        return
    kept = []
    if left:
        for item in left:
            print(f"  {item['branch']:<36} {where(item)}")
        if confirm(f"Close these {len(left)} merged branches (worktree sessions are moved here first)?", ask):
            kept += [item["branch"] for item in left if not close_branch(item)]
    for item in gone:
        if confirm(f"{item['branch']} ({where(item)}): PR #{item['pr']} was closed unmerged. Delete it? "
                   f"(recover with: git branch {item['branch']} {item['tip']})", ask):
            if not close_branch(item):
                kept.append(item["branch"])
    run(["git", "worktree", "prune"], check=False)
    if kept:
        print(f"Kept (see above): {', '.join(kept)}")


def merge(number, slow=False, ask=input):
    must_be_main_checkout()
    pr = json.loads(gh("pr", "view", str(number), "--json",
                       "number,title,state,isDraft,headRefName,baseRefName,body,url,files,mergeCommit"))
    if pr["state"] == "MERGED":
        return resume(pr, ask)
    if pr["state"] != "OPEN":
        raise SystemExit(f"#{number} is {pr['state'].lower()}")
    if pr.get("baseRefName", "main") != "main":       # stacked on another PR: it would merge into that branch, tested against main
        raise SystemExit(f"#{number} is built on {pr['baseRefName']}, not main: merge that PR first, and GitHub moves this one "
                         "onto main once its branch is closed")
    branch = pr["headRefName"]
    run(["git", "fetch", "--prune", "origin"])
    ref = f"origin/{branch}"
    if (problems := pull_blockers(ref, [f["path"] for f in pr["files"]])):
        raise SystemExit(f"not merging #{number}, nothing changed:\n" + "\n".join(f"- {p}" for p in problems))
    print(f"\n#{number} {pr['title']}\n{pr['url']}\n\n{pr['body'] or '(no description)'}\n")
    if (missing := missing_sections(pr["body"], [f["path"] for f in pr["files"]])):
        print(f"MISSING from the description: {', '.join(missing)}")
    if "Review" in missing and not confirm(
            f"\nNo review recorded, and it changes {', '.join(needs_review(f['path'] for f in pr['files']))}. Ask for /code-review on "
            f"the branch and its findings in ## Review, or merge without one?", ask):
        print("Not merged: no review.")
        return
    files, names = overlaps(ref)
    print(f"\nChanged on both main and the branch since it split: {', '.join(files) or 'nothing'}")
    if names:
        print(f"ADDED ON BOTH SIDES (a duplicate to resolve, conflict or not): {', '.join(names)}")
    touched = [f["path"] for f in pr["files"]]
    slow = slow or any(f in SLOW_FILES for f in touched)
    print(f"\nTesting the merged tree ({'all tests: policy.py/dataset.py changed, or --slow' if slow else 'fast tests'})")
    if not test_merged(ref, slow=slow):
        raise SystemExit(f"not merging #{number}: the merged tree fails (above)")
    if not confirm(f"\nTests pass on main + {branch}. Merge #{number} into main?", ask):
        print("Not merged.")
        return
    if pr["isDraft"]:
        gh("pr", "ready", str(number))
    gh("pr", "merge", str(number), "--merge")
    finish(number, branch, ask)


def finish(number, branch, ask=input, auto=False):
    """Everything after the merge on GitHub: pull main here, test it, close the branch, offer the review. Each step is safe to run
    again, so resume() can start it over. auto: no question, and no pull while an arm script runs from here (it would change the
    code under a live run; the checkout's warnings then say it's behind, and the next run pulls)."""
    if auto and (running := arm_scripts_running()):
        print(f"Not pulling main here yet: {running[0]} is running. The next queue run or merge_pr.py pulls it.")
    else:
        run(["git", "pull", "--ff-only", "origin", "main"])
        if run(test_command(), check=False):
            print("WARNING: the fast tests fail on main after the pull: fix main before the next hardware session")
    run(["git", "fetch", "--prune", "origin"])
    item = next((i for i in checkout.leftovers() if i["branch"] == branch), None)
    if item and close_branch(item):
        print(f"\nMerged #{number} and closed {branch}.")
    else:
        print(f"\nMerged #{number}; {branch} isn't closed (above): python scripts/merge_pr.py --close")
    if not auto:
        offer_review(ask)


def arm_scripts_running():
    """Real (not --sim) runs of an arm script on this machine, as `pgrep -af` lines."""
    gate = _arm_gate()
    out = subprocess.run(["pgrep", "-af", r"scripts/(%s)\.py" % "|".join(gate.ARM_SCRIPTS)], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "--sim" not in line.split() and "--help" not in line.split() and "pgrep" not in line]


def _arm_gate():
    """.claude/hooks/arm_run_gate.py, the one list of the scripts that move the arm."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("arm_run_gate", Path(config.REPO) / ".claude" / "hooks" / "arm_run_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


# --- the merge queue (--auto) ---

@contextlib.contextmanager
def queue_lock(repo=None):
    """One queue run at a time on this machine: they share the scratch worktree and the main checkout."""
    common = checkout.git("rev-parse", "--path-format=absolute", "--git-common-dir", repo=repo)
    with open(Path(common) / "merge_queue.lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another merge queue run is going: waiting for it")
            fcntl.flock(f, fcntl.LOCK_EX)
        yield


def section_text(body, name):
    """What's under a markdown heading starting with name, template comments left out."""
    text = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    found = re.search(r"^#+\s*%s[^\n]*\n((?:(?!^#).*\n?)*)" % re.escape(name), text, re.M | re.I)
    return found.group(1).strip() if found else ""


def held(pr, ref, repo=None, base=checkout.BASE):
    """Why the queue leaves this PR for the owner, or None: a label, a decision file, a change to the default policy, new weights."""
    labels = {label["name"] for label in pr.get("labels") or []}
    if labels & set(HOLD_LABELS):
        return f"labelled {', '.join(sorted(labels & set(HOLD_LABELS)))}"
    touched = [f["path"] for f in pr["files"]]
    if (decisions := [f for f in touched if f in HOLD_FILES or (f.startswith("assets/") and f.endswith(".pt"))]):
        return f"changes {', '.join(decisions)}"
    if "data/policies.json" in touched:
        point = checkout.git("merge-base", base, ref, repo=repo)
        before, after = (json.loads(checkout.git("show", f"{r}:data/policies.json", repo=repo) or "{}") for r in (point, ref))
        if (before.get("default"), before.get("default_note")) != (after.get("default"), after.get("default_note")):
            return "changes the default policy"
    return None


def verify_merged(ref, slow=False, repo=None, base=checkout.BASE):
    """(passed, report): the tests and the sim runs on base + ref, merged in a scratch worktree, with the end of each one's output."""
    import review_brief
    repo = Path(repo or config.REPO)
    scratch = repo / ".claude" / "worktrees" / "_merge_check"
    if scratch.exists():
        run(["git", "worktree", "remove", "--force", str(scratch)], cwd=repo, check=False)
    run(["git", "worktree", "add", "--detach", str(scratch), base], cwd=repo)
    try:
        if run(["git", "merge", "--no-ff", "--no-commit", ref], cwd=scratch, check=False):
            return False, f"{ref} doesn't merge cleanly into {base}: merge main into the branch and resolve it"
        report, passed = [], True
        checks = [("tests", test_command(slow), review_brief.TESTS_TIMEOUT_S * (3 if slow else 1), "")]
        checks += [(f"{script}.py {' '.join(opts)}", [sys.executable, f"scripts/{script}.py", *opts], review_brief.SIM_TIMEOUT_S,
                    review_brief.ENTER_PRESSES) for script, opts in review_brief.SIM_RUNS.items()]
        for name, command, timeout, answers in checks:
            code, tail = review_brief.run_quietly(command, scratch, timeout, answers)
            report.append(f"{name}: exit {code}\n```\n{tail}\n```")
            passed = passed and code == 0
            print(f"  {name}: exit {code}")
        return passed, "\n".join(report)
    finally:
        run(["git", "worktree", "remove", "--force", str(scratch)], cwd=repo, check=False)


def send_back(number, why):
    """A PR the queue won't merge: the reason as a comment, and back to draft, so it isn't picked up again until it's ready."""
    gh("pr", "comment", str(number), "--body", f"**The merge queue didn't merge this** (`merge_pr.py --auto`):\n\n{why}\n\n"
       "Fix it on the branch, mark the PR ready again, and run `python scripts/merge_pr.py --auto <number>` (or `/hand-back`).")
    gh("pr", "ready", str(number), "--undo")
    return f"sent back: {why.splitlines()[0][:160]}"


def auto_one(number, repo=None):
    """One PR through the queue: "merged", "held: ...", "sent back: ..." or "skipped: ..."."""
    pr = json.loads(gh("pr", "view", str(number), "--json",
                       "number,title,state,isDraft,headRefName,baseRefName,body,files,labels,headRefOid"))
    if pr["state"] != "OPEN":
        return f"skipped: it's {pr['state'].lower()}"
    if pr["isDraft"]:
        return "skipped: a draft (mark it ready when it is)"
    if pr["baseRefName"] != "main":
        return f"skipped: built on {pr['baseRefName']}; it moves onto main when that merges"
    branch, ref = pr["headRefName"], f"origin/{pr['headRefName']}"
    run(["git", "fetch", "--prune", "origin"])
    if (why := held(pr, ref, repo)):
        gh("api", "-X", "POST", f"repos/{{owner}}/{{repo}}/issues/{number}/labels", "-f", "labels[]=needs-owner")   # REST: see retarget_stacked
        return f"held: {why}"
    touched = [f["path"] for f in pr["files"]]
    if (missing := missing_sections(pr["body"], touched)):
        return send_back(number, f"The description is missing or has nothing under: {', '.join(missing)}.")
    if needs_review(touched) and "code-review" not in section_text(pr["body"], "Review").lower():
        return send_back(number, f"It changes {', '.join(needs_review(touched))}, and ## Review doesn't say what /code-review found.")
    if (problems := pull_blockers(ref, touched, repo)):
        raise SystemExit("The main checkout can't take a merge (nothing merged):\n" + "\n".join(f"- {p}" for p in problems))
    files, names = overlaps(ref, repo=repo)
    if names:
        return send_back(number, f"Added on both main and the branch: {', '.join(names)}. Reconcile them (one of each) first.")
    slow = any(f in SLOW_FILES for f in touched)
    for _ in range(QUEUE_RETRIES):
        main_before = checkout.git("rev-parse", checkout.BASE, repo=repo)
        head = checkout.git("rev-parse", ref, repo=repo)
        print(f"\n#{number}: testing main {main_before[:7]} + {branch} {head[:7]}" + (" (slow too)" if slow else ""))
        passed, report = verify_merged(ref, slow=slow, repo=repo)
        if not passed:
            return send_back(number, f"On main {main_before[:7]} + {head[:7]}:\n\n{report[-6000:]}")
        run(["git", "fetch", "--prune", "origin"])
        if checkout.git("rev-parse", checkout.BASE, repo=repo) != main_before:
            print("  main moved while testing: again on the new main")
            continue
        merged, err = gh_try("pr", "merge", str(number), "--merge", "--match-head-commit", head)
        if not merged:
            print(f"  GitHub didn't merge it ({err[:200]}): again")
            run(["git", "fetch", "--prune", "origin"])
            continue
        finish(number, branch, auto=True)
        return "merged"
    return send_back(number, f"main kept moving: tested {QUEUE_RETRIES} times and it changed under each.")


def auto(number=None, repo=None):
    """The queue: one PR, or every open ready PR oldest first. Returns {number: result}. Starts the post-merge review once a batch
    that merged anything leaves nothing ready."""
    must_be_main_checkout()
    results = {}
    with queue_lock(repo):
        if number is not None:
            numbers = [number]
        else:
            prs = json.loads(gh("pr", "list", "--json", "number,isDraft,createdAt"))
            numbers = [pr["number"] for pr in sorted(prs, key=lambda p: p["createdAt"]) if not pr["isDraft"]]
        for n in numbers:
            results[n] = auto_one(n, repo)
            print(f"#{n}: {results[n]}")
        ready = [pr for pr in json.loads(gh("pr", "list", "--json", "number,isDraft")) if not pr["isDraft"]]
        if "merged" in results.values() and not ready and checkout.unreviewed_merges():
            print(f"\n{len(checkout.unreviewed_merges())} merged PR(s) since the last review, none ready: starting it in the background")
            start_review()
    return results


def resume(pr, ask=input):
    """A PR merged on GitHub whose steps here didn't all run: on 2026-11-04 #76 merged, then an SSH drop stopped the pull, and running
    this again only said "#76 is merged", leaving main here untested and the branch open. Picks up from the pull, or says it's done."""
    number, branch = pr["number"], pr["headRefName"]
    run(["git", "fetch", "--prune", "origin"])
    commit = (pr.get("mergeCommit") or {}).get("oid")
    pulled = bool(commit) and subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=config.REPO,
                                             env=checkout.ENV, capture_output=True).returncode == 0
    if pulled and not any(i["branch"] == branch for i in checkout.leftovers()):
        raise SystemExit(f"#{number} is merged, pulled here and closed: nothing left to do")
    print(f"#{number} is merged on GitHub, but " + ("its branch isn't closed" if pulled else "main here doesn't have it yet")
          + ": picking up from the pull")
    if not pulled and (problems := pull_blockers(checkout.BASE, [f["path"] for f in pr["files"]])):
        raise SystemExit(f"#{number} can't be pulled yet:\n" + "\n".join(f"- {p}" for p in problems))
    finish(number, branch, ask)


def start_review():
    """The post-merge review, as a background session of its own: `claude agents` lists it, and it ends in a draft PR. It enters its
    own worktree, so this checkout isn't touched."""
    if not shutil.which("claude"):
        raise SystemExit("claude isn't on the PATH: open Claude Code here and run /post-merge-review")
    subprocess.run(["claude", "--bg", "--agent", "merge-reviewer", REVIEW_PROMPT], cwd=config.REPO, env=checkout.claude_env(), check=True)


def offer_review(ask=input):
    """The review belongs at the end of a batch, not after each merge: asks only once no PR is waiting, and says so otherwise."""
    merges = checkout.unreviewed_merges()
    if not merges:
        return
    waiting = json.loads(gh("pr", "list", "--json", "number,isDraft"))
    if [pr for pr in waiting if not pr["isDraft"]] or waiting:
        print(f"\n{len(merges)} merged PR(s) since the last review, {len(waiting)} still open. When the batch is done: "
              "python scripts/merge_pr.py --review")
    elif confirm(f"\n{len(merges)} merged PR(s) since the last review, none left open. Start the review in the background (uses your Claude plan's limits: past reviews took ~150 calls)?", ask):
        start_review()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("number", nargs="?", type=int, help="the PR to check, approve and merge")
    p.add_argument("--slow", action="store_true", help="run every test on the merged tree, not just the fast ones")
    p.add_argument("--close", action="store_true", help="close every merged branch (worktree, local, GitHub); asks first")
    p.add_argument("--review", action="store_true", help="start the post-merge review of what's merged since the last one, in the background")
    p.add_argument("--auto", action="store_true", help="the merge queue: no questions; with a number, just that PR (see above)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.auto:
        results = auto(args.number)
        if args.number is not None and results.get(args.number) != "merged":
            raise SystemExit(f"#{args.number}: {results.get(args.number)}")
    elif args.review:
        start_review()
    elif args.close:
        must_be_main_checkout()
        run(["git", "fetch", "--prune", "origin"])
        close_all()
    elif args.number:
        merge(args.number, slow=args.slow)
    else:
        show_status()


if __name__ == "__main__":
    main()
