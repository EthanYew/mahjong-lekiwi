"""Approve and merge a branch's PR, then close the branch (CLAUDE.md, "Worktrees and merging").

    python scripts/merge_pr.py               # open PRs, and merged branches/worktrees not closed yet (changes nothing)
    python scripts/merge_pr.py 34            # check PR 34, ask, merge it, pull main, test main, close its branch
    python scripts/merge_pr.py 34 --slow     # ... with the slow (GPU) tests on the merged tree too
    python scripts/merge_pr.py --close       # close every merged branch: worktree, local and GitHub branch (asks)

Run it from the main checkout. Before asking, it shows what you're approving:
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
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import checkout, config

SECTIONS = ["What changed", "Files", "Overlaps", "Unverified on the arm"]
SLOW_FILES = ("mahjong/policy.py", "mahjong/dataset.py")      # CLAUDE.md: run the slow tests too
DEF = re.compile(r"^\+\s*def (\w+)\(|^\+([A-Z][A-Z0-9_]+) = |add_argument\(\s*\"(--[\w-]+)\"")


def run(cmd, cwd=None, check=True):
    print("$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd or config.REPO)
    if check and result.returncode:
        raise SystemExit(f"stopped: {cmd[0]} exited {result.returncode}")
    return result.returncode


def gh(*args):
    result = subprocess.run(["gh", *args], cwd=config.REPO, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"gh {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def confirm(question, ask=input):
    return ask(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def missing_sections(body):
    """The PR description's sections that aren't there (a markdown heading each)."""
    headings = {h.strip().lower() for h in re.findall(r"^#+\s*(.+)$", body or "", re.M)}
    return [s for s in SECTIONS if not any(h.startswith(s.lower()) for h in headings)]


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
        if folder.name.startswith("sim_"):
            continue                    # a simulated run's reference, not a session
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
        run(["git", "push", "origin", "--delete", name], cwd=repo, check=False)
    return True


def must_be_main_checkout():
    if checkout.is_worktree():
        raise SystemExit("run this from the main checkout (hardware runs there, and it pulls main)")
    if checkout.git("rev-parse", "--abbrev-ref", "HEAD") != "main":
        raise SystemExit("the main checkout isn't on main: switch back first")
    if (files := checkout.changed_code()):
        raise SystemExit(f"uncommitted code in the main checkout: {', '.join(files)} (commit or move it first)")


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
        print(f"\nPR closed without merging (--close asks about these separately):")
        for item in gone:
            print(f"  {item['branch']:<36} {where(item)}  (#{item['pr']}, last commit {item['tip']})")
    print(f"\nTo approve one: python scripts/merge_pr.py <number>")


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
    pr = json.loads(gh("pr", "view", str(number), "--json", "number,title,state,isDraft,headRefName,body,url,files"))
    if pr["state"] != "OPEN":
        raise SystemExit(f"#{number} is {pr['state'].lower()}")
    branch = pr["headRefName"]
    run(["git", "fetch", "--prune", "origin"])
    ref = f"origin/{branch}"
    print(f"\n#{number} {pr['title']}\n{pr['url']}\n\n{pr['body'] or '(no description)'}\n")
    if (missing := missing_sections(pr["body"])):
        print(f"MISSING from the description: {', '.join(missing)}")
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
    run(["git", "pull", "--ff-only", "origin", "main"])
    if run(test_command(), check=False):
        print("WARNING: the fast tests fail on main after the pull: fix main before the next hardware session")
    run(["git", "fetch", "--prune", "origin"])
    item = next((i for i in checkout.leftovers() if i["branch"] == branch), None)
    if item and close_branch(item):
        print(f"\nMerged #{number} and closed {branch}.")
    else:
        print(f"\nMerged #{number}; {branch} isn't closed (above): python scripts/merge_pr.py --close")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("number", nargs="?", type=int, help="the PR to check, approve and merge")
    p.add_argument("--slow", action="store_true", help="run every test on the merged tree, not just the fast ones")
    p.add_argument("--close", action="store_true", help="close every merged branch (worktree, local, GitHub); asks first")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.close:
        must_be_main_checkout()
        run(["git", "fetch", "--prune", "origin"])
        close_all()
    elif args.number:
        merge(args.number, slow=args.slow)
    else:
        show_status()


if __name__ == "__main__":
    main()
