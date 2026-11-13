"""The git checkout a script runs from, and the branches and worktrees around it.

Hardware runs from the main checkout (sessions/ is per checkout), so it should run the code that's on main:
on 2026-10-29 the main checkout was 2 merges behind GitHub, PR #33's change to returns included. Sessions
warn when it's behind origin/main (as of the last fetch: no network here), has uncommitted code, or is a
worktree. scripts/merge_pr.py and scripts/status.py use the rest: which branches are merged and which
worktrees are left to close.
"""
import json
import os
import subprocess
from pathlib import Path

from . import config

BASE = "origin/main"
REVIEW_LOG = "data/post_merge_reviews.jsonl"     # one line per review, written by its PR (.claude/agents/merge-reviewer.md)
# Inside a git hook (the pre-commit tests), git sets these for the repo being committed to, and a `git -C other`
# would follow them there instead
ENV = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_PREFIX")}


def claude_env():
    """For any `claude` these scripts start: the claude.ai login, never an ANTHROPIC_API_KEY in the shell. A headless run takes that key
    over the login, which billed the owner $4.57 unasked on 2026-11-05, though their interactive sessions had rejected the key."""
    return {k: v for k, v in ENV.items() if k != "ANTHROPIC_API_KEY"}
# Written by the scripts as they run (corrections, reviews, layouts...), not code: uncommitted is normal
DATA_PREFIXES = ("data/", "sessions/")


def git(*args, repo=None, check=False, strip=True):
    """git's stdout, stripped ("" if git fails, unless check)."""
    try:
        result = subprocess.run(["git", "-C", str(repo or config.REPO), *args], capture_output=True, text=True,
                                timeout=60, env=ENV)
    except (OSError, subprocess.TimeoutExpired):
        if check:
            raise
        return ""
    if check and result.returncode:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    if result.returncode:
        return ""
    return result.stdout.strip() if strip else result.stdout


def behind(repo=None, base=BASE):
    """Commits on base (as last fetched) that HEAD doesn't have, or None if there's no base."""
    count = git("rev-list", "--count", f"HEAD..{base}", repo=repo)
    return int(count) if count.isdigit() else None


def unpushed(repo=None, base=BASE):
    """Commits on HEAD that base (as last fetched) doesn't have, "<hash> <subject>" each, or [] if there's no base."""
    return git("log", "--format=%h %s", f"{base}..HEAD", repo=repo).splitlines()


def _changed(repo=None):
    lines = git("status", "--porcelain", "--untracked-files=no", repo=repo, strip=False).splitlines()
    return [line[3:] for line in lines if line[3:]]


def changed_code(repo=None):
    """Tracked files with uncommitted changes, data files left out."""
    return [path for path in _changed(repo) if not path.startswith(DATA_PREFIXES)]


def changed_data(repo=None):
    """Tracked data files with uncommitted changes: normal while a script is logging, but a pull changing one stops."""
    return [path for path in _changed(repo) if path.startswith(DATA_PREFIXES)]


def is_worktree(repo=None):
    git_dir, common = git("rev-parse", "--absolute-git-dir", repo=repo), git("rev-parse", "--git-common-dir", repo=repo)
    return bool(git_dir) and Path(git_dir) != (Path(repo or config.REPO) / common).resolve()


def last_reviewed(repo=None, base=BASE):
    """The newest commit a post-merge review covered that's on base, or None. A review's PR writes it, so a review only counts
    once it has merged."""
    path = Path(repo or config.REPO) / REVIEW_LOG
    if not path.exists():
        return None
    reviewed = [json.loads(line)["reviewed_to"] for line in path.read_text().splitlines() if line.strip()]
    on_base = [sha for sha in reviewed if git("rev-list", "--count", f"{base}..{sha}", repo=repo) == "0"]
    return min(on_base, key=lambda sha: int(git("rev-list", "--count", f"{sha}..{base}", repo=repo)), default=None)


def unreviewed_merges(repo=None, base=BASE):
    """Merged PRs on base since the last post-merge review, "<hash> <subject>" each. A review's own PR doesn't count, or
    each review would ask for another. [] if no review has been logged yet."""
    start = last_reviewed(repo, base)
    if not start:
        return []
    return [line for line in git("log", "--first-parent", "--merges", "--format=%h %s", f"{start}..{base}", repo=repo).splitlines()
            if "Merge pull request" in line and "post-merge-review" not in line]


def warnings(repo=None):
    """What's off about running hardware from this checkout, one line each."""
    out = []
    if (merges := unreviewed_merges(repo)):
        out.append(f"{len(merges)} merged PR(s) with no post-merge review yet: python scripts/merge_pr.py --review")
    if (n := behind(repo)):
        out.append(f"this checkout is {n} commit(s) behind {BASE} (as last fetched): git pull first, or the "
                   f"session runs older code")
    if (files := changed_code(repo)):
        out.append(f"uncommitted code: {', '.join(files[:5])}{' ...' if len(files) > 5 else ''}")
    if is_worktree(repo):
        out.append("this is a worktree: its sessions/ go when it's removed (run hardware from the main checkout)")
    return out


def state(repo=None):
    """For the session_start event: commit, branch, commits behind origin/main, uncommitted code."""
    return {"branch": git("rev-parse", "--abbrev-ref", "HEAD", repo=repo), "behind": behind(repo),
            "changed_code": changed_code(repo), "worktree": is_worktree(repo)}


def worktrees(repo=None):
    """[{"path", "branch"}] for every worktree but the main checkout."""
    found, current = [], {}
    for line in git("worktree", "list", "--porcelain", repo=repo).splitlines() + [""]:
        if line.startswith("worktree "):
            current = {"path": Path(line.split(" ", 1)[1]), "branch": None}
        elif line.startswith("branch "):
            current["branch"] = line.split(" ", 1)[1].removeprefix("refs/heads/")
        elif not line and current:
            found.append(current)
            current = {}
    return found[1:]


def merged(branch, repo=None, base=BASE):
    """Every commit of branch is on base, by content (a merge, a rebase or a cherry-pick all count)."""
    out = git("cherry", base, branch, repo=repo)
    return git("rev-parse", "--verify", "--quiet", branch, repo=repo) != "" and \
        not any(line.startswith("+") for line in out.splitlines())


def local_branches(repo=None):
    return [b for b in git("for-each-ref", "--format=%(refname:short)", "refs/heads", repo=repo).splitlines()
            if b not in ("main", "master")]


def remote_branches(repo=None, remote="origin"):
    return [b.removeprefix(f"{remote}/") for b in
            git("for-each-ref", "--format=%(refname:short)", f"refs/remotes/{remote}", repo=repo).splitlines()
            if b not in (f"{remote}/main", f"{remote}/master", f"{remote}/HEAD", remote)]


def leftovers(repo=None, base=BASE):
    """CLAUDE.md's closing step not done: merged branches still with a worktree, a local or a remote branch."""
    trees = {w["branch"]: w["path"] for w in worktrees(repo) if w["branch"]}
    local, remote = set(local_branches(repo)), set(remote_branches(repo))
    return [{"branch": name, "worktree": trees.get(name), "local": name in local, "remote": name in remote}
            for name in sorted(set(trees) | local | remote)
            if merged(name if name in local else f"origin/{name}", repo=repo, base=base)]
