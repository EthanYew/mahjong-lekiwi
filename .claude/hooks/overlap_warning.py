"""Claude Code calls this after every Edit and Write (`.claude/settings.json`): if another open branch already changes the file,
it tells Claude, once per file per session, while the change is still small.

CLAUDE.md: two branches shouldn't change the same shared file, because git catches overlapping lines, not overlapping intent.
That rule only lived in the docs, and merge_pr.py only shows an overlap at merge time: 10 of the 12 PRs from #67 to #78 changed
play.py. It never blocks; it adds a note for Claude to act on (merge the other first, or say so in the PR's Overlaps).

Open branches are GitHub's as last fetched (refs/remotes/origin), less those already in origin/main and those this branch is
built on. Each branch's files are cached in the repo's git dir and redone only when the remote refs change.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = "origin/main"
# Git sets these for a hook it runs (a commit's tests): left in, every git call here would act on that repo, not the edited one
ENV = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_PREFIX")}


def git(*args, cwd):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=ENV)
    return result.stdout.strip() if result.returncode == 0 else None


def branch_files(top):
    """{branch: [files it changes since it left origin/main]} for every open remote branch, cached by the remote refs."""
    refs = git("for-each-ref", "--format=%(refname:short) %(objectname)", "refs/remotes/origin", cwd=top) or ""
    key = hashlib.sha1(refs.encode()).hexdigest()
    cache_path = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=top)) / "claude_overlap_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    if cache.get("key") != key:
        files = {}
        for line in refs.splitlines():
            ref = line.split()[0]
            if ref in (BASE, "origin/HEAD", "origin") or ref.endswith("/HEAD"):
                continue
            cherry = git("cherry", BASE, ref, cwd=top)
            if cherry is None or not any(c.startswith("+") for c in cherry.splitlines()):
                continue                                    # already on main, by content
            point = git("merge-base", BASE, ref, cwd=top)
            changed = git("diff", "--name-only", point, ref, cwd=top) if point else None
            if changed:
                files[ref.removeprefix("origin/")] = changed.splitlines()
        warned = cache.get("warned", {})
        cache = {"key": key, "files": files, "warned": dict(list(warned.items())[-20:])}    # the last 20 sessions
    return cache, cache_path


def main(stdin=sys.stdin, stdout=sys.stdout):
    try:
        event = json.load(stdin)
    except ValueError:
        return
    path = (event.get("tool_input") or {}).get("file_path")
    if not path:
        return
    folder = Path(path).parent
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    top = git("rev-parse", "--show-toplevel", cwd=folder)
    if not top:
        return
    try:
        relative = str(Path(path).resolve().relative_to(Path(top).resolve()))
    except ValueError:
        return
    current = git("rev-parse", "--abbrev-ref", "HEAD", cwd=top)
    cache, cache_path = branch_files(top)
    others = sorted(b for b, files in cache["files"].items() if relative in files and b != current
                    and git("merge-base", "--is-ancestor", f"origin/{b}", "HEAD", cwd=top) is None)  # not this one's base
    warned = cache["warned"].setdefault(event.get("session_id", ""), [])
    if others and relative not in warned:
        warned.append(relative)
        stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext":
            f"{relative} is also changed by open branch(es) {', '.join(others)}, not yet on main. CLAUDE.md: two branches "
            "shouldn't change the same file, since git catches overlapping lines, not intent. Check what they change there; "
            "if it overlaps, merge the other first or build on it, and name it under ## Overlaps in this branch's PR."}}))
    try:
        cache_path.write_text(json.dumps(cache))
    except OSError:
        pass


if __name__ == "__main__":
    main()
