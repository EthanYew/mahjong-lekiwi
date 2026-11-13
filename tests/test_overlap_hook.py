""".claude/hooks/overlap_warning.py: after an Edit or Write, a note if another open branch changes the same file."""
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".claude" / "hooks"))

import overlap_warning
from mahjong import checkout


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=checkout.ENV).stdout.strip()


def commit(repo, path, text, message):
    (Path(repo) / path).parent.mkdir(parents=True, exist_ok=True)
    (Path(repo) / path).write_text(text)
    git(repo, "add", path)
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path):
    origin, main = tmp_path / "origin.git", tmp_path / "main"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True, env=checkout.ENV)
    subprocess.run(["git", "clone", "-q", str(origin), str(main)], check=True, capture_output=True, env=checkout.ENV)
    for key, value in (("user.email", "t@t"), ("user.name", "t"), ("core.hooksPath", "/dev/null")):
        git(main, "config", key, value)
    git(main, "checkout", "-q", "-b", "main")
    commit(main, "scripts/bench.py", "x = 1\n", "start")
    git(main, "push", "-q", "-u", "origin", "main")
    return main


def edited(path, session="s1"):
    out = io.StringIO()
    overlap_warning.main(io.StringIO(json.dumps({"session_id": session, "tool_input": {"file_path": str(path)}})), out)
    return json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"] if out.getvalue() else None


def test_an_edit_to_a_file_another_open_branch_changes_gets_one_note(repo, tmp_path):
    git(repo, "checkout", "-q", "-b", "other")
    commit(repo, "scripts/bench.py", "x = 2\n", "other's change")
    git(repo, "push", "-q", "origin", "other")
    tree = tmp_path / "mine"
    git(repo, "worktree", "add", "-q", "-b", "mine", str(tree), "origin/main")
    (tree / "scripts/bench.py").write_text("x = 3\n")
    (tree / "scripts/record.py").write_text("y = 1\n")
    note = edited(tree / "scripts/bench.py")
    assert "scripts/bench.py is also changed by open branch(es) other" in note
    assert edited(tree / "scripts/bench.py") is None                  # once a session
    assert edited(tree / "scripts/bench.py", session="s2") is not None
    assert edited(tree / "scripts/record.py") is None                 # nobody else changes it


def test_a_merged_branch_or_this_branch_s_own_base_is_not_an_overlap(repo, tmp_path):
    git(repo, "checkout", "-q", "-b", "base")
    commit(repo, "scripts/bench.py", "x = 2\n", "base's change")
    git(repo, "push", "-q", "origin", "base")
    git(repo, "checkout", "-q", "-b", "stacked")                       # built on base: base's changes are its own
    assert edited(repo / "scripts/bench.py") is None
    git(repo, "push", "-q", "origin", "base:main")                     # base merged
    git(repo, "fetch", "-q", "origin")
    git(repo, "checkout", "-q", "main"), git(repo, "reset", "-q", "--hard", "origin/main")
    assert edited(repo / "scripts/bench.py", session="s3") is None


def test_a_file_outside_any_repo_or_bad_input_is_left_alone(tmp_path):
    assert edited(tmp_path / "loose.txt") is None
    overlap_warning.main(io.StringIO("not json"), io.StringIO())
