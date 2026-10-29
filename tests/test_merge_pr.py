"""merge_pr.py and mahjong/checkout.py on a scratch repo with its own origin: what a merge would bring in, which
branches are merged, and CLAUDE.md's closing step (no GitHub: the gh calls aren't exercised)."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merge_pr
from mahjong import checkout


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, path, text, message):
    (Path(repo) / path).parent.mkdir(parents=True, exist_ok=True)
    (Path(repo) / path).write_text(text)
    git(repo, "add", path)
    git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path):
    origin, main = tmp_path / "origin.git", tmp_path / "main"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(main)], check=True, capture_output=True)
    git(main, "config", "user.email", "t@t"), git(main, "config", "user.name", "t")
    git(main, "checkout", "-q", "-b", "main")
    commit(main, "mahjong/arm.py", "def move_to():\n    pass\n", "start")
    git(main, "push", "-q", "-u", "origin", "main")
    return main


def branch_in_worktree(repo, name, path, text):
    tree = repo / ".claude" / "worktrees" / name
    git(repo, "worktree", "add", "-q", "-b", name, str(tree))
    commit(tree, path, text, f"{name} work")
    git(tree, "push", "-q", "-u", "origin", name)
    return tree


def test_the_same_function_added_on_both_sides_is_reported_though_git_merges_it_cleanly(repo):
    branch_in_worktree(repo, "feature", "scripts/bench.py",
                       'def held_check():\n    pass\nRADIUS_M = 1\np.add_argument("--hover-jitter-mm")\n')
    commit(repo, "scripts/record.py", 'def held_check():\n    pass\np.add_argument("--hover-jitter-mm")\n', "main side")
    commit(repo, "scripts/bench.py", "# main touched it too\n", "main bench")
    git(repo, "push", "-q", "origin", "main")
    git(repo, "fetch", "-q", "origin")
    files, names = merge_pr.overlaps("origin/feature", repo=repo)
    assert files == ["scripts/bench.py"] and names == ["--hover-jitter-mm", "held_check"]


def test_a_merged_branch_is_closed_and_its_sessions_kept_but_an_unmerged_one_is_left(repo):
    tree = branch_in_worktree(repo, "done", "mahjong/x.py", "X = 1\n")
    (tree / "sessions" / "20261029_120000_bench").mkdir(parents=True)
    (tree / "sessions" / "20261029_120000_bench" / "events.jsonl").write_text("{}\n")
    branch_in_worktree(repo, "wip", "mahjong/y.py", "Y = 1\n")
    git(repo, "merge", "-q", "--no-ff", "-m", "Merge done", "done")
    git(repo, "push", "-q", "origin", "main")
    git(repo, "fetch", "-q", "origin")

    left = checkout.leftovers(repo=repo)
    assert [i["branch"] for i in left] == ["done"] and left[0]["worktree"] == tree and left[0]["remote"]
    assert merge_pr.close_branch(left[0], repo=repo)
    assert (repo / "sessions" / "20261029_120000_bench" / "events.jsonl").exists()
    assert not tree.exists() and "done" not in checkout.local_branches(repo)
    git(repo, "fetch", "-q", "--prune", "origin")
    assert "done" not in checkout.remote_branches(repo) and "wip" in checkout.remote_branches(repo)
    assert checkout.leftovers(repo=repo) == []


def test_a_worktree_with_uncommitted_code_is_not_removed(repo):
    tree = branch_in_worktree(repo, "done", "mahjong/x.py", "X = 1\n")
    git(repo, "merge", "-q", "--no-ff", "-m", "Merge done", "done")
    git(repo, "push", "-q", "origin", "main")
    git(repo, "fetch", "-q", "origin")
    (tree / "mahjong" / "x.py").write_text("X = 2\n")
    (tree / "data").mkdir()
    assert checkout.changed_code(tree) == ["mahjong/x.py"]
    assert not merge_pr.close_branch(checkout.leftovers(repo=repo)[0], repo=repo, push=False) and tree.exists()


def test_the_checkout_warns_when_main_moved_on_without_it(repo, tmp_path):
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", str(tmp_path / "origin.git"), str(other)], check=True, capture_output=True)
    git(other, "config", "user.email", "t@t"), git(other, "config", "user.name", "t")
    commit(other, "mahjong/arm.py", "def move_to():\n    return 1\n", "merged elsewhere")
    git(other, "push", "-q", "origin", "main")
    assert checkout.warnings(repo) == []
    git(repo, "fetch", "-q", "origin")
    assert checkout.behind(repo) == 1 and "1 commit(s) behind" in checkout.warnings(repo)[0]
    commit(repo, "data/hover_corrections.jsonl", "{}\n", "data")
    (repo / "data" / "hover_corrections.jsonl").write_text("{}\n{}\n")
    assert checkout.changed_code(repo) == []          # data files the scripts append to aren't code


def test_the_merged_tree_is_tested_in_a_scratch_worktree_and_a_conflict_stops_it(repo, monkeypatch):
    monkeypatch.setattr(merge_pr, "test_command", lambda slow=False: ["sh", "-c", "ls mahjong > ../seen.txt"])
    branch_in_worktree(repo, "clean", "mahjong/new.py", "N = 1\n")
    git(repo, "fetch", "-q", "origin")
    assert merge_pr.test_merged("origin/clean", repo=repo)
    seen = (repo / ".claude" / "worktrees" / "seen.txt").read_text().split()
    assert seen == ["arm.py", "new.py"]                       # main's files and the branch's, merged
    assert not (repo / ".claude" / "worktrees" / "_merge_check").exists()

    branch_in_worktree(repo, "clash", "mahjong/arm.py", "def move_to():\n    return 2\n")
    commit(repo, "mahjong/arm.py", "def move_to():\n    return 3\n", "main changed the same line")
    git(repo, "push", "-q", "origin", "main")
    git(repo, "fetch", "-q", "origin")
    assert not merge_pr.test_merged("origin/clash", repo=repo)
    assert not (repo / ".claude" / "worktrees" / "_merge_check").exists()


def test_a_pr_description_is_checked_for_its_sections():
    body = "## What changed\nx\n## Files\ny\n## Overlaps\nnone\n"
    assert merge_pr.missing_sections(body) == ["Unverified on the arm"]
    assert merge_pr.missing_sections(body + "## Unverified on the arm\nall of it\n") == []
