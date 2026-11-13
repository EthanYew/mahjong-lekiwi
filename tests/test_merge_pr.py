"""merge_pr.py and mahjong/checkout.py on a scratch repo with its own origin: what a merge would bring in, which
branches are merged, and CLAUDE.md's closing step (no GitHub: the gh calls aren't exercised)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merge_pr
from mahjong import checkout


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=checkout.ENV).stdout.strip()


def commit(repo, path, text, message):
    (Path(repo) / path).parent.mkdir(parents=True, exist_ok=True)
    (Path(repo) / path).write_text(text)
    git(repo, "add", path)
    git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path):
    origin, main = tmp_path / "origin.git", tmp_path / "main"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True, env=checkout.ENV)
    subprocess.run(["git", "clone", "-q", str(origin), str(main)], check=True, capture_output=True, env=checkout.ENV)
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
    subprocess.run(["git", "clone", "-q", str(tmp_path / "origin.git"), str(other)], check=True, capture_output=True, env=checkout.ENV)
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


def pulls_cleanly(repo):
    return subprocess.run(["git", "-C", str(repo), "pull", "-q", "--ff-only", "origin", "main"], capture_output=True,
                          env=checkout.ENV).returncode == 0


def merged_on_github(repo, tmp_path, path, text):
    """origin/main moves on as a merge would, and a PR branch with the same file, as last fetched here."""
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", str(tmp_path / "origin.git"), str(other)], check=True, capture_output=True,
                   env=checkout.ENV)
    git(other, "config", "user.email", "t@t"), git(other, "config", "user.name", "t")
    commit(other, path, text, "the PR")
    git(other, "push", "-q", "origin", "main", "main:feature")
    git(repo, "fetch", "-q", "origin")


def test_local_commits_github_lacks_stop_a_merge_until_pushed(repo, tmp_path):
    git(repo, "fetch", "-q", "origin")
    assert merge_pr.pull_blockers("origin/main", [], repo) == []
    commit(repo, "data/hover_corrections.jsonl", "{}\n", "data updates")
    problems = merge_pr.pull_blockers("origin/main", [], repo)
    assert len(problems) == 1 and "1 commit(s)" in problems[0] and "data updates" in problems[0]
    git(repo, "push", "-q", "origin", "main")
    git(repo, "fetch", "-q", "origin")
    assert merge_pr.pull_blockers("origin/main", [], repo) == []


def test_those_commits_are_what_makes_the_pull_fail(repo, tmp_path):
    commit(repo, "data/hover_corrections.jsonl", "{}\n", "data updates")
    merged_on_github(repo, tmp_path, "mahjong/x.py", "X = 1\n")
    assert merge_pr.pull_blockers("origin/feature", ["mahjong/x.py"], repo)
    assert not pulls_cleanly(repo)


def test_being_behind_github_alone_doesn_t_stop_a_merge(repo, tmp_path):
    merged_on_github(repo, tmp_path, "mahjong/x.py", "X = 1\n")
    assert merge_pr.pull_blockers("origin/feature", ["mahjong/x.py"], repo) == []
    assert pulls_cleanly(repo)


def test_uncommitted_data_only_matters_if_the_pr_changes_that_file(repo, tmp_path):
    commit(repo, "data/hover_corrections.jsonl", "{}\n", "start")
    commit(repo, "data/return_corrections.jsonl", "{}\n", "start")
    git(repo, "push", "-q", "origin", "main")
    (repo / "data" / "hover_corrections.jsonl").write_text("{}\n{\"new\": 1}\n")
    merged_on_github(repo, tmp_path, "data/return_corrections.jsonl", "{}\n{\"pr\": 1}\n")
    assert merge_pr.pull_blockers("origin/feature", ["data/return_corrections.jsonl"], repo) == []
    assert pulls_cleanly(repo)                            # the bench's uncommitted samples came through
    assert "new" in (repo / "data" / "hover_corrections.jsonl").read_text()


def test_uncommitted_data_the_pr_also_changes_stops_a_merge_unless_it_s_the_same_lines_staged(repo, tmp_path):
    commit(repo, "data/hover_corrections.jsonl", "{}\n", "start")
    git(repo, "push", "-q", "origin", "main")
    (repo / "data" / "hover_corrections.jsonl").write_text("{}\n{\"mine\": 1}\n")
    merged_on_github(repo, tmp_path, "data/hover_corrections.jsonl", "{}\n{\"pr\": 1}\n")
    problems = merge_pr.pull_blockers("origin/feature", ["data/hover_corrections.jsonl"], repo)
    assert len(problems) == 1 and "hover_corrections" in problems[0]
    assert not pulls_cleanly(repo)
    (repo / "data" / "hover_corrections.jsonl").write_text("{}\n{\"pr\": 1}\n")     # PR #63: review lines staged here too
    git(repo, "add", "data/hover_corrections.jsonl")
    assert merge_pr.pull_blockers("origin/feature", ["data/hover_corrections.jsonl"], repo) == []
    assert pulls_cleanly(repo)


def test_a_blocked_merge_stops_before_it_asks_or_touches_github(repo, monkeypatch):
    from mahjong import config
    monkeypatch.setattr(config, "REPO", repo)
    git(repo, "fetch", "-q", "origin")
    commit(repo, "data/hover_corrections.jsonl", "{}\n", "data updates")
    calls = []

    def gh(*args):
        calls.append(args)
        return ('{"number": 7, "title": "t", "state": "OPEN", "isDraft": true, "headRefName": "feature", "body": "", '
                '"url": "u", "files": [{"path": "mahjong/x.py"}]}')
    monkeypatch.setattr(merge_pr, "gh", gh)

    def never(question):
        raise AssertionError("it asked")
    with pytest.raises(SystemExit, match="not merging #7, nothing changed"):
        merge_pr.merge(7, ask=never)
    assert [c[:2] for c in calls] == [("pr", "view")]


def test_a_pr_description_is_checked_for_its_sections():
    body = "## What changed\nx\n## Files\ny\n## Overlaps\nnone\n"
    assert merge_pr.missing_sections(body) == ["Unverified on the arm"]
    assert merge_pr.missing_sections(body + "## Unverified on the arm\nall of it\n") == []
    full = body + "## Unverified on the arm\nall of it\n"
    assert merge_pr.missing_sections(full, ["README.md", "mahjong/session.py"]) == []
    assert merge_pr.missing_sections(full, ["scripts/play.py"]) == ["Review"]
    assert merge_pr.missing_sections(full + "## Review\n/code-review: 2 findings, both fixed\n", ["scripts/play.py"]) == []


def test_closing_leaves_a_simulated_session_behind_though_its_name_looks_real(repo):
    """move_sessions only skipped sim_ folders, and 40 sim runs named like real ones reached the real sessions/ by 2026-11-13."""
    tree = repo / "tree"
    for name, start in (("20261113_092441_bench", {"type": "session_start", "sim": True}),
                        ("20261113_100710_bench", {"type": "session_start", "sim": False})):
        (tree / "sessions" / name).mkdir(parents=True)
        (tree / "sessions" / name / "events.jsonl").write_text(json.dumps(start) + "\n")
    assert merge_pr.move_sessions(tree, repo=repo) == []
    assert sorted(p.name for p in (repo / "sessions").iterdir()) == ["20261113_100710_bench"]


def test_a_merge_whose_pull_failed_is_finished_by_running_it_again(repo, monkeypatch):
    """#76 merged on GitHub, an SSH drop stopped the pull, and a rerun only said it was merged: untested and never closed."""
    from mahjong import config
    monkeypatch.setattr(config, "REPO", repo)
    branch_in_worktree(repo, "feature", "mahjong/x.py", "X = 1\n")
    tip = git(repo, "rev-parse", "feature")
    git(repo, "push", "-q", "origin", "feature:main")              # merged on GitHub; main here never pulled it
    monkeypatch.setattr(merge_pr, "gh", lambda *a: json.dumps(
        {"number": 7, "title": "t", "state": "MERGED", "isDraft": False, "headRefName": "feature", "body": "", "url": "u",
         "files": [{"path": "mahjong/x.py"}], "mergeCommit": {"oid": tip}}))
    monkeypatch.setattr(merge_pr, "test_command", lambda slow=False: ["true"])
    monkeypatch.setattr(merge_pr, "offer_review", lambda ask=None: None)
    merge_pr.merge(7, ask=lambda q: "y")
    assert git(repo, "rev-parse", "HEAD") == tip and not (repo / ".claude" / "worktrees" / "feature").exists()
    assert "feature" not in git(repo, "branch", "--list", "feature")
    with pytest.raises(SystemExit, match="nothing left to do"):
        merge_pr.merge(7, ask=lambda q: "y")


def test_a_pr_changing_play_without_a_review_asks_before_testing(repo, monkeypatch, capsys):
    from mahjong import config
    monkeypatch.setattr(config, "REPO", repo)
    body = "## What changed\nx\n## Files\ny\n## Overlaps\nnone\n## Unverified on the arm\nall\n"
    monkeypatch.setattr(merge_pr, "gh", lambda *a: json.dumps(
        {"number": 7, "title": "t", "state": "OPEN", "isDraft": True, "headRefName": "feature", "body": body, "url": "u",
         "files": [{"path": "scripts/play.py"}]}))
    monkeypatch.setattr(merge_pr, "test_merged", lambda *a, **k: pytest.fail("tested an unreviewed PR the person turned down"))
    questions = []
    merge_pr.merge(7, ask=lambda q: questions.append(q) or "n")
    assert "scripts/play.py" in questions[0] and "Not merged: no review." in capsys.readouterr().out
