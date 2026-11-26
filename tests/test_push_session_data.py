"""scripts/push_session_data.py: a session's samples committed and pushed to main in one step, nothing else with them."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import push_session_data
from mahjong import checkout

ATTRIBUTES = "data/hover_corrections.jsonl merge=union\ndata/post_merge_reviews.jsonl merge=union\n"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=checkout.ENV).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    origin, main = tmp_path / "origin.git", tmp_path / "main"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True, env=checkout.ENV)
    subprocess.run(["git", "clone", "-q", str(origin), str(main)], check=True, capture_output=True, env=checkout.ENV)
    for key, value in (("user.email", "t@t"), ("user.name", "t"), ("core.hooksPath", "/dev/null")):
        git(main, "config", key, value)
    git(main, "checkout", "-q", "-b", "main")
    (main / "data").mkdir()
    for name, text in ((".gitattributes", ATTRIBUTES), ("data/hover_corrections.jsonl", "{}\n"),
                       ("data/post_merge_reviews.jsonl", "{}\n"), ("data/layouts.json", "{}\n")):
        (main / name).write_text(text)
    git(main, "add", "-A"), git(main, "commit", "-q", "-m", "start"), git(main, "push", "-q", "-u", "origin", "main")
    return main


def session(sessions, name, sim=False):
    (sessions / name).mkdir(parents=True)
    (sessions / name / "events.jsonl").write_text(json.dumps({"type": "session_start", "sim": sim}) + "\n")


def test_the_samples_go_to_github_named_after_the_last_real_session(repo, tmp_path):
    sessions = tmp_path / "sessions"
    session(sessions, "20261113_100710_bench"), session(sessions, "20261113_120000_bench", sim=True)
    for name in ("data/hover_corrections.jsonl", "data/post_merge_reviews.jsonl", "data/layouts.json"):
        (repo / name).write_text("{}\n{}\n")
    assert checkout.unsaved_samples(repo) == ["data/hover_corrections.jsonl"]   # not the review log, not a layout
    push_session_data.main([], ask=lambda q: "y", repo=repo, sessions_dir=sessions)
    assert git(repo, "log", "-1", "--format=%s", "origin/main") == "Samples from 20261113_100710_bench"
    assert git(repo, "show", "--name-only", "--format=", "origin/main") == "data/hover_corrections.jsonl"
    assert checkout.changed_data(repo) == ["data/layouts.json", "data/post_merge_reviews.jsonl"]


def test_a_dry_run_or_a_no_commits_nothing(repo, tmp_path):
    (repo / "data/hover_corrections.jsonl").write_text("{}\n{}\n")
    push_session_data.main(["--dry-run"], ask=lambda q: pytest.fail("a dry run asked"), repo=repo, sessions_dir=tmp_path)
    push_session_data.main([], ask=lambda q: "n", repo=repo, sessions_dir=tmp_path)
    assert git(repo, "log", "--format=%s") == "start"


def test_another_unpushed_commit_stops_it_before_anything_is_committed(repo, tmp_path):
    (repo / "mahjong.py").write_text("x = 1\n")
    git(repo, "add", "mahjong.py"), git(repo, "commit", "-q", "-m", "local code, never meant for GitHub")
    (repo / "data/hover_corrections.jsonl").write_text("{}\n{}\n")
    with pytest.raises(SystemExit, match="1 commit"):
        push_session_data.main([], ask=lambda q: "y", repo=repo, sessions_dir=tmp_path)
    assert git(repo, "log", "-1", "--format=%s", "origin/main") == "start"


def test_a_worktree_is_refused(repo, tmp_path):
    tree = tmp_path / "tree"
    git(repo, "worktree", "add", "-q", "-b", "other", str(tree))
    with pytest.raises(SystemExit, match="main checkout"):
        push_session_data.main([], repo=tree, sessions_dir=tmp_path)
