"""The post-merge review's plumbing: what counts as unreviewed, the brief, and a backtest's scratch copy (no model is called)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merge_pr
import review_backtest
import review_brief
from mahjong import checkout


def sh(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=checkout.ENV)
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True, env=checkout.ENV).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """main with a first commit, then PR #1 and a review's own PR #2 merged, and a branch for origin/main to point at."""
    root = tmp_path / "repo"
    root.mkdir()
    sh(root, "init", "-q", "-b", "main")
    sh(root, "config", "user.email", "t@example.com")
    sh(root, "config", "user.name", "t")
    sh(root, "config", "core.hooksPath", "/dev/null")
    (root / "scripts").mkdir()
    (root / "scripts" / "a.py").write_text("x = 1\n")
    sh(root, "add", ".")
    first = sh(root, "commit", "-qm", "first")
    for number, branch, name in ((1, "worktree-feature", "b.py"), (2, "worktree-post-merge-review-abc", "c.py")):
        sh(root, "checkout", "-qb", branch)
        (root / "scripts" / name).write_text("y = 2\n")
        sh(root, "add", ".")
        sh(root, "commit", "-qm", f"work on {name}")
        sh(root, "checkout", "-q", "main")
        sh(root, "merge", "-q", "--no-ff", branch, "-m", f"Merge pull request #{number} from EthanYew/{branch}")
    sh(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    return root, first


def log_review(root, sha):
    (root / "data").mkdir(exist_ok=True)
    with (root / "data" / "post_merge_reviews.jsonl").open("a") as f:
        f.write(json.dumps({"reviewed_to": sha}) + "\n")


def test_no_log_means_nothing_to_nag_about(repo):
    root, _ = repo
    assert checkout.last_reviewed(root) is None
    assert checkout.unreviewed_merges(root) == []


def test_merges_since_the_last_review_leave_out_a_reviews_own_pr(repo):
    root, first = repo
    log_review(root, first)
    merges = checkout.unreviewed_merges(root)
    assert len(merges) == 1 and "#1" in merges[0]
    assert any("post-merge review" in w for w in checkout.warnings(root))


def test_nothing_is_unreviewed_once_a_review_reaches_the_tip(repo):
    root, first = repo
    log_review(root, first)
    log_review(root, sh(root, "rev-parse", "HEAD"))
    assert checkout.unreviewed_merges(root) == []


def test_a_review_of_a_commit_not_on_main_doesnt_count(repo):
    """A review's line arrives with its PR, but a branch that was never merged mustn't mark anything reviewed."""
    root, first = repo
    sh(root, "checkout", "-qb", "elsewhere")
    (root / "scripts" / "d.py").write_text("z = 3\n")
    sh(root, "add", ".")
    other = sh(root, "commit", "-qm", "not on main")
    sh(root, "checkout", "-q", "main")
    log_review(root, first)
    log_review(root, other)
    assert checkout.last_reviewed(root) == first


def test_sections_takes_only_what_the_review_needs():
    body = "## What changed\nlots\n\n## Overlaps\n- play.py\n\n## Unverified on the arm\nall of it\n"
    assert review_brief.sections(body) == {"Overlaps": "- play.py", "Unverified on the arm": "all of it"}
    assert review_brief.sections(None) == {}


def test_the_brief_lists_merges_and_what_several_of_them_changed(repo):
    root, first = repo
    text = review_brief.brief(first, "HEAD", root, bodies={"1": "## Overlaps\nnothing else\n"}, run=False)
    assert "Merge pull request #1" in text and "**Overlaps:** nothing else" in text
    assert "1 commit" not in text.splitlines()[2]       # two on the main line


def test_a_sim_run_that_waits_for_an_answer_is_reported_not_left_hanging(tmp_path):
    code, tail = review_brief.run_quietly([sys.executable, "-c", "input()"], tmp_path, timeout=20)
    assert code != 0 and "EOF" in tail


def test_the_review_is_offered_only_when_no_pr_is_waiting(repo, monkeypatch, capsys):
    root, first = repo
    log_review(root, first)
    monkeypatch.setattr(merge_pr.checkout, "unreviewed_merges", lambda: ["abc Merge pull request #1"])
    started = []
    monkeypatch.setattr(merge_pr, "start_review", lambda: started.append(True))
    monkeypatch.setattr(merge_pr, "gh", lambda *a: json.dumps([{"number": 9, "isDraft": True}]))
    merge_pr.offer_review(ask=lambda q: pytest.fail("asked with a PR still open"))
    assert "--review" in capsys.readouterr().out and not started
    monkeypatch.setattr(merge_pr, "gh", lambda *a: "[]")
    merge_pr.offer_review(ask=lambda q: "y")
    assert started == [True]


def test_a_scratch_copy_has_no_future_no_remote_and_only_earlier_sessions(repo, tmp_path):
    root, first = repo
    sessions = tmp_path / "sessions"
    for name in ("20261101_100000_bench", "20261105_100000_bench"):
        (sessions / name).mkdir(parents=True)
        (sessions / name / "events.jsonl").write_text("{}\n")
        (sessions / name / "wrist.mp4").write_text("video")
    target = review_backtest.scratch_copy(first, tmp_path / "scratch", root, "20261102", sessions)
    log = subprocess.run(["git", "log", "--all", "--format=%s"], cwd=target, capture_output=True, text=True, env=checkout.ENV).stdout
    assert log.strip() == "first"
    assert subprocess.run(["git", "remote"], cwd=target, capture_output=True, text=True, env=checkout.ENV).stdout == ""
    assert [p.name for p in (target / "sessions").iterdir()] == ["20261101_100000_bench"]
    assert (target / "sessions" / "20261101_100000_bench" / "events.jsonl").exists()
    assert not (target / "sessions" / "20261101_100000_bench" / "wrist.mp4").exists()


def test_a_backtest_refuses_an_arm_script_where_the_real_gate_would_ask(tmp_path):
    review_backtest.add_reviewer(tmp_path)
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    payload = json.dumps({"tool_input": {"command": "python scripts/bench.py --trials 3"}})
    out = subprocess.run(["sh", "-c", command], cwd=tmp_path, input=payload, capture_output=True, text=True).stdout
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    payload = json.dumps({"tool_input": {"command": "python scripts/bench.py --sim --trials 3"}})
    assert subprocess.run(["sh", "-c", command], cwd=tmp_path, input=payload, capture_output=True, text=True).stdout.strip() == ""


def test_no_claude_these_scripts_start_can_bill_an_api_key(monkeypatch):
    """A key in the shell wins over the login in a headless run: three backtests spent $4.57 of it, unasked."""
    monkeypatch.setitem(checkout.ENV, "ANTHROPIC_API_KEY", "x")
    assert "ANTHROPIC_API_KEY" not in checkout.claude_env()
    seen = {}

    def fake_run(cmd, **kwargs):
        seen[cmd[0]] = kwargs["env"]

    monkeypatch.setattr(merge_pr.shutil, "which", lambda name: "/bin/claude")
    monkeypatch.setattr(merge_pr.subprocess, "run", fake_run)
    merge_pr.start_review()
    assert "ANTHROPIC_API_KEY" not in seen["claude"]
    for name in ("review_backtest.py", "merge_pr.py"):
        source = (Path(__file__).resolve().parents[1] / "scripts" / name).read_text()
        for call in source.split('["claude"')[1:]:
            assert "env=checkout.claude_env()" in call[:1000], f"{name} starts a claude that could bill a key"


def test_a_stopped_run_says_why():
    assert review_backtest.stopped_because('{"result": "Credit balance is too low"}') == "Credit balance is too low"
    assert review_backtest.stopped_because("not json") == "not json"


def test_tally_counts_hits_and_partials():
    known = [{"grade": "hit"}, {"grade": "partial"}, {"grade": "miss"}, {"grade": "hit"}]
    assert review_backtest.tally(known) == {"hit": 2, "partial": 1, "of": 4}


def test_every_case_says_how_blind_it_is_and_lists_what_counts():
    cases = review_backtest.load_cases()
    assert cases["scoring"]
    for name, case in cases["cases"].items():
        assert case["base"] and case["head"] and case["blindness"], name
        ids = [k["id"] for k in case["known"]]
        assert ids and len(ids) == len(set(ids)), name
