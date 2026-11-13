"""scripts/merge_pr.py --auto: the merge queue, in throwaway repos with GitHub faked. A PR merges only on a clean, tested,
reviewed tree; decisions wait for the owner; a moved main is tested again; nothing is pulled under a running arm."""
import fcntl
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merge_pr  # noqa: E402
from mahjong import config
from test_merge_pr import branch_in_worktree, commit, git, make_repo

BODY = ("## What changed\nx\n## Files\ny\n## Overlaps\nnone\n## Unverified on the arm\nnothing moves\n"
        "## Review\n/code-review medium: 1 finding, fixed\n")


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


class GitHub:
    """The gh calls the queue makes, answered from a dict PR, recorded. Merging pushes the branch onto origin's main."""
    def __init__(self, repo, pr, merge_ok=True):
        self.repo, self.pr, self.merge_ok, self.calls = repo, pr, merge_ok, []

    def gh(self, *args):
        self.calls.append(args)
        if args[:2] == ("pr", "view"):
            return json.dumps(self.pr)
        if args[:2] == ("pr", "list"):
            return json.dumps([] if "--base" in args or self.pr["state"] != "OPEN" else
                              [{"number": self.pr["number"], "isDraft": self.pr["isDraft"], "createdAt": "2026-11-13T00:00:00Z"}])
        return ""

    def gh_try(self, *args):
        self.calls.append(args)
        if args[:2] == ("pr", "merge") and self.merge_ok:         # a merge commit on origin's main, as GitHub makes one
            scratch = self.repo.parent / "github_merge"
            git(self.repo, "fetch", "-q", "origin")
            git(self.repo, "worktree", "add", "-q", "--detach", str(scratch), "origin/main")
            git(scratch, "merge", "-q", "--no-ff", "-m", "Merge pull request", f"origin/{self.pr['headRefName']}")
            git(scratch, "push", "-q", "origin", "HEAD:main")
            git(self.repo, "worktree", "remove", "--force", str(scratch))
            self.pr["state"] = "MERGED"
            return True, ""
        return False, "head moved"

    def did(self, *prefix):
        return [c for c in self.calls if c[:len(prefix)] == prefix]


def queue(repo, monkeypatch, files=("mahjong/x.py",), body=BODY, labels=(), passes=True):
    monkeypatch.setattr(config, "REPO", repo)
    branch_in_worktree(repo, "feature", files[0], "X = 1\n")
    git(repo, "fetch", "-q", "origin")
    pr = {"number": 7, "title": "t", "state": "OPEN", "isDraft": False, "headRefName": "feature", "baseRefName": "main",
          "body": body, "files": [{"path": f} for f in files], "labels": [{"name": n} for n in labels]}
    hub = GitHub(repo, pr)
    monkeypatch.setattr(merge_pr, "gh", hub.gh)
    monkeypatch.setattr(merge_pr, "gh_try", hub.gh_try)
    tested = []
    monkeypatch.setattr(merge_pr, "verify_merged", lambda ref, slow=False, repo=None: (tested.append(ref), (passes, "tests: exit 1"))[1])
    monkeypatch.setattr(merge_pr, "test_command", lambda slow=False: ["true"])
    monkeypatch.setattr(merge_pr, "arm_scripts_running", lambda: [])
    monkeypatch.setattr(merge_pr, "start_review", lambda: hub.calls.append(("review",)))
    return hub, tested


def test_a_clean_tested_pr_merges_is_pulled_and_closed(repo, monkeypatch):
    hub, tested = queue(repo, monkeypatch)
    head = git(repo, "rev-parse", "origin/feature")
    assert merge_pr.auto_one(7, repo) == "merged"
    assert tested == ["origin/feature"] and hub.did("pr", "merge")[0][-1] == head          # only the head it tested
    assert git(repo, "rev-parse", "HEAD") == git(repo, "rev-parse", "origin/main")     # pulled here
    assert not (repo / ".claude" / "worktrees" / "feature").exists()                    # closed


@pytest.mark.parametrize("files, labels, why", [
    (("data/standard_setup.json",), (), "changes data/standard_setup.json"),
    (("assets/tile_detector.pt",), (), "changes assets/tile_detector.pt"),
    (("mahjong/x.py",), ("owner",), "labelled owner"),
])
def test_a_decision_waits_for_the_owner(repo, monkeypatch, files, labels, why):
    hub, tested = queue(repo, monkeypatch, files=files, labels=labels)
    assert merge_pr.auto_one(7, repo) == f"held: {why}"
    assert tested == [] and not hub.did("pr", "merge")
    assert any("labels[]=needs-owner" in c for c in hub.did("api"))


def test_a_change_to_the_default_policy_waits_but_registering_one_doesnt(repo, monkeypatch):
    commit(repo, "data/policies.json", json.dumps({"default": "a", "policies": {"a": {}}}), "registry")
    git(repo, "push", "-q", "origin", "main")
    hub, _ = queue(repo, monkeypatch, files=("data/policies.json",))
    tree = repo / ".claude" / "worktrees" / "feature"
    commit(tree, "data/policies.json", json.dumps({"default": "b", "policies": {"a": {}, "b": {}}}), "new default")
    git(tree, "push", "-q", "origin", "feature")
    git(repo, "fetch", "-q", "origin")
    assert merge_pr.auto_one(7, repo) == "held: changes the default policy"
    commit(tree, "data/policies.json", json.dumps({"default": "a", "policies": {"a": {}, "b": {}}}), "only registered")
    git(tree, "push", "-q", "origin", "feature")
    assert merge_pr.auto_one(7, repo) == "merged"


@pytest.mark.parametrize("files, body, why", [
    (("mahjong/x.py",), BODY.replace("## Overlaps\nnone\n", "## Overlaps\n<!-- the template's comment -->\n"), "Overlaps"),
    (("scripts/play.py",), BODY.replace("/code-review medium: 1 finding, fixed", "looked fine"), "/code-review"),
])
def test_an_unfinished_description_is_sent_back(repo, monkeypatch, files, body, why):
    hub, tested = queue(repo, monkeypatch, files=files, body=body)
    assert merge_pr.auto_one(7, repo).startswith("sent back")
    assert tested == [] and why in hub.did("pr", "comment")[0][-1] and hub.did("pr", "ready", "7", "--undo")


def test_failing_tests_on_the_merged_tree_send_it_back_with_the_output(repo, monkeypatch):
    hub, _ = queue(repo, monkeypatch, passes=False)
    assert merge_pr.auto_one(7, repo).startswith("sent back")
    assert "tests: exit 1" in hub.did("pr", "comment")[0][-1] and not hub.did("pr", "merge")


def test_main_moving_while_it_tests_is_tested_again(repo, monkeypatch):
    hub, tested = queue(repo, monkeypatch)
    calls = []

    def verify(ref, slow=False, repo_=None, **_):
        calls.append(git(repo, "rev-parse", "origin/main"))
        if len(calls) == 1:                         # another PR lands on main during the first run
            commit(repo, "mahjong/other.py", "Y = 1\n", "someone else's merge")
            git(repo, "push", "-q", "origin", "main")
        return True, ""
    monkeypatch.setattr(merge_pr, "verify_merged", lambda ref, slow=False, repo=None: verify(ref))
    assert merge_pr.auto_one(7, repo) == "merged"
    assert len(calls) == 2 and calls[0] != calls[1]


def test_nothing_is_pulled_while_an_arm_script_runs_here(repo, monkeypatch, capsys):
    hub, _ = queue(repo, monkeypatch)
    before = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(merge_pr, "arm_scripts_running", lambda: ["4242 python scripts/base_cam_collect.py"])
    assert merge_pr.auto_one(7, repo) == "merged"
    assert git(repo, "rev-parse", "HEAD") == before and "Not pulling main here yet" in capsys.readouterr().out


def test_a_pr_built_on_the_branch_is_moved_onto_main_before_the_branch_goes(repo, monkeypatch):
    hub, _ = queue(repo, monkeypatch)
    hub.gh = lambda *a: (hub.calls.append(a), json.dumps([{"number": 9}]) if a[:2] == ("pr", "list") else
                         json.dumps(hub.pr) if a[:2] == ("pr", "view") else "")[1]
    monkeypatch.setattr(merge_pr, "gh", hub.gh)
    assert merge_pr.auto_one(7, repo) == "merged"
    assert ("api", "-X", "PATCH", "repos/{owner}/{repo}/pulls/9", "-f", "base=main") in hub.calls


def test_only_one_queue_runs_at_a_time(repo):
    with merge_pr.queue_lock(repo):
        common = git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
        with open(f"{common}/merge_queue.lock", "w") as f, pytest.raises(BlockingIOError):
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_a_sim_run_isnt_an_arm_script_running(monkeypatch):
    import subprocess
    out = "11 python scripts/bench.py --sim --trials 3\n12 python scripts/play.py --layout even-hand13\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("R", (), {"stdout": out})())
    assert merge_pr.arm_scripts_running() == ["12 python scripts/play.py --layout even-hand13"]


def test_a_branch_that_conflicts_with_main_is_never_tested(repo, monkeypatch):
    monkeypatch.setattr(config, "REPO", repo)
    branch_in_worktree(repo, "feature", "mahjong/arm.py", "def move_to():\n    return 1\n")
    commit(repo, "mahjong/arm.py", "def move_to():\n    return 2\n", "main's own change")
    git(repo, "push", "-q", "origin", "main")
    git(repo, "fetch", "-q", "origin")
    passed, report = merge_pr.verify_merged("origin/feature", repo=repo)
    assert not passed and "doesn't merge cleanly" in report
    assert not (repo / ".claude" / "worktrees" / "_merge_check").exists()


def test_a_branch_pushed_while_queued_is_checked_again_from_the_start(repo, monkeypatch):
    """A head that moved after the hold and description checks could otherwise merge a decision file unchecked."""
    hub, tested = queue(repo, monkeypatch)
    hub.pr["headRefOid"] = "0" * 40                   # what GitHub said before a push
    views = []
    view = hub.gh

    def gh(*args):
        if args[:2] == ("pr", "view"):
            views.append(1)
            if len(views) > 1:
                hub.pr["headRefOid"] = git(repo, "rev-parse", "origin/feature")
        return view(*args)
    monkeypatch.setattr(merge_pr, "gh", gh)
    assert merge_pr.auto_one(7, repo) == "merged" and len(views) == 2 and len(tested) == 1


def test_github_refusing_for_another_reason_is_sent_back_with_its_words(repo, monkeypatch):
    hub, tested = queue(repo, monkeypatch)
    monkeypatch.setattr(merge_pr, "gh_try", lambda *a: (False, "GraphQL: Resource not accessible by integration"))
    assert merge_pr.auto_one(7, repo).startswith("sent back")
    assert len(tested) == 1 and "Resource not accessible" in hub.did("pr", "comment")[0][-1]


def test_a_held_pr_doesnt_hold_back_the_batch_review(repo, monkeypatch):
    hub, _ = queue(repo, monkeypatch)
    listing = [{"number": 7, "isDraft": False, "createdAt": "2026-11-13T00:00:00Z", "baseRefName": "main", "labels": []}]
    gh = hub.gh

    def gh_with_a_held_one(*args):
        if args[:2] == ("pr", "list"):
            open_now = [p for p in listing if hub.pr["state"] == "OPEN" or p["number"] != 7]
            return json.dumps(open_now + [{"number": 8, "isDraft": False, "createdAt": "2026-11-13T01:00:00Z", "baseRefName": "main",
                                           "labels": [{"name": "needs-owner"}]}])
        return gh(*args)
    monkeypatch.setattr(merge_pr, "gh", gh_with_a_held_one)
    def one(n, repo=None):
        if n == 7:
            hub.pr["state"] = "MERGED"
            return "merged"
        return "held: changes data/eval_split.json"
    monkeypatch.setattr(merge_pr, "auto_one", one)
    monkeypatch.setattr(merge_pr.checkout, "unreviewed_merges", lambda *a, **k: ["abc Merge pull request #7"])
    assert merge_pr.auto(repo=repo) == {7: "merged", 8: "held: changes data/eval_split.json"}
    assert ("review",) in hub.calls
