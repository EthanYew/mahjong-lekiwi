import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def no_face_model(monkeypatch, request):
    """Base-camera photos are saved with faces blurred by a pose model (mahjong/privacy.py): tests that save them run without
    the weights and find no one. tests/test_privacy.py drives the real blur with its own finder, and one `slow` test the model."""
    if "slow" in request.keywords:
        return
    from mahjong import privacy
    monkeypatch.setattr(privacy, "find_faces", lambda image: [])
    monkeypatch.setattr(privacy, "require", lambda: None)


@pytest.fixture(autouse=True)
def no_holdup_review(monkeypatch, tmp_path_factory):
    """Fake hold-up sessions reuse real session names, so a person's real verdicts (data/holdup_review.jsonl) would apply
    to them: every test starts with an empty review of its own."""
    from mahjong import base_cam
    monkeypatch.setattr(base_cam, "REVIEW", tmp_path_factory.mktemp("review") / "holdup_review.jsonl")


@pytest.fixture(autouse=True)
def motion_watchdog(request):
    """Every command a simulated arm was sent in this test, checked against the motion invariants (mahjong/motion_watch.py):
    any end-to-end sim run that moves the arm wrongly fails, whatever it was testing. A test that sends a bad command on purpose
    is marked motion_unwatched. MOTION_WATCH_REPORT=<file> records instead of failing (for calibrating the thresholds)."""
    import os
    from mahjong.sim import SimRobot
    SimRobot.live.clear()
    SimRobot.watching = True
    yield
    SimRobot.watching = False
    found = [(kind, detail, i) for robot in SimRobot.live for kind, detail, i in robot.violations]
    SimRobot.live.clear()
    if not found or "motion_unwatched" in request.keywords:
        return
    if (report := os.environ.get("MOTION_WATCH_REPORT")):
        import json
        with open(report, "a") as f:
            f.write(json.dumps({"test": request.node.nodeid, "found": found[:20], "n": len(found)}) + "\n")
        return
    pytest.fail("the simulated arm broke a motion invariant (mahjong/motion_watch.py):\n"
                + "\n".join(f"  command {i}: {kind}: {detail}" for kind, detail, i in found[:10])
                + (f"\n  ... {len(found) - 10} more" if len(found) > 10 else ""), pytrace=False)
