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
