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
