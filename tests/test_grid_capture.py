"""grid_capture.py (the arm photographs a checkerboard from the wrist at spread-out poses) and mahjong/wrist_grid.py, with fakes."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import config, wrist_grid
from mahjong.session import read_events


def board_image(squares=(10, 7), px=40, margin=60):
    """A checkerboard of `squares` squares, drawn on white."""
    h, w = squares[1] * px + 2 * margin, squares[0] * px + 2 * margin
    image = np.full((h, w, 3), 255, np.uint8)
    for r in range(squares[1]):
        for c in range(squares[0]):
            if (r + c) % 2 == 0:
                image[margin + r * px: margin + (r + 1) * px, margin + c * px: margin + (c + 1) * px] = 0
    return image


def test_the_board_is_found_whole_and_not_when_cut_off():
    image = board_image()
    corners = wrist_grid.find_grid(image)
    assert corners is not None and corners.shape == (54, 2)
    assert wrist_grid.find_grid(image[:, : image.shape[1] // 2]) is None      # half the board is no board


def test_the_poses_spread_over_every_height_and_roll_and_the_disc():
    import grid_capture
    poses = grid_capture.plan_poses(24, 0.03, [0.06, 0.08, 0.10, 0.12], 25.0)
    assert len(poses) == 24 and poses[0] == (0.0, 0.0, 0.06, 0.0)
    pairs = {(round(c, 3), round(r)) for _, _, c, r in poses[1:]}
    assert len(pairs) == 12                                                   # each of 4 heights meets each of 3 rolls
    offsets = np.array([(dx, dy) for dx, dy, _, _ in poses])
    assert np.hypot(*offsets.T).max() <= 0.03 + 1e-9
    assert offsets[:, 0].min() < -0.01 and offsets[:, 0].max() > 0.01 and offsets[:, 1].min() < -0.01 and offsets[:, 1].max() > 0.01


def test_sim_run_saves_a_pose_event_each_and_ends_at_rest(tmp_path, monkeypatch, capsys):
    import grid_capture
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    out = grid_capture.main(["--sim", "--poses", "6", "--frames", "1"])
    events = read_events(out)
    assert next(e for e in events if e["type"] == "grid_located")["source"] == "sim"
    poses = [e for e in events if e["type"] == "grid_pose"]
    assert len(poses) == 6
    assert all(set(e["measured"]) == set(config.JOINTS) for e in poses if "skipped" not in e)
    assert all(p.suffix == ".jpg" for p in (out / "frames_partial").iterdir())     # the sim's blank view never shows a board
    assert "fewer than 15" in capsys.readouterr().out
    assert any(e["type"] == "grid_capture_done" for e in events)


def test_a_malformed_board_position_stops_before_the_arm_moves(tmp_path, monkeypatch):
    import grid_capture
    import pytest
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    with pytest.raises((ValueError, IndexError)):
        grid_capture.main(["--sim", "--grid-xy", "0.17"])
    assert not list(tmp_path.iterdir())


def test_a_hover_lower_than_the_policies_is_refused_before_the_arm_moves(tmp_path, monkeypatch):
    import grid_capture
    import pytest
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    for value in ("10,20", "-5", "60,0"):
        with pytest.raises(SystemExit):
            grid_capture.main(["--sim", "--clearances-mm", value])
    assert not list(tmp_path.iterdir())


def test_an_unreachable_board_is_reported_as_skipped_not_as_not_seen(tmp_path, monkeypatch, capsys):
    import grid_capture
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    out = grid_capture.main(["--sim", "--poses", "3", "--grid-xy", "0.9,0.9"])
    assert all(e.get("skipped") for e in read_events(out) if e["type"] == "grid_pose")
    printed = capsys.readouterr().out
    assert "3 out of reach" in printed and "Frames in" not in printed
