"""Pickup phases and the hover→grasp check (mahjong/quality.py) on synthetic demos built with the IK."""
import numpy as np
import pytest

from mahjong import config, kinematics, quality

FPS = 30
X, Y = 0.20, 0.0


def pose(x, y, offset, gripper):
    command, _ = kinematics.hover_command(x, y, offset, gripper)
    return np.array([command[j] for j in config.JOINTS])


def ramp(a, b, n):
    """n frames from pose a to pose b (joint space: straight enough over a few cm), b included."""
    return [a + (b - a) * k / n for k in range(1, n + 1)]


def demo(steer_y_m=0.010, pause_s=1.0, jitter_m=0.0):
    """Narrow at the hover, hold still `pause_s`, steer `steer_y_m` in y, descend, close, hold."""
    hx, hy = X, Y + jitter_m
    wide, hover = pose(hx, hy, 0.045, 90), pose(hx, hy, 0.045, 27)
    steered, low = pose(hx, hy + steer_y_m, 0.045, 27), pose(hx, hy + steer_y_m, 0.010, 27)
    shut = low.copy(); shut[5] = 0
    frames = [wide] + ramp(wide, hover, 29) + [hover] * int(pause_s * FPS)
    frames += ramp(hover, steered, 30) + ramp(steered, low, 45) + [shut] * 60
    return np.array(frames)


def test_phases_time_the_wait_the_approach_and_its_detour():
    p = quality.phases(demo(pause_s=1.0), FPS)
    assert p["narrow_s"] == pytest.approx(1.0, abs=0.1)
    assert p["wait_s"] == pytest.approx(1.0, abs=0.1)          # still at the hover before the first move
    assert p["approach_s"] == pytest.approx(2.5, abs=0.15)     # steer 1 s + descend 1.5 s
    assert p["pause_s"] == 0 and p["approach_rev"] == 0
    assert 1.05 < p["path_ratio"] < 1.5                        # steer then descend: an L, not the diagonal
    assert p["after_s"] == pytest.approx(2.0, abs=0.1) and p["wobble_mm"] < 0.2


def test_hover_move_is_the_steer_with_a_deliberate_offset_added_back():
    assert quality.hover_move(demo(steer_y_m=0.010)) == pytest.approx([0, 10], abs=0.5)
    # hover moved +6 mm on purpose, operator steered 6 mm less to the same tile: same miss of the hover
    assert quality.hover_move(demo(steer_y_m=0.004, jitter_m=0.006), (0.0, 0.006)) == pytest.approx([0, 10], abs=0.5)
    assert quality.hover_move(demo()[:50]) is None                                  # never closed


def test_drift_compares_only_demos_that_steered():
    """2026-10-26: a third of v2's demos went straight down, so medians that counted them read as a drift."""
    record = {"detection_xy": [X, Y], "spot": "F3"}
    today = {0: (demo(0.010), record), 1: (demo(0.012), record)}
    reference = {i: (demo(0.0), {"detection_xy": [X, Y]}) for i in range(3)}
    reference.update({i: (demo(0.011), {"detection_xy": [X + 0.005, Y]}) for i in range(3, 6)})
    row = quality.drift_table(today, reference).loc["F3"]
    assert row["reference"] == 6 and row["reference_straight_down"] == pytest.approx(0.5)
    assert row["shift_y"] == pytest.approx(0.0, abs=1.0)                   # the steered ones agree
    lonely = quality.drift_table(today, {0: (demo(0.0), {"detection_xy": [0.15, 0.15]})}).loc["F3"]
    assert np.isnan(lonely["shift_y"]) and lonely["reference"] == 0


def test_dropped_episodes_are_kept_per_dataset_with_their_reason(tmp_path):
    from mahjong.quality import drop_episodes, dropped_episodes
    path = tmp_path / "dropped.json"
    assert dropped_episodes("a/b", path) == {}
    drop_episodes("a/b", [39, 8], "searching", path)
    drop_episodes("c/d", [1], "other", path)
    assert dropped_episodes("a/b", path) == {8: "searching", 39: "searching"}
    assert drop_episodes("a/b", [8, 2], "again", path) == {2: "again", 8: "again", 39: "searching"}
    assert dropped_episodes("c/d", path) == {1: "other"}


def test_no_dataset_lists_every_dropped_episode_with_its_reason(tmp_path, monkeypatch, capsys):
    import quality_report
    path = tmp_path / "dropped.json"
    monkeypatch.setattr(quality, "DROPPED_FILE", path)
    monkeypatch.setattr("sys.argv", ["quality_report.py"])
    quality_report.main()
    assert "No episodes" in capsys.readouterr().out
    quality.drop_episodes("a/b", [39, 8], "searching")
    quality.drop_episodes("a/b", [13], "closed high")
    quality.drop_episodes("c/d", [1], "other")
    quality.drop_episodes("e/f", [2], "undone")
    quality.undrop_episodes("e/f", [2])
    quality_report.main()
    out = capsys.readouterr().out
    assert "4 episodes in 2 datasets" in out and "e/f" not in out
    assert "a/b: 3\n  8, 39: searching\n  13: closed high" in out and "c/d: 1\n  1: other" in out
    monkeypatch.setattr("sys.argv", ["quality_report.py", "--drop", "3"])
    with pytest.raises(SystemExit):                          # never drops without saying from what
        quality_report.main()
    assert "--dataset is required with --drop" in capsys.readouterr().err


def test_gripper_fixes_and_undropping(tmp_path):
    from mahjong.quality import drop_episodes, fix_gripper, gripper_fixes, undrop_episodes
    fixes, drops = tmp_path / "fixes.json", tmp_path / "dropped.json"
    drop_episodes("a/b", [75, 86], "loose grip", drops)
    assert fix_gripper("a/b", [75], "stopped at 5.5%", fixes) == {75: "stopped at 5.5%"}
    assert undrop_episodes("a/b", [75], drops) == {86: "loose grip"} and gripper_fixes("a/b", fixes) == {75: "stopped at 5.5%"}


# --- grip reach: demos that closed short of the tile's middle (2026-11-23) ---

def closing_at(x, y, frames=20, open_frames=10):
    """Open at a pose low over (x, y), then shut there."""
    low = pose(x, y, 0.010, 27)
    shut = low.copy(); shut[5] = 0
    return np.array([low] * open_frames + [shut] * frames)


def closed_x(x, y):
    q = closing_at(x, y)[-1]
    return float(kinematics.tool_point(dict(zip(config.JOINTS, q)))[0])


def labels(reach=40.0, n=4):
    """Unmarked bench attempts with the same reach over a grid of tiles, and one marked low 6 mm short."""
    rows = [{"session": "s", "attempt_dir": f"{i}", "policy": "p", "spot": "S1", "detection_xy": [x, y],
             "reach_mm": reach, "low": False} for i, (x, y) in enumerate((x, y) for x in np.linspace(0.14, 0.19, n)
                                                                           for y in np.linspace(-0.1, 0.1, n))]
    return rows + [{**rows[0], "attempt_dir": "low", "reach_mm": reach - 6, "low": True}]


def test_episode_reach_is_the_close_less_the_tile_and_a_dagger_one_counts_from_the_takeover():
    det = [0.17, 0.02]
    joints = closing_at(0.18, 0.02)
    reach, height = quality.episode_reach(joints, joints, {"detection_xy": det})
    assert reach == pytest.approx((closed_x(0.18, 0.02) - det[0]) * 1000, abs=0.01) and height is not None
    # the policy shut at x 0.16, the person took over at frame 30 and closed at 0.18: the person's close is what's trained on
    dagger = np.vstack([closing_at(0.16, 0.02), closing_at(0.18, 0.02)])
    record = {"detection_xy": det, "policy": "v13", "takeover_frame": 30}
    assert quality.episode_reach(dagger, dagger, record)[0] == pytest.approx(reach, abs=0.01)
    assert quality.episode_reach(dagger, dagger, {**record, "takeover_frame": None}) == (None, None)   # never trained on
    assert quality.episode_reach(joints, joints, {}) == (None, None)                                    # no detection
    assert quality.episode_reach(joints[:10], joints[:10], {"detection_xy": det}) == (None, None)       # never shut


def test_reach_rows_propose_only_judged_training_episodes_short_enough():
    ref = quality.ReachReference(labels())
    assert ref.expected([0.16, 0.0]) == pytest.approx(40.0) and ref.marked_low == 1
    row = lambda reach, **rec: quality.reach_row(1, {"detection_xy": [0.16, 0.0], **rec}, reach, 5.0, ref)
    assert row(35.0)["propose"] and row(35.0)["basis"] == "measured"
    assert row(35.0)["shortfall_mm"] == pytest.approx(-5.0)
    assert not row(37.0)["propose"]                                                   # 3 mm short: within the cutoff
    spot_only = {"placement_status": "saved_position"}
    assert not row(35.0, **spot_only)["propose"] and row(33.0, **spot_only)["propose"]   # 6 mm where it's the spot, not the tile
    far = quality.reach_row(1, {"detection_xy": [0.30, 0.0]}, 20.0, 5.0, ref)
    assert far["basis"] == "outside" and not far["propose"]                           # no bench attempts there to judge by
    assert not row(30.0, slice="familiar")["propose"]                                 # an eval slice
    assert not quality.reach_row(1, {"detection_xy": [0.16, 0.0]}, 30.0, 5.0, ref, excluded={1})["propose"]   # held out
    assert not quality.reach_row(1, {"detection_xy": [0.16, 0.0]}, 30.0, 5.0, ref, dropped={1})["propose"]
    assert quality.reach_row(1, {}, None, None, ref)["basis"] == "no close"


def test_reach_calibration_counts_marked_low_caught_and_unmarked_flagged():
    ref = quality.ReachReference(labels())
    assert quality.reach_calibration(labels(), ref, (4, 7)) == [(4, 1.0, 0.0), (7, 0.0, 0.0)]


def test_thin_spots_warn_when_drops_empty_a_spot():
    import pandas as pd
    t = pd.DataFrame({"spot": ["S8", "S8", "S8", "S1", "S1", "S1", "S1", ""],
                      "propose": [True, True, False, True, False, False, False, True], "dropped": False, "heldout": False,
                      "slice": "train"})
    assert quality.thin_spots(t) == {"S8": (3, 1)}     # S1 keeps 3 of 4; unnamed spots aren't grouped
    t.loc[4, "dropped"] = True
    assert quality.thin_spots(t) == {"S1": (3, 2), "S8": (3, 1)}   # now 2 left: under 3


def test_bench_reach_labels_read_marked_sessions_only(tmp_path):
    import json
    def session(name, events):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    attempt = {"type": "attempt", "policy": "v13", "spot": "S1", "detection_xy": [0.15, 0.0], "grasp_xy": [0.19, 0.0]}
    marked = [{"type": "session_start", "sim": False}, {**attempt, "attempt_dir": "001"}, {**attempt, "attempt_dir": "002"},
              {"type": "keys", "attempt_dir": "002", "presses": [{"key": "d", "kind": "grip_low"}]},
              {**attempt, "attempt_dir": "003", "face_down": True}, {**attempt, "attempt_dir": "004", "person_grasp_xy": [0.2, 0]}]
    session("20261120_100000_bench", marked)
    session("20261120_110000_bench", marked[:3])                                  # nothing marked: unmarked means nothing
    session("20261120_120000_bench", [{**marked[0], "sim": True}] + marked[1:])   # sim
    session("20261101_100000_bench", marked)                                      # before marking was close
    session(next(iter(quality.LONE_TILE_SESSIONS)), marked)                       # a lone tile grips elsewhere
    rows = quality.bench_reach_labels(tmp_path)
    assert [(r["session"], r["attempt_dir"], r["low"]) for r in rows] == [
        ("20261120_100000_bench", "001", False), ("20261120_100000_bench", "002", True)]
    assert rows[0]["reach_mm"] == pytest.approx(40.0)


def test_grip_reach_review_fixes(tmp_path, monkeypatch):
    """A recording that isn't on disk is skipped rather than crashing; marks and attempts need an attempt_dir; eval-slice
    episodes don't count toward a spot's training episodes."""
    import json
    import pandas as pd
    monkeypatch.setattr(quality, "dataset_root", lambda repo: tmp_path / "nothing")
    with pytest.raises(SystemExit):
        quality.grip_reach_table("a/missing", quality.ReachReference(labels()))
    folder = tmp_path / "20261120_100000_bench"
    folder.mkdir()
    attempt = {"type": "attempt", "policy": "v13", "spot": "S1", "detection_xy": [0.15, 0.0], "grasp_xy": [0.19, 0.0]}
    events = [{"type": "keys", "presses": [{"key": "d", "kind": "grip_low"}]}, {**attempt, "attempt_dir": "001"},
              dict(attempt), {"type": "keys", "attempt_dir": "001", "presses": [{"key": "d", "kind": "grip_low"}]}]
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    assert [(r["attempt_dir"], r["low"]) for r in quality.bench_reach_labels(tmp_path)] == [("001", True)]
    t = pd.DataFrame({"spot": ["S1"] * 4, "slice": ["train", "train", "familiar", "familiar"],
                      "propose": [True, False, False, False], "dropped": False, "heldout": False})
    assert quality.thin_spots(t) == {"S1": (2, 1)}
