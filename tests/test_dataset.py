import json

import numpy as np

from mahjong import config, dataset, kinematics
from mahjong.dataset import idle_mask, is_pre_takeover_fix, keep_mask, narrowing_end, takeover_sag_end


def episode(n=200, narrow=True, sag=False):
    """A synthetic pickup: still, gripper 90 -> 21, still, descend, close, lift."""
    hover, _ = kinematics.hover_command(0.20, -0.14, 0.045, gripper=90.0)
    start = np.array([hover[j] for j in config.JOINTS])
    actions = np.tile(start, (n, 1))
    if narrow:
        actions[20:60, 5] = np.linspace(90, 21, 40)
        actions[60:, 5] = 21
    actions[100:140, 1] += np.linspace(0, 15, 40)        # descend
    actions[140:150, 5] = np.linspace(21, 0, 10)          # close
    actions[150:, 5] = 0
    actions[150:, 1] = actions[139, 1] - np.linspace(0, 20, n - 150)
    states = actions.copy()
    if sag:
        actions[0] = states[0]
        states[1:6, 1] += np.linspace(0.5, 1.5, 5)        # the takeover drop
        states[6:, 1] += 1.5
    else:
        actions[:, 1] -= 1.0                               # post-fix: the command leads the sagged state throughout
    return states, actions, actions - actions[0]


def test_idle_frames_are_the_still_stretches():
    _, actions, deltas = episode()
    idle = idle_mask(deltas)
    assert idle[:10].all() and idle[75:90].all()          # still at the hover, still after narrowing
    assert not idle[110:130].any()                         # descending


def test_narrowing_ends_where_the_gripper_reaches_its_pregrasp_opening():
    _, actions, _ = episode()
    assert 55 <= narrowing_end(actions) <= 60
    _, actions, _ = episode(narrow=False)
    assert narrowing_end(actions) == 0


def test_takeover_sag_only_in_pre_fix_episodes():
    states, actions, deltas = episode(sag=True)
    assert is_pre_takeover_fix(states, actions)
    assert 4 <= takeover_sag_end(states) <= 8
    states, actions, deltas = episode(sag=False)
    assert not is_pre_takeover_fix(states, actions)


def test_keep_mask_starts_after_narrowing_when_asked():
    states, actions, deltas = episode()
    keep = keep_mask(states, actions, deltas, trim_narrowing=True)
    assert np.flatnonzero(keep)[0] >= 55
    assert keep[110:130].all()


def write_log(root, name, rows):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_episode_log_reads_either_recorders_log(tmp_path, monkeypatch):
    import lerobot.utils.constants
    monkeypatch.setattr(lerobot.utils.constants, "HF_LEROBOT_HOME", tmp_path)
    write_log(tmp_path / "u/new", dataset.EPISODE_LOG, [{"episode_index": 0, "tile_name": "8D"}])
    write_log(tmp_path / "u/old", "episode_teleop_log.jsonl", [{"episode_index": 3, "tile_name": "4D"}])
    assert dataset.episode_log("u/new") == {0: {"episode_index": 0, "tile_name": "8D"}}
    assert dataset.episode_log("u/old")[3]["tile_name"] == "4D"
    assert dataset.episode_log("u/none") == {}


def test_hover_settings_are_the_recorded_hover_and_the_trimmed_start_gripper(tmp_path, monkeypatch):
    import lerobot.utils.constants
    monkeypatch.setattr(lerobot.utils.constants, "HF_LEROBOT_HOME", tmp_path)
    hover = {"offset_m": 0.05, "clearance_m": 0.035, "gripper": 90.0}
    write_log(tmp_path / "u/new", dataset.EPISODE_LOG, [{"episode_index": i, "hover": hover} for i in range(3)])
    assert dataset.hover_settings("u/new", {"start_gripper_median": 27.24}) == \
        {"offset_m": 0.05, "clearance_m": 0.035, "gripper": 27.2}
    # the pre-rebuild log has no hover: config defaults (how v2 and v2n were registered)
    write_log(tmp_path / "u/old", "episode_teleop_log.jsonl", [{"episode_index": 0, "hover_offset_m": 0.06}])
    assert dataset.hover_settings("u/old", {"start_gripper_median": 90.0}) == \
        {"offset_m": config.HOVER_OFFSET_M, "clearance_m": config.HOVER_CLEARANCE_M, "gripper": 90.0}
