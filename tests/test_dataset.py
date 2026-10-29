import pytest
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


def test_finish_close_carries_a_short_close_on_to_zero_at_its_own_rate():
    from mahjong.dataset import finish_close
    grip = [27.2, 27.2, 25.2, 23.2, 21.2, 19.2, 17.2, 15.2, 13.2, 11.2, 9.2, 7.2, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
    fixed = finish_close(grip)
    assert list(fixed[:13]) == grip[:13]                      # untouched up to where the operator stopped
    assert fixed[13] == pytest.approx(3.5) and fixed[14] == pytest.approx(1.5) and list(fixed[15:]) == [0.0] * 4
    assert list(finish_close([27.2, 10.0, 0.0, 0.0])) == [27.2, 10.0, 0.0, 0.0]   # closed fully: unchanged
    opened = finish_close([27.2, 15.0, 7.0, 7.0, 30.0, 30.0])
    assert list(opened[4:]) == [30.0, 30.0]                   # a later release is left alone


def test_keep_mask_drops_everything_before_a_dagger_takeover():
    states, actions, deltas = episode()
    keep = keep_mask(states, actions, deltas, trim_idle=False, start=120)
    assert not keep[:120].any() and keep[120:].all()


def test_idle_right_after_a_dagger_takeover_is_trimmed():
    # The policy drifts until the takeover at 80, then the person holds still until the descent. The idle
    # window mustn't reach back into the policy's dropped frames and keep the still ones (146 frames in v4).
    states, actions, deltas = episode()
    deltas[60:80, 0] = np.linspace(-5, 0, 20)
    keep = keep_mask(states, actions, deltas, trim_sag=False, start=80)
    assert not keep[:95].any()
    assert keep[110:130].all()


def wrist_frame():
    """A synthetic wrist frame (RGB): black table top with the lamp's brown glare, a white face with a red print,
    a green tile back and a cyan jaw."""
    frame = np.full((224, 224, 3), (25, 22, 20), np.uint8)
    frame[:, 120:] = (150, 95, 50)                       # glare on the mat
    frame[60:140, 60:110] = (240, 240, 235)              # a face...
    frame[90:100, 80:90] = (200, 30, 40)                 # ...with its print
    frame[140:150, 60:110] = (60, 150, 80)               # the tile's green back
    frame[170:224, 10:40] = (40, 200, 210)               # a jaw
    return frame


def test_a_surface_swap_replaces_the_mat_and_its_glare_but_not_tiles_or_jaws():
    frame = wrist_frame()
    kind, texture = dataset.random_surface(np.random.default_rng(1), kind="grid")
    out = dataset.replace_surface(frame, texture)
    assert kind == "grid" and out.shape == frame.shape
    for rows, cols in ((slice(60, 140), slice(60, 110)), (slice(140, 150), slice(60, 110)), (slice(170, 224), slice(10, 40))):
        assert (out[rows, cols] == frame[rows, cols]).all()             # face (print too), back, jaw
    assert (out[10:50, 5:50] != frame[10:50, 5:50]).any(axis=2).mean() > 0.9     # the black table top
    assert (out[10:50, 150:220] != frame[10:50, 150:220]).any(axis=2).mean() > 0.9   # the glare on it


def test_episode_surfaces_are_repeatable_and_about_the_asked_fraction():
    kinds = [dataset.episode_surface("Yewy/x", ep, fraction=0.7)[0] for ep in range(200)]
    assert kinds == [dataset.episode_surface("Yewy/x", ep, fraction=0.7)[0] for ep in range(200)]
    assert 0.6 < sum(k is not None for k in kinds) / 200 < 0.8
    assert set(k for k in kinds if k) == set(dataset.SURFACES)
    assert dataset.episode_surface("Yewy/x", 3, fraction=0.0) == (None, None)


def test_augment_turns_on_colour_jitter_without_moving_the_image():
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("make_policy", config.REPO / "scripts" / "make_policy.py")
    make_policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_policy)
    n = {"dataset": "Yewy/d", "policy_repo": "Yewy/p", "output_dir": "out", "job": "j"}
    assert not any("image_transforms" in a for a in make_policy.train_command(n, 1000, False))
    command = make_policy.train_command(n, 1000, False, augment=True)
    assert "--dataset.image_transforms.enable=true" in command
    tfs = json.loads(next(a for a in command if a.startswith("--dataset.image_transforms.tfs=")).split("=", 1)[1])
    assert "affine" not in tfs and {"brightness", "contrast", "hue"} <= set(tfs)
