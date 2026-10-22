"""Policy loading and behaviour. Slow: loads policies from the local HF cache and runs on the GPU.

    pytest -m slow
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import pytest
import torch

from mahjong import config

pytestmark = pytest.mark.slow
FREEZE = config.DATA / "fixtures" / "freeze_frames"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@pytest.fixture(scope="module")
def v2():
    from mahjong.policy import Policy
    return Policy("pickup_mat_cl")


@pytest.fixture(scope="module")
def v1():
    from mahjong.policy import Policy
    return Policy("pickup_mat_cl_v1")


@pytest.fixture(scope="module")
def dataset(v2):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(v2.spec["dataset"])


def bgr(item):
    return (item["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)[:, :, ::-1]


def test_processors_match_mahjongbots_manual_normalisation(v2, dataset):
    """The saved processors normalise exactly as MahjongBot did by hand (ImageNet image stats,
    dataset state/action stats) — so switching to them changes nothing for existing policies."""
    stats = dataset.meta.stats
    s_mean, s_std = (torch.tensor(stats["observation.state"][k], dtype=torch.float32) for k in ("mean", "std"))
    a_mean, a_std = (np.array(stats["action"][k], dtype=np.float32) for k in ("mean", "std"))
    for i in (0, len(dataset) // 2, len(dataset) - 1):
        item = dataset[i]
        frame = bgr(item)
        state = item["observation.state"].numpy()
        image = torch.from_numpy(np.ascontiguousarray(frame[:, :, ::-1])).permute(2, 0, 1).float() / 255.0
        manual = {"observation.images.wrist": ((image - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).cuda(),
                  "observation.state": ((torch.tensor(state) - s_mean) / s_std).unsqueeze(0).cuda()}
        with torch.no_grad():
            old = v2.act.predict_action_chunk(manual).squeeze(0).cpu().numpy() * a_std + a_mean
        # 0.002 deg at most: float32 rounding and a no-op 224->224 resize
        assert np.abs(v2.plan(frame, state) - old).max() < 0.01


def test_replay_follows_recorded_actions(v2, dataset):
    for ep in (10, 50, 95):
        meta = dataset.meta.episodes[ep]
        v2.reset()
        errors, still = [], []
        for i in range(int(meta["dataset_from_index"]), int(meta["dataset_to_index"])):
            item = dataset[i]
            predicted = v2.step(bgr(item), item["observation.state"].numpy())
            errors.append(np.abs(predicted - item["action"].numpy()).mean())
            still.append(np.abs(item["action"].numpy()).mean())
        assert np.mean(errors) < 1.0 < 10.0 < np.mean(still)


def descent_onset(policy, frame_name, gripper_delta):
    plan = policy.plan(cv2.imread(str(FREEZE / frame_name)), np.array([0, 0, 0, 0, 0, gripper_delta], np.float32))
    moved = np.abs(plan[:, 1:3] - plan[0, 1:3]).max(axis=1)
    return next((k for k in range(len(plan)) if moved[k] > 2.0), None), plan[-1, 1] - plan[0, 1]


@pytest.mark.parametrize("attempt", ["180956_8D", "181050_8D", "181145_8D"])
def test_v2_plans_the_descent_where_v1_froze(v1, v2, attempt):
    """Frames from the 2026-10-22 attempts v1 froze on, after the gripper opened: v1 plans to wait
    ~3s (or forever); v2 plans the descent within ~2s."""
    v1_onset, _ = descent_onset(v1, f"{attempt}_step0300.jpg", -31.0)
    v2_onset, v2_lift = descent_onset(v2, f"{attempt}_step0300.jpg", -31.0)
    assert v1_onset is None or v1_onset >= 85
    assert v2_onset is not None and v2_onset <= 65 and v2_lift > 3.0
