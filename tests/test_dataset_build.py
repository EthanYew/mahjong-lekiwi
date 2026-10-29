"""dataset.build on real (tiny) LeRobot datasets: a recorded one plus a DAgger one added after it, as
make_policy.py --add does. Runs in a subprocess: lerobot reads HF_LEROBOT_HOME when it's imported."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from mahjong import config, dataset

NAMES = [f"{j}.pos" for j in config.JOINTS]
FEATURES = {k: {"dtype": "float32", "shape": (6,), "names": NAMES}
            for k in ("observation.state", "action", "observation.state_raw", "action_raw")}
FEATURES["observation.images.wrist"] = {"dtype": "video", "shape": (48, 64, 3), "names": ["h", "w", "c"]}
HOVER = np.array([0, 10, 20, 30, 40, 27], np.float32)


def record(repo, episodes, rows):
    ds = LeRobotDataset.create(repo_id=repo, fps=config.FPS, features=FEATURES, robot_type="so101_follower",
                               use_videos=True)
    for n in episodes:
        for i in range(n):            # every frame moves: nothing is idle
            state = HOVER + np.float32(i)
            ds.add_frame({"observation.state": state - HOVER, "action": state + 1 - HOVER,
                          "observation.state_raw": state, "action_raw": state + 1,
                          "observation.images.wrist": np.full((48, 64, 3), i, np.uint8), "task": "pick"})
        ds.save_episode()
    ds.finalize()
    (HF_LEROBOT_HOME / repo / dataset.EPISODE_LOG).write_text("".join(json.dumps(r) + "\n" for r in rows))


record("t/demos", [20], [{"episode_index": 0}])
record("t/dagger", [20, 20], [{"episode_index": 0, "policy": "p", "takeover_frame": 8},
                              {"episode_index": 1, "policy": "p", "takeover_frame": None}])
summary = dataset.build("t/demos", "t/built", trim_idle=False, size=(24, 32), extra=[("t/dagger", None, ())],
                        log=lambda *a: None)
record("t/green", [20], [{"episode_index": 0, "conditions": {"surface": "green cutting mat"}}])
swapped = dataset.build("t/demos", "t/swapped", trim_idle=False, size=(24, 32), extra=[("t/green", None, ())],
                        replace_mat=1.0, log=lambda *a: None)
built = LeRobotDataset("t/built")
h = built.hf_dataset
ep = np.array(h["episode_index"])
state, raw = np.stack(h["observation.state"]), np.stack(h["observation.state_raw"])
print(json.dumps({"summary": summary, "lengths": [int((ep == e).sum()) for e in range(built.num_episodes)],
                  "first_state": [state[ep == e][0].tolist() for e in range(built.num_episodes)],
                  "first_raw": [raw[ep == e][0].tolist() for e in range(built.num_episodes)],
                  "swapped_surfaces": swapped["surfaces"]}))
'''


def test_a_dagger_episode_keeps_its_frames_from_the_takeover_as_deltas_from_the_hover(tmp_path):
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT)], capture_output=True, text=True, cwd=tmp_path,
                         env={**os.environ, "HF_LEROBOT_HOME": str(tmp_path / "lerobot")}, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    summary = result["summary"]
    assert summary["sources"] == [["t/demos", 0], ["t/dagger", 0]]      # the one with no takeover left out
    assert summary["source_episodes"] == [0] and summary["dagger_episodes"] == [1]
    assert result["lengths"] == [20, 12]
    assert result["first_state"][0] == [0.0] * 6                        # a demo starts at a zero delta
    assert result["first_state"][1] == [8.0] * 6                        # DAgger: 8 frames on from the hover
    assert result["first_raw"][1] == [8, 18, 28, 38, 48, 35]
    assert summary["surfaces"] == [None, None]                          # replace_mat off
    black, green = result["swapped_surfaces"]                           # replace_mat=1.0: every black-top episode,
    assert black is not None and green is None                          # never one recorded on another surface
