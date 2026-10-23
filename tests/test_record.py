"""record.py end to end on simulated hardware: every saved episode gets its episode-log row, with the
hover jitter it was recorded from. (2026-10-23: the row was built after the offset had been cleared, so
every save crashed right after save_episode(), and no test ran the save path.)"""
import argparse
import json
import math
import random
import sys
import types
from pathlib import Path

import pytest

from mahjong import config, hardware
from mahjong.hover_correction import parse_offsets
from mahjong.session import read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_next_offset_cycles_a_list_by_saved_episode_and_is_seeded():
    import record
    fixed = argparse.Namespace(hover_offsets_mm=parse_offsets("6x,-6y"), hover_jitter_mm=0.0, jitter_zero_fraction=0.2)
    assert [record.next_offset(fixed, None, n) for n in range(3)] == [(0.006, 0.0), (0.0, -0.006), (0.006, 0.0)]
    jitter = argparse.Namespace(hover_offsets_mm=None, hover_jitter_mm=8.0, jitter_zero_fraction=0.0)
    draws = [[record.next_offset(jitter, random.Random(7), n) for n in range(3)] for _ in range(2)]
    assert draws[0] == draws[1] and all(0 < math.hypot(*o) <= 0.008 for o in draws[0])
    off = argparse.Namespace(hover_offsets_mm=None, hover_jitter_mm=0.0, jitter_zero_fraction=0.2)
    assert record.next_offset(off, random.Random(0), 0) == (0.0, 0.0)


class FakeDataset:
    def __init__(self, root):
        self.root, self.meta, self.frames, self.saved = root, types.SimpleNamespace(total_episodes=0), [], []

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved.append(len(self.frames))
        self.frames = []
        self.meta.total_episodes += 1

    def clear_episode_buffer(self):
        self.frames = []

    def has_pending_frames(self):
        return bool(self.frames)

    def finalize(self):
        pass


class FakeGamepad:
    name = "fake gamepad"

    def start(self, arm):
        pass

    def step(self, arm, dt):
        return arm.commanded()

    def close(self):
        pass


def run_record(tmp_path, monkeypatch, argv, script):
    """record.main(argv) on the sim arm, camera and mat, picking 8D each time. `script`: what the
    keyboard does, in order — ("hover", keys) when the hover is shown, ("rec", keys) after 3 recorded
    frames; keys beyond → (exit_early) e.g. {"rerecord_episode": True} for ←. Returns (dataset, session dir)."""
    import builtins
    import record
    from mahjong.arm import Arm
    from mahjong.sim import SimCamera, SimPerception, SimRobot

    def connect(needed, gamepad_kwargs=None):
        mat = SimPerception()
        arm = Arm(SimRobot(on_release=mat.release), sleep=lambda s: None)
        arm.at_connect = {"arm_at_rest": True, "off_rest_deg": 0.0}
        return {"arm": arm, "wrist_camera": SimCamera(), "realsense": mat, "gamepad": FakeGamepad()}

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    listener = types.SimpleNamespace(stop=lambda: None)
    steps, frames = iter(script), []

    def show(camera, disabled, text, frame=None):
        phase = "hover" if text.startswith("hovering") else "rec"
        if phase == "rec":
            frames.append(text)
            if len(frames) < 3:
                return
            frames.clear()
        expected, keys = next(steps)
        assert expected == phase
        events.update(exit_early=True, **keys)

    dataset = FakeDataset(tmp_path / "dataset")
    dataset.root.mkdir()
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(hardware, "connect", connect)
    monkeypatch.setattr(record, "open_dataset", lambda repo_id: dataset)
    monkeypatch.setattr(record, "show", show)
    monkeypatch.setitem(sys.modules, "lerobot.common.control_utils",
                        types.SimpleNamespace(init_keyboard_listener=lambda: (listener, events)))
    monkeypatch.setattr(builtins, "input", lambda prompt="": "8D" if "Which tile" in prompt else "")
    record.main(["--dataset", "test/record", "--episodes", "2", *argv])
    (session,) = (tmp_path / "sessions").iterdir()
    return dataset, session


def episode_rows(dataset):
    return [json.loads(line) for line in (dataset.root / "episode_log.jsonl").read_text().splitlines()]


def test_each_saved_episode_logs_the_fixed_offset_it_was_recorded_from(tmp_path, monkeypatch):
    script = [("hover", {}), ("rec", {}), ("hover", {}), ("rec", {})]
    dataset, session = run_record(tmp_path, monkeypatch, ["--slice", "jitter", "--hover-offsets-mm", "6x,-6y"], script)
    rows = episode_rows(dataset)
    assert [r["episode_index"] for r in rows] == [0, 1] and dataset.saved == [3, 3]
    assert [r["hover_jitter_xy"] for r in rows] == [[0.006, 0.0], [0.0, -0.006]]
    for r in rows:
        assert r["slice"] == "jitter" and r["tile_name"] == "8D"
        assert r["hover_xy"] == pytest.approx([u + j for u, j in zip(r["hover_unjittered_xy"], r["hover_jitter_xy"])])
    events = read_events(session)
    assert sum(e["type"] == "episode_saved" for e in events) == 2
    assert events[-1]["type"] == "session_end" and events[-1]["reason"] == "quit"


def test_a_discarded_episode_keeps_its_random_offset_for_the_retry(tmp_path, monkeypatch):
    script = [("hover", {}), ("rec", {}), ("hover", {}), ("rec", {"rerecord_episode": True}), ("hover", {}), ("rec", {})]
    dataset, session = run_record(tmp_path, monkeypatch, ["--hover-jitter-mm", "8", "--jitter-zero-fraction", "0",
                                                          "--jitter-seed", "3"], script)
    rows = episode_rows(dataset)
    assert len(rows) == 2 and dataset.saved == [3, 3]
    events = read_events(session)
    hovers = [e["hover_jitter_xy"] for e in events if e["type"] == "hover"]
    assert len(hovers) == 3 and hovers[1] == hovers[2] != hovers[0]       # the retry kept the discarded one's draw
    assert [r["hover_jitter_xy"] for r in rows] == [hovers[0], hovers[2]]
    assert all(0 < math.hypot(*r["hover_jitter_xy"]) <= 0.008 for r in rows)
    (discard,) = [e for e in events if e["type"] == "episode_discarded"]
    assert discard["tile"] == "8D" and discard["frames"] == 3
