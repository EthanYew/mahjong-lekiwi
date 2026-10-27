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

from mahjong import config, hardware, kinematics
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


def run_record(tmp_path, monkeypatch, argv, script, answers=None, prompts=None, gamepad=FakeGamepad, mats=None):
    """record.main(argv) on the sim arm, camera and mat, picking 8D each time. `script`: what the
    keyboard does, in order — ("hover", keys) when the hover is shown, ("rec", keys) after 3 recorded
    frames; keys beyond → (exit_early) e.g. {"rerecord_episode": True} for ←. answers: {prompt substring: reply}
for input() (else Enter, and 8D for "Which tile"); prompts, if a list, collects every prompt. Returns
(dataset, session dir)."""
    import builtins
    import record
    from mahjong.arm import Arm
    from mahjong.sim import SimCamera, SimPerception, SimRobot

    def connect(needed, gamepad_kwargs=None):
        mat = SimPerception()
        if mats is not None:
            mats.append(mat)
        arm = Arm(SimRobot(on_release=mat.release), sleep=lambda s: None)
        arm.at_connect = {"arm_at_rest": True, "off_rest_deg": 0.0}
        return {"arm": arm, "wrist_camera": SimCamera(), "realsense": mat, "gamepad": gamepad()}

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
    def reply(prompt=""):
        if prompts is not None:
            prompts.append(prompt)
        replies = {"Which tile": "8D", **(answers or {})}
        return next((r for key, r in replies.items() if key in prompt), "")

    monkeypatch.setattr(builtins, "input", reply)
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
    assert discard["roll_command_travel_deg"] == 0 and "roll_lag_max_deg" in discard     # no stick input in sim
    assert all("roll_command_change_deg" in r for r in rows)


def test_roll_trace_tells_a_commanded_roll_from_a_turned_wrist():
    from scripts.record import RollTrace
    roll = RollTrace()
    for commanded, measured in [(-56, -56), (-56, -60), (-50, -51), (-53, -53)]:
        roll.add(commanded, measured)
    assert roll.summary() == {"roll_command_change_deg": 3, "roll_command_travel_deg": 9, "roll_lag_max_deg": 4}
    assert RollTrace().summary() == {}


DISCARD_ONCE = [("hover", {}), ("rec", {"rerecord_episode": True}), ("hover", {}), ("rec", {}), ("hover", {}), ("rec", {})]


def test_after_a_discard_enter_hovers_over_the_same_tile_without_choosing_it_again(tmp_path, monkeypatch):
    prompts = []
    dataset, session = run_record(tmp_path, monkeypatch, ["--hover-offsets-mm", "6x"], DISCARD_ONCE, prompts=prompts)
    assert dataset.saved == [3, 3]
    assert sum("Which tile" in q for q in prompts) == 2          # first episode, second episode: not the retry
    assert any("Same tile" in q and "Enter = y" in q for q in prompts)
    hovers = [e for e in read_events(session) if e["type"] == "hover"]
    assert [h["same_tile_retry"] for h in hovers] == [False, True, False]
    assert hovers[1]["hover_xy"] == hovers[0]["hover_xy"] and hovers[1]["hover_jitter_xy"] == [0.006, 0.0]


def test_after_a_discard_n_goes_back_to_rest_and_asks_for_the_tile(tmp_path, monkeypatch):
    prompts = []
    dataset, session = run_record(tmp_path, monkeypatch, [], DISCARD_ONCE, answers={"Same tile": "n"}, prompts=prompts)
    assert dataset.saved == [3, 3] and sum("Which tile" in q for q in prompts) == 3
    assert not any(e.get("same_tile_retry") for e in read_events(session) if e["type"] == "hover")


def test_a_knocked_tile_discard_defaults_to_choosing_again(tmp_path, monkeypatch):
    prompts = []
    run_record(tmp_path, monkeypatch, [], DISCARD_ONCE, answers={"Why discard": "2"}, prompts=prompts)
    assert any("Same tile" in q and "Enter = n" in q for q in prompts)
    assert sum("Which tile" in q for q in prompts) == 3


class GrippingGamepad(FakeGamepad):
    """Closes the jaws on the tile under the hover: the sim mat hands it over, the gripper stalls on it."""
    def start(self, arm):
        self.mat = arm.robot.on_release.__self__
        x, y = kinematics.tool_point(arm.commanded())[:2]
        self.tile = self.mat.nearest(x, y)

    def step(self, arm, dt):
        if not arm.robot.holding:
            arm.send({**arm.commanded(), "gripper": 0.0})
            arm.robot.holding = True
            self.mat.grasp(self.tile)
        return arm.commanded()


def test_a_held_tile_is_set_back_down_where_it_was_picked_not_dropped_at_rest(tmp_path, monkeypatch):
    mats = []
    script = [("hover", {}), ("rec", {}), ("hover", {}), ("rec", {"rerecord_episode": True}),
              ("hover", {}), ("rec", {})]
    dataset, session = run_record(tmp_path, monkeypatch, [], script, gamepad=GrippingGamepad, mats=mats)
    assert dataset.saved == [3, 3]
    events = read_events(session)
    returns = [e for e in events if e["type"] == "return"]
    assert len(returns) == 3 and all(e["tile"] == "8D" for e in returns)       # saved, discarded, saved
    (mat,) = mats
    assert mat.held is None and any(t["tile_name"] == "8D" for t in mat.tiles)   # on the mat, not dropped
    tile = next(t for t in mat.tiles if t["tile_name"] == "8D")
    assert math.dist(tile["xy"], returns[-1]["asked_xy"]) < 0.002
    # a returned tile has moved, so a discard doesn't offer the same hover again
    hovers = [e for e in events if e["type"] == "hover"]
    assert [h["same_tile_retry"] for h in hovers] == [False, False, False]


def test_no_return_leaves_the_tile_to_be_released_at_rest(tmp_path, monkeypatch):
    mats = []
    dataset, session = run_record(tmp_path, monkeypatch, ["--no-return", "--episodes", "1"],
                                  [("hover", {}), ("rec", {})], gamepad=GrippingGamepad, mats=mats)
    assert dataset.saved == [3] and not [e for e in read_events(session) if e["type"] == "return"]
    assert not any(t["tile_name"] == "8D" for t in mats[0].tiles)       # released high at rest: off the mat
