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

import numpy as np
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

    def touched(self):
        return False

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


def test_a_return_beyond_the_ik_walks_back_toward_the_hover(monkeypatch):
    """2026-10-27: teleop closed on 9B 29cm out at the left edge; even 15mm closer was out of the IK's reach."""
    import record
    calls = []

    class Arm:
        def holding(self):
            return True, 0.0, 11.0

        def place(self, x, y, offset, log=print):
            calls.append((x, y))
            return {"xy": [x, y], "shift_m": 0.0} if len(calls) == 3 else None

    events = []
    session = types.SimpleNamespace(event=lambda kind, **kw: events.append((kind, kw)))
    shut = [0.0] * 6
    monkeypatch.setattr(kinematics, "grasp_point", lambda actions: (0, [0.229, 0.223]))
    assert record.return_tile(Arm(), [shut], {"hover_xy": [0.209, 0.203], "offset_m": 0.05}, session, "9B")
    assert calls[0] == (0.229, 0.223) and calls[2] == pytest.approx((0.219, 0.213))
    assert events == [("return", {"tile": "9B", "asked_xy": [0.229, 0.223], "toward_hover": 0.5,
                                  "xy": pytest.approx([0.219, 0.213]), "shift_m": 0.0})]


class DescendingPolicy:
    """Stands in for a registered ACT policy: from its hover, lower the shoulder a degree a step."""
    id, hover = "fake_v3c", {"offset_m": 0.05, "clearance_m": 0.045, "gripper": 26.8}

    def __init__(self, policy_id=None):
        self.k = 0

    def reset(self):
        self.k = 0

    def step(self, frame, state_delta):
        self.k += 1
        return np.array([0.0, self.k, 0.0, 0.0, 0.0, 0.0])


class TouchedAt(FakeGamepad):
    """Someone takes the gamepad on its `at`-th check (never, if None); counts its starts."""
    at = 2

    def __init__(self):
        self.checks, self.starts = 0, []

    def touched(self):
        self.checks += 1
        return self.at is not None and self.checks >= self.at

    def start(self, arm):
        self.starts.append(dict(arm.commanded()))


def run_dagger(tmp_path, monkeypatch, gamepad, argv=()):
    import mahjong.policy
    monkeypatch.setattr(mahjong.policy, "Policy", DescendingPolicy)
    return run_record(tmp_path, monkeypatch, ["--policy", "fake_v3c", "--episodes", "1", *argv],
                      [("hover", {}), ("rec", {})], gamepad=gamepad)


def test_dagger_the_policy_drives_until_the_gamepad_is_touched_then_you_have_it(tmp_path, monkeypatch):
    pads = []

    def pad():
        pads.append(TouchedAt())
        return pads[-1]
    dataset, session = run_dagger(tmp_path, monkeypatch, pad)
    (row,) = episode_rows(dataset)
    assert row["policy"] == "fake_v3c" and row["takeover_frame"] == 1 and row["hover"] == DescendingPolicy.hover
    assert dataset.saved == [3] and len(pads[0].starts) == 1      # the gamepad started once, at the takeover
    (takeover,) = [e for e in read_events(session) if e["type"] == "intervention"]
    assert takeover["kind"] == "takeover" and takeover["frame"] == 1


def test_dagger_frames_are_the_policy_s_commands_then_the_gamepad_s(tmp_path, monkeypatch):
    frames = []
    monkeypatch.setattr(FakeDataset, "add_frame", lambda self, f: (frames.append(f), self.frames.append(f)))
    run_dagger(tmp_path, monkeypatch, TouchedAt)
    first, second, third = (f["action"] for f in frames)
    assert first[1] == pytest.approx(1.0, abs=0.05)              # the policy's first step, from the hover
    assert second[1] == pytest.approx(first[1], abs=0.05)         # the gamepad holds where it took over
    assert third[1] == pytest.approx(first[1], abs=0.05)


def test_dagger_without_a_takeover_is_saved_and_says_so(tmp_path, monkeypatch):
    class Untouched(TouchedAt):
        at = None
    dataset, session = run_dagger(tmp_path, monkeypatch, Untouched)
    (row,) = episode_rows(dataset)
    assert row["policy"] == "fake_v3c" and row["takeover_frame"] is None
    assert not [e for e in read_events(session) if e["type"] == "intervention"]


def test_dagger_episodes_are_training_data_only(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="slice"):
        run_dagger(tmp_path, monkeypatch, TouchedAt, ["--slice", "jitter"])


def test_session_report_counts_dagger_takeovers(tmp_path, monkeypatch, capsys):
    import session_report
    _, session = run_dagger(tmp_path, monkeypatch, TouchedAt)
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["session_report.py", str(session)])
    session_report.main()
    out = capsys.readouterr().out
    assert "DAgger (fake_v3c): 1 episode(s) saved, 1 taken over at 0.0s" in out
    assert "takeover             rescue" in out and "rescues: 1 in 1 episodes" in out
