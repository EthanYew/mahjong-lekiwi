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

    def arm_takeover(self):
        pass

    def takeover_pressed(self):
        return False

    def close(self):
        pass


def run_record(tmp_path, monkeypatch, argv, script, answers=None, prompts=None, gamepad=FakeGamepad, mats=None,
               cameras=None):
    """record.main(argv) on the sim arm, camera and mat, picking 8D each time. `script`: what the
    keyboard does, in order — ("hover", keys) when the hover is shown, ("rec", keys) after 3 recorded
    frames; keys beyond → (exit_early) e.g. {"rerecord_episode": True} for ←. answers: {prompt substring: reply}
for input() (else Enter, and 8D for "Which tile"); prompts, if a list, collects every prompt. Returns
(dataset, session dir)."""
    import builtins
    import record
    from mahjong.arm import Arm
    from mahjong.sim import SimCamera, SimPerception, SimRobot

    def connect(needed, gamepad_kwargs=None, **kwargs):
        mat = SimPerception()
        if mats is not None:
            mats.append(mat)
        arm = Arm(SimRobot(on_release=mat.release), sleep=lambda s: None)
        arm.at_connect = {"arm_at_rest": True, "off_rest_deg": 0.0}
        camera = cameras.pop(0) if cameras else SimCamera()         # cameras: handed out in turn (reopens too)
        pad = gamepad()
        pad.kwargs = gamepad_kwargs
        return {"arm": arm, "wrist_camera": camera, "realsense": mat, "gamepad": pad}

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    listener = types.SimpleNamespace(stop=lambda: None)
    steps, frames = iter(script), []

    def show(camera, disabled, text, frame=None, last=None):
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
    monkeypatch.setattr(config, "RETURN_CORRECTIONS", tmp_path / "return_corrections.jsonl")
    monkeypatch.setattr(hardware, "connect", connect)
    monkeypatch.setattr(record, "open_dataset", lambda repo_id: dataset)
    monkeypatch.setattr(record, "show", show)
    monkeypatch.setattr(record, "KEY_REPEAT_S", 0.0)       # the scripted → comes 3 frames in, well inside it
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
    """Closes the jaws on the tile under the hover: the sim mat hands it over, the gripper stalls on it. Run
    with --no-correction: the sim has no camera-to-arm offset for the hover correction to undo, so a corrected
    hover closes ~50mm off the tile and the return lands it there."""
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


def test_a_held_tile_is_set_back_down_on_the_mat_not_dropped_at_rest(tmp_path, monkeypatch):
    mats = []
    script = [("hover", {}), ("rec", {}), ("hover", {}), ("rec", {"rerecord_episode": True}),
              ("hover", {}), ("rec", {})]
    dataset, session = run_record(tmp_path, monkeypatch, ["--no-correction"], script, gamepad=GrippingGamepad, mats=mats)
    assert dataset.saved == [3, 3]
    events = read_events(session)
    returns = [e for e in events if e["type"] == "return"]
    assert len(returns) == 3 and all(e["tile"] == "8D" for e in returns)       # saved, discarded, saved
    (mat,) = mats
    assert mat.held is None and any(t["tile_name"] == "8D" for t in mat.tiles)   # on the mat, not dropped
    tile = next(t for t in mat.tiles if t["tile_name"] == "8D")
    assert math.dist(tile["xy"], returns[-1]["asked_xy"]) < 0.002
    landed = [e for e in events if e["type"] == "return_landed"]
    assert len(landed) == 3 and all(e["landed_xy"] for e in landed)
    assert len((tmp_path / "return_corrections.jsonl").read_text().splitlines()) == 3   # samples for the model
    # a returned tile has moved, so a discard doesn't offer the same hover again
    hovers = [e for e in events if e["type"] == "hover"]
    assert [h["same_tile_retry"] for h in hovers] == [False, False, False]


class StallingGamepad(GrippingGamepad):
    """A person who stops closing once the jaws stall on the tile: commanded 8.8, reading 11."""
    def step(self, arm, dt):
        if not arm.robot.holding:
            arm.send({**arm.commanded(), "gripper": 8.8})
            arm.robot.holding = True
            self.mat.grasp(self.tile)
        return arm.commanded()


def test_a_tile_held_by_jaws_a_person_stopped_closing_is_still_returned(tmp_path, monkeypatch):
    """2026-10-30: 20 of 35 saved episodes ended commanded 5-12 on the tile; holds_tile saw empty jaws, so no
    return, and the tile fell out at the next hover's opening."""
    mats = []
    dataset, session = run_record(tmp_path, monkeypatch, ["--no-correction", "--episodes", "1"],
                                  [("hover", {}), ("rec", {})], gamepad=StallingGamepad, mats=mats)
    assert dataset.saved == [3]
    assert len([e for e in read_events(session) if e["type"] == "return"]) == 1
    assert mats[0].held is None and any(t["tile_name"] == "8D" for t in mats[0].tiles)


def test_the_gamepad_gets_the_wider_deadzone_and_a_resting_thumb_is_said(tmp_path, monkeypatch, capsys):
    """2026-10-28: a thumb resting on the left stick, past XLeRobot's 0.15, pulled the arm 87mm (collect_corrections);
    record hands over the gamepad at every episode, so it gets the same deadzone and warning."""
    pads = []

    class ThumbOnStick(FakeGamepad):
        def __init__(self):
            pads.append(self)

        def off_centre(self):
            return {"left": (0.0, -0.3)}
    _, session = run_record(tmp_path, monkeypatch, ["--episodes", "1"], [("hover", {}), ("rec", {})], gamepad=ThumbOnStick)
    assert pads[0].kwargs["deadzone"] == 0.35
    assert pads[0].kwargs["left_stick_max_mm_s"] == 30.0
    assert "left stick reads (+0.00, -0.30)" in capsys.readouterr().out
    (event,) = [e for e in read_events(session) if e["type"] == "stick_off_centre"]
    assert event["tile"] == "8D" and event["sticks"] == {"left": [0.0, -0.3]}


def test_no_return_leaves_the_tile_to_be_released_at_rest(tmp_path, monkeypatch):
    mats = []
    dataset, session = run_record(tmp_path, monkeypatch, ["--no-return", "--episodes", "1"],
                                  [("hover", {}), ("rec", {})], gamepad=GrippingGamepad, mats=mats)
    assert dataset.saved == [3] and not [e for e in read_events(session) if e["type"] == "return"]
    assert not any(t["tile_name"] == "8D" for t in mats[0].tiles)       # released high at rest: off the mat


def test_returns_aim_at_the_home_so_an_off_centre_grasp_doesnt_walk_the_tile(tmp_path, monkeypatch):
    """2026-10-28: returning each tile where the jaws closed walked it 25-31mm in one recording. Every grasp
    here is 6mm off the tile (the hover offset); returns aimed at home land 6mm off it, not 6mm more each time."""
    mats = []
    script = [("hover", {}), ("rec", {})] * 3
    dataset, session = run_record(tmp_path, monkeypatch, ["--no-correction", "--hover-offsets-mm", "6x",
                                                          "--episodes", "3"], script, gamepad=GrippingGamepad, mats=mats)
    assert dataset.saved == [3, 3, 3]
    events = read_events(session)
    landed = [e for e in events if e["type"] == "return_landed"]
    assert len(landed) == 3 and len({tuple(e["home_xy"]) for e in landed}) == 1
    assert all(e["error_m"] < 0.008 for e in landed)
    (home,) = {tuple(e["home_xy"]) for e in landed}
    tile = next(t for t in mats[0].tiles if t["tile_name"] == "8D")
    assert math.dist(tile["xy"], home) < 0.008                 # grasp-point returns: ~18mm by now


def test_session_report_shows_where_a_recording_s_returns_landed(tmp_path, monkeypatch, capsys):
    import session_report
    _, session = run_record(tmp_path, monkeypatch, ["--no-correction", "--hover-offsets-mm", "6x"],
                            [("hover", {}), ("rec", {})] * 2, gamepad=GrippingGamepad)
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["session_report.py", str(session)])
    session_report.main()
    out = capsys.readouterr().out
    assert "returns landed 6mm from home (median of 2" in out
    assert "return rise: the tip rose" in out and "swing to rest from" in out        # Arm.place's measured rise
    assert "return release: " in out and "from where the jaws closed" in out


def test_a_return_beyond_the_ik_walks_back_toward_the_hover(monkeypatch):
    """2026-10-27: teleop closed on 9B 29cm out at the left edge; even 15mm closer was out of the IK's reach."""
    import record
    from mahjong.return_correction import ReturnCorrection
    calls = []

    class Arm:
        def holding(self):
            return True, 0.0, 11.0

        def place(self, x, y, offset, log=print, grasp_clearance_m=None):
            calls.append((x, y))
            return {"xy": [x, y], "shift_m": 0.0} if len(calls) == 3 else None

    events = []
    session = types.SimpleNamespace(event=lambda kind, **kw: events.append((kind, kw)))
    shut = [0.0] * 6
    monkeypatch.setattr(kinematics, "grasp_point", lambda actions: (0, [0.229, 0.223]))
    monkeypatch.setattr(record, "check_landing", lambda *a, **kw: (None, False))
    held = {"tile_name": "9B", "detection_xy": [0.209, 0.203], "home_xy": [0.209, 0.203], "attempt_dir": None,
            "spot": None}
    realsense = types.SimpleNamespace(detect_tiles=lambda: [])
    assert record.return_tile(Arm(), [shut], {"hover_xy": [0.209, 0.203], "offset_m": 0.05}, held,
                              ReturnCorrection([]), realsense, session, None, DescendingPolicy.hover, "teleop")
    assert calls[0] == pytest.approx((0.229, 0.223)) and calls[2] == pytest.approx((0.219, 0.213))
    kind, ret = events[0]
    assert kind == "return" and ret["asked_xy"] == pytest.approx([0.229, 0.223]) and ret["toward_hover"] == 0.5
    assert ret["xy"] == pytest.approx([0.219, 0.213]) and ret["home_xy"] == [0.209, 0.203]


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
    """Someone presses A on the `at`-th check (never, if None); counts its starts."""
    at = 2

    def __init__(self):
        self.checks, self.starts = 0, []

    def takeover_pressed(self):
        self.checks += 1
        return self.at is not None and self.checks >= self.at

    def start(self, arm):
        self.starts.append(dict(arm.commanded()))


def run_dagger(tmp_path, monkeypatch, gamepad, argv=()):
    import mahjong.policy
    monkeypatch.setattr(mahjong.policy, "Policy", DescendingPolicy)
    return run_record(tmp_path, monkeypatch, ["--policy", "fake_v3c", "--episodes", "1", *argv],
                      [("hover", {}), ("rec", {})], gamepad=gamepad)


def test_dagger_the_policy_drives_until_a_is_pressed_then_you_have_it(tmp_path, monkeypatch):
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


class LostCamera:
    """A wrist camera that has stopped giving frames (2026-10-28), until it's released and reopened."""
    def __init__(self):
        self.reads, self.released = 0, False

    def read(self):
        self.reads += 1
        return False, None

    def flush(self, frames=10):
        pass

    def release(self):
        self.released = True


def test_an_episode_the_wrist_camera_dropped_out_of_is_discarded_and_the_camera_reopened(tmp_path, monkeypatch):
    from mahjong.sim import SimCamera
    lost = LostCamera()
    script = [("hover", {}), ("hover", {}), ("rec", {}), ("hover", {}), ("rec", {})]
    dataset, session = run_record(tmp_path, monkeypatch, [], script, cameras=[lost, SimCamera()])
    assert lost.released and lost.reads == 30                           # 1 s at 30 fps, then reopened
    discarded = [e for e in read_events(session) if e["type"] == "episode_discarded"]
    assert [e["reason"] for e in discarded] == ["wrist camera lost"]    # no question asked
    assert len(dataset.saved) == 2                                      # both episodes recorded after the reopen


class TornCamera:
    """A wrist camera that keeps streaming but tears (20261113_100710_bench): here every row jumps from one frame to the next."""
    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        return True, np.full((480, 640, 3), 255 * (self.reads % 2), dtype=np.uint8)

    def flush(self, frames=10):
        pass

    def release(self):
        pass


def test_an_episode_on_torn_frames_says_how_many_in_its_log_row(tmp_path, monkeypatch):
    """policy.run stops on torn frames; a recording trains on every frame, so its row has to say they were torn."""
    script = [("hover", {}), ("rec", {}), ("hover", {}), ("rec", {})]
    dataset, _ = run_record(tmp_path, monkeypatch, [], script, cameras=[TornCamera()])
    torn, = episode_rows(dataset)[:1]
    assert torn["frames"] >= 2 and torn["torn_frames"] == torn["frames"] - 1      # every frame after the first


def test_frames_that_jump_once_the_jaws_hold_arent_counted_as_torn(tmp_path, monkeypatch, capsys):
    """Counted until the first hold, as policy.run does: a fast lift after it blurs 1-3 frames into a tear's numbers, and
    warning on those fired on ordinary episodes."""
    import record
    monkeypatch.setattr(record, "holds_tile", lambda command, measured: True)       # held from the first frame
    dataset, _ = run_record(tmp_path, monkeypatch, [], [("hover", {}), ("rec", {})] * 2, cameras=[TornCamera()])
    assert [r["torn_frames"] for r in episode_rows(dataset)] == [0, 0]
    assert "torn wrist camera frames in it" not in capsys.readouterr().out


def test_a_few_torn_frames_are_counted_but_not_warned_about(tmp_path, monkeypatch, capsys):
    """A fast move blurs 1-3 frames into a tear's numbers: under camera.TORN_STOP_FRAMES, the row says so and nothing warns."""
    import record
    calls = iter(range(10 ** 6))
    monkeypatch.setattr(record, "torn", lambda previous, frame: next(calls) in (1, 2))      # two, inside the first episode
    dataset, _ = run_record(tmp_path, monkeypatch, [], [("hover", {}), ("rec", {})] * 2)
    assert [r["torn_frames"] for r in episode_rows(dataset)] == [2, 0]
    assert "torn wrist camera frames in it" not in capsys.readouterr().out


def test_an_episode_torn_before_the_hold_is_warned_about(tmp_path, monkeypatch, capsys):
    import record
    monkeypatch.setattr(record, "TORN_STOP_FRAMES", 2)          # a test episode is 3 frames
    run_record(tmp_path, monkeypatch, [], [("hover", {}), ("rec", {})] * 2, cameras=[TornCamera()])
    assert "torn wrist camera frames in it (USB data lost)" in capsys.readouterr().out


def test_an_episode_on_clean_frames_has_no_torn_ones(tmp_path, monkeypatch):
    from mahjong.sim import SimCamera
    dataset, _ = run_record(tmp_path, monkeypatch, [], [("hover", {}), ("rec", {})] * 2, cameras=[SimCamera()])
    assert [r["torn_frames"] for r in episode_rows(dataset)] == [0, 0]


def test_keys_pressed_while_the_camera_was_lost_dont_answer_the_same_tile_question(tmp_path, monkeypatch):
    """The ← / → pressed while recording sat in stdin: the first answer to "Same tile" came out garbled, so it
    was asked twice and only the second answer counted. Nothing flushed them on the camera-lost path."""
    import record
    from mahjong.sim import SimCamera
    log = []
    monkeypatch.setattr(record, "drain_stdin", lambda: log.append("drain"))
    script = [("hover", {}), ("hover", {}), ("rec", {}), ("hover", {}), ("rec", {})]
    run_record(tmp_path, monkeypatch, [], script, cameras=[LostCamera(), SimCamera()], prompts=log)
    same = next(i for i, q in enumerate(log) if q != "drain" and "Same tile" in q)
    assert log[same - 1] == "drain"


def test_a_second_right_arrow_straight_after_the_first_doesnt_save(tmp_path, monkeypatch):
    """The → that starts recording, counted twice, used to end the episode at once."""
    import builtins, record
    from mahjong.arm import Arm
    from mahjong.sim import SimCamera, SimPerception, SimRobot

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    frames, clock = [], [0.0]

    def show(camera, disabled, text, frame=None, last=None):
        if text.startswith("hovering"):
            events["exit_early"] = True            # → to start
        else:
            frames.append(text)
            if len(frames) == 1:
                events["exit_early"] = True        # the same press, counted again 0.0s in
            elif len(frames) == 5:
                clock[0] += 1.0
                events["exit_early"] = True        # a real → to save

    def connect(needed, gamepad_kwargs=None):
        mat = SimPerception()
        arm = Arm(SimRobot(on_release=mat.release), sleep=lambda s: None)
        arm.at_connect = {"arm_at_rest": True, "off_rest_deg": 0.0}
        return {"arm": arm, "wrist_camera": SimCamera(), "realsense": mat, "gamepad": FakeGamepad()}

    dataset = FakeDataset(tmp_path / "dataset")
    dataset.root.mkdir()
    real_time = record.time.time
    monkeypatch.setattr(record.time, "time", lambda: real_time() + clock[0])
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(hardware, "connect", connect)
    monkeypatch.setattr(record, "open_dataset", lambda repo_id: dataset)
    monkeypatch.setattr(record, "show", show)
    monkeypatch.setitem(sys.modules, "lerobot.common.control_utils",
                        types.SimpleNamespace(init_keyboard_listener=lambda: (types.SimpleNamespace(stop=lambda: None), events)))
    monkeypatch.setattr(builtins, "input", lambda prompt="": "8D" if "Which tile" in prompt else "")
    record.main(["--dataset", "test/record", "--episodes", "1"])
    assert dataset.saved == [5]
