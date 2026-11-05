"""collect_corrections.py end to end on simulated hardware: the wrist view while lining up, the frame saved
with each correction, and a wrist camera lost mid-alignment reopened without losing the tile."""
import json
import sys
import types
from pathlib import Path

from mahjong import config, hardware
from mahjong.session import read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class FakeGamepad:
    """Presses → (or ← for the tiles in `skip`, numbered from 1) after `steps` steps of lining up."""
    def __init__(self, events, skip=(), steps=5, off=None):
        self.events, self.skip, self.every, self.steps, self.starts = events, set(skip), steps, 0, 0
        self.off, self.kwargs, self.shown_inputs = off or {}, None, []

    def start(self, arm):
        self.starts += 1

    def off_centre(self):
        return dict(self.off)

    def step(self, arm, dt):
        self.steps += 1
        if self.steps % self.every == 0:
            self.events.update(exit_early=True, rerecord_episode=(self.steps // self.every) in self.skip)
        command = arm.commanded()
        arm.send(command)
        self.last = {"measured": arm.measured(), "buttons": [], "sticks": {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0},
                     "raw": {"lx": 0.0, "ly": -0.3 if self.off else 0.0, "ry": 0.0}}
        return command

    def close(self):
        pass


class DroppingCamera:
    """Gives frames, then none from the 3rd read on (a wrist camera dropping out)."""
    simulated = False

    def __init__(self):
        self.reads, self.released = 0, False

    def read(self):
        import numpy as np
        self.reads += 1
        return (True, np.zeros((480, 640, 3), np.uint8)) if self.reads < 3 else (False, None)

    def release(self):
        self.released = True


def run(tmp_path, monkeypatch, argv=(), skip=(), camera=None, steps=5, off=None):
    """collect_corrections.main on the sim arm and mat, one scan, then q. Returns (session dir, shown frames,
    the gamepad, the connect calls)."""
    import builtins
    import collect_corrections
    from mahjong.arm import Arm
    from mahjong.sim import SimPerception, SimRobot, SimWristCamera

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    gamepad, shown, connects = FakeGamepad(events, skip, steps, off), [], []

    def connect(needed, gamepad_kwargs=None, **kwargs):
        connects.append(list(needed))
        gamepad.kwargs = gamepad.kwargs or gamepad_kwargs
        if needed == ["wrist_camera"]:           # a reopen
            return {"wrist_camera": SimWristCamera(arm, mat)}
        devices = {"arm": arm, "realsense": mat, "gamepad": gamepad}
        if "wrist_camera" in needed:
            devices["wrist_camera"] = camera or SimWristCamera(arm, mat)
        return devices

    mat = SimPerception()
    mat.close = lambda: None
    arm = Arm(SimRobot(), sleep=lambda s: None)
    arm.at_connect = {"arm_at_rest": True, "off_rest_deg": 0.0}
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(config, "HOVER_CORRECTIONS", tmp_path / "hover_corrections.jsonl")
    monkeypatch.setattr(hardware, "connect", connect)
    def show(frame, text, last=None):
        shown.append(text)
        gamepad.shown_inputs.append(last)
    monkeypatch.setattr(collect_corrections, "show", show)
    monkeypatch.setattr(collect_corrections.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setitem(sys.modules, "lerobot.common.control_utils", types.SimpleNamespace(
        init_keyboard_listener=lambda: (types.SimpleNamespace(stop=lambda: None), events)))
    monkeypatch.setattr(builtins, "input", lambda prompt="": "q" if "scan again" in prompt else "")   # setup questions: Enter
    collect_corrections.main(list(argv))
    (session,) = (tmp_path / "sessions").iterdir()
    return session, shown, gamepad, connects


def test_the_wrist_view_is_shown_while_lining_up_and_each_record_keeps_its_frame(tmp_path, monkeypatch):
    session, shown, _, connects = run(tmp_path, monkeypatch, skip={2})
    assert "wrist_camera" in connects[0]
    assert shown and all("line up" in text for text in shown)
    events = read_events(session)
    recorded = [e for e in events if e["type"] == "correction_recorded"]
    skipped = [e for e in events if e["type"] == "correction_skipped"]
    assert len(skipped) == 1 and len(recorded) >= 1
    for e in recorded:
        assert (session / e["wrist_frame"]).is_file()
    rows = [json.loads(line) for line in (tmp_path / "hover_corrections.jsonl").read_text().splitlines()]
    assert len(rows) == len(recorded) and all("wrist_frame" not in r for r in rows)   # the log stays fitting data


def test_the_gamepad_gets_the_wider_deadzone_and_its_readings_go_to_the_window(tmp_path, monkeypatch):
    import collect_corrections
    _, _, gamepad, _ = run(tmp_path, monkeypatch)
    assert gamepad.kwargs["deadzone"] == collect_corrections.DEADZONE == 0.35
    assert gamepad.kwargs["left_stick_max_mm_s"] == collect_corrections.LEFT_STICK_MAX_MM_S
    assert gamepad.shown_inputs and all(last and "raw" in last for last in gamepad.shown_inputs)
    _, _, gamepad, _ = run(tmp_path / "again", monkeypatch, argv=["--deadzone", "0.5"])
    assert gamepad.kwargs["deadzone"] == 0.5


def test_a_stick_off_centre_at_the_handoff_is_said_and_logged(tmp_path, monkeypatch, capsys):
    session, _, _, _ = run(tmp_path, monkeypatch, off={"left": (0.0, -0.3)})
    assert "left stick reads (+0.00, -0.30)" in capsys.readouterr().out
    events = [e for e in read_events(session) if e["type"] == "stick_off_centre"]
    assert events and events[0]["sticks"] == {"left": [0.0, -0.3]}


def test_every_lined_up_tile_is_traced_and_the_report_reads_the_traces(tmp_path, monkeypatch):
    import handoff_report
    session, _, gamepad, _ = run(tmp_path, monkeypatch, skip={2})
    events = read_events(session)
    traces = [e for e in events if e["type"] == "handoff_trace"]
    lined_up = [e for e in events if e["type"] in ("correction_recorded", "correction_skipped")]
    assert len(traces) == len(lined_up) >= 2                          # skipped tiles are traced too
    for e in traces:
        rows = [json.loads(line) for line in (session / e["trace"]).read_text().splitlines()]
        assert len(rows) == e["steps"] + 1 == gamepad.every + 1        # the seed, then one row per step
        assert e["idle_command_moved_mm"] == 0 and e["first_input_s"] is None
        assert e["worst_sag_mm"] > 0                                   # the sim arm sags under its command
    summaries = handoff_report.report(session, log=lambda *a: None)
    assert len(summaries) == len(traces)
    assert all(path.with_suffix(".png").is_file() for path, _ in summaries)


def test_no_view_needs_no_wrist_camera(tmp_path, monkeypatch):
    session, shown, _, connects = run(tmp_path, monkeypatch, argv=["--no-view"])
    assert "wrist_camera" not in connects[0] and not shown
    recorded = [e for e in read_events(session) if e["type"] == "correction_recorded"]
    assert recorded and all("wrist_frame" not in e for e in recorded)


def test_a_lost_wrist_camera_is_reopened_and_lining_up_goes_on(tmp_path, monkeypatch):
    camera = DroppingCamera()      # lining up lasts long enough for its failed reads to count as lost
    monkeypatch.setattr(hardware.wrist_usb, "recover", lambda opener: (None, {"outcome": "not_back", "seconds": 0.0}))
    session, shown, gamepad, connects = run(tmp_path, monkeypatch, camera=camera, steps=int(config.FPS * 1.5))
    assert camera.released and connects.count(["wrist_camera"]) == 1
    recorded = [e for e in read_events(session) if e["type"] == "correction_recorded"]
    assert gamepad.starts == len(recorded) + 1           # the reopen picked lining up up again from the command
    assert recorded and all((session / e["wrist_frame"]).is_file() for e in recorded)
