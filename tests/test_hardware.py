"""Preflight: missing hardware is reported with a retry / quit choice, before any session exists."""
import sys
from pathlib import Path

import pytest

from mahjong import config, hardware
from mahjong.hardware import NotReady

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class Device:
    def __init__(self, name):
        self.name, self.closed = name, False

    def disconnect(self):
        self.closed = True

    release = close = disconnect


def flaky(failures, name):
    """An opener failing `failures` times before working."""
    calls = {"n": 0}

    def opener(**kwargs):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise NotReady(f"{name} unplugged")
        return Device(name)
    return opener


def test_retry_until_everything_is_there(capsys):
    answers = iter(["", ""])
    openers = {"arm": flaky(2, "arm"), "wrist_camera": flaky(0, "wrist")}
    devices = hardware.connect(["arm", "wrist_camera"], ask=lambda _: next(answers), openers=openers)
    assert set(devices) == {"arm", "wrist_camera"}
    assert "arm unplugged" in capsys.readouterr().out


def test_quit_closes_what_opened_and_exits():
    wrist = Device("wrist")
    openers = {"arm": flaky(9, "arm"), "wrist_camera": lambda **k: wrist}
    with pytest.raises(SystemExit):
        hardware.connect(["arm", "wrist_camera"], ask=lambda _: "q", openers=openers)
    assert wrist.closed


def test_unexpected_errors_are_offered_a_retry_too(capsys):
    def broken(**kwargs):
        raise OSError("device busy")
    with pytest.raises(SystemExit):
        hardware.connect(["realsense"], ask=lambda _: "q", openers={"realsense": broken})
    assert "OSError: device busy" in capsys.readouterr().out


def test_bench_creates_no_session_when_you_quit_at_preflight(tmp_path, monkeypatch):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setitem(hardware.OPENERS, "arm", flaky(9, "arm"))
    monkeypatch.setitem(hardware.OPENERS, "wrist_camera", lambda **k: Device("wrist"))
    monkeypatch.setitem(hardware.OPENERS, "realsense", lambda **k: Device("rs"))
    monkeypatch.setattr("builtins.input", lambda _: "q")
    with pytest.raises(SystemExit):
        bench.Bench(bench.parse_args([]))
    assert list(tmp_path.iterdir()) == []


class Camera:
    label = "wrist camera"

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def test_a_camera_at_the_wrong_resolution_is_not_ready(tmp_path, monkeypatch):
    """20261112_155318_holdup: a replug left the wrist camera at 12 Mbit/s and 160x120, and the reopen took it."""
    import numpy as np
    (tmp_path / "3-8").mkdir()
    (tmp_path / "3-8" / "speed").write_text("12\n")
    monkeypatch.setattr(hardware.wrist_usb, "port_of", lambda path: "3-8")
    camera = Camera()
    with pytest.raises(NotReady, match=r"160x120 frames, not 640x480, USB link at 12 Mbit/s"):
        hardware.check_resolution(camera, np.zeros((120, 160, 3), np.uint8), "/dev/wristcam", sys_bus=tmp_path)
    assert camera.released


def test_a_camera_at_the_recorded_resolution_passes(tmp_path):
    import numpy as np
    camera = Camera()
    hardware.check_resolution(camera, np.zeros((480, 640, 3), np.uint8), "/dev/wristcam", sys_bus=tmp_path)
    assert not camera.released


def test_link_speed_unreadable_is_none(tmp_path):
    assert hardware.wrist_usb.link_speed("3-8", sys_bus=tmp_path) is None
