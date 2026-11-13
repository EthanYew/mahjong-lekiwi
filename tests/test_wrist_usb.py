"""Getting the wrist camera back with no one touching it, what that logs, and the root helper's refusals."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from mahjong import config, hardware, wrist_usb      # noqa: E402
from mahjong.session import Session, read_events      # noqa: E402


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def opens_after(tries):
    state = {"tries": 0}

    def opener():
        state["tries"] += 1
        if state["tries"] <= tries:
            raise hardware.NotReady("not yet")
        return "camera"
    return opener


def no_cycle(port):
    raise AssertionError("no power cycle was needed")


def test_a_drop_that_comes_back_needs_no_power_cycle():
    clock = Clock()
    camera, record = wrist_usb.recover(opens_after(3), cycle=no_cycle, clock=clock, sleep=clock.sleep)
    assert camera == "camera" and record["outcome"] == "came_back" and record["seconds"] == 0.75


def test_a_dead_port_goes_straight_to_the_person_with_no_power_cycle():
    """The default: the cycle recovered 0 of 9 real losses, so it no longer runs or delays the prompt."""
    clock = Clock()
    camera, record = wrist_usb.recover(opens_after(10 ** 6), clock=clock, sleep=clock.sleep)
    assert camera is None and record["outcome"] == "not_back"
    assert "cycled" not in record and "cycle_failed" not in record
    assert clock.now == wrist_usb.RETURN_WAIT_S      # no AFTER_CYCLE_WAIT_S on top


def test_a_dead_port_gets_one_power_cycle_after_the_wait():
    """Still available on request (cycle=power_cycle), for trying it again on a new cable or port."""
    clock, cycles = Clock(), []
    tries = int(wrist_usb.RETURN_WAIT_S / wrist_usb.POLL_S) + 3

    def cycle(port):
        cycles.append((port, clock.now))
        return True, None
    camera, record = wrist_usb.recover(opens_after(tries), cycle=cycle, clock=clock, sleep=clock.sleep)
    assert camera == "camera" and record["outcome"] == "power_cycle" and record["cycled"] is True
    assert len(cycles) == 1 and cycles[0][1] == wrist_usb.RETURN_WAIT_S


def test_without_the_helper_it_says_why_and_does_not_wait_again():
    clock = Clock()
    camera, record = wrist_usb.recover(opens_after(10 ** 6), cycle=lambda port: (False, "not installed"), clock=clock,
                                       sleep=clock.sleep)
    assert camera is None and record["outcome"] == "not_back" and record["cycle_failed"] == "not installed"
    assert clock.now == wrist_usb.RETURN_WAIT_S


def test_power_cycle_never_asks_for_a_password_and_reports_a_refusal(tmp_path):
    helper = tmp_path / "helper"
    helper.write_text("")
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 4, stdout="", stderr="leaving it alone")
    assert wrist_usb.power_cycle("3-8", run=run, helper=str(helper)) == (False, "leaving it alone")
    assert seen == [["sudo", "-n", str(helper), "3-8"]]
    assert wrist_usb.power_cycle("3-8", helper=str(tmp_path / "missing"))[0] is False


def test_port_of_reads_the_port_from_sysfs(tmp_path):
    interface = tmp_path / "pci" / "usb3" / "3-8" / "3-8:1.0"
    interface.mkdir(parents=True)
    video = tmp_path / "class" / "video9"
    video.mkdir(parents=True)
    (video / "device").symlink_to(interface)
    assert wrist_usb.port_of(str(tmp_path / "video9"), sys_class=str(tmp_path / "class")) == "3-8"
    assert wrist_usb.port_of(str(tmp_path / "video1"), sys_class=str(tmp_path / "class")) is None


class Camera:
    def __init__(self, frames=10 ** 9):
        self.frames, self.released = frames, False

    def read(self):
        self.frames -= 1
        return self.frames >= 0, None

    def release(self):
        self.released = True


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(wrist_usb, "RETURN_WAIT_S", 0.05)
    monkeypatch.setattr(wrist_usb, "AFTER_CYCLE_WAIT_S", 0.0)
    monkeypatch.setattr(wrist_usb, "POLL_S", 0.01)
    monkeypatch.setattr(wrist_usb, "SETTLE_S", 0.01)
    monkeypatch.setattr(wrist_usb.recover, "__defaults__",
                        (0.05, 0.0, lambda port: (False, "not installed"), *wrist_usb.recover.__defaults__[3:]))
    monkeypatch.setattr("mahjong.camera.LOST_S", 0.01)
    monkeypatch.setattr("mahjong.camera.read_within", lambda camera, **k: camera.read())


def test_reopening_logs_a_return_with_no_prompt(tmp_path, monkeypatch, quick):
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    session = Session("bench", ask_setup=False)
    old, new = Camera(), Camera()
    got = hardware.reopen_wrist_camera(old, ask=lambda _: pytest.fail("asked a person"), openers={"wrist_camera": lambda: new})
    session.close()
    assert got is new and old.released
    record = next(e for e in read_events(session.dir) if e["type"] == "wrist_camera_recovery")
    assert record["outcome"] == "came_back"


def test_a_camera_that_drops_again_at_once_does_not_count_as_back(tmp_path, monkeypatch, quick):
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    flapping, steady = Camera(frames=0), Camera()
    opened = iter([flapping, steady])
    got = hardware.reopen_wrist_camera(Camera(), ask=lambda _: pytest.fail("asked a person"),
                                       openers={"wrist_camera": lambda: next(opened)})
    assert got is steady and flapping.released


def test_a_person_is_only_asked_after_the_rest_and_quitting_is_logged(tmp_path, monkeypatch, quick):
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    session = Session("bench", ask_setup=False)

    def gone():
        raise hardware.NotReady("gone")
    with pytest.raises(SystemExit):
        hardware.reopen_wrist_camera(Camera(), ask=lambda _: "q", openers={"wrist_camera": gone})
    session.close()
    record = next(e for e in read_events(session.dir) if e["type"] == "wrist_camera_recovery")
    assert record["outcome"] == "quit" and record["cycle_failed"] == "not installed"


def test_summary_says_how_losses_ended():
    events = [{"type": "wrist_camera_recovery", "outcome": "came_back", "seconds": 1.0},
              {"type": "wrist_camera_recovery", "outcome": "came_back", "seconds": 3.0},
              {"type": "wrist_camera_recovery", "outcome": "person", "seconds": 40.0}, {"type": "attempt"}]
    assert wrist_usb.recovery_summary(events) == ("3 losses: 2 came back by itself (median 2.0s), "
                                                  "1 a person at the prompt (median 40.0s)")
    assert wrist_usb.recovery_summary([{"type": "attempt"}]) is None


def test_startup_cycles_only_a_camera_that_is_gone(monkeypatch):
    calls = []
    monkeypatch.setattr(wrist_usb, "recover", lambda opener, wait_s: calls.append(wait_s) or (None, {}))
    monkeypatch.setitem(hardware.OPENERS, "wrist_camera", lambda **k: Camera())
    monkeypatch.setattr(config, "WRIST_CAM", "/nonexistent/wristcam")
    hardware.connect(["wrist_camera"])
    assert calls == [0]
    monkeypatch.setattr(config, "WRIST_CAM", __file__)      # there, maybe in use by another program
    hardware.connect(["wrist_camera"])
    assert calls == [0]


# --- the root helper, against a fake /sys ---

def fake_port(root, bus="3", number="8", device=None, peer_device=False):
    port = root / f"{bus}-0:1.0" / f"usb{bus}-port{number}"
    port.mkdir(parents=True)
    (port / "disable").write_text("0\n")
    if device:
        (port / "device").mkdir()
        (port / "device" / "idVendor").write_text(device[0] + "\n")
        (port / "device" / "idProduct").write_text(device[1] + "\n")
    if peer_device:
        (port / "peer" / "device").mkdir(parents=True)
    return port


def helper(tmp_path, root):
    script = (REPO / "scripts/wrist_usb/port-cycle.sh").read_text()
    script = script.replace("DEVICES=/sys/bus/usb/devices", f"DEVICES={root}").replace("OFF_S=3", "OFF_S=0")
    path = tmp_path / "cycle.sh"
    path.write_text(script)
    return path


def run_helper(path, *args):
    return subprocess.run(["sh", str(path), *args], capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_helper_cycles_the_cameras_port_and_leaves_it_on(tmp_path):
    root = tmp_path / "sys"
    port = fake_port(root, device=("05a3", "9230"))
    result = run_helper(helper(tmp_path, root), "3-8")
    assert result.returncode == 0, result.stderr
    assert (port / "disable").read_text().strip() == "0"
    empty = fake_port(root, number="9")           # already fallen off: nothing there to check
    assert run_helper(helper(tmp_path, root), "3-9").returncode == 0
    assert (empty / "disable").read_text().strip() == "0"


@pytest.mark.parametrize("argument", ["", "3-8;reboot", "../../etc", "3-8/../3-3", "-h", "3"])
def test_helper_refuses_anything_but_a_port_name(tmp_path, argument):
    assert run_helper(helper(tmp_path, tmp_path), argument).returncode == 2


def test_helper_refuses_a_port_with_something_else_on_it(tmp_path):
    root = tmp_path / "sys"
    arm = fake_port(root, number="3", device=("1a86", "55d3"))
    assert run_helper(helper(tmp_path, root), "3-3").returncode == 4
    assert (arm / "disable").read_text().strip() == "0"
    twin = fake_port(root, number="4", device=("05a3", "9230"), peer_device=True)
    assert run_helper(helper(tmp_path, root), "3-4").returncode == 4
    assert (twin / "disable").read_text().strip() == "0"
    assert run_helper(helper(tmp_path, root), "3-5").returncode == 3      # no such port


def test_helper_finds_a_port_on_a_hub(tmp_path):
    root = tmp_path / "sys"
    port = root / "3-2:1.0" / "3-2-port4"
    port.mkdir(parents=True)
    (port / "disable").write_text("0\n")
    assert run_helper(helper(tmp_path, root), "3-2.4").returncode == 0


# --- the watcher ---

def test_classify_only_takes_lines_about_the_port():
    import wrist_usb_watch as w
    assert w.classify("kernel: usb 3-8: USB disconnect, device number 5", "3-8") == "disconnect"
    assert w.classify("kernel: usb usb3-port8: disabled by hub (EMI?), re-enabling...", "3-8") == "emi"
    assert w.classify("kernel: usb 3-8: device not accepting address 84, error -71", "3-8") == "protocol_error"
    assert w.classify("kernel: usb 3-8.1: USB disconnect, device number 5", "3-8") is None
    assert w.classify("kernel: usb 3-3: USB disconnect, device number 5", "3-8") is None
    assert w.classify("kernel: usb 4-3: USB disconnect", "3-2.4") is None
    assert w.classify("kernel: usb 3-2-port4: Cannot enable", "3-2.4") == "other"


def test_classify_sees_a_camera_that_enumerates_but_cannot_talk():
    """20261113_123953_usbwatch: 51 of these counted as "other", which summary didn't print, so a camera giving no
    frames at all read as a clean run."""
    import wrist_usb_watch as w
    assert w.classify("kernel: uvcvideo 3-8:1.1: Failed to set UVC probe control : -110 (exp. 26).", "3-8") == "control_timeout"
    assert w.classify("kernel: uvcvideo 3-8:1.0: Failed to query (GET_DEF) UVC control 9 on unit 2: -110 (exp. 2).",
                      "3-8") == "control_timeout"
    assert w.classify("kernel: usb 3-8: new low-speed USB device number 126 using xhci_hcd", "3-8") == "speed_fallback"
    assert w.classify("kernel: usb 3-8: new full-speed USB device number 5 using xhci_hcd", "3-8") == "speed_fallback"
    assert w.classify("kernel: usb 3-8: new high-speed USB device number 12 using xhci_hcd", "3-8") == "other"


def test_summary_prints_every_kind_including_other_and_old_watches():
    import wrist_usb_watch as w
    line = w.summary("more-slack", {"seconds": 600.1, "frames": 0, "drops": 0, "worst_gap_s": 0.0, "control_timeout": 51,
                                    "speed_fallback": 2, "other": 3, "disconnect": 0, "emi": 0, "protocol_error": 0})
    assert "51 -110" in line and "2 slow" in line and "3 other" in line and "0 frames" in line
    assert "nan" not in w.summary("old", {"seconds": 60.0, "drops": 0})      # a watch from before these kinds existed
    # a kernel-only watch has no frames by design, so it must not read as a camera giving none
    assert "kernel" in w.summary("bench", {"seconds": 60.0, "frames": 0, "kernel_only": True, "control_timeout": 9})
    assert "0 frames" not in w.summary("bench", {"seconds": 60.0, "frames": 0, "kernel_only": True})


def test_watch_sim_runs_end_to_end_and_reports(tmp_path, monkeypatch, capsys):
    import wrist_usb_watch as w
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    w.main(["--sim", "--label", "by-hand", "--seconds", "0.3", "--port", "3-8"])
    folder = next(tmp_path.glob("*_usbwatch"))
    events = read_events(folder)
    end = next(e for e in events if e["type"] == "watch_end")
    assert end["label"] == "by-hand" and end["totals"]["drops"] == 1 and end["totals"]["disconnect"] == 1
    assert any(e["type"] == "watch_back" for e in events)
    capsys.readouterr()
    w.main(["--report"])
    out = capsys.readouterr().out
    assert "by-hand" in out and "none logged yet" in out


def test_watch_kernel_only_never_opens_the_camera():
    import wrist_usb_watch as w

    class Kernel:
        def __init__(self):
            self.lines = ["kernel: usb 3-8: USB disconnect, device number 5"]

        def drain(self):
            lines, self.lines = self.lines, []
            return lines
    clock = Clock()
    totals = w.watch(Kernel(), "3-8", 1.0, lambda *a, **k: None, open_camera=None, clock=clock, sleep=clock.sleep)
    assert totals["disconnect"] == 1 and totals["drops"] == 0 and totals["frames"] == 0


def test_a_camera_at_the_wrong_size_ends_the_wait_at_once():
    """A link stuck slow doesn't come back by waiting; the person was told not to replug for 10s while it did."""
    tries = []

    def slow():
        tries.append(1)
        raise hardware.WrongResolution("160x120 frames")
    camera, record = wrist_usb.recover(slow, wait_s=10, clock=lambda: 0.0, sleep=lambda s: pytest.fail("waited"))
    assert camera is None and record["outcome"] == "not_back" and tries == [1] and record["refused"] == "160x120 frames"


def test_a_reopen_on_a_slow_link_says_replug_now_and_logs_why(tmp_path, monkeypatch, quick, capsys):
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    session = Session("bench", ask_setup=False)
    new = Camera()
    opened = iter([hardware.WrongResolution("160x120 frames, USB link at 12 Mbit/s"), new])

    def opener():
        got = next(opened)
        if isinstance(got, Exception):
            raise got
        return got
    got = hardware.reopen_wrist_camera(Camera(), ask=lambda _: "", openers={"wrist_camera": opener})
    session.close()
    assert got is new and "came back wrong (160x120 frames, USB link at 12 Mbit/s). Replug it now." in capsys.readouterr().out
    record = next(e for e in read_events(session.dir) if e["type"] == "wrist_camera_recovery")
    assert record["outcome"] == "person" and record["refused"].startswith("160x120")
    assert "of these, 1 came back wrong" in wrist_usb.recovery_summary([record])


def test_a_wrong_size_on_a_link_that_reads_fast_is_not_refused_at_once(tmp_path):
    """The first frame of a camera still coming up can be the wrong size; only a slow link says waiting won't fix it."""
    import numpy as np
    (tmp_path / "3-8").mkdir()
    (tmp_path / "3-8" / "speed").write_text("480\n")
    import unittest.mock
    with unittest.mock.patch.object(hardware.wrist_usb, "port_of", lambda path: "3-8"):
        with pytest.raises(hardware.NotReady) as refused:
            hardware.check_resolution(type("C", (), {"label": "wrist camera", "release": lambda self: None})(),
                                      np.zeros((120, 160, 3), np.uint8), "/dev/wristcam", sys_bus=tmp_path)
    assert not isinstance(refused.value, hardware.WrongResolution)
