"""Getting a wrist camera back that fell off USB, without a person if possible.

The kernel log says this is an electrical fault on the camera's USB link, not a driver one (`error -71`, `disabled by
hub (EMI?)`; docs/invariant_notes.md), so this doesn't fix the cause, only what a drop costs. Most drops come back by
themselves in under a second, some flap for a few seconds, and in some the port stays dead until a replug. So: wait a
little, cut the port's power once, then ask a person.

Whether the wait and the power cycle earn their keep isn't known yet: older recovery times can't say, since they
include how fast a person replugged. Every recovery is logged (`wrist_camera_recovery`) with how it ended, so
`recovery_summary` answers it from sessions where nobody touched the cable first.

The power cycle needs root: scripts/wrist_usb/setup.sh (once) installs a helper that only cycles the camera's own port.
Without it, this step is skipped and everything else works.
"""
import os
import statistics
import subprocess
import time
from pathlib import Path

from . import config

HELPER = "/usr/local/sbin/wristcam-port-cycle"
RETURN_WAIT_S = 10.0        # how long a drop gets to come back by itself; flapping (a drop within 5s of the last) settles in this
AFTER_CYCLE_WAIT_S = 10.0   # and after the port's power comes back
SETTLE_S = 1.0              # a camera that's back must keep giving frames this long
POLL_S = 0.25
CYCLE_TIMEOUT_S = 15.0
OUTCOMES = ("came_back", "power_cycle", "person", "quit")
OUTCOME_TEXT = {"came_back": "came back by itself", "power_cycle": "after a power cycle",
                "person": "a person at the prompt", "quit": "quit at the prompt"}

_last_port = None


def port_of(path=config.WRIST_CAM, sys_class="/sys/class/video4linux"):
    """The USB port a video device is on ("3-8", "3-2.4"), or None if it isn't there."""
    device = Path(sys_class) / os.path.basename(os.path.realpath(path)) / "device"
    try:
        interface = device.resolve(strict=True).name      # "3-8:1.0"
    except OSError:
        return None
    return interface.split(":")[0] if ":" in interface else None


def link_speed(port, sys_bus="/sys/bus/usb/devices"):
    """The USB link's speed in Mbit/s (480 high speed, 12 full speed), or None if it can't be read. A camera that comes
    back at 12 gives only 160x120 frames (2026-11-12, after a replug)."""
    try:
        return int(float((Path(sys_bus) / port / "speed").read_text().strip()))
    except (OSError, ValueError, TypeError):
        return None


def remember_port(path=config.WRIST_CAM):
    """Note the camera's port while it's there: once it has dropped, nothing says where it was."""
    global _last_port
    _last_port = port_of(path) or _last_port


def current_port():
    return _last_port or config.WRIST_USB_PORT


def power_cycle(port=None, run=subprocess.run, helper=HELPER):
    """Cut the port's power once: (True, None), or (False, why). Never asks for a password. The kernel's own "attempt
    power cycle" hasn't been enough on this port, and many ports can't really cut power, so this is a try, not a fix."""
    if not os.path.exists(helper):
        return False, "not installed (sudo scripts/wrist_usb/setup.sh)"
    try:
        result = run(["sudo", "-n", helper, port or current_port()], capture_output=True, text=True,
                     timeout=CYCLE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"{type(e).__name__}: {e}"
    if result.returncode:
        return False, (result.stderr or result.stdout or f"exit {result.returncode}").strip()
    return True, None


def _wait_for(open_camera, wait_s, clock, sleep):
    """A camera from open_camera() within wait_s, or None; tries at least once."""
    end = clock() + wait_s
    while True:
        try:
            return open_camera()
        except Exception:
            pass
        if clock() >= end:
            return None
        sleep(POLL_S)


def recover(open_camera, wait_s=RETURN_WAIT_S, after_cycle_s=AFTER_CYCLE_WAIT_S, cycle=power_cycle, clock=time.monotonic,
            sleep=time.sleep):
    """(camera or None, record) with no person involved. The record is what `wrist_camera_recovery` logs."""
    start = clock()
    record = {"port": current_port()}

    def done(camera, outcome):
        return camera, {**record, "outcome": outcome, "seconds": round(clock() - start, 2)}

    camera = _wait_for(open_camera, wait_s, clock, sleep)
    if camera is not None:
        return done(camera, "came_back")
    cycled, why = cycle(record["port"])
    record.update(cycled=cycled, **({"cycle_failed": why} if why else {}))
    if cycled:
        print(f"Wrist camera: power-cycled USB port {record['port']}.")
        camera = _wait_for(open_camera, after_cycle_s, clock, sleep)
        if camera is not None:
            return done(camera, "power_cycle")
    else:
        print(f"Wrist camera: no power cycle: {why}.")
    return done(None, "not_back")


def recovery_summary(events):
    """One line on how wrist camera losses ended, from `events`; None if there were none."""
    records = [e for e in events if e.get("type") == "wrist_camera_recovery"]
    if not records:
        return None
    parts = []
    for outcome in OUTCOMES:
        seconds = [r["seconds"] for r in records if r.get("outcome") == outcome]
        if seconds:
            parts.append(f"{len(seconds)} {OUTCOME_TEXT[outcome]} (median {statistics.median(seconds):.1f}s)")
    return f"{len(records)} losses: " + ", ".join(parts)
