"""Connect the hardware a script needs before anything else happens.

Every device is tried; anything missing is listed with what to do about it, and you can plug it in
and retry, or quit — before a session folder is created, so a run that never started leaves nothing
behind. (2026-10-23: a bench run with the LeKiwi unplugged died in lerobot's "Full found motor list:
{}" and left an empty session.)
"""
import os
import time
from pathlib import Path

from . import config, wrist_usb


class NotReady(Exception):
    """A device that isn't there or isn't answering, with what to do about it."""


class WrongResolution(NotReady):
    """A camera giving frames of the wrong size: a USB link stuck slow, which only a replug fixes, so a reopen's wait for
    it to come back by itself ends at once (wrist_usb._wait_for) rather than tell the person not to replug it for 10s."""
    waiting_wont_fix = True


def open_arm():
    from .arm import Arm
    if not Path(config.FOLLOWER_PORT).exists():
        raise NotReady(f"no USB port {config.FOLLOWER_PORT}: plug in the LeKiwi board's USB cable")
    try:
        return Arm.connect()
    except RuntimeError as e:          # lerobot: the port opened but the motors didn't answer
        raise NotReady("the LeKiwi board is connected but the arm's motors aren't answering: switch the "
                       "LeKiwi on (battery) and check the arm's servo cable") from e
    except ConnectionError as e:
        raise NotReady(f"couldn't open {config.FOLLOWER_PORT} (another program using it?)") from e


def check_resolution(camera, frame, path, sys_bus="/sys/bus/usb/devices"):
    """Raise NotReady (releasing the camera) unless `frame` is config.WRIST_RESOLUTION. On 2026-11-12 a replugged wrist
    camera came back on a 12 Mbit/s USB link giving 160x120 magenta frames; it opened and read fine, so the reopen took
    it, and the policy ran on them to its time cap (20261112_155318_holdup, 050)."""
    width, height = config.WRIST_RESOLUTION
    if frame is None or frame.shape[:2] == (height, width):
        return
    camera.release()
    port = wrist_usb.port_of(path)
    speed = wrist_usb.link_speed(port, sys_bus) if port else None
    link = f", USB link at {speed:g} Mbit/s instead of 480" if speed is not None and speed < 480 else ""
    raise WrongResolution(f"the {camera.label} gives {frame.shape[1]}x{frame.shape[0]} frames, not {width}x{height}{link}: "
                   "unplug it and plug it back in firmly (a slow link means the cable or its plug isn't making good contact)")


def open_wrist_camera():
    from .camera import WristCamera
    if not os.path.exists(config.WRIST_CAM):
        raise NotReady(f"no wrist camera at {config.WRIST_CAM}: plug in / replug the wrist camera's USB")
    camera = WristCamera()
    ok, frame = camera.read()
    if not ok:
        camera.release()
        raise NotReady("the wrist camera opened but gives no frames: replug it")
    wrist_usb.remember_port()          # before the size check: a slow camera is still on this port
    check_resolution(camera, frame, config.WRIST_CAM)
    return camera


def open_realsense():
    import pyrealsense2 as rs
    serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
    if config.REALSENSE_SERIAL not in serials:
        raise NotReady(f"RealSense {config.REALSENSE_SERIAL} not found (seen: {serials or 'none'}): plug it in")
    from .perception import RealSense
    return RealSense()


def open_gamepad(**kwargs):
    from .gamepad import GamepadTeleop
    try:
        return GamepadTeleop(**kwargs)
    except RuntimeError as e:
        raise NotReady(f"{e} — switch the controller on / pair it") from e


def open_base_camera():
    from .camera import BaseCamera, read_within
    if not os.path.exists(config.BASE_CAM):
        raise NotReady(f"no base camera at {config.BASE_CAM}: plug it in, and check `ls -l /dev/basecam /dev/wristcam` "
                       "(docs/face_down_plan.md, 'The camera fix')")
    camera = BaseCamera()
    ok, frame = read_within(camera)
    if not ok:
        camera.release()
        raise NotReady("the base camera opened but gives no frames: replug it")
    check_resolution(camera, frame, config.BASE_CAM)
    return camera


OPENERS = {"arm": open_arm, "wrist_camera": open_wrist_camera, "realsense": open_realsense, "gamepad": open_gamepad,
           "base_camera": open_base_camera}
CLOSE = {"arm": lambda a: a.disconnect(), "wrist_camera": lambda c: c.release(), "realsense": lambda r: r.close(),
         "gamepad": lambda g: g.close(), "base_camera": lambda c: c.release()}


def _settled(open_camera):
    """A camera that keeps giving frames for wrist_usb.SETTLE_S: a flapping one opens fine and drops again at once."""
    from .camera import read_within
    camera = open_camera()
    end = time.monotonic() + wrist_usb.SETTLE_S
    while time.monotonic() < end:
        if not read_within(camera)[0]:
            camera.release()
            raise NotReady("the wrist camera dropped again straight away")
    return camera


def _log_recovery(record):
    from .session import Session
    print(f"Wrist camera recovery: {record['outcome']} after {record['seconds']}s.")
    if Session.current is not None:
        Session.current.event("wrist_camera_recovery", **record)


def reopen_wrist_camera(camera, ask=None, openers=None):
    """Release a wrist camera that stopped giving frames (camera.LOST_S) and get it back: give it a moment, then the
    usual retry / quit (connect). No power cycle: it has never recovered one, 0 of 9 (wrist_usb). Each one logs how it
    ended (wrist_usb). Returns the new camera; a simulated one comes back as is."""
    if getattr(camera, "simulated", False):
        return camera
    try:
        camera.release()
    except Exception:
        pass
    print(f"Wrist camera lost: waiting up to {wrist_usb.RETURN_WAIT_S:.0f}s for it to come back. Don't replug it yet.")
    from .perception import drain_stdin
    drain_stdin()          # keys pressed while it was lost would answer connect's retry prompt
    opener = (openers or OPENERS)["wrist_camera"]
    recovered, record = wrist_usb.recover(lambda: _settled(opener))
    if recovered is None:
        drain_stdin()      # and so would keys pressed during the wait
        start = time.monotonic()
        try:
            recovered = connect(["wrist_camera"], ask=ask, openers=openers, recover_wrist=False)["wrist_camera"]
        except SystemExit:
            _log_recovery({**record, "outcome": "quit", "seconds": round(record["seconds"] + time.monotonic() - start, 2)})
            raise
        record = {**record, "outcome": "person", "seconds": round(record["seconds"] + time.monotonic() - start, 2)}
    _log_recovery(record)
    return recovered


def connect(needed, gamepad_kwargs=None, ask=None, openers=None, recover_wrist=True):
    """Open every device in `needed` (names from OPENERS). Returns {name: device}. Lists what's missing
    and offers retry (Enter) or quit (q, raises SystemExit after closing whatever did open). A wrist camera that's gone
    from USB gets one port power cycle before that (never one that's there: another program may be using it)."""
    ask = ask or input
    if recover_wrist and openers is None and "wrist_camera" in needed and not os.path.exists(config.WRIST_CAM):
        print(f"No wrist camera at {config.WRIST_CAM}: trying a power cycle of its USB port.")
        camera, _ = wrist_usb.recover(OPENERS["wrist_camera"], wait_s=0)
        if camera is not None:
            camera.release()
    openers = openers or OPENERS
    devices = {}
    while True:
        problems = []
        for name in needed:
            if name in devices:
                continue
            try:
                kwargs = (gamepad_kwargs or {}) if name == "gamepad" else {}
                devices[name] = openers[name](**kwargs)
            except NotReady as e:
                problems.append(f"  {name}: {e}")
            except Exception as e:     # anything unexpected: still offer a retry rather than a traceback
                problems.append(f"  {name}: {type(e).__name__}: {e}")
        if not problems:
            return devices
        print("Not ready:\n" + "\n".join(problems))
        if ask("Fix that, then Enter to retry (q to quit): ").strip().lower() in ("q", "quit"):
            for name, device in devices.items():
                try:
                    CLOSE[name](device)
                except Exception:
                    pass
            raise SystemExit("Quit before starting — no session was created.")
