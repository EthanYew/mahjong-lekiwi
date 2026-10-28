"""Connect the hardware a script needs before anything else happens.

Every device is tried; anything missing is listed with what to do about it, and you can plug it in
and retry, or quit — before a session folder is created, so a run that never started leaves nothing
behind. (2026-10-23: a bench run with the LeKiwi unplugged died in lerobot's "Full found motor list:
{}" and left an empty session.)
"""
import os
from pathlib import Path

from . import config


class NotReady(Exception):
    """A device that isn't there or isn't answering, with what to do about it."""


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


def open_wrist_camera():
    from .camera import WristCamera
    if not os.path.exists(config.WRIST_CAM):
        raise NotReady(f"no wrist camera at {config.WRIST_CAM}: plug in / replug the wrist camera's USB")
    camera = WristCamera()
    ok, _ = camera.read()
    if not ok:
        camera.release()
        raise NotReady("the wrist camera opened but gives no frames: replug it")
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


OPENERS = {"arm": open_arm, "wrist_camera": open_wrist_camera, "realsense": open_realsense, "gamepad": open_gamepad}
CLOSE = {"arm": lambda a: a.disconnect(), "wrist_camera": lambda c: c.release(), "realsense": lambda r: r.close(),
         "gamepad": lambda g: g.close()}


def reopen_wrist_camera(camera, ask=None, openers=None):
    """Release a wrist camera that stopped giving frames (camera.LOST_S) and open it again; if it won't
    open, the usual retry / quit (connect). Returns the new camera; a simulated one comes back as is."""
    if getattr(camera, "simulated", False):
        return camera
    try:
        camera.release()
    except Exception:
        pass
    print("Wrist camera lost: reopening it.")
    return connect(["wrist_camera"], ask=ask, openers=openers)["wrist_camera"]


def connect(needed, gamepad_kwargs=None, ask=None, openers=None):
    """Open every device in `needed` (names from OPENERS). Returns {name: device}. Lists what's missing
    and offers retry (Enter) or quit (q, raises SystemExit after closing whatever did open)."""
    ask = ask or input
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
