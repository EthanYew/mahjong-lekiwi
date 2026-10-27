"""Gamepad Cartesian teleop, from XLeRobot's examples (gamepad_utils.py, 10_so101_gamepad_ee_control.py),
imported from that checkout so control-mapping changes made there carry over.

Controls (Switch labels): left stick reach / pan, right stick height / wrist roll, L/R pitch,
ZL/ZR gripper close/open, -/+ speed, HOME (hold) back to where teleop started. The right stick moves
height or roll, never both: pulling it down leaked into rx past XLeRobot's 25° snap (2026-10-27).
"""
import importlib.util
import sys

from . import config


def _load_xlerobot():
    for path in (config.XLEROBOT_EXAMPLES, config.XLEROBOT_MODEL):
        if not path.is_dir():
            raise RuntimeError(f"XLeRobot gamepad code not found at {path} (config.XLEROBOT_EXAMPLES)")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import gamepad_utils
    spec = importlib.util.spec_from_file_location(
        "so101_gamepad_ee_control", config.XLEROBOT_EXAMPLES / "10_so101_gamepad_ee_control.py")
    ee_control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ee_control)
    return gamepad_utils, ee_control


class CommittedSticks:
    """A stick moves the arm only after being held the same way for commit_s, then fades in over the
    next commit_s: flicks and quickly corrected wrong-way starts never reach the arm (a low-pass
    only spreads a flick out — same distance travelled). Release still stops at once. Sticks in
    `disabled` read 0. In each `exclusive` pair only one axis moves at a time: the first to move (the
    larger, if both start together) keeps the stick until it is back at centre. Buttons pass through."""

    def __init__(self, gamepad, commit_s, stick_names, disabled=(), exclusive=()):
        self.gamepad, self.commit_s = gamepad, commit_s
        self.stick_names, self.disabled = list(stick_names), set(disabled)
        self.exclusive = [tuple(pair) for pair in exclusive]
        self.reset()

    def reset(self):
        self._held, self._values, self._owner = {}, {}, {}

    def _read(self):
        raw = {name: 0.0 if name in self.disabled else self.gamepad.stick(name) for name in self.stick_names}
        for pair in self.exclusive:
            if not any(raw.get(name) for name in pair):
                self._owner.pop(pair, None)          # back at centre: either axis may take it next
                continue
            owner = self._owner.setdefault(pair, max(pair, key=lambda name: abs(raw.get(name, 0.0))))
            for name in pair:
                if name != owner and name in raw:
                    raw[name] = 0.0
        return raw

    def tick(self, dt):
        values = self._read()
        for name in self.stick_names:
            raw = values[name]
            sign = (raw > 0) - (raw < 0)
            prev_sign, held = self._held.get(name, (0, 0.0))
            held = 0.0 if sign == 0 else (held + dt if sign == prev_sign else dt)
            self._held[name] = (sign, held)
            gain = 1.0 if self.commit_s <= 0 else min(max((held - self.commit_s) / self.commit_s, 0.0), 1.0)
            self._values[name] = raw * gain

    def stick(self, name):
        return self._values.get(name, 0.0)

    def __getattr__(self, attr):
        return getattr(self.gamepad, attr)


EXCLUSIVE_AXES = [("rx", "ry")]       # wrist roll / height


class GamepadTeleop:
    """Drives an Arm from the gamepad. start() seeds from the arm's *commanded* pose — seeding from
    the measured (sagged) pose re-commands the arm lower and it drops at takeover."""

    def __init__(self, device=None, commit_s=0.1, locked_sticks=(), locked_buttons=()):
        self.utils, self.ee = _load_xlerobot()
        self.pad = self.utils.Gamepad(self.utils.find_gamepad(device))
        self.sticks = CommittedSticks(self.pad, commit_s, self.utils.STICK_CODES, disabled=locked_sticks,
                                      exclusive=EXCLUSIVE_AXES)
        self.locked_buttons = set(locked_buttons)
        self.speed_index = self.ee.DEFAULT_SPEED_INDEX
        self.teleop = None

    @property
    def name(self):
        return f"{self.pad.device.name} ({self.pad.device.path})"

    def start(self, arm):
        self.teleop = self.ee.EETeleop(self.ee.SO101Kinematics(), arm.commanded(), arm.joint_limits())
        self.sticks.reset()
        self.pad.just_pressed(self.pad.pressed())     # a button held while waiting isn't a new press

    def touched(self):
        """Any stick off centre (past XLeRobot's deadzone) or any button held: someone has the pad. No
        commit delay, so a DAgger takeover starts on the first push."""
        return (any(self.pad.stick(name) for name in self.utils.STICK_CODES if name not in self.sticks.disabled)
                or bool(self.pad.pressed() - self.locked_buttons))

    def step(self, arm, dt):
        """Read the pad, send the next command, return it."""
        self.sticks.tick(dt)
        buttons = self.pad.pressed() - self.locked_buttons
        new = self.pad.just_pressed(buttons)
        if "plus" in new or "minus" in new:
            self.speed_index = min(max(self.speed_index + ("plus" in new) - ("minus" in new), 0),
                                   len(self.ee.SPEED_LEVELS) - 1)
            print(f"  speed {self.ee.SPEED_LEVELS[self.speed_index]:.2f}x")
        targets = self.teleop.update(arm.measured(), self.sticks, buttons, self.ee.SPEED_LEVELS[self.speed_index], dt)
        command = {j: float(v) for j, v in targets.items()}
        arm.send(command)
        return command

    def close(self):
        self.pad.device.close()
