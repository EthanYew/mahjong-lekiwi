---
paths:
  - "mahjong/gamepad.py"
  - "mahjong/handoff_trace.py"
  - "scripts/collect_corrections.py"
  - "scripts/handoff_report.py"
  - "scripts/record.py"
---
Teleop and what the sticks do. Evidence in `docs/invariant_notes.md`.

- **The right stick moves height or wrist roll, never both** (`gamepad.EXCLUSIVE_AXES`): a pull down leaked
  into roll past XLeRobot's 25° snap and turned the wrist mid-descent (2026-10-27, 7B).
- **Hand corrections record the commanded tool point, not where the jaws were — and that's fine.** A
  settled hover sags the same way the lined-up arm does, so the commanded fit already carries it (a refit on
  the measured point moves the hover only 0-4mm); the noise is the lining up (~9mm RMS, same tile twice).
  Every gamepad handoff uses a 0.35 deadzone (`gamepad.DEADZONE`) and warns of a stick off centre
  (`gamepad.warn_off_centre`): a resting thumb past XLeRobot's 0.15 pulled the arm 87mm. Teleop is traced
  (`mahjong/handoff_trace.py`, `scripts/handoff_report.py`). Evidence: `docs/invariant_notes.md`.
- **Stick speed is capped at the jaws, by scaling the stick's whole travel** (`gamepad.cap_sticks`: left stick 30mm/s,
  height 50). XLeRobot gives pan a fixed deg/s, ~130mm/s at the jaws at full reach and 0.5x, so the left stick lurched
  once past the deadzone. Clipping the top would keep that steep start. Demos descend at 50, so that cap costs them
  nothing. The L/R tilt buttons (~92mm/s at the tool point) and wrist roll aren't capped. Unconfirmed on the arm.
