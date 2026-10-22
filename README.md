# mahjong-lekiwi

An SO101 arm on a LeKiwi base that picks up mahjong tiles: a RealSense finds the tile, a closed-loop
IK hover puts the gripper over it, and an ACT policy grasps it from the wrist camera.

## Setup

- `lerobot` conda env (`~/miniforge3/envs/lerobot/bin/python`), with `pytest`.
- XLeRobot checkout at `~/Projects/XLeRobot` (gamepad teleop is imported from it).
- Follower calibration `ethan_lekiwi_arm`; wrist camera at `/dev/wristcam`; RealSense serial in
  `mahjong/config.py`.

## Workflow

```
python scripts/collect_corrections.py                  # measure the hover's error (once per calibration)
python -m mahjong.hover_correction                     # check the fit
python scripts/record.py --dataset Yewy/<name>         # record gamepad demonstrations
python scripts/quality_report.py --dataset Yewy/<name> # review hesitant episodes
python scripts/make_policy.py --source Yewy/<name> --version v3 --dry-run   # then without --dry-run
python scripts/bench.py --policy <id> --trials 5 --tile 8D                  # test it
python scripts/session_report.py                       # what happened
```

`python -m pytest -m "not slow"` checks the motion, kinematics, correction, dataset trims and bench
without hardware. See `CLAUDE.md` for the rules the code follows and why.
