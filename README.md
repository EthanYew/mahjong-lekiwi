# mahjong-lekiwi

A low-cost robot arm that picks up, reads and plays mahjong tiles. An SO101 arm on a LeKiwi mobile base finds a tile
with a depth camera, moves its gripper over it, and grasps it with a learned policy driven by its wrist camera.

## How it works

```
 RealSense, above the mat       Inverse kinematics             ACT policy, wrist camera
 ┌──────────────────────┐      ┌───────────────────────┐      ┌───────────────────────┐
 │ find and name every  │ ───▶ │ hover over the tile,  │ ───▶ │ learned grasp: close  │
 │ tile on the mat      │      │ corrected, from above │      │ on the tile and lift  │
 └──────────────────────┘      └───────────────────────┘      └───────────────────────┘
```

1. **Find.** A YOLO tile detector on the RealSense image locates every tile on the mat and names it.
2. **Hover.** Inverse kinematics (joint angles for a target point) puts the gripper above the tile. A correction model,
   fitted from gamepad corrections on the real arm, removes the arm's systematic error.
3. **Grasp.** An [ACT](https://tonyzhaozh.github.io/aloha/) policy, trained with [LeRobot](https://github.com/huggingface/lerobot)
   on recorded demonstrations and DAgger corrections (a person taking over when the policy goes wrong), closes on the tile.
4. **Play.** Game logic decides what to keep and discard. A second read from the wrist camera checks tile names, and the
   base camera reads face-down tiles the arm holds up after drawing them from the wall.

## Hardware

- SO101 follower arm on a LeKiwi base
- Intel RealSense depth camera above the mat
- USB wrist camera on the gripper
- The LeKiwi's front camera (the "base camera")
- A gamepad, for teleoperation, corrections and taking over a pickup

## Getting started

### Requirements

- Linux with a CUDA GPU (training and the slow tests)
- A conda env with [LeRobot](https://github.com/huggingface/lerobot) installed, plus `pytest`
- An [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) checkout at `~/Projects/XLeRobot` (its gamepad teleop is imported)
- Your arm's calibration ID, camera paths and RealSense serial set in `mahjong/config.py`

### Install

```bash
git clone https://github.com/EthanYew/mahjong-lekiwi.git
cd mahjong-lekiwi
git config core.hooksPath scripts/hooks     # runs the fast tests before each commit
```

Run every script from the repo root with the LeRobot env's Python. Every script takes `--help`. The detector and
classifier weights (`assets/`) and the committed state the scripts read (`data/`) come with the clone; the grasp
policies download from the Hugging Face Hub, and some are private.

### Try it without the robot

```bash
python -m pytest -m "not slow" -n 8                    # fast tests, no hardware
python scripts/bench.py --sim --trials 3 --tile 8D     # a pickup bench in simulation
python scripts/play.py --sim                           # a hand of mahjong in simulation
python scripts/status.py                               # the current default policy, bench results and evidence
```

### On the arm

```bash
python scripts/setup_check.py                          # check the workspace matches the saved reference
python scripts/bench.py --trials 5 --tile 8D           # test pickups on one tile
python scripts/play.py --layout even-hand13            # play a hand
```

## The development loop

```
collect_corrections.py → record.py → quality_report.py → make_policy.py → offline_eval.py → bench.py → session_report.py
 measure hover error      demos       drop bad episodes    train + register  score offline    test on arm  what happened
```

Each hardware run writes a session folder under `sessions/` (not in git). `session_report.py` summarises it.

## Project structure

| Path | What's there |
|---|---|
| `mahjong/` | The library: motion (`arm.py`, `kinematics.py`), perception (`detector.py`, `camera.py`), policies (`policy.py`), game logic (`game.py`) |
| `scripts/` | Everything a person runs: benches, recording, training, review tools |
| `data/` | Committed state: registered policies, layouts, corrections, evaluation results |
| `assets/` | Trained weights for the detector and classifiers, and the camera calibration |
| `tests/` | Fast tests (no hardware) and slow tests (load policies, need a GPU) |
| `docs/` | Plans, the scorecard and the full usage guide |

## Documentation

- **[docs/usage.md](docs/usage.md)**: the full command reference, with every script and option, and how each step is run
- [docs/code_map.md](docs/code_map.md): what each file is for
- [ROADMAP.md](ROADMAP.md): what's done and what's next, each item with its evidence
- [docs/scorecard.md](docs/scorecard.md): how a new policy is judged against the current one
- [docs/policy_history.md](docs/policy_history.md): how each policy version was trained and what its data captured

## Results

Live numbers come from `data/`, not from this file: `python scripts/status.py` prints the current default policy with
its evidence and the bench totals for each policy. [ROADMAP.md](ROADMAP.md)'s status table records each milestone with
the session that showed it.

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot) and the SO101 arm, the
[LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) base, [XLeRobot](https://github.com/Vector-Wangel/XLeRobot),
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [Roboflow](https://roboflow.com) for tile labelling.
Development was done with [Claude Code](https://claude.com/claude-code).
