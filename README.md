# mahjong-lekiwi

An SO101 arm on a LeKiwi base that picks up mahjong tiles: a RealSense (the depth camera looking at
the mat) finds the tile, a closed-loop IK hover (inverse kinematics: joint angles for a target point,
plus a measured correction) puts the gripper over it, and an ACT policy (the learned model) grasps it
from the wrist camera.

## Setup

- `lerobot` conda env (`~/miniforge3/envs/lerobot/bin/python`), with `pytest`.
- XLeRobot checkout at `~/Projects/XLeRobot` (gamepad teleop is imported from it).
- Follower calibration `ethan_lekiwi_arm`; wrist camera at `/dev/wristcam`; RealSense serial in
  `mahjong/config.py`.

Run everything from the repo root with that Python (`python` below means
`~/miniforge3/envs/lerobot/bin/python`). Every script takes `--help`.

## The usual loop

```
collect_corrections.py  →  record.py  →  quality_report.py  →  make_policy.py  →  offline_eval.py  →  bench.py  →  session_report.py
 measure hover error       demos         drop bad episodes      train + register    score offline      test on arm   what happened
```

Hardware scripts connect to everything first (retry / quit if something's unplugged), then ask the
setup questions (daylight, lamp, ceiling lights...), then write a session folder `sessions/<time>_<kind>/`.
`sessions/` belongs to the checkout you run from and isn't in git: run hardware from the main checkout,
not a worktree, or the sessions go when the worktree does.

## Quick reference

| I want to... | Command |
|---|---|
| Check nothing's broken (no hardware, seconds) | `python -m pytest -m "not slow"` |
| Run the bench without the robot | `python scripts/bench.py --sim --trials 3 --tile 8D` |
| Test a policy on the arm, interactively | `python scripts/bench.py` |
| Run N pickup trials on one tile | `python scripts/bench.py --trials 5 --tile 8D` |
| Test a specific policy | `python scripts/bench.py --policy pickup_mat_cl_v1 --trials 5 --tile 8D` |
| Check a policy steers to the tile (vs. a memorised move) | `python scripts/bench.py --trials 15 --tile 4D --hover-perturb-mm 0,6x,-6x,6y,-6y` |
| See what happened in the last session | `python scripts/session_report.py` |
| Compare all bench sessions | `python scripts/session_report.py --all` |
| Record demonstrations | `python scripts/record.py --dataset Yewy/<new_name>` |
| Train a policy | `python scripts/make_policy.py --source Yewy/<name> --version v3 --dry-run` (then drop `--dry-run`) |
| Score a policy offline | `python scripts/offline_eval.py --policy pickup_mat_cl --against pickup_mat_cl_v1` |
| Set up tile positions on the mat | `python scripts/layout.py --name <layout> --teach` |
| Check eval spots are new (or familiar) to a policy | `python scripts/spot_novelty.py <layout>` |
| Record training demos that steer to the tile | `python scripts/record.py --dataset Yewy/<new_name> --hover-jitter-mm 6` |
| Compare two policies across the mat on the arm | `python scripts/bench.py --policy pickup_mat_cl,pickup_mat_cl_v2n --layout <layout> --trials 2` (2 rounds at every spot, both policies each round) |
| Take photos of tiles for the tile detector (no arm) | `python scripts/capture_frames.py` (`--exam` for exam frames) |
| Export the frames the detector struggled with, for labelling | `python scripts/mine_frames.py --list` (then drop `--list`) |
| Score the tile detector on its exam | `python scripts/detector_eval.py --exam-dir data/detector/<exam export>` |
| Retrain the tile detector | `python scripts/train_detector.py --data <export>/data.yaml --name <name> --dry-run` (then drop `--dry-run`) |

Registered policies live in `data/policies.json` (default: `pickup_mat_cl`; also `pickup_mat_cl_v1`, and
`pickup_mat_cl_v2n`: v2's episodes with the gripper narrowing trimmed and the eval episodes left out).

---

## Testing on the arm: `scripts/bench.py`

The main test bench. Detects tiles, hovers over one, runs the policy, checks success, puts the tile back.

```
python scripts/bench.py                                   # interactive menu, default policy
python scripts/bench.py --policy pickup_mat_cl_v1         # interactive, another policy
python scripts/bench.py --trials 5 --tile 8D              # 5 pickup→return trials on 8D, then quit
python scripts/bench.py --sim --trials 3 --tile 8D        # same, all simulated (no hardware)
python scripts/bench.py --trials 15 --tile 4D --hover-perturb-mm 0,6x,-6x,6y,-6y
python scripts/bench.py --policy pickup_mat_cl,pickup_mat_cl_v2n --layout eval_new --trials 2   # paired: 2 rounds x 6 spots x 2 policies = 24 pickups
```

| Option | What it does |
|---|---|
| `--policy ID` | policy from `data/policies.json` (default: the registry's default). `a,b` compares two, paired: rounds alternate a b / b a from the same hover |
| `--layout NAME` | trials walk this layout's spots (`scripts/layout.py`); the report shows success by spot and policy |
| `--trials N` | run N rounds of pickup/return trials, then quit. A round is one attempt per policy, at every spot with `--layout`: pickups = N × policies × spots. Without `--tile` or `--layout`, it asks which tile |
| `--tile NAME` | tile for `--trials`, e.g. `8D` |
| `--sim` | simulated arm, cameras, detection and policy |
| `--hover-perturb-mm LIST` | shift the hover on purpose, cycling through e.g. `0,6x,-6x,6y,-6y` (mm). A policy that steers follows the tile; one that replays a memorised move follows the offset. `session_report.py` prints which. |
| `--device PATH` | gamepad device path (only for teach-discard) |

**Menu commands** (type the number or name): `1 hover`, `2 pickup`, `3 return`, `4 discard`,
`5 teach-discard` (drive the arm over the discard bin with the gamepad and save it), `6 trials`,
`7 rest`, `8 note` (log something you did by hand), `9 setup` (lighting/battery changed mid-session),
`10 quit`.

**During a pickup:** `Esc` stops the policy. `→` marks "something happened here" without stopping it
(a hand on the tile, a caught drop, the arm bumping a neighbour); you're asked what afterwards. A
pickup stops itself once the tile has been held and lifted for 1.5s (max 16s). Every attempt saves the
full wrist video (`wrist.mp4`) in its folder, so a new policy can be replayed on it.

After each attempt the RealSense rescans the mat. If tiles moved, the bench shows the change and
asks: Enter = "as the scan says", or mark it a detection glitch. Every scan also saves the frames the
detector looked at in the session's `scans/` folder, for improving the detector (see below).

## Reading results: `scripts/session_report.py`

```
python scripts/session_report.py                                 # the latest session
python scripts/session_report.py sessions/20261024_101500_bench  # a specific one
python scripts/session_report.py --all                           # every bench session side by side, with its setup
```

Shows the setup, success per tile and policy, stalls, clearances, ACT seconds, interventions (rescues
per 10 attempts, clean vs. assisted successes) and saves a contact sheet (one image row per attempt).

## Recording demonstrations: `scripts/record.py`

The arm hovers over a tile, then you take over with the gamepad. `→` starts recording / saves,
`←` discards, `Esc` ends. After a discard it asks *Same tile, same hover again?*: Enter goes
straight back to the same hover (same tile, same jitter offset, no trip to rest), `n` goes to rest and
asks for a tile. Enter means no after "knocked a tile" or "hover was wrong".

A tile still in the jaws when an episode ends (saved, discarded or Esc) is set back down where the jaws
closed on it, then the arm rests — no putting it back by hand. It's then a little off from where it was
detected, so a discard doesn't offer the same hover again. Move tiles to new places now and then: the arm
puts each one back where it was, and v3 needs spots spread across the mat.

```
python scripts/record.py --dataset Yewy/mahjong_mat_pickup_v3
python scripts/record.py --dataset Yewy/mahjong_mat_pickup_v3 --layout eval_v1 --repeats 2
python scripts/record.py --dataset Yewy/mahjong_mat_eval_v1 --slice new --layout eval_new
python scripts/record.py --dataset Yewy/mahjong_mat_pickup_v3_jitter --hover-jitter-mm 6      # training: random hover offsets
python scripts/record.py --dataset Yewy/mahjong_mat_eval_v1 --slice jitter --layout eval_new --hover-offsets-mm 6x,-6y
```

| Option | What it does |
|---|---|
| `--dataset REPO` | **required**. A new name for a new hover; never append to a dataset recorded from a different hover. (One eval set holds all its slices: each episode's row logs its slice, layout and offset) |
| `--task TEXT` | task string stored with the episodes |
| `--episodes N` | stop after N saved episodes (default 200) |
| `--slice` | `train` (default), or an eval-only set: `familiar`, `new`, `dim`, `tilt`, `jitter`. Eval sets are never trained on |
| `--hover-jitter-mm R` | move each episode's hover by a random offset within R mm (training data: the policy has to steer to the tile). An episode keeps its offset through re-hovers and discards |
| `--jitter-zero-fraction F`, `--jitter-seed N` | with `--hover-jitter-mm`: share of episodes left at the clean hover (default 0.2); random seed |
| `--hover-offsets-mm LIST` | cycle fixed offsets instead (not with `--hover-jitter-mm`), e.g. `6x,-6y` — for the `jitter` eval slice, so it can be recorded again identically |
| `--layout NAME` | guides you to place each tile on a taught spot (outline turns green when it's right) |
| `--repeats N` | with `--layout`: episodes per spot (default 2) |
| `--start-spot ID` | with `--layout`: start at this spot |
| `--hover-gripper PCT` | gripper opening at the hover (default 90%). A narrower start skips the slow narrowing: v2n's trimmed episodes start at 27.2, so record at that to compare with v2n. Changing it means a new dataset |
| `--hover-offset-mm`, `--clearance-mm` | hover position/height tweaks |
| `--no-correction` | hover over the raw detection |
| `--no-view` | no wrist camera window |
| `--no-return` | don't set a held tile back down: it's released at rest, as before |
| `--stick-commit-s S` | a gamepad stick moves the arm only once held that long (default 0.1), so flicks don't. The right stick moves height *or* wrist roll: whichever moves first, until it's back at centre |
| `--device PATH` | gamepad device path (default: the first gamepad found) |

## Reviewing a dataset: `scripts/quality_report.py`

Ranks episodes by hesitation / jerky motion, so you can drop the worst before training. Then the
pickup phases per slice: gripper narrowing, wait at the hover, approach to the close (with its pauses,
reversals and detour: path over the straight line), time after the close, and wobble. Wobble has been
~0.3 mm in every recording: what looks jittery is stop-go corrections, not noise a filter would remove.

```
python scripts/quality_report.py --dataset Yewy/mahjong_mat_pickup_v3
python scripts/quality_report.py --dataset Yewy/mahjong_mat_pickup_v3 --episode 37   # when each reversal happened
python scripts/quality_report.py --dataset Yewy/mahjong_mat_eval_v1 --phases          # phases per episode
python scripts/quality_report.py --dataset Yewy/mahjong_mat_eval_v1 --against Yewy/mahjong_mat_pickup_lekiwi_cl_hover
python scripts/quality_report.py --dataset Yewy/mahjong_mat_pickup_jitter_v3 --drop 8,39 --reason "searching"
```

`--drop EPS` leaves reviewed episodes out of training without deleting them: they're listed in
`data/dropped_episodes.json` (with `--reason`), marked `x` in the table, and `make_policy.py` skips them.
The recording keeps its episode numbers, so you can go on recording into it.

`--against REPO` compares each spot's hover→grasp move with the other recording's at the same place
(within 15 mm), counting only demos that steered. A demo lowered straight down only says the tile was
close enough for the jaws (a third of v2's were). Shifts beyond ~6 mm in one direction at most spots
are worth a `collect_corrections.py` check; a few spots differing is as likely to be how the demos were done.

Options: `--worst N` (rows shown, default 15), `--flag SCORE` (threshold, default 4), `--csv PATH` (save the table),
`--phases`, `--against REPO`, `--drop EPS --reason TEXT`.

## Training a policy: `scripts/make_policy.py`

One command from recorded dataset to registered policy: quality summary → build the training dataset
(trims idle frames, leaves out eval episodes) → upload → train ACT (**hours**) → score offline →
register in `data/policies.json`.

```
python scripts/make_policy.py --source Yewy/mahjong_mat_pickup_v3 --version v3 --dry-run   # check the plan first
python scripts/make_policy.py --source Yewy/mahjong_mat_pickup_v3 --version v3 --default   # registers pickup_mat_cl_v3
```

| Option | What it does |
|---|---|
| `--source REPO`, `--version vN` | **required**: the recorded dataset, and a version tag for output names |
| `--policy-id ID` | registry name (default `pickup_mat_cl_<version>`, like the registered ones) |
| `--default` | make it the default policy for `bench.py` |
| `--dry-run` | print the plan and stop. Always do this first |
| `--trim-narrowing` | also cut the slow gripper narrowing at the start of each episode |
| `--steps N` | training steps (default ~131 passes over the data) |
| `--private` | upload as private (default is public under `Yewy/`) |
| `--skip-build`, `--skip-upload`, `--skip-train` | resume a run partway through |

## Scoring offline: `scripts/offline_eval.py`

Scores a policy on recorded episodes it never trained on (GPU, minutes, no arm). Good for screening;
the bench decides what ships.

```
python scripts/offline_eval.py --policy pickup_mat_cl                                  # scores per slice
python scripts/offline_eval.py --policy pickup_mat_cl --against pickup_mat_cl_v1       # + pass/fail gates vs a reference
python scripts/offline_eval.py --policy pickup_mat_cl --by spot                        # or --by lighting
python scripts/offline_eval.py --show                                                  # what's in the eval split
python scripts/offline_eval.py --make-split Yewy/<training_dataset>                    # hold out 1 per spot, >= 10% (v2: 15/104)
python scripts/offline_eval.py --add-eval-set Yewy/mahjong_mat_eval_v1                 # register an eval-only recording
```

Other options: `--max-episodes N` (quick look), `--stride N` (score every Nth frame, default 3),
`--fraction`/`--seed`/`--force` (for `--make-split`).
Don't change an existing split, or scores stop being comparable.

## Hover correction (measuring where the arm really goes)

The arm sags and the camera calibration isn't perfect, so the hover is corrected by measured errors
stored in `data/hover_corrections.jsonl`. Redo this after recalibrating the RealSense.

```
python scripts/collect_corrections.py           # line the jaws up over each tile with the gamepad, → saves, ← skips, Esc ends
python scripts/collect_corrections.py --raw     # start from the raw detection instead of the current correction
python -m mahjong.hover_correction              # the current fit, per-tile errors
python -m mahjong.hover_correction --sources gamepad,demo:0.5    # what the hover would become with these sources

python scripts/hover_samples.py --dataset Yewy/mahjong_mat_pickup_lekiwi_cl_hover   # import grasp points from demos (repeatable)
python scripts/hover_samples.py --sessions                                          # ...from every bench session
python scripts/hover_samples.py --sessions sessions/20261023_123549_bench           # ...or just these
```

`collect_corrections.py --device PATH` picks the gamepad (default: the first one found).

Which sources are live is set in `data/hover_model.json`. Don't change the live hover for v2-style
policies (they replay a memorised move from it; see `CLAUDE.md`).

## Tile layouts: `scripts/layout.py`

Named sets of target spots on the mat (`data/layouts.json`), drawn on the RealSense view: green =
tile placed right (within 5mm / 6°), amber = close, red = off. Used by `record.py --layout` and
`bench.py --layout`. Commit `data/layouts.json` once a layout is used for a recording (the spots are in
the arm's base frame, so the set can be recorded again); `data/layouts/` (ghost photos, captures) is
only valid for the current camera placement and stays out of git.

```
python scripts/layout.py --list                                   # all layouts (add --name X for its spots)
python scripts/layout.py --name eval_v1 --teach                   # place tiles, d detects them, t saves them as spots
python scripts/layout.py --name eval_v1 --teach --append --prefix N   # add more spots (N1, N2, ...)
python scripts/layout.py --name eval_v1                           # live view: n/p spot, a all, g ghost, c capture, q quit
python scripts/layout.py --name eval_v1 --add T1 --like S2 --yaw 15   # a tilted copy of a spot
python scripts/layout.py --name eval_v1 --add N7 0.24 0.17        # a spot at x y (metres, arm base frame)
python scripts/layout.py --name eval_v1 --remove N7
```

In the live view, `c` captures every spot at once (averaged over 30 frames): offset, yaw, how much each
wobbles, and whether the yaw reading drifts across the mat. It prints a table and saves
`data/layouts/<name>_capture_<time>.json`. Outlines are smoothed over ~10 frames, and a green spot
stays green up to 1.5× the tolerance, so a tile at the edge doesn't flicker.

`--sim` uses a rendered frame instead of the camera; `--note` describes the layout when teaching.

### Is a spot new to the policy? `scripts/spot_novelty.py`

Each spot's distance to the nearest tile in a training recording's episode log: **familiar** within
10 mm, **new** from 25 mm, "near a demo" in between. v2's rows are ~6 cm apart along y, so the newest
reachable spots are between the rows (x ≈ 18-19 cm).

```
python scripts/spot_novelty.py                         # every layout, against v2's recording
python scripts/spot_novelty.py eval_familiar eval_new
python scripts/spot_novelty.py eval_new --against Yewy/mahjong_mat_pickup_v3_jitter   # repeatable
```

## The tile detector: photos, labels, exam, retraining

The RealSense finds tiles with a YOLO detector (`assets/tile_detector.pt`). It was trained in July on
a different table and room, so it sometimes misses a tile or gives it the wrong name. Improving it is
a loop:

```
capture_frames.py / bench.py  →  mine_frames.py  →  Roboflow  →  train_detector.py  →  detector_eval.py
 save what it saw                 pick + pre-label     correct      fine-tune             score on the exam
```

**1. Photos.** Every bench scan saves its 5 frames, the depth image and every box YOLO found (down to
0.25 confidence) in `sessions/<session>/scans/<n>_<label>/`. `capture_frames.py` does the same without
the arm: a live view with the detector's boxes (green at or above the 0.5 cut, amber below), space
takes a scan, `l` says the lighting changed, `q` quits. The keys and what to change before each scan
are on screen, and under the view a panel counts every tile name scanned so far (as the detector read
it): red none yet, amber below `--target` (default 3), green there. A scan whose tiles barely moved
since the last one is flagged in amber: move or swap 5-8 tiles before the next.

```
python scripts/capture_frames.py             # photos to train on
python scripts/capture_frames.py --exam      # photos for the exam (one session per lighting)
python scripts/capture_frames.py --target 5  # aim for 5 of each name
```

Good captures: 12-15 face-up tiles from several suits across the whole mat (edges and corners too),
one or two scans per arrangement, and a changed scene before every scan (move or swap tiles, turn a
few, push some together). Many scans of one unchanged scene add little: they're near copies.

**2. The exam.** A fixed set of labelled photos of the current scene that is **never trained on**, so
scores mean something. Its sessions are listed in `data/detector_exam.json` (commit it). Take one
`--exam` session per lighting (daylight, lamp, ceiling lights), about 40-60 scans in all, with tile
arrangements of their own. Export them with `mine_frames.py --exam`, label them in a **separate**
Roboflow project, and download that as a YOLO export (e.g. to `data/detector/exam_labelled/`).
Score the live detector first: that's the baseline.

```
python scripts/detector_eval.py --exam-dir data/detector/exam_labelled                # the live detector
python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --by lighting
python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --weights outputs/detector/v2/weights/best.pt --against assets/tile_detector.pt
```

It reports **found** (labelled tiles it put a box on), **named right** (of those found), missed
tiles, extra boxes on nothing, and its commonest mix-ups (e.g. `4D->5D x6`). Scores are per frame and
go in `data/detector_eval/`. Frames in the folder that aren't from an exam session are left out.

**3. Mining and retraining.** `mine_frames.py` picks the frames worth labelling: a tile whose name
changed across a scan's 5 frames (flicker), a tile seen in some frames and not others (dropout), a
tile below 0.7 confidence (low), or a scan you called a detection glitch at the bench. `--all` adds a
plain frame from every other scan (up to `--per-session`, default 15), since the detector has never
seen this scene at all. Exam and simulated sessions are never exported for training.

```
python scripts/mine_frames.py --list                    # what it would take
python scripts/mine_frames.py --all --since 20261027    # export to data/detector/mined_<time>/
```

Upload the export to the Roboflow training project (the July project, "Mahjong tiles on table"): its
`labels/` holds the detector's own boxes as a start, and `manifest.csv` says why each frame was
picked. Correct the boxes, make a new dataset version, and download it as a YOLO export. Then:

```
python scripts/train_detector.py --data data/detector/train_v2/data.yaml --name v2 --dry-run   # counts, exam check
python scripts/train_detector.py --data data/detector/train_v2/data.yaml --name v2             # GPU, ~an hour
python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --weights outputs/detector/v2/weights/best.pt --against assets/tile_detector.pt
```

`train_detector.py` starts from the live weights with their July recipe (1280 px, no mirror flips: a
mirrored tile face isn't a real tile) and refuses a dataset with any exam frame in it. Ship the new
weights only if they beat the old ones on the exam: copy `best.pt` to `assets/tile_detector.pt` and
commit it with the scores. Exports and downloads under `data/detector/` aren't in git; Roboflow keeps
the labels.

## Tests

```
python -m pytest -m "not slow"    # fast: motion, kinematics, correction, dataset trims, bench (no hardware)
python -m pytest                  # everything; slow tests load policies and need the GPU
```

Git hooks run the fast tests (~20 s) on every commit and every merge, so a broken invariant, or two
branches that break each other, can't be committed. Turn them on once per clone:
`git config core.hooksPath scripts/hooks`. Merging worktrees: see "Worktrees and merging" in `CLAUDE.md`.

Run the fast tests before calling a change hardware-ready (and the slow ones too if `policy.py` or
`dataset.py` changed). See `CLAUDE.md` for the rules the code follows and why.
