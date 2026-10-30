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
| Check nothing's broken (no hardware, ~11 s) | `python -m pytest -m "not slow" -n 8` |
| See where things stand (default policy and why, bench totals, layouts, the drop zone, this checkout) | `python scripts/status.py` |
| Approve and merge a branch's PR, then clean up after it | `python scripts/merge_pr.py` (lists them), then `python scripts/merge_pr.py <number>` |
| Make another policy the default | `python scripts/status.py --set-default <id> --evidence "sessions/<bench session>: <scores>" --why "..."` |
| Check the workspace is set up as before (after packing up / moving things) | `python scripts/setup_check.py` (once, on a good setup: `--save-reference`) |
| Run the bench without the robot | `python scripts/bench.py --sim --trials 3 --tile 8D` |
| Test a policy on the arm, interactively | `python scripts/bench.py` |
| Run N pickup trials on one tile | `python scripts/bench.py --trials 5 --tile 8D` |
| Test a specific policy | `python scripts/bench.py --policy pickup_mat_cl_v1 --trials 5 --tile 8D` |
| Check a policy steers to the tile (vs. a memorised move) | `python scripts/bench.py --trials 15 --tile 4D --hover-perturb-mm 0,6x,-6x,6y,-6y` |
| See what happened in the last session | `python scripts/session_report.py` |
| See how the arm sagged / drifted while lining up corrections | `python scripts/handoff_report.py` |
| See how well returns land tiles back where they were | `python -m mahjong.return_correction` |
| Compare all bench sessions | `python scripts/session_report.py --all` |
| Check the hand's slots and the drop zone on the bench before playing | `python scripts/bench.py --layout hand14 --trials 2` (a tile on every spot) |
| Play a hand on its own (you only put drawn tiles on the drop zone) | `python scripts/play.py --layout hand13` |
| Play a hand without the robot | `python scripts/play.py --sim` |
| Measure the hover's real error with the gamepad (first step of the loop; after a recalibration) | `python scripts/collect_corrections.py` |
| Record demonstrations | `python scripts/record.py --dataset Yewy/<new_name>` |
| Find the hesitant / jerky demos and leave them out of training | `python scripts/quality_report.py --dataset Yewy/<name>` (then `--drop 8,39 --reason "..."`) |
| Add hover-correction samples from demos or older bench sessions | `python scripts/hover_samples.py --dataset Yewy/<name>` / `--sessions` |
| Add return-correction samples from older bench sessions | `python scripts/return_samples.py` |
| Train a policy | `python scripts/make_policy.py --source Yewy/<name> --version v3 --dry-run` (then drop `--dry-run`) |
| Score a policy offline | `python scripts/offline_eval.py --policy pickup_mat_cl --against pickup_mat_cl_v1` |
| Set up tile positions on the mat | `python scripts/layout.py --name <layout> --teach` |
| Check eval spots are new (or familiar) to a policy | `python scripts/spot_novelty.py <layout>` |
| Record training demos that steer to the tile | `python scripts/record.py --dataset Yewy/<new_name> --hover-jitter-mm 6` |
| Correct the default policy where it goes wrong (DAgger) | `python scripts/record.py --dataset Yewy/mahjong_mat_dagger_v6 --policy pickup_mat_cl_v6` (press A on the gamepad to take over) |
| Train on the corrections too | `python scripts/make_policy.py --source Yewy/<name> --add Yewy/mahjong_mat_dagger_v3c --add Yewy/<new dagger set> --version <next> --dry-run` |
| Compare two policies across the mat on the arm | `python scripts/bench.py --policy pickup_mat_cl,pickup_mat_cl_v2n --layout <layout> --trials 2` (2 rounds at every spot, both policies each round) |
| Take photos of tiles for the tile detector (no arm) | `python scripts/capture_frames.py` (`--exam` for exam frames) |
| Export the frames the detector struggled with, for labelling | `python scripts/mine_frames.py --list` (then drop `--list`) |
| Score the tile detector on its exam | `python scripts/detector_eval.py --exam-dir data/detector/<exam export>` |
| Photograph every tile on the mat with the wrist camera from jittered hovers | `python scripts/wrist_capture.py` (rings when a set is done) |
| Check the wrist photos of each tile name (mark wrong ones, rename misfiled ones) | `python scripts/wrist_review.py` |
| Re-split a Roboflow download so no scene is in train and valid | `python scripts/split_by_scene.py --data <download>/data.yaml --out data/detector/train_<v> --list` (then drop `--list`) |
| Retrain the tile detector | `python scripts/train_detector.py --data <export>/data.yaml --name <name> --dry-run` (then drop `--dry-run`) |

Registered policies live in `data/policies.json` (default: `pickup_mat_cl_v6`, v3c's hover-jittered demos,
DAgger corrections of v3c and of v4 at 4D/7D, and 33 green-mat demos, trained with `--augment --replace-mat 0.7`;
also `pickup_mat_cl_v5` (without the green demos or augmentation; fails on the green mat), `pickup_mat_cl_v4`
(without v4's corrections),
`pickup_mat_cl_v3c`, `pickup_mat_cl_v3b`, `pickup_mat_cl_v3` (demos from jittered
hovers, so they'd steer to the tile; the perturbation test says only partly, ROADMAP 1), `pickup_mat_cl` (v2),
`pickup_mat_cl_v1`, and `pickup_mat_cl_v2n`: v2's episodes with
the gripper narrowing trimmed and the eval episodes left out). Why the default is the default, and how each
did on the bench: `python scripts/status.py` (tests/test_docs.py keeps this paragraph in step).

---

## Checking the setup after a pack-up: `scripts/setup_check.py`

The policies, the hover correction and the return correction assume the RealSense, the arm and the table
are where they were when they were recorded and fitted. A few mm off and every hover lands off-target with
no error. `setup_check.py` compares the setup with a reference taken once on a setup you trust:

| Check | What it measures | Warn / fail |
|---|---|---|
| camera | where the LeKiwi base and the folded arm appear in the RealSense image (1px ≈ 1.5mm on the mat) | 2 / 5mm, 0.5 / 1.5° |
| table | the mat's height and tilt from the RealSense depth, vs the reference | 3 / 6mm, 2 / 4mm per 10cm |
| tiles | a tile placed to green on 3 spots; the arm hovers over its measured position with the hover correction stored in the reference; where the tile sits in the wrist view. Graded on the median spot; its angle is shown, not graded | median 4 / 7mm, size 10 / 20% |
| brightness | the mat (RealSense) and the wrist view, vs the reference | outside 0.7-1.4x |
| parked | where the arm settles when switched off at the end (a recalibrated servo shows here) | 3 / 6° |

```
python scripts/setup_check.py --save-reference     # once, on a good setup (e.g. right after a good bench session); commit data/setup_reference/
python scripts/setup_check.py                      # after every setup; exits 1 if a check fails
python scripts/setup_check.py --skip-tiles         # camera, table, parked pose only
python scripts/setup_check.py --sim --save-reference && python scripts/setup_check.py --sim   # no hardware
```

If the camera check isn't green, a window blends the reference with the live view and draws an arrow: nudge
the tripod (or the LeKiwi) until the box turns green, then Enter. Moving it back keeps `T_cam_to_base` and
both corrections valid; recalibrating sets the corrections aside. If a tile isn't found (by the RealSense
at its spot, or in the wrist view at the hover), the check asks you to reposition it: the placing window
opens again to chase green, then it looks again (Enter), or `s` skips the spot, `q` stops the tile test.

The tile test asks only whether the RealSense and the arm still agree about where a tile is, so it leaves out
what legitimately changes:
- **The hover correction.** The reference stores the correction it hovered with at each spot, and a check uses
  that one, so refitting the correction doesn't move the tile in the wrist view. How far the live correction
  now differs at each spot is printed, not graded. (Refitting it from 23 to 27 samples moved the hover 4-8mm at
  the check spots, about what the first real check reported, on 2026-10-29.)
- **How the tile was placed.** The hover follows the tile's measured position, so placement within green
  cancels. Its angle in the wrist view (green allows ±6°) is shown next to how differently the RealSense saw it
  placed, and isn't graded.
- **Noise at one spot.** The test is graded on the median spot: a moved camera or arm shifts them all. One spot
  past the line makes it a warning.

Saving a reference shows each wrist frame with the tile it found crossed: Enter if the cross is on the tile, `r`
to reposition and look again. A reference saved before this asks to be saved again.

Each run writes `sessions/<time>_setup/`
(the RealSense frame, `camera_vs_reference.jpg`, each spot's `wrist_<spot>_vs_reference.jpg`, and a
`setup_check` event).

| Option | What it does |
|---|---|
| `--save-reference` | measure this setup and save it as the reference (asks before replacing one) |
| `--layout NAME`, `--spots IDS` | with `--save-reference`: the check spots (default `eval_familiar`, `F2,F3,F6`); a check uses the reference's |
| `--skip-tiles` | no tile test |
| `--reference DIR` | another reference folder (default `data/setup_reference`; `sessions/sim_setup_reference` with `--sim`) |
| `--no-view` | no alignment window |
| `--sim` | simulated arm, cameras and mat |

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
| `--trials N` | run N rounds of pickup/return trials, then quit. A round is one attempt per policy, at every spot with `--layout`: pickups = N × policies × spots. Without `--tile` or `--layout`, it asks which tile. Every return aims the tile at its home (see below), so it doesn't walk off |
| `--tile NAME` | tile for `--trials`, e.g. `8D` |
| `--sim` | simulated arm, cameras, detection and policy |
| `--hover-perturb-mm LIST` | shift the hover on purpose, cycling through e.g. `0,6x,-6x,6y,-6y` (mm). A policy that steers follows the tile; one that replays a memorised move follows the offset. `session_report.py` prints which. |
| `--device PATH` | gamepad device path (only for teach-discard) |

**Menu commands** (type the number or name): `1 hover`, `2 pickup`, `3 return`, `4 discard`,
`5 teach-discard` (drive the arm over the discard bin with the gamepad and save it), `6 trials`,
`7 rest`, `8 note` (log something you did by hand), `9 setup` (lighting/battery/surface changed mid-session),
`10 quit`.

**A tile the trials can't find** (not detected by name, or nothing near a `--layout` spot) is first looked
at with the wrist camera: the arm hovers where it was first seen and checks for a tile face between the jaw
tips. If it's there (the detector flickered) the pickup goes ahead from there. If not, it stops for you:
put it back where it belongs, Enter rescans (logged as a reset), `s` skips the spot (`--tile`: ends the
trials). An unattended run waits there.

**While the arm picks up or returns a tile, say what you see with a key** (no Enter; the terminal prints
`[b] bumped another tile (noted)`; the keys are printed as each pickup and return starts). Keys typed at a
prompt or the menu don't count.

| Key | Means | Counts as |
|---|---|---|
| `b` | the arm bumped another tile | a collision (failure; the success stays clean), and a neighbour that moved is a knock without asking |
| `p` | the arm pushed or turned the tile it's picking up | a failure |
| `l` | look at this (nothing touched) | an observation |

`Esc` stops the policy. `→` marks "something happened here" without stopping it (a hand on the tile, a
caught drop); a `→` with no key is asked about right after that attempt (not at the end of a batch). A pickup stops itself once the tile
has been held and lifted for 1.5s (max 16s). Every attempt saves the full wrist video (`wrist.mp4`) in
its folder, so a new policy can be replayed on it.

After each pickup and return the RealSense rescans the mat. A tile the scan says went is looked at with the
wrist camera first (once the arm has put down what it holds); one still there is logged as a detection
glitch, with the wrist frame in the session's `wrist_checks/`. What else changed is settled like this:

- **a neighbour** (within 85mm of the tile the arm handled) **moved or turned**: if you pressed `b` during
  that pickup or return, it's a knock; if not, you're asked right then (Enter = it moved, or a glitch);
- **anything else** (tiles further away, tiles gone or appearing, changes between attempts): no question.
  The next scans decide: undone by a later scan, it was the detector (a glitch); still so 3 scans later
  (about one attempt), it happened: a knock near the arm, "moved by something else" further away, a reset
  between attempts. The end of the trials lists the tiles whose detections flickered (`3B x2`).

Every scan also saves the frames the detector looked at in the session's `scans/` folder, for improving
the detector (see below).

**Returns go home.** A tile's home is where it was first seen this session (per layout spot, or per
tile name; a tile found more than 40mm from its home was moved on purpose and gets a new one; moved
on purpose by less, the next return puts it back, so restart the bench to give it a new home there). A return
aims the tile at its home, not where the jaws closed, so errors don't add up round after round, using
the return correction (below). The scan after a return checks where it landed: over 12mm from home it
says so; over 25mm, or out of the hover's reach, it asks you to put the tile back on its home (Enter
when done; logged as a reset).

## Reading results: `scripts/session_report.py`

```
python scripts/session_report.py                                 # the latest session
python scripts/session_report.py sessions/20261024_101500_bench  # a specific one
python scripts/session_report.py --all                           # every bench session side by side, with its setup
```

Shows the setup, success per tile and policy, stalls, clearances, ACT seconds, interventions (rescues
per 10 attempts, clean vs. assisted successes) and saves a contact sheet (one image row per attempt).
For a played hand (`play.py`) it adds the result, the grade, what you were asked and the turn times;
`--all` ends with one line per hand.

## Autonomous play: `scripts/play.py`

The robot plays one closed hand (no calls, no flowers). Your only job is to put each drawn tile on the
drop zone and press Enter. Why it works the way it does, and the targets it's measured against:
`docs/autonomy_plan.md`.

```
python scripts/play.py --layout hand13                 # play a hand
python scripts/play.py --layout hand13 --teach-zone    # find the drop-zone marker again first
python scripts/play.py --layout hand13 --resume        # carry on the latest play session's hand
python scripts/play.py --sim                           # no hardware: a fixed hand that wins in 3 draws
```

Before the first hand:
1. **The hand slots**: teach a 13-spot layout in two rows (13 in one row don't fit the reach) with
   `python scripts/layout.py --name hand13 --teach`, inside the area pickups were checked on.
2. **The test layout** `hand14`: the same 13 slots plus one spot where the drawn tile will go (today S13, WW at
   (0.223, −0.158), at the edge of the reach). Check it with `bench.py --layout hand14 --trials 2`, a tile on
   every spot: pickups and returns at every slot *and* at the drop zone.
3. **The drop zone**: lay ArUco marker 3 (DICT_6X6_250, as in MahjongBot) flat on the mat, uncovered, on
   hand14's extra spot. The first run finds it and saves its position (`data/drop_zone.json`, tied to the
   RealSense calibration). The drawn tile goes on top of it.
4. **The discard pose**: `bench.py`, `teach-discard`, over the container.

Play reaches the drop zone only once a hand is under way, so it won't start a hand (on the arm) until it's
checked: hand14 must still hold all of hand13's slots, have a spot within 25mm of the taught zone, and some
bench session must have picked up and returned a tile there. If not, it prints the commands that fix it
(e.g. `layout.py --name hand14 --remove S13` then `--add S13 X Y` at the zone, then the bench walk) and
stops. `--skip-zone-test` plays anyway, logged. `status.py` shows the same check.

A hand: it reads the 13 tiles (asking about any it isn't sure of), then each turn reads the drawn tile,
checks the table still matches its hand, chooses the discard (lowest shanten, then the most improving
tiles still left, then kinds), drops it in the container, and if it came from the hand moves the drawn
tile into the slot it left, aimed so it lands on the slot's home. Type `end` at the draw prompt when the
wall is empty. At the end you check its hand against the table: Enter if it's right, or the wrong slots
as `H3=5D`.

It asks rather than guess: a name the votes over the scans aren't clearly for, a slot that reads
differently from the hand, a fifth copy of a tile, a pickup that failed 3 times (you move the tile). Each
question is logged, and the hand is graded **clean** (nothing asked), **attended** (questions only),
**rescued** (you touched something) or **silent error** (the end check found a tile it had wrong).
Every step is checkpointed: after Ctrl-C or a stop, `--resume` carries on (it refuses while the arm
holds a tile).

| Option | What it does |
|---|---|
| `--layout NAME` | **required** (except `--sim`): the 13 hand slots |
| `--policy ID` | policy from `data/policies.json` (default: the registry's default) |
| `--teach-zone` | find the drop-zone marker again (after moving it or the camera); it must be uncovered |
| `--resume [FOLDER]` | carry on a play session's hand (default: the latest) |
| `--skip-zone-test` | play even if the bench hasn't picked up and returned a tile on the drop zone (logged) |
| `--sim` | simulated arm, cameras, mat and policy |

## Recording demonstrations: `scripts/record.py`

The arm hovers over a tile, then you take over with the gamepad. `→` starts recording / saves,
`←` discards, `Esc` ends. After a discard it asks *Same tile, same hover again?*: Enter goes
straight back to the same hover (same tile, same jitter offset, no trip to rest), `n` goes to rest and
asks for a tile. Enter means no after "knocked a tile" or "hover was wrong".

A tile still in the jaws when an episode ends (saved, discarded or Esc) is set back down at its home (where
it was first seen this session, as the bench does: aimed with the return model, so tiles don't walk across
the mat episode after episode), then the arm rests — no putting it back by hand. A RealSense scan then logs
where it landed (a sample for `data/return_corrections.jsonl`); past 25mm from home, or out of the hover's
reach, you're asked to put it back. It's a little off from where it was detected, so a discard doesn't offer
the same hover again. Move tiles to new places now and then (more than 40mm, or they're returned to their
old home): v3 needs spots spread across the mat.

```
python scripts/record.py --dataset Yewy/mahjong_mat_pickup_v3
python scripts/record.py --dataset Yewy/mahjong_mat_pickup_v3 --layout eval_v1 --repeats 2
python scripts/record.py --dataset Yewy/mahjong_mat_eval_v1 --slice new --layout eval_new
python scripts/record.py --dataset Yewy/mahjong_mat_pickup_v3_jitter --hover-jitter-mm 6      # training: random hover offsets
python scripts/record.py --dataset Yewy/mahjong_mat_eval_v1 --slice jitter --layout eval_new --hover-offsets-mm 6x,-6y
python scripts/record.py --dataset Yewy/mahjong_mat_dagger_v6 --policy pickup_mat_cl_v6                  # DAgger corrections
```

**DAgger (`--policy ID`)**: the policy does the pickup and you correct it where it goes wrong. After
the hover, `→` starts the episode with the policy driving. Press **A** (the rightmost face button) and
the policy stops and you have the arm, from where the policy last put it, with the gripper
held. Sticks and other buttons do nothing until then, and an A held before `→` doesn't count. The wrist
window shows `POLICY drives` or `YOU`. Finish the pickup, then `→` saves as usual. Every frame is recorded. Training
(`make_policy.py --add`) keeps only your frames, from the takeover on. An episode where you never took
over is saved, but it isn't trained on. Take over as soon as you see the mistake coming (heading low,
about to drag the table or bump a neighbour), not after it. The policy's own hover (offset, clearance,
gripper) replaces `--hover-*`. Each takeover is logged as a rescue, and `session_report.py` lists them
with when you took over. Use a dataset of their own, never an eval slice.

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
| `--hover-offset-mm`, `--clearance-mm` | the pincer tip is commanded this far above the grasp point (default 45), then settled until it measures this far (default 40) |
| `--no-correction` | hover over the raw detection |
| `--no-view` | no wrist camera window |
| `--no-return` | don't set a held tile back down: it's released at rest, as before |
| `--policy ID` | DAgger: this registered policy drives from its own hover until you press A on the gamepad (see above) |
| `--stick-commit-s S` | a gamepad stick moves the arm only once held that long (default 0.1), so flicks don't. The right stick moves height *or* wrist roll: whichever moves first, until it's back at centre |
| `--deadzone F` | stick deadzone, fraction of its travel (default 0.35, as collect_corrections: a thumb resting past XLeRobot's 0.15 pulled the arm 87mm). The wrist window's bottom line shows what the sticks read (red while they move the arm), and a stick off centre when you take over is said in the terminal. Lower it if gentle pushes don't register |
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
`--fix-gripper EPS` is for an episode whose only fault is a gripper command that stopped short of 0 on
the tile: `make_policy.py` builds it with the command carried on to 0 at the rate it was closing
(`data/episode_fixes.json`). The jaws had already stalled on the tile, so the frames still match.

`--against REPO` compares each spot's hover→grasp move with the other recording's at the same place
(within 15 mm), counting only demos that steered. A demo lowered straight down only says the tile was
close enough for the jaws (a third of v2's were). Shifts beyond ~6 mm in one direction at most spots
are worth a `collect_corrections.py` check; a few spots differing is as likely to be how the demos were done.

Options: `--worst N` (rows shown, default 15), `--flag SCORE` (threshold, default 4), `--csv PATH` (save the table),
`--phases`, `--against REPO`, `--drop EPS --reason TEXT`, `--fix-gripper EPS --reason TEXT`.

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
| `--add REPO` | train on another recorded dataset as well (repeatable), e.g. DAgger corrections: only their frames from the takeover on |
| `--trim-narrowing` | also cut the slow gripper narrowing at the start of each episode |
| `--augment` | colour jitter while training (brightness, contrast, saturation, hue, sharpness; no shifting or rotating the image, the jaws stay put) |
| `--replace-mat FRACTION` | swap the black table top for another surface (grid mat, plain colour, wood, cloth, paper) in that fraction of episodes, e.g. `0.7`; `--dry-run` saves a preview sheet next to the training output to check before training |
| `--steps N` | training steps (default ~131 passes over the data) |
| `--private` | upload as private (default is public under `Yewy/`) |
| `--skip-build`, `--skip-upload`, `--skip-train` | resume a run partway through |

**What the names mean.** There are three kinds of repo under `Yewy/`, and only the first is yours to name:

| Kind | Made by | Name |
|---|---|---|
| recorded dataset (raw episodes) | `record.py --dataset` | whatever you choose, e.g. `mahjong_mat_pickup_jitter_v3`, `mahjong_mat_dagger_v3c` |
| training dataset (trimmed, 224px) | `make_policy.py` | `<--source>_<version>_224`, e.g. `mahjong_mat_pickup_jitter_v3_v4_224` |
| policy | `make_policy.py` | `<--source>_<version>_policy`, registered as `pickup_mat_cl_<version>` |

So in `jitter_v3_v4`, `jitter_v3` is the recording and `v4` the policy version. `--add` datasets don't
show in the name, only in the policy's `notes` in `data/policies.json` (v4 = `jitter_v3` + `dagger_v3c`).
Record DAgger corrections to a new dataset named after the policy that drove, e.g.
`mahjong_mat_dagger_v4` for `--policy pickup_mat_cl_v4`, never to a `_224` or `_policy` repo.

## Scoring offline: `scripts/offline_eval.py`

Scores a policy on recorded episodes it never trained on (GPU, minutes, no arm). Good for screening;
the bench decides what ships.

```
python scripts/offline_eval.py --policy pickup_mat_cl                                  # scores per slice
python scripts/offline_eval.py --policy pickup_mat_cl --against pickup_mat_cl_v1       # + pass/fail gates vs a reference
python scripts/offline_eval.py --policy pickup_mat_cl --by spot                        # or --by lighting / surface (default: --by slice)
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

`collect_corrections.py --device PATH` picks the gamepad (default: the first one found). A `wrist` window
shows the wrist camera while you line up, and the frame at → is saved as `sessions/<time>_corrections/aligned/<n>_<tile>.jpg`;
`--no-view` runs without the window (and without the wrist camera plugged in).

The window's bottom line is what the gamepad reads: each stick as it reaches the arm, with its raw position
in brackets, red while it's moving the arm and green while it isn't. If a stick rests off centre when you
take over, the terminal says so: let go of it. `--deadzone` (default 0.35 of the stick's travel; XLeRobot's
own is 0.15) is wider so a thumb resting on a stick doesn't drive the arm; push a bit further for the same
slow speed.

Each tile's teleop is traced too (`handoff/<n>_<tile>.jsonl`: command, measured joints and gamepad input
every step), and a few lines after → / ← say what the arm did once you had it. To look back at a session:

```
python scripts/handoff_report.py                 # the latest session with traces: a table per tile + a plot each
python scripts/handoff_report.py sessions/20261029_101500_corrections --no-plot
```

| Column | Means |
|---|---|
| sag at handoff / at end | measured − commanded tool point (x reach, y robot's left, z up), mm. At the end, it's error the correction records (it stores the commanded point) |
| drift before input | how far the arm moved under an unchanged command before you touched anything: sinking servos |
| cmd idle | how far the *command* moved in steps with nothing pressed: should be 0 (else a teleop bug) |
| IK deg | how far the hover is from XLeRobot's own IK solution: the first stick push eases onto it |

In the plots (`handoff/<n>_<tile>.png`) a red (measured) line leaving a flat blue (commanded) one is the
servos; blue moving outside the grey (input) stretches is the teleop.

Which sources are live is set in `data/hover_model.json`. Don't change the live hover for v2-style
policies (they replay a memorised move from it; see `CLAUDE.md`).

## Return correction (setting tiles down where they were)

The jaws pull a tile only partway to their centre, so where a returned tile lands depends on how
off-centre that grasp was. Every bench return logs where it was set down and where the RealSense saw
it land (`data/return_corrections.jsonl`); the bench fits from these and aims each return so the tile
lands at its home. The fit is used only if it predicts landings better than "the tile keeps its offset
in the jaws", by leave-one-position-out error. Like the hover correction, it's tied to the RealSense
calibration.

```
python -m mahjong.return_correction                 # the fit: landing error per position, before and after
python scripts/return_samples.py                    # import returns from every bench session (bench.py adds its own)
python scripts/return_samples.py sessions/20261028_090511_bench   # ...or just these
```

`session_report.py` prints how far a session's returns landed from home, and how the open jaws left the tile:
how far the measured tip rose of the 30mm commanded after the release, how far it moved sideways doing so,
and how high it was when the swing to rest began (a return goes higher over the tile first if it's under
60mm). A return lets the tile go just above where the jaws closed on it at the pickup (1mm, measured), so it
lands as it was held instead of dropping; the report's "return release" line says how far above that each
one actually let go.

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
python scripts/layout.py --name eval_v1                           # live view: n/p spot, a all, g ghost, r reach, c capture, q quit
python scripts/layout.py --name eval_v1 --add T1 --like S2 --yaw 15   # a tilted copy of a spot
python scripts/layout.py --name eval_v1 --add N7 0.24 0.17        # a spot at x y (metres, arm base frame)
python scripts/layout.py --name eval_v1 --remove N7
```

**Is it in reach?** `r` (live view and `--teach`) tints the mat where the arm reaches a tile comfortably
(green), only at the edge of its reach (amber), or not at all (untinted), and prints the spots outside the
green. Amber means the hover gets there but at grasp height the wrist is within 8mm of the gamepad's reach
or past it: on hand14 that's 1D and WW, the spots pickups and corrections struggled with. Keep new spots
in the green. It goes through the hover correction as the bench does, and takes a few seconds the first time.

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
arrangements of their own. Export them with `mine_frames.py --exam`, drag the whole export folder
(with its `data.yaml`) into a **separate** Roboflow project, label them, and download that as a YOLO
export with no augmentation and no resize (e.g. to `data/detector/exam_labelled/`).
Score the live detector first: that's the baseline.

```
python scripts/detector_eval.py --exam-dir data/detector/exam_labelled                # the live detector
python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --by lighting     # or --by session
python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --ignore flowers   # tiles that matter for play
python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --weights outputs/detector/v2/weights/best.pt --against assets/tile_detector.pt
```

It reports **found** (labelled tiles it put a box on), **named right** (of those found), **right
overall** (found and named right: the tile accuracy), missed tiles, extra boxes on nothing, and its
commonest mix-ups (e.g. `4D->5D x6`; `--mixups N` prints N, default 8). `--ignore flowers` (or names, e.g. `flowers,EW`) leaves those
tiles out: boxes on them count for nothing, while another tile read as one is still a mix-up. A label
drawn twice on one tile is merged and listed, so you can delete the copy in Roboflow. **Wrong names**
says how many found tiles it named wrong and the highest wrong-name rate the exam can't rule out (95%):
0 of 99 still allows 3%, so showing under 0.3% (what autonomous play needs) takes ~1000 exam tiles.
Scores are per frame and go in `data/detector_eval/` (`_without_<names>` when some are left out). Frames in the
folder that aren't from an exam session are left out.

Baseline (2026-10-27, live detector, exam v4: 15 frames, dim + lamp + ceiling lights): without
flowers 102 tiles, 85.3% found, 80.5% named right, **68.6% right overall**, no extra boxes. Mostly
neighbouring numbers within a suit (6/7C, 2/3C, 5/8C, 3D/5D, 2/3/4B).

**3. Mining and retraining.** `mine_frames.py` picks the frames worth labelling: a tile whose name
changed across a scan's 5 frames (flicker), a tile seen in some frames and not others (dropout), a
tile below 0.7 confidence (low), or a scan you called a detection glitch at the bench. `--all` adds a
plain frame from every other scan (up to `--per-session`, default 15), since the detector has never
seen this scene at all. Exam and simulated sessions are never exported for training.

```
python scripts/mine_frames.py --list                    # what it would take
python scripts/mine_frames.py --all --since 20261027    # export to data/detector/mined_<time>/
python scripts/mine_frames.py --all 20261027_213821_capture   # one session, by folder or name
```

An export is laid out as Roboflow's own YOLO export: `data.yaml` (the label map Roboflow asks for) and
`train/images/` + `train/labels/` (`test/` for the exam), the labels being the detector's own boxes
as a start (one box per tile: where the detector read a tile as two names, only the surer one). **Drag
the whole export folder into Roboflow's upload, not just the images**, for the training project (the
July project, "Mahjong tiles on table"). Why each frame was picked is in `<export>_manifest.csv` beside
the folder. Correct the boxes, make a new dataset version (no augmentation: the training adds its
own), and download it as a YOLO export.

Then re-split it by scene. Roboflow splits frame by frame, and the July frames were each shot at 4
exposures, so the same scene sat in train and valid (134 of v4's 182 valid frames): the valid score
then says nothing and `best.pt` is picked on it. `split_by_scene.py` pools the export's splits, groups
frames by their labels (same names in the same places), and deals whole scenes to valid (`--valid`,
default 20%, for the July frames and the session frames apart). It also keeps only the 38 tile names
(`aruco_marker` would change the model's head), merges a label drawn twice, and leaves out (and lists)
any frame with two names on one tile, for you to fix in Roboflow.

```
python scripts/split_by_scene.py --data "data/detector/<download>/data.yaml" --out data/detector/train_v5 --list
python scripts/split_by_scene.py --data "data/detector/<download>/data.yaml" --out data/detector/train_v5
python scripts/train_detector.py --data data/detector/train_v5/data.yaml --name v5 --dry-run   # counts, exam check
python scripts/train_detector.py --data data/detector/train_v5/data.yaml --name v5             # GPU
# --start WEIGHTS (default: the live ones), --epochs N (default 40), --batch N (default 6, GPU memory at imgsz 1280)
python scripts/detector_eval.py --exam-dir data/detector/exam_labelled --weights outputs/detector/v5/weights/best.pt --against assets/tile_detector.pt
```

v4 (2026-10-27: the July frames + 21 of the current scene, split by Roboflow) lost to the live detector
on the exam, 52.4% vs 66.7% right overall: it named 93% of what it found right but found only 56%,
less sure of the current scene the longer it trained (95% of its frames were July ones).

v6x3 (2026-10-28, live since: `tile_detector@0c4b15eb`): v6 = the July frames + 190 session frames
(capture and bench scans), split by scene, with the session frames repeated 3× in train. Without
flowers 97.1% found, 100% named right, **97.1% right overall** (was 68.6%); with flowers 94.4% (was
66.7%). v6 without the repeats: 95.1% / 90.5% (its misses were 1F/3F). No training frame shares more
than 17% of an exam frame's tiles, so this isn't a leak. Box centres moved < 0.1 px against the old
weights (the hover correction holds).

The exam scores Roboflow's re-encoded copies of the frames: on the session's own frames (what the
robot sees) the old weights got 81% right, not 69%, so absolute numbers undersell every detector; the
comparisons hold, since both score on the same copies.

`train_detector.py` starts from the live weights with their July recipe (1280 px, no mirror flips: a
mirrored tile face isn't a real tile) and refuses a dataset with any exam frame in it. Ship the new
weights only if they beat the old ones on the exam: copy `best.pt` to `assets/tile_detector.pt` and
commit it with the scores. Exports and downloads under `data/detector/` aren't in git; Roboflow keeps
the labels.

**4. Wrist-camera photos (for a name check at the hover).** The wrist camera sees a tile from ~4 cm at
the hover, far sharper than the RealSense. `wrist_capture.py` collects photos to train a check there:
it scans the mat, lists the tiles left to right with the detector's names (numbered on the RealSense
view) for you to confirm, then hovers over each one `--views` times (default 8): the first centred at
40 mm, the rest moved by up to `--jitter-mm` (8), at heights cycled from `--clearances-mm` (40,55,70)
and the wrist rolled by up to `--roll-deg` (15). It never touches a tile. When the set is done the arm
rests, the terminal rings (plus a desktop notification), and it prints every name's wrist views at the
hover from all sources (its own sessions, bench attempts' first frames, recorded episodes) and which
are still below `--target` (10); put down a new set and press Enter. A wrist camera that stops giving
frames is reopened and the view taken again (as the bench and record.py do).

```
python scripts/wrist_capture.py                      # until you press q at a set change
python scripts/wrist_capture.py --views 12 --sets 3
python scripts/wrist_capture.py --sim --sets 1       # no hardware
```

Also `--frames N` (wrist frames saved per view, default 2), `--seed N` (the jitter and roll), `--no-view`
(no RealSense / wrist windows).

At the names prompt: Enter starts, `3=2C` renames tile 3, `-3` leaves it out, `r` scans again, `q`
quits. Photos land in `sessions/<time>_wrist/wrist/<NAME>/` (one folder per name), with a `wrist_view`
event per view. A view takes ~8 s, so 8 views of 13 tiles is ~15 minutes a set. As of 2026-10-28 the
recordings and bench runs had no wrist views of 1-5C, 7-9C, the flowers, 1B or 5B: put those down first.

**Checking the wrist photos.** A photo's name came from the detector (you confirm it in wrist_capture,
not at the bench), so a misdetected tile's photos can slip in under the wrong name. `wrist_review.py`
shows each name's photos, one row per physical tile (all its views), with the bench attempts' hover
frames packed by session. Click a photo to mark it wrong, click a row's label for the whole tile,
right-click to give it its right name (asked in the terminal: a 2C filed as 3C becomes a 2C photo),
Shift+click to see it at full size (several tiles are in view; the one between the jaws is the one).
Space marks the page's unmarked photos ok and moves on; `p` back, `]` `[` change name, `q` quits.

```
python scripts/wrist_review.py                  # every name, what isn't reviewed yet
python scripts/wrist_review.py --name 3C --only
python scripts/wrist_review.py --all            # reviewed photos too
```

Nothing is moved or deleted: the photos stay in their sessions (the record), and each verdict is a
line in `data/wrist_review.jsonl` (commit it; it merges by union). The counts in wrist_capture.py and
any training set read the photos through it (`mahjong/wrist_photos.py`): a renamed photo counts under
its right name, one marked wrong without a name doesn't count.

## Where things stand: `scripts/status.py`

Reads the state from the files that decide it, so it can't go stale the way a doc does:

```
python scripts/status.py
python scripts/status.py --set-default pickup_mat_cl_v5 --evidence "sessions/20261101_100000_bench: hand14 paired with v4, 40/40 vs 38/40" --why "fewer S12 misses"
```

It prints this checkout against origin/main (and warns if it's behind or has uncommitted code), the default
policy with the evidence it was chosen on, every registered policy's bench total over all sessions, the eval
sets and detector exam, the layouts and their notes, the drop zone and whether the bench has checked a pickup
there, and how many merged branches are waiting to be closed.

**Changing the default policy**: always with `--set-default ID --evidence ...` (and `--why`). It writes a
`default_note` next to the default in `data/policies.json`; a test fails if the default changes without one.
Then commit `data/policies.json` and update ROADMAP.md's status table.

Every hardware session also warns at its start if the checkout is behind origin/main (as last fetched), has
uncommitted code, or is a worktree, and logs that in `session_start`.

## Approving merges: `scripts/merge_pr.py`

Branches (usually Claude's, each in a worktree) come back as draft PRs. You approve each one from the main
checkout:

```
python scripts/merge_pr.py              # open PRs, and merged branches/worktrees not cleaned up yet
python scripts/merge_pr.py 34           # check PR 34, ask, merge, pull main, test, clean up its branch
python scripts/merge_pr.py 34 --slow    # also the GPU tests on the merged tree (automatic if policy.py/dataset.py changed)
python scripts/merge_pr.py --close      # clean up every merged branch (asks first)
```

What `merge_pr.py 34` shows you before it asks:
1. The PR's description, and which of its sections are missing: **What changed**, **Files**, **Overlaps**,
   **Unverified on the arm**. The last one is what to try on hardware after merging.
2. Files both `main` and the branch changed since the branch started, and any function, constant or option
   added on **both** sides. Git merges those without a conflict, and that's how duplicate options and
   helpers broke `record.py` on 2026-10-23. If it lists one, ask Claude to reconcile before you say yes.
3. The tests on `main` + the branch merged together, in a scratch folder. If they fail, or the merge
   conflicts, it stops: ask Claude to merge `main` into the branch and fix it.

On `y` it merges on GitHub, pulls `main` into this checkout (hardware runs here), reruns the fast tests on
it, and cleans up: the branch's worktree's `sessions/` are moved into this checkout's, then the worktree,
the local branch and the GitHub branch are deleted. A worktree with uncommitted changes is left alone and
said so. `--close` does the same for anything merged earlier, and asks separately about branches whose PR
was closed without merging (printing the commit they can be recovered from).

Don't merge on GitHub's page: it runs no tests, and this checkout stays behind.

## Tests

```
python -m pytest -m "not slow" -n 8   # fast: motion, kinematics, correction, dataset trims, bench, docs (~11 s)
python -m pytest -m "not slow"        # the same without pytest-xdist (~45 s)
python -m pytest                      # everything; slow tests load policies and need the GPU
```

Git hooks run the fast tests on every commit and every merge (in parallel when pytest-xdist is installed:
`pip install pytest-xdist` in the lerobot env), so a broken invariant, or two branches that break each
other, can't be committed. Turn them on once per clone: `git config core.hooksPath scripts/hooks`.
`tests/test_docs.py` checks this README and CLAUDE.md against `data/` and the scripts: every option they
mention exists, every script has a section, the default policy and every registered one are named right.
Merging worktrees: `merge_pr.py` above, and "Worktrees and merging" in `CLAUDE.md`.

Run the fast tests before calling a change hardware-ready (and the slow ones too if `policy.py` or
`dataset.py` changed). See `CLAUDE.md` for the rules the code follows and why.
