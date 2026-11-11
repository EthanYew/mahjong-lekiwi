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
| See which tiles get misread, and how much it matters | `python scripts/name_report.py` |
| Approve and merge a branch's PR, then clean up after it | `python scripts/merge_pr.py` (lists them), then `python scripts/merge_pr.py <number>` |
| Make another policy the default | `python scripts/status.py --set-default <id> --evidence "sessions/<bench session>: <scores>" --why "..."` |
| Check the workspace is set up as before (after packing up / moving things) | `python scripts/setup_check.py` (once, on a good setup: `--save-reference`) |
| Run the bench without the robot | `python scripts/bench.py --sim --trials 3 --tile 8D` |
| Test a policy on the arm, interactively | `python scripts/bench.py` |
| Run N pickup trials on one tile | `python scripts/bench.py --trials 5 --tile 8D` |
| Run the bench with no one watching, then check what the scans said moved | `python scripts/bench.py --unattended --layout even-hand14 --trials 2`, then `python scripts/bump_report.py` |
| Test a specific policy | `python scripts/bench.py --policy pickup_mat_cl_v1 --trials 5 --tile 8D` |
| Check a policy steers to the tile (vs. a memorised move) | `python scripts/bench.py --trials 15 --tile 4D --hover-perturb-mm 0,6x,-6x,6y,-6y` |
| Try a narrower (or wider) gripper opening at the hover | `python scripts/bench.py --layout hand14 --trials 3 --hover-gripper 25` |
| Try a lower hover where the policy closes high (S8) | `python scripts/bench.py --trials 3 --tile 1B --hover-lower-mm 6` (lay that tile on S8) |
| See what happened in the last session | `python scripts/session_report.py` |
| Find which part makes the wrist camera drop, and how drops ended | `python scripts/wrist_usb_watch.py --label still`, then `--report` |
| See how the arm sagged / drifted while lining up corrections | `python scripts/handoff_report.py` |
| See how well returns land tiles back where they were | `python -m mahjong.return_correction` |
| Compare all bench sessions | `python scripts/session_report.py --all` |
| Check the hand's slots and the drop zone on the bench before playing | `python scripts/bench.py --layout even-hand14 --trials 2` (a tile on every spot), then `--carry S13,S1` added (a tile on S13, S1 empty) |
| Check a new policy iteration against the default (`docs/scorecard.md`) | `python scripts/bench.py --policy BASELINE,CANDIDATE --layout even-hand14 --trials 1`, then `session_report.py` |
| Check a policy is ready for scored hands | `python scripts/bench.py --policy ID --layout even-hand14 --trials 2`, then `session_report.py` |
| Play a hand on its own (you only put drawn tiles on the drop zone) | `python scripts/play.py --layout even-hand13` |
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
| Correct the default policy where it goes wrong (DAgger) | `python scripts/record.py --dataset Yewy/mahjong_mat_dagger_v8 --policy pickup_mat_cl_v8` (press A on the gamepad to take over) |
| Train on the corrections too | `python scripts/make_policy.py --source Yewy/<name> --add Yewy/mahjong_mat_dagger_v3c --add Yewy/<new dagger set> --version <next> --dry-run` |
| Compare two policies across the mat on the arm | `python scripts/bench.py --policy pickup_mat_cl,pickup_mat_cl_v2n --layout <layout> --trials 2` (2 rounds at every spot, both policies each round) |
| Take photos of tiles for the tile detector (no arm) | `python scripts/capture_frames.py` (`--exam` for exam frames) |
| Export the frames the detector struggled with, for labelling | `python scripts/mine_frames.py --list` (then drop `--list`) |
| Score the tile detector on its exam | `python scripts/detector_eval.py --exam-dir data/detector/<exam export>` |
| Photograph every tile on the mat with the wrist camera from jittered hovers | `python scripts/wrist_capture.py` (rings when a set is done) |
| Check the wrist photos of each tile name (mark wrong ones, rename misfiled ones) | `python scripts/wrist_review.py` |
| Check labelled detector frames by eye: one picture per tile name | `python scripts/label_sheets.py data/detector/<folder>` |
| Train or score the wrist-camera name check | `python scripts/train_name_check.py`, `python scripts/name_check_eval.py` |
| Hold tiles up to the base camera and see if anything reads them (face-down plan, test 1; no arm) | `python scripts/holdup_capture.py`, then `python scripts/holdup_score.py --installed`, then `python scripts/holdup_check.py` |
| Drive the arm by gamepad watching the wrist and base cameras, and save poses that hold a tile up to the base camera | `python scripts/teleop_view.py` |
| Have the arm collect base-camera photos of face-down tiles (check the saved poses with nothing in the jaws first) | `python scripts/holdup_arm.py --empty`, then `python scripts/holdup_arm.py` |
| Try a whole face-down turn: draw, show, read, you check, discard (crude, to find where it breaks) | `python scripts/crude_run.py --hand <13 names> --turns 6` |
| Test drawing face-down tiles from a wall, at one station you feed by hand | `python scripts/wall_test.py --stations F --cases flat --trials 10`; `--report` scores every session so far |
| Re-split a Roboflow download so no scene is in train and valid | `python scripts/split_by_scene.py --data <download>/data.yaml --out data/detector/train_<v> --list` (then drop `--list`) |
| Score a played hand frame by frame against a person's names, or build training frames from it | `python scripts/hand_frames.py <session>` (`--out`, `--onto`) |
| Retrain the tile detector | `python scripts/train_detector.py --data <export>/data.yaml --name <name> --dry-run` (then drop `--dry-run`) |

Registered policies live in `data/policies.json` (default: `pickup_mat_cl_v8`, 37 standard-mat demos + other surfaces + black-top
jittered demos, no DAgger, `--trim-narrowing`, hover 20.6%; the owner's pick on 2026-11-05 as the best trained, `status.py` has the evidence;
also `pickup_mat_cl_v9` (v8's data + 80 DAgger corrections of v8 on wall picks at station F, 16 of them with the green hand in view; trained 2026-11-10, private on the Hub, not benched yet),
`pickup_mat_cl_v7` (`pickup_mat_cl_v6`'s data + 52 brown/white paper demos, 30 of them closing short as recorded, the default on 2026-11-04),
`pickup_mat_cl_v6` (v3c's hover-jittered demos, DAgger corrections of v3c and of v4 at 4D/7D, and 33 green-mat demos,
trained with `--augment --replace-mat 0.7`),
`pickup_mat_cl_v5` (without the green demos or augmentation; fails on the green mat), `pickup_mat_cl_v4`
(without v4's corrections),
`pickup_mat_cl_v3c`, `pickup_mat_cl_v3b`, `pickup_mat_cl_v3` (demos from jittered
hovers, so they'd steer to the tile; the perturbation test says only partly, ROADMAP 1), `pickup_mat_cl` (v2),
`pickup_mat_cl_v1`, and `pickup_mat_cl_v2n`: v2's episodes with
the gripper narrowing trimmed and the eval episodes left out). Why the default is the default, and how each
did on the bench: `python scripts/status.py` (tests/test_docs.py keeps this paragraph in step). How the versions evolved, what each one's data captured and how each was trained: `docs/policy_history.md`.

---

## When the wrist camera drops out: `scripts/wrist_usb_watch.py`

A lost wrist camera now tries to come back with no one touching it. Every script waits up to 10s, cuts the power of the
camera's USB port once, and only then shows "Fix that, then Enter". **Don't replug during the wait**: a replug there would
be counted as the camera coming back by itself. `session_report.py` prints how each loss ended, and
`wrist_usb_watch.py --report` totals them.

The power cycle and turning autosuspend off need root, so run this once per machine (`--remove` takes it out again):

```
sudo scripts/wrist_usb/setup.sh
```

It installs a udev rule, a helper in `/usr/local/sbin`, and a sudoers entry that lets your user run that helper without a
password. The helper takes only a port name and refuses any port with something on it other than the wrist camera.
Without the setup, the wait and the prompt still work. Many ports can't really cut power, so the cycle may do nothing:
the logged outcomes will say.

To find which part fails, watch the link while you try one thing at a time. The watcher moves nothing and never recovers
the camera, so it shows what the link does by itself:

```
python scripts/wrist_usb_watch.py --label still --seconds 600     # arm left alone: the control
python scripts/wrist_usb_watch.py --label by-hand --seconds 300   # motors off, move the arm by hand, right side especially
python scripts/wrist_usb_watch.py --kernel-only --label bench     # 2nd terminal during a bench or teleop run: motors on
python scripts/wrist_usb_watch.py --label new-cable --seconds 300 # then swap one part at a time
python scripts/wrist_usb_watch.py --report                        # every watch, and how real losses ended
```

Drops by hand point to the cable or connector; drops only with the motors on point to electrical noise; drops that follow
the camera onto a new cable point to the camera board. Run it from the main checkout, like other hardware scripts.
Options: `--label`, `--seconds`, `--kernel-only` (the camera stays with another script), `--port` (default: where the
camera is now), `--report`, `--sim`.

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

Saving a reference shows each wrist frame with a cross on the tile: click the middle of the tile in the window if
the cross is off it, then Enter (`r` to reposition and look again). A check finds the tile by the reference's picture of
it, so each spot needs the same tile every time: saving asks which you used, and a check asks for that one. A reference
saved before this asks to be saved again.

On a pale or patterned mat the RealSense can't read a tile's turn, so green means position only (in `layout.py` and
`record.py` too): line the outline up by eye. Placing finds tiles with the tile detector, whatever the mat.

When the detector can't name a tile but still boxes it (down to 0.1 confidence), the tile counts as there and no one
looks: `tile_faint` events. A tile a spot's scan can't name is picked up from that box, logged as `unnamed` with `name_source` saying how it was found (never under a spot id or layout label).

Otherwise, on a pale mat the bench's wrist look can't tell a tile from the mat. So when the scans lose a tile it asks you
the first time at that place, then compares the view with what it saw there.

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
python scripts/bench.py --unattended --layout even-hand14 --trials 2     # no one watching: nothing is asked once the trials start
python scripts/bench.py --layout even-hand14 --trials 2 --carry S13,S1   # carry a tile from the drop zone to a slot and back, as a hand does
```

| Option | What it does |
|---|---|
| `--policy ID` | policy from `data/policies.json` (default: the registry's default). `a,b` compares two, paired: rounds alternate a b / b a from the same hover |
| `--unattended` | with `--trials` and `--tile` or `--layout`: run with no one watching (see below) |
| `--layout NAME` | trials walk this layout's spots (`scripts/layout.py`); the report shows success by spot and policy |
| `--trials N` | run N rounds of pickup/return trials, then quit. A round is one attempt per policy, at every spot with `--layout`: pickups = N × policies × spots. Without `--tile` or `--layout`, it asks which tile. Every return aims the tile at its home (see below), so it doesn't walk off |
| `--tile NAME` | tile for `--trials`, e.g. `8D` |
| `--sim` | simulated arm, cameras, detection and policy |
| `--hover-perturb-mm LIST` | shift the hover on purpose, cycling through e.g. `0,6x,-6x,6y,-6y` (mm). A policy that steers follows the tile; one that replays a memorised move follows the offset. `session_report.py` prints which. |
| `--hover-gripper PCT` | gripper opening at the hover for every policy, instead of the one its episodes started at (`data/policies.json`; v6 26.7). The policy moves the gripper as changes from there, so its narrowing on the way down shifts by the same amount. Logged per attempt; `session_report.py` flags the session. Far outside the training opening is untested territory for the policy |
| `--hover-lower-mm MM` | hover this many mm lower than the policy's own, up to 10 (negative = higher, to -15): the hover's offset and the clearance the settle aims for move together. For a spot where the policy closes high. On 2026-11-05 v8 at S8 closed 24.5mm up, and with 4 lower 21-23mm: the close moves about 0.6mm per mm of hover, so it cannot reach the 10mm that picks (127 of 130) inside the arm's 10mm limit. Read `grasp_clearance_m` in `session_report.py`. An experiment: the attempts save no hover samples, the session is flagged in `session_report.py` and is no counted result. The policy was trained from its own hover, so a lower start is a different image |
| `--carry FROM,TO` | with `--layout`: trials carry one tile between these two spots, across the mat and back (a tile on FROM, TO empty) instead of returning it where it came from. A round is two carries. Every carry's landing is logged; from the drop zone's spot, it's what `play.py` checks before a hand. Needs one policy and someone watching. For a first run, consider leaving TO's neighbours off |
| `--device PATH` | gamepad device path (only for teach-discard) |

**Menu commands** (type the number or name): `1 hover`, `2 pickup`, `3 return`, `4 discard`,
`5 teach-discard` (drive the arm over the discard bin with the gamepad and save it), `6 trials`,
`7 rest`, `8 note` (log something you did by hand), `9 setup` (lighting/battery/surface changed mid-session),
`10 quit`.

**A tile the trials can't find** (not detected by name, or nothing near a `--layout` spot) is first looked
at with the wrist camera: the arm hovers where it was first seen and checks for a tile face between the jaw
tips. If it's there (the detector flickered) the pickup goes ahead from there. If not, it stops for you:
put it back where it belongs, Enter rescans (logged as a reset), `s` skips the spot (`--tile`: ends the
trials). `--unattended` skips it without asking.

**`--unattended`** asks nothing once the trials start (the setup questions come first, so stay for those). Ctrl-C stops
it. Where a run with you there would ask, it does this instead, and `bump_report.py` shows the result:

| With you there | Unattended |
|---|---|
| A neighbour moved or turned: asked what moved it | a self-reported bump, unless a later scan shows it undone (a glitch). It counts as a knock until you say otherwise |
| A change with the arm at rest: taken as you moving tiles | watched the same way, as a tile moved away from the arm, not a reset |
| A tile not found: asked to put it back | spot skipped (`--tile`: the trials end) |
| A return landing over 25mm off its home, or out of reach: asked to put it back | left there (`reset_skipped` event). The next pickup starts from there if the hover finds it, else the spot is skipped |
| A pale surface the wrist look can't judge: asked if a tile's there | left unknown, and the scans settle it |
| Wrist camera won't reopen: retry prompt | the run ends |
| A marked or stopped attempt: asked why | not asked; listed as unexplained |

It doesn't stop by itself after repeated misses or knocks: it runs its whole plan, so the mat can drift out of shape.
One closing scan at the end gives a bump in the last return a later scan to settle against.

**While the arm picks up or returns a tile, say what you see with a key** (no Enter; the terminal prints
`[b] bumped another tile (noted)`; the keys are printed as each pickup and return starts). Keys typed at a
prompt or the menu don't count.

| Key | Means | Counts as |
|---|---|---|
| `b` | the arm bumped another tile | a collision (failure; the success stays clean), and a neighbour that moved is a knock without asking |
| `p` | the arm pushed or turned the tile it's picking up | a failure |
| `d` | gripped low on the tile (near its bottom) | precarious, it may slip: counted ("gripped low") but the attempt stands; works during the return too, while you watch it carried |
| `u` | gripped high on the tile (near its top) | the same, "gripped high" |
| `r` | I moved a tile by hand (e.g. put a knocked one back for the next run) | what that pickup's or return's scan sees change was you: straightening a tile a knock left (a rescue), else putting tiles back (a reset); never a knock or a glitch |
| `l` | look at this (nothing touched) | an observation |

Keys (these letters, `→`, `←`, `Esc`) are read from the bench's own terminal tab while the arm works,
so type them there; what you type in another tab or window isn't heard (it used to be: 2026-10-30).
`Esc` stops the policy. `→` marks "something happened here" without stopping it (a hand on the tile, a
caught drop); a `→` with no key is asked about right after that attempt (not at the end of a batch). A pickup stops itself once the tile
has been held and lifted for 1.5s (max 16s). Every attempt saves the full wrist video (`wrist.mp4`) in
its folder, so a new policy can be replayed on it. The `hover` command saves a wrist photo in a folder of its
own (`wrist_0000.jpg`, named like a pickup's first frame), and `wrist_review.py` and `wrist_capture.py` count it with them.

After each pickup and return the RealSense rescans the mat. A tile the scan says went is looked at with the
wrist camera first (once the arm has put down what it holds); one still there is logged as a detection
glitch, with the wrist frame in the session's `wrist_checks/`. What else changed is settled like this:

- **you pressed `r`** during that pickup or return: the changes were yours (with `b` as well, the
  neighbours are asked about, since either could have moved them);
- **a neighbour** (within 85mm of the tile the arm handled) **moved or turned**: if you pressed `b` during
  that pickup or return, it's a knock; if not, you're asked right then what moved it: the arm (Enter), you
  by hand, or nothing (a glitch);
- **anything else** (tiles further away, tiles gone or appearing, changes between attempts): no question.
  The next scans decide: undone by a later scan, it was the detector (a glitch), unless you said you moved
  tiles (`r`, or a `note`): then it was a knocked tile you put back; still so 3 scans later (about one
  attempt), it happened: a knock near the arm, "moved by something else" further away (you straightening
  it, if it started where a knock left a tile), a reset between attempts. The end of the trials lists the
  tiles whose detections flickered (`3B x2`).

Moved tiles between commands? `note` → straightened / put back: the next scan's changes are yours, not
counted again. The same goes for a discard you said missed, and in `play.py` for a tile you moved after the
robot gave up or put on an empty slot.

Every scan also saves the frames the detector looked at in the session's `scans/` folder, for improving
the detector (see below).

**Returns go home.** A tile's home is where it was first seen this session (per layout spot, or per
tile name; a tile found more than 40mm from its home was moved on purpose and gets a new one; moved
on purpose by less, the next return puts it back, so restart the bench to give it a new home there). A return
aims the tile at its home, not where the jaws closed, so errors don't add up round after round, using
the return correction (below). The scan after a return checks where it landed: over 12mm from home it
says so; over 25mm, or out of the hover's reach, it asks you to put the tile back on its home (Enter
when done; logged as a reset).

### Saying what's on the mat

Trials on a `--layout` start with one scan and one question: each spot with what the scan reads there. Enter if they're
right, `S7=2B` to correct one, `-S7` for no tile there, `r` to scan again, `q` to leave them unchecked (those placements then
go unscored in `name_report.py`). `--unattended` asks this too, before the trials start. If a tile has to be put back
mid-run, type its name at that prompt if it's a different tile.

## Checking what the scans said moved: `scripts/bump_report.py`

Goes through a bench session's pickups and returns, for you to say which tiles really moved. `bench.py --unattended`
writes the page as it ends; any bench session with scans works.

```
python scripts/bump_report.py                       # the latest bench session, opens in your browser
python scripts/bump_report.py sessions/<folder>
python scripts/bump_report.py --no-serve            # only write <session>/bump_report.html (buttons won't save)
```

| Option | What it does |
|---|---|
| `--no-serve` | write the page and stop. Opened as a file it's readable, but its buttons can't save |
| `--port N` | port for the page, on this machine only (default: any free one) |

Each pickup and return is a card: the mat before and after, cropped around the tile the arm handled, and the same two
frames flipping back and forth with nothing drawn on them, which is the easiest way to see a small shift. Below it, a row
per neighbouring tile with what the scans said and **Moved** / **Didn't move**. Cards where the scans saw a change are
open; quiet ones are closed, so open one if you think the scans missed something (under 5mm doesn't count as a change).

Clicks save straight away, and the latest click on a tile stands. A verdict is a `scan_verdict` line in the session's
`events.jsonl`, and `session_report.py`, the scorecard and `status.py` follow it: **Didn't move** turns a bump into a
glitch, and **Moved** on a neighbour the scans said nothing about makes it a knock. Until you check them, self-reported
bumps count as knocks.

## Which tiles get misread: `scripts/name_report.py`

```
python scripts/name_report.py                    # every tile, per detector and surface, worst first
python scripts/name_report.py --by lighting      # also split by lighting (or spot, or session; repeatable)
python scripts/name_report.py --tile 8C --list   # one tile, and each placement behind it
python scripts/name_report.py --since 20261103   # sessions from that date (name prefix) on
python scripts/name_report.py --json out.json    # the same rows and placements, as a file
python scripts/name_report.py --sim              # include simulated sessions (left out by default)
python scripts/name_report.py --sessions-dir ../other/sessions   # another checkout's sessions
```

A *placement* is one physical tile at one place, until a person or a knock changes that. The report scores every read of
it against what a person said was there: bench attempts and hovers (at a spot or not) and `wrist_capture.py` tiles, the
last as "read once" since one scan can't vote. Exam sessions are never read. Its sample size is placements and sessions, not scans. A placement ends up
**accepted wrong** (the votes cleared their bar for a wrong name, so the robot would act without asking), **asks**, or
accepted right.

| Band | Meaning (thresholds are first guesses) |
|---|---|
| severe | a wrong name was accepted without asking |
| costly | a person is asked on 25%+ of placements |
| flaky | votes right, but 10%+ of scans wrong or unnamed |
| fine | none of those, over 5+ placements and 3+ sessions |
| not enough data | anything else; names with no placement at all are listed under the table |

"(up to)" is the highest accepted-wrong rate the placements so far can't rule out (95%). The truth comes from your check of
the mat before trials, your answers during a hand, and `wrist_review.py`, whose verdicts overrule the others (the report
counts how often). The first lines also show what play's extra checks would have done (`identity.doubt`); the bands stay on
the votes alone. Older sessions have only reviewed photos, and their votes are replayed over the whole placement, so they
say less than a played hand will. `status.py` prints the worst tiles and `session_report.py` the session's. Run it from
the main checkout. What's collected and why: `docs/name_data.md`.

## Reading results: `scripts/session_report.py`

```
python scripts/session_report.py                                 # the latest session
python scripts/session_report.py sessions/20261024_101500_bench  # a specific one
python scripts/session_report.py --all                           # every bench session side by side, with its setup
```

Shows the setup, success per tile and policy, stalls, clearances, ACT seconds, the scorecard (clean attempts,
and a ship / revert / extend verdict when the session paired two policies: `docs/scorecard.md`), interventions
(rescues per 10 attempts, unassisted vs. assisted successes) and saves a contact sheet (one image row per attempt).
Bench sessions also get close height: how high the tip was as the jaws closed, per policy and spot. Under 10mm
picked 127/130 and 20mm or more 3/25 (`docs/retrospectives/2026-11-03_eval_validity.md`). It explains a run; it doesn't decide one.
For a played hand (`play.py`) it adds the result, the grade, what you were asked and the turn times;
`--all` ends with one line per hand.

## Autonomous play: `scripts/play.py`

The robot plays one closed hand (no calls, no flowers). Your only job is to put each drawn tile on the
drop zone and press Enter. Why it works the way it does, and the targets it's measured against:
`docs/autonomy_plan.md`.

```
python scripts/play.py --layout even-hand13                 # play a hand
python scripts/play.py --layout even-hand13 --teach-zone    # move the drop zone: read the marker for its new position
python scripts/play.py --layout even-hand13 --resume        # carry on the latest play session's hand
python scripts/play.py --sim                           # no hardware: a fixed hand that wins in 3 draws
```

Before the first hand:
1. **The hand slots**: `even-hand13`, 13 spots in two rows (13 in one row don't fit the reach), part of the standard
   setup (`data/standard_setup.json`).
2. **The test layout** `even-hand14`: the same 13 slots plus one spot where the drawn tile will go (S13, WW at
   (0.192, −0.165)). Check it with `bench.py --layout even-hand14 --trials 2`, a tile on
   every spot: pickups and returns at every slot *and* at the drop zone.
3. **The drop zone**: lay ArUco marker 3 (DICT_6X6_250, as in MahjongBot) flat on the mat, uncovered, on
   even-hand14's extra spot. The first run finds it and saves its position (`data/drop_zone.json`, tied to the
   RealSense calibration). Every hand after that uses the saved position and never reads the marker, so you can move it or
   take it off the mat; the window outlines the zone, and the startup line gives its position. Only `--teach-zone` reads
   the marker again (a recalibrated RealSense asks for it too).
4. **The discard pose**: `bench.py`, `teach-discard`, over the container.

Play reaches the drop zone only once a hand is under way, so it won't start a hand (on the arm) until it's
checked: even-hand14 must still hold all of even-hand13's slots, have a spot within 25mm of the taught zone, and bench
sessions must have picked up and returned a tile there, and carried at least 2 from there to a slot
(`bench.py --layout even-hand14 --trials 2 --carry S13,S1`) landing a median of 12mm or less from home. If not, it prints the commands that fix it
(e.g. `layout.py --name even-hand14 --remove S13` then `--add S13 X Y` at the zone, then the bench walk) and
stops. `--skip-zone-test` plays anyway, logged. `status.py` shows the same check.

Each turn waits for a tile on the drop zone: after you press Enter, a scan has to see one within 25mm of the zone, or it asks again
and says where the zone is. A tile it can only box, not name, counts as there, and it asks you the name.

A hand: it reads the 13 tiles (asking about any it isn't sure of), then **shows you the whole hand to check against the table**:
Enter if it's right, or the wrong ones as `S9=5C`. It won't ask for a drawn tile until you've done this (`--resume` asks too, if
the hand never had it), and the draw prompt takes `S9=5C` as well, for a tile you spot later. Then each turn it reads the drawn tile,
checks the table still matches its hand, chooses the discard (lowest shanten, then the most improving
tiles still left, then kinds), drops it in the container, and if it came from the hand moves the drawn
tile into the slot it left, aimed so it lands on the slot's home. Each turn the terminal and the window say what it's after:
the tiles that win once it's ready, else how far from ready it is and which draws get it closer, with the copies of each still
left (4 minus its hand and its discards; there are no other players' tiles to count). Type `end` at the draw prompt when the
wall is empty. At the end you check its hand against the table and its discards (`D1`, `D2`...) against the
container: Enter if it's right, or the wrong slots
as `H3=5D`.

It asks rather than guess: a name the votes over the scans aren't clearly for, a read of 7C, 4B, 3C, EW, 6C, 5C, 2C or 9C (names
other tiles get misread as: `identity.ALWAYS_ASK_READS`, a stopgap, `docs/name_data.md`), a read under 0.80 confidence, a slot
that reads differently from the hand, a fifth copy of a tile, a pickup that failed 3 times (you move the tile). Each
question is logged, and the hand is graded **clean** (nothing asked; pressing Enter at the hand check doesn't count, correcting
a tile there does), **attended** (questions only), **rescued** (you touched something) or **silent error** (the end check found a
tile it had wrong).

Your word on a tile holds: once you've named it, or pressed Enter at the hand check, later scans that read it differently are ignored
(and logged) until the tile moves more than 5mm. Then you're asked once, after the wrist camera's look, and your answer holds from
there. The tile's word ends when it leaves the slot. The wrist camera disagreeing at a pickup still asks about a tile you only passed
with Enter, since that's a new read. `--resume` starts without these words, so a tile that reads differently is asked about once again.

A read on that list gets the wrist camera's look first: the arm hovers over the tile before you're asked. While registering the
hand, if the camera agrees you aren't asked about that tile (you still check it with the rest of the hand); if it disagrees you
are, with both reads. A drawn tile on the list is always asked about, since no hand check follows it. Any question about a drawn
tile gets the look first, whatever made it uncertain (split votes, a low-confidence read, the list), with the camera's read in the
question or a note that it got none. A camera found lost at a read is reopened and read again.

A window shows what each question is about: the mat zoomed in on the tiles, every slot labelled with the robot's name for it, and
the wrist camera's view beside the tile in question once the arm has looked. When you're reviewing names (the hand check, a name
question, a slot that doesn't match, a fifth copy), the tiles to look at stand out from a dimmed mat, thick orange (thick yellow:
the one being asked about now). Each is labelled with what it might be (`S9 5C or 9C`: the robot has 5C, it could be 9C) and listed
under the picture with every other name and why. At the hand check that means another name in the votes (5% or more), a read on the
list above, a read under 0.80, or the wrist camera disagreeing; you still check all 13, and the marks ask nothing and don't affect
the grade. While the arm is over the mat the picture is the latest scan's, taken at rest and marked `[mat from before the arm moved]`,
so the arm never covers the tiles, and each look saves it beside the wrist photo (`overview_before.jpg`). You answer in the terminal, so click back on it if the window takes the keyboard. `--no-view` turns it off.

Before each pickup, at the hover, the wrist camera reads the tile too (`assets/name_check.pt`). If it agrees nothing changes. If
it disagrees you're asked, with the arm waiting over the tile, and if your answer changes the name of the tile about to be
discarded, the discard is chosen again and the arm goes straight to it. It never names a tile on its own.

A tile the policy picked up goes straight from there to the container or the slot, without a trip to rest. Everything else goes back
to rest first, a look with the wrist camera included, so the RealSense has a clear view whenever it needs one, and the scan after a
tile is set down answers for its pickup too. A miss, or a key you pressed during the pickup, goes to rest and scans at once. `--via-rest`
goes to rest and scans after every pickup too, as the bench does.

Every step is checkpointed: after Ctrl-C or a stop, `--resume` carries on (it refuses while the arm
holds a tile).

| Option | What it does |
|---|---|
| `--layout NAME` | **required** (except `--sim`): the 13 hand slots |
| `--policy ID` | policy from `data/policies.json` (default: the registry's default) |
| `--teach-zone` | move the drop zone: read the marker for its new position, which later hands carry over; it must be uncovered |
| `--resume [FOLDER]` | carry on a play session's hand (default: the latest) |
| `--skip-zone-test` | play even if the bench hasn't picked up and returned a tile on the drop zone (logged) |
| `--no-name-check` | play without the wrist camera's second read |
| `--read-hand` | after registering the hand, hover over all 13 tiles and read each with the wrist camera (a tile never discarded is otherwise never read) |
| `--via-rest` | go to rest and scan after every pickup before setting the tile down, as the bench does (slower) |
| `--no-view` | no window showing what each question is about; the terminal only |
| `--sim` | simulated arm, cameras, mat and policy (no second read) |

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
python scripts/record.py --dataset Yewy/mahjong_mat_dagger_v8 --policy pickup_mat_cl_v8                  # DAgger corrections
python scripts/record.py --dataset Yewy/mahjong_wall_dagger_v8 --policy pickup_mat_cl_v8 --wall-station F --hover-jitter-mm 6   # DAgger on wall picks
python scripts/record.py --dataset Yewy/mahjong_zone_dagger_<v> --policy <ID> --wall-station zone --wall-layers flat --hover-jitter-mm 6   # DAgger at the drop zone, on the policy crude_run.py drove
python scripts/record.py --dataset Yewy/mahjong_facedown_dagger_<v> --policy <ID> --layout even-hand14 --face-down --start-spot S11 --repeats 3 --hover-jitter-mm 6   # DAgger on green-back tiles at layout spots
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

**Wall picks (`--wall-station ID`)**: aims at that station of the layout `wall-stations` instead of a tile the detector found, because it can't see green backs. Before each episode the terminal says what to lay, with a side view, and
waits for Enter. The layers in `--wall-layers` are taken in turn, and a discarded episode tries the same layer again. The hovers are `wall_test.py`'s: a top tile's is one tile higher and `--wall-top-back-mm` nearer the robot, and a bottom tile's is the policy's own. A
tile still held afterwards goes in the discard container, which needs a discard pose. For DAgger, take over before the jaws close and bring them down so they close low on the tile and centred along it: a grip near the top edge lets the tile swing round in the jaws.

**Face-down spots (`--layout NAME --face-down`)**: any spot of a layout with the tiles green side up, e.g. the hand's right end with `--layout even-hand14 --face-down --start-spot S11 --repeats 3`. The detector can't see a green back, so each episode names the spot, waits for Enter (`s` skips the spot, `q` quits) and aims at the spot's saved position; `--hover-jitter-mm` stands in for where a real tile sits. Lay the layout's other tiles green side up as they'd be. A held tile goes in the discard container afterwards: a scan can't check where a returned green back landed. Each log row has `face_down`.

**The drop zone (`--wall-station zone --wall-layers flat`)**: the same, aimed at the drop zone, for the tile `crude_run.py` sets there and picks up
again. Lay the crude run's hand green side up first (S13 stays empty: the zone is there), then one green-back tile on the zone each episode; the
hover jitter stands in for where the arm's own set-down leaves it.

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
| `--face-down` | with `--layout`: tiles lie green side up, so each episode aims at the spot's saved position instead of a detected tile; a tile still held afterwards goes in the discard container (see below) |
| `--hover-gripper PCT` | gripper opening at the hover (default 90%). A narrower start skips the slow narrowing: v2n's trimmed episodes start at 27.2, so record at that to compare with v2n. Changing it means a new dataset |
| `--hover-offset-mm`, `--clearance-mm` | the pincer tip is commanded this far above the grasp point (default 45), then settled until it measures this far (default 40) |
| `--no-correction` | hover over the raw detection |
| `--no-view` | no wrist camera window |
| `--no-return` | don't set a held tile back down: it's released at rest, as before |
| `--policy ID` | DAgger: this registered policy drives from its own hover until you press A on the gamepad (see above) |
| `--wall-station ID` | record wall picks at this station of `wall-stations` (see above), or `zone` for the drop zone's saved position (`data/drop_zone.json`); not with `--layout` |
| `--wall-layers LIST` | with `--wall-station`: `top`, `bottom` and `flat` (one tile on its own, as on the zone), taken in turn (default `top,bottom`); repeat one for more of it, e.g. `top,top,bottom` |
| `--wall-top-back-mm N` | with `--wall-station`: a top tile's hover starts this much nearer the robot (default 10, what worked at F) |
| `--stick-commit-s S` | a gamepad stick moves the arm only once held that long (default 0.1), so flicks don't. The right stick moves height *or* wrist roll: whichever moves first, until it's back at centre |
| `--deadzone F` | stick deadzone, fraction of its travel (default 0.35, as collect_corrections: a thumb resting past XLeRobot's 0.15 pulled the arm 87mm). The wrist window's bottom line shows what the sticks read (red while they move the arm), and a stick off centre when you take over is said in the terminal. Lower it if gentle pushes don't register |
| `--left-stick-max-mm-s F` | the jaws' speed at a full push of the left stick (pan, reach), with lighter pushes scaled down from it, so it can't jump to a fast move just past the deadzone (default 30, 0 for no cap). The right stick's height is capped at 50 mm/s the same way, so `+` past 0.5x changes little. The L/R tilt buttons and wrist roll aren't capped. The window's line then shows the scaled stick: a full push reads below 1, its raw value in brackets |
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
python scripts/quality_report.py --dataset mahjong_surfaces_standard_mat_eval_v1 --clean Yewy/mahjong_surfaces_standard_mat_eval_v1_clean
```

`--drop EPS` leaves reviewed episodes out of training without deleting them: they're listed in
`data/dropped_episodes.json` (with `--reason`), marked `x` in the table, and `make_policy.py` skips them.
The recording keeps its episode numbers, so you can go on recording into it.
`--fix-gripper EPS` is for an episode whose only fault is a gripper command that stopped short of 0 on
the tile: `make_policy.py` builds it with the command carried on to 0 at the rate it was closing
(`data/episode_fixes.json`). The jaws had already stalled on the tile, so the frames still match.

`--clean REPO` writes a copy under a new name with every stop in the commands cut to one frame (the build's idle
trim leaves about half of a stop-go demo's still frames) and dropped episodes left out; each row of its episode log
names its source episode. The recording isn't changed. A move backwards stays: cutting it would make the arm jump.
A cleaned eval set runs faster than the demos a policy trained on, so read its `arm_mae` and `grasp_mm`, not its timing:
on `mahjong_surfaces_standard_mat_eval_v1` the descent lag went from 14 to 29 steps and the close error from 0.34s to
0.76s for v6 with its arm error barely moved, past gates calibrated on the standard trim.

`--against REPO` compares each spot's hover→grasp move with the other recording's at the same place
(within 15 mm), counting only demos that steered. A demo lowered straight down only says the tile was
close enough for the jaws (a third of v2's were). Shifts beyond ~6 mm in one direction at most spots
are worth a `collect_corrections.py` check; a few spots differing is as likely to be how the demos were done.

Options: `--worst N` (rows shown, default 15), `--flag SCORE` (threshold, default 4), `--csv PATH` (save the table),
`--phases`, `--against REPO`, `--drop EPS --reason TEXT`, `--fix-gripper EPS --reason TEXT`, `--clean REPO`.

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
slow speed. `--left-stick-max-mm-s` (default 30, 0 for no cap) is the jaws' speed at a full push of the left
stick, as in `record.py`.

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

## The wrist-camera name check

A small classifier reads a tile's name from the wrist camera at the hover, and is never told what the detector said. It trains on
wrist photos a person named (`wrist_capture.py` views, photos reviewed in `wrist_review.py`). Its exam is whole sessions in
`data/name_check_exam.json`, which never train it. `docs/name_check_plan.md` has the evidence and the gate.

```
python scripts/train_name_check.py --dry-run          # the photos it would train on
python scripts/train_name_check.py                    # weights in outputs/name_check/name_check.pt (GPU, about a minute)
python scripts/name_check_eval.py --weights outputs/name_check/name_check.pt
python scripts/name_check_eval.py --list              # what the exam holds
```

`train_name_check.py`: `--out PATH`, `--seed N` (default 0: choose before any score), `--epochs N`, `--sessions-dir DIR`, `--dry-run`.
`name_check_eval.py`: `--weights PATH` (default: the installed `assets/name_check.pt`), `--sessions-dir DIR`, `--list`. It refuses
weights that trained on an exam session, and scores go in `data/name_check_eval/`. The score that matters is **accepted and wrong**:
the two agree on a wrong name, so nothing catches it. To install new weights, copy them to `assets/name_check.pt` and commit them
with the score.

## Holding tiles up to the base camera (face-down plan, test 1)

`docs/face_down_plan.md`'s first test: no arm, just the LeKiwi base's front camera (`/dev/basecam`, its own USB port: check
`ls -l /dev/basecam /dev/wristcam` first). You hold each tile up wherever the arm can actually hold it (close, at an angle; no fixed distances), right way up and turned 180°, by day and
again in the evening; then the name classifier and the detector are scored on the photos.

```
python scripts/holdup_capture.py                                  # 7 tiles x 2 ways up x 4 holds, lighting asked first
python scripts/holdup_capture.py --tiles 5C,3C,2C,9C --holds 6 --frames 4
python scripts/holdup_capture.py --sim                            # synthetic camera, no hardware
python scripts/holdup_score.py                                    # is training a reader paying off? every holdup_check.py result
python scripts/holdup_score.py --installed                        # the installed readers on the latest holdup session (test 1)
python scripts/holdup_score.py --installed sessions/<time>_holdup --weights outputs/name_check/name_check.pt
python scripts/holdup_check.py --mark-boxes                       # once per taught pose: drag a box round where the held tile shows
python scripts/holdup_check.py                                    # can a classifier learn this view? trains on some photos, scores others
python scripts/holdup_check.py --sessions sessions/<time>_holdup --by hold --seeds 3
python scripts/holdup_check.py --sessions sessions/<evening>_holdup,sessions/<day>_holdup --test-session <day>_holdup   # another light
python scripts/holdup_check.py --framing whole                    # the whole frame (hand-held photos have no tile box)
python scripts/teleop_view.py                                     # drive by gamepad watching the wrist and base cameras (arm)
python scripts/teleop_view.py --go manual_1 --name base_view_     # start at a saved pose; s saves the next one
python scripts/teleop_view.py --sim
```

`holdup_capture.py`: `--tiles A,B` (default 5C,3C,2C,9C,7C,1B,EW), `--holds N` per tile and way up (4; a new place and angle each), `--frames N` per shot
(3), `--gap-s` between a shot's frames, `--no-view` (Enter in the terminal instead of the preview window's space), `--sim`. Keys
in the window: space takes the shot, `s` skips, `b` redoes the last, `q` quits. Photos go to
`sessions/<time>_holdup/photos/<NAME>/`; the name you hold is the truth.
`holdup_score.py`: `SESSION` (default: every real holdup session; with `--installed`, the latest). By default it lists every
`holdup_check.py` result, oldest first: what each trained and tested on (another session, or the same one, which is optimistic), the
fresh reader's right rate, the control with the tile greyed out and the best constant answer, the per-tile rates and confusions,
then which results have a clean control (within 15 points of the constant) and which of those were on a session never trained on:
that last one is what says training pays off. Nothing is trained or run; it reads the JSON files. `--installed` is test 1: the
installed readers, `--weights` (the classifier) and `--detector`, which were trained on other views and read ~0% here. It prints rates per slice (all,
each way up, each daylight setting) and per tile, for three framings of the classifier (the wrist camera's
crop, a square round the detector's best box, the whole frame; each read as it is and upside down, reported apart) and for
the detector, then writes `holdup_scores.json` and `holdup_sheet.jpg` (every photo with its guesses, wrong ones in red: look
through it). A reader counts as reading only if it is right on two or more tiles: one that answers "3C" to everything is right
on every 3C photo and nothing else. The plan's stop is "nothing reads anywhere the arm can hold a tile": the last line says
if anything did.

`holdup_check.py`: `--sessions A,B` (default: every real holdup session; simulated ones never count), `--by auto|session|hold`
(auto holds out the newest session when every tile in it is also in an older one, else the last `--test-holds` 2 holds of each
tile and way up, which is optimistic: the same light and hands are on both sides), `--test-session NAME` (hold out that
session and train on the others in `--sessions`, whatever `--by` says; only the tiles another session has are tested, the rest
left out and counted: train on an evening, test on a day, or a new room; it writes `holdup_check_held_out.json` into that
session), `--framing tile|whole` (tile), `--epochs`
(150), `--seeds` (3, for the check and again for the control), `--mark-boxes`. It is not a fine-tune of the installed weights:
it trains a fresh YOLO11n-cls from its ImageNet weights, for just the tiles you held, and scores the held-out photos against
chance and the best constant answer. `tile` framing crops each photo to the tile box of the pose the arm held it at (a box per
`base_view_*` pose in `data/camera_poses.json`, drawn with `--mark-boxes` on the average of that pose's photos: Enter saves, `c`
keeps the old one); photos with no box, such as hand-held ones, are left out. The control trains again on the whole frame with
the tile box greyed out: if that beats the best constant answer by 15 points, the verdict is BACKGROUND (the scene names the
tile, as on 2026-11-06 when each tile's holds were taken in a row). With no boxes there is no control, and a pass reads
UNCONTROLLED. The bar (70% or more, on 3+ tiles, well above the constant answer, to be worth a real capture; 40% or less to
look at the photos first) is in `docs/face_down_plan.md` test 1b. It writes `holdup_check.json` into the newest session.

`teleop_view.py` (moves the arm: asks a person to run it): the gamepad drives the arm as in `collect_corrections.py`, and one
window shows the wrist and base cameras side by side with the commanded pose, the tool point and its distance from the
shoulder. Keys in the window: `s` saves the commanded pose as the next `--name` (default `base_view_`) in
`data/camera_poses.json`, `c` saves both frames to the session's `captures/`, `g` returns to `--go`'s pose, `q` rests and
quits. Options: `--go POSE`, `--name PREFIX`, `--device PATH` (gamepad), `--sim`. `data/camera_poses.json` starts with the two
poses placed by hand on 2026-11-05 (`manual_1`, `manual_2`: measured, not commanded).

`crude_run.py` (moves the arm: asks a person to run it; `docs/face_down_plan.md`, test 8): the whole face-down turn, crude on purpose, for a list of where it breaks. Lay 13 tiles green side up on the hand's slots and name them with `--hand`, in the layout's order.

```
python scripts/holdup_check.py --sessions sessions/20261106_195636_holdup,sessions/20261109_100829_holdup,sessions/20261109_122910_holdup --save-reader   # once: train the reader (GPU, minutes)
python scripts/crude_run.py --hand 1C,2C,3C,5C,5C,7C,7C,9C,9C,1B,1B,2C,3C --turns 6        # a wall one tile high
python scripts/crude_run.py --hand ... --turns 6 --stacked --top-back-mm 10                 # two high: top, bottom, feed
python scripts/crude_run.py --hand ... --sim                                                # simulated arm and cameras
```

A turn: the arm draws from the station, holds the tile up at each saved `base_view_*` pose, and sets it on the drop zone. A window shows the photos and the reader's guess: Enter accepts it, or type the name, and `X` says the face wasn't visible. `game.py` chooses the discard, and the arm plays it. Whatever fails is written down
(`crude_break`, printed as "Noted for the list") and you do that step by hand, in the outline the window shows; each turn ends by asking if anything else went wrong. If the jaws read too low for the held check but not as low as empty (the held reading fell from 9.5 to 3.0 over 2026-11-09), it asks whether there's a tile in them, and carries on if so.
`--leave-empty S7` is for a slot with no tile on the mat: still named in `--hand`, never chosen as a discard. A tile on S7 sits right behind station F and threw the draw off. Other options: `--station ID` (default `F`), `--layout NAME` (default `even-hand13`), `--policy ID`. The reader only knows the tiles it trained on (`--save-reader`, saved to `assets/holdup_reader.pt`); without it you name every tile. Results: `sessions/<time>_crude/`, with `crude_summary.json`. It needs a discard pose and a taught drop zone.

`wall_test.py` (moves the arm: asks a person to run it; `docs/face_down_plan.md`, "The wall: one station"): v8 draws face-down tiles from one saved spot, a station, while
you feed the wall by hand. Stations are in the layout `wall-stations` (`F` in front of the hand, `L` on its left; `F2` 5mm further out and `L2` 10mm further left, for more room beside the hand's tiles, e.g. `--stations F2,L2`, `record.py --wall-station L2`): lay a tile face up to green with `layout.py --name wall-stations`, tape around it, then flip it.

```
python scripts/wall_test.py --hover-check --stations F,L                  # 1. no policy: is the hover centred on a face-up tile?
python scripts/wall_test.py --push-check --stations F --trials 10         # 2. no arm: does a pushed row land in the same place?
python scripts/wall_test.py --stations F,L --cases control,flat --trials 5    # 3. which station, alternating
python scripts/wall_test.py --stations F --cases strip --trials 10        # 4. tiles touching end to end (then top, then bottom)
python scripts/wall_test.py --stations F --cases cycle --trials 12        # 7. the real thing: top, bottom, you feed, and so on
python scripts/wall_test.py --report                                      # every real session so far, scored
python scripts/wall_test.py --sim --stations F,L --cases flat --trials 2  # simulated arm and cameras
```

`--cases`: `control` (face up), `flat`, `strip` (a tile touching its far end), `top` (the hover one tile higher, the next stack behind), `bottom` (the next stack behind), or `cycle` alone at one station, where
`--trials` counts draws. Each attempt: a RealSense frame with the arm at rest, the policy, one base-camera photo if the jaws read held, the tile into the discard container, another RealSense frame, then one key from you, worst first:
`t` a person touched it, `2` lifted two tiles, `w` took the wrong tile, `k` knocked the top tile off, `n` moved the next stack, `p` pushed its own tile, `d` dropped it on the way, `a` closed on air, `c` clean (`q` quits). Enter isn't an answer. A cycle's non-clean
draw asks you to put the wall right. `--push-check` wants the wall laid with the lead stack's top tile face up, and passes if the worst push is within 3mm of where pushes usually land.
`--hover-back-mm N` starts every pick N mm nearer the robot than the station (up to 20; negative is further out), to test whether a pick landing too far out is an aiming error, e.g.
`--stations F,L --cases top --trials 5 --hover-back-mm 10`; `--report` keeps shifted tries apart from unshifted ones. Leave the mat alone until the outcome keys appear, since the photo after each try is taken just before them.
`--top-back-mm N` shifts top tiles only and leaves bottom tiles at the regular hover, which is what worked at F: `--stations F --cases cycle --trials 12 --top-back-mm 10`. A run that wasn't the test it was logged as (a wall laid another way, say)
is left out of `--report` with `--exclude-session SESSION --why "..."`; its files stay.
Other options: `--layout`, `--policy`. Sessions: `sessions/<time>_wall/` (`wall_attempt` events; per attempt `wrist.mp4`, `table_before_after.jpg`, `base.jpg`; `wall_summary.json`), `_wall_push`, `_wall_hover`. It needs a discard pose (`bench.py`'s teach-discard) and a saved `base_view_*` pose. Run `setup_check.py` first, and from the main checkout.

`holdup_arm.py` (moves the arm: asks a person to run it; face-down plan, tests 1b and 2): the arm picks up a face-down tile you laid
on a layout spot, holds it up to the base camera at the saved `base_view_*` poses, saves frames and sets it back down.

```
python scripts/holdup_arm.py --empty                       # first: visit each saved pose with nothing in the jaws, one Enter each
python scripts/holdup_arm.py                               # collect what is short: default tiles, 6 holds per way up
python scripts/holdup_arm.py --tiles 1B,EW,7C --spots S5,S9 --rounds 3
python scripts/holdup_arm.py --sim                         # simulated arm and cameras
```

`--empty` writes `sessions/<time>_holdup_dry/` with a base and a wrist frame per pose (`dry_run/`, `dry_run_sheet.jpg`) and prints each
pose's height above the table (flagged under 30mm), reach and nearest joint limit. A collecting run counts the holds already in every
real `*_holdup` session (by hand and by arm) and lays the tiles furthest short in groups, one on each `--spots` spot. Each round
picks every spot once (pick, rest, every pose with a small nudge, `--jitter-deg` 2 or 0 = as taught, rest, set it back down), so
holds of different tiles alternate; after `--rounds` rounds (at most 2, fewer if fewer holds are short) every tile moves one spot
along and is laid the other way, until each tile has been held from every spot both ways up. A group short of tiles is filled
with ones already at target, so no tile is collected alone. This order is what `holdup_check.py`'s control asks for: collected a
tile at a time, the scene named the tile (2026-11-06). Several tiles need two or more spots. A tile is laid one of two ways: face
down where the layout's face-up tile would lie (flipped sideways about its long edge, the top of its picture at the same end), or
that turned 180° on the mat. Which way up the camera sees each depends on the poses, so after the session's first hold it shows that photo
and asks whether the picture reads right way up (`u`, recorded as `upright`) or upside down (`d`, `turned`), and labels every later hold
from the answer (in that window: `u` or `d`). A miss, a drop or an unreachable spot stops that tile and asks you.
Where to lay each tile comes up as a live RealSense window, as in `layout.py`: the target spot's outline, the tile's name, and the short
end where the top of its picture goes drawn thick and labelled TOP (`--layout-top far|near` says which end of a spot a face-up layout
tile's picture points to; `far` = away from the robot, which is a guess until you check it on one tile). The detector can't see a
face-down tile, so nothing turns green: line it up by eye. Space/Enter = laid, `s` skips, `q` quits; the window closes before the arm
moves. After a miss, or a tile lost on the way to a pose, the same window opens again for that tile and laying (same TOP end) to put
it back by hand: Space/Enter tries again, `s` skips the tile, `q` quits. `--no-view` (and `--sim`) use a terminal prompt instead
and don't open the RealSense. Other options: `--tiles`, `--target-holds`, `--layout` (the standard setup's),
`--hover-lower-mm` (as `bench.py`'s, for a spot where the policy closes high), `--spots` (S5,S7,S9, a tile on each; S8 missed every pickup on 2026-11-05), `--max-misses` (2: that many misses in a row on a spot and
it sits out the rest of that arrangement, instead of asking forever), `--poses`, `--frames` per pose (3), `--settle-s`, `--gap-s`, `--seed`, `--policy`. Photos and
`holdup_shot` events (with `source: arm` and each frame's pose) go where the hand-held ones do, so `holdup_score.py` and `holdup_check.py` read them.

### Blurring faces in the base camera's photos

The base camera looks out over the room, so a person near it is in the photos. Everything the base camera saves is blurred as it is
saved (`mahjong/privacy.py`: the hold-up photos, `view_check.jpg`, the dry run's base frames, `teleop_view.py`'s base captures), with
a pose model in `assets/yolo11n-pose.pt` (Ultralytics' public yolo11n-pose weights, 6 MB, CPU). The scripts stop before touching
hardware if the file is missing and print the `curl` that fetches it. For photos already on disk:

```
python scripts/blur_faces.py                          # dry run over every session: counts, and a sheet of what would change
python scripts/blur_faces.py --apply                  # rewrite them blurred (the originals are not kept)
python scripts/blur_faces.py sessions/<time>_holdup --sheet /tmp/faces.jpg
```

`blur_faces.py` covers `photos/**`, `view_check.jpg`, `dry_run/*_base.jpg` and `captures/*_base.jpg` of each session; it changes nothing
without `--apply` and always writes a sheet of the photos it flagged, as they would look. Look through it, and through some photos it
did not flag: it lowers the chance of a face in a photo, it doesn't rule it out. A person anywhere near the camera is found, even
turned away or half out of the frame, and where too little of the head shows the top of their box is blurred. A "person" covering more
than 65% of the frame is ignored, because the held tile seen close up passes for one (it was blurred whole on the first try).
Haar face detection, tried first, blurred the printing on the tile and still missed faces.

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

Current exam (Roboflow v7, downloaded to `data/detector/exam_v7`): 103 frames over 4 capture sessions, 48 of them on the
standard mat, where the live detector gets 80.9% right overall (`--by session` shows it).

To check a labelled set by eye, `label_sheets.py` makes one picture per tile name, of every box labelled with it. A wrong label is
the odd one out, and `lists.txt` beside the sheets names each crop's file so you can fix it in Roboflow:

```
python scripts/label_sheets.py data/detector/exam_20261104_mat_all             # into <folder>_sheets/
python scripts/label_sheets.py data/detector/exam_20261104_mat_all --names 2C,3C --out /tmp/sheets
```

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

v8 (2026-11-04, live since: `tile_detector@47bbd02d`): v7x3's training set + 93 frames of a played hand (`hand_frames.py --onto`, no
repeats: they're near copies already). On the exam, without flowers: **98.8% right overall** (v7x3: 97.6%), **98.3% on the standard
mat** (v7x3: 96.8%). Box centres moved a median 0.16 px, at most 0.76. Not yet run on the arm.

v7x3 (2026-11-04, live until v8: `tile_detector@24a163b4`): v6's frames + 61 of the standard mat (Roboflow training v7, split by
scene, session frames repeated 3× in train). On the current exam, without flowers: **97.6% right overall** (v6x3: 86.2%), and
**96.8% on the standard mat** (v6x3: 80.9%). Box centres moved a median 0.1-0.15 px. Not yet run on the arm.

v6x3 (2026-10-28, live until v7x3: `tile_detector@0c4b15eb`): v6 = the July frames + 190 session frames
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

**Frames a person already named.** `scripts/hand_frames.py` labels a played hand's or a bench's frames from a person's answers
(names) and the detector's boxes, so they don't need correcting in Roboflow (`mahjong/hand_truth.py`):

```
python scripts/hand_frames.py 20261104_183142_play                                     # per tile: right, wrong, missed
python scripts/hand_frames.py 20261104_183142_play --out data/detector/train_v8 --onto data/detector/train_v7x3
python scripts/hand_frames.py 20261104_150542_bench --weights NEW.pt --against assets/tile_detector.pt
# --sessions-dir DIR (from a worktree), --per-arrangement N (default 2), --per-hard-name N (default 40)
```

**4. Wrist-camera photos (for a name check at the hover).** The wrist camera sees a tile from ~4 cm at
the hover, far sharper than the RealSense. `wrist_capture.py` collects photos to train a check there:
it scans the mat, lists the tiles left to right with the detector's names (numbered on the RealSense
view) for you to confirm, then hovers over each one `--views` times (default 8): the first centred at
40 mm, the rest moved by up to `--jitter-mm` (8), at heights cycled from `--clearances-mm` (40,55,70)
and the wrist rolled by up to `--roll-deg` (15). It never touches a tile. When the set is done the arm
rests, the terminal rings (plus a desktop notification), and it prints every name's wrist views at the
hover from all sources (its own sessions, bench attempts' first frames and hover photos, recorded episodes) and which
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
there, the worst-read tiles on the standard surface (`name_report.py`), and how many merged branches are waiting to be
closed.

**Changing the default policy**: on a branch, never on `main`, always with `--set-default ID --evidence ...` (and `--why`;
`--since YYYY-MM-DD` for a decision made earlier). It writes a `default_note` next to the default in `data/policies.json`; a
test fails if the default changes without one. It then lists what still names the old default: the README policy list's
`(default: ...)`, `record.py --policy` DAgger examples in README.md and CLAUDE.md, and ROADMAP.md's status table. The
pre-commit tests fail until all agree, so change them in the same branch, commit them with `data/policies.json`, and merge
the PR with `merge_pr.py`. Changed alone on `main` (2026-11-06), the default could be neither committed (the docs tests) nor
pulled past (merge_pr.py stops on uncommitted changes to a file the PR also changes).

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

It stops before asking, with nothing changed, if the pull after the merge would fail: `main` here has commits GitHub doesn't, or
the PR changes a data file with uncommitted changes here. Uncommitted data the PR doesn't touch is fine.

### Data files and merging

The arm's scripts append to files in `data/` as they run, so this checkout is often "dirty" with data. That's normal.

| What | Where it goes | When |
|---|---|---|
| Code, docs, anything a person reviews | a branch and a PR, merged with `merge_pr.py` | when it's ready |
| Samples the scripts log (`hover_corrections.jsonl`, `return_corrections.jsonl`, reviews) | straight to `main`, in one step below | after a hardware session |
| A decision (the default policy, exam lists, scores, new weights) | a PR, with its evidence; commit an exam list the session it's added, so no training can pick those frames up | when it's made |

Save a session's samples in one step, so a commit never sits on `main` here without being on GitHub:

```
git add data/hover_corrections.jsonl data/return_corrections.jsonl
git commit -m "Samples from 20261104_150542_bench"
git pull --rebase origin main && git push origin main
```

Name only the files you mean to save, and don't start a merge between those commands. Branches don't carry samples: they reach
`main` from this checkout, where the hardware runs, and these three files merge by union if two ever meet.

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
