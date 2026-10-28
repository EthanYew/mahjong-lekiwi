# mahjong-lekiwi

SO101 arm on a LeKiwi base picking up mahjong tiles: RealSense detects a tile → closed-loop IK hover
over it (measured correction, settled from above) → an ACT policy grasps from the wrist camera.
Rebuilt 2026-10-23 from `~/Projects/MahjongBot` (archive; don't import from it).

## Running

- Python: `~/miniforge3/envs/lerobot/bin/python`, from the repo root (scripts add it to `sys.path`).
- Fast tests (no hardware, seconds): `python -m pytest -m "not slow"`. Everything: `python -m pytest`
  (slow = loads policies from the HF cache, needs the GPU). A pre-commit hook runs the fast tests
  (~20 s); turn it on once per clone with `git config core.hooksPath scripts/hooks`.
- No-hardware run of the whole bench: `python scripts/bench.py --sim --trials 3 --tile 8D` (without
  `--tile` it asks which tile). Paired comparison across the mat: `--policy a,b --layout NAME --trials N`
  (`--trials` counts rounds: N at every spot, each policy once per round, alternating a b / b a from the
  same hover offset, so N × 2 × spots pickups; each attempt saves `wrist.mp4`).
- DAgger: `record.py --dataset Yewy/<new> --policy pickup_mat_cl_v3c`. The policy drives, and the first
  gamepad touch hands you the arm. Training uses only the frames from the takeover on, as deltas from
  the hover: `make_policy.py --source <main> --add <dagger>`.
- Training recordings: `record.py --hover-jitter-mm 6`, so the policy finds the tile in the image (v2
  learned a fixed move per position). Eval recordings have no random jitter; the golden set's `jitter`
  slice uses a fixed list (`--hover-offsets-mm 6x,-6y`) so it can be recorded again identically.
- Hardware sessions write `sessions/<time>_<kind>/`; read one with `python scripts/session_report.py
  [folder]` (setup, success per tile/policy, stalls, clearances, ACT seconds, interventions, how far
  returns landed from home, contact sheet) before asking what happened; `--all` compares every bench session with its setup conditions.
  Run hardware scripts from the main checkout: `sessions/` is per-checkout and gitignored, so a
  worktree's sessions go with it.
- Interventions (`mahjong/interventions.py` has the classes and why, ROADMAP 3): → during ACT marks a
  moment without stopping it, and the RealSense scans around each attempt catch knocked or replaced
  tiles, which the bench asks you to confirm (YOLO flickers). "Touched the tile to help" is a rescue
  (the success becomes assisted); "the arm bumped a tile, no one helped" is a failure that leaves a
  success clean. The scorecard number is rescues per 10 attempts.
- Hardware scripts connect everything first (`mahjong/hardware.py`: retry / quit if something's
  unplugged), then ask the setup questions, then create the session folder.
- A pickup stops itself once the tile is held and lifted for 1.5s (`policy.py`), capped at 16s.
- After packing up or moving anything: `python scripts/setup_check.py` (camera vs robot, table plane, a tile
  in the wrist view at the hover on 3 spots, parked pose, brightness) against `data/setup_reference/`, taken
  with `--save-reference` on a setup that worked. Nudge the camera back to green rather than recalibrating.
- Autonomous play: `python scripts/play.py --layout hand13` (`--sim` without hardware). One session is one
  hand; it asks instead of guessing a tile name, and the hand grades clean / attended / rescued / silent
  error (`interventions.hand_grade`). Silent wrong tiles are a gate, not a target: one stops play.
- Score a policy offline (GPU, minutes): `python scripts/offline_eval.py --policy pickup_mat_cl --against
  pickup_mat_cl_v1 [--by spot|lighting]`; tables per slice and the gates.

## Layout

| | |
|---|---|
| `mahjong/config.py` | the one setup: ports, cameras, table plane, rest poses, defaults |
| `mahjong/kinematics.py` | closed-form FK/IK; `pincer_tip`, `hover_command`, `grasp_point` |
| `mahjong/arm.py` | all motion (`hover`, `place`, `rest`, `go_to_pose`) |
| `mahjong/policy.py` | registry (`data/policies.json`), loading, the ACT control loop |
| `mahjong/perception.py`, `camera.py`, `gamepad.py` | RealSense tiles, wrist camera, XLeRobot gamepad |
| `mahjong/hover_correction.py` | measured hover correction (`data/hover_corrections.jsonl`; live sources in `data/hover_model.json`) |
| `mahjong/return_correction.py`, `scripts/return_samples.py` | where to set a held tile down so it lands at its home (`data/return_corrections.jsonl`, from each return's after-return scan) |
| `mahjong/dataset.py`, `quality.py` | trims + re-anchoring + resize; episode quality |
| `mahjong/session.py`, `sim.py` | session folders; simulated arm/camera/perception/policy |
| `mahjong/setup_check.py`, `scripts/setup_check.py` | is the workspace where it was: measurements vs a reference (`data/setup_reference/`) |
| `mahjong/interventions.py` | what a person had to do: classes, questions, derived from session events |
| `scripts/play.py`, `mahjong/game.py`, `identity.py`, `markers.py` | autonomous play (`docs/autonomy_plan.md`): the turn loop on the bench's pieces; shanten and the discard choice; name votes per hand slot; the ArUco drop zone (`data/drop_zone.json`) |
| `scripts/` | `record`, `collect_corrections`, `hover_samples`, `bench`, `session_report`, `quality_report`, `make_policy` |
| `mahjong/layout.py`, `scripts/layout.py`, `spot_novelty.py` | tile layouts (`data/layouts.json`, committed once a recording uses one; ghost photos and captures in `data/layouts/` stay local) drawn on the RealSense view; place tiles to green; how far each spot is from a training recording's tiles |
| `mahjong/offline_eval.py`, `scripts/offline_eval.py` | scores per slice on episodes never trained on (`data/eval_split.json`) |
| `mahjong/detector.py`, `scripts/capture_frames.py`, `mine_frames.py`, `detector_eval.py`, `split_by_scene.py`, `train_detector.py`, `wrist_capture.py`, `wrist_review.py`, `mahjong/wrist_photos.py` | the tile detector's loop: scan frames saved in `sessions/*/scans/`, frames it struggled with exported pre-labelled for Roboflow, downloads re-split by scene, wrist-camera photos of each tile from jittered hovers (gathered from the sessions, reviewed in `data/wrist_review.jsonl`), the exam (`data/detector_exam.json`), fine-tuning (ROADMAP 10) |

## Invariants (each cost a debugging round on hardware)

- **Never start motion from the measured pose.** The servos (lerobot's I=0) sit 5-13mm below their
  command; re-commanding the measured pose makes the arm drop again. Interpolate from `Goal_Position`
  (`Arm.move_to`), seed teleop from `Arm.commanded()`. `tests/test_arm.py` catches a regression.
- **Arrive at hover heights from above** — a P-only servo holding against gravity stops further short
  arriving from below. `Arm._settle` handles it.
- **Hold the gripper's commanded value** until a release. Policies grip by commanding 0 (jaws stall at
  ~11 on a tile); commanding any fixed "closed" value or the measured reading loosens or opens them.
- **A return opens only just past the jaws' reading on the tile** (~11 + `GRIPPER_RELEASE_PAST_HELD`
  6 ≈ 17%): 25% (`GRIPPER_RELEASE`, the widest opening held around a tile in recordings, still used
  over the discard container) turned a neighbour on 2026-10-23. Unconfirmed on hardware that 17% always
  lets go. The policy's own descent opens to its hover gripper (90% for v1/v2, 27.2 for v2n), which
  only a new dataset or trimmed episodes change.
- **Heights are above the grasp point** (`config.grasp_z`, table + 5mm); the tile top is ~15mm higher.
- **Rest is where the unpowered arm settles** (`config.REST_POSE`, from `sessions/parked_pose.json`). The old
  folded rest sagged ~8° onto the stops at every disconnect, and each connect then lifted the arm via
  `RAISED_REST` and folded it back before doing anything. Re-measure it if the arm or its stops change.
- **The right stick moves height or wrist roll, never both** (`gamepad.EXCLUSIVE_AXES`): a pull down leaked
  into roll past XLeRobot's 25° snap and turned the wrist mid-descent (2026-10-27, 7B).
- **Returns aim at the tile's home, never where the jaws closed.** Returning to each grasp point carried
  every return's error into the next pickup: tiles walked 77mm (4D, 9 rounds) and 100mm (6C, 10 rounds)
  toward the edge of the reach on 2026-10-27. Over 95 bench returns a pick + return moved a tile 5.7mm
  (median), leaning +2.8mm in y; the jaws pull a tile only partway to their centre (55-65% of its offset
  in the jaws stays with it), so where it lands depends on that grasp, which the return model
  (`mahjong/return_correction.py`) takes as input (landing error 10.5mm → 6.8mm RMS, leave-one-position-out).
  On the arm (`20261028_102732_bench`, 15 rounds at 6C, v3c) returns landed 3mm from home (median; worst
  4mm) with no build-up, vs 11mm (worst 17mm) at 6C the day before; other spots unchecked. Homes last the
  session, by spot or tile name: a tile moved on purpose less than `HOME_RADIUS_M` (40mm) is put back at
  its old home. The fit pools every policy; one that grips differently may need its own
  (`docs/retrospectives/2026-10-28_return_homes.md`).
- **Moves between rest and the table go via `RAISED_REST`** so the gripper clears the LeKiwi base.
- **The policy reaches further than the IK will plan.** It closed on 4D 11-12mm beyond the hover,
  ~0.29m from the shoulder, outside the IK's envelope (15° tilt, 10° joint margins), so the return
  was refused. `Arm.place` moves a return up to 15mm toward the shoulder, checking every height down
  to the release (reachable at the approach alone would drop the tile from there).
- **Datasets store joint deltas from each episode's first frame** (gripper too). A policy only works from
  the hover its episodes started at — that hover is stored per policy in `data/policies.json` and the
  bench uses it. Changing the hover (e.g. gripper 90% → 27.2, v2n) means a new dataset, or trimming old
  episodes to match (`dataset.build(trim_narrowing=True)`). Two policies compare one change only if they
  start from the same hover (v3 vs v2n: record v3 at `--hover-gripper 27.2`).
- **Idle frames teach waiting.** A policy has no clock; still frames all look the same, so it learns to
  wait at the hover forever (v1). Training datasets drop idle frames (`dataset.build`); new
  recordings should avoid pauses before the descent.
- **Lighting is part of the policy's input.** v2 was recorded with the lamp behind the robot on and
  froze on the first row in a dim evening room without it — but went 9/9 in daylight with the lamp
  off (2026-10-23). What matters is enough light on the mat, not the lamp itself. Every session
  records daylight / lamp / ceiling lights as structured `conditions`
  (`config.SETUP_QUESTIONS`); compare across sessions with `session_report.py --all`.
- **v2 repeats a memorised x/y move instead of steering to the tile** (measured at 4D on 2026-10-23:
  the grasp followed a deliberately offset hover at ~1 mm per mm). The demos' hover→grasp move is
  systematic per position (affine/23 misses ~12mm RMS in a pattern), and v2 learned it per position. So
  for v2: don't change the live hover (`data/hover_model.json`), and don't fit bench samples (grasp =
  hover + fixed move). Success rate hides this, since the jaws tolerate ±6mm; check a policy with
  `bench.py --hover-perturb-mm`. Compare bench and demo shifts per position, never against a mat-wide median.
- **Wrist roll runs out at the left edge.** The hover turns the roll to keep the jaws square to the mat:
  ~−147° at (13, +19) cm, 23° from the −170° command limit. A tile turned clockwise more than ~20° there
  is only reachable by rolling ~+150° the other way, jaws flipped and the wrist image upside down from
  every other demo (golden set T1, 2026-10-26). Expect any policy to fail it; don't place training tiles so.
- **Computed heights drift with reach: far spots are ~10-15mm lower than the FK says.** Same settled hover
  (tip 40mm above the grasp point by FK): a ruler showed 10-15mm above the tile top at far spots, 20mm+
  near (2026-10-28). The demos agree: their lowest FK tip rises from +10mm (r 0.15-0.18m) to +25mm
  (r 0.24-0.30m) while the jaws grip the tile at the same depth. The settle and the tip floor both use
  FK, so neither sees it; likely link/gear flex under load or a tilted table plane. At (0.19, +0.19) both
  v3b and v3c then went 13mm deeper than the demos there and bumped a neighbour once.
- **The wrist camera can drop out mid-session** (2026-10-28: from the 3rd of 10 v3c trials every read
  failed, and the policy ran blind to the 16s cap, so 8 misses were scored that said nothing about v3c).
  No frames for `camera.LOST_S` (1s) means it's lost, in every script: `policy.run` stops (`stop_reason`
  "camera") and the bench records `attempt_void` and runs it again; record.py discards the episode
  ("wrist camera lost"); wrist_capture.py takes the view again. Each reopens it with
  `hardware.reopen_wrist_camera` (retry / quit if it won't open): use that in any new script.
- The hover correction, the return correction and the table plane are tied to `assets/T_cam_to_base.npy`;
  recalibrating the RealSense invalidates both corrections (records are set aside automatically; returns
  then aim at home + the grasp's offset until new returns are logged) and may move the plane.

## Conventions

- Before calling a change hardware-ready: fast tests pass (slow too if `policy.py`/`dataset.py`
  changed), and say what was checked offline vs what still needs the arm.
- New recordings go to a new dataset name (`record.py --dataset`, required); never append to one recorded
  from a different hover (record.py refuses different features). One eval set holds all its slices and
  tile layouts: each episode's log row records its slice, layout spot and hover offset.
- Dataset → policy: `scripts/make_policy.py` (settle its options with `--dry-run` first; training takes
  hours). Uploads are public under `Yewy/` unless `--private`.
- **Eval episodes are never trained on**: held-out episodes and eval-only recordings in
  `data/eval_split.json` (`make_policy` leaves them out; `--slice` other than `train` marks eval recordings).
  Folding an eval set into training retires it: record the next one. Don't change an existing split
  (scores stop comparing). Offline scores screen; the bench decides what ships.
- **The tile detector's exam is never trained on either**: whole capture sessions listed in
  `data/detector_exam.json`, labelled in a Roboflow project of their own (`mine_frames.py` and
  `train_detector.py` keep them out). New weights replace `assets/tile_detector.pt` only if they beat it
  on the exam (`detector_eval.py --against`).
- Learned something about the hardware? Add it to the invariants above, with the reason.
- Retrospectives / session write-ups: **What changed / What was learned / What's next**, in
  `docs/retrospectives/<date>_<topic>.md` with a timeline and the session folders they draw on.
- **`README.md` is the user's command reference** (quick "I want to..." table + one section per script
  with example commands and options). Before merging into `main`, update it for any change to what a
  person runs: a new or removed script under `scripts/` or `python -m mahjong.*` entry point, an added/
  renamed/removed option, a changed default, new menu commands or keys (bench, record, layout), or new
  policy ids worth naming. Check every flag it mentions still exists (`--help`). Internal-only changes
  don't need it.
- Commits: one logical change each, message says why; push to `origin` (private GitHub).

## Worktrees and merging

Several worktrees run in parallel. On 2026-10-23 seven merged in one evening, and every bug that cost
time merged *without a conflict*: two `--hover-jitter-mm` options (record.py wouldn't start), a log row
built after its offset was cleared, and two copies each of an episode-log reader and the gripper
constants. Git catches overlapping lines, not overlapping intent. So:

- **Before starting parallel work**: a helper, constant or option that more than one task will need
  lands on `main` first (a small commit), and the branches build on it. Two branches shouldn't both
  change `bench.py`, `record.py` or `config.py`: do those tasks one after the other, or merge the first
  before the second starts on the shared file.
- **In a branch**: touch CLAUDE.md, ROADMAP.md and README.md only for that branch's own items; the merge
  reconciles them (they took 14, 11 and 3 commits from separate branches that day and drifted). Run
  hardware from the main checkout (sessions are per-checkout).
- **Handing a branch back**: merge or rebase the latest `main` into it, rerun the tests, push, and open a
  draft PR (`gh pr create --draft`) whose body says what changed and why, which files it touched, what
  it overlaps with on `main` or other open branches, and what's unverified on the arm. The merging
  session reads that instead of guessing intent from a diff.
- **Merging**: one branch at a time into `main`, each followed by the fast tests (the
  `pre-merge-commit` hook runs them; a conflicted merge's `git commit` runs `pre-commit`). Never
  `--no-verify` a merge. Before merging, compare what the branch adds with what `main` gained since
  the branch point (`git diff main...BRANCH` vs `git log BRANCH..main`): a new function, constant or
  option with the same purpose on both sides is a duplicate to resolve, conflict or not.
- **After a batch of merges**: one review pass over the merged range (duplicates, docs vs `--help`)
  before the next hardware session.
- **Tests that catch a clean-but-broken merge** are end-to-end sim runs of a script's main path
  (`tests/test_bench.py`, `tests/test_record.py`). A new user-run script gets one.
- **Closing**: once merged, `git worktree remove <path>` and `git branch -d <branch>` (and the
  `origin/` branch). `data/hover_corrections.jsonl` and `data/return_corrections.jsonl`
  merge by union (`.gitattributes`): both sides' samples are kept.

## Open items

The sequence to v3 (golden eval set, jittered dataset), the scorecard, autonomous play and DAgger are
in `ROADMAP.md`.

- Gripper stored as a delta ties every policy to its hover opening; consider absolute gripper values for
  the next dataset (needs `policy.run` + `dataset.build` to treat the gripper separately).
- Second row and far corners untested with v2; far corners (e.g. 7B) are out of reach after correction.
- RealSense recalibration at table height across the workspace would shrink the correction needed.
- Recording/deploying at servo I=8 would remove most sag; needs a new dataset and corrections.
- Autonomous play runs in `--sim` only so far (`scripts/play.py`): `hand13`, the drop zone and its marker are untaught; ROADMAP 8 has the steps on the arm.
