# mahjong-lekiwi

SO101 arm on a LeKiwi base picking up mahjong tiles: RealSense detects a tile → closed-loop IK hover
over it (measured correction, settled from above) → an ACT policy grasps from the wrist camera.
Rebuilt 2026-10-23 from `~/Projects/MahjongBot` (archive; don't import from it).

## Running

- Python: `~/miniforge3/envs/lerobot/bin/python`, from the repo root (scripts add it to `sys.path`).
- Fast tests (no hardware): `python -m pytest -m "not slow" -n 8` (~11 s with pytest-xdist; ~45 s without
  `-n`). Everything: `python -m pytest` (slow = loads policies from the HF cache, needs the GPU). A pre-commit
  hook runs the fast tests; turn it on once per clone with `git config core.hooksPath scripts/hooks`.
- **Where things stand**: `python scripts/status.py` (default policy and why, bench totals per policy, eval
  sets, layouts, the drop zone's bench check, this checkout vs origin/main). It reads `data/`; trust it over
  any doc. The default changes only with `status.py --set-default ID --evidence ...`.
- No-hardware run of the whole bench: `python scripts/bench.py --sim --trials 3 --tile 8D` (without
  `--tile` it asks which tile). Paired comparison across the mat: `--policy a,b --layout NAME --trials N`
  (`--trials` counts rounds: N at every spot, each policy once per round, alternating a b / b a from the
  same hover offset, so N × 2 × spots pickups; each attempt saves `wrist.mp4`).
- DAgger (on the default policy): `record.py --dataset Yewy/<new> --policy pickup_mat_cl_v6`. The policy drives, and a press of A
  (the rightmost face button, `gamepad.TAKEOVER_KEY`) hands you the arm. Training uses only the frames from the takeover on, as deltas from
  the hover: `make_policy.py --source <main> --add <dagger>`.
- Training recordings: `record.py --hover-jitter-mm 6`, so the policy finds the tile in the image (v2
  learned a fixed move per position). Eval recordings have no random jitter; the golden set's `jitter`
  slice uses a fixed list (`--hover-offsets-mm 6x,-6y`) so it can be recorded again identically.
- Hardware sessions write `sessions/<time>_<kind>/`; read one with `python scripts/session_report.py
  [folder]` (setup, success per tile/policy, stalls, clearances, ACT seconds, interventions, how far
  returns landed from home, contact sheet) before asking what happened; `--all` compares every bench session with its setup conditions.
  Run hardware scripts from the main checkout: `sessions/` is per-checkout and gitignored, so a
  worktree's sessions go with it.
- Interventions (`mahjong/interventions.py` has the classes and why, ROADMAP 3): keys pressed while the arm
  picks up or returns say what happened as it happens (`b` bumped another tile, `p` pushed its own, `l`
  look; `interventions.KEYS`, printed as each pickup and return starts), → marks a moment such as a hand
  on the tile (asked right after, unless a key said it). The RealSense scans around each attempt catch knocked or replaced
  tiles: a neighbour that moved or turned is a knock if `b` was pressed, else asked then; every other change
  (far tiles, gone / appeared) is settled by the next scans (`interventions.ChangeWatch`: undone = YOLO
  flicker, still there 3 scans on = real), not asked. No questions wait for the end of a batch. "Touched the tile to help" is a rescue
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
  error (`interventions.hand_grade`). Silent wrong tiles are a gate, not a target: one stops play. Bench the
  hand with `bench.py --layout hand14` (hand13 + the drop zone's spot): play won't start until the zone is.
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
| `mahjong/handoff_trace.py`, `scripts/handoff_report.py` | what the arm does once the gamepad has it (collect_corrections' `handoff/` traces): sag, drift, command moves |
| `mahjong/hover_correction.py` | measured hover correction (`data/hover_corrections.jsonl`; live sources in `data/hover_model.json`) |
| `mahjong/return_correction.py`, `scripts/return_samples.py` | where to set a held tile down so it lands at its home (`data/return_corrections.jsonl`, from each return's after-return scan) |
| `mahjong/dataset.py`, `quality.py` | trims + re-anchoring + resize; episode quality |
| `mahjong/session.py`, `sim.py` | session folders; simulated arm/camera/perception/policy |
| `mahjong/setup_check.py`, `scripts/setup_check.py` | is the workspace where it was: measurements vs a reference (`data/setup_reference/`) |
| `mahjong/interventions.py` | what a person had to do: classes, questions, derived from session events |
| `mahjong/wrist_check.py` | is a tile the scans lost still there: a face between the jaw tips at the hover (the bench looks before asking) |
| `scripts/play.py`, `mahjong/game.py`, `identity.py`, `markers.py` | autonomous play (`docs/autonomy_plan.md`): the turn loop on the bench's pieces; shanten and the discard choice; name votes per hand slot; the ArUco drop zone (`data/drop_zone.json`) |
| `scripts/` | `record`, `collect_corrections`, `hover_samples`, `bench`, `session_report`, `quality_report`, `make_policy` |
| `scripts/status.py`, `scripts/merge_pr.py`, `mahjong/checkout.py` | where things stand (from `data/`); approve + merge a PR and close its branch; the checkout vs origin/main, merged branches and worktrees |
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
  lets go. The policy's own descent opens to its hover gripper (90% for v1/v2, 27.2 for v2n, 26.7-26.8 for
  v3-v4), which only a new dataset or trimmed episodes change.
- **A return lets go where that tile's jaws closed, not at a fixed height.** Grasps close with the measured tip
  6-8mm above the grasp point near the base and 12-14mm at full reach (flex), v4 5mm lower than v3c, and once
  at -0.2 to 2mm (7D/4D, 2026-10-29). The fixed 10mm release let go of v4's tiles a median 3.3mm above where
  they were held (6-9mm that day): they dropped, tilted, and landed further from home (3.4mm under a 3mm drop,
  6.2mm at 6mm+; 137 returns). `Arm.place` now descends (only down, `_set_down`) to 1mm above the grasp's
  own height (`kinematics.grasp_clearance`, from the pickup's measured joints). Unconfirmed on the arm.
- **Heights are above the grasp point** (`config.grasp_z`, table + 5mm); the tile top is ~15mm higher.
- **Rest is where the unpowered arm settles** (`config.REST_POSE`, from `sessions/parked_pose.json`). The old
  folded rest sagged ~8° onto the stops at every disconnect, and each connect then lifted the arm via
  `RAISED_REST` and folded it back before doing anything. Re-measure it if the arm or its stops change.
- **The right stick moves height or wrist roll, never both** (`gamepad.EXCLUSIVE_AXES`): a pull down leaked
  into roll past XLeRobot's 25° snap and turned the wrist mid-descent (2026-10-27, 7B).
- **Returns aim at the tile's home, never where the jaws closed.** Returning to each grasp point carried
  every return's error into the next (tiles walked 77-100mm, 2026-10-27). The jaws pull a tile only partway
  to their centre, so the return model (`mahjong/return_correction.py`) takes the grasp as input; bench and
  record.py share the homes and landing check (`return_correction.Homes`, `check_landing`). A tile moved on
  purpose less than `HOME_RADIUS_M` (40mm) is put back at its old home. The fit pools every policy.
  Evidence: `docs/invariant_notes.md`.
- **Moves between rest and the table go via `RAISED_REST`** so the gripper clears the LeKiwi base.
- **Every move off the table rises straight up first** (`Arm.lift`, called by `rest`, `hover` and `place`, so
  bench, record and every other script share it): 5mm steps keeping the jaws' tilt and roll
  (`kinematics.straight_up`), up to 80mm. The old lift was one joint-space move to the IK's pose, which curved
  through the tiles, and it was skipped when the wrist had been tilted more than 30° from that pose: a discarded
  teleop then swung to rest from the mat.
- **A person stops closing once the jaws stall on a tile** (commanded 5-12, reading 11-13: 20 of 35 episodes of
  `mahjong_surfaces_mat_v1`, 2026-10-30). `holds_tile` can't tell that from empty jaws, so the tile wasn't
  returned. `Arm.squeeze` shuts jaws that read under 16 before asking (record after every episode, bench after
  the policy). Teleop episodes like that also have no `grasp_frame` (no command under 5), so their return aims
  from the hover and lets go at the default 10mm.
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
- **So is the surface under the tiles.** Every recording through v5 is on the black table top. On a green cutting mat
  with a white grid (2026-10-29, `20261029_180643_bench`) v5 closed its jaws 45mm up in the air once and bumped
  a neighbour the next time. `make_policy.py --augment --replace-mat 0.7` trains on colour jitter and on frames
  whose mat is swapped for other surfaces (`dataset.surface_keep` keeps faces, green backs and the jaws; faces in
  the dim recordings are found by growing from their lit part). The swap is fixed in the image, while a real mat
  moves with the camera, so record a few real episodes on each new surface as well, and score on ones never
  trained on. Each session records the surface (`config.SETUP_QUESTIONS`, earlier ones are the black top) and
  `offline_eval.py --by surface` scores per surface. Run `setup_check.py` on a new surface first: on the
  green mat (~2mm thick) the table plane read 4.1mm higher and tilted 4.4mm per 10cm in x
  (`20261029_191321_setup`), more than its thickness, so check it lies flat.
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
- **The far row is at the arm's full reach, not past what the servos can hold.** Held 10s at the 1D hover
  (base r 0.287m) the shoulder and elbow loaded 8% and 13% (overload trips at 80% for 2s), the tip
  steady (2026-10-28). But the wrist is 3mm from the gamepad's reach limit at 1D's grasp point, and
  past it at WW (0.291m). Pushing out there does nothing, and lowering slides the gripper back toward
  the base along the reach circle, which looks like a sag. `gamepad.warn_if_at_reach` says so after
  the hover (record, collect_corrections). Reach such tiles by tilting the jaws forward (L/R), or keep
  tiles inside base r ~0.27m. `layout.py`'s `r` shows that boundary on the mat (`layout.reach_band`): on
  hand14 1D and WW are at the edge, and both v4 and v5 missed WW with a neighbour beside it (2026-10-29).
- **Hand corrections record the commanded tool point, not where the jaws were — and that's fine.** A
  settled hover sags the same way the lined-up arm does, so the commanded fit already carries it (a refit on
  the measured point moves the hover only 0-4mm); the noise is the lining up (~9mm RMS, same tile twice).
  Every gamepad handoff uses a 0.35 deadzone (`gamepad.DEADZONE`) and warns of a stick off centre
  (`gamepad.warn_off_centre`): a resting thumb past XLeRobot's 0.15 pulled the arm 87mm. Teleop is traced
  (`mahjong/handoff_trace.py`, `scripts/handoff_report.py`). Evidence: `docs/invariant_notes.md`.
- **The wrist camera can drop out mid-session** (2026-10-28: 8 blind misses scored against v3c). No frames
  for `camera.LOST_S` (1s) means it's lost in every script: `policy.run` stops (`stop_reason` "camera"), the
  bench voids and reruns the attempt, record.py discards the episode. Reopen with
  `hardware.reopen_wrist_camera` in any new script. Evidence: `docs/invariant_notes.md`. It can also fall off
  USB altogether (`/dev/wristcam` gone, 2026-10-30 `20261030_115706_bench`, 3 dropouts in 5 min first): only a
  replug brings it back, and the bench waits at "Fix that, then Enter".
- **The wrist look sees a tile face, not which tile.** Between the jaw tips at a policy's hover, the hovered tile's
  face fills a fixed disc of the wrist image (`wrist_check.EXPECTED_PX`; fill >= 0.73 on 103 black-top frames, 0
  without). A tile knocked to within 30mm of a lost one's spot fills it too (3D for 1B, 2026-10-30), so the bench
  doesn't look there (`wrist_check.CROWDED_M`). Not measured on other surfaces: brown paper is pale too.
- **The drop zone is benched before play.** Play reaches the drawn tile's spot only mid-hand, at the far edge
  (WW, see the far-row invariant). `hand14` is `hand13` + that spot (S13); `play.py` refuses to start until a
  bench session has picked up and returned a tile within 25mm of the taught zone (`markers.zone_test`).
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
  policy ids worth naming. `tests/test_docs.py` fails on an option no script defines, a script with no
  section, a registered policy it doesn't name, or a default it gets wrong. Internal-only changes don't need it.
- **State lives in `data/`, not in the docs.** Don't write "the default is X" or "not yet tried" where a file
  decides it; point at `status.py`. ROADMAP.md's status table is the one place for what's done and next, each
  row with its evidence (a session folder or commit); update it in the branch that changes the state.
- **A ship decision is recorded where it's made**: `status.py --set-default ID --evidence <sessions and scores>
  --why "..."`, a row in ROADMAP.md's status, and a retrospective if the run taught something.
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
- **Handing a branch back** (Claude's job; the person approves the merge): merge the latest `main` into it,
  rerun the tests, push, and open a draft PR (`gh pr create --draft`) whose body has these sections, which
  `merge_pr.py` checks for: `## What changed` (and why), `## Files`, `## Overlaps` (with `main` or other open
  branches), `## Unverified on the arm`. Don't merge it yourself unless the person asks.
- **Merging** (the person): `python scripts/merge_pr.py N` from the main checkout, one PR at a time. It shows
  the description, what `main` and the branch both changed since they split (and any function, constant or
  option added on both sides: a duplicate to resolve, conflict or not), runs the tests on `main` + the branch
  in a scratch worktree, asks, merges, pulls `main` here and tests it. Don't merge on GitHub's page: it skips
  every test (29 of the first 40 merges did), and the main checkout isn't pulled. Never `--no-verify` a merge.
- **Closing** (part of `merge_pr.py N`): the branch's worktree `sessions/` are moved into the main checkout's,
  then the worktree, local branch and GitHub branch are removed. `python scripts/merge_pr.py` lists anything
  merged but not closed; `--close` closes it (35 branches had piled up by 2026-10-29). A Claude session can't
  delete worktrees or branches on its own: it says to run this.
- **After a batch of merges**: one review pass over the merged range (duplicates) before the next hardware
  session. `tests/test_docs.py` checks the docs against `data/` and the scripts' options on every commit.
- **Tests that catch a clean-but-broken merge** are end-to-end sim runs of a script's main path
  (`tests/test_bench.py`, `tests/test_record.py`). A new user-run script gets one.
- `data/hover_corrections.jsonl`, `data/return_corrections.jsonl` and `data/wrist_review.jsonl` merge by union
  (`.gitattributes`): both sides' samples are kept.

## Open items

What's done and next (policies, eval sets, the scorecard, autonomous play, DAgger, the detector) is in
`ROADMAP.md`'s status table.

- Gripper stored as a delta ties every policy to its hover opening; consider absolute gripper values for
  the next dataset (needs `policy.run` + `dataset.build` to treat the gripper separately).
- Far corners (e.g. 7B) are out of reach after correction.
- RealSense recalibration at table height across the workspace would shrink the correction needed.
- Recording/deploying at servo I=8 would remove most sag; needs a new dataset and corrections.
- Autonomous play runs in `--sim` only so far (`scripts/play.py`): the drop zone and its marker are untaught;
  ROADMAP 8 has the steps on the arm.
