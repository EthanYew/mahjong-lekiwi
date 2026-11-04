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
- Interventions (`mahjong/interventions.py` has the classes and why, ROADMAP 3): keys pressed in the bench's own
  terminal while the arm picks up or returns say what happened (`interventions.KEYS`, printed as each starts: `b`
  bumped another tile, `p` pushed its own, `r` moved by hand, `d` / `u` gripped low / high, `l` look; → marks a moment).
  Scans around each attempt catch the rest: a neighbour that moved is a knock if `b` was pressed, else asked then; other
  changes are settled by the next scans. No questions wait for the end of a batch. "Touched the tile" is a rescue; "the
  arm bumped a tile, no one helped" is a failure. The scorecard's numbers: `docs/scorecard.md`. Rules and cases:
  `docs/invariant_notes.md`.
- Unattended: `bench.py --unattended --layout NAME --trials N` asks nothing once the trials start. A neighbour the scans
  see move is a self-reported bump (settled by later scans; counts as a knock); a person checks them afterwards with
  `python scripts/bump_report.py`, whose verdicts (`scan_verdict` events) `interventions.derive` applies. README has what
  replaces each question. A new question in the bench's trials path needs an unattended answer: `tests/test_bump_report.py`
  fails any `input()`.
- Hardware scripts connect everything first (`mahjong/hardware.py`: retry / quit if something's
  unplugged), then ask the setup questions, then create the session folder.
- A command that may run an arm script prompts for a person, in auto mode too (`.claude/hooks/arm_run_gate.py`, a
  hook because no permission rule can exempt `--sim`); reads, `--sim` and `--help` don't. A new script that moves
  the arm goes in its `ARM_SCRIPTS`.
- A pickup stops itself once the tile is held and lifted for 1.5s (`policy.py`), capped at 16s.
- After packing up or moving anything: `python scripts/setup_check.py` (camera vs robot, table plane, a tile
  in the wrist view at the hover on 3 spots, parked pose, brightness) against `data/setup_reference/`, taken
  with `--save-reference` on a setup that worked. Nudge the camera back to green rather than recalibrating.
- Autonomous play: `python scripts/play.py --layout even-hand13` (`--sim` without hardware). One session is one
  hand; it asks instead of guessing a tile name, and the hand grades clean / attended / rescued / silent
  error (`interventions.hand_grade`). Silent wrong tiles are a gate, not a target: one stops play. Bench the
  hand with `bench.py --layout even-hand14` (even-hand13 + the drop zone's spot): play won't start until the zone is.
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
| `mahjong/bump_report.py`, `scripts/bump_report.py` | every pickup and return before / after from the scans, for a person to say which tiles really moved (`scan_verdict` events) |
| `mahjong/placements.py`, `name_ledger.py`, `scripts/name_report.py`, `docs/name_data.md` | which tiles get misread and how much it matters: a person's word on what's at each spot (bench, before trials; play's answers and audit), every read scored against it per placement, with reviewed wrist photos for older sessions |
| `mahjong/setup_check.py`, `scripts/setup_check.py` | is the workspace where it was: measurements vs a reference (`data/setup_reference/`) |
| `mahjong/interventions.py` | what a person had to do: classes, questions, derived from session events |
| `mahjong/scorecard.py`, `docs/scorecard.md` | clean attempts (picked, nothing bumped, no one helped); the iteration check, the play-ready check and the strict paired rule |
| `mahjong/wrist_check.py` | is a tile the scans lost still there: a face between the jaw tips at the hover (the bench looks before asking) |
| `mahjong/name_check.py`, `scripts/train_name_check.py`, `name_check_eval.py` | a second read of a tile's name from the wrist camera at the hover: in play, agreement changes nothing and disagreement asks; its exam is `data/name_check_exam.json` |
| `scripts/label_sheets.py` | one picture per tile name of every box labelled with it, to check a labelled set by eye |
| `mahjong/hand_view.py` | play's window: the mat zoomed on the tiles, every slot named, the tiles under review picked out with what they could be (`identity.Votes.worth_a_look`; it only marks, never asks), the wrist view of the tile asked about |
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
- **A return lets go where that tile's jaws closed, not at a fixed height.** Grasp height varies with reach and
  policy, and a release above it drops and tilts the tile, so it lands further from home. `Arm.place` descends (only
  down, `_set_down`) to 1mm above the grasp's own height (`kinematics.grasp_clearance`). Unconfirmed on the arm.
  Evidence: `docs/invariant_notes.md`.
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
  start from the same hover (v3 vs v2n: record v3 at `--hover-gripper 27.2`). `--hover-gripper` below a policy's own
  hover closes the jaws that much further on the way down: v6 and v7 at 21% commanded 12-15%, a tile reads ~11%, and 22 of
  28 attempts pushed the tile (2026-11-03, the likely cause, not proven). Mixing recordings needs their gripper deltas to
  agree; deltas start from the measured reading, which lags the command. `docs/retrospectives/2026-11-03_standard_mat_data_review.md`.
- **Idle frames teach waiting.** A policy has no clock; still frames all look the same, so it learns to
  wait at the hover forever (v1). Training datasets drop idle frames (`dataset.build`); new
  recordings should avoid pauses before the descent.
- **Lighting is part of the policy's input.** v2 was recorded with the lamp behind the robot on and
  froze on the first row in a dim evening room without it — but went 9/9 in daylight with the lamp
  off (2026-10-23). What matters is enough light on the mat, not the lamp itself. Every session
  records daylight / lamp / ceiling lights as structured `conditions`
  (`config.SETUP_QUESTIONS`); compare across sessions with `session_report.py --all`.
- **So is the surface under the tiles.** v5, trained only on the black top, closed its jaws 45mm in the air on a
  green grid mat. `make_policy.py --augment --replace-mat 0.7` helps, but the swap is fixed in the image, so also record a
  few real episodes on each new surface and score on ones never trained on (`offline_eval.py --by surface`). Each session
  records its surface (`config.SETUP_QUESTIONS`). Run `setup_check.py` on a new surface first, and check it lies flat.
  Evidence: `docs/invariant_notes.md`.
- **v2 repeats a memorised x/y move instead of steering to the tile.** Success rate hides this, since the jaws
  tolerate ±6mm; check a policy with `bench.py --hover-perturb-mm`. For v2: don't change the live hover
  (`data/hover_model.json`) or fit bench samples. Compare bench and demo shifts per position, never against a mat-wide
  median. Evidence: `docs/invariant_notes.md`.
- **Wrist roll runs out at the left edge.** The hover turns the roll to keep the jaws square to the mat:
  ~−147° at (13, +19) cm, 23° from the −170° command limit. A tile turned clockwise more than ~20° there
  is only reachable by rolling ~+150° the other way, jaws flipped and the wrist image upside down from
  every other demo (golden set T1, 2026-10-26). Expect any policy to fail it; don't place training tiles so.
- **Computed heights drift with reach: far spots are ~10-15mm lower than the FK says.** The settle and the tip floor
  both use FK, so neither sees it; likely flex under load or a tilted table plane. Evidence: `docs/invariant_notes.md`.
- **The far row is at the arm's full reach, not past what the servos can hold.** Pushing past the gamepad's reach
  limit does nothing, and lowering slides the gripper back along the reach circle, which looks like a sag.
  `gamepad.warn_if_at_reach` says so after the hover. Tilt the jaws forward (L/R), or keep tiles inside base r ~0.27m
  (`layout.py`'s `r` shows it). Evidence: `docs/invariant_notes.md`.
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
- **Closing high is what a miss looks like, and only the arm shows it.** Tip under 10mm above the grasp point as the
  jaws close: 127/130 picked; 20mm or more: 3/25 (295 real attempts, `grasp_clearance_m`; `session_report.py` prints it).
  Teacher-forced scores don't see it: v6 and v7 tied on arm, grasp and timing error over 53 demos they never trained on,
  while the bench had them at 6/14 vs 11/14. Evidence: `docs/retrospectives/2026-11-03_eval_validity.md`.
- **The wrist camera can drop out mid-session.** No frames for `camera.LOST_S` means it's lost in every script:
  `policy.run` stops (`stop_reason` "camera"), the bench voids and reruns, record.py discards the episode. Reopen with
  `hardware.reopen_wrist_camera` in any new script. If `/dev/wristcam` is gone, only a replug helps. Evidence:
  `docs/invariant_notes.md`.
- **The wrist look sees a tile face, not which tile.** A knocked tile near a lost one fills the same disc, so the
  bench doesn't look within `wrist_check.CROWDED_M`. A pale surface fills it with no tile at all, so without enough dark
  surface in view (`wrist_check.MIN_DARK_SHARE`) it can only say the view matches an earlier look at that place, and a
  person is asked the first time (`wrist_check.PlaceViews`). Evidence: `docs/invariant_notes.md`.
- **Whether a tile is there and which tile it is are separate questions.** The detector boxes tiles it can't name (3C at a
  median 0.18, so it looked gone in 45 of 46 scans), so a box down to `config.DETECTION_PRESENCE_CONFIDENCE` proves a tile
  is there (`perception.faint_tiles`, `tile_faint` events), and only a name over `DETECTION_CONFIDENCE` is acted on as one.
  A layout's labels are what taught it, not what's on the mat: never use them as a tile's name. Evidence:
  `docs/invariant_notes.md`.
- **A tile's white face can't be found by colour on a pale mat**: 1 good read in 950 on the striped mat, and slices of
  stripe passed as tiles. Where it matters, use what a tile looks like: the detector's box on the RealSense
  (`layout.find_tile`, which can't read yaw there, so green is position only) and the reference's picture of the tile in
  the wrist view (`setup_check.match_tile`, so a spot needs the same tile each time). Evidence: `docs/invariant_notes.md`.
- **A stricter rule for a problem tile has to go on the name read, not the tile.** The robot never sees "8C", only the "7C"
  it reads, so real 7Cs pay too (`identity.ALWAYS_ASK_READS`, a stopgap in play). Confidence alone isn't enough: wrong EW and
  7C reads overlap real ones. Evidence: `docs/name_data.md`.
- **A second read may only add questions until it passes its exam.** Agreement changes nothing and disagreement asks
  (`name_check`, `play.py`): the two together can still agree on a wrong name, and the gate is 0 of those towards ~1000 reads.
  One exception, at the owner's request: registering the hand, agreement on a read from `identity.ALWAYS_ASK_READS` moves its
  question into the hand check a person does on all 13 before the first draw. A drawn tile has no such check, so it's still asked.
- **The votes can be sure of a wrong name, and nothing asks.** A 5C read as 9C at 0.91 of the votes (S9, `20261104_164541_play`), and
  the only way to say so was to stop the hand. So a person checks the hand before the first draw (`Player.confirm_hand`) and can
  correct a tile at any draw prompt. Enter isn't graded; a correction is a `recount`, and counts as a wrong name in the ledger.
- **A read a person overruled is no news until the tile moves.** The detector's wrong reads are steady, not flicker: S9 read 9C in 201 of
  202 scans, the name a person gave outvoted within 2-3 scans, and 53 re-asks in one hand changed no answer (`20261104_183142_play`). So a
  person's word, the hand check's Enter too, holds against scans until the tile moves past `interventions.KNOCK_M` (`Player.word_holds`). The most
  a re-asked tile had moved was 2.9mm. It doesn't hold against the wrist camera at the hover. Unconfirmed on the arm.
- **Play's window has its own process** (`mahjong/hand_view.py`). The main thread sits in the policy loop and `input()`, and an OpenCV
  window nobody pumps gets marked "not responding". It draws on a zoom of the tiles: on the real layout a tile is ~30px of the frame, and
  labels on the whole frame overprint each other (only seen rendering a real frame; the tests pass either way).
  Questions during a pickup need the key listener stopped first (`keys.stop()`): it holds the terminal in cbreak and eats the answer.
- **The drop zone is benched before play.** Play reaches the drawn tile's spot only mid-hand, at the far end of
  the rows (WW). `even-hand14` is `even-hand13` + that spot (S13); `play.py` refuses to start until a
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
- **Counted bench results run on the standard setup** (`data/standard_setup.json`: one mat, one layout), with a passing
  `setup_check.py` and one hover opening within a comparison. Other surfaces are generalisation checks and never block.
  Which check answers which question: `docs/scorecard.md`. Don't change the setup to explain away a result.
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
  It refuses, before touching GitHub, if that pull would fail (`pull_blockers`): #65 merged on GitHub, then stopped at the pull
  because `main` here had a commit GitHub didn't, so nothing was pulled, tested or closed.
- **Data the scripts log** (samples, reviews) goes straight to `main` from the main checkout, committed and pushed in one step
  after a hardware session (README, "Data files and merging"); uncommitted until then is normal. A branch doesn't carry it. A decision
  (default policy, exam list, scores, weights) goes in a PR with its evidence. A Claude session doesn't push to `main`.
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
