# mahjong-lekiwi

SO101 arm on a LeKiwi base picking up mahjong tiles: RealSense detects a tile → closed-loop IK hover
over it (measured correction, settled from above) → an ACT policy grasps from the wrist camera.
Rebuilt 2026-10-23 from `~/Projects/MahjongBot` (archive; don't import from it).

## Running

- Python: `~/miniforge3/envs/lerobot/bin/python`, from the repo root (scripts add it to `sys.path`).
- Fast tests (no hardware, seconds): `python -m pytest -m "not slow"`. Everything: `python -m pytest`
  (slow = loads policies from the HF cache, needs the GPU). A pre-commit hook runs the fast tests
  (~13 s); turn it on once per clone with `git config core.hooksPath scripts/hooks`.
- No-hardware run of the whole bench: `python scripts/bench.py --sim --trials 3 --tile 8D` (without
  `--tile` it asks which tile). Paired comparison across the mat: `--policy a,b --layout NAME --trials N`
  (rounds alternate a b / b a from the same hover offset; each attempt saves `wrist.mp4`).
- Training recordings: `record.py --hover-jitter-mm 6`, so the policy finds the tile in the image (v2
  learned a fixed move per position). Eval recordings have no random jitter; the golden set's `jitter`
  slice uses a fixed list (`--hover-offsets-mm 6x,-6y`) so it can be recorded again identically.
- Hardware sessions write `sessions/<time>_<kind>/`; read one with `python scripts/session_report.py
  [folder]` (setup, success per tile/policy, stalls, clearances, ACT seconds, interventions, contact
  sheet) before asking what happened; `--all` compares every bench session with its setup conditions.
- Interventions (`mahjong/interventions.py`, ROADMAP 3): during a bench pickup, → marks the moment
  without stopping ACT (asked what happened afterwards; Esc still stops it). "Touched the tile to help"
  is a rescue (the success becomes assisted); "the arm bumped a tile, no one helped" is a counted
  failure that leaves a success clean. The bench's `note`
  logs anything else a person did, and `setup` logs a mid-session change (lamp, battery).
  Knocked tiles and tiles put back by hand come from the RealSense scans logged around each attempt;
  the bench shows each change and asks (Enter: as the scans say, or "a detection glitch": YOLO
  flickers, e.g. a 0.59 tile reading 0.68 and "appearing"; a tile turned in place shows as its
  detection box changing shape). The scorecard number is rescues per 10 attempts; resets and
  overrides are counted apart.
- Hardware scripts connect everything first (`mahjong/hardware.py`: retry / quit if something's
  unplugged), then ask the setup questions, then create the session folder.
- A pickup stops itself once the tile is held and lifted for 1.5s (`policy.py`), capped at 16s.
- Score a policy offline (GPU, minutes): `python scripts/offline_eval.py --policy pickup_mat_cl --against
  pickup_mat_cl_v1 [--by spot|lighting]`; tables per slice and the gates.

## Layout

| | |
|---|---|
| `mahjong/config.py` | the one setup: ports, cameras, table plane, rest poses, defaults |
| `mahjong/kinematics.py` | closed-form FK/IK; `pincer_tip`, `hover_command` |
| `mahjong/arm.py` | all motion (`hover`, `place`, `rest`, `go_to_pose`) |
| `mahjong/policy.py` | registry (`data/policies.json`), loading, the ACT control loop |
| `mahjong/perception.py`, `camera.py`, `gamepad.py` | RealSense tiles, wrist camera, XLeRobot gamepad |
| `mahjong/hover_correction.py` | measured hover correction (`data/hover_corrections.jsonl`; live sources in `data/hover_model.json`) |
| `mahjong/dataset.py`, `quality.py` | trims + re-anchoring + resize; episode quality |
| `mahjong/session.py`, `sim.py` | session folders; simulated arm/camera/perception/policy |
| `mahjong/interventions.py` | what a person had to do: classes, questions, derived from session events |
| `scripts/` | `record`, `collect_corrections`, `hover_samples`, `bench`, `session_report`, `quality_report`, `make_policy` |
| `mahjong/layout.py`, `scripts/layout.py` | tile layouts (`data/layouts.json`) drawn on the RealSense view; place tiles to green |
| `mahjong/offline_eval.py`, `scripts/offline_eval.py` | scores per slice on episodes never trained on (`data/eval_split.json`) |

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
  lets go. The policy's own descent opens to its hover gripper (90%), which only a new dataset changes.
- **Heights are above the grasp point** (`config.grasp_z`, table + 5mm); the tile top is ~15mm higher.
- **Moves between rest and the table go via `RAISED_REST`** so the gripper clears the LeKiwi base.
- **The policy reaches further than the IK will plan.** It closed on 4D 11-12mm beyond the hover,
  ~0.29m from the shoulder, outside the IK's envelope (15° tilt, 10° joint margins), so the return
  was refused. `Arm.place` moves a return up to 15mm toward the shoulder, checking every height down
  to the release (reachable at the approach alone would drop the tile from there).
- **Datasets store joint deltas from each episode's first frame** (gripper too). A policy only works from
  the hover its episodes started at — that hover is stored per policy in `data/policies.json` and the
  bench uses it. Changing the hover (e.g. gripper 90% → 23%) means a new dataset, or trimming old
  episodes to match (`dataset.build(trim_narrowing=True)`).
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
- The hover correction and the table plane are tied to `assets/T_cam_to_base.npy`; recalibrating the
  RealSense invalidates the correction (records are set aside automatically) and may move the plane.

## Conventions

- Before calling a change hardware-ready: fast tests pass (slow too if `policy.py`/`dataset.py`
  changed), and say what was checked offline vs what still needs the arm.
- New recordings go to a new dataset name (`record.py --dataset`, required); never append to one with a
  different layout or hover.
- Dataset → policy: `scripts/make_policy.py` (settle its options with `--dry-run` first; training takes
  hours). Uploads are public under `Yewy/` unless `--private`.
- **Eval episodes are never trained on**: held-out episodes and eval-only recordings in
  `data/eval_split.json` (`make_policy` leaves them out; `--slice` other than `train` marks eval recordings).
  Folding an eval set into training retires it: record the next one. Don't change an existing split
  (scores stop comparing). Offline scores screen; the bench decides what ships.
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

## Open items

Bigger plans (evaluation set, scorecard, interventions, hover learning from bench runs) are in
`ROADMAP.md`.

- Gripper stored as a delta ties every policy to its hover opening; consider absolute gripper values for
  the next dataset (needs `policy.run` + `dataset.build` to treat the gripper separately).
- Second row and far corners untested with v2; far corners (e.g. 7B) are out of reach after correction.
- RealSense recalibration at table height across the workspace would shrink the correction needed.
- Recording/deploying at servo I=8 would remove most sag; needs a new dataset and corrections.
- Autonomous play isn't ported yet.
