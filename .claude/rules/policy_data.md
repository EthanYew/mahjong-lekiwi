---
paths:
  - "mahjong/policy.py"
  - "mahjong/dataset.py"
  - "mahjong/quality.py"
  - "mahjong/offline_eval.py"
  - "scripts/record.py"
  - "scripts/make_policy.py"
  - "mahjong/recipe.py"
  - "scripts/quality_report.py"
  - "scripts/offline_eval.py"
  - "scripts/spot_novelty.py"
  - "data/policies.json"
  - "data/eval_split.json"
  - "mahjong/takeover.py"
  - "scripts/train_remote.py"
---
Recording, datasets, training and scoring a policy. Evidence in `docs/invariant_notes.md`.

- DAgger (on the default policy; wall picks add `--wall-station F`): `record.py --dataset Yewy/<new> --policy pickup_mat_cl_v11`. The policy drives, and a press of A
  (the rightmost face button, `gamepad.TAKEOVER_KEY`) hands you the arm. Training uses only the frames from the takeover on, as deltas from
  the hover: `make_policy.py --source <main> --add <dagger>`. The same from wherever a pickup fails in `bench.py`, `base_cam_collect.py`,
  `wall_test.py` or `crude_run.py`: `--dagger-dataset Yewy/<new>` (`mahjong/takeover.py`; A again saves, Esc throws away; a takeover is a
  rescue, so the attempt isn't clean).
- Training recordings: `record.py --hover-jitter-mm 6`, so the policy finds the tile in the image (v2
  learned a fixed move per position). Eval recordings have no random jitter; the golden set's `jitter`
  slice uses a fixed list (`--hover-offsets-mm 6x,-6y`) so it can be recorded again identically.
- A pickup stops itself once the tile is held and lifted for 1.5s (`policy.py`), capped at 16s.
- Score a policy offline (GPU, minutes): `python scripts/offline_eval.py --policy pickup_mat_cl --against
  pickup_mat_cl_v1 [--by spot|lighting]`; tables per slice and the gates.
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
- **Closing high is what a miss looks like, and only the arm shows it.** Tip under 10mm above the grasp point as the
  jaws close: 127/130 picked; 20mm or more: 3/25 (295 real attempts, `grasp_clearance_m`; `session_report.py` prints it).
  Teacher-forced scores don't see it: v6 and v7 tied on arm, grasp and timing error over 53 demos they never trained on,
  while the bench had them at 6/14 vs 11/14. Evidence: `docs/retrospectives/2026-11-03_eval_validity.md`.
- New recordings go to a new dataset name (`record.py --dataset`, required); never append to one recorded
  from a different hover (record.py refuses different features). One eval set holds all its slices and
  tile layouts: each episode's log row records its slice, layout spot and hover offset.
- Dataset → policy: `scripts/make_policy.py` (settle its options with `--dry-run` first; training takes
  hours). Uploads are public under `Yewy/` unless `--private`. **A new version is the default's datasets and options plus the
  new recording**: v12 (2026-11-17) was started from a one-dataset example, trained on 15k frames where v11 had 88k, and was
  thrown away. `mahjong/recipe.py` (used by `make_policy.py` and `train_remote.py`) stops a run that leaves out one of the default's datasets (`--leave-out REPO` to mean it),
  and the dry run prints the default's frames and steps beside the new run's.
- **Eval episodes are never trained on**: held-out episodes and eval-only recordings in
  `data/eval_split.json` (`make_policy` leaves them out; `--slice` other than `train` marks eval recordings).
  Folding an eval set into training retires it: record the next one. Don't change an existing split
  (scores stop comparing). Offline scores screen; the bench decides what ships.
- **A lone tile closes higher than one in a full layout, and S8 is where it crosses the line** (partly confirmed, 2026-11-05). What seems to matter is white
  face-up tiles near the target: about 8mm with them, 13-18mm with none or only green backs, and turning the lights on doesn't close the gap (a pattern across 4
  sessions, not a deliberate test: `docs/retrospectives/2026-11-09_face_down_progress.md`).
  v8 at S8, from the same hover pose and 41mm hover as when it picked 6/6 at 5-9mm among a full hand, closed 21-28mm up on a lone
  tile and missed 7/7, face down and face up; with neighbours around it, 14-16mm and 1 of 2 completed picked. A hover 4mm lower
  closed only 2-3mm lower, so the hover isn't the lever. Bench and collect near-row spots with their neighbours in place, and
  record demos on isolated tiles. Evidence: `docs/retrospectives/2026-11-05_arm_collects_base_camera_photos.md`.
