# How the pickup policies evolved

Written 2026-11-11, while v10 trains. It draws on `data/policies.json` (each policy's sources, frames and steps), the datasets'
`episode_log.jsonl`, `python scripts/status.py` (bench totals), ROADMAP.md and the retrospectives named below. Live numbers belong to
`status.py`. This page is the story of how they got there.

## How a policy is made (the same for every version)

- **Every version is trained from scratch.** `scripts/make_policy.py` runs `lerobot-train --policy.type=act` with no `--policy.path`,
  so no version starts from an earlier one's weights. The only pretrained part is ACT's image encoder (ResNet18 with ImageNet weights,
  lerobot's default). Nothing has been fine-tuned so far.
- **The data accumulates instead.** A new version is usually the last one's datasets plus a new recording, all trained together.
  Twice the recipe changed and older data was left out (v2n and v8, below).
- **DAgger sets count only their corrections.** In a DAgger recording (DAgger: the policy drives and a person takes over with the
  gamepad when it goes wrong), only the frames from the takeover on are trained on. An episode the policy finished alone isn't
  trained on at all. Demos (a person drives the whole pickup) count in full.
- **Some episodes are held back.** Held-out episodes (`data/eval_split.json`) and the eval-only recordings are never trained on.
  Dropped episodes (`data/dropped_episodes.json`) are left out after review.
- **Benchmarks are separate from training.** Each version is scored two ways after training. Offline, its checkpoints are scored
  on episodes they never saw (`offline_eval.py`). Since 2026-11-03 that's only a crash check, because the scores didn't rank working
  policies (`docs/retrospectives/2026-11-03_eval_validity.md`). On the arm, `bench.py` runs real pickups, and that's what
  decides which policy becomes the default.

A policy only works from the hover it was recorded from (its starting pose over the tile, gripper opening included; CLAUDE.md,
"Datasets store joint deltas"). That's why the hover opening appears in the table and why data recorded at one opening can't
always be mixed with another.

## The versions

Bench figures are totals from `status.py` (clean = picked up, nothing bumped, no one helped). They mix layouts, surfaces and how
closely bumps were watched, so they show each version's history, not a ranking. Only paired sessions compare two policies.

| Version | Trained | New in it | Training data (all from scratch) | Hover opening | Bench, clean | Outcome |
|---|---|---|---|---|---|---|
| v1 | 2026-10-22 | First policy | 104 gamepad demos from the closed-loop hover, rows 1-2, black table top, untrimmed | 90% | never benched by `bench.py` | Froze at the hover on 3 of 5 tries: still frames taught it to wait |
| v2 (`pickup_mat_cl`) | 2026-10-22 | Idle frames trimmed | v1's 104 episodes, idle and sag trimmed | 90% | 29/29 | Worked in daylight; froze in a dim room without the lamp. Repeated a memorised move instead of steering to the tile |
| v2n | 2026-10-27 | Baseline at a narrower opening | v2's episodes, gripper narrowing trimmed, 15 held out | 27.2% | 7/10 | The baseline for v3 (paired 10/10 vs 9/10) |
| v3 | 2026-10-27 | Hover jitter: demos start up to ~12mm off the tile, so the policy must look for it | 88 jittered demos, black top (`mahjong_mat_pickup_jitter_v3`) | 26.8% | 23/28 | Steered partly, not fully (ROADMAP 1) |
| v3b | 2026-10-28 | Bad demos removed | v3's demos minus 33 that showed the bench's failures (55 left) | 26.7% | 29/32 | |
| v3c | 2026-10-28 | More demos | v3b's 55 + 38 evening demos (95) | 26.8% | 96/125 | Default from 10-28 |
| v4 | 2026-10-29 | **First DAgger round** | v3c's demos + 52 corrections of v3c, mostly at the far spots 1D, 5D, WD (`mahjong_mat_dagger_v3c`) | 26.7% | 58/76 | Default |
| v5 | 2026-10-29 | Second DAgger round | v4's data + 30 corrections of v4 at 4D and 7D (`mahjong_mat_dagger_v4`) | 26.6% | 41/102 | Default 10-29; closed its jaws 45mm in the air on a green mat: it had only seen the black top |
| v6 | 2026-10-30 | **Other surfaces** | v5's data + 38 green cutting mat demos, with colour jitter and the black top swapped for other surfaces in 70% of episodes in training (`--augment --replace-mat 0.7`) | 26.7% | 108/159 | Default 10-30; knocked neighbours on tight spacing; failed badly in a real hand |
| v7 | 2026-11-03 | More surfaces | v6's data with the green set replaced by `mahjong_surfaces_mat_v1` (90 demos: green mat, brown and white paper) | 26.7% | 22/42 | Default 11-04; 29/30 clean over 2 hands |
| v8 | 2026-11-03 | **New recipe: the standard mat, narrower hover** | 37 demos on the standard mat at all 14 spots of `even-hand14` + 71 other-surface demos + 93 black-top jittered demos. **No DAgger sets**: v3c's and v4's corrections were left out, because they narrow the jaws from a 26% opening and would close them on the tile from 21% | 20.6% | 169/186 | **The default since 11-05**: play-ready 42/42, one hand 14/14 |
| v9 | 2026-11-10 | **First face-down data** | v8's data + 80 corrections of v8 drawing face-down tiles from a wall at station F, top and bottom of a stack (`mahjong_wall_dagger_v8`, 16 with the green-backed hand in view) | 20.6% | 38/43 | Drew 8/8 from the wall in crude runs (v8 0/7); 2/8 picking off the drop zone; missed face up at S12-S14 |
| v10 | 2026-11-12 | **Face-down hand spots + station L** | v9's data + 80 corrections of v9 (`mahjong_facedown_dagger_v9`, below); 83,214 frames kept, 114,000 steps (resumed from step 16,285 after the PC shut down) | 20.6% | not benched | Registered, not the default. Offline crash check passed: no freezes, held-out arm error 0.64° (v9 0.64°, v8 0.68°) |

Evidence by row: v1/v2 `docs/retrospectives/2026-10-23_hover_learning.md`; v2n-v3c ROADMAP "v2n / v3 / v3b / v3c" and commits
`a93ba25`, `3ef3311`, `31e8201`, `8776426`; v4-v6 ROADMAP "DAgger (9)" and `docs/retrospectives/2026-10-30_v4_to_v6_surfaces.md`; v7
`docs/retrospectives/2026-11-04_first_hands.md`; v8 `docs/retrospectives/2026-11-03_standard_mat_data_review.md`; v9
`docs/retrospectives/2026-11-11_crude_run_zone.md` and `docs/face_down_plan.md`.

## What each dataset captured

| Dataset | Recorded | Kind | Scenario | Used by |
|---|---|---|---|---|
| `mahjong_mat_pickup_lekiwi_cl_hover` | 10-22 | 104 demos | Face up, black table top, rows 1-2, lamp on, exact hover | v1, v2, v2n |
| `mahjong_mat_pickup_jitter_v3` | 10-27 | 132 demos (93 trained on now) | Face up, black top, single tiles across the mat, hover up to ~12mm off on purpose, dim and dark rooms | v3 to v10 |
| `mahjong_mat_dagger_v3c` | 10-28 | 52 DAgger on v3c | Face up, black top, far spots (1D, 5D, WD, WW) | v4 to v7 |
| `mahjong_mat_dagger_v4` | 10-29 | 32 DAgger on v4 | Face up, black top, 4D and 7D, bright daylight | v5 to v7 |
| `mahjong_green_cutting_mat_v1` | 10-29 | 38 demos | Face up, green cutting mat (also inside `surfaces_mat_v1`) | v6 |
| `mahjong_surfaces_mat_v1` | 10-29/30 | 90 demos (71 trained on) | Face up on green mat (38), brown paper (28), white paper (24) | v7 to v10 |
| `mahjong_surfaces_standard_mat_v1` | 11-03 | 56 demos (37 trained on) | Face up on the standard mat, a full 14-tile hand (`even-hand14`), every spot, bright | v8 to v10 |
| `mahjong_wall_dagger_v8` | 11-09/10 | 81 DAgger on v8 (80 trained on) | Face down, a wall at station F: top tile of a stack (41) and bottom tile (40) | v9, v10 |
| `mahjong_facedown_dagger_v9` | 11-11 | 84 DAgger on v9 (80 trained on) | Face down: the hand's own spots with every tile green side up, S11-S14 (30, including rotated tiles and recoveries from a failed grasp) and S1-S10 (30), plus top and bottom tiles at station L (24). Dim and dark rooms | v10 |

The eval recordings (`mahjong_mat_eval_v1`, `mahjong_mat_eval_surfaces_v1`) and held-out episodes are left out of all of these.

## Three shifts in the recipe

1. **v1 → v3c: learning to pick at all.** Trimming idle frames stopped the freezes, and hover jitter made the policy look for the
   tile rather than replay a move. It still only partly steers (ROADMAP 1).
2. **v4 → v7: corrections and surfaces.** DAgger fixed specific spots. Other surfaces were added once v5 showed a policy only knows
   the surface it was trained on.
3. **v8 onward: one standard setup, then face down.** The hover opening went to 20.6% to stop the open jaws knocking neighbours. That
   meant leaving out the DAgger sets recorded at 26%, and adding demos of a full hand on the standard mat. Face-down data started with
   v9 and grows with v10.

## v10's training data

v10 is not trained only on tonight's episodes. Like every version, it's trained from scratch on all its datasets together, and most of
that is face up. Raw frames counted per source (before idle trimming; DAgger sets from the takeover on):

| Source | Episodes | Share of frames |
|---|---|---|
| Face up, standard mat, full hand (demos) | 37 | 10% |
| Face up, green mat and paper (demos) | 71 | 20% |
| Face up, black top, jittered (demos) | 93 | 26% |
| Face down, wall at F (corrections of v8) | 80 | 19% |
| Face down, hand spots and station L (corrections of v9, 2026-11-11) | 80 | 24% |

So about 56% face up and 44% face down. The face-up data is why v8 and v9 pick well face up. Face-down hand spots (S1-S14 with green
backs) appear for the first time in v10. Before tonight the only face-down data was the wall at F, where the hand was in view in
only 16 episodes.

## Open questions this history raises

- **Fine-tuning hasn't been tried.** Starting v10 from v9's weights on only the new corrections would take less time on the GPU but
  could lose face-up skill. Every result above comes from full retraining.
- **Few face-down corrections per spot.** Trained on in v10: S12-S14 8 each, S11 5, S1-S10 3 each, except S3, S4 and S7 with 2
  (episodes 61, 63 and 72 left out). Whether that's enough is for the bench to say, at those spots, with the hand laid.
