# 2026-10-23: offline evaluation set and tile layout guide (ROADMAP 2)

Branch `worktree-offline-eval`, commits `48dc298`..`2b6dbaa`. Nothing here needed the arm. Recording
the eval set (What's next) does.

## Terms

- **Policy**: the trained ACT model (*Action Chunking with Transformers*). It looks at the
  wrist-camera image plus the joint angles and plans the next ~100 arm movements (a **chunk**).
- **Offline evaluation**: scoring a policy on recorded episodes on the computer, with the arm off. It asks:
  "shown this exact image and arm pose, would the policy have done what the human did?"
- **Held out**: episodes kept out of training, so a policy is scored on things it has never seen.
- **Slice**: a labelled group of eval episodes (familiar spots, new spots, dim light, tilted tiles),
  scored separately and never averaged together.
- **Teacher-forced**: at every frame the policy gets the *human's* real image and pose, not the
  pose its own earlier moves would have led to.
- **Open loop**: the policy's plan is compared with the demo, not executed.
- **Freeze**: the v1 failure where the policy waits at the hover and never descends.
- **Gate**: a pass/fail check a new policy must pass before it's worth bench time.
- **Yaw**: a tile's rotation on the mat, seen from above. 0° = square to the mat (long side along the arm's x).

## What changed

**Tile layout guide.** `mahjong/layout.py`, `scripts/layout.py`, `data/layouts.json`.
- A layout is a named list of target spots (x, y, yaw) in the arm's base frame.
- Each spot is drawn on the live RealSense image as a tile outline. Detection maps pixel → table; this
  maps table → pixel, using the same hand-eye transform and the table plane at the tile's top face.
- The tile actually at a spot is found by its white face on the dark mat (bright, low saturation).
  Its centre and yaw are measured on the table plane. The outline turns amber within 8 mm / 10°
  and green within 4 mm / 5°.
- `--teach` saves the tiles on the mat as spots, plus a "ghost" photo you can blend in (`g`). `--add ID
  --like SPOT --yaw 15` makes a tilted copy of a spot.
- `RealSense` now exposes its colour intrinsics. `SimPerception` renders its tiles, so the overlay works
  in `--sim`.

![Overlay on a real RealSense frame](img/2026-10-23_layout_overlay_real_frame.jpg)

*The finder on a real frame from that morning's bench session. Every tile on the mat is found. The
left-hand outlines are amber/red because those tiles really are turned 14-21°. The outlines in the
corners and off the mat are seeds my test script took from bright clutter; no tile was found there,
correctly.*

**Recording labels.** `record.py --slice {train,familiar,new,dim,tilt}`, `--layout NAME`, `--repeats N`,
`--start-spot ID`.
- Every episode record now stores its slice and the session's lighting answers. `Session` keeps
  `conditions` and `notes` on the object.
- With a layout, the live view shows where the next tile goes. The tile at that spot is picked
  without the menu, and the record stores the target spot plus where the tile was actually measured.

**Held-out split.** `data/eval_split.json`.
- 15 of v2's 104 episodes are held out: one, at random (seed 0), from each spot with ≥ 3 episodes.
- `dataset.build(episodes=...)` builds a chosen subset, and its summary maps built episodes back to
  source indices. `make_policy.py` leaves the held-out episodes out of training, and its dry run
  shows the count.

**Offline scoring.** `mahjong/offline_eval.py`, `scripts/offline_eval.py`, results in
`data/offline_eval/<policy>.json`. Per episode:

| Score | How | Teacher-forced? |
|---|---|---|
| arm error / gripper error | first planned action vs the human's, every 3rd frame | yes |
| descent onset, freeze | plan from the hover (where the gripper narrowing ends): steps until lift/elbow move 2° vs the human's | open loop |
| close timing | plan made 1 s before the human closed: when it closes vs when the human did | open loop |
| grasp point, roll error | that same plan: jaw position (mm) and wrist roll at its close vs the human's | open loop |

- Eval episodes are built like training data, with the narrowing trimmed for a 23 %-hover policy, so one
  90 %-hover recording scores both kinds.
- `make_policy.py` scores every checkpoint and runs the gates against the current default policy.

**Docs**: `CLAUDE.md` (layout rows, how to run the offline eval, "eval episodes are never trained on",
the open item to measure `TILE_FACE_M`) and ROADMAP 2.

## What was learned

- **v2's 104 episodes cover only ~15 spots.** Tiles went back within ~3 mm of the same place. All were
  recorded 11:00-13:00 on 2026-10-22, and lighting and yaw weren't recorded. 16 demos rolled the wrist
  > 10° (6 > 20°), so some tiles were a little crooked.
- **The old "replay error" can't see a freeze.** On the 15 held-out episodes:

  | | freezes | descent starts | closes off by | arm error | grasp point |
  |---|---|---|---|---|---|
  | v2 (9/9 on the arm) | 0 | +1 step | 0.06 s | 0.21° | 0.9 mm |
  | v1 (froze 3/5 on the arm) | **4** | **+21 steps** | **0.37 s** | 0.22° | 1.4 mm |

  Fed the human's pose at every frame, a policy that just waits is only off by the small lag between
  command and position. The same happened on a synthetic test with a frozen stand-in policy, and it's why
  close timing moved from teacher-forced to open loop. So the **gates** are: no freezes, mean
  descent lag ≤ 10 steps, close timing ≤ 0.25 s, plus arm error ≤ 1.25 × reference + 0.1° and grasp point ≤
  reference + 3 mm. `tests/test_offline_eval_slow.py` checks that v2 passes and v1 fails.
- **Caveat:** v1 and v2 trained on these 15 episodes, so the absolute numbers are optimistic. They show
  the scoring works, not how good the policies are. **No split of v2's data can ever compare v2 with v3
  fairly; only the fresh recording below can.**
- **RealSense geometry.** The camera is ~75 cm above the base, looking almost straight down, with image-up =
  base +x. With fx ≈ 590 px (fitted; the colour intrinsics weren't saved before), logged detections
  project within ~3 px of the tiles, at ~1.5 mm per pixel. A tile is only ~24 × 18 px, so measured size
  is ~1 px (1.5-3 mm) too big per side. Position and yaw stay within ~1.5 mm / ~1.5°.
- **Repeatability over accuracy.** The outlines land on the same patch of mat every session as long as
  the RealSense isn't moved, even if the calibration is a few mm off overall. Spots are in the base frame,
  so they survive a recalibration. The ghost photo doesn't.

## What's next

1. ~~**Measure a tile's top face**~~ — done: 35 × 26 mm is right (confirmed by hand, 2026-10-23); the camera
   reads ~38 × 29.5 mm including the visible side faces.
2. ~~**Try the live view on the hardware**~~ — done 2026-10-23: teaching and viewing a layout work.
   Still worth a glance during the first recording: the episode log's `placed_xy` against the YOLO
   `detection_xy` for tiles placed to green (expect within ~4 mm).
3. **Record `Yewy/mahjong_mat_eval_v1`** (~36 episodes, 90 % hover, never trained on). Spend episodes on
   spots, not repeats: one offline score already averages ~200 frames.

   | `--slice` | spots | conditions | episodes |
   |---|---|---|---|
   | familiar | 4 spots v2 trained on (row 1 and row 2) | bright daylight, square | 4 × 2 |
   | new | 6 new: between row-1 spots, row 2 left/right, reachable edges | bright, square | 6 × 2 |
   | dim | the 4 familiar spots | evening, lamp off, ceiling off | 4 × 2 |
   | tilt | 2 familiar spots at −30/−15/+15/+30° | bright | 2 × 4 (reported, not gated) |

   ```
   python scripts/layout.py --name eval_familiar --teach
   python scripts/layout.py --name eval_tilt --teach            # or --add T1 --like S2 --yaw 15 on a copy
   python scripts/record.py --dataset Yewy/mahjong_mat_eval_v1 --slice familiar --layout eval_familiar
   python scripts/offline_eval.py --add-eval-set Yewy/mahjong_mat_eval_v1
   python scripts/offline_eval.py --policy pickup_mat_cl --by lighting
   ```

4. **Check the dim slice predicts the hardware freeze**: v2 should start descending late there. If it
   doesn't, the offline score misses what failed on the arm, and the gates need rethinking.
5. **Re-check the gate thresholds** on the eval recording once it exists. They were set on episodes
   both policies trained on.
6. **Full wrist video per bench attempt** (ROADMAP 2 and 5), so new policies can be replayed on attempts
   the old one succeeded at.
7. **The next dataset (ROADMAP 7)**: record with `--slice train` so it carries lighting labels. Don't
   fold `eval_v1` into it. If you do, retire it and record `eval_v2`.

## Limits

Offline scores screen; the bench decides. A policy 2 mm off early sees different images from then on,
and errors snowball. Teacher-forced and short open-loop scores can't show that.
