# Autonomous play: review of the MahjongBot script, targets and guardrails, port plan

Written 2026-10-28 (AEDT), agreed the same day; the record of why `scripts/play.py` is built the way it is.
Sections 1-6 and 8 are the reasoning and the targets; section 7 the build order (what's built so far is
in ROADMAP 8). Decisions are listed under "Decided".

## Context

The goal is **autonomous play**: the robot plays a hand of mahjong, and a person's only job is to put
each newly drawn tile on the drop zone. Before trusting that, we want targets and guardrails showing
the setup is robust, accurate and consistent. Two requirements are fixed:
- **Silent wrong tiles are unacceptable.** The robot must never act on a tile name it got wrong
  without noticing.
- **A clean full hand** is the headline target (defined in section 3).

`~/Projects/MahjongBot/mahjong_autonomous_play.py` (1197 lines, Aug 2026) already has the turn
structure: an ArUco-marked drop zone, choosing the discard, and checkpoints. This note reviews it
against what `mahjong-lekiwi` has learned since. That includes the return correction merged today
(`adfbd03`, `bd4f6a7`): tiles go back to their home, aimed with a model that predicts where they
land. Then it sets the targets and a port plan. (CLAUDE.md says not to import from the archive, so
the script is a design to rebuild, not code to reuse.)

**Terms used below.** *Shanten*: how many tiles a hand is from ready (0 = one tile from winning,
called *tenpai*; −1 = won). *Ukeire*: how many kinds of tile would improve the hand. *Home / slot*: a
tile's assigned place on the mat. *RMS*: the typical size of an error. *Rule of three*: zero failures
in *n* tries means the true rate is below 3/*n* (95 % confidence).

---

## 1. The old script's one real game (2026-08-18)

`mahjong_play_sessions/20260818_154140/checkpoint.json`: **won in 14 turns, 42 minutes (~3 min a
turn).** 5 of the 14 discards were the drawn tile itself. That gives about **23 pickups and 23
set-downs per hand** (a hand-tile discard costs 2 pickups: the discard, then the drawn tile into its
slot). **8 of the 13 starting tiles needed a person to confirm their name** (7 were below the 0.80
confidence bar). On top of that, a person answered y/n after every grasp. So the loop worked, but a
person was involved in nearly every step.

## 2. Review of `mahjong_autonomous_play.py`

### Keep (good design, port it)
- **The phase machine** (REGISTER_HAND → AWAIT_DRAW → REGISTER_DRAW → SELECT_DISCARD →
  EXECUTE_DISCARD → PLACE_DRAWN), with an **atomic checkpoint after every change** and resume. It's
  the right shape.
- **The discard choice**: lowest shanten after the discard, ties broken by most ukeire, then by
  "discard the drawn tile". It's pure and testable, and the last tie-break also saves a manipulation.
  The **win check on all 14 tiles** before choosing a discard is correct.
- **Refusing to guess**: registration loops until exactly 13 non-bonus tiles are seen; a rejected
  tile clears state and re-scans; stdin is flushed before prompts.
- **`--dry-run --fixture`**: the whole decision path runs without hardware.
- **The drop-zone marker resolved once while uncovered and cached**, since a tile on it hides it.

### Bugs and gaps (fix in the port)
| # | What | Why it matters |
|---|---|---|
| 1 | **A person confirms every grasp** (`confirm_grasp_success`, always asked) | ~23 asks a hand: a clean hand is impossible. lekiwi has an automatic hold check (`policy.run` early stop, `Arm.holding`) plus before/after scans |
| 2 | **Discard marker not visible → `return False` while holding the tile**, and the phase retries `strategy.pick` with the tile still in the jaws | The arm tries to pick while holding a tile |
| 3 | **Unbounded retries**: IK failure, failed discard and failed placement all `continue` forever, re-aiming at the same stored position | No cap, no ask, and a tile knocked by the failed try is aimed at where it *was* |
| 4 | **Slot positions fixed at registration, never re-scanned** | lekiwi measured tiles walking 77-100 mm over 10 pick/return rounds. Stale positions are how a pick lands on the wrong tile, which is a **silent wrong tile** |
| 5 | **No conservation check or re-scan of the hand** per turn | A knocked, swapped or misplaced tile goes unnoticed: a silent error |
| 6 | **The grasp check is one detection frame within 20 mm** (`tile_still_present_near`) | YOLO flickers; lekiwi uses 5-frame scans and `interventions.scene_diff` |
| 7 | **Placement opens to `GRIPPER_OPEN`** at a fixed z, no settle, aims where told | Breaks three lekiwi invariants: open only just past the held reading (25 % turned a neighbour), arrive from above, aim with the return correction |
| 8 | **The wrist re-check runs the table-view YOLO on the wrist photo** | That model was trained on a 24 px tile seen from 0.8 m, so a close-up is a different input. Its answer isn't an independent second read |
| 9 | **What the arm holds isn't checkpointed** | A resume between pick and discard re-picks with a tile in the jaws (same as #2) |
| 10 | **Ukeire counts tile kinds, not copies left** | Ignores visible discards. A strategy weakness, not a safety one |
| 11 | Old stack: ikpy `move_to_tile`, `pickup_table` policy, 39-class ArUco-aware detector | lekiwi's detector has 38 classes and no ArUco class, so a marker beside a tile must be masked out geometrically |
| 12 | **Name clash**: lekiwi's package is called `mahjong`, so `from mahjong.shanten import Shanten` fails from the repo root (checked: library 2.0.0 is installed but shadowed) | The port needs its own shanten code or a renamed import |

### The drop zone (ArUco) in the lekiwi setup
- **Decided: marker under the tile, as before; pivot if it causes trouble.** Two things to watch,
  each a pivot trigger (then move the marker beside the zone, at a fixed offset): (1) first-try
  pickups at the zone fall below the ~95 % seen on the mat, since every policy was trained on tiles
  on the mat and the black-and-white pattern is new to the wrist camera; (2) the 38-class detector
  (no ArUco class) reads the marker's edge around the tile as a tile, or misnames the drawn tile. The
  bench can test both before play: `--layout` with a spot on the marker, paired against a mat spot.
- **Superseded 2026-11-04: the marker is read only to teach the zone**, and later hands carry the saved position over (`.claude/rules/play.md`). The original decision:
  **Re-detect the marker every turn once the zone is empty** (just after the drawn tile is picked).
  If it moved more than a few mm, the LeKiwi base or camera moved: pause. This catches the
  calibration drift that ties hover correction and table plane to `T_cam_to_base.npy`. Cache the
  position with the calibration fingerprint (`hover_correction.setup_fingerprint()`), not bare.
- **Site the zone where the policy is verified**: away from the left edge (wrist roll runs out) and
  the far spots (heights drift 10-15 mm low). Size it from the perturbation test: a person places
  tiles roughly, so the zone is how far off and how turned a tile can be and still be picked.
  Outside the zone → ask "move it into the zone" (an ask, not a failed pick).

### What lekiwi already has for each piece
| Old | lekiwi equivalent |
|---|---|
| `detect_tiles` + `merge_tile_detections` | `RealSense.detect_tiles` (5 frames) + `perception.merge_tiles` |
| `move_to_tile` hover, `pickup_table` ACT | `Arm.hover` (closed-loop, from above) + `policy.run` with the default policy (`pickup_mat_cl_v3c`), early stop, camera-dropout void |
| `place_on_table` | `Arm.place` + `ReturnCorrection.target(home, detection_xy, grasp_xy)` + the after-return landing check (`Bench.return_tile`, `check_landing`) |
| discard marker | taught discard pose (`bench.py teach-discard`, `Arm.go_to_pose`), `discard_missed` question |
| checkpoint only | `Session` folder: `events.jsonl`, scans, interventions, `session_report.py` |
| — | `interventions.scene_diff` / `MatWatch` (knocks), `hardware.py` preflight, `layout.py` taught spots |

---

## 3. "A clean full hand", spelled out

**Definition.** A clean hand runs from registering the 13 starting tiles to a win (or the wall running
out). In it, **a person does nothing but put each drawn tile in the drop zone**:
- no **rescues** (touching the arm or a tile, a tile put back, a discard fished out of the bin, Esc);
- no **asks** (name confirmations, "is it held?", "move it into the zone", a recount);
- no **resets** (`tile_reset` after a return landed > 25 mm from home);
- and at the end-of-hand **audit** (a person checks the robot's hand and discards against the table)
  **zero mismatches**.

Two weaker grades are reported next to it: **attended** (only asks, nobody touched anything) and
**rescued**.

**Silent wrong tile = 0 is separate from clean, and stricter.** A clean hand is a target, to be met
more and more often. A silent error is a **gate**: any one stops play, and the cause goes back down
the verification ladder (section 6) before more play. The design trades asks for safety whenever
they conflict: when unsure, ask. An ask costs cleanliness; a silent error costs correctness.

**What it takes, in numbers.** A hand has about 23 pickups, 23 set-downs and 27 name reads (13 at
registration, 14 drawn). If every one has to go right without help, the chance of a clean hand is
the product:

| each event succeeds without help | clean hand (~73 events) |
|---|---|
| 99 % | 48 % |
| 99.5 % | 69 % |
| 99.7 % | 80 % |
| 99.9 % | 93 % |

**Names are still the gap, but it has narrowed (update 2026-10-28 afternoon).** The old game
auto-accepted only 5 of 13 names (38 %). Since then:
- **Detector v6x3** (live then, `2fee0eb`; `ROADMAP.md` has what's live now): on the exam of that day, without flowers, it found 97.1 % of tiles and
  named all 99 it found correctly.
- **The `scan-voting` branch** (`ea055a9`, not merged yet): the 5 frames of a scan vote on each name.
  98.0 % right and 0 mix-ups; a split read is left out instead of misnamed.

That's the safe direction: an uncertain tile becomes an ask, not a silent error. Two limits remain:
- **The ask rate.** ~2-3 % of tiles missed per read × 27 reads ≈ 0.5-0.8 asks a hand. A clean hand
  needs ~0.5 %.
- **The exam is too small to rule out silent errors.** It's one capture session (15 frames, 102
  tiles, one lighting). 99 of 99 named right only shows the wrong-name rate is below ~3 % (rule of
  three). To show < 0.3 % takes ~1000 tile reads with no wrong name, across the lighting conditions.
  ROADMAP 10 step 1 planned one exam session per lighting; only one exists.

So the second read (next section) is still needed, and **growing the exam is now as important as
growing the training set**.

**Proposed headline target:** **≥ 8 of 10 consecutive hands clean, 0 silent errors**, across the
lighting conditions. 10 hands is ~730 events: with zero silent errors, the rule of three bounds the
silent-error rate below ~0.4 % per event (backed by the far larger offline exams in section 4).

### Face-down hands (decided 2026-11-20, the owner)

Face-down tiles are the format for play against people, so they get the same target: **≥ 8 of 10 consecutive hands clean, 0 silent
errors** (`scripts/play_face_down.py`). What differs from the face-up hand above:

- **Unattended hands count.** Wrong names can get in silently, so they are caught afterwards: every hold's photo gets a verdict
  (`base_cam_review.py --play`), `play_face_down.py --report` scores every read against it, and the end check compares the 13
  to the table. One wrong name found is a silent error: a gate, play stops. A hand not yet reviewed counts for nothing.
- **Clean** = no rescue (a step done by hand), no reset, no ask (a correction of a name is an ask; pressing Enter on a read isn't,
  as `interventions.hand_grade` grades it), end check all matched, review found no wrong read. `mahjong/hand_tally.py` counts
  this, and `status.py` prints it: hands before the first one still waiting for review count, an unfinished hand counts as failed. The wall feed (a person pushing the next stack up) is part of the
  format, not a rescue. Neighbours the arm knocks aren't seen in play (no scans), so only the end check can show them.
- **Steps per hand, face down** (my count from the turn plan, not measured): registration is 13 × (pickup, read, set-down) = 39;
  each turn adds a draw pickup, a read, a drop-zone set-down, a discard pickup and a discard set-down = 5. So N ≈ 39 + 5T
  for T turns (= tiles drawn). T is not fixed: the logged hands ran 15, 19, 34 and 43 turns (`20261119_142253_play_fd`,
  `20261104_231918_play`, `20261104_183142_play`, `20261117_200555_play_fd`), so the 14 draws above was an assumed typical hand,
  not a derived number.
- **The formula.** P(hand clean) ≈ p^N when each step succeeds without help with probability p. To get a hand clean 80 % of the
  time, p must be 0.8^(1/N): ≈ 99.80 % at T = 14 (N = 109), 99.87 % at T = 26 (169), 99.91 % at T = 43 (254). Face-down pickups
  on the default policy (v13) are ~94 % clean (79/84), which gives ≈ 24 % for 23 pickups: far short.
- **The test itself.** A system whose real clean rate is 80 % a hand passes 8 of 10 about 68 % of the time; at 90 %, 93 %; at 70 %,
  38 %. So a pass shows "good enough", a fail doesn't show "bad".
- **Evidence so far: 2 won hands, 0 of the 10.** `20261117_200555_play_fd` (43 turns, reads confirmed) and `20261119_142253_play_fd`
  (15 turns, unattended); both graded clean by `session_report.py`, which has no scans in play to see a knocked tile. Neither
  counts yet: the 10 are consecutive hands on one setup, reviewed.

## 4. No silent wrong tiles: the identity guardrails

A silent wrong tile can enter in three places. Each gets its own check:

**a. Reading a name wrong.** Accept a name only when **two independent reads agree**:
1. the RealSense vote (`merge_tiles` on the `scan-voting` branch: confidence summed per name across
   frames, ≥ 2 frames, `votes` kept). For play, add a bar on the **margin** between the top two names
   (the branch's split read: 8C 2.46 vs 9C 1.53), set from the exam;
2. an independent second read. For the drawn tile: the drop-zone crop classifier (section 8.3).
   For hand tiles: the accumulated vote from registration on (8.1-8.2), checked once by a person at
   registration if needed. A wrist-camera classifier (8.5) is a fallback only: the old wrist check
   did poorly.

Agreement → accept. Disagreement, or either read unsure → ask with the wrist photo. Measured
offline on exams it never trained on: **"both agree and both wrong" must be 0** on the detector exam
plus a wrist exam (thousands of reads, far more than play can test). The share of disagreements is
the ask rate, which the clean-hand numbers above need near 0.5 %.

**b. Picking the wrong physical tile** (stale position, a neighbour). Before ACT starts:
- re-scan and aim at the **current** detection nearest the slot, never the stored position
  (bug #4);
- the current detection at the slot must carry the expected name in its accumulated vote, and no
  other tile may be within the merge radius. Otherwise don't descend; ask. This is the last check
  before a discard, which can't be undone. If a wrist classifier passes its exam (8.5), its read at
  the hover is added here;
- after the pickup: the slot is empty in the scan and nothing else moved (`scene_diff`).

**c. The robot's picture of the hand drifting from the table.** Every turn:
- **conservation**: the scan's names at the 13 slots equal the expected hand (old hand + drawn −
  discard); no face more than 4 times counting visible discards;
- **after a placement**: the scan sees the right name at the slot (the landing check already finds
  the tile by name near its home), and the slot's stored position becomes where it actually
  landed.

Any mismatch → stop and ask for a recount. Never "fix" it automatically.

## 5. Placement: using today's return correction

The return model aims a set-down so the tile lands at a target. It takes the target, the tile's
detection and the grasp, so it **works for any target, not only the tile's own home**. That means a
drawn tile can be aimed straight at the vacated slot (bug #7 fixed by reuse).
- **Checked on the arm at one spot** (`20261028_102732_bench`, 6C, v3c): 15 rounds landed a median
  of 3 mm from home (worst 4 mm), with no build-up, vs 11 mm the day before. Other spots, the reach's
  edge and the shift toward the shoulder are unchecked. Hand slots and the drop zone need that check
  (a `--layout hand13` walk) before play.
- **Key homes by slot, not tile name.** The bench keys a home by layout spot *or* tile name
  (`Bench.home_for`). A hand holds pairs and triples of the same name, so a name key would merge
  two slots' homes. In play every home is a slot id, with the vacated slot reused for the drawn
  tile.
- The fit pools every policy; a policy that grips differently (DAgger, v4) may need its own
  (retrospective `2026-10-28_return_homes.md`). Compare its landings before trusting the fit.
- Keep the bench thresholds: warn past 12 mm (`RETURN_DRIFT_WARN_M`), ask for a reset past 25 mm
  (`RETURN_DRIFT_RESET_M`). In play a warning also updates the slot to where the tile landed.
- **Each play placement is a new return sample** (`return_corrections.jsonl`), so the model improves
  with play, and its new positions (hand slots, drop zone) get measured.
- **The hand layout is a real design question.** `Arm.place` assumes tiles sit ≥ 57 mm apart; 13 at
  that spacing is ~74 cm, more than the verified reach. Plan: teach a `layout.py` layout `hand13`
  (2 rows inside the verified area, plus the drop zone and the discard pose). Pick the spacing with a
  bench test: tight-row pickups and placements at e.g. 57 / 50 / 45 mm, counting knocks. Check that
  the training demos had neighbours that close; if not, that's a DAgger target.

## 6. Targets, guardrails, verification

**Targets** (per condition row: lighting × battery × reach; never pooled):
| Target | Bar | Source |
|---|---|---|
| Silent wrong tiles | **0, a gate** | audit, conservation check, exams |
| Clean hands | ≥ 8/10 consecutive | session events |
| Clean face-down hands (`play_face_down.py`, attended or unattended; see "Face-down hands" below) | ≥ 8/10 consecutive, 0 silent errors | session events + every hold reviewed + end check |
| Asks per hand | ≤ 1 on the way there | interventions |
| Names auto-accepted (two reads agree) | ≥ 99.5 % (today ~97-98 % per read); agree-and-wrong 0 over ~1000 exam reads across lightings | `detector_eval.py` + drop-zone / wrist exams |
| First-try pickup | ≥ 95 % per slot and in the zone | bench `--layout hand13` |
| Eventual pickup (≤ 2 automatic retries, re-scanned) | ≥ 99.7 % | play |
| Landing | ≤ 12 mm from target at 95 %; neighbours < `KNOCK_M` (5 mm) | landing check, `scene_diff` |
| Drop-zone tolerance | measured: offset and turn picked at ≥ 95 % | `bench.py --hover-perturb-mm` at the zone |
| Turn time | p90 (90 % of turns are faster) ≤ ~45 s (old: ~3 min) | timestamps |
| Endurance | targets hold from a full battery to low and after 1 h | soak with voltage and temperature logged |

**Guardrails during play** (pause and ask rather than guess):
- each turn: both cameras delivering, battery and servo temperature within limits, mat brightness in
  the tested range, calibration fingerprint unchanged (the marker check this list had is gone: section 2);
- each action: the identity checks of section 4; retries capped at 2 with a re-scan each time, then
  ask; never pick while holding (state checkpointed, checked with `Arm.holding()` at resume); IK or
  reach refusals ask, never loop.

**Verification ladder** (each rung before the next):
1. Offline / sim: the turn loop end to end in `--sim` with fixtures (a test like
   `tests/test_bench.py`); shanten/discard unit tests against the old script's choices; the two
   exams.
2. Bench components: `hand13` pickups and placements, drop-zone perturbation, tight-row spacing.
3. A scripted hand: a fixed wall order (the fixture's draws, set out by a person), so it can be run
   again identically.
4. Soak: 10 hands across the conditions → the headline numbers.
5. Live games, still logged and audited.

## 7. Port plan

1. **`mahjong/game.py`** (pure, no hardware): hand state, the shanten/ukeire discard choice,
   14-tile win check, conservation check, copies-left ukeire. Our own shanten code (bug #12), tested
   against the library's answers generated outside the repo.
2. **Shared pick / return / discard helpers out of `scripts/bench.py` into `mahjong/`** (pickup with
   scans, `home_for`, `return_tile` + `check_landing`, discard). Land this on `main` first as its own
   commit (CLAUDE.md: shared helpers before parallel branches).
3. **ArUco drop zone**: `mahjong/markers.py` (detect, cache with fingerprint, mask the marker square,
   offset to the zone).
4. **Identity**: merge `scan-voting` (it changes `merge_tiles`, which the bench, record and play all
   use; merge before branches build on it). Then: sticky per-tile votes across scans; a drop-zone
   capture mode modelled on `wrist_capture.py`; agree-and-wrong and ask rate in `detector_eval.py`;
   more exam sessions (section 8).

Open branches to sequence around: `scan-voting` (perception.py, detector.py) and
`wrist-capture-window` (wrist_capture.py) are unmerged; step 2's bench.py extraction should wait until
neither touches bench.py.
5. **`scripts/play.py`**: the phase machine on lekiwi's `Session`, `hardware.py` preflight, setup
   questions, events, and a checkpoint that includes what the arm holds; `--sim`; end-of-hand audit
   prompt. Add `session_report.py` lines for hands (clean / attended / rescued, asks, silent errors).
6. README section, ROADMAP 8 update, a sim test for `play.py`.

Files touched: new `mahjong/game.py`, `mahjong/markers.py`, `scripts/play.py`, tests; `scripts/bench.py`
(step 2 only), `scripts/session_report.py`, README, ROADMAP. Nothing in MahjongBot changes.

## Decided (2026-10-28)
- Drop-zone marker **under the tile** as before; pivot to beside it if the triggers in section 2 fire.
- **2-row hand** (`hand13`): 13 tiles don't fit comfortably in one row within the reach.
- **No calls** (pon/chi/kan), no flowers. The discard choice tracks **copies left** (4 − in hand −
  visible discards) next to ukeire: rank by shanten, then by the number of improving *tiles* left
  (copies), then by ukeire kinds, then by "discard the drawn tile".

## 8. Making tile detection robust (the biggest blocker)

More photos of the current workspace (ROADMAP 10 step 2) is the base. The ideas below go beyond
"more data", ordered by expected payoff for the effort. The first three use the game's structure to
cut how much reading has to be right.

1. **Read each tile once, then carry its name.** Names only change when a tile is touched. After
   registration, the hand's names come from the game state, not from re-reading. Each turn needs
   **one new read: the drawn tile**. Re-scans of the hand check positions and look for disagreement
   (a flag for a recount), never re-name a tile. That cuts the risky reads per hand from "every tile
   every scan" to 13 + 14.
2. **Sticky, accumulated votes per tile.** The `scan-voting` branch votes within one scan's 5
   frames and keeps each name's sum (`votes`). Extend that across scans: follow each tile by
   position (slot) over the session and add each scan's `votes` to a running total. Hundreds of
   frames per hand tile turn a split read into a clear majority, or a clear ask. Reset a tile's total
   only when `scene_diff` says it was touched. Merge `scan-voting` first; this builds on its format.
3. **Treat the drop zone as a reading station.** Every drawn tile is read at the same spot, so build
   for that spot: fixed light over it (a small diffuse LED), fixed RealSense exposure and white
   balance (auto exposure changes how a tile looks frame to frame), a crop at a known pixel box,
   and a small **crop classifier trained only on drop-zone crops**. Optional: a cheap USB camera a
   few cm above the zone, for many times the 24×18 px the RealSense gets now.
4. **Self-labelling capture (no Roboflow work).** When the name is known, labels are free.
   `wrist_capture.py` (merged, `e35eb17`) is already this pattern for the wrist camera: confirm the
   names once, then many views per tile, saved per name. Reuse it:
   - *drop zone*: the same flow with the tile on the marker, saving RealSense crops of the zone
     (and wrist views) per name. Prompt "place 5D", capture, step through all 34 faces (× copies,
     × lighting). The crop classifier's training set in an evening;
   - *robot-driven*: in a bench loop the arm picks a tile whose name a person confirmed and sets it
     down at new hand slots and angles (the return model aims it). Each RealSense scan is a labelled
     frame of the current scene, unattended, across lighting through the day;
   - *existing data*: record and bench logs already name the tile of every episode and attempt
     (`record.py` log rows, bench `attempt` events + `wrist.mp4`). Hundreds of labelled
     hover-frame wrist images, after a quick check that the logged names were right.
5. **Why the old wrist check did poorly, and whether to retry it.** From the code, it ran the
   *table* YOLO (trained on a 24 px tile from 0.8 m) on a close-up wrist photo, from 3 cm, on the old
   arm and camera: wrong model, likely too close to focus. That says little about a classifier trained
   on wrist images. `wrist_capture.py` now collects exactly that data (40/55/70 mm, jittered, rolled),
   so it's cheap to score on an exam before deciding. Names with no wrist views yet: 1-5C, 7-9C, the
   flowers, 1B, 5B. It stays a fallback **second read for asks**, not the main path, given your
   experience.
   **Split by capture session, never by photo**, for any new classifier (wrist or drop zone). 8 views
   of one tile are near-copies; spread over train and valid, they inflate the score.
   `split_by_scene.py` fixed exactly this for the detector (134 of v4's 182 valid frames had a train
   twin). Keep the exam a separate capture session too.
6. **Use the confusion list.** The exam's mix-ups (`detector_eval.py`) are mostly near-identical
   faces (2/3, 7/8 of a suit). Accept per pair: a read in a known-confused pair needs a higher margin
   between the top two classes, or a second read. Check the suit and the number separately (suit is
   easy from colour and pattern).
7. **More pixels per tile from the RealSense.** Colour at 1280×720 or 1920×1080 (ROADMAP 10 idea),
   or the camera lower and closer now that play uses a known area (`hand13` + zone). Decide **before**
   the big labelling round, since it changes the input. Moving the camera means recalibrating, which
   resets the hover and return corrections.
8. **Control the light** over the whole mat (diffuse, fixed). It removes the biggest variable for the
   detector *and* for ACT (the lighting invariant).
9. **Cheap at inference**: test-time augmentation (flip/scale) or two scales, voted. Only if the exam
   shows a gain.

10. **Grow the exam before trusting the silent-error bar.** One exam session per lighting (daylight,
    lamp only, ceiling only, dim), including tiles in the `hand13` rows and on the drop-zone marker:
    ~1000 tile reads in all. Note: it scores Roboflow's re-encoded frames, which undersell every
    detector (old weights: 69 % there vs 81 % on the robot's own frames). Comparisons hold, absolute
    numbers don't.
11. **What already worked, keep doing it** (docs/usage.md, v6x3): current-scene frames repeated 3× in
    training, splits by scene, one box per tile in pre-labels. The next round's frames should come
    from the play layout itself (`hand13` + the zone), since that's the scene play sees.

**Measure every one on the exam** (`data/detector_exam.json`, never trained on). Add two numbers to
`detector_eval.py`: **agree-and-wrong** (the silent-error rate; must be 0) and the **ask rate** (how
often reads disagree). A new detector ships only if both improve.
