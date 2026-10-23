# Roadmap

Bigger steps after the 2026-10-23 bench improvements (early stop, preflight, setup questions).
Roughly in order: the evaluation pieces first, because they decide whether everything after them
helped.

**Sequence from here (decided 2026-10-23):**
1. Trimmed-v2 baseline `pickup_mat_cl_v2n` (7): v2's episodes with the gripper narrowing trimmed,
   the 15 eval episodes left out, hover gripper ~23 %. Scored offline against v2, then benched paired.
   It isolates the hover change, so v3 vs v2n measures the jitter alone. v2 stays registered.
2. Hover jitter in `record.py` (7): done.
3. Record the golden eval set `Yewy/mahjong_mat_eval_v1` (2), including the new `jitter` slice. Do this
   before any jittered training data exists, so it can never leak into training.
4. Check the eval on v2: it should do clearly worse on `jitter` (grasp point off by ~the offset). If
   not, the slice can't detect memorisation.
5. Record the jittered training set, train v3, score it offline, bench it with the perturbation test
   (expect slope ~0) and paired against v2n.
6. DAgger on v3 (9): needs a takeover feature first.
7. Only then the hover A/B (1, Next 3).

## 1. Every bench run improves the hover

Each attempt logs the detection, the hover target (`hover_xy`) and where ACT closed the gripper
(`grasp_xy`). On 2026-10-23 ACT moved +6 to +10 mm in x and +1 to +11 mm in y from the hover before
grasping. That looked like a drifted hover against the demos' *overall* median (0 / +3.5 mm), but
compared **position by position** it isn't: the demos were recorded from the same correction
(`affine/23`) and moved the same way at the same places (near 4D: demos +11/+11, bench +9/+9).
The affine correction misses by ~12 mm RMS, in a pattern that depends on where the tile is on the mat,
and the operator corrected for it in every demo, so ACT learned to as well.

**Would a better hover confuse v2?** The policy only learns where it started from the wrist image
(its state input is relative to the hover). If it steers by the tile in the image, a better hover just
means a smaller move. But in the demos, the correction goes with mat position, so it may have learned
"here, move +13 mm" from the background instead. Then it would overshoot exactly where the hover improves most.
Trimming can't fix that (the offset is in every frame, unlike the gripper narrowing in 7); the cure is
a new dataset recorded with deliberate hover jitter.

Done (2026-10-23):
- Correction samples carry a `source` (gamepad / demo / bench); `data/hover_model.json` says which the
  live hover is fitted from (gamepad only). The check behind a fit leaves out a whole position at a
  time, so repeated attempts at one tile can't vouch for each other.
- `scripts/hover_samples.py` imported the 104 v2 demos and the 9 bench grasps; the bench saves a sample
  per success itself. `python -m mahjong.hover_correction --sources gamepad,demo,bench` shows the
  candidate, its change from the live hover (up to ~9 mm at the corners), and what the live hover misses
  per position and source. No drift between demo and bench so far.
- The hover settle logs sag (measured − commanded tool point) at each step.
- Caveat: where the operator didn't steer (x ≥ 0.21 in the demos: exactly 0/0), a demo sample only says
  "close enough", so it pulls a refit toward the current hover. Weight demos below bench.

**Sensitivity test result (2026-10-23 12:35, `20261023_123549_bench`): v2 repeats a fixed move.**
At 4D it moved ~+10/+10 mm from the hover whatever the hover offset; the grasp followed the hover at
1.26 (x) / 0.88 (y) mm per mm. It went 15/15 only because the jaws tolerate ±6 mm. So for v2 the live
hover stays as it is, and bench samples are circular (grasp = hover + fixed move): don't fit them.
Write-up: `docs/retrospectives/2026-10-23_hover_learning.md`.

Next:
1. The same test at an x ≥ 0.21 tile, returning the tile to the grasp point minus the offset, so it
   doesn't drift into its neighbours.
2. A jittered dataset (7; `record.py --hover-jitter-mm` built 2026-10-23). Rerun this test on the
   policy trained from it; expect a slope near 0.
3. Only then a **paired A/B** of a better hover vs the live one (`--correction a,b`, the same
   alternation as `--policy a,b` in 5), judged by the hover→grasp shift in mm as well as success,
   and **switching** by editing `data/hover_model.json` in one commit that cites the A/B session.
4. Bench attempts across the mat (5): most bench data so far is one position (4D).

## 2. An offline evaluation set

**Built 2026-10-23** (`mahjong/offline_eval.py`, `scripts/offline_eval.py`, `mahjong/layout.py`,
`scripts/layout.py`). What's left needs the arm: record `Yewy/mahjong_mat_eval_v1` (below). Write-up:
`docs/retrospectives/2026-10-23_offline_eval.md`.

- **Split.** `data/eval_split.json` holds out 15 of v2's 104 episodes, one per spot with ≥ 3 episodes
  (seeded). `make_policy` leaves them out of training. v1 and v2 trained on all 104, so scores on them
  check the plumbing, not true quality. **Only a fresh recording can compare v2 with v3 fairly.**
- **Calibration** on those 15 episodes:

  | | freezes | descent lag | close timing | arm error | grasp point |
  |---|---|---|---|---|---|
  | v2 (9/9 on the arm) | 0 | +1 step | 0.06 s | 0.21° | 0.9 mm |
  | v1 (froze 3/5) | 4 | +21 steps | 0.37 s | 0.22° | 1.4 mm |

  Teacher-forced arm error, today's "replay error", can't tell them apart: fed the demo's state, a
  policy that waits is only the command-to-position lag off. The gates are therefore freezes, a mean descent lag ≤ 10
  steps, and close timing ≤ 0.25 s, plus "not much worse than the reference" on arm error and grasp point.
- **The eval recording** (eval-only, ~36 episodes, 90 % hover so trimming the narrowing also scores a
  23 %-hover policy). Spend episodes on spots, not repeats: one offline score already averages ~200 frames.

  | `--slice` | spots | conditions | episodes |
  |---|---|---|---|
  | familiar | 4 spots v2 trained on (row 1 and row 2) | bright, square | 4 × 2 |
  | new | 6 new spots: between row-1 spots, row 2 left/right, reachable edges | bright, square | 6 × 2 |
  | dim | the 4 familiar spots | evening, lamp off, ceiling off (the freeze condition) | 4 × 2 |
  | tilt | 2 familiar spots, ±15° and ±30° | bright | 2 × 4 (reported, not gated) |
  | jitter | the 6 new spots | bright, square, hover moved `--hover-offsets-mm 6x,-6y` | 6 × 2 |

  Why a jitter slice: offline scores compare predictions with the demo. From a clean hover, v2's
  memorised per-position move *is* the demo's move, so it scores well there. Only a demo that had to
  steer from an offset hover can tell "steers to the tile" from "repeats a move". The offsets are a
  fixed list, not random, so the slice can be recorded again identically.

  Steps:
  1. ~~Measure a tile's top face~~: 35 × 26 mm, confirmed 2026-10-23; teach and view checked on hardware.
  2. Teach the layouts: `layout.py --name eval_familiar --teach`, then `--add T1 --like S2 --yaw 15` for the
     tilted spots.
  3. Record: `record.py --dataset Yewy/mahjong_mat_eval_v1 --slice familiar --layout eval_familiar`;
     the jitter slice: `--slice jitter --layout eval_new --hover-offsets-mm 6x,-6y`.
  4. Register it: `offline_eval.py --add-eval-set Yewy/mahjong_mat_eval_v1`.
  5. Check the dim slice predicts the hardware freeze: v2 should start descending late there.

The original plan:

Contact sheets show what happened. They can't score a new policy. What can is the
**(wrist image, state) → action** data in the recorded datasets.

- **Held-out episodes.** Choose a fixed ~10 % of episodes, spread across positions on the mat. List
  them in `data/eval_split.json` and have `make_policy` leave them out of training. Today v1 and v2
  are scored on episodes they trained on, so their 0.21-0.25° replay error is optimistic.
- **Metrics per held-out episode:** action error (the arm joints and the gripper, separately), error
  in when the gripper closes, the plan step where the descent starts (the freeze probe), and the
  predicted grasp point vs the demo's. All broken down by position.
- **Bench attempts as regression cases.** Save the full wrist video per attempt, not one frame every
  100 steps. A new policy can then be replayed on the same frames, open loop. There's no ground
  truth: these are the old policy's own actions. Still, a big difference from the old policy on an
  attempt that *succeeded* is worth flagging.
- **Limits.** Offline action error correlates only loosely with real success. A policy that drifts
  2 mm early sees different images from then on (compounding error). Use the offline numbers as a
  gate that catches freezes and gross regressions. Use the bench to decide what ships.

## 3. Logged interventions

Each time a person steps in, it should be an event, not a memory. **Done (2026-10-23):**
`mahjong/interventions.py`, which bench, record and collect_corrections log into.
`session_report.py` lists them.

Only one class measures the policy, so each intervention has a class:

- **rescue**: unplanned; something went wrong and a person fixed it. *The scorecard number:
  rescues per 10 attempts* (and attempts per rescue).
- **reset**: part of the protocol (tiles put back after a discard, the mat rearranged). It measures
  how far autonomous play (8) is, not the policy.
- **override**: a person's verdict against the gripper's (the success prompt). That's the success
  check's error.
- **failure**: a knocked neighbour. Not a step-in, but someone has to fix it.
- **observation**: listed, not counted.

A person is asked only when something happened. The rest is derived from what's logged anyway.

| Intervention | Class | Captured by |
|---|---|---|
| Esc abort (+ why) | rescue | `stop_reason`; one question afterwards |
| **→ during ACT** (a mark: the policy carries on) | from the answer: touched the tile / held the arm or base / caught a drop / other → rescue; "just look" → observation; Enter → `unspecified` rescue | `marks_s`; asked after the attempt, or at the end of a batch of trials |
| Neighbour knocked | failure | `detections` scans before/after each pickup and return, matched by position |
| Tiles put back / moved by hand | reset (straightened, if it was a knocked tile: rescue) | scans while idle at the menu |
| Discard missed the container | rescue | "in the container?" after a discard |
| Tile not found / unreachable | rescue (cancelled: observation) | `attempt_skipped` |
| Ctrl-C mid-command, crash, killed | rescue | `intervention`, `session_end.reason`, no `session_end` |
| Hover rejected in `record.py` (←) | rescue (of the hover) | `intervention` |
| Success overridden | override | `gripper_says_held` ≠ `success` |
| Arm not at rest at connect | observation | `connect` |
| Anything else | as chosen | bench `note` command |

A success with a rescue-class mark is **assisted**, reported apart from clean successes. Teaching
the discard pose is setup, not an intervention. A lamp switched on mid-session is the bench `setup`
command.

The 2026-10-23 log, read again with this: the 7B pickup (far corner) knocked 8D. Its detection
moved 6.4 mm, and it sits rotated in `overview_after.jpg`. Nobody noticed. Untouched tiles move
1-1.5 mm scan to scan, so the knock threshold is 5 mm. Tiles were also put back by hand after every
discard; that session had no after-scans, so it can't show them.

Not yet:
- A pre-ACT gamepad nudge. It would be a correction sample (1) *and* make the attempt
  non-autonomous, so it would need its own field and scorecard column.
- A contact signal from `act_trace` (measured vs commanded beyond the sag band: a hand on the arm).
  It can be derived offline from saved traces.
- Return accuracy from the after-return scan.

Still to check on the arm: one session that does each of these on purpose, and `KNOCK_M` against
that session's scan jitter.

## 4. A scorecard for ship or revert

A fixed protocol, so numbers compare across sessions and policies.

- **Standard eval:** for example 4 positions (first-row left, centre, right; second-row centre) × 5
  attempts, with the same lighting conditions recorded each time. ~10 min with early stop.
- **Scorecard per policy:** success by position, mean ACT seconds, stalls, slips, tip-floor steps,
  interventions per 10 attempts, the offline gates (2), and the hover→grasp shift (1).
- **Decision rule (draft):**
  - *ship* if the offline gates pass, no position gets worse by more than noise, and there's no new
    failure type;
  - *revert* if success falls on any position or interventions rise.
- **Be honest about sample size** (exact binomial, 95 %). 9/9 → true rate 66-100 %; 19/20 → 75-100 %;
  48/50 → 86-100 %. Against a fixed bar, 20 attempts of a 95 % policy clear "> 70 %" only 74 % of the
  time (50 attempts: always). Comparing two policies is harder: at 20 attempts each, a real 95 % vs
  70 % difference is detected 45 % of the time (40 each: 82 %), and 95 % vs 90 % is out of reach
  (2 %; 41 % even at 200 each). So use the bench to catch big regressions and failure modes, not
  small gains; compare policies **paired**, alternating them on the same positions in one session
  (5), so lighting and battery are the same for both.
- `session_report.py --all` is the start. It would grow into a per-policy table over all sessions.

## 5. Bench features those need

- `--policy a,b`: alternate two policies on the same tile, for paired comparisons.
- `--positions`: walk a list of tiles or positions (the standard eval in 4), with a success map over
  the mat.
- Full wrist video per attempt (for 2).
- Auto-return / re-detect when a trial's tile isn't where it was.

## 6. Housekeeping

- A pre-commit hook that runs the 5 s fast tests, so a motion change that breaks an invariant can't
  be committed.
- A Claude Code skill / slash command: "review the latest session". It runs `session_report`, reads
  the contact sheet and compares the run with `--all`.

## 7. The next dataset

- Hover with the jaws at ~23 % instead of 90 % (skips ~3.5 s of narrowing per episode). The hover is
  per policy, so this is a data change. Because the narrowing is a time prefix, trimming the v2
  episodes (`trim_narrowing`) and retraining also works without new recordings. A/B it against v2
  at the current hover before changing the hover too.
  **In progress 2026-10-23:** `pickup_mat_cl_v2n` (`make_policy.py --source
  Yewy/mahjong_mat_pickup_lekiwi_cl_hover --version v2n --trim-narrowing`): 89 episodes, the 15 eval
  episodes left out, so unlike v2 its offline scores are honest. It is the baseline for v3. If it does
  worse than v2, record v3 at 90 % instead.
- Record with deliberate random hover jitter (±~6 mm), so the policy has to find the tile in
  the image rather than in the mat position. Hover improvements (1) are then safe without a retrain.
  **Built 2026-10-23:** `record.py --hover-jitter-mm R` (uniform over the disc, `--jitter-zero-fraction`
  0.2 kept clean, `--jitter-seed`); an episode keeps its offset through ← and discards. Each log row
  has `hover_jitter_xy`. Jittered demos stay valid hover-correction samples (grasp − detection).
  Open: how many of v2's clean episodes to mix in; they carry the memorised per-position move.
- More episodes at the edges and on the second row. No pauses mid-episode.
- Consider an absolute gripper value instead of a delta (see the CLAUDE.md conventions).
- Leave the eval split (2) out from the start.

## 8. Autonomous play

Detect → pick → place / discard as a loop on top of the bench's pieces (hover, `policy.run`,
place, discard). Do this after 4, so it builds on a policy the scorecard has shipped.

## 9. DAgger: corrections on the policy's own mistakes

Comes after v3 (sequence step 6), not before 8: DAgger fixes the situations the *deployed* policy gets
itself into, so doing it on v2 would teach fixes for a policy that's being replaced. Today → only marks
a moment and ACT carries on; DAgger needs a **takeover**: the operator takes the gamepad mid-pickup, and
the frames from then on are recorded with the operator's actions as labels. Constraints:

- Seed the gamepad from `Arm.commanded()`, never the measured pose (the takeover drop).
- Hold the policy's gripper command until the operator changes it.
- Store deltas from the attempt's hover (its first frame), not the takeover frame, or the labels mean
  something different from every other episode. The pre-takeover frames are the policy's own and
  shouldn't be trained on as demos (keep them, flagged).
- Log each takeover as a rescue (3), so the scorecard still counts it.
- DAgger data never goes into the eval set; the golden set (2) judges v4 against v3.
