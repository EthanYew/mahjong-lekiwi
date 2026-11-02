# Roadmap

## Status

The one place for what's done and what's next: one row per item, each with its evidence. Update the row in
the branch that changes it. Live numbers (the default policy and why, bench totals per policy, layouts, the
drop zone) come from `python scripts/status.py`, which reads `data/`; `tests/test_docs.py` checks this table
names the current default. Sections below hold the reasoning.

| Item | State (2026-11-02) | Evidence | Next |
|---|---|---|---|
| Default policy | **`pickup_mat_cl_v6`**: v5's data + 33 green-mat demos, `--augment --replace-mat 0.7` (9) | `data/policies.json` default_note; `20261029_234437_bench` (green mat, close-hand14: 28/28, 0 bumped, 0 rescues; v5 there 2/3, 2 bumped, `20261030_010109_bench`); offline `data/offline_eval/pickup_mat_cl_v6.json` | Pair v6 and v7 on `even-hand14`, black top, and decide by `docs/scorecard.md`; perturbation test (1) |
| Steering vs memorised move (1) | **Not solved.** Jittered data cut it but not to ~0: v3 x +1.26 / y +0.91 mm per mm, v3b +1.57 / +0.38, v3c +0.56 / +0.37 (another v3b run +1.19 / +0.27) | `20261027_160415`, `20261027_192340`, `20261028_084359`, `20261028_092447` (bench `--hover-perturb-mm`, 4D) | The test on v4, and at an x ≥ 0.21 tile. Hover A/B (1, Next 3) waits for a slope near 0 |
| Golden eval set (2) | Recorded and registered: `Yewy/mahjong_mat_eval_v1`; layouts `eval_familiar`, `eval_new`, `eval_tilt` | `7ed5e58`, `905f35a`; `data/eval_split.json` | Score v4 on it (`offline_eval.py --policy pickup_mat_cl_v4 --against pickup_mat_cl_v3c`); check v2 does worse on `jitter` (sequence step 4, not found in `data/offline_eval/`) |
| v2n / v3 / v3b / v3c (7) | Trained, scored offline, benched. v2n vs v3 paired 10/10 vs 9/10 | `20261027_161356_bench`; `data/offline_eval/`; `python scripts/status.py` | Retire from the bench once v4 holds up |
| DAgger (9) | Round 1 on v3c → v4 (shipped). Round 2 on v4 (`Yewy/mahjong_mat_dagger_v4`, episodes 20, 25 dropped) → `pickup_mat_cl_v5` (default 2026-10-29), then v5's data + green-mat demos with augmentation → **`pickup_mat_cl_v6`**, the default since 2026-10-30 | `0957a61`, `b6e2095`; `20261029_111756_record`, `20261029_115104_record`; `data/offline_eval/pickup_mat_cl_v5@*` | Round 3 on v5 (`Yewy/mahjong_mat_dagger_v5`): WW with a neighbour beside it, or move WW inside comfortable reach; v5's pause at the 4D hover |
| Returns (8) | Aim at home (3mm median at 6C); let go at the grasp's own height (#33) | `docs/retrospectives/2026-10-28_return_homes.md`; `bf75394` | #33's release height unconfirmed on the arm: check `session_report` "return release" on the next bench |
| Autonomous play (8) | Built, `--sim` only. `hand13` taught; `hand14` = hand13 + the drawn tile's spot S13 (WW), benched 73/75 (S13 6/6; S12 the only misses, 1/3) | `20261029_094543_bench`; `data/layouts.json` | Teach the drop zone on hand14's S13 (play checks it was benched, `markers.zone_test`); discard pose; the scripted hand (8, Next 2) |
| Black top (v5, v6, v7) | v6 ran there on 2026-10-30: picked 99/108 but clean 72/108, knocks traced to the open jaws on tight spacing. v7 (registered, not the default) 14/14 picked, 8/14 clean, unpaired | `20261030_142320_bench` (v5 vs v6, close-hand14: clean 9/28 vs 14/28), `20261030_174042_bench` (v6, 21% hover, even-hand14: clean 11/14), `20261030_175235_bench` (v7); `docs/retrospectives/2026-10-30_close_layout_bumps.md` | The v6 / v7 pairing above |
| Scorecard (4) | Built: clean attempts, ranges, a paired verdict. **The ship rule is proposed, waiting on sign-off** | `docs/scorecard.md`; `mahjong/scorecard.py` | Sign off the rule; add the hover and control-code cards when one is needed |
| Tile detector (10) | v6x3 live: 97.1% right on the exam without flowers (was 68.6%). Exam: 2 capture sessions | `2fee0eb`; `data/detector_exam.json` | More exam sessions (one per lighting, ~1000 tiles for the wrong-name bound); record.py's scans don't save frames yet |
| Harness (6) | Fast tests 96 → 11 s; `merge_pr.py` (merged-tree tests, closing step), `status.py`, `test_docs.py`, checkout warnings | this table's branch (`worktree-harness-docs-merge`) | `merge_pr.py --close` for the 35 merged branches left over |

**The sequence decided 2026-10-23** (v2n baseline → hover jitter → golden set → check the eval on v2 → v3 →
DAgger → hover A/B) ran to DAgger in six days. Two steps didn't come out as planned: the eval was never
checked on v2's `jitter` slice (step 4), and v3's perturbation slope isn't ~0 (step 5). Both are rows above.

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
1. The same test at an x ≥ 0.21 tile. Returns now go to the tile's home (2026-10-28, below in 8), so
   a perturbed hover no longer drifts the tile into its neighbours.
2. A jittered dataset (7; `record.py --hover-jitter-mm` built 2026-10-23). Rerun this test on the
   policy trained from it; expect a slope near 0.
3. Only then a **paired A/B** of a better hover vs the live one (`--correction a,b`, the same
   alternation as `--policy a,b` in 5), judged by the hover→grasp shift in mm as well as success,
   and **switching** by editing `data/hover_model.json` in one commit that cites the A/B session.
4. ~~Bench attempts across the mat~~: done, `--layout` walks (eval_new, hand14: 75 attempts over 13 spots
   on 2026-10-29). The perturbation tests are still all at 4D.

## 2. An offline evaluation set

**Built 2026-10-23** (`mahjong/offline_eval.py`, `scripts/offline_eval.py`, `mahjong/layout.py`,
`scripts/layout.py`); **`Yewy/mahjong_mat_eval_v1` recorded and registered 2026-10-27** (`7ed5e58`; layouts
`905f35a`). Still open: v2 on the `jitter` slice (the 2026-10-23 sequence's step 4, in Status), and v4's scores. Write-up:
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
- **The eval recording** (eval-only, ~48 episodes, 90 % hover so trimming the narrowing also scores a
  narrowed-hover policy like v2n). Spend episodes on spots, not repeats: one offline score already averages ~200 frames.

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
  2. Teach the layouts: `layout.py --name eval_familiar --teach` (d detects, t saves), then
     `--add T1 --like S2 --yaw 15` for the tilted spots. Commit `data/layouts.json` (the spots); the ghost
     photos in `data/layouts/` stay local. (The scratch `angle_check` grid was dropped 2026-10-23.)
  3. Record: `record.py --dataset Yewy/mahjong_mat_eval_v1 --slice familiar --layout eval_familiar`;
     the jitter slice: `--slice jitter --layout eval_new --hover-offsets-mm 6x,-6y`.
  4. Register it: `offline_eval.py --add-eval-set Yewy/mahjong_mat_eval_v1`.
  5. Check the dim slice predicts the hardware freeze: v2 should start descending late there.

Not yet: **bench attempts as regression cases.** Every attempt now saves `wrist.mp4` (5), so a new
policy can be replayed on the same frames, open loop. There's no ground truth (these are the old
policy's own actions), but a big difference from the old policy on an attempt that *succeeded* is
worth flagging. The limits of offline scores (compounding error) are in `mahjong/offline_eval.py`.

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

Checked on the arm 2026-10-23 (test sessions run from the interventions worktree, not kept): the knock
detection and the confirm-at-the-bench flow work; marking "touched tile" on a descent that needed no
help is what led to the separate `collided` mark. Still to check: one session that does each row above
on purpose, and `KNOCK_M` against that session's scan jitter.

## 4. A scorecard for ship or revert

**Built 2026-11-02 for policy changes: `docs/scorecard.md`.** What follows is the original reasoning.

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
- **Draft format (2026-10-23):** a scorecard page with a rationale per criterion and a glossary
  (EXTEND renamed RETEST; guardrails split into hard and soft), drafted outside the repo (plan
  `~/.claude/plans/reviewing-item-4-in-soft-blanket.md`). Not yet in the repo: write it up as
  `docs/scorecard.md` once settled, then build it into `session_report.py`.

## 5. Bench features those need

**Done (2026-10-23)**, checked in `--sim` only; still needs a session on the arm:
- `--policy a,b`: trials run in rounds, each policy once per round from the same hover offset and
  its own hover, alternating order (a b, b a) so a drift over the session doesn't favour one.
  `--trials N` counts rounds: N × policies × spots pickups with `--layout`.
- `--layout NAME` (instead of `--positions`): trials walk the spots of a layout taught with
  `scripts/layout.py`, picking the tile detected nearest each spot. An empty spot is skipped (an
  observation) and the walk goes on. `session_report.py` prints success per spot x policy.
- `wrist.mp4` per attempt: every frame `policy.run` saw (for replaying a new policy, 2). Check on the
  arm that encoding keeps the loop at 30 Hz.
- Re-detect: already the case. Every trial scans before choosing, and the scans around it flag moved
  tiles (3).

Return accuracy, moved to 8, is done (2026-10-28): returns go to the tile's home.

## 6. Housekeeping

- ~~A pre-commit hook that runs the fast tests~~ Done: `scripts/hooks/pre-commit` (~20 s), on with
  `git config core.hooksPath scripts/hooks`; `pre-merge-commit` runs it on merges too.
- ~~Review the merged state after the 2026-10-23 worktree merges~~ Done (8 commits): the `record.py`
  save crash, one copy of each duplicated helper/constant (episode-log reader, gripper thresholds,
  held check, `grasp_point`, default hover), `bench.parse_args` for tests, `--help` and docs vs code.
  How to avoid a repeat is in CLAUDE.md "Worktrees and merging" (`data/hover_corrections.jsonl` now
  merges by union).
- End-to-end sim tests for the user-run scripts that have none (they're what catch a merge that's
  clean but broken): `layout.py` teach/view/capture, `make_policy.py` (a `--skip-build/--skip-train`
  path), `offline_eval.py` main and `--make-split/--add-eval-set`, `quality_report.py`, `mine_frames.py`.
  (collect_corrections and `mahjong/quality.py` have theirs.)
- ~~Review the scripts against each other after the 2026-10-23..28 worktrees~~ Done 2026-10-29 (the
  consolidate-* branches): record.py returns to homes and logs return samples, the gamepad deadzone and off-centre
  warning at every handoff, play reviews marked pickups, one copy of each shared helper.
- ~~Docs that go stale, merges that skip the tests, branches never closed~~ Done 2026-10-29: `tests/test_docs.py`
  (docs vs `data/` and the scripts' options), `scripts/status.py` (state from `data/`; the default changes with
  its evidence), `scripts/merge_pr.py` (tests the merged tree, then pulls main and closes the branch), session
  warnings when the checkout is behind origin/main, and the fast tests 96 → 11 s (FK without `np.block`,
  pytest-xdist in the hook).
- Optional: a Claude Code skill / slash command, "review the latest session". It runs `session_report`, reads
  the contact sheet and compares the run with `--all`. Less needed now that `session_report` covers
  interventions and the CLAUDE.md says to run it first.

## 7. The next dataset

- Hover with the jaws narrowed (v2n: 27.2 %) instead of 90 % (skips ~3.5 s of narrowing per episode). The hover is
  per policy, so this is a data change. Because the narrowing is a time prefix, trimming the v2
  episodes (`trim_narrowing`) and retraining also works without new recordings. A/B it against v2
  at the current hover before changing the hover too.
  **Done since (2026-10-27/28):** v3 (88 jittered demos, `Yewy/mahjong_mat_pickup_jitter_v3`), v3b and v3c
  (95 demos) trained from v2n's narrowed hover, scored offline and benched; v3c was the default until v4 (9).
  The rest of this item is the v2n history.
  **2026-10-23:** `pickup_mat_cl_v2n` (`make_policy.py --source
  Yewy/mahjong_mat_pickup_lekiwi_cl_hover --version v2n --trim-narrowing`): 89 episodes, the 15 eval
  episodes left out, so unlike v2 its offline scores are honest. It is the baseline for v3. If it does
  worse than v2, record v3 at 90 % instead.
  **Offline (2026-10-23 18:00):** 0 freezes, lag −3 steps, close 0.18 s, grasp 1.8 mm: pass. Arm error
  0.70° vs v2's 0.21°: FAIL. v2 on the same post-narrowing frames is 0.25° and v2n on its own training
  episodes 0.23°, so the gap is mostly that v2 trained on the 15 eval episodes. The "not much worse
  than the reference" gates can't judge a policy against v2 until the golden set exists (neither has
  trained on it). Worst spot 21,+13: grasp 12 mm, plan descends 48 steps early. The bench decides it:
  paired v2n vs v2 at a few spots.
- Record with deliberate random hover jitter (±~6 mm), so the policy has to find the tile in
  the image rather than in the mat position. Hover improvements (1) are then safe without a retrain.
  **Built 2026-10-23:** `record.py --hover-jitter-mm R` (uniform over the disc, `--jitter-zero-fraction`
  0.2 kept clean, `--jitter-seed`); an episode keeps its offset through ← and discards. Each log row
  has `hover_jitter_xy`. Jittered demos stay valid hover-correction samples (grasp − detection).
  Record from v2n's hover (`--hover-gripper 27.2`); `make_policy.py --version v3` registers
  `pickup_mat_cl_v3`.
  Open: how many of v2's clean episodes to mix in; they carry the memorised per-position move.
- More episodes at the edges and on the second row. No pauses mid-episode.
- Consider an absolute gripper value instead of a delta (CLAUDE.md, Open items).
- Leave the eval split (2) out from the start.

## 8. Autonomous play

Detect → pick → place / discard as a loop on top of the bench's pieces (hover, `policy.run`,
place, discard). Do this after 4, so it builds on a policy the scorecard has shipped. The bench's
`--layout` walk is the same loop shape.

**Return accuracy (from 5): done 2026-10-28.** Every bench return aims at the tile's home (where it was
first seen this session) through `mahjong/return_correction.py`, a fit of where a set-down tile lands
given how off-centre the jaws held it. Each after-return scan logs the landing. 15 rounds at 6C landed
3 mm from home (median, worst 4 mm), vs 11 mm the day before
(`docs/retrospectives/2026-10-28_return_homes.md`). The play loop should place with
`ReturnCorrection.target(home, detection, grasp)`. Still to check: other spots, the reach's edge, a
`--layout` walk.

**Plan (2026-10-28): `docs/autonomy_plan.md`**: a review of MahjongBot's `mahjong_autonomous_play.py`,
what a clean hand means (a person only puts drawn tiles down; ~23 pickups, 23 set-downs and 27 name reads
a hand, so each must work ~99.7% of the time for 8 in 10 hands to be clean), no silent wrong tiles as a
gate, and the targets and guardrails. Decided: ArUco marker under the drawn tile (move it beside the zone
if zone pickups fall below the mat's ~95% or the detector reads its edge), a 2-row hand, no calls,
copies left tracked with ukeire.

**Built (2026-10-28), checked in `--sim` only:** `scripts/play.py` (README, "Autonomous play") on the
bench's pieces; `mahjong/game.py` (our own shanten, checked against the `mahjong` library, which this
package's name shadows; the discard choice; the hand checks); `mahjong/identity.py` (name votes per slot
over every scan, reset when a tile moves; reads the scans' per-frame `votes`);
`mahjong/markers.py` (the drop zone); the `ask` intervention class and `interventions.hand_grade`;
`session_report.py` prints a hand, `--all` every hand; `detector_eval.py` prints the wrong-name bound.

Next, on the arm:
1. ~~Teach `hand13`~~ (done) and the drop zone. `hand14` is the bench's version of the hand: hand13's slots
   plus S13 (WW, (0.223, −0.158), at the edge of the reach), where the drawn tile goes. Benched 2026-10-29
   (`20261029_094543_bench`, 73/75; S13 6/6). Lay the marker on S13: `play.py` won't start until a bench
   session has picked up and returned a tile within 25mm of the taught zone (`markers.zone_test`). Then
   paired pickups on the marker vs a mat spot (the marker's pivot trigger).
2. A scripted hand: a fixed wall set out by hand, so it can be played again identically.
3. Set `identity.py`'s bars from the exam (they're guesses), after more exam sessions (one per
   lighting, ~1000 tiles: section 8 of the plan).
4. The soak: 10 hands across the lighting conditions, battery full to low. Target: 8 of 10 clean, no
   silent errors.
5. Not built yet: the drop-zone reading station and its self-labelling capture, a wrist name check (only
   if a classifier trained on `wrist_capture.py`'s photos passes an exam), battery / servo-temperature /
   brightness health checks each turn.

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

**Built (2026-10-28):** `record.py --policy ID` (README, DAgger). The policy drives until the first
press of A (`GamepadTeleop.takeover_pressed`; any touch took over at t=0 from stick drift), then the gamepad starts from `Arm.commanded()`.
Every frame is recorded, with `policy` / `takeover_frame` in the episode-log row and a `takeover`
rescue in the session. `dataset.build` keeps the frames from the takeover on, anchored at the hover
(frame 0), and leaves out episodes with no takeover. `make_policy.py --add REPO` trains on such a
dataset next to the main one.

**Round 1 done (2026-10-28/29):** corrections on v3c (`Yewy/mahjong_mat_dagger_v3c`) → `pickup_mat_cl_v4`
(34195 frames, 47000 steps), default since 2026-10-29 after the hand14 paired run (Status). **Round 2:**
`Yewy/mahjong_mat_dagger_v4`, corrections on v4 → `pickup_mat_cl_v5` (both DAgger sets, registered 2026-10-29,
`b6e2095`), default since 2026-10-29 after the paired bench against v4 (Status). Still worth targeting: the far spots, where v3b and v3c descended ~13mm deeper
than the demos (FK heights drift with reach, CLAUDE.md), and S12 (7D), the only hand14 misses.

## 10. The tile detector

The YOLO detector (`assets/tile_detector.pt`) misses tiles now and then and often names one wrong. It
was trained on 2026-07-21 on 495 RealSense frames of another table and room, with tiles scattered and
turned. Since 2026-10-20 the camera is on the rear tripod over this mat, with rows of tiles, a
keyboard, a person and the discard bin in view: a scene it has never trained on. A tile is ~24×18 px
in the 848×480 frame, so similar faces (2/3, 7/8 of a suit) are close calls. Inference already runs
at 1280 px like training (no scale mismatch to fix).

**Built (2026-10-27): photos → exam → mining → retraining** (README, "The tile detector"):
- Every bench scan saves its 5 frames, depth and every box down to 0.25 confidence
  (`sessions/*/scans/`); `scripts/capture_frames.py` takes scans without the arm.
- The exam: whole `--exam` capture sessions listed in `data/detector_exam.json`, labelled in their own
  Roboflow project, never trained on. `scripts/detector_eval.py` scores found / named right / missed /
  extra and the mix-ups, by lighting.
- `scripts/mine_frames.py` exports the frames it struggled with (flicker, dropout, low confidence,
  glitch answers) and plain current-scene frames, pre-labelled for Roboflow. `scripts/train_detector.py`
  fine-tunes from the live weights with their July recipe, and refuses exam frames.

**Shipped 2026-10-28: v6x3** (`2fee0eb`): 97.1% right overall on the exam without flowers (the live
detector was 68.6%), README "The tile detector". The exam has 2 capture sessions so far
(`data/detector_exam.json`).

**Next, on the hardware:**
1. Capture the rest of the exam: one `--exam` session per lighting (daylight, lamp, ceiling), 40-60 scans in all,
   arrangements of their own (rows, tiles pushed together, turned tiles, the tilt layout's angles).
   Label it, score the live detector: the baseline. Commit `data/detector_exam.json` and the scores.
2. Training frames: a few `capture_frames.py` sessions plus the bench's scans, `mine_frames.py --all`,
   label them in the July project, train, score against the live one, ship only on a better exam.
3. Not yet: `record.py`'s scans don't save their frames (the bench and capture do). Add it in its own
   change (record.py is shared with other work), then recording sessions feed mining too.

**Ideas for later (plan steps 4-6), only if the exam still shows the problem they solve:**

- *Depth tile candidates (missed tiles).* The table plane is already fitted (`config.table_z`), so
  depth needs no per-frame RANSAC: project the scan's depth (now saved as `depth.npz`) into the base
  frame, subtract the plane, keep 10-30 mm (a tile is 20 mm tall; depth noise is ~2-3 mm at 0.82 m),
  and mask to the mat (the arm, base, bin, keyboard and a hand stand above the plane too). Split
  touching tiles by the 35×26 mm face. Then each blob without a YOLO box is a missed tile (a mining
  trigger, and a crop to classify), and a box with no blob is likely a false detection. **Keep aiming at
  the box centre**: the hover correction and v2's memorised move are fitted to `detection_xy`, and a
  blob centroid would shift it systematically. Depth edges blur by 2-4 px on a 24 px tile: fine for a
  crop, not an exact box. On the dark desk `layout.find_tile` (white face on dark) does much of this
  from colour already; depth adds robustness to light and to a light background. Test on synthetic
  depth (plane + raised blocks, a touching pair, a tall object) and on saved scans.
- *A crop classifier (wrong names).* Re-running the full-scene YOLO on an upscaled 24 px crop of the
  same frame gives it no new pixels and an input unlike its training. Better: a small 38-class
  classifier (e.g. YOLO-cls nano on ~64 px crops) trained on crops cut from the labelled boxes, run on
  *every* tile, its vote combined with YOLO's. `merge_tiles` takes the single most confident read
  today; voting across the 5 frames (and both models) would steady flicker too. Only worth it if the
  exam's mix-up list is still long after retraining.
- *More pixels per tile.* Stream colour at 1280×720 or 1920×1080 (depth aligned to it): ~1.5-2.3× the
  pixels on each face. It changes the detector's input, so it means a new capture and labelling round:
  decide before step 2's big labelling round if the mix-ups are between near-identical faces. The
  sharpest view of all is the wrist camera at the hover (~4 cm above the tile): a name check there,
  before ACT starts, would catch a wrong tile before the arm commits to it.
