# Roadmap

Bigger steps after the 2026-10-23 bench improvements (early stop, preflight, setup questions).
Roughly in order: the evaluation pieces first, because they decide whether everything after them
helped.

## 1. Every bench run improves the hover

Each attempt already logs the detection, the hover target (`hover_xy`) and where ACT closed the
gripper (`grasp_xy`). On 2026-10-23 ACT moved **+6 to +10 mm in x and +1 to +11 mm in y** from the
hover before grasping, every time (one 3D attempt: +14/+26). In the recorded demos the same shift is
a median of **0 / +3.5 mm**. So the hover probably still sits ~5-8 mm off, and ACT quietly makes
up for it — the margin it has left for a harder position.

- Turn each *successful* attempt into a hover-correction sample: target = `grasp_xy` minus the demos'
  typical hover→grasp shift. Store them as `source: "bench"` in `hover_corrections.jsonl`, weighted
  below gamepad samples. Refit when there are enough, and keep the leave-one-out check so a bad batch
  can't make it worse.
- Also log the settle itself (commanded vs measured tip at each step), so sag can be tracked over
  battery level and time.
- Caveat: most of the data so far is one position (4D). It needs attempts across the mat (5).

## 2. An offline evaluation set

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

Each time a person steps in, it should be an event, not a memory. Candidates:

| Intervention | How it's logged today | Make it |
|---|---|---|
| Esc abort | `stop_reason: aborted` | ✓ already |
| Success overridden by hand | `gripper_says_held` ≠ `success` | ✓ derivable |
| Tile reset by hand after a discard | — | bench asks "tile back on the mat?" / a key |
| Neighbouring tile knocked | — | key during an attempt; or compare overview before/after |
| Re-detection needed (tile missed or merged) | — | log detections per choose() |
| Arm unstuck / restarted / torque off | — | session_end reason + count |
| Hover nudged with the gamepad | — | if a pre-ACT nudge is added |

Metric: **interventions per 10 attempts**, and attempts between interventions.

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
  per policy, so this is a data change.
- More episodes at the edges and on the second row. No pauses mid-episode.
- Consider an absolute gripper value instead of a delta (see the CLAUDE.md conventions).
- Leave the eval split (2) out from the start.

## 8. Autonomous play

Detect → pick → place / discard as a loop on top of the bench's pieces (hover, `policy.run`,
place, discard). Do this after 4, so it builds on a policy the scorecard has shipped.
