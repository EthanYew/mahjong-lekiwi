---
name: clean-dataset
description: Review a recording before training and propose which episodes to drop, with reasons. Use when the user says they recorded episodes, clean them up, check this recording, or which episodes are bad.
context: fork
model: sonnet
---

Review this recording: $ARGUMENTS. Propose; the owner decides. Don't run `--drop`, `--fix-gripper` or `--clean`, and don't change `data/`.

1. **The setup.** `python scripts/quality_report.py --dataset REPO`, then `--phases`. From its `episode_log.jsonl`: the hover opening, the jitter, the slices, the surface. Does the opening match the recordings it'll train with? Mixing needs their gripper deltas to agree (`.claude/rules/policy_data.md`).
2. **The flagged ones.** `--episode N` for each, plus the top of the ranking even if nothing is flagged: `quality_report.py` flagged none of the standard mat recording's 56 episodes on 2026-11-03, and a person's review still dropped some.
3. **Look.** Save frames from the wrist video of each suspect and view them before judging. A number says an episode was odd, not why.
4. **`--against REPO`** when it should steer like an earlier recording at the same places.
5. **`--grip-reach --dataset REPO`** for demos that closed short of the tile's middle, toward the robot ("gripped low"), judged against the bench attempts the owner didn't mark low. Run it from the main checkout: it reads the bench sessions. Each proposed episode still needs a look at its close.

**Not flaws, so don't propose dropping for these:**
- A DAgger (correction) episode narrowing its jaws early, near hover height. The operator does it on purpose so open jaws don't hit neighbouring tiles.
- A gap between jaws and tile at the close. Demos leave a margin on purpose. Judge a close by how far it strayed from where jaws usually close at that place, not by its distance from the detection.

**Real reasons, from past reviews:** narrowing during the descent in a non-DAgger recording (`--trim-narrowing` may cover it, so say how much it would cut), holding about a tile's width (11%) on the way down, a long searching path, a wrist pitched unlike the rest, and a close that stopped short (`--fix-gripper` keeps the episode, so prefer it). Pauses alone aren't a reason: `dataset.build` already drops idle frames.

**Leave alone:** eval splits (`data/eval_split.json`) and held-out episodes. Dropping one changes what a score means, so flag it instead. Note whether a drop leaves any spot thin, using the log's layout spot.

Return a table of episode, proposed action, reason and the evidence you looked at, then how many remain per spot, then the exact `quality_report.py --drop ... --reason "..."` commands for the owner to approve. `docs/retrospectives/2026-11-03_standard_mat_data_review.md` is the model. Say plainly if the recording looks fine.
