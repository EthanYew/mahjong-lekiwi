---
paths:
  - "scripts/merge_pr.py"
  - "mahjong/checkout.py"
  - "scripts/review_brief.py"
  - "mahjong/code_health.py"
  - ".gitattributes"
  - ".claude/hooks/overlap_warning.py"
  - "tests/test_overlap_hook.py"
---
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
- **Handing a branch back** (Claude's job, `/hand-back`): merge the latest `main` into it, ruff and the tests, `/code-review` (`/hand-back`
  says how far; no ready PR while a bug it found is open), and a ready PR whose body has `## What changed` (and why), `## Files`,
  `## Overlaps` (with `main` or other open branches), `## Review` (each finding and what was done) and `## Unverified on the arm`.
  Then the queue merges it: `python scripts/merge_pr.py --auto N` (README, "The merge queue").
  It refuses rather than asks: an unfinished description, a missing review for `REVIEW_FILES`, a name added on both sides, a
  merged tree that fails the tests or the sim runs. A sent-back PR has the output as a comment; fix and requeue. A decision
  (`HOLD_FILES`, weights, the default policy) gets `needs-owner` and waits.
- **Merging by hand** (the person, for held PRs): `python scripts/merge_pr.py N` from the main checkout, one PR at a time. It shows
  the description, what `main` and the branch both changed since they split (and any function, constant or
  option added on both sides: a duplicate to resolve, conflict or not), runs the tests on `main` + the branch
  in a scratch worktree, asks, merges, pulls `main` here and tests it. Don't merge on GitHub's page: it skips
  every test (29 of the first 40 merges did), and the main checkout isn't pulled. Never `--no-verify` a merge.
  It refuses, before touching GitHub, if that pull would fail (`pull_blockers`): #65 merged on GitHub, then stopped at the pull
  because `main` here had a commit GitHub didn't, so nothing was pulled, tested or closed.
- **Data the scripts log** (samples, reviews) goes straight to `main` from the main checkout, committed and pushed in one step
  after a hardware session (README, "Data files and merging"); uncommitted until then is normal. A branch doesn't carry it. A decision
  (default policy, exam list, scores, weights) goes in a PR with its evidence. A Claude session doesn't push to `main` (samples go by `save_samples.py`, merges by the queue).
- **Closing** (part of `merge_pr.py N`): the branch's worktree `sessions/` are moved into the main checkout's,
  then the worktree, local branch and GitHub branch are removed. `python scripts/merge_pr.py` lists anything
  merged but not closed; `--close` closes it (35 branches had piled up by 2026-10-29). A Claude session can't
  delete worktrees or branches on its own: it says to run this.
- **After a batch of merges**: a review in a fresh context before the next hardware session (`merge_pr.py --review`, or
  `/post-merge-review`). It marks the range reviewed when its PR merges (`data/post_merge_reviews.jsonl`); until then `status.py` and hardware
  sessions count what it hasn't seen. `tests/test_docs.py` checks the docs against `data/` and the scripts' options on every commit.
- **Tests that catch a clean-but-broken merge** are end-to-end sim runs of a script's main path
  (`tests/test_bench.py`, `tests/test_record.py`). A new user-run script gets one.
- `data/hover_corrections.jsonl`, `data/return_corrections.jsonl` and `data/wrist_review.jsonl` merge by union
  (`.gitattributes`): both sides' samples are kept.
- **An edit to a file another open branch changes gets a note** (`.claude/hooks/overlap_warning.py`, after every Edit and Write,
  once per file per session). 10 of the 12 PRs from #67 to #78 changed `play.py`, and `merge_pr.py` only shows an overlap at
  merge time. Open branches are `origin`'s as last fetched, less those on `main` and this branch's own base. It never blocks:
  act on it (merge the other first, or build on it) and name it under `## Overlaps`.
