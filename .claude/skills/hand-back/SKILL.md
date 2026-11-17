---
name: hand-back
description: Verify a finished branch and merge it through the queue, so no person has to. Use when the work in a worktree is done, the user says ship it, hand it back, merge it, or open the PR.
model: sonnet
---

A branch is merged by `python scripts/merge_pr.py --auto N` (the queue), not by a person, once it passes everything below. A round
is one pass through steps 1-5 (a send-back starts the next); at most 3 rounds; if still not clean, stop and say what's left. Never `--no-verify`, never push to `main`.

1. **Up to date**: `git fetch origin && git merge origin/main`; resolve conflicts by keeping both sides' intent. If the work was planned
   before recent merges, run `/recheck-main` first.
2. **Checks**, all from the worktree:
   - `~/miniforge3/envs/lerobot/bin/python -m ruff check .`
   - `~/miniforge3/envs/lerobot/bin/python -m pytest -m "not slow" -n 8 -q` (every sim arm command is checked by the motion watchdog;
     a watchdog failure is a motion bug until shown otherwise: never mark a test `motion_unwatched` to pass it)
   - the slow tests too if `mahjong/policy.py` or `mahjong/dataset.py` changed: `python -m pytest -m slow -q`
   - `python scripts/review_brief.py --range origin/main..HEAD` is the reviewer's view; its sim runs are the queue's too.
3. **Review**: `/code-review medium` once on `git diff origin/main...HEAD`. Fix what's clearly a bug and rerun 2. If you fixed
   anything, note `HEAD` before the first fix (after step 1's merge) and run `/code-review low` on just that diff
   (`git diff <that commit>..HEAD`) and the callers of what it changed, not the whole branch again. If that finds a bug, fix it and
   rerun 2 once more; anything still open goes in `## Review` and you stop and say so. Note each finding and what was done (fixed,
   or declined and why).
4. **The PR body**, written to a file in the scratchpad and passed with `--body-file` (never inline: quotes and backticks break):
   `## What changed` (and why), `## Files`, `## Overlaps` (with main and open branches; the overlap hook's notes go here), `## Review`
   (the /code-review findings and what was done), `## Unverified on the arm` (what's checked offline, what needs hardware).
5. **Open it ready, then queue it**: `gh pr create --base main --title "..." --body-file <file>` (not `--draft`), then, from the main
   checkout, `cd <main checkout> && python scripts/merge_pr.py --auto <N>`.
   - **merged**: say so, with the PR link. The queue pulls main and closes the branch.
   - **sent back**: read the PR's newest comment, fix on the branch, `gh pr ready <N>`, and go round again from 1.
   - **held** (`needs-owner`: it changes a decision, such as the default policy, an exam or eval split, the standard setup, the
     scorecard or installed weights): stop; tell the user plainly it is waiting for them and why (`merge_pr.py <N>` merges it).
6. End with one plain line: merged, or not merged and what it waits on.

Real arm runs still ask a person (the arm gate); everything here is `--sim`.
