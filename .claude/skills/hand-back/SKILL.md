---
name: hand-back
description: Verify a finished branch and merge it through the queue, so no person has to. Use when the work in a worktree is done, the user says ship it, hand it back, merge it, or open the PR.
model: sonnet
---

A branch is merged by `python scripts/merge_pr.py --auto N` (the queue), not by a person, once it passes everything below. A round
is one pass through steps 1-5 (a send-back starts the next); at most 3 rounds; if still not clean, stop and say what's left.
Never `--no-verify`, never push to `main`.

1. **Up to date**: `git fetch origin && git merge origin/main`; resolve conflicts by keeping both sides' intent. If the work was planned
   before recent merges, run `/recheck-main` first.
2. **Checks**, all from the worktree:
   - `~/miniforge3/envs/lerobot/bin/python -m ruff check .`
   - `~/miniforge3/envs/lerobot/bin/python -m pytest -m "not slow" -n 8 -q` (every sim arm command is checked by the motion watchdog;
     a watchdog failure is a motion bug until shown otherwise: never mark a test `motion_unwatched` to pass it)
   - the slow tests too if `mahjong/policy.py` or `mahjong/dataset.py` changed: `python -m pytest -m slow -q`
   - `python scripts/review_brief.py --range origin/main..HEAD` is the reviewer's view; its sim runs are the queue's too.
   - if the diff touches `piq/` (the local model's prompt, skills, questions or launcher): `python piq/check.py --ask` from the worktree. It runs the
     model's regression questions (about 3 minutes, needs Ollama) and a test can't. A FAIL is a bug in `piq/` until shown otherwise; a stale question
     means this branch changed what it assumed, so update the question. If Ollama isn't reachable, say so under `## Unverified on the arm`, don't skip it silently.
3. **Review**: `/code-review medium` on `git diff origin/main...HEAD`, every round. Fix what's clearly a bug and **commit it**, rerun
   2, then one `/code-review low` on all of this round's fixes (`<commit before them>..HEAD`; name the changed functions so it
   checks their callers). At most 2 low passes a round; a bug the second one finds is open. A bug still open: stop before opening
   or re-readying the PR and say what's left. Note each finding and what was done (fixed, or declined and why; a declined finding isn't open).
4. **The PR body**, written to a file in the scratchpad and passed with `--body-file` (never inline: quotes and backticks break):
   `## What changed` (and why), `## Files`, `## Overlaps` (with main and open branches; the overlap hook's notes go here), `## Review`
   (every round's /code-review findings and what was done), `## Unverified on the arm` (what's checked offline, what needs hardware).
5. **Push, open it ready, then queue it**: `git push -u origin HEAD`. Round 1: `gh pr create --base main --title "..."
   --body-file <file>` (not `--draft`). After a send-back: `gh pr edit <N> --body-file <file>` and `gh pr ready <N>`. Then, from the
   main checkout, `cd <main checkout> && python scripts/merge_pr.py --auto <N>`.
   - **merged**: say so, with the PR link. The queue pulls main and closes the branch.
   - **sent back**: read the PR's newest comment, fix on the branch, and go round again from 1.
   - **held** (`needs-owner`: it changes a decision, such as the default policy, an exam or eval split, the standard setup, the
     scorecard or installed weights): stop; tell the user plainly it is waiting for them and why (`merge_pr.py <N>` merges it).
6. If a low review ran, say that a plain `/code-review` may now reuse the low level, so type `/code-review medium` next time. Then end
   with one plain line: merged, or not merged and what it waits on.

Real arm runs still ask a person (the arm gate); everything here is `--sim`.
