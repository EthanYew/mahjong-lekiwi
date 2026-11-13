---
name: post-merge-review
description: Review what merged to main since the last review, in a fresh context, and fix what's clearly broken in a draft PR. Use when the user says PRs were just merged, asks to clean up after merges, or asks to review a batch of changes.
context: fork
agent: merge-reviewer
effort: high
---

Review the merges since the last review. Arguments, if any: $ARGUMENTS

If the arguments say `findings only`, change nothing. If they name a range (`A..B`), review that range. Otherwise run
`~/miniforge3/envs/lerobot/bin/python scripts/review_brief.py` from the repo root, which works out the range and shows what to start from.
