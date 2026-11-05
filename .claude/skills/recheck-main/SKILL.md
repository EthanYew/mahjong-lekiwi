---
name: recheck-main
description: Check a plan or an open branch against what has merged since it was written. Use when the user says PRs merged, is this plan still right, or asks if a branch is still needed. For bugs in the merged code itself, use /post-merge-review.
context: fork
model: sonnet
---

Compare this with what has merged since: $ARGUMENTS (a plan's path or text, a branch or a PR; none given means the current branch).

Git catches overlapping lines, not overlapping intent, so read for intent. Run `git fetch origin` and work against `origin/main`. Read diffs, not whole files.

- **A plan:** find when it was written, from its file's commit or the date in it. List what merged since (`git log --merges`, and `gh pr view N` for what each says it does). For each step of the plan: still needed, already done (by which PR), or now wrong (and why).
- **A branch:** from `git merge-base`, what did `main` and the branch each change? Look for the same function, constant, option or threshold added on both sides under any name, a helper the branch could now reuse, and an assumption a merged PR broke, such as a default, a log field or an order of events.
- **Lost work:** anything the plan or branch relied on that a later merge removed or reverted.

Return a short table of each item, its verdict and the PR or commit behind it, then what to change, smallest first. Say plainly if it all still holds. Change nothing.
