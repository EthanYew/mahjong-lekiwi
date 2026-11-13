---
name: next-run
description: Turn where things stand into one plan for the next run on the arm or the next recording. Use when the user asks what to run next, is this enough episodes, why this comparison, or feels they're playing whack-a-mole.
context: fork
---

Plan one run. What the user is weighing, if anything: $ARGUMENTS

The trap this exists for is changing several conditions at once and learning nothing. Say so if the plan on the table does that.

Read, in this order, and no more than you need: `python scripts/status.py`, the status table in `ROADMAP.md`, `python scripts/session_report.py` on the latest
session and `--all` for what changed between sessions, `docs/scorecard.md`. Sessions are in the main checkout. Don't read old conversation logs.

Return exactly this, in plain language, each line short:

1. **The failure this targets**, with the session it was seen in and how often. No failure with evidence means no run: say what would produce some.
2. **The one thing that changes**, and everything held the same (mat, layout, hover opening, light). Counted results use `data/standard_setup.json`.
3. **The prediction**, as a number, and the result that would prove it wrong.
4. **The command**, ready to paste, with how long it takes on the arm.
5. **How much is enough**: from `docs/scorecard.md`'s rules, the size at which this gives an answer, and what an inconclusive result at that size would and wouldn't mean.
6. **What it can't tell us**, and **what happens next** on each outcome.

If two runs are close, recommend one and say in a line why not the other. Change nothing, and start no arm script.
