---
name: should-ship
description: Decide whether a policy should become the default. Use when the user asks to ship, compare or make a policy the default, or asks which policy is better.
---

Apply `docs/scorecard.md`; don't invent a new standard for this decision.

1. Find a session that ran both policies with `bench.py --policy baseline,candidate`. If there's none, give the command to run. Don't decide from totals across sessions.
2. Run `python scripts/session_report.py <session>` and read the verdict, with the offline scores (`data/offline_eval/`) as a screen only.
3. Report the verdict, the clean pairs, the p value and any spot it names as worse. Say what the session can't show, such as other surfaces or layouts.
4. An Extend means run more. If the user still wants to ship, the benefit they name goes in the evidence.
5. Only the user makes it the default. On their word: `python scripts/status.py --set-default ID --evidence "<sessions, scores, benefit>" --why "..."`, then update `ROADMAP.md`'s status in a branch.
