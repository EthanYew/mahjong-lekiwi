---
name: should-ship
description: Decide whether a policy should become the default. Use when the user asks to ship, compare or make a policy the default, which policy is better, or whether a policy is ready for a hand of play.
---

Apply `docs/scorecard.md`; don't invent a new standard for this decision.

1. Choose the check by the question. Is it a better base to keep building on: the iteration check. Is it worth spending scored hands on: the play-ready check. A claim that one policy is better, or replacing a play-ready policy: the strict paired rule.
2. Find a session that fits it, on the standard setup (`data/standard_setup.json`). If there's none, give the command to run. Don't decide from totals across sessions.
3. Run `python scripts/session_report.py <session>` and read the result, with the offline scores (`data/offline_eval/`) as a screen only.
4. Report the result, the clean counts and any spot it names as worse. For the iteration check, ask what failure the version was meant to fix, and check it improved with no new failure type or freeze. Say what the session can't show, such as other surfaces.
5. An Extend is the rule's usual answer at these sizes, not evidence of no difference. If the user still wants to ship, the benefit they name goes in the evidence.
6. Only the user makes it the default. On their word: `python scripts/status.py --set-default ID --evidence "<sessions, scores, benefit>" --why "..."`, then update `ROADMAP.md`'s status in a branch.
