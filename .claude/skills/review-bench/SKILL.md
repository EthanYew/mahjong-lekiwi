---
name: review-bench
description: Review a bench session on the arm. Use when the user says they ran the bench or asks what happened in a session, what failed, or to review the latest data.
---

Report what the session shows and what it can't. Read before asking the user what happened.

1. Run `python scripts/session_report.py [folder]`, from the main checkout (sessions are per-checkout). No folder means the latest.
2. Lead with the scorecard: clean attempts and their 95% range, then picked. A session pairing two policies has a verdict.
3. Open the contact sheet it names, and look at the attempts tagged KNOCK, HIT or RESCUE. Group failures by cause and spot, and name the attempt folders.
4. Compare only within paired sessions. Use `session_report.py --all` to spot a change in setup (light, surface, layout), not to rank policies.
5. Check what the user said against the logs. If they differ, or a press looks wrong, say so; a person can misremember a batch.

End with the single next step and what it would tell us. Update `ROADMAP.md` only if the state changed, and only in a branch.
