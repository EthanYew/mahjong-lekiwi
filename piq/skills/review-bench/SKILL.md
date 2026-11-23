---
name: review-bench
description: Review a bench session on the SO101 arm (mahjong-lekiwi). Use when the user says they ran the bench, or asks what happened in a session, what failed, or to review the latest data.
---

Report what the session shows and what it can't. This is read-only: you have no way to run commands or edit files.
Anything that would change a file is a suggestion for the user to take to Claude Code.

The session, and what the user said about it, come after `/skill:review-bench`.

1. **Get the report.** If `python scripts/session_report.py` output is not already in this conversation, stop and ask
   the user to run it themselves:
   `!~/miniforge3/envs/lerobot/bin/python scripts/session_report.py [folder]`
   (no folder means the latest; run from the main checkout, because `sessions/` is per-checkout). Wait for it.
2. **Lead with the scorecard:** clean attempts and their 95% range, then picked. A session pairing two policies has a verdict.
3. **Look at the failures.** The report names a contact sheet (an image). Read it only if you can see images, and say
   what is clearly visible, not what you guess. Group attempts tagged KNOCK, HIT or RESCUE by cause and spot, and name
   the attempt folders. Never read `events.jsonl` whole; use `grep` on the one attempt folder.
4. **Compare only within paired sessions.** `session_report.py --all` shows a change in setup (light, surface, layout);
   it is not for ranking policies. If you need it, ask the user to run it with `!`.
5. **Check what the user said against the logs.** If they differ, or a press looks wrong, say so; a person can misremember a batch.
6. **Unconfirmed rules.** `grep -rn "Unconfirmed" .claude/rules/`. List each line this session settled, with its evidence,
   and propose the wording change. Most sessions settle none: say so in one line.

Rules for this skill: say what the logs show, never what the hardware "probably" did; if the report is missing a number,
say it is missing. Keep reads small (`grep -n`, then `read` with offset and limit); the report is already the summary.

End with the single next step and what it would tell us. If `ROADMAP.md` or a rule should change, say so and leave the
edit for Claude Code in a branch.
