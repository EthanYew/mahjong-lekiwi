---
name: debug-arm
description: Diagnose odd behaviour on the arm from its logs, before touching code. Use when the user says it's sagging, drifting, it keeps missing, a tile moved, or they're finding something hard to diagnose.
context: fork
---

The symptom: $ARGUMENTS

Sag and drift took many rounds of PRs on 2026-10-28 because fixes came before a measurement. Change no code here. Never start an arm script; `--sim` only.

1. **Read the rules first.** The rule in `.claude/rules/` for this part (`motion.md`, `returns.md`, `gamepad.md`, `perception.md`), then `docs/invariant_notes.md` if one looks close. Many symptoms are a known invariant: the servos sit 5-13mm below command, far spots read low, lowering at full reach slides the gripper back. If it's one of those, say so and stop.
2. **Read the logs.** Sessions are in the main checkout's `sessions/`. `python scripts/session_report.py <folder>`; `python scripts/handoff_report.py` for anything under the gamepad; `--all` to see what setup changed. Write a script over `events.jsonl` rather than reading it raw.
3. **Build a timeline.** When did it start? Line up sessions against `git log` and `data/` changes, and the setup conditions each session recorded. Say which change came just before, without yet calling it the cause.
4. **Separate what's measured from what's commanded.** Which number shows the symptom, and where was it logged? Check the person's account against the logs; they can differ.

Return:
- **What the logs show**, with the session folders and numbers. Say what they can't show.
- **One hypothesis**, not a list, with the number it predicts.
- **One test on the arm**: the command, what to watch, how long, and the result that would rule the hypothesis out. It changes one thing.
- **The fix only if the evidence already settles it**, as a proposal.

If this teaches something new about the hardware, propose the line for the rule, with its reason.
