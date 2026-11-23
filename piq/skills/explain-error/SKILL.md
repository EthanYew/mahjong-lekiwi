---
name: explain-error
description: Explain a pasted error message, traceback or failing test output from this repo, by finding the code that raised it. Use when the user pastes a traceback, an exception, a pytest failure or a log line and asks what it means or why it happened.
---

Explain what went wrong and where. Read-only: you can't run anything or change anything, and you must not present a
guess as a diagnosis.

1. **Find the frame that matters.** In the traceback, take the innermost frame inside this repo (paths under `mahjong/`
   or `scripts/`; skip `site-packages`, lerobot, torch, pyrealsense2). Note the exception type and message, quoting its numbers exactly.
2. **Read that code, narrowly.** `grep -n` the function in that file (or use the symbol index), then `read` about 40 lines
   around the line. Also read the caller one frame up if the message alone doesn't explain the input.
3. **Check for a written rule.** Find the area in CLAUDE.md's rules table (arm.py → `motion.md`, a policy → `policy_data.md`, a
   camera → `perception.md`) and `grep -n` that file for the function or term. Many failures here are guards that were added on purpose after a debugging round,
   and the rule gives the reason. Quote it if it applies.
4. **Answer in this shape:**
   - **What raised it:** the file, line and condition, in one or two sentences.
   - **Why, from the values in the message:** use the actual numbers or names pasted. If the values don't settle it, say so.
   - **Likely causes**, at most three, most likely first, each marked *read from the code* or *guess*.
   - **What to check next:** commands for the user to run themselves (`!command`), nothing that moves the arm.
5. **Hardware errors** (serial port, torque, camera timeout, a motor not responding): say what the code does when it
   sees the error, then stop. Don't speculate about wiring or power. Suggest `/debug-arm` in Claude Code.
6. Don't propose a patch. If the fix is clear, describe it in a sentence and say to ask Claude Code for the change.
