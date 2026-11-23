---
name: brief-for-claude
description: Compress pasted logs, a session report, a diff, an error or a long local conversation into a short brief the user can paste into Claude Code. Use when the user says to summarise this for Claude, write a handoff, or prepare context to take to the frontier model.
---

The point is to spend fewer tokens in Claude Code without losing what it needs. You are the filter, not the analyst.
Read whatever the user pointed at (paste, files, or this conversation), then write the brief.

**Format.** Put the brief in one fenced block so it can be copied, under 300 words:

```
Goal: <one sentence: what I'm trying to find out or change>
Observed:
- <fact, with file:line or the exact log line>   (at most 6)
Ruled out: <what was checked and found fine, or "nothing yet">
Ask: <one specific thing for Claude to do>
Suggested skill: <e.g. /debug-arm, /review-bench, /should-ship, or none>
```

**Rules**
1. Copy error lines, numbers, scores, paths and session folder names exactly. Never round, reword or recompute a number.
2. Only put in "Observed" what the material shows. Anything you inferred goes on its own line beginning `Guess:`, so
   Claude knows to check it. A local 12B model summarised this; say so in one short line at the end: `(summarised locally; verify)`.
3. Leave out anything Claude can get itself from the repo: file contents, what a rule says, what `CLAUDE.md` already covers.
   Include a code excerpt only if it is under 20 lines and the point depends on it.
4. Never include secrets or machine details: API keys, tokens, SSH addresses, hostnames, usernames, the training PC's details.
   The repo is public. Replace them with `<redacted>`.
5. If the material is too thin to brief (no error text, no numbers), say what is missing and ask the user for it instead of padding.
6. One ask only. If there are several, pick the one that unblocks the rest and list the others under it as "Later".

After the block, say in one line what you left out on purpose, so the user can add it back.
