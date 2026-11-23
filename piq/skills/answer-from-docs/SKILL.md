---
name: answer-from-docs
description: Answer "according to my docs" or "what does the documentation say" questions about this repo, such as what a rule requires, how a command is used, or what is still needed before a decision. Use when the answer is in CLAUDE.md, .claude/rules/, docs/ or ROADMAP.md rather than in code.
---

Answer from the written docs, and show where each statement came from. Read-only.

1. **Pick at most two sources.** CLAUDE.md is already in your context; its table says which `.claude/rules/*.md` covers
   which topic. Other places: `docs/usage.md` (commands and options), `docs/code_map.md` (what each file is for),
   `ROADMAP.md` (what is done and next), `docs/scorecard.md`, `docs/autonomy_plan.md`, `docs/invariant_notes.md` (evidence behind rules).
2. **Read the section, not the file.** `docs/usage.md` is 140 KB and `ROADMAP.md` is large. Use the symbol index
   (`grep -n "docs/usage.md" <index>`) or `grep -n` for a keyword to find the heading, then `read` about 60 lines from there.
   Rule files are short enough to read whole.
   **Never `read` a whole script to look for an option.** Options are in `docs/usage.md` under the script's own heading (the table after
   the examples) and in the script's `add_argument` lines: `grep -n 'add_argument' scripts/<name>.py` lists them all in one call.
   A 100 KB file read costs ~15k tokens of a 65k window; three of them end the conversation.
2b. **Decide "it doesn't exist" early.** Search the usage heading and the `add_argument` list once each. If the thing asked for
   (pause, resume, start at a spot, a flag) is in neither, it does not exist: say so, name the nearest real option, and stop.
   Do not open other scripts hoping to find it, and do not re-read or re-weigh. If a similar name exists in another script
   (`--resume` is `play.py`'s, not `bench.py`'s), say which script has it and that the one asked about does not.
3. **State lives in `data/`, not in the docs.** If the question is "where do things stand", "what is the default policy"
   or "what has been run", don't answer from docs. Ask the user to run
   `!~/miniforge3/envs/lerobot/bin/python scripts/status.py` and answer from that output.
4. **Answer in this shape:**
   - the answer in a few sentences, or a checklist if the question is "what else do I need to do";
   - **Source:** one bullet per claim, with path, heading and line;
   - **Not documented:** anything the user asked that the docs do not say. Say so; do not fill the gap from general knowledge;
   - a line marked `Unconfirmed` in a rule is not settled: say it is unconfirmed.
5. For a "do I need more tests before X" question, separate what the docs require from what you only suspect, and
   never say something has passed: you can't see the results, only what the docs ask for.

You can't run commands or edit files. If the answer needs a change to a rule or a run on the arm, say so and leave it for Claude Code.

## A reviewed bad answer (2026-11-22), so you don't repeat it
Asked "what's the command to pause and resume a bench session, or start one from a specific spot in a layout", a local model
(gemma4 12B, 64k window) read `docs/usage.md`, `scripts/bench.py` and `scripts/play.py` whole (about 60k tokens), then produced
no answer: one turn of 164,000 characters of reasoning that repeated the same five lines 319 times, until the window was full and the
session errored after nine minutes. The answer was short and was in the docs: `bench.py` has no pause, resume or start-at-spot;
`quit` (or Ctrl-C at the menu) ends it, Ctrl-C mid-command returns to the menu, `--layout` walks every spot, and `--resume` belongs to `play.py`.
What failed: whole-file reads, no rule for concluding that a feature is absent, and reasoning that kept re-checking an unanswerable premise.
A good answer is two or three tool calls (grep the heading, read one table) and a plain "there is no such option; here is the nearest one".
If you notice yourself writing "Wait," and restating a finding you already have, stop and answer.
