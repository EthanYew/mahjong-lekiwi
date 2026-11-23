---
name: check-ship-readiness
description: List what is still outstanding before a policy can become the default, using the scorecard and the paired bench report. Use when the user asks what tests or reviews are left to clear a policy (v14, v15...) as the default, whether a bench result is enough, or what a session means for shipping. Read-only; the ship decision is the user's.
---

You gather evidence and sort it into what is computed, what the user must judge, and what is still to run. You never decide.
Read-only: you can't run commands or edit files. Apply `docs/scorecard.md`; don't invent a standard.

## 1. Get the two reports, from the user, one at a time
The user may not know which session to use. Work it out for them. Each command goes in its own message, and the `!` must be the very first character of the line (no leading space, one command per message). Give them the exact line to paste.

**First,** if status.py output isn't in this conversation, stop and ask for:
`! ~/miniforge3/envs/lerobot/bin/python scripts/status.py`
(the current default, per-setup clean counts, owed list). Wait.

**Then pick the session yourself from that output.** In the policy table, the "last bench" column gives a session id for the candidate. Add `_bench` to get its folder (`20261120_181225` is `sessions/20261120_181225_bench`). Check it is a pairing: the default's row should show the same id (or the user says so). If both rows show the same id, it is the paired session; say "I'm using <folder> because both <candidate> and <default> list it as their last bench". Then ask for:
`! ~/miniforge3/envs/lerobot/bin/python scripts/session_report.py sessions/<folder>`

**If the candidate's last bench differs from the default's, or the candidate shows `-` / `never`,** don't guess. Say what you see (for example "v14's last bench is X, v13's is Y, so I can't tell which is the paired run"), and ask the user to run `!ls -t sessions | grep _bench | head -8` (newest first) and say which one was the v13-vs-candidate run. A session that benched the candidate alone is not a pairing and can't give an iteration-check verdict: say so.

Do not read `status.py`'s source, `data/policies.json` (37 KB) or list `sessions/`: they don't answer the question and cost ~15k tokens.
State lives in `data/`; take the default and the numbers from these two reports, never from memory or from an old note.

## 2. Read the rules, narrowly
`docs/scorecard.md` is long. `grep -n "^## \|^###" docs/scorecard.md`, then `read` about 40 lines each of: "Which check answers which question",
"The standard setup", "Two counted setups" (the face-up round), "The iteration check", "The play-ready check". Pick the check by the question:
changing the default = the **iteration check**; claiming "better" = the strict paired rule; spending scored hands = the play-ready check.
To find what the candidate was built to fix: `grep -n "<candidate id>" ROADMAP.md` and read that row.

## 3. Report what the session computed
Quote exactly from `session_report.py`: clean counts and ranges for both policies, the iteration-check line, the strict-paired line, the play-ready lines. An EXTEND on the strict rule is the usual answer at these sizes, not evidence of no difference.

## 4. List what is outstanding, in this order, each with its source
**a. Is the session countable?** Standard mat and layout (the report says if not); the setup check passed (sessions don't log it: the user's step, so ask); one hover opening in the pair; session not voided; after an unattended run, bumps checked with `bump_report.py`.
**b. Face-up round.** A candidate benched only face down needs one paired round against the face-up best, face up, that doesn't end in Drop: `bench.py --policy <face-up best>,<candidate> --layout even-hand14 --trials 1`. Look at status.py's `face up <candidate>` line first; if it exists, say what it shows.
**c. The user's judgment calls** (the iteration check computes only the clean count): did the named failure improve; is there a new failure type; any freeze. Hand them the facts: failures and knocks by spot, and the close-height table (median mm, how many at 13 mm+, spots where it closed higher). Say plainly where the candidate looks worse, even when it is ahead on clean.
**d. The decision and the record.** Only the user decides. On their word: `status.py --set-default ID --evidence "<sessions, scores, benefit>" --why "..."`, then ROADMAP.md's status in a branch (Claude Code does that).

## 5. Not requirements. Don't put these in the checklist
- status.py's "Owed and overdue" list (unmerged branches, `save_samples.py`, Unconfirmed invariants): housekeeping. At most one line: "housekeeping, not needed to clear it".
- The face-down hands tally ("10 hands, 8+ clean"): the play-readiness proof. The scorecard says the default is "the working best" and says nothing about being ready to play.
- The offline close-timing gate: report-only since 2026-11-17. Offline scores are a screen only; say whether `data/offline_eval/<id>*` exists and whether it shows a freeze.
- The default's old decision note (`evidence` / `why`): dated text. "No face-up bench yet" was true on the day it was written; check it against today's status lines before repeating it.

## 6. Answer in this shape
**Result:** start with both policies' clean counts as the report prints them (for example "v14 28/28 clean, v13 25/28"), then **By check:** iteration / strict paired / play-ready, each one line. **Still needed:** numbered, each tagged *computed*, *your call* or *to run*, with source; the last item is always "your decision, then `status.py --set-default` with evidence" (item d above). **Can't tell:** what the logs don't show (other surfaces, whether a setup check was done). Then stop; no verdict.

Before you answer, check: Did I state the candidate's actual result first? Is every "required" item in the scorecard, not just in status.py? Did I read or invent nothing about state? Did I label each item?

## A reviewed bad answer (2026-11-22), so you don't repeat it
Asked "what is outstanding to clear v14 as the default" after a face-down paired bench against v13, a local model answered from `status.py` alone, read nothing else, and:
- never said v14's result, which was the point of the question (28/28 clean against v13's 25/28; iteration check AHEAD 3-0; strict paired EXTEND p 0.25; play-ready face down PASS, with the setup check still the user's);
- listed a face-up bench "to establish a baseline" quoting a stale note: the action was right (the scorecard requires the face-up round) but the reason and the source were wrong;
- told the user to pass the offline close-timing gate, which is report-only, and v14's offline screen already existed;
- listed the 11 Unconfirmed invariants and `save_samples.py` as ship requirements: housekeeping;
- counted the face-down hands tally as a requirement: it is the play-ready proof;
- missed the real review item: v14 closed higher (median 9.4 mm against 8.0, six attempts at 13 mm+ against none), a possible new failure type for the user to judge.
Its quoted figures were accurate; what failed was relevance, staleness and the missing result. A good answer opens with the result, lists three to five items that each trace to the scorecard, and names the judgment calls with their facts. (Those numbers are from that day; always take current ones from the reports.)
