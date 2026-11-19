---
name: write-retro
description: Write up a run of work as a retrospective. Use when the user says document this for posterity, write this up, or asks for a retrospective.
---

Write `docs/retrospectives/<date>_<topic>.md` in the main checkout, not a worktree: the folder is gitignored and kept out of the
public repo, so a retrospective written in a worktree is lost when the worktree goes, and none is committed. Retrospectives feed the owner's public writeup, so a reader must be able to check every claim.

**Structure:** one paragraph saying what happened and what it means, then **What changed / What was learned / What's next**, with a timeline and the session folders and PRs it draws on. Define a term the first time you use it. `docs/retrospectives/2026-11-04_first_hands.md` is a good model.

**Every claim is traceable:**
- Each number names its source: a session folder, a commit, a PR or a data file. Recompute it from that source rather than copying from memory or another doc. If a number took a script to get, save the script under `docs/retrospectives/evidence/<date>_<topic>/`.
- A number you can't trace gets cut, or labelled as an estimate.
- Separate what was measured from what's judgement, and what's checked offline from what's been seen on the arm.
- Give sizes with results: 22/22 isn't 100%. Include what failed and what's still unknown, not just the wins.

**Who did what:** say what the owner decided, and what Claude built or found. Quote the owner's own words for a decision or a correction when they matter. Don't credit the owner with analysis Claude did, or Claude with a call the owner made. If Claude wrote it, say so at the top, and ask a person to check the conclusions.

**State lives in `data/`:** don't write "the default is X" as a standing fact; date it. If the run changed the state, update `ROADMAP.md`'s status row. If it taught something about the hardware, add the rule to `.claude/rules/`.

Don't touch other retrospectives: they're a record of what was believed then. If this one overturns an earlier claim, say so here and link it.
