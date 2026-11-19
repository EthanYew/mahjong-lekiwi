---
paths:
  - "scripts/bench.py"
  - "scripts/session_report.py"
  - "scripts/bump_report.py"
  - "scripts/status.py"
  - "mahjong/interventions.py"
  - "mahjong/scorecard.py"
  - "mahjong/session.py"
  - "mahjong/bump_report.py"
  - "tests/test_bench.py"
  - "tests/test_bump_report.py"
  - ".claude/hooks/arm_run_gate.py"
  - "tests/test_arm_run_gate.py"
---
Running and counting the bench, what a person did, what a session records.

- No-hardware run of the whole bench: `python scripts/bench.py --sim --trials 3 --tile 8D` (without
  `--tile` it asks which tile). Paired comparison across the mat: `--policy a,b --layout NAME --trials N`
  (`--trials` counts rounds: N at every spot, each policy once per round, alternating a b / b a from the
  same hover offset, so N × 2 × spots pickups; each attempt saves `wrist.mp4`).
- Hardware sessions write `sessions/<time>_<kind>/`; read one with `python scripts/session_report.py
  [folder]` (setup, success per tile/policy, stalls, clearances, ACT seconds, interventions, how far
  returns landed from home, contact sheet) before asking what happened; `--all` compares every bench session with its setup conditions.
  Run hardware scripts from the main checkout: `sessions/` is per-checkout and gitignored, so a
  worktree's sessions go with it.
- Interventions (`mahjong/interventions.py` has the classes and why, ROADMAP 3): keys pressed in the bench's own
  terminal while the arm picks up or returns say what happened (`interventions.KEYS`, printed as each starts: `b`
  bumped another tile, `p` pushed its own, `r` moved by hand, `d` / `u` / `s` gripped low / high / shallow (jaw tips only), `l` look; → marks a moment). The report's "where to record
  more" puts misses, these keys, high closes and camera losses per spot.
  Scans around each attempt catch the rest: a neighbour that moved is a knock if `b` was pressed, else asked then; other
  changes are settled by the next scans. No questions wait for the end of a batch. "Touched the tile" is a rescue; "the
  arm bumped a tile, no one helped" is a failure. The scorecard's numbers: `docs/scorecard.md`. Rules and cases:
  `docs/invariant_notes.md`.
- Unattended: `bench.py --unattended --layout NAME --trials N` asks nothing once the trials start. A neighbour the scans
  see move is a self-reported bump (settled by later scans; counts as a knock); a person checks them afterwards with
  `python scripts/bump_report.py`, whose verdicts (`scan_verdict` events) `interventions.derive` applies. `docs/usage.md` has what
  replaces each question. A new question in the bench's trials path needs an unattended answer: `tests/test_bump_report.py`
  fails any `input()`.
- Hardware scripts connect everything first (`mahjong/hardware.py`: retry / quit if something's
  unplugged), then ask the setup questions, then create the session folder.
- **A power cut damages only the end of a session**: on 2026-11-18 `20261118_230033_wrist`'s `events.jsonl` ended in 566 zero
  bytes and its last 9 photos were 0-byte files, and every script that lists sessions stopped on the log. So read logs with
  `session.read_jsonl` (it leaves out a zero-filled or cut-off tail and says so, but raises on damage before the last whole
  line), append with `session.append_jsonl` (it moves a damaged tail to `<name>.cut` first, or the next record merges with
  the cut line and is lost), and list photos with `wrist_photos.written` (whole JPEGs only); never a hand-rolled loop.
- **Counted bench results run on the standard setup** (`data/standard_setup.json`: one mat, one layout), with a passing
  `setup_check.py` and one hover opening within a comparison. Other surfaces are generalisation checks and never block.
  Which check answers which question: `docs/scorecard.md`. Don't change the setup to explain away a result.
- A command that may run an arm script prompts for a person, in auto mode too (`.claude/hooks/arm_run_gate.py`, a
  hook because no permission rule can exempt `--sim`); nothing else does: reads, `--sim`, `--help`, linters, flags that
  never connect (`wall_test.py --report`), command substitutions that only read, and `python -c` code or a helper script
  (e.g. a patch in the job folder) with nothing that could start a program. Only a likely real arm run should prompt; a
  prompt on anything else is a hook bug to fix in `tests/test_arm_run_gate.py`. A new script that moves the arm goes in
  its `ARM_SCRIPTS`.
