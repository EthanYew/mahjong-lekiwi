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
  bumped another tile, `p` pushed its own, `r` moved by hand, `d` / `u` gripped low / high, `l` look; → marks a moment).
  Scans around each attempt catch the rest: a neighbour that moved is a knock if `b` was pressed, else asked then; other
  changes are settled by the next scans. No questions wait for the end of a batch. "Touched the tile" is a rescue; "the
  arm bumped a tile, no one helped" is a failure. The scorecard's numbers: `docs/scorecard.md`. Rules and cases:
  `docs/invariant_notes.md`.
- Unattended: `bench.py --unattended --layout NAME --trials N` asks nothing once the trials start. A neighbour the scans
  see move is a self-reported bump (settled by later scans; counts as a knock); a person checks them afterwards with
  `python scripts/bump_report.py`, whose verdicts (`scan_verdict` events) `interventions.derive` applies. README has what
  replaces each question. A new question in the bench's trials path needs an unattended answer: `tests/test_bump_report.py`
  fails any `input()`.
- Hardware scripts connect everything first (`mahjong/hardware.py`: retry / quit if something's
  unplugged), then ask the setup questions, then create the session folder.
- **Counted bench results run on the standard setup** (`data/standard_setup.json`: one mat, one layout), with a passing
  `setup_check.py` and one hover opening within a comparison. Other surfaces are generalisation checks and never block.
  Which check answers which question: `docs/scorecard.md`. Don't change the setup to explain away a result.
