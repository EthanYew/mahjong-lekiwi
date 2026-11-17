# mahjong-lekiwi

SO101 arm on a LeKiwi base picking up mahjong tiles: RealSense detects a tile → closed-loop IK hover
over it (measured correction, settled from above) → an ACT policy grasps from the wrist camera.
Rebuilt 2026-10-23 from `~/Projects/MahjongBot` (archive; don't import from it).

## Running

- Python: `~/miniforge3/envs/lerobot/bin/python`, from the repo root (scripts add it to `sys.path`).
- Fast tests (no hardware): `python -m pytest -m "not slow" -n 8`. Everything: `python -m pytest` (slow = loads policies,
  needs the GPU). A pre-commit hook runs the fast tests; turn it on once per clone with `git config core.hooksPath scripts/hooks`.
- **Where things stand**: `python scripts/status.py`. It reads `data/`; trust it over any doc. The default changes only
  with `status.py --set-default ID --evidence ...`.
- Commands and options: `README.md`. What each file is for: `docs/code_map.md`.
- Run hardware scripts from the main checkout: `sessions/` is per-checkout and gitignored. Read a session with
  `python scripts/session_report.py [folder]` before asking what happened.
- A command that may run an arm script prompts for a person, in auto mode too (`.claude/hooks/arm_run_gate.py`); reads,
  `--sim`, `--help`, linters and flags that never connect don't (a prompt on those is a hook bug: `tests/test_arm_run_gate.py`). A new script that moves the arm goes in its `ARM_SCRIPTS`.

## Rules for each part of the code

Each invariant cost a debugging round on hardware. They live in `.claude/rules/` and load when you read a file they cover.
**Writing a new script, or working from memory of a file: read the rule first.** All motion goes through `mahjong/arm.py`.

| Rule | Covers |
|---|---|
| `motion.md` | never start from the measured pose, arrive from above, hold the gripper, rest, lift, travel in play, reach limits |
| `returns.md` | how far to open, where to let go, aiming at home, landings, carried tiles, squeeze, reach on a return |
| `gamepad.md` | stick axes and speed caps, what hand corrections record |
| `policy_data.md` | DAgger, jitter, joint deltas tied to a hover, idle frames, lighting, surface, closing high, eval episodes never trained on |
| `perception.md` | setup check, wrist camera loss, wrist look, presence vs name, pale mats, the detector's exam |
| `play.md` | autonomous play, stricter reads, second read, votes, a person's word, the window, the drop zone |
| `bench.md` | sim runs, sessions, interventions, unattended runs, what counts as a bench result |
| `face_down.md` | the wall station, the base camera's reader and its controls, face blurring |
| `merging.md` | the full merge and closing rules, and the incidents behind them |

Learned something about the hardware? Add it to the rule it belongs to, with the reason; full evidence goes in
`docs/invariant_notes.md`. Keep this file short: `tests/test_docs.py` fails it past its size limit.

## Conventions

- Before calling a change hardware-ready: fast tests pass (slow too if `policy.py`/`dataset.py` changed), and say what was
  checked offline vs what still needs the arm.
- **`README.md` is the user's command reference.** Before merging, update it for any change to what a person runs: a script,
  an option, a default, menu commands or keys, new policy ids. `tests/test_docs.py` checks it.
- **State lives in `data/`, not in the docs.** Don't write "the default is X" where a file decides it; point at `status.py`.
  ROADMAP.md's status table is the one place for what's done and next, each row with its evidence; update it in the branch
  that changes the state.
- **A ship decision is recorded where it's made**: `status.py --set-default ID --evidence <sessions and scores> --why "..."`,
  a row in ROADMAP.md's status, and a retrospective if the run taught something.
- Commits: one logical change each, message says why; push to `origin` (private GitHub).

## Worktrees and merging

Git catches overlapping lines, not overlapping intent. `merging.md` has the detail.

- Parallel work: a helper, constant or option several tasks need lands on `main` first. Two branches shouldn't both change
  `bench.py`, `record.py` or `config.py`; `.claude/hooks/overlap_warning.py` says so after the first edit to a file another open branch changes. In a branch, touch this file, ROADMAP.md and README.md only for that branch's items.
- **Handing a branch back is `/hand-back`**: merge `main` in, ruff and the tests (the motion watchdog checks every sim arm command),
  `/code-review` (one medium pass, then low on the fixes), a ready PR with `## What changed`, `## Files`, `## Overlaps`, `## Review`, `## Unverified on the arm`,
  then the queue, `python scripts/merge_pr.py --auto N` from the main checkout, which merges it, pulls and closes the branch.
- **Only decisions wait for the person** (`needs-owner`: the default policy, eval/exam splits, the standard setup, the scorecard,
  installed weights; `merge_pr.HOLD_FILES`); they merge those with `merge_pr.py N`. Merge only through the queue, never `gh pr merge`
  or a push to `main`; never `--no-verify`.
- A new user-run script gets an end-to-end sim test (`tests/test_bench.py`, `tests/test_record.py`).
