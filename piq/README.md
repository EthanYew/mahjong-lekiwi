# piq: a local model for read-only questions about this repo

`piq` runs pi (a small coding-agent harness) against a local Ollama model (gemma4 12B) with read-only tools (read, grep, find, ls). It is for
quick questions ("what does `--carry` do?", "what's still needed to clear v14?") that don't need Claude Code. It can't run, edit or move anything.

## What is in this folder, and what keeps it true

| File | What it is | Kept true by |
|---|---|---|
| `prompt.md` | the rules the model is given every time: four HARD RULES, then style and scope | `tests/test_piq.py`: the hard rules are still there; every path it cites exists |
| `skills/*/SKILL.md` | task recipes the model can load (`answer-from-docs`, `review-bench`, `check-ship-readiness`, `explain-error`, `brief-for-claude`) | `tests/test_piq.py`: cited paths and sizes, quoted headings still exist somewhere in the docs, rules or scripts |
| `index.py` | at every launch writes a symbol index of the big files and **generated facts**: each script's options (helpers like `takeover.add_argument` included) and menu commands, read from the code | `tests/test_piq.py`: the generated options equal each script's own `--help` |
| `regression-questions.json` | questions the model must still answer well: two false-premise, one true behaviour, one real option; each with a *premise* about the repo | `tests/test_piq.py`: a flipped premise fails the branch that flipped it |
| `piq` | the launcher | `piq-check` hashes it |
| `check.py` | the checks a test can't do: Ollama's real limits, base model and pi versions, and the model's own answers (`piq-check --ask`) | run by `/hand-back` when a branch touches `piq/` |

Facts about the code are **generated, never written by hand** in `prompt.md` or a skill: a hand-written list of flags is wrong the first time a script changes.

## When you change the repo

- A flag or menu command: nothing to do. The facts are regenerated at launch and `tests/test_piq.py` compares them with `--help`.
- A doc file or heading that `piq/` cites, or a size it quotes: update `piq/` in the same branch. The test names the file and the line's claim.
- You add the option a false-premise question was about (say, a pause command): the test fails with the question's id. Turn it into a question about the real option, or retire it.
- You change `prompt.md`, a skill, the Modelfiles or the model: run `piq-check --ask` (about 3 minutes, needs Ollama). It stamps the setup as verified.

## Install (once per machine)

`piq` and `piq-check` are symlinks into the main checkout, so they follow `main`. Run `piq` from any checkout or worktree: it uses that checkout's `piq/` folder.

```
ln -sf <main checkout>/piq/piq ~/.local/bin/piq
ln -sf <main checkout>/piq/check.py ~/.local/bin/piq-check
```

pi's own settings (default model, compaction, thinking level) are in `~/.pi/agent/settings.json` and the model sizes in `~/.pi/agent/models.json`. Both stay on the
machine. The Modelfiles (`~/.ollama/Modelfile.gemma4-64k`, `-32k`) must set `num_ctx` (and pi's `contextWindow` the same), `num_predict` and `repeat_penalty`;
`piq-check` fails if Ollama runs something else.

## Why each guard exists

On 2026-11-22 a bench question made a 12B model read three 100 KB files and reason in a loop until the window was full (nine minutes, then an error). With the loop
capped, the next false-premise question got an invented `--start-at` flag. Both were a stale or missing fact meeting a question with no answer in the docs. The
hand-written bench facts were replaced by generated ones, the "absence is an answer" rule moved to the top of the prompt, and the rest is tested here.
