---
name: train-policy
description: Train a policy on the PC over SSH and follow it to a registered, screened policy. Use when the user says train it on the other machine, start training, has training finished, or how is v-something doing.
model: sonnet
---

Training runs on the PC (Windows, WSL inside) through `scripts/train_remote.py`, which detaches the run so it outlives SSH and
keeps Windows awake (README, "Training on a remote machine"). The machine's details are in `data/train_machine.json`, gitignored,
never in the repo, which may go public. No such file: copy `data/train_machine.example.json` and ask for the values (memory
`train-machine` has them).

Asked whether it's finished: skip to step 4. Check once per ask; don't sit in a session polling a run that takes hours.

1. **Settle the options here**: `python scripts/train_remote.py --source ... [--add ...] --version vN --dry-run` (no connection;
   prints the scripts it would upload). The options are `make_policy.py`'s. The dataset name is fixed once uploaded, and uploads are
   public under `Yewy/` unless `--private`. Show the user the plan and the one thing that changed from the last version, and get a yes.
   The PC trains from `origin/main` (the script runs `git checkout -B main origin/main` there), so anything `make_policy.py` reads
   (`data/dropped_episodes.json`, `data/eval_split.json`) must already be pushed to `main`.
2. **Sync the recordings** the PC lacks. `train_remote.py` doesn't: a local recording (no `Yewy/` upload yet) exists only here.
   On the PC, in `~/.cache/huggingface/lerobot/Yewy`: `rsync -a -e "ssh -o BatchMode=yes" <this machine>:~/.cache/huggingface/lerobot/Yewy/{a,b} ./`,
   then compare `wc -l */episode_log.jsonl` with here.
3. **Start it**: the same command without `--dry-run`. It warns if the PC's `data/` has uncommitted changes the pull would overwrite;
   say so rather than answering for the user. Overnight: `--shutdown-train` (and `--shutdown-local` only if the user asks).
4. **Check progress**: `python scripts/train_remote.py --status --version vN`. Report the step, the steps left and the time that
   implies. A failure is diagnosed from the log it names.
5. **Once it has finished**, in a branch (registering is a decision, so it doesn't go straight to `main`, and it isn't made the default):
   - `python scripts/train_remote.py --sync --version vN` brings back `data/policies.json` and `data/offline_eval/<id>@*.json`.
   - Add the line it prints to README.md's policy list, run the fast tests, and open a draft PR.
   - The offline screen is in those files: `python scripts/offline_eval.py --policy <id> --against <default>` for tables by slice. It screens; the bench decides.
6. End with the paired bench command against the default (`/should-ship` has the rule) and what result would mean what.
