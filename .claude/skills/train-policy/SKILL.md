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

1. **Settle the options here, starting from the default's recipe.** A new version is the default's datasets and options plus the
   new recording, never a README example: v12 (2026-11-17) copied one and trained on 15k frames where v11 had 88k, so it couldn't
   be compared with v11 and the run was thrown away. The default's `notes` in `data/policies.json` start with its datasets
   (`make_policy.py from A + B + ...`; newer entries have `datasets`) and name its options (`mat swapped in 70%` is `--replace-mat 0.7`,
   `colour jitter` is `--augment`). Then `python scripts/train_remote.py --source A --add B ... --add <new> --version vN --dry-run`
   (no connection). It stops if a dataset of the default's is missing and prints the `--add` flags; `--leave-out REPO` only when
   the user wants one dropped. Its plan, make_policy's own dry run, gives the frames and steps next to the default's: a new version
   with more data should be no smaller, so a drop means something is missing. The dataset name is fixed once uploaded, and uploads
   are public under `Yewy/` unless `--private`. Show the user the plan and the one thing that changed from the default (the new
   recording, usually), and get a yes.
   The PC trains from `origin/main` (the script runs `git checkout -B main origin/main` there), so anything `make_policy.py` reads
   (`data/dropped_episodes.json`, `data/eval_split.json`) must already be pushed to `main`.
2. **Sync the recordings** the PC lacks. `train_remote.py` doesn't: a local recording (no `Yewy/` upload yet) exists only here.
   On the PC, in `~/.cache/huggingface/lerobot/Yewy`: `rsync -a -e "ssh -o BatchMode=yes" <this machine>:~/.cache/huggingface/lerobot/Yewy/{a,b} ./`,
   then compare `wc -l */episode_log.jsonl` with here.
3. **Start it**: the same command without `--dry-run`. It warns if the PC's `data/` has uncommitted changes the pull would overwrite;
   say so rather than answering for the user. Shutting down: `--shutdown-train` at the start powers the PC off as training ends,
   before the sync can copy anything, so use it only when the results can wait until the PC is switched on again. "Shut both down
   when it's done" is step 5's `--sync --version vN --shutdown-train --shutdown-local`, once the PR is open.
4. **Check progress**: `python scripts/train_remote.py --status --version vN`. Report the step, the steps left and the time that
   implies. A failure is diagnosed from the log it names.
5. **Once it has finished**, in a branch (registering is a decision, so it doesn't go straight to `main`, and it isn't made the default):
   - `python scripts/train_remote.py --sync --version vN` brings back `data/policies.json` and `data/offline_eval/<id>@*.json`.
   - Add the line it prints to README.md's policy list, run the fast tests, and open a draft PR.
   - The offline screen is in those files: `python scripts/offline_eval.py --policy <id> --against <default>` for tables by slice. It screens; the bench decides.
6. End with the paired bench command against the default (`/should-ship` has the rule) and what result would mean what.
