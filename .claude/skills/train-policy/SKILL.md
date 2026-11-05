---
name: train-policy
description: Train a policy on the PC over SSH and follow it to a registered, screened policy. Use when the user says train it on the other machine, start training, has training finished, or how is v-something doing.
---

Training runs on the PC (Windows, WSL inside) so it can outlive the session. The address, distro and home folder are in memory
(`train-machine`), never in the repo, which may go public. No such memory: ask. Below, `<host>`, `<distro>` and `<home>` are those.

Asked whether it's finished: skip to step 5. Check once per ask; don't sit in a session polling a run that takes hours.

Every remote command takes this form, a script file on stdin (quoting through Windows and WSL breaks anything else):
`ssh -o BatchMode=yes -o ConnectTimeout=10 <host> 'wsl -d <distro> -- bash -l -s' < script.sh`

1. **Settle the options here**: `python scripts/make_policy.py --source ... [--add ...] --version vN --dry-run`. The dataset name is fixed once
   uploaded, and uploads are public under `Yewy/` unless `--private`. Show the user the plan, what one thing changed from the last version, and get a yes.
   Anything `make_policy.py` reads (`data/dropped_episodes.json`, `data/eval_split.json`) must already be on `main`, or the PC won't have it.
2. **Check the PC**: in `~/Projects/mahjong-lekiwi`, `git status --short` clean and `git pull --ff-only`, then `git log -1`; the GPU free
   (`nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`, `pgrep -af "make_policy|lerobot-train"`); `df -h ~`.
3. **Sync the recordings** the PC lacks, pulling from this machine, in `~/.cache/huggingface/lerobot/Yewy`:
   `rsync -a -e "ssh -o BatchMode=yes" <this machine>:~/.cache/huggingface/lerobot/Yewy/{a,b} ./`. Compare `wc -l */episode_log.jsonl` with here.
4. **Start it** as a scheduled task, so it outlives SSH. Write `~/run_vN.sh` on the PC:
   ```bash
   #!/bin/bash
   cd ~/Projects/mahjong-lekiwi
   echo "started $(date)" > ~/make_policy_vN.log
   ~/miniforge3/envs/lerobot/bin/python -u scripts/make_policy.py <the settled options> >> ~/make_policy_vN.log 2>&1 < /dev/null
   echo "exit $? $(date)" >> ~/make_policy_vN.log
   ```
   Then, on Windows and not inside WSL, dated far ahead so it can't fire again by itself:
   `ssh <host> 'schtasks /create /tn mahjong_vN /tr "wsl.exe -d <distro> -- bash -l <home>/run_vN.sh" /sc once /st 00:01 /sd 01/01/2030 /f && schtasks /run /tn mahjong_vN'`
   Wait a minute, and confirm it's running with step 5.
5. **Check progress**: `pgrep -af make_policy` and `tr '\r' '\n' < ~/make_policy_vN.log | grep -v "^Svt\|Fetching\|Loading" | tail -5`. Report the step, the steps left and
   the time that implies. A last line of `exit N` means it stopped; anything but `exit 0` is a failure to diagnose from the log.
6. **Once it exits 0**:
   - Delete the task: `ssh <host> 'schtasks /delete /tn mahjong_vN /f'`.
   - Bring back what it wrote on the PC (`data/policies.json`, `data/offline_eval/<id>@*.json`) into a branch here, name the policy in `README.md`, and open a draft PR.
     Registering is a decision, so it doesn't go straight to `main`, and it isn't made the default.
   - The offline screen is in those files: `python scripts/offline_eval.py --policy <id> --against <default>` for tables by slice. It screens; the bench decides.
7. End with the paired bench command against the default (`/should-ship` has the rule) and what result would mean what.
