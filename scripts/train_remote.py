"""Start and monitor an ACT training run on a remote Windows/WSL machine.

Connection details live in data/train_machine.json (gitignored — copy from
data/train_machine.example.json). The remote runs Windows OpenSSH with WSL.
Training runs detached from the SSH session via Win32_Process, so the job
survives when SSH disconnects and Windows sleep is suppressed by a PowerShell
script.

Start a training run (detaches). A new version is the default's datasets plus the new recording: the
script stops with the --add flags to paste if the list leaves one of the default's out (make_policy.recipe_gap),
and prints make_policy.py's plan, frames and steps, before it connects:

    python scripts/train_remote.py --source <the default's source> --add <each of its --add> \\
        --add Yewy/<new recording> --augment --replace-mat 0.7 --private --version vN

Check whether a detached run has finished:

    python scripts/train_remote.py --status --version vN

Sync results back after a detached run (updates data/policies.json and
data/offline_eval/; prints what README.md still needs), then power both machines off:

    python scripts/train_remote.py --sync --version vN --shutdown-train --shutdown-local

--shutdown-train when starting powers the training machine off as training ends, before anything is synced.

Dry-run to see what would be generated without connecting:

    python scripts/train_remote.py ... --dry-run
"""
import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_policy  # noqa: E402 — the recipe check, shared so both scripts refuse the same runs
from mahjong import config  # noqa: E402

CONFIG_PATH = config.DATA / "train_machine.json"
EXAMPLE_PATH = config.DATA / "train_machine.example.json"
POLL_INTERVAL_S = 60


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config():
    if not CONFIG_PATH.exists():
        raise SystemExit(
            "data/train_machine.json not found.\n"
            "Copy data/train_machine.example.json to data/train_machine.json "
            "and fill in your machine's details."
        )
    cfg = json.loads(CONFIG_PATH.read_text())
    for key in ("ssh", "wsl_distro", "wsl_home", "windows_home", "repo", "lerobot"):
        if key not in cfg:
            raise SystemExit(f"data/train_machine.json is missing '{key}' (see data/train_machine.example.json)")
    return cfg


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

def run_script(cfg, version, make_policy_args, comment=""):
    """Content of the WSL bash script that runs make_policy.py."""
    log = f"{cfg['lerobot']}/outputs/train/{version}_make_policy.log"
    cmd = ("python -u scripts/make_policy.py "
           + " ".join(f"'{a}'" if " " in a else a for a in make_policy_args))
    lines = ["#!/bin/bash"]
    if comment:
        lines.append(f"# {comment}")
    lines += [
        "export PATH=\"$HOME/miniforge3/envs/lerobot/bin:$PATH\"",
        f"LOG={log}",
        f"cd {cfg['repo']}",
        f"rm -f ~/{version}_exit",
        f"{cmd} > \"$LOG\" 2>&1 < /dev/null",
        f"echo $? > ~/{version}_exit",
    ]
    return "\n".join(lines) + "\n"


def keepawake_script(cfg, version, shutdown_train=False):
    """Content of the Windows PowerShell keepawake script."""
    # Escape the path for WSL: convert backslashes, spaces quoted
    wsl_run = f"~/{version}_run.sh"
    lines = [
        "# Keep Windows from sleeping during training",
        "Add-Type -Namespace Win32 -Name PowCfg -MemberDefinition @\"",
        "    [DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint esFlags);",
        "\"@",
        # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
        "[Win32.PowCfg]::SetThreadExecutionState(0x80000001)",
        "",
        f"wsl.exe -d {cfg['wsl_distro']} -- bash \"{wsl_run}\"",
        "",
        "# Release",
        "[Win32.PowCfg]::SetThreadExecutionState(0x80000000)",
    ]
    if shutdown_train:
        lines += [
            "",
            "# Shut down after training",
            "shutdown /s /t 60 /c \"Training complete\"",
        ]
    return "\n".join(lines) + "\n"


def make_policy_args(args):
    """Build the make_policy.py argument list from train_remote's args."""
    parts = ["--source", args.source, "--version", args.version]
    for repo in args.add:
        parts += ["--add", repo]
    for repo in args.leave_out:
        parts += ["--leave-out", repo]
    if args.trim_narrowing:
        parts.append("--trim-narrowing")
    if args.augment:
        parts.append("--augment")
    if args.replace_mat:
        parts += ["--replace-mat", str(args.replace_mat)]
    if args.steps:
        parts += ["--steps", str(args.steps)]
    if args.private:
        parts.append("--private")
    if args.default:
        parts.append("--default")
    return parts


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def ssh_run(cfg, windows_cmd, check=True):
    """Run a command on the Windows SSH host (not WSL)."""
    result = subprocess.run(["ssh", cfg["ssh"], windows_cmd],
                            capture_output=True, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"SSH command failed:\n{result.stderr.strip()}")
    return result


def wsl_run(cfg, bash_cmd, check=True):
    """Run a bash command inside WSL on the remote machine."""
    return ssh_run(cfg, f"wsl -d {cfg['wsl_distro']} -- bash -c {_sh_quote(bash_cmd)}", check=check)


def _sh_quote(s):
    """Single-quote a string for a Windows SSH command line."""
    return "'" + s.replace("'", "'\\''") + "'"


def wsl_write(cfg, remote_path, content):
    """Write a file to a WSL path on the remote machine via SSH stdin."""
    # Use base64 to avoid quoting issues with the content
    import base64
    b64 = base64.b64encode(content.encode()).decode()
    wsl_run(cfg, f"echo {b64} | base64 -d > {remote_path} && chmod +x {remote_path}")


def windows_write(cfg, windows_path, content):
    """Write a file to a Windows path on the remote machine."""
    import base64
    b64 = base64.b64encode(content.encode()).decode()
    # PowerShell: decode base64 and write as UTF-8
    ps_cmd = (f"[System.IO.File]::WriteAllBytes('{windows_path}', "
              f"[Convert]::FromBase64String('{b64}'))")
    ssh_run(cfg, f"powershell -Command \"{ps_cmd}\"")


# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------

def cmd_start(args, cfg):
    version = args.version
    mp_args = make_policy_args(args)
    policy_id = args.policy_id or f"pickup_mat_cl_{version}"

    if args.dry_run:
        display_cfg = cfg or {k: f"<{k}>" for k in ("lerobot", "repo", "wsl_distro", "windows_home", "ssh")}
        run_sh_content = run_script(display_cfg, version, mp_args,
                                    comment=f"{policy_id}: started by train_remote.py")
        ps1_content = keepawake_script(display_cfg, version, shutdown_train=args.shutdown_train)
        log_path = f"{display_cfg['lerobot']}/outputs/train/{version}_make_policy.log"
        print(f"Would write ~/{version}_run.sh:")
        print(textwrap.indent(run_sh_content, "  "))
        print(f"Would write <windows_home>\\{version}_keepawake.ps1:")
        print(textwrap.indent(ps1_content, "  "))
        print(f"Would start detached via Win32_Process, log -> {log_path}")
        if args.shutdown_train:
            print("  --shutdown-train: machine shuts down 60s after training")
        if args.shutdown_local:
            print("  --shutdown-local: this machine would shut down after sync")
        return

    run_sh_content = run_script(
        cfg, version, mp_args,
        comment=f"{policy_id}: started by train_remote.py on {time.strftime('%Y-%m-%d')}"
    )
    ps1_content = keepawake_script(cfg, version, shutdown_train=args.shutdown_train)
    run_sh_remote = f"~/{version}_run.sh"
    ps1_remote = f"{cfg['windows_home']}\\{version}_keepawake.ps1"
    log_remote = f"{cfg['lerobot']}/outputs/train/{version}_make_policy.log"

    # Check the remote's data/ for uncommitted changes that would be overwritten
    result = wsl_run(cfg, f"cd {cfg['repo']} && git status --short data/", check=False)
    if result.stdout.strip():
        print(f"Warning: remote {cfg['repo']} has uncommitted data/ changes:")
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
        print("These will be overwritten by 'git pull'. Check the remote first if this matters.")
        answer = input("Continue? [y/N] ").strip().lower()
        if answer != "y":
            raise SystemExit("Aborted.")

    # Ensure remote is on the latest main
    print("Pulling latest main on the remote...")
    wsl_run(cfg, f"cd {cfg['repo']} && git fetch origin && git checkout -B main origin/main")

    # Upload scripts
    print(f"Writing {run_sh_remote}...")
    wsl_write(cfg, run_sh_remote, run_sh_content)
    print(f"Writing {ps1_remote}...")
    windows_write(cfg, ps1_remote, ps1_content)

    # Launch detached via Win32_Process (outlives the SSH session)
    print("Starting training (detached from SSH)...")
    ps_launch = (
        f"Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        f"-Arguments @{{CommandLine='powershell.exe -ExecutionPolicy Bypass "
        f"-File \"{ps1_remote}\"'}}"
    )
    ssh_run(cfg, f"powershell -Command \"{ps_launch}\"")

    print(f"\nTraining started. Log: {cfg['ssh']}:{log_remote}")
    print(f"Check progress:  python scripts/train_remote.py --status --version {version}")
    print(f"Sync when done:  python scripts/train_remote.py --sync --version {version}")
    if args.shutdown_local:
        print("\n--shutdown-local: will shut down this machine after sync.")
        print("Run the sync command manually when training is done, then it will shut down.")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def cmd_status(args, cfg):
    version = args.version
    exit_remote = f"~/{version}_exit"
    log_remote = f"{cfg['lerobot']}/outputs/train/{version}_make_policy.log"

    result = wsl_run(cfg, f"cat {exit_remote} 2>/dev/null; echo __NOFILE__", check=False)
    out = result.stdout.strip()
    if "__NOFILE__" in out and out.strip() == "__NOFILE__":
        print(f"Training for {version} has not finished yet (no exit file).")
    else:
        code = out.replace("__NOFILE__", "").strip()
        if code == "0":
            print(f"Training for {version} complete (exit 0). Ready to sync.")
            print(f"  python scripts/train_remote.py --sync --version {version}")
        else:
            print(f"Training for {version} exited with code {code}. Check the log:")
            print(f"  ssh {cfg['ssh']} \"wsl -d {cfg['wsl_distro']} -- bash -c 'tail -50 {log_remote}'\"")
        return

    # Also show tail of log if it exists
    tail = wsl_run(cfg, f"tail -20 {log_remote} 2>/dev/null || echo '(log not found)'", check=False)
    print("\nRecent log output:")
    for line in tail.stdout.strip().splitlines():
        print(f"  {line}")


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def cmd_sync(args, cfg):
    version = args.version
    policy_id = args.policy_id or f"pickup_mat_cl_{version}"
    exit_remote = f"~/{version}_exit"

    # Confirm training is done
    result = wsl_run(cfg, f"cat {exit_remote} 2>/dev/null || echo __NOFILE__", check=False)
    out = result.stdout.strip()
    if "__NOFILE__" in out:
        raise SystemExit(f"No exit file for {version} — training may still be running. "
                         f"Check with: python scripts/train_remote.py --status --version {version}")
    code = out.strip()
    if code != "0":
        raise SystemExit(f"Training for {version} exited with code {code}. Sync aborted.")

    # Read the new policy entry from the remote's policies.json
    remote_policies_path = f"{cfg['repo']}/data/policies.json"
    result = wsl_run(cfg, f"cat {remote_policies_path}")
    remote_reg = json.loads(result.stdout)
    if policy_id not in remote_reg["policies"]:
        raise SystemExit(f"'{policy_id}' not found in remote policies.json — did make_policy.py complete step 7?")

    # Merge the new policy entry into local policies.json
    local_reg = json.loads(config.POLICIES.read_text())
    if policy_id in local_reg["policies"]:
        print(f"Note: {policy_id} already in local policies.json — overwriting with remote version.")
    local_reg["policies"][policy_id] = remote_reg["policies"][policy_id]
    config.POLICIES.write_text(json.dumps(local_reg, indent=2) + "\n")
    print(f"Updated data/policies.json with {policy_id}.")

    # Copy offline eval JSONs
    remote_eval_dir = f"{cfg['repo']}/data/offline_eval"
    result = wsl_run(cfg, f"ls {remote_eval_dir}/{policy_id}@*.json 2>/dev/null || true", check=False)
    remote_files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
    if remote_files:
        local_eval_dir = config.DATA / "offline_eval"
        for remote_path in remote_files:
            fname = Path(remote_path).name
            content = wsl_run(cfg, f"cat {remote_path}").stdout
            (local_eval_dir / fname).write_text(content)
        print(f"Copied {len(remote_files)} offline eval file(s) to data/offline_eval/.")
    else:
        print("No offline eval files found on remote.")

    # Print README reminder
    readme_check = config.DATA.parent / "README.md"
    readme = readme_check.read_text()
    if f"`{policy_id}`" not in readme:
        print(f"\nREADME.md does not yet name `{policy_id}`.")
        entry = remote_reg["policies"][policy_id]
        notes = entry.get("notes", "")
        # Pull key facts from the notes string for the README line
        import re
        frames = re.search(r"(\d+) frames", notes)
        steps = re.search(r"(\d+) steps", notes)
        datasets = re.search(r"make_policy\.py from (.*?);", notes)
        hint = []
        if datasets:
            hint.append(f"data: {datasets.group(1)}")
        if frames:
            hint.append(f"{frames.group(1)} frames")
        if steps:
            hint.append(f"{steps.group(1)} steps")
        hint.append(f"trained {time.strftime('%Y-%m-%d')}")
        print("Add this to the policies list in README.md (then run the fast tests):")
        print(f"  `{policy_id}` ({'; '.join(hint)}),")
    else:
        print(f"README.md already names `{policy_id}`.")

    if args.shutdown_train:
        print("\nShutting down the training machine in 60 seconds.")
        ssh_run(cfg, 'shutdown /s /t 60 /c "Synced"', check=False)
    if args.shutdown_local:
        print("\nShutting down this machine in 60 seconds (Ctrl-C to cancel)...")
        time.sleep(60)
        subprocess.run(["sudo", "shutdown", "-h", "now"])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # Mode selectors (mutually exclusive with the training args)
    p.add_argument("--sync", action="store_true",
                   help="sync results for --version back from the training machine "
                        "(updates data/policies.json and data/offline_eval/)")
    p.add_argument("--status", action="store_true",
                   help="check whether a detached training run for --version has finished")

    # make_policy.py pass-through (required for starting a run, also needed for --sync/--status)
    p.add_argument("--source",
                   help="recorded dataset repo id (required to start training)")
    p.add_argument("--version", required=True,
                   help="e.g. v12; used in output names and to identify a run")
    p.add_argument("--policy-id", default=None,
                   help="registry id (default: pickup_mat_cl_<version>)")
    p.add_argument("--add", action="append", default=[], metavar="REPO",
                   help="another recorded dataset to train on (repeatable)")
    p.add_argument("--leave-out", action="append", default=[], metavar="REPO",
                   help="a dataset the default trained on that this run leaves out on purpose (repeatable)")
    p.add_argument("--trim-narrowing", action="store_true",
                   help="drop the opening gripper narrowing from each episode")
    p.add_argument("--augment", action="store_true",
                   help="colour jitter during training")
    p.add_argument("--replace-mat", type=float, default=0.0, metavar="FRACTION",
                   help="swap the black table top in this fraction of episodes, e.g. 0.7")
    p.add_argument("--steps", type=int, default=None,
                   help="training steps (default: ~131 passes over the data)")
    p.add_argument("--private", action="store_true",
                   help="upload dataset and policy as private")
    p.add_argument("--default", action="store_true",
                   help="make it the default policy")

    # Remote-run options
    p.add_argument("--shutdown-train", action="store_true",
                   help="starting a run: shut down the training machine 60s after training completes (--sync then "
                        "waits until it's switched on again); with --sync: shut it down once the files are copied")
    p.add_argument("--shutdown-local", action="store_true",
                   help="shut down this machine after the sync step completes")
    p.add_argument("--dry-run", action="store_true",
                   help="print the scripts that would be generated and uploaded; do not connect")

    return p.parse_args(argv)


def main():
    args = parse_args()

    if args.dry_run and (args.sync or args.status):
        raise SystemExit("--dry-run is only for starting a training run, not --sync or --status")

    if args.sync or args.status:
        cfg = load_config()
        if args.sync:
            cmd_sync(args, cfg)
        else:
            cmd_status(args, cfg)
        return

    if not args.source:
        raise SystemExit("--source is required to start a training run "
                         "(or use --sync/--status to manage an existing one)")

    # Before anything connects: the dataset list against the default's, then the plan with its frame and step
    # counts. The scripts alone don't show a run's size; v12 was started from them at a sixth of v11's frames.
    make_policy.check_recipe(args.source, args.add, args.leave_out)
    if args.shutdown_train:
        print("Note: --shutdown-train powers the training machine off when training ends, and --sync needs it on. "
              "To copy the results back and then power it off, leave it out here and run "
              f"--sync --version {args.version} --shutdown-train when it's done.\n")
    local_plan(args)
    if args.dry_run:
        cmd_start(args, cfg=None)
        return

    cfg = load_config()
    cmd_start(args, cfg)


def local_plan(args):
    """make_policy.py's own dry run, here: the episodes, frames and steps the remote run would train on. Needs the
    recordings on this machine (they're recorded here); without them it says so and carries on."""
    command = [sys.executable, str(Path(__file__).with_name("make_policy.py")),
               *make_policy_args(args), "--dry-run"]
    print("The plan, from make_policy.py --dry-run on this machine:")
    result = subprocess.run(command, capture_output=True, text=True)
    print(textwrap.indent((result.stdout + result.stderr).strip(), "  ") + "\n")
    if result.returncode != 0:
        print("  (it couldn't plan here: check the recordings are on this machine before trusting the size)\n")


if __name__ == "__main__":
    main()
