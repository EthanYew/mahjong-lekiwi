"""train_remote.py: script generation, config loading, and sync logic (no SSH)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_remote


FAKE_CFG = {
    "ssh": "user@testhost",
    "wsl_distro": "Ubuntu-24.04",
    "wsl_home": "/home/user",
    "windows_home": "C:\\Users\\user",
    "repo": "~/Projects/mahjong-lekiwi",
    "lerobot": "~/Projects/lerobot",
}


def test_run_script_contains_make_policy_command():
    args = ["--source", "mahjong_surfaces_standard_mat_v1", "--version", "v99",
            "--add", "Yewy/dagger_v98", "--augment", "--trim-narrowing", "--private",
            "--dry-run"]
    parsed = train_remote.parse_args(args)
    mp_args = train_remote.make_policy_args(parsed)
    script = train_remote.run_script(FAKE_CFG, "v99", mp_args)
    assert "make_policy.py" in script
    assert "--source" in script and "mahjong_surfaces_standard_mat_v1" in script
    assert "--version" in script and "v99" in script
    assert "--add" in script and "Yewy/dagger_v98" in script
    assert "--augment" in script
    assert "--trim-narrowing" in script
    assert "--private" in script
    assert "v99_exit" in script
    assert "v99_make_policy.log" in script


def test_run_script_exit_code_captured():
    script = train_remote.run_script(FAKE_CFG, "v99", ["--source", "ds", "--version", "v99"])
    assert "echo $?" in script
    assert "v99_exit" in script


def test_keepawake_script_calls_wsl_run_sh():
    ps1 = train_remote.keepawake_script(FAKE_CFG, "v99")
    assert "wsl.exe" in ps1
    assert "v99_run.sh" in ps1
    assert "SetThreadExecutionState" in ps1


def test_keepawake_shutdown_flag():
    ps1_no_shutdown = train_remote.keepawake_script(FAKE_CFG, "v99", shutdown_train=False)
    ps1_shutdown = train_remote.keepawake_script(FAKE_CFG, "v99", shutdown_train=True)
    assert "shutdown" not in ps1_no_shutdown
    assert "shutdown /s" in ps1_shutdown


def test_make_policy_args_passthrough():
    parsed = train_remote.parse_args(
        ["--source", "ds", "--version", "v99",
         "--add", "Yewy/a", "--add", "Yewy/b",
         "--trim-narrowing", "--augment", "--replace-mat", "0.7",
         "--steps", "50000", "--private", "--default"]
    )
    args = train_remote.make_policy_args(parsed)
    assert "--source" in args and "ds" in args
    assert "--version" in args and "v99" in args
    assert args.count("--add") == 2
    assert "--trim-narrowing" in args
    assert "--augment" in args
    assert "--replace-mat" in args and "0.7" in args
    assert "--steps" in args and "50000" in args
    assert "--private" in args
    assert "--default" in args


def test_make_policy_args_minimal():
    parsed = train_remote.parse_args(["--source", "ds", "--version", "v99"])
    args = train_remote.make_policy_args(parsed)
    assert "--trim-narrowing" not in args
    assert "--augment" not in args
    assert "--private" not in args
    assert "--replace-mat" not in args


def test_config_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(train_remote, "CONFIG_PATH", tmp_path / "train_machine.json")
    with pytest.raises(SystemExit, match="train_machine.json not found"):
        train_remote.load_config()


def test_config_missing_key_raises(tmp_path, monkeypatch):
    cfg = tmp_path / "train_machine.json"
    cfg.write_text(json.dumps({"ssh": "u@h"}))
    monkeypatch.setattr(train_remote, "CONFIG_PATH", cfg)
    with pytest.raises(SystemExit, match="missing 'wsl_distro'"):
        train_remote.load_config()


def test_config_loads(tmp_path, monkeypatch):
    cfg = tmp_path / "train_machine.json"
    cfg.write_text(json.dumps(FAKE_CFG))
    monkeypatch.setattr(train_remote, "CONFIG_PATH", cfg)
    result = train_remote.load_config()
    assert result["ssh"] == "user@testhost"


def test_dry_run_prints_scripts(capsys):
    # dry-run should print scripts without connecting
    args = train_remote.parse_args(
        ["--source", "mahjong_surfaces_standard_mat_v1", "--version", "v99",
         "--augment", "--trim-narrowing", "--private", "--dry-run"]
    )
    train_remote.cmd_start(args, cfg=None)
    out = capsys.readouterr().out
    assert "make_policy.py" in out
    assert "SetThreadExecutionState" in out
    assert "v99_run.sh" in out


def test_dry_run_shutdown_train_noted(capsys):
    args = train_remote.parse_args(
        ["--source", "ds", "--version", "v99", "--dry-run", "--shutdown-train"]
    )
    train_remote.cmd_start(args, cfg=None)
    out = capsys.readouterr().out
    assert "--shutdown-train" in out or "shuts down" in out.lower()


def test_sync_missing_exit_file_raises(tmp_path, monkeypatch):
    def fake_wsl_run(cfg, cmd, check=True):
        class R:
            stdout = "__NOFILE__\n"
            returncode = 0
        return R()
    monkeypatch.setattr(train_remote, "wsl_run", fake_wsl_run)
    args = train_remote.parse_args(["--version", "v99", "--sync"])
    with pytest.raises(SystemExit, match="may still be running"):
        train_remote.cmd_sync(args, FAKE_CFG)


def test_sync_nonzero_exit_raises(tmp_path, monkeypatch):
    def fake_wsl_run(cfg, cmd, check=True):
        class R:
            stdout = "1\n"
            returncode = 0
        return R()
    monkeypatch.setattr(train_remote, "wsl_run", fake_wsl_run)
    args = train_remote.parse_args(["--version", "v99", "--sync"])
    with pytest.raises(SystemExit, match="exited with code 1"):
        train_remote.cmd_sync(args, FAKE_CFG)


def test_wsl_command_has_no_single_quotes_and_round_trips():
    import base64
    cmd = "cd ~/repo && git fetch origin && echo 'it''s' | grep \"x\" $HOME"
    line = train_remote.wsl_command(FAKE_CFG, cmd)
    assert "'" not in line
    b64 = line.split("echo ")[1].split(" |")[0]
    assert base64.b64decode(b64).decode() == cmd

# --- the recipe check, as train_remote runs it (the check itself: tests/test_recipe.py) ---

V11 = {"default": "old", "policies": {"old": {"datasets": ["src", "Yewy/a"]}}}


def _start(monkeypatch, argv, here=V11, mains=V11):
    """main() with the registries handed in and nothing able to connect; returns what ran, in order."""
    calls = []
    monkeypatch.setattr(sys, "argv", ["train_remote.py", *argv])
    monkeypatch.setattr(train_remote.recipe, "registry", lambda: here)
    monkeypatch.setattr(train_remote, "mains_registry", lambda: mains)
    monkeypatch.setattr(train_remote, "local_plan", lambda args: calls.append("plan"))
    monkeypatch.setattr(train_remote, "load_config", lambda: FAKE_CFG)
    monkeypatch.setattr(train_remote, "cmd_start", lambda args, cfg: calls.append(("start", cfg)))
    train_remote.main()
    return calls


def test_a_start_missing_one_of_the_defaults_datasets_stops_before_it_connects(monkeypatch):
    with pytest.raises(SystemExit, match="--add Yewy/a"):
        _start(monkeypatch, ["--source", "src", "--version", "v99", "--add", "Yewy/new"])


def test_a_start_with_the_defaults_datasets_goes_ahead(monkeypatch):
    calls = _start(monkeypatch, ["--source", "src", "--version", "v99", "--add", "Yewy/a", "--add", "Yewy/new"])
    assert calls == [("start", FAKE_CFG)]


def test_the_dry_run_plans_here_and_leaves_the_check_to_make_policys_own(monkeypatch):
    # make_policy --dry-run runs the check inside the plan: checking here too printed every warning twice
    calls = _start(monkeypatch, ["--source", "src", "--version", "v99", "--dry-run"])
    assert calls == ["plan", ("start", None)]


def test_a_default_that_isnt_mains_stops_a_start_and_warns_a_dry_run(monkeypatch, capsys):
    newer = {"default": "newer", "policies": {"newer": {"datasets": ["src"]}}}
    with pytest.raises(SystemExit, match="origin/main's is newer"):
        _start(monkeypatch, ["--source", "src", "--version", "v99", "--add", "Yewy/a"], mains=newer)
    _start(monkeypatch, ["--source", "src", "--version", "v99", "--dry-run"], mains=newer)
    assert "origin/main's is newer" in capsys.readouterr().out


def test_a_failed_local_plan_stops_the_dry_run():
    def failed(command):
        return type("Done", (), {"returncode": 1})()
    with pytest.raises(SystemExit, match="couldn't plan"):
        train_remote.local_plan(train_remote.parse_args(["--source", "s", "--version", "v99"]), run=failed)


def test_leave_out_and_policy_id_are_passed_to_make_policy():
    parsed = train_remote.parse_args(["--source", "s", "--version", "v99", "--policy-id", "pickup_mat_cl_v99b",
                                      "--leave-out", "Yewy/old_set"])
    args = train_remote.make_policy_args(parsed)
    assert args[args.index("--policy-id") + 1] == "pickup_mat_cl_v99b"
    assert args[args.index("--leave-out") + 1] == "Yewy/old_set"


def test_the_status_hint_for_a_failed_run_has_no_single_quotes(monkeypatch, capsys):
    # the Windows end hands it to cmd.exe, which doesn't honour them (the fault wsl_command fixed)
    monkeypatch.setattr(train_remote, "wsl_run", lambda cfg, cmd, check=True: type("R", (), {"stdout": "1\n"})())
    train_remote.cmd_status(train_remote.parse_args(["--version", "v99", "--status"]), FAKE_CFG)
    hint = [line for line in capsys.readouterr().out.splitlines() if line.strip().startswith("ssh ")]
    assert hint and "'" not in hint[0]
