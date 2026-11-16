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


# --- the recipe check: a new version keeps the default's datasets (v12, 2026-11-17) ---

def _registry(default_entry):
    return {"default": "old", "policies": {"old": default_entry}}


OLD_NOTES = {"notes": "make_policy.py from mahjong_surfaces_standard_mat_v1 + Yewy/mahjong_surfaces_mat_v1 + "
                      "Yewy/mahjong_facedown_dagger_v10; 88075 frames, 120000 steps; colour jitter (AUGMENT); "}


def test_recipe_gap_reads_a_policy_registered_before_the_datasets_field():
    reg = _registry(OLD_NOTES)
    # v12's command: the source and the new recording only
    default, missing = train_remote.load_make_policy().recipe_gap(
        "mahjong_surfaces_standard_mat_v1", ["Yewy/mahjong_facedown_dagger_v11"], reg=reg)
    assert default == "old"
    assert missing == ["mahjong_surfaces_mat_v1", "mahjong_facedown_dagger_v10"]


def test_recipe_gap_is_empty_for_the_default_recipe_plus_a_new_recording_with_or_without_the_hub_user():
    reg = _registry(OLD_NOTES)
    add = ["mahjong_surfaces_mat_v1", "Yewy/mahjong_facedown_dagger_v10", "Yewy/mahjong_facedown_dagger_v11"]
    assert train_remote.load_make_policy().recipe_gap("Yewy/mahjong_surfaces_standard_mat_v1", add, reg=reg)[1] == []


def test_recipe_gap_prefers_the_datasets_field_and_honours_leave_out():
    reg = _registry({"datasets": ["src", "Yewy/a", "Yewy/b"], "notes": "make_policy.py from src; 1 frames"})
    assert train_remote.load_make_policy().recipe_gap("src", [], reg=reg)[1] == ["a", "b"]
    assert train_remote.load_make_policy().recipe_gap("src", [], leave_out=["Yewy/a"], reg=reg)[1] == ["b"]


def test_check_recipe_stops_with_the_flags_to_paste():
    with pytest.raises(SystemExit) as stopped:
        train_remote.load_make_policy().check_recipe("mahjong_surfaces_standard_mat_v1", [], reg=_registry(OLD_NOTES))
    message = str(stopped.value)
    assert "--add Yewy/mahjong_surfaces_mat_v1 --add Yewy/mahjong_facedown_dagger_v10" in message
    assert "--leave-out" in message


def test_start_checks_the_recipe_before_it_connects(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_remote.py", "--source", "mahjong_surfaces_standard_mat_v1",
                                      "--version", "v99", "--add", "Yewy/new"])
    monkeypatch.setattr(train_remote, "check_default_is_mains", lambda: None)
    monkeypatch.setattr(train_remote.load_make_policy(), "recipe_gap", lambda *a, **k: ("old", ["missing_set"]))
    monkeypatch.setattr(train_remote, "load_config", lambda: pytest.fail("connected before checking the recipe"))
    with pytest.raises(SystemExit, match="missing_set"):
        train_remote.main()


def test_leave_out_is_passed_to_make_policy():
    parsed = train_remote.parse_args(["--source", "s", "--version", "v99", "--leave-out", "Yewy/old_set"])
    assert train_remote.make_policy_args(parsed)[-2:] == ["--leave-out", "Yewy/old_set"]


def test_a_default_whose_recipe_cant_be_read_warns_instead_of_passing_silently(capsys):
    reg = _registry({"notes": "hand-registered from the archive"})
    assert train_remote.load_make_policy().recipe_gap("src", [], reg=reg)[1] is None
    train_remote.load_make_policy().check_recipe("src", [], reg=reg)
    assert "check this run's list against it by hand" in capsys.readouterr().out


def test_policy_id_is_passed_to_make_policy():
    parsed = train_remote.parse_args(["--source", "s", "--version", "v99", "--policy-id", "pickup_mat_cl_v99b"])
    assert "pickup_mat_cl_v99b" in train_remote.make_policy_args(parsed)


@pytest.mark.parametrize("dry_run", [True, False])
def test_only_the_dry_run_plans_locally(monkeypatch, dry_run):
    monkeypatch.setattr(sys, "argv", ["train_remote.py", "--source", "s", "--version", "v99"] + (["--dry-run"] * dry_run))
    calls = []
    monkeypatch.setattr(train_remote.load_make_policy(), "check_recipe", lambda *a, **k: calls.append("check"))
    monkeypatch.setattr(train_remote, "check_default_is_mains", lambda: None)
    monkeypatch.setattr(train_remote, "local_plan", lambda args: calls.append("plan"))
    monkeypatch.setattr(train_remote, "load_config", lambda: FAKE_CFG)
    monkeypatch.setattr(train_remote, "cmd_start", lambda args, cfg: calls.append(("start", cfg)))
    train_remote.main()
    assert calls == (["check", "plan", ("start", None)] if dry_run else ["check", ("start", FAKE_CFG)])


def test_a_failed_local_plan_stops_the_dry_run(monkeypatch):
    monkeypatch.setattr(train_remote.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 1})())
    with pytest.raises(SystemExit, match="couldn't plan"):
        train_remote.local_plan(train_remote.parse_args(["--source", "s", "--version", "v99"]))


@pytest.mark.parametrize("skip_train", [False, True])
def test_make_policy_checks_the_recipe_unless_it_only_registers(monkeypatch, skip_train):
    mp = train_remote.load_make_policy()
    import mahjong.quality

    class Checked(Exception):
        pass

    class PastTheCheck(Exception):
        pass

    monkeypatch.setattr(sys, "argv", ["make_policy.py", "--source", "s", "--version", "v99"] + (["--skip-train"] * skip_train))
    monkeypatch.setattr(mp, "check_recipe", lambda *a, **k: (_ for _ in ()).throw(Checked()))
    monkeypatch.setattr(mahjong.quality, "quality_table", lambda *a, **k: (_ for _ in ()).throw(PastTheCheck()))
    with pytest.raises(PastTheCheck if skip_train else Checked):
        mp.main()


def test_options_that_differ_from_the_default_warn(capsys):
    reg = _registry(OLD_NOTES | {"notes": OLD_NOTES["notes"] + "mat swapped in 70% of episodes; "})
    add = ["mahjong_surfaces_mat_v1", "mahjong_facedown_dagger_v10"]
    train_remote.load_make_policy().check_recipe("mahjong_surfaces_standard_mat_v1", add, reg=reg,
                                                 options={"augment": True, "replace_mat": 0.7})
    assert "Warning" not in capsys.readouterr().out
    train_remote.load_make_policy().check_recipe("mahjong_surfaces_standard_mat_v1", add, reg=reg,
                                                 options={"augment": False, "replace_mat": 0.0})
    out = capsys.readouterr().out
    assert "--augment off" in out and "--replace-mat 0" in out


def test_a_leave_out_the_default_never_trained_on_warns(capsys):
    reg = _registry(OLD_NOTES)
    add = ["mahjong_surfaces_mat_v1", "mahjong_facedown_dagger_v10"]
    train_remote.load_make_policy().check_recipe("mahjong_surfaces_standard_mat_v1", add, leave_out=["Yewy/typo_v09"], reg=reg)
    assert "--leave-out Yewy/typo_v09" in capsys.readouterr().out


def test_notes_without_a_semicolon_after_the_list_still_read():
    reg = _registry({"notes": "make_policy.py from src + Yewy/a"})
    assert train_remote.load_make_policy().recipe_gap("src", [], reg=reg)[1] == ["a"]


def test_a_default_that_isnt_mains_stops_the_start(monkeypatch, tmp_path):
    local = tmp_path / "policies.json"
    local.write_text(json.dumps({"default": "pickup_mat_cl_v12"}))
    monkeypatch.setattr(train_remote.config, "POLICIES", local)
    shown = json.dumps({"default": "pickup_mat_cl_v11"})
    monkeypatch.setattr(train_remote.subprocess, "run",
                        lambda cmd, **k: type("R", (), {"returncode": 0, "stdout": shown})())
    with pytest.raises(SystemExit, match="origin/main's is pickup_mat_cl_v11"):
        train_remote.check_default_is_mains()
