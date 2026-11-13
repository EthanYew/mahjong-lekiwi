"""The hook that asks before an arm script runs: real runs prompt, reads and --sim don't."""
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / ".claude" / "hooks" / "arm_run_gate.py"
PY = "~/miniforge3/envs/lerobot/bin/python"

spec = importlib.util.spec_from_file_location("arm_run_gate", HOOK)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize("command", [
    f"{PY} scripts/bench.py --trials 3",
    "python scripts/play.py --layout even-hand13",
    f"HF_HUB_OFFLINE=1 {PY} scripts/record.py --dataset Yewy/x",
    f"timeout 300 {PY} scripts/setup_check.py 2>&1 | tail -3",
    f"cd ~/Projects/mahjong-lekiwi && {PY} scripts/collect_corrections.py",
    f"{PY} /home/ethanYew/Projects/mahjong-lekiwi/scripts/wrist_capture.py",
    f"nohup {PY} -u scripts/bench.py &",
    "./scripts/bench.py",
    f"{PY} scripts/bench.py --sim; {PY} scripts/bench.py",            # only the first part is simulated
    f"{PY} scripts/record.py --sim",                                  # no --sim there, so nothing vouches for it
    f"{PY} -c \"import runpy; runpy.run_path('scripts/bench.py')\" --sim",
    f"{PY} -m somemodule scripts/bench.py --sim",
    f"{PY} scripts/other.py scripts/bench.py --sim",
    f"P={PY}; $P scripts/bench.py",
    "$UNSET scripts/bench.py --sim",                                  # no way to tell what that runs
    "P=/bin/sh; $P scripts/bench.py --sim",
    f"cd ~/Projects/mahjong-lekiwi && {PY} - <<'EOF'\nimport subprocess; subprocess.run(['python', 'scripts/bench.py'])\nEOF",
    f"git status; echo \"$({PY} scripts/bench.py)\"",
    f"bash -c '{PY} scripts/bench.py'",
    f"echo scripts/bench.py | xargs {PY}",
    f"echo \"$({PY} scripts/bench.py)\"",
    f"for f in scripts/bench.py; do {PY} $f; done",
    f"{PY} - <<'EOF'\nimport subprocess; subprocess.run(['python', 'scripts/bench.py'])\nEOF",
    f"if {PY} scripts/bench.py; then echo done; fi",
    "python scripts/bench.py 'unterminated",
    f"{PY} $(echo scripts/bench.py)",                                 # the substitution hands python the script
    f"{PY} \"$(echo scripts/bench.py)\" --sim",
    f"echo `{PY} scripts/bench.py`",
    f"{PY} -c \"import subprocess; subprocess.run(['python', 'scripts/bench.py'])\"",
    f"{PY} -c \"exec(open('scripts/bench.py').read())\"",
    f"{PY} scripts/wall_test.py --stations F",                        # a real wall test, not --report
    f"{PY} scripts/bench.py --report",                                # --report is wall_test's alone
    # Ways round the 2026-11-13 changes that must still ask.
    f"T=/tmp/x && mkfifo $T/in && nohup {PY} -u scripts/bench.py --policy a,b < $T/in > $T/log 2>&1 &",
    f"for f in scripts/bench.py; do bash -c \"{PY} $f\"; done",
    f"L=scripts/bench.py; {PY} $L",
    f"L='scripts/play.py scripts/bench.py'; for f in $L; do {PY} $f; done",
    f"{PY} - <<'EOF'\nimport subprocess, sys\nsubprocess.run([sys.executable, 'scripts/bench.py'])\nEOF",
    f"{PY} - <<'EOF'\nimport subprocess\ncmd = ['python', 'x']\nsubprocess.run(cmd)\n# scripts/bench.py\nEOF",
    f"{PY} - <<'EOF'\nimport os\nos.system('python scripts/play.py --layout even-hand13')\nEOF",
    f"{PY} - <<'EOF'\nfrom scripts import bench\nbench.main()\nEOF",
    f"echo x | xargs -n1 {PY} scripts/bench.py",
    f"/usr/bin/time -f '%e' {PY} scripts/bench.py",
    f"timeout 30 bash -c '{PY} scripts/play.py --layout even-hand13'",
    # Found by /code-review on 2026-11-14: each slipped past the parsed check.
    f"{PY} -c \"import subprocess as s; s.run(['python', 'scripts/bench.py'])\"",
    f"{PY} -c \"from subprocess import run; run(['python', 'scripts/bench.py'])\"",
    f"{PY} -c \"from scripts import bench; bench.main([])\"",
    f"cd scripts && {PY} -c \"import bench; bench.main()\"",
    f"cat <<'EOF' | {PY}\nimport subprocess; subprocess.run(['python', 'scripts/bench.py'])\nEOF",
    f"{PY} -c \"import runpy; runpy.run_module('scripts.bench', run_name='__main__')\"",
    f"{PY} -c \"import importlib; importlib.import_module('bench').main()\"",
])
def test_a_run_of_an_arm_script_asks(command):
    assert gate.reason_to_ask(command)


@pytest.mark.parametrize("command", [
    # The commands that needed a manual approval under the rules this hook replaces.
    'grep -c "def " /home/ethanYew/Projects/mahjong-lekiwi/scripts/bench.py',
    "sed -n 415,500p scripts/bench.py; grep -rn wrist_look --include=*.py scripts mahjong | grep -v '^scripts/bench.py' | head",
    "git log --format='%h %ad %s' --date=iso -S'def wrist_look' -- scripts/bench.py | tail -1",
    "ls -l scripts/bench.py scripts/record.py scripts/play.py",
    "sed -n 1,30p scripts/wrist_capture.py",
    f"timeout 170 {PY} scripts/bench.py --sim --policy a,b --layout even-hand14 --trials 1 </dev/null 2>&1 | tail -10 | cut -c1-170",
    f"yes '' | timeout 200 {PY} scripts/play.py --sim --layout even-hand13 2>&1 | tail -7",
    f"{PY} scripts/setup_check.py --sim --save-reference --reference /tmp/ref < /dev/null 2>&1 | tail -4",
    f"{PY} scripts/wrist_capture.py --sim",
    # Help, for scripts with or without --sim.
    f"{PY} scripts/record.py --help > /dev/null",
    f"{PY} scripts/collect_corrections.py -h",
    # Python on the same line isn't a run of the script.
    f"{PY} -m pytest tests/test_bench.py -q; grep -n sim scripts/bench.py",
    "grep python scripts/bench.py",
    "# python scripts/bench.py",
    "cat > notes.txt <<'EOF'\npython scripts/bench.py\nEOF",
    # Text and edits that mention a script without running it.
    "gh pr create --title 'Fix scripts/bench.py' --body \"$(cat <<'EOF'\npython scripts/bench.py --trials 3\nEOF\n)\"",
    "git add scripts/bench.py && git commit -q -m \"$(cat <<'EOF'\nscripts/bench.py: why\nEOF\n)\"",
    f"cd ~/Projects/mahjong-lekiwi && {PY} - <<'EOF'\np='scripts/bench.py'; s=open(p).read()\nEOF",
    "cp scripts/play.py /tmp/play.bak",
    f"{PY} -m py_compile scripts/record.py scripts/layout.py",
    f"{PY} -m pyflakes scripts/record.py",
    f"P={PY}; $P scripts/wrist_capture.py --help",
    f"P={PY}; $P scripts/setup_check.py --sim --save-reference 2>&1",
    # Prompts that came up in the sessions to 2026-11-11 without being arm runs.
    f"{PY} scripts/wall_test.py --report 2>&1",
    f"{PY} scripts/wall_test.py --exclude-session 20261109_1200_wall --why test",
    f"{PY} -m ruff check --select F scripts/base_cam_collect.py scripts/bench.py scripts/wall_test.py",
    "ruff check scripts/bench.py",
    f"{PY} -m mypy scripts/play.py",
    "python3 -c \"\nfrom pathlib import Path\nprint(Path('scripts/bench.py').read_text().count('import'))\"",
    "git diff $(git merge-base HEAD main) -- scripts/bench.py",
    "for f in $(git ls-files scripts); do grep -c def $f; done; wc -l scripts/bench.py",
    "echo \"$(grep -c def scripts/bench.py)\"",
    "wc -l `ls scripts/bench.py`",
    # Prompts in the sessions to 2026-11-13 that weren't arm runs (the transcripts, home paths taken out).
    'gh pr create --draft --title "Torn frames" --body "## What changed\nRun \\`scripts/bench.py --sim\\` to see it.\n"',
    "gh pr create --draft --title \"Wall stations; the drop zone\" --body-file - <<'EOF'\npython scripts/record.py --wall-station zone\nEOF",
    'grep -n "^- \\`scripts/base_cam_collect.py\\`: the arm collects" docs/face_down_plan.md',
    f"{PY} - <<'PYEOF'\nimport subprocess\nbase = subprocess.check_output(['git', 'merge-base', 'HEAD', 'origin/main'], text=True)\n"
    "print(open('scripts/bench.py').read().count(base))\nPYEOF",
    f"{PY} - <<'EOF'\ns = 'import record\\nmonkeypatch.setattr(hardware, \"connect\", None)'\nopen('tests/t.py', 'w').write(s)  # scripts/record.py\nEOF",
    "python3 -B - <<'EOF'\nsrc = open('.claude/hooks/arm_run_gate.py').read(); ns = {}\nexec(compile(src, 'g', 'exec'), ns)\n"
    "print(ns['reason_to_ask']('python scripts/bench.py'))\nEOF",
    'L="mahjong/base_cam.py scripts/base_cam_collect.py"; for f in $L; do cp $f /tmp/x/$f; git show origin/main:$f > $f; done',
    "for f in CLAUDE.md scripts/play.py; do /usr/bin/time -f '  %es' python3 -I .claude/hooks/overlap_warning.py <<< \"$PWD/$f\"; done",
    "pkill -f 'sleep infinity' ; timeout 30 bash -c 'while pgrep -f \"scripts/bench.py --policy\" >/dev/null; do sleep 1; done'",
    "M=$(git log --merges -40 --format=%h); echo \"$M\" | tail -1 | xargs git log -1 --format='%h'; "
    "for f in scripts/bench.py scripts/play.py; do git log --format=%h -- $f | wc -l; done",
    "grep -rln 'hardware.connect' scripts/ | xargs -n1 basename; grep -c def scripts/bench.py",
    # Not arm scripts.
    f"{PY} scripts/status.py",
    f"{PY} scripts/session_report.py --all",
    "rm -rf build",
    "",
])
def test_anything_else_stays_silent(command):
    assert gate.reason_to_ask(command, cwd=str(REPO)) is None


@pytest.mark.parametrize("helper, asks", [
    ("from pathlib import Path\nimport sys\np = Path(sys.argv[1]); p.write_text(p.read_text().replace('a', 'b'))\n", False),
    ("import subprocess, sys\nsubprocess.run([sys.executable, sys.argv[1]])\n", True),
    ("import runpy, sys\nrunpy.run_path(sys.argv[1])\n", True),
    ("import sys\nsys.path.insert(0, 'scripts')\nimport bench\nbench.main()\n", True),
    ("from mahjong import hardware\nhardware.connect([\"arm\"])\n", True),
])
def test_a_helper_script_handed_an_arm_script_is_read_first(tmp_path, helper, asks):
    """A patch script in the job folder that edits play.py's text asked on 2026-11-10; one that could run it must still ask."""
    (tmp_path / "patch.py").write_text(helper)
    for command in (f"{PY} {tmp_path}/patch.py scripts/play.py", f"{PY} patch.py scripts/play.py"):
        assert bool(gate.reason_to_ask(command, cwd=str(tmp_path))) == asks
    assert gate.reason_to_ask(f"{PY} missing.py scripts/play.py", cwd=str(tmp_path))    # can't read it: ask


def run_hook(stdin):
    return subprocess.run([sys.executable, "-I", str(HOOK)], input=stdin, capture_output=True, text=True, timeout=30)


def call(command):
    return json.dumps({"tool_name": "Bash", "permission_mode": "auto", "tool_input": {"command": command}})


def test_hook_asks_in_the_format_claude_code_reads():
    done = run_hook(call("python scripts/bench.py"))
    assert done.returncode == 0
    out = json.loads(done.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert out["permissionDecision"] == "ask"
    assert out["permissionDecisionReason"]


def test_hook_has_no_opinion_on_a_read():
    done = run_hook(call("grep -n def scripts/bench.py"))
    assert done.returncode == 0 and done.stdout == ""


def test_hook_asks_when_its_input_is_broken():
    assert json.loads(run_hook("not json").stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_settings_run_the_hook_and_blocks_if_it_fails():
    settings = json.loads((REPO / ".claude" / "settings.json").read_text())
    entry = settings["hooks"]["PreToolUse"][0]
    assert set(entry["matcher"].split("|")) == {"Bash", "Monitor"}
    hook = entry["hooks"][0]
    assert "arm_run_gate.py" in hook["command"]
    assert hook["onFailure"] == "block"


def test_hook_is_found_from_a_worktree_and_from_outside_a_repo(tmp_path):
    """On 2026-11-03 a path only under CLAUDE_PROJECT_DIR blocked every command in a worktree whose main lacked the file."""
    settings = json.loads((REPO / ".claude" / "settings.json").read_text())
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    stdin = call("python scripts/bench.py")
    empty_project = tmp_path / "main"
    (empty_project / ".claude").mkdir(parents=True)
    env = {"PATH": os.environ["PATH"], "CLAUDE_PROJECT_DIR": str(empty_project)}
    in_repo = subprocess.run(["bash", "-c", command], input=stdin, cwd=REPO, env=env, capture_output=True, text=True, timeout=30)
    assert json.loads(in_repo.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"
    env["CLAUDE_PROJECT_DIR"] = str(REPO)
    outside = subprocess.run(["bash", "-c", command], input=stdin, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert json.loads(outside.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_no_ask_rule_undoes_the_sim_exemption():
    """A matching ask rule prompts whatever the hook says, so the arm scripts can't also have one."""
    ask = json.loads((REPO / ".claude" / "settings.json").read_text())["permissions"].get("ask", [])
    assert not [rule for rule in ask if gate.MENTION.search(rule)]


def test_every_script_that_connects_the_arm_is_gated():
    """A script connects the arm itself, or through bench.Bench (play.py) or face_down_rig.Rig (wall_test, crude_run...): the
    hand-kept list missed the indirect ones until 2026-11-13."""
    moves = re.compile(r"hardware\.connect\(\s*\[\s*\"arm\"|\bbench\.\w*Bench\b|class \w+\(\s*(?:bench\.)?Bench\b|\bRig\(|face_down_rig")
    connects = {p.stem for p in (REPO / "scripts").glob("*.py") if moves.search(p.read_text())}
    assert {"bench", "play", "wall_test", "crude_run", "base_cam_collect"} <= connects
    assert connects <= set(gate.ARM_SCRIPTS), f"move the arm but aren't gated: {sorted(connects - set(gate.ARM_SCRIPTS))}"


def test_sim_exemption_only_for_scripts_that_define_it():
    defines = {p.stem for p in (REPO / "scripts").glob("*.py")
               if re.search(r"add_argument\(\s*\"--sim\"", p.read_text())}
    assert set(gate.SIM_SCRIPTS) == defines & set(gate.ARM_SCRIPTS)
