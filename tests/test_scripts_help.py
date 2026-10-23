"""Every user-run script builds its options: `--help` exits cleanly. Two branches each adding an option of
the same name merged without a conflict and broke record.py before any hardware was touched
(2026-10-23); only bench's and record's parsers are also driven by other tests."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO / "scripts").glob("*.py"))
COMMANDS = [[str(s)] for s in SCRIPTS] + [["-m", "mahjong.hover_correction"]]


@pytest.mark.parametrize("command", COMMANDS, ids=[Path(c[-1]).stem for c in COMMANDS])
def test_help_runs(command):
    result = subprocess.run([sys.executable, *command, "--help"], capture_output=True, text=True, timeout=120, cwd=REPO)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout
