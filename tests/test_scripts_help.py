"""Every user-run script builds its options: `--help` exits cleanly. Two branches each adding an option of
the same name merged without a conflict and broke record.py before any hardware was touched
(2026-10-23); nothing else imports a script's parser."""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.stem for s in SCRIPTS])
def test_help_runs(script):
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout
