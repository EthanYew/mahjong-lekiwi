"""scripts/status.py: the report reads the registry and the bench sessions; --set-default keeps its evidence."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import status
from mahjong import config


def bench_session(sessions, name, policy, results):
    folder = sessions / f"{name}_bench"
    folder.mkdir(parents=True)
    (folder / "events.jsonl").write_text("".join(
        json.dumps({"type": "attempt", "policy": policy, "success": ok}) + "\n" for ok in results))


def test_the_report_counts_each_policys_bench_attempts_and_marks_the_default(tmp_path, capsys):
    bench_session(tmp_path, "20261029_094543", "pickup_mat_cl_v4", [True, True, False])
    bench_session(tmp_path, "20261029_123745", "pickup_mat_cl_v4", [True])
    results = status.bench_results(tmp_path)
    assert results["pickup_mat_cl_v4"] == {"attempts": 4, "picked": 3, "sessions": 2, "last": "20261029_123745"}
    text = status.report(sessions_dir=tmp_path)
    default = json.loads(config.POLICIES.read_text())["default"]
    assert any(line.startswith(default) and line.endswith("<- default") for line in text.splitlines())
    assert "hand14" in text and "Eval sets" in text


def test_set_default_writes_the_evidence_with_it(tmp_path, monkeypatch):
    policies = tmp_path / "policies.json"
    shutil.copy(config.POLICIES, policies)
    monkeypatch.setattr(config, "POLICIES", policies)
    with pytest.raises(SystemExit):
        status.parse_args(["--set-default", "pickup_mat_cl_v3c"])            # no evidence: refused
    status.main(["--set-default", "pickup_mat_cl_v3c", "--evidence", "sessions/x_bench: 10/10", "--why", "test"])
    reg = json.loads(policies.read_text())
    assert reg["default"] == "pickup_mat_cl_v3c"
    assert reg["default_note"]["policy"] == "pickup_mat_cl_v3c" and reg["default_note"]["evidence"] == "sessions/x_bench: 10/10"
    with pytest.raises(SystemExit):
        status.main(["--set-default", "no_such_policy", "--evidence", "x"])
