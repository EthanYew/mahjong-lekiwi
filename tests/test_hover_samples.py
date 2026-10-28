"""Importing hover-correction samples from recorded demos and bench sessions."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mahjong import config, kinematics
from mahjong.hover_correction import read_all

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hover_samples  # noqa: E402


def episode(x, y, close_at=10, steps=20):
    """Absolute actions over (x, y): hover with the jaws open, then shut from `close_at`."""
    pose = kinematics.hover_command(x, y, 0.007, 90.0)[0]
    actions = np.array([[pose[j] for j in config.JOINTS]] * steps)
    actions[close_at:, 5] = 0.0
    return actions


def test_a_demo_sample_is_where_the_gripper_closed_minus_the_detection():
    log = {0: {"episode_index": 0, "timestamp": "2026-10-22T11:07:04", "tile_name": "8D",
               "detection_xy": [0.195, 0.05], "hover_xy": [0.2, 0.06]},
           1: {"episode_index": 1, "session": "20261022_120000_record", "detection_xy": [0.15, 0.0]}}
    actions = {0: episode(0.2, 0.06), 1: episode(0.15, 0.0), 2: episode(0.2, 0.0)}
    actions[1][:, 5] = 90.0                       # never closed: no sample
    (r,) = hover_samples.demo_records("Yewy/demo", actions, log)
    assert r["source"] == "demo" and (r["dataset"], r["episode_index"]) == ("Yewy/demo", 0)
    assert r["delta_xy"] == pytest.approx([0.005, 0.01], abs=0.001)
    assert hover_samples.row_time(log[1]) == "2026-10-22T12:00:00"


def write_session(folder, attempts, sim=False):
    folder.mkdir()
    events = [{"type": "session_start", "kind": "bench", "sim": sim}] + attempts
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return folder


def attempt(n, success=True, grasp_xy=(0.205, 0.06)):
    return {"time": f"2026-10-23T10:2{n}:00", "type": "attempt", "tile": "4D", "policy": "p", "detection_xy": [0.2, 0.05],
            "hover_xy": [0.2, 0.06], "success": success, "grasp_xy": list(grasp_xy) if grasp_xy else None,
            "attempt_dir": f"00{n}_4D"}


def test_bench_samples_are_successful_real_attempts(tmp_path):
    folder = write_session(tmp_path / "20261023_101808_bench",
                           [attempt(1), attempt(2, success=False), attempt(3, grasp_xy=None)])
    (r,) = hover_samples.bench_records(folder)
    assert (r["source"], r["attempt_dir"], r["timestamp"]) == ("bench", "001_4D", "2026-10-23T10:21:00")
    assert r["delta_xy"] == pytest.approx([0.005, 0.01])
    assert hover_samples.bench_records(write_session(tmp_path / "sim_bench", [attempt(1)], sim=True)) == []


def test_a_bench_attempt_from_a_perturbed_hover_is_imported_excluded(tmp_path):
    """bench.py excludes these as it writes them; a backfill used to import them as clean samples."""
    folder = write_session(tmp_path / "20261023_101808_bench",
                           [attempt(1), {**attempt(2), "hover_perturb_xy": [0.006, 0.0]},
                            {**attempt(3), "hover_perturb_xy": [0.0, 0.0]}])
    records = hover_samples.bench_records(folder)
    assert ["excluded" in r for r in records] == [False, True, False]


def test_import_skips_duplicates_and_samples_older_than_the_setup(tmp_path):
    path = tmp_path / "c.jsonl"
    folder = write_session(tmp_path / "20261023_101808_bench", [attempt(1), attempt(2)])
    records = hover_samples.bench_records(folder)
    assert hover_samples.import_records(records, path, since="2026-10-23T10:21:30") == (1, 0, 1)
    assert hover_samples.import_records(records, path, since="2026-10-23T10:00:00") == (1, 1, 0)
    assert hover_samples.import_records(records, path, since="2026-10-23T10:00:00") == (0, 2, 0)
    assert [r["attempt_dir"] for r in read_all(path)] == ["002_4D", "001_4D"]
