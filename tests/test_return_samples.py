"""Importing return-correction samples from bench sessions."""
import json
import sys
from pathlib import Path

from mahjong.hover_correction import read_all

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import return_samples  # noqa: E402


def write_session(folder, events, sim=False):
    folder.mkdir(parents=True)
    rows = [{"type": "session_start", "time": "2026-10-23T10:18:08", "sim": sim}] + events
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in rows))
    return folder


def cycle(i, landed=(0.203, 0.001)):
    """One pickup + return of 4D, its after-return scan seeing the tile at `landed` (None: not seen)."""
    tiles = [{"tile_name": "4D", "xy": list(landed)}] if landed else []
    return [{"type": "attempt", "time": f"2026-10-23T10:2{i}:00", "tile": "4D", "attempt_dir": f"00{i}_4D",
             "detection_xy": [0.2, 0.0], "hover_xy": [0.23, 0.01], "grasp_xy": [0.235, 0.012], "policy": "p",
             "success": True},
            {"type": "return", "xy": [0.236, 0.013]},
            {"type": "detections", "label": "after_return", "tiles": tiles}]


def test_returns_with_a_landing_are_samples_and_import_once(tmp_path):
    folder = write_session(tmp_path / "20261023_101808_bench", cycle(1) + cycle(2, landed=None) + cycle(3))
    records = return_samples.session_records(folder)
    assert [r["attempt_dir"] for r in records] == ["001_4D", "003_4D"]
    r = records[0]
    assert (r["target_xy"], r["landed_xy"], r["timestamp"]) == ([0.236, 0.013], [0.203, 0.001], "2026-10-23T10:21:00")
    assert return_samples.session_records(write_session(tmp_path / "sim_bench", cycle(1), sim=True)) == []
    path = tmp_path / "r.jsonl"
    assert return_samples.import_records(records, path, since="2026-10-23T10:22:00") == (1, 0, 1)
    assert return_samples.import_records(records, path, since="2026-10-23T10:00:00") == (1, 1, 0)
    assert [r["attempt_dir"] for r in read_all(path)] == ["003_4D", "001_4D"]
