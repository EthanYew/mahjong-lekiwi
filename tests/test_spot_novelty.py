"""spot_novelty.py end to end: a taught layout against a recording's episode log."""
import json
import sys
from pathlib import Path

from mahjong import config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_spots_are_rated_by_distance_to_the_nearest_demo(tmp_path, monkeypatch, capsys):
    import spot_novelty
    layouts = tmp_path / "layouts.json"
    layouts.write_text(json.dumps({"eval": {"spots": [
        {"id": "F1", "xy": [0.150, 0.000], "yaw_deg": 0.0},
        {"id": "N1", "xy": [0.185, 0.000], "yaw_deg": 0.0},
        {"id": "T1", "xy": [0.150, 0.015], "yaw_deg": 30.0}]}}))
    monkeypatch.setattr(config, "LAYOUTS", layouts)
    demos = {0: {"detection_xy": [0.152, 0.000]}, 1: {"detection_xy": [0.220, 0.000]}, 2: {}}
    monkeypatch.setattr(spot_novelty, "episode_log", lambda repo: demos)
    spot_novelty.main(["eval"])
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines() if line.startswith("  ")}
    assert "2.0 mm  familiar" in lines["F1"]
    assert "33.0 mm  new" in lines["N1"]
    assert "near a demo" in lines["T1"] and "+30°" in lines["T1"]
