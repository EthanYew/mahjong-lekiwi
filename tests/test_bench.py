"""The bench end to end on simulated hardware: trials write a complete, reportable session."""
import argparse
import sys
from pathlib import Path

import pytest

from mahjong import config
from mahjong.hover_correction import read_all
from mahjong.session import read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_sim_trials_write_a_complete_session(tmp_path, monkeypatch):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    args = argparse.Namespace(policy=None, trials=2, tile="8D", sim=True, device=None, hover_perturb_mm=None)
    b = bench.Bench(args)
    try:
        b.arm.rest()
        b.trials(2, "8D")
    finally:
        b.close()
    session = next(tmp_path.iterdir())
    events = read_events(session)
    attempts = [e for e in events if e["type"] == "attempt"]
    assert [a["success"] for a in attempts] == [True, True]
    assert sum(e["type"] == "return" for e in events) == 2
    assert len({a["attempt_dir"] for a in attempts}) == 2
    for a in attempts:
        assert (session / a["attempt_dir"] / "overview_before.jpg").exists()
    assert (session / "session.log").read_text()
    samples = read_all(session / "hover_corrections.jsonl")     # sim samples stay out of data/
    assert [r["source"] for r in samples] == ["bench", "bench"] and not samples[0].get("excluded")


def test_perturbed_trials_offset_the_hover_and_set_their_samples_aside(tmp_path, monkeypatch):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    offsets = bench.parse_perturb("0,6x")
    assert offsets == [(0.0, 0.0), (0.006, 0.0)]
    args = argparse.Namespace(policy=None, trials=2, tile="8D", sim=True, device=None, hover_perturb_mm=offsets)
    b = bench.Bench(args)
    try:
        b.arm.rest()
        b.trials(2, "8D")
    finally:
        b.close()
    session = next(tmp_path.iterdir())
    first, second = [e for e in read_events(session) if e["type"] == "attempt"]
    assert (first["hover_perturb_xy"], second["hover_perturb_xy"]) == ([0.0, 0.0], [0.006, 0.0])
    assert second["hover_xy"][0] - first["hover_xy"][0] == pytest.approx(0.006)
    assert first["correction"] == b.correction.name and len(first["hover_sag_xyz_m"]) == 3
    plain, perturbed = read_all(session / "hover_corrections.jsonl")
    assert not plain.get("excluded") and perturbed["excluded"]
