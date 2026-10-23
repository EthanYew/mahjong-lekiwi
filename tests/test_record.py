"""record.py's hover jitter: random, bounded, and off unless asked for."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import record  # noqa: E402


def test_no_jitter_unless_asked():
    assert record.jitter_offset(np.random.default_rng(0), 0.0) == (0.0, 0.0)


def test_jitter_stays_within_the_bound_and_covers_it():
    rng = np.random.default_rng(0)
    offsets = np.array([record.jitter_offset(rng, 6.0) for _ in range(500)])
    assert np.abs(offsets).max() <= 0.006
    assert (offsets.min(axis=0) < -0.005).all() and (offsets.max(axis=0) > 0.005).all()   # both axes, both ways
    assert abs(np.corrcoef(offsets.T)[0, 1]) < 0.15                                       # x and y drawn apart
