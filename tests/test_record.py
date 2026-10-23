"""record.py's hover jitter: random offsets within the radius, clean episodes kept, fixed lists cycled."""
import math
import random
import sys
from pathlib import Path

import pytest

from mahjong.hover_correction import parse_offsets, random_offset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_random_offsets_stay_in_the_disc_and_fill_it():
    rng = random.Random(0)
    offsets = [random_offset(rng, 0.008) for _ in range(4000)]
    radii = [math.hypot(*o) for o in offsets]
    assert max(radii) <= 0.008
    # uniform by area: half the draws land beyond r / sqrt(2)
    assert 0.45 < sum(r > 0.008 / math.sqrt(2) for r in radii) / len(radii) < 0.55
    assert 0.45 < sum(o[0] > 0 for o in offsets) / len(offsets) < 0.55


def test_zero_fraction_keeps_clean_hovers():
    rng = random.Random(1)
    offsets = [random_offset(rng, 0.006, zero_fraction=0.2) for _ in range(2000)]
    assert 0.17 < offsets.count((0.0, 0.0)) / len(offsets) < 0.23
    assert random_offset(rng, 0.0) == (0.0, 0.0)


def test_parse_offsets():
    assert parse_offsets("0, 6x,-6y") == [(0.0, 0.0), (0.006, 0.0), (0.0, -0.006)]
    with pytest.raises(Exception):
        parse_offsets("6z")


def test_next_offset_cycles_a_list_by_saved_episode_and_is_seeded():
    import argparse
    import record
    fixed = argparse.Namespace(hover_offsets_mm=parse_offsets("6x,-6y"), hover_jitter_mm=0.0, jitter_zero_fraction=0.2)
    assert [record.next_offset(fixed, None, n) for n in range(3)] == [(0.006, 0.0), (0.0, -0.006), (0.006, 0.0)]
    jitter = argparse.Namespace(hover_offsets_mm=None, hover_jitter_mm=8.0, jitter_zero_fraction=0.0)
    draws = [[record.next_offset(jitter, random.Random(7), n) for n in range(3)] for _ in range(2)]
    assert draws[0] == draws[1] and all(0 < math.hypot(*o) <= 0.008 for o in draws[0])
    off = argparse.Namespace(hover_offsets_mm=None, hover_jitter_mm=0.0, jitter_zero_fraction=0.2)
    assert record.next_offset(off, random.Random(0), 0) == (0.0, 0.0)
