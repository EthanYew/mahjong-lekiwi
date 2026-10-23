import json
import math
import random

import numpy as np
import pytest

from mahjong import config
from mahjong.hover_correction import (HoverCorrection, active_sources, grasp_record, parse_offsets, parse_sources,
                                      random_offset, setup_fingerprint, source_of, MAX_CORRECTION_M)


def records(xy, delta_fn, noise=0.002, seed=1, **extra):
    rng = np.random.default_rng(seed)
    return [{"detection_xy": [x, y], "delta_xy": (np.array(delta_fn(x, y)) + rng.normal(0, noise, 2)).tolist(), **extra}
            for x, y in xy]


def tiles(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0.14, 0.27, n), rng.uniform(-0.18, 0.18, n)])


def affine_truth(x, y):
    a, s = math.radians(2.0), 1.06
    return (s * (math.cos(a) * x - math.sin(a) * y) - x + 0.005, s * (math.sin(a) * x + math.cos(a) * y) - y)


def test_offset_data_fits_an_offset():
    model = HoverCorrection(records(tiles(18), lambda x, y: (0.014, 0.004)))
    assert model.kind == "offset"
    assert np.allclose(model.delta_at(0.18, -0.17), (0.014, 0.004), atol=0.002)


def test_growing_error_fits_an_affine_map():
    model = HoverCorrection(records(tiles(18), affine_truth))
    assert model.kind == "affine"
    assert np.allclose(model.delta_at(0.18, -0.17), affine_truth(0.18, -0.17), atol=0.002)


def test_too_few_points_stay_an_offset():
    assert HoverCorrection(records(tiles(5), affine_truth)).kind == "offset"


def test_correction_is_capped_and_held_at_the_edge():
    model = HoverCorrection([{"detection_xy": [0.2, 0.0], "delta_xy": [0.2, 0.0]}])
    assert math.hypot(*model.delta_at(0.2, 0.0)) == pytest.approx(MAX_CORRECTION_M)
    model = HoverCorrection(records(tiles(18), affine_truth))
    assert model.delta_at(0.20, -0.35) == model.delta_at(0.20, model.bounds[0][1] - 0.03)


def test_migrated_records_reproduce_the_old_correction():
    old = json.loads((config.DATA / "fixtures" / "old_hover_correction.json").read_text())
    model = HoverCorrection.load()
    operator_excluded = [r for r in model.excluded if source_of(r) == "gamepad"]   # not e.g. perturbed bench samples
    assert (model.kind, model.n, len(operator_excluded)) == (old["kind"], old["n"], 2)
    for x, y, dx, dy in old["deltas"]:
        assert model.delta_at(x, y) == pytest.approx((dx, dy), abs=1e-9)


def test_other_setups_are_set_aside(tmp_path):
    path = tmp_path / "c.jsonl"
    rows = records(tiles(4), lambda x, y: (0.01, 0.0), **setup_fingerprint())
    rows += records(tiles(3), lambda x, y: (0.04, 0.04), follower_id="other", t_cam_to_base_sha="x")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    model = HoverCorrection.load(path)
    assert (model.n, len(model.set_aside)) == (4, 3)


def test_only_listed_sources_are_fitted(tmp_path):
    path = tmp_path / "c.jsonl"
    rows = records(tiles(8), lambda x, y: (0.01, 0.0), **setup_fingerprint())
    rows += records(tiles(5, seed=1), lambda x, y: (0.03, 0.0), source="demo", **setup_fingerprint())
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    gamepad = HoverCorrection.load(path, sources={"gamepad": 1.0})
    assert (gamepad.n, len(gamepad.unused)) == (8, 5)
    assert gamepad.delta_at(0.2, 0.0)[0] == pytest.approx(0.01, abs=0.002)
    both = HoverCorrection.load(path, sources={"gamepad": 1.0, "demo": 1.0})
    assert (both.n, both.sources["demo"]) == (13, 5)


def test_live_sources_default_to_gamepad_only(tmp_path):
    assert active_sources(tmp_path / "missing.json") == {"gamepad": 1.0}
    assert parse_sources("gamepad, demo:0.5,bench:0.25") == {"gamepad": 1.0, "demo": 0.5, "bench": 0.25}


def test_weights_set_how_much_a_source_counts():
    rows = records(tiles(6), lambda x, y: (0.010, 0.0), noise=0) + records(tiles(6, 1), lambda x, y: (0.020, 0.0), noise=0, source="bench")
    assert HoverCorrection(rows).delta_at(0.2, 0.0)[0] == pytest.approx(0.015, abs=1e-6)
    assert HoverCorrection(rows, weights={"bench": 0.25}).delta_at(0.2, 0.0)[0] == pytest.approx(0.012, abs=1e-6)


def test_repeats_at_one_position_dont_vouch_for_each_other():
    """The check leaves out a whole position, so five attempts at one tile count as one tile."""
    rows = records(tiles(12), affine_truth, noise=0.004)
    once, repeated = HoverCorrection(rows), HoverCorrection(rows * 5)
    for kind in ("offset", "affine"):
        assert repeated.loo_rms[kind] == pytest.approx(once.loo_rms[kind])


def test_grasp_record_is_the_grasp_minus_the_detection():
    r = grasp_record("bench", (0.20, 0.10), (0.208, 0.112), session="s")
    assert r["delta_xy"] == pytest.approx([0.008, 0.012]) and r["source"] == "bench" and r["session"] == "s"


# --- deliberate hover offsets (record.py --hover-jitter-mm / --hover-offsets-mm, bench --hover-perturb-mm) ---

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
