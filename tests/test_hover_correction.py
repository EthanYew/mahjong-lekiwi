import json
import math

import numpy as np
import pytest

from mahjong import config
from mahjong.hover_correction import HoverCorrection, setup_fingerprint, MAX_CORRECTION_M


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
    assert (model.kind, model.n, len(model.excluded)) == (old["kind"], old["n"], 2)
    for x, y, dx, dy in old["deltas"]:
        assert model.delta_at(x, y) == pytest.approx((dx, dy), abs=1e-9)


def test_other_setups_are_set_aside(tmp_path):
    path = tmp_path / "c.jsonl"
    rows = records(tiles(4), lambda x, y: (0.01, 0.0), **setup_fingerprint())
    rows += records(tiles(3), lambda x, y: (0.04, 0.04), follower_id="other", t_cam_to_base_sha="x")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    model = HoverCorrection.load(path)
    assert (model.n, len(model.set_aside)) == (4, 3)
