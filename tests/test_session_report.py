"""session_report's reading of a hover perturbation test."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import session_report  # noqa: E402


def attempts(follows):
    """Attempts at offsets 0 / +-6mm in x and +6mm in y; the grasp moves `follows` x the offset."""
    out = []
    for dx, dy in [(0, 0), (0.006, 0), (-0.006, 0), (0, 0.006)] * 2:
        out.append({"policy": "p", "tile": "4D", "success": True, "detection_xy": [0.2, 0.05],
                    "hover_perturb_xy": [dx, dy], "grasp_xy": [0.21 + follows * dx, 0.06 + follows * dy]})
    out.append({**out[1], "success": False, "grasp_xy": [0.5, 0.5]})     # misses don't count
    return out


@pytest.mark.parametrize("follows", [0.0, 0.4, 1.0])
def test_the_slope_says_how_much_the_grasp_follows_the_hover(follows):
    slopes = session_report.perturbation_slopes(attempts(follows))
    assert slopes["x"] == (pytest.approx(follows), 6) and slopes["y"] == (pytest.approx(follows), 4)


def test_no_slope_without_offsets_both_ways():
    plain = [{**a, "hover_perturb_xy": [0.0, 0.0]} for a in attempts(1.0)]
    assert session_report.perturbation_slopes(plain) == {}
