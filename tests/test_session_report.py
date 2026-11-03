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


def test_close_height_is_given_per_policy_and_spot_and_says_which_closed_lower(capsys):
    def attempt(policy, spot, mm, success=True):
        return {"policy": policy, "spot": spot, "tile": "4D", "success": success, "grasp_clearance_m": mm / 1000}
    session_report.close_height_report([attempt("a", "S1", 22, False), attempt("a", "S2", 12), attempt("b", "S1", 8),
                                        attempt("b", "S2", 9), {"policy": "b", "spot": "S3", "tile": "4D", "success": False}])
    lines = capsys.readouterr().out.splitlines()
    row = next(line for line in lines if line.strip().startswith("a ")).split()
    assert row == ["a", "2/2", "17.0", "12.0", "22.0", "0", "1"]    # closed, median, picked, missed, under 10, 20+
    row = next(line for line in lines if line.strip().startswith("b ")).split()
    assert row == ["b", "2/3", "8.5", "8.5", "-", "2", "0"]         # one attempt never closed
    assert next(line for line in lines if line.strip().startswith("S1 ")).split() == ["S1", "22.0x", "8.0"]
    assert any("b closed lower than a at 2 of 2 spots" in line for line in lines)


def test_no_close_height_section_for_sessions_from_before_it_was_logged(capsys):
    session_report.close_height_report([{"policy": "a", "spot": "S1", "tile": "4D", "success": True}])
    assert capsys.readouterr().out == ""


def test_the_interventions_section_counts_rescues_and_lists_the_rest(capsys):
    def attempt(n, **fields):
        return {"type": "attempt", "attempt_dir": f"00{n}_4D", "tile": "4D", "success": True,
                "gripper_says_held": True, "stop_reason": "done", **fields}
    events = [{"type": "session_start"}, attempt(1, marks_s=[6.2], mark_kind="touched_tile"),
              attempt(2, marks_s=[3.0], mark_kind="look"),
              {"type": "intervention", "kind": "tile_reset", "attempt_dir": "002_4D"},
              {"type": "session_end", "reason": "quit"}]
    found = session_report.interventions_report(events, [e for e in events if e["type"] == "attempt"], "attempt")
    out = capsys.readouterr().out
    assert len(found) == 3
    assert "rescues: 1 in 2 attempts (5.0 per 10 attempts, one per 2.0)" in out
    assert "resets 1" in out and "observations 1" in out and "1 unassisted + 1 assisted" in out
    assert "marked at 6.2s" in out and "no scans logged" in out
