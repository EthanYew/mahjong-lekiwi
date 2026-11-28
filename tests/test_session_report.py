"""session_report's reading of a hover perturbation test."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import session_report  # noqa: E402


def test_a_face_down_hand_lists_its_tiles_not_its_slots(capsys):
    """play_face_down.py's hand_end keeps the hand as {slot: name} and the tile it won on apart; play.py's as a list.
    (Post-merge review of #125: the report printed 'hand S1 S2 S3 ...'.)"""
    end = {"type": "hand_end", "result": "win", "turns": 3, "grade": "clean", "silent_errors": 0,
           "hand": {"S1": "1C", "S2": "2C", "S14": "5D"}, "won_on": "5D", "discards": ["4D"]}
    session_report.play_report([end], [])
    out = capsys.readouterr().out
    assert "hand 1C 2C 5D, won on 5D; discards 4D" in out and "S1" not in out


def test_a_face_down_sim_win_reports_its_pickups_hand_and_verdict(monkeypatch, tmp_path, capsys):
    """A face-down hand logs as face-up play does (mahjong/play_log.py): its picks fill the pickup table, and the hand says
    whether the win is verified. Its end check covers the tile it won on, so a clean check verifies it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_play_face_down import LIGHT, WINS_ON_5D, run, script_in
    script = script_in(monkeypatch, tmp_path, WINS_ON_5D + ["5D"])
    folder, _ = run(script, ["--unattended"], [*LIGHT, "", ""])
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["session_report.py", str(folder)])
    session_report.main()
    out = capsys.readouterr().out
    assert "14 pickup attempt(s), 14 picked up" in out          # the 13 read, then the draw it won on
    assert ", won on 5D;" in out and "Win verified: a person vouched for all 14 tiles" in out
    assert "Unattended: the reader named every tile" in out and "Winning 5D: a person checked it" in out


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
    assert row == ["a", "2/2", "17.0", "12.0", "22.0", "0", "1", "1"]    # closed, median, picked, missed, under 10, 13+, 20+
    row = next(line for line in lines if line.strip().startswith("b ")).split()
    assert row == ["b", "2/3", "8.5", "8.5", "-", "2", "0", "0"]         # one attempt never closed
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


def test_where_to_record_lists_spots_with_misses_remarks_and_the_camera(capsys):
    def attempt(n, spot, success=True, **fields):
        return {"type": "attempt", "attempt_dir": f"00{n}", "policy": "p", "spot": spot, "tile": "face_down",
                "success": success, "stop_reason": "done", "torn_frames": 0, **fields}

    def keys(n, *kinds):
        return {"type": "keys", "attempt_dir": f"00{n}", "during": "pickup",
                "presses": [{"key": k, "kind": k, "t_s": 1.0} for k in kinds]}
    events = [attempt(1, "S1"), attempt(2, "S1"),
              attempt(3, "S6", False, stop_reason="cap"), keys(3, "pushed_target"),
              attempt(4, "S6", False, stop_reason="camera"),
              {"type": "attempt_void", "attempt_dir": "005", "policy": "p", "spot": "S10"},
              attempt(6, "S10"), keys(6, "grip_shallow", "grip_shallow"),
              attempt(7, "S12", grasp_clearance_m=0.024)]
    found = session_report.interventions.derive(events)
    session_report.recording_focus(events, [e for e in events if e["type"] == "attempt"], found)
    lines = capsys.readouterr().out.splitlines()
    row = lambda spot: next(line for line in lines if line.strip().startswith(spot + " ")).split()
    # columns: missed, pushed, early, bumped, low, high, shallow, 20mm+, camera
    assert row("S6") == ["S6", "2/2", "1", "-", "-", "-", "-", "-", "-", "1", "stopped"]
    assert row("S10") == ["S10", "0/1", "-", "-", "-", "-", "-", "1", "-", "1", "voided"]    # attempts, not presses
    focus = lines[lines.index("  record more at (attempts with trouble, most first):") + 1:]
    assert [line.split(":")[0].strip() for line in focus] == ["S6", "S10", "S12"]      # S1 had no trouble
    assert "the camera stopped 1 of them" in focus[0]
    assert "shallow 1" in focus[1] and "1 voided and rerun" in focus[1]
    assert "closed 20mm+ up 1" in focus[2] and "no camera loss" in focus[2]


def test_a_session_from_before_the_torn_frame_check_doesnt_say_the_camera_was_fine(capsys):
    """20261113_100710_bench: 34 of 53 attempts on torn frames, logged before policy.run counted them (no torn_frames)."""
    events = [{"type": "attempt", "attempt_dir": "001", "policy": "p", "spot": "S2", "tile": "4D", "success": False,
               "stop_reason": "cap", "grasp_clearance_m": 0.05}]
    session_report.recording_focus(events, events, [])
    out = capsys.readouterr().out
    assert "no camera loss" not in out
    assert "torn frames weren't checked" in out


def test_a_pre_check_spot_with_a_voided_attempt_still_says_tears_werent_checked(capsys):
    """A lost-camera void on the spot used to win: "no counted attempt lost the camera" hid the unchecked tears."""
    attempts = [{"type": "attempt", "attempt_dir": "001", "policy": "p", "spot": "S3", "tile": "4D", "success": False,
                 "stop_reason": "cap"}]
    session_report.recording_focus(attempts + [{"type": "attempt_void", "spot": "S3", "policy": "p"}], attempts, [])
    out = capsys.readouterr().out
    assert "torn frames weren't checked in 1" in out and "1 voided and rerun" in out
    assert "no counted attempt lost the camera" not in out


def test_a_spot_s_camera_remark_says_every_camera_problem_it_had(capsys):
    """Stopped, a few torn, unchecked and voided were one chain: the first that applied hid the rest."""
    attempts = [{"type": "attempt", "attempt_dir": f"00{n}", "policy": "p", "spot": "S4", "tile": "4D", "success": n != 1,
                 **fields} for n, fields in enumerate([{"stop_reason": "camera", "torn_frames": 5},
                                                       {"stop_reason": "done", "torn_frames": 2},
                                                       {"stop_reason": "done"}], start=1)]
    session_report.recording_focus(attempts + [{"type": "attempt_void", "spot": "S4", "policy": "p"}], attempts, [])
    line = next(line for line in capsys.readouterr().out.splitlines() if line.strip().startswith("S4:"))
    assert "stopped 1" in line and "1 saw a few torn" in line and "weren't checked in 1" in line and "1 voided" in line


def test_no_recording_focus_without_spots(capsys):
    session_report.recording_focus([], [{"policy": "p", "tile": "4D", "success": False, "attempt_dir": "001"}], [])
    assert capsys.readouterr().out == ""
