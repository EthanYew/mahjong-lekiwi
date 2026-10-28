"""The closed-form kinematics against MahjongBot's URDF FK and the old closed-loop hover, via fixtures
recorded from the old code (data/fixtures)."""
import json

import numpy as np
import pytest

from mahjong import config, kinematics

FIXTURES = config.DATA / "fixtures"


def test_pincer_tip_matches_urdf_fk():
    data = np.load(FIXTURES / "urdf_pincer_tips.npz")
    names = [str(n) for n in data["joint_names"]]
    errors = [np.linalg.norm(kinematics.pincer_tip(dict(zip(names, q))) - tip)
              for q, tip in zip(data["joints"], data["pincer_tip"])]
    assert max(errors) < 1e-5   # 0.01mm


def test_inverse_round_trip():
    for target in ([0.20, -0.14, -0.05], [0.16, 0.18, -0.06], [0.27, 0.0, -0.08]):
        q = kinematics.inverse(target)
        assert np.linalg.norm(kinematics.tool_point(q) - target) < 1e-5
        assert q["wrist_roll"] == pytest.approx(q["shoulder_pan"] - 90)


def test_hover_commands_match_old_closed_loop_stack():
    cases = json.loads((FIXTURES / "old_hover_commands.json").read_text())["cases"]
    for case in cases:
        command, tilt = kinematics.hover_command(case["x"], case["y"], case["offset"], gripper=90.0)
        old = case["act_start"]
        assert (command is None) == (old is None), case
        if old is None:
            continue
        assert tilt == case["act_tilt"]
        for joint in config.ARM_JOINTS:
            assert command[joint] == pytest.approx(old[joint], abs=0.01), (case, joint)


def test_reachability_matches_old_closed_loop_stack():
    cases = json.loads((FIXTURES / "old_hover_commands.json").read_text())["cases"]
    for case in cases:
        q, tilt = kinematics.find_reachable_pose([case["x"], case["y"], config.grasp_z(case["x"], case["y"]) + 0.03])
        assert (q is not None) == case["reachable"], case
        if q is not None:
            assert tilt == case["reach_tilt"]
            for joint in config.ARM_JOINTS:
                assert q[joint] == pytest.approx(case["reach_cfg"][joint], abs=0.01)


def test_grasp_frame_is_the_first_step_commanding_the_gripper_shut():
    import numpy as np
    from mahjong.kinematics import grasp_frame, grasp_point
    open_, shut = [0.0, 0.0, 0.0, 0.0, 0.0, 30.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert grasp_frame([]) is None and grasp_frame([open_, open_]) is None
    assert grasp_frame([open_, shut, shut]) == 1 == grasp_frame(np.array([open_, shut, shut]))
    assert grasp_point([open_, shut])[0] == 1 and grasp_point([]) == (None, None)
