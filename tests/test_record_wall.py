"""record.py --wall-station end to end on simulated hardware: wall picks aimed at a saved station, from wall_test.py's hovers."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mahjong import config, hardware, layout, markers, wall_test
from mahjong.hover_correction import HoverCorrection
from test_record import DescendingPolicy, TouchedAt, episode_rows, run_record

TWO_EPISODES = [("hover", {}), ("rec", {}), ("hover", {}), ("rec", {})]


def run_wall(tmp_path, monkeypatch, script, argv=(), prompts=None):
    import mahjong.policy
    monkeypatch.setattr(mahjong.policy, "Policy", DescendingPolicy)
    return run_record(tmp_path, monkeypatch, ["--policy", "fake_v3c", "--wall-station", "F", *argv], script, prompts=prompts, gamepad=TouchedAt)


def test_wall_episodes_aim_at_the_station_and_take_the_layers_in_turn(tmp_path, monkeypatch):
    prompts = []
    dataset, _ = run_wall(tmp_path, monkeypatch, TWO_EPISODES, prompts=prompts)
    top, bottom = episode_rows(dataset)
    station = next(s for s in layout.load(wall_test.LAYOUT)["spots"] if s["id"] == "F")
    x, y = HoverCorrection.load().apply(*station["xy"])
    assert (top["wall_layer"], bottom["wall_layer"]) == ("top", "bottom") and top["spot"] == "F" and top["detection_xy"] == station["xy"]
    assert top["hover_xy"] == pytest.approx([x - 0.010, y]) and bottom["hover_xy"] == pytest.approx([x, y])     # only the top tile starts nearer the robot
    assert top["hover_raise_m"] == config.TILE_THICKNESS_M and bottom["hover_raise_m"] == 0.0
    assert top["hover_clearance_m"] - bottom["hover_clearance_m"] == pytest.approx(config.TILE_THICKNESS_M, abs=0.004)
    assert top["hover"] == bottom["hover"] == DescendingPolicy.hover       # what make_policy reads: the policy's own, not the wall's raise
    assert top["policy"] == "fake_v3c" and top["takeover_frame"] == 1
    assert not any("Which tile" in p for p in prompts) and sum("once it's laid" in p for p in prompts) == 2


def test_wall_episodes_say_what_to_lay(tmp_path, monkeypatch, capsys):
    run_wall(tmp_path, monkeypatch, TWO_EPISODES)
    out = capsys.readouterr().out
    assert "[top, episode 1/2] Clear station F" in out and "4 tiles, GREEN side up" in out
    assert "[bottom, episode 2/2] Clear station F" in out and "NOTHING on top" in out and "W = white side up" in out


def test_a_discarded_wall_episode_tries_the_same_layer_again(tmp_path, monkeypatch):
    dataset, _ = run_wall(tmp_path, monkeypatch, [("hover", {}), ("rec", {"rerecord_episode": True}), ("hover", {}), ("rec", {})], argv=["--episodes", "1"])
    assert [r["wall_layer"] for r in episode_rows(dataset)] == ["top"]


def test_a_repeated_layer_gets_more_of_its_episodes(tmp_path, monkeypatch):
    dataset, _ = run_wall(tmp_path, monkeypatch, TWO_EPISODES, argv=["--wall-layers", "bottom,bottom,top"])
    assert [r["wall_layer"] for r in episode_rows(dataset)] == ["bottom", "bottom"]


def test_q_at_the_laying_prompt_ends_the_recording(tmp_path, monkeypatch):
    import mahjong.policy
    monkeypatch.setattr(mahjong.policy, "Policy", DescendingPolicy)
    dataset, _ = run_record(tmp_path, monkeypatch, ["--policy", "fake_v3c", "--wall-station", "F"], [], answers={"once it's laid": "q"}, gamepad=TouchedAt)
    assert dataset.saved == []


def test_a_held_wall_tile_goes_in_the_container_and_isnt_returned(tmp_path, monkeypatch):
    import record
    calls = []
    monkeypatch.setattr(record, "discard_tile", lambda arm: calls.append("discard") or True)
    monkeypatch.setattr(record, "return_tile", lambda *a, **kw: pytest.fail("a wall tile can't go back on its stack"))
    run_wall(tmp_path, monkeypatch, TWO_EPISODES)
    assert calls == ["discard", "discard"]


def test_discard_tile_lets_go_over_the_container_and_only_when_holding():
    import record
    log = []

    class Arm:
        held = True

        def holding(self):
            return self.held, 0.0, 11.0

        def rest(self):
            log.append("rest")

        def go_to_pose(self, pose):
            log.append("container")

        def set_gripper(self, value):
            log.append(("open", value))

    assert record.discard_tile(Arm()) is True and log == ["rest", "container", ("open", config.GRIPPER_RELEASE), "rest"]
    log.clear()
    Arm.held = False
    assert record.discard_tile(Arm()) is False and log == []


def test_zone_episodes_aim_at_the_drop_zone_from_the_policys_own_hover(tmp_path, monkeypatch, capsys):
    import mahjong.policy
    monkeypatch.setattr(mahjong.policy, "Policy", DescendingPolicy)
    monkeypatch.setattr(markers, "load_zone", lambda: {"xy": [0.2028, -0.1723]})
    dataset, _ = run_record(tmp_path, monkeypatch, ["--policy", "fake_v3c", "--wall-station", "zone", "--wall-layers", "flat"], TWO_EPISODES, gamepad=TouchedAt)
    rows = episode_rows(dataset)
    x, y = HoverCorrection.load().apply(0.2028, -0.1723)
    assert [r["wall_layer"] for r in rows] == ["flat", "flat"] and rows[0]["spot"] == "zone" and rows[0]["detection_xy"] == [0.2028, -0.1723]
    assert rows[0]["hover_xy"] == pytest.approx([x, y]) and rows[0]["hover_raise_m"] == 0.0 and rows[0]["hover_back_m"] == 0.0
    assert "Clear station zone" in capsys.readouterr().out


def test_zone_episodes_need_a_taught_zone(monkeypatch):
    import record
    monkeypatch.setattr(markers, "load_zone", lambda: None)
    monkeypatch.setattr(hardware, "connect", lambda *a, **kw: pytest.fail("connected"))
    monkeypatch.setitem(sys.modules, "lerobot.common.control_utils", types.SimpleNamespace(init_keyboard_listener=None))
    with pytest.raises(SystemExit, match="teach-zone"):
        record.main(["--dataset", "test/record", "--wall-station", "zone", "--wall-layers", "flat"])


@pytest.mark.parametrize("argv", [["--wall-station", "X"], ["--wall-station", "F", "--wall-layers", "middle"], ["--wall-station", "F", "--layout", "even-hand14"],
                                  ["--wall-station", "F", "--wall-top-back-mm", "25"]])
def test_a_bad_wall_request_fails_before_anything_connects(monkeypatch, argv):
    import record
    monkeypatch.setattr(hardware, "connect", lambda *a, **kw: pytest.fail("connected"))
    monkeypatch.setitem(sys.modules, "lerobot.common.control_utils", types.SimpleNamespace(init_keyboard_listener=None))
    with pytest.raises(SystemExit):
        record.main(["--dataset", "test/record", *argv])
