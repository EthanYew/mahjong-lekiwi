"""The crude full run: its plan and record of breaks, and the whole script on the simulated arm."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import camera_poses, config, crude_run, game, base_cam, name_check
from mahjong.session import read_events

POSE = {j: 0.0 for j in config.ARM_JOINTS}
SLOTS = [f"S{i}" for i in range(1, 14)]
HAND = "1C,2C,3C,5C,5C,7C,7C,9C,9C,1B,1B,2C,3C"
LIGHT = ["", "", ""]


def test_a_hand_is_13_real_tiles_with_no_more_than_four_of_one():
    hand = crude_run.parse_hand(HAND, SLOTS)
    assert hand["S1"] == "1C" and hand["S13"] == "3C"
    for bad in ("1C,2C", ",".join(["1C"] * 12 + ["ZZ"]), ",".join(["1C"] * 13)):
        with pytest.raises(SystemExit):
            crude_run.parse_hand(bad, SLOTS)


def test_a_flat_wall_is_fed_every_draw_and_a_stacked_one_every_second():
    assert [(d["layer"], d["feed_before"]) for d in crude_run.draw_plan(3, False)] == [("flat", False), ("flat", True), ("flat", True)]
    assert [(d["layer"], d["feed_before"]) for d in crude_run.draw_plan(3, True)] == [("top", False), ("bottom", False), ("top", True)]


def test_the_summary_counts_breaks_per_step_and_scores_the_reader_only_where_it_guessed():
    breaks = [{"turn": 1, "step": "draw", "what": "missed", "by_hand": False}, {"turn": 2, "step": "draw", "what": "missed", "by_hand": True}]
    reads = [{"guess": "5C", "name": "5C", "face_hidden": False}, {"guess": "3C", "name": "2C", "face_hidden": False}, {"guess": None, "name": "9C", "face_hidden": True}]
    result = crude_run.summary(2, breaks, reads)
    assert result["breaks_per_step"] == {"draw": 2} and result["by_hand"] == 1
    assert (result["reader_right"], result["reader_guesses"], result["reads"], result["faces_not_visible"]) == (1, 2, 3, 1)
    assert any("turn 2 draw: missed (done by hand)" in line for line in crude_run.summary_lines(result, breaks))


def test_a_shallow_reading_is_between_empty_jaws_and_a_held_tile():
    assert crude_run.maybe_shallow(0.0, 3.3) and crude_run.maybe_shallow(0.0, 5.0)
    assert not crude_run.maybe_shallow(0.0, 0.9) and not crude_run.maybe_shallow(0.0, 7.5)      # empty; a proper hold
    assert not crude_run.maybe_shallow(20.0, 23.0)                                              # the jaws weren't shut


def test_a_slot_left_empty_is_never_the_discard_and_the_drawn_tile_always_can_be():
    candidates = [{"index": 6, "tile_name": "7C"}, {"index": 2, "tile_name": "3C"}, {"index": game.HAND_SIZE, "tile_name": "5C"}]
    assert crude_run.choose_allowed(candidates, SLOTS, ["S7"])["index"] == 2
    assert crude_run.choose_allowed(candidates, SLOTS, ["S7", "S3"])["index"] == game.HAND_SIZE
    with pytest.raises(SystemExit):
        crude_run.parse_slots("S99", SLOTS)
    assert crude_run.parse_slots(" s7 ,S3", SLOTS) == ["S7", "S3"]


def test_every_instruction_fits_the_camera_window():
    for step in (crude_run.flat_wall_step("F", 7), crude_run.flat_feed_step("F"), crude_run.place_step("the drop zone"), crude_run.place_step("S14")):
        assert all(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] <= 828 for line in step["lines"])


class FakeReader:
    classes = ["5C", "9C"]

    def read_many(self, crops):
        return [{"name": "5C", "confidence": 0.8}, {"name": "9C", "confidence": 0.9}, {"name": "5C", "confidence": 0.7}][:len(crops)]


def test_the_held_tiles_name_comes_from_every_pose_with_a_box():
    frames = [np.zeros((480, 640, 3), np.uint8)] * 3
    read = base_cam.read_held(FakeReader(), frames, ["a", "b", "c"], {"a": [0, 0, 100, 100], "b": [0, 0, 100, 100], "c": [0, 0, 100, 100]})
    assert read["name"] == "5C" and read["confidence"] == pytest.approx(0.5) and read["votes"]["9C"] == 0.9
    assert base_cam.read_held(None, frames, ["a"] * 3, {"a": [0, 0, 100, 100]}) is None
    assert base_cam.read_held(FakeReader(), frames, ["z"] * 3, {"a": [0, 0, 100, 100]}) is None


# --- the whole script on the simulated arm ---

def script_in(monkeypatch, tmp_path):
    import crude_run as script
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(config, "CAMERA_POSES", tmp_path / "poses.json")
    monkeypatch.setattr(base_cam, "READER", tmp_path / "none.pt")
    camera_poses.save("base_view_1", POSE, "test")
    camera_poses.save("base_view_2", POSE, "test")
    return script


def run(script, argv, answers):
    prompts, answers = [], iter(answers)

    def ask(text):
        prompts.append(text)
        return next(answers, "q")
    return script.main(["--sim", "--hand", HAND, *argv], ask=ask), prompts


def events_of(folder, kind):
    return [e for e in read_events(folder) if e["type"] == kind]


def test_a_simulated_turn_draws_shows_reads_and_plays_the_discard(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder, prompts = run(script, ["--turns", "1"], [*LIGHT, "", "", "5C", ""])       # lighting, lay the hand, lay the wall, the name, nothing else went wrong
    assert any("ONE tile high" in p for p in prompts) and any("Lay the hand GREEN side up" in p for p in prompts)
    assert [e["step"] for e in events_of(folder, "crude_set_down")][0] == "set_on_zone"
    (read,) = events_of(folder, "crude_read")
    assert read["name"] == "5C" and read["guess"] is None
    (choice,) = events_of(folder, "crude_choice")
    assert choice["drawn"] == "5C" and choice["discard"]
    assert [e["step"] for e in events_of(folder, "crude_pick")][0] == "draw" and "pick_from_zone" in [e["step"] for e in events_of(folder, "crude_pick")]
    assert not events_of(folder, "crude_break")
    summary = json.loads((folder / "crude_summary.json").read_text())
    assert summary["turns_done"] == 1 and summary["breaks"] == 0
    draw_dir = folder / events_of(folder, "crude_pick")[0]["attempt_dir"]
    assert len(list(draw_dir.glob("base_*.jpg"))) == 2           # one photo per saved pose


def test_the_hand_remembers_a_kept_tile_in_the_slot_it_left(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    monkeypatch.setattr(game, "choose_discard", lambda hand, drawn, seen: (None, [{"index": 2, "tile_name": hand[2], "shanten_after": 1}]))
    folder, _ = run(script, ["--turns", "2"], [*LIGHT, "", "", "5C", "", "", "9C", ""])
    choices = events_of(folder, "crude_choice")
    assert choices[0]["slot"] == "S3" and choices[0]["discard"] == "3C"
    assert choices[1]["discard"] == "5C"                         # S3 holds the tile drawn on turn 1
    steps = [e["step"] for e in events_of(folder, "crude_set_down")]
    assert steps.count("place_in_hand") == 2


def test_a_stacked_run_starts_top_tiles_nearer_the_robot_and_tells_you_to_feed(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    folder, _ = run(script, ["--turns", "2", "--stacked"], [*LIGHT, "", "", "5C", "", "9C", ""])
    draws = [e for e in events_of(folder, "crude_pick") if e["step"] == "draw"]
    assert [(e["hover_raise_m"], e["hover_back_m"]) for e in draws] == [(config.TILE_THICKNESS_M, 0.010), (0.0, 0.0)]


def test_a_top_tile_is_set_down_from_the_mats_height_not_its_own(monkeypatch, tmp_path):
    from mahjong.arm import Arm
    script = script_in(monkeypatch, tmp_path)
    real, placed = Arm.place, []

    def place(self, x, y, offset, grasp_clearance_m=None, **kw):
        placed.append((offset, grasp_clearance_m))
        return real(self, x, y, offset, grasp_clearance_m=grasp_clearance_m, **kw)
    monkeypatch.setattr(Arm, "place", place)
    folder, _ = run(script, ["--turns", "1", "--stacked"], [*LIGHT, "", "", "5C", ""])
    draw = next(e for e in events_of(folder, "crude_pick") if e["step"] == "draw")
    assert placed[0][0] < 0.07 and placed[0][1] == pytest.approx(draw["grasp_clearance_m"] - config.TILE_THICKNESS_M)


def test_a_miss_is_written_down_tried_again_and_can_be_done_by_hand(monkeypatch, tmp_path):
    from mahjong import face_down_rig
    script = script_in(monkeypatch, tmp_path)
    real, calls = face_down_rig.Rig.pick, []

    def missing(self, name, spot, raise_m=0.0, back_m=0.0):
        pick = real(self, name, spot, raise_m=raise_m, back_m=back_m)
        calls.append(name)
        if len(calls) == 1:          # the first draw comes up empty; let go of what the simulated policy always holds
            self.arm.set_gripper(90.0)
            pick["holding"] = False
        return pick
    monkeypatch.setattr(face_down_rig.Rig, "pick", missing)
    folder, _ = run(script, ["--turns", "1"], [*LIGHT, "", "", "", "5C", ""])          # ... Enter = try again
    (miss,) = events_of(folder, "crude_break")
    assert miss["step"] == "draw" and "missed" in miss["what"] and not miss["by_hand"]
    assert json.loads((folder / "crude_summary.json").read_text())["breaks_per_step"] == {"draw": 1}


def shallow_first_pick(monkeypatch, stop_reason="cap"):
    """The first pick reads as a shallow grip: Rig.pick says not held, and the arm goes on reading 3.3."""
    from mahjong import face_down_rig
    from mahjong.arm import Arm
    real, calls, shallow_now = face_down_rig.Rig.pick, [], [False]

    def pick(self, name, spot, raise_m=0.0, back_m=0.0):
        shallow_now[0] = False          # only the first pick is shallow
        out = real(self, name, spot, raise_m=raise_m, back_m=back_m)
        calls.append(name)
        if len(calls) == 1:
            out.update(holding=False, gripper_command=0.0, gripper_measured=3.3)
            out["run"]["stop_reason"] = stop_reason
            shallow_now[0] = True
        return out
    monkeypatch.setattr(face_down_rig.Rig, "pick", pick)
    holding = Arm.holding

    def holding_shallow(self):
        return (False, 0.0, 3.3) if shallow_now[0] else holding(self)
    monkeypatch.setattr(Arm, "holding", holding_shallow)
    return calls


def test_a_shallow_grip_a_person_confirms_carries_on_and_isnt_a_miss(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    calls = shallow_first_pick(monkeypatch)
    folder, prompts = run(script, ["--turns", "1"], [*LIGHT, "", "", "y", "5C", ""])
    assert any("Is there a tile in the jaws" in p for p in prompts)
    assert calls[0].endswith("draw_flat") and sum("draw_flat" in c for c in calls) == 1         # no second try, which would open the jaws
    assert not events_of(folder, "crude_break")
    first = events_of(folder, "crude_pick")[0]
    assert first["held"] is True and first["shallow"] is True and first["jaw_reading"] == 3.3
    assert [e["step"] for e in events_of(folder, "crude_set_down")][0] == "set_on_zone"
    assert json.loads((folder / "crude_summary.json").read_text())["shallow_grips"] == 1


def test_a_shallow_reading_with_no_tile_is_still_a_miss(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    shallow_first_pick(monkeypatch)
    folder, _ = run(script, ["--turns", "1"], [*LIGHT, "", "", "n", "", "5C", ""])
    assert [b["step"] for b in events_of(folder, "crude_break")] == ["draw"]


def test_a_tile_in_the_jaws_when_the_camera_drops_isnt_retried(monkeypatch, tmp_path):
    from mahjong import hardware
    script = script_in(monkeypatch, tmp_path)
    reopened = []
    monkeypatch.setattr(hardware, "reopen_wrist_camera", lambda cam: reopened.append(1) or cam)
    calls = shallow_first_pick(monkeypatch, stop_reason="camera")
    folder, _ = run(script, ["--turns", "1"], [*LIGHT, "", "", "y", "5C", ""])
    assert reopened == [1] and sum("draw_flat" in c for c in calls) == 1


def test_a_missed_draw_says_the_camera_never_saw_it_and_outlines_the_zone(monkeypatch, tmp_path, capsys):
    from mahjong import face_down_rig
    script = script_in(monkeypatch, tmp_path)
    real = face_down_rig.Rig.pick

    def missing(self, name, spot, raise_m=0.0, back_m=0.0):
        pick = real(self, name, spot, raise_m=raise_m, back_m=back_m)
        if "draw" in name:
            self.arm.set_gripper(90.0)
            pick["holding"] = False
        return pick
    monkeypatch.setattr(face_down_rig.Rig, "pick", missing)
    folder, prompts = run(script, ["--turns", "1"], [*LIGHT, "", "", "h", "", "", "3C", ""])     # h = by hand; the zone step; Enter isn't a name; then the name
    assert any("[by hand] Put the tile on the drop zone" in p for p in prompts)
    assert any("The camera never saw this tile" in p for p in prompts)
    out = capsys.readouterr().out
    assert "no guess to accept" in out and "Noted for the list: draw" in out and "BROKE" not in out
    assert events_of(folder, "crude_read")[0]["name"] == "3C"


def test_a_slot_left_empty_is_said_when_laying_the_hand_and_never_discarded(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    monkeypatch.setattr(game, "choose_discard", lambda hand, drawn, seen: (None, [{"index": 6, "tile_name": hand[6]}, {"index": 2, "tile_name": hand[2]}]))
    folder, prompts = run(script, ["--turns", "1", "--leave-empty", "S7"], [*LIGHT, "", "", "5C", ""])
    assert any("S7 7C (leave this slot EMPTY" in p for p in prompts)
    (choice,) = events_of(folder, "crude_choice")
    assert choice["slot"] == "S3"


def test_x_says_the_face_wasnt_visible_and_a_guess_at_it_isnt_scored(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    monkeypatch.setattr(base_cam, "load_reader", lambda: FakeReader())
    monkeypatch.setattr(base_cam, "read_held", lambda *a: {"name": "5C", "confidence": 0.9, "votes": {"5C": 0.9}})
    folder, _ = run(script, ["--turns", "1"], [*LIGHT, "", "", "x", "2C", ""])
    (read,) = events_of(folder, "crude_read")
    assert read["name"] == "2C" and read["face_hidden"] is True and read["guess"] is None
    assert [b["step"] for b in events_of(folder, "crude_break")] == ["read"]


def test_enter_accepts_the_readers_guess_and_it_is_scored(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path)
    monkeypatch.setattr(base_cam, "load_reader", lambda: FakeReader())
    monkeypatch.setattr(base_cam, "read_held", lambda *a: {"name": "5C", "confidence": 0.9, "votes": {"5C": 0.9}})
    folder, _ = run(script, ["--turns", "1"], [*LIGHT, "", "", "", ""])
    (read,) = events_of(folder, "crude_read")
    assert read["name"] == "5C" and read["guess"] == "5C"
    assert json.loads((folder / "crude_summary.json").read_text())["reader_right"] == 1


def test_a_name_that_isnt_a_tile_is_asked_again(monkeypatch, tmp_path, capsys):
    script = script_in(monkeypatch, tmp_path)
    folder, _ = run(script, ["--turns", "1"], [*LIGHT, "", "", "ZZ", "5C", ""])
    assert events_of(folder, "crude_read")[0]["name"] == "5C" and "isn't a tile name" in capsys.readouterr().out


def test_typing_over_the_photos_edits_and_returns_on_enter():
    import crude_run as script
    frames = [np.zeros((480, 640, 3), np.uint8)] * 2
    keys = iter([255, ord("9"), 255, ord("x"), 8, ord("c"), 13])       # 255: OpenCV's "no key this time"
    assert script.type_in_window(frames, None, show=lambda image: None, wait_key=lambda: next(keys)) == "9C"
    keys = iter([27])
    assert script.type_in_window(frames, {"name": "5C", "confidence": 0.9}, show=lambda image: None, wait_key=lambda: next(keys)) is None


@pytest.mark.parametrize("argv", [["--station", "X"], ["--turns", "0"], ["--layout", "nope"]])
def test_a_bad_request_fails_before_any_hardware(monkeypatch, tmp_path, argv):
    script = script_in(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        script.main(["--sim", "--hand", HAND, *argv], ask=lambda prompt: pytest.fail("asked a question"))


def test_saving_a_reader_trains_on_every_photo_and_writes_it(monkeypatch, tmp_path):
    import base_cam_train
    image = np.zeros((480, 640, 3), np.uint8)
    rows = []
    for i, name in enumerate(("5C", "9C")):
        path = tmp_path / f"{i}.png"
        cv2.imwrite(str(path), image)
        rows.append({"truth": name, "path": path, "session": "s", "pose": "base_view_1"})
    args = base_cam_train.parse_args(["--save-reader"])
    monkeypatch.setattr(base_cam, "READER", tmp_path / "reader.pt")
    seen = []

    def train(photos, seed, epochs, log, crop_fn):
        seen.extend(cv2.imread(str(p["path"])).shape for p in photos)
        return {"classes": ["5C", "9C"], "model": None}
    saved = base_cam_train.save_reader(rows, args, {"base_view_1": [0, 0, 100, 100]}, train)
    assert saved.exists() and len(seen) == 2 and seen[0][:2] == (name_check.INPUT_PX, name_check.INPUT_PX)
