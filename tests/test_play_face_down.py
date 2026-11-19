"""Face-down play (scripts/play_face_down.py, mahjong/face_down_play.py) on the simulated arm: the hand read by the arm and
checked once, turns from a wall two high until the wall's empty, the end check; --unattended asking nothing but the wall;
every hold saved for the reader and kept out of training until reviewed; --report and --resume."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import base_cam, camera_poses, config, face_down_play, interventions, photo_review
from mahjong.session import read_events

POSE = {j: 0.0 for j in config.ARM_JOINTS}
HAND = ["1C", "2C", "3C", "5C", "5C", "7C", "7C", "9C", "9C", "1B", "1B", "2C", "3C"]
SLOTS = [f"S{i}" for i in (*range(1, 13), 14)]         # even-hand13: even-hand14's ids without S13, the drop zone's spot
LIGHT = ["", "", ""]


class Reader:
    classes = ["5C", "9C"]


def script_in(monkeypatch, tmp_path, reads, unattended_reader=True):
    """The script with a sim's sessions in tmp_path and the reader's names in order, one per hold: a second look at the same
    hold (the same first frame) gets the same name, surer. reads: names, or (name, confidence)."""
    import play_face_down as script
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(config, "CAMERA_POSES", tmp_path / "poses.json")
    camera_poses.save("base_view_1", POSE, "test")
    camera_poses.save("base_view_2", POSE, "test")
    if reads is None:
        monkeypatch.setattr(base_cam, "READER", tmp_path / "none.pt")
        return script
    monkeypatch.setattr(base_cam, "load_reader", lambda: Reader())
    queue = [r if isinstance(r, tuple) or r is None else (r, 0.9) for r in reads]       # None: a hold the reader has no name for
    by_hold = {}

    def read_held(reader, frames, from_pose, boxes):
        if not frames:
            return None
        key = id(frames[0])         # the frame is kept below, so its id can't be reused by a later hold's
        if key not in by_hold:
            by_hold[key] = (queue.pop(0) if queue else ("9B", 0.9), len(frames), frames[0])
        read, first_look, _ = by_hold[key]
        if read is None:
            return None
        name, confidence = read
        if len(frames) > first_look:        # a second look's photos added: surer
            confidence = 0.95
        return {"name": name, "confidence": confidence, "votes": {name: confidence}}
    monkeypatch.setattr(base_cam, "read_held", read_held)
    return script


def run(script, argv, answers):
    prompts, answers = [], iter(answers)

    def ask(text):
        prompts.append(text)
        return next(answers, "q")
    return script.main(["--sim", "--station", "F", *argv], ask=ask), prompts


def events_of(folder, kind):
    return [e for e in read_events(folder) if e["type"] == kind]


def test_the_turns_come_from_a_wall_two_high_with_no_end():
    plan = face_down_play.turn_plan()
    first = [next(plan) for _ in range(5)]
    assert [(p["turn"], p["layer"], p["feed_before"]) for p in first] == [
        (1, "top", False), (2, "bottom", False), (3, "top", True), (4, "bottom", False), (5, "top", True)]
    assert next(face_down_play.turn_plan(4))["layer"] == "bottom"


def test_the_hand_sheet_and_the_laying_window_draw():
    from mahjong import layout
    slots = SLOTS
    hand = dict(zip(slots, HAND))
    guesses = {s: {"name": n, "confidence": 0.9} for s, n in hand.items()}
    guesses["S2"] = None
    sheet = face_down_play.hand_sheet(slots, {}, hand, guesses)
    assert sheet.shape[2] == 3 and sheet.shape[1] > 7 * 150
    spots = layout.load("even-hand13")["spots"]
    intrinsics = {"fx": 600.0, "fy": 600.0, "ppx": 424.0, "ppy": 240.0}
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 0.6]
    out = face_down_play.lay_hand_frame(np.zeros((480, 848, 3), np.uint8), spots, {"id": "L2", "xy": [0.2, 0.15]},
                                        {"id": "zone", "xy": [0.2, -0.17]}, intrinsics, T, face_down_play.lay_lines("L2", 4, slots))
    assert out.shape == (480, 848, 3)


def test_an_attended_hand_is_read_checked_once_played_and_checked_at_the_end(monkeypatch, tmp_path):
    reads = HAND[:2] + ["4C"] + HAND[3:] + ["4D", "8B"]       # S3 misread as 4C: the hand check says 3C
    script = script_in(monkeypatch, tmp_path, reads)
    answers = [*LIGHT, "",                  # lighting, the hand and the wall laid
               "S3=3C", "",                 # the hand check: one fix, then all right
               "", "",                      # turn 1: the draw's read accepted, nothing went wrong
               "", "end",                   # turn 2: accepted, then the wall's empty
               ""]                          # the end check: every tile matches
    folder, prompts = run(script, [], answers)
    assert any("GREEN side up" in p for p in prompts)
    assert len(events_of(folder, "fd_register_read")) == 13
    (confirmed,) = events_of(folder, "fd_hand_confirmed")
    assert confirmed["fixed"] == {"S3": "3C"} and confirmed["hand"]["S3"] == "3C"
    asked = [i for i in events_of(folder, "intervention") if i["kind"] == "name_corrected"]
    assert [(i["slot"], i["read"], i["name"]) for i in asked] == [("S3", "4C", "3C")]
    assert [c["drawn"] for c in events_of(folder, "discard_chosen")] == ["4D", "8B"]
    shots = events_of(folder, "holdup_shot")
    assert len(shots) == 15 and all(s["review_first"] and s["filed_by"] == "person" for s in shots)
    assert next(s for s in shots if s["spot"] == "S3")["tile"] == "3C"     # filed under the checked name, the reader's beside it
    assert base_cam.rows(folder) == []                                      # nothing for training until reviewed
    (end,) = events_of(folder, "hand_end")
    assert end["result"] == "end" and end["grade"] == "attended" and end["corrected"] == 1 and end["end_check"]
    assert json.loads((folder / "checkpoint.json").read_text())["phase"] == "end"
    summary = json.loads((folder / "play_fd_summary.json").read_text())
    assert summary["holds_waiting_for_review"] == 15 and summary["reads"] == 15


def test_the_hand_check_refuses_five_of_a_tile_and_a_name_that_isnt_one(monkeypatch, tmp_path, capsys):
    script = script_in(monkeypatch, tmp_path, HAND + ["4D"])
    answers = [*LIGHT, "", "S1=5C S2=5C S3=5C", "S1=ZZ", "S1=1C S2=2C S3=3C", "", "", "end", ""]
    folder, _ = run(script, [], answers)
    out = capsys.readouterr().out
    assert "more than 4 of 5C" in out and "ZZ isn't one" in out
    assert events_of(folder, "fd_hand_confirmed")[0]["hand"] == dict(zip(SLOTS, HAND))


def test_unattended_asks_only_for_the_hand_the_wall_and_the_end_check(monkeypatch, tmp_path):
    reads = HAND[:4] + [("5C", 0.4)] + HAND[5:] + ["4D", "8B"]        # S5 read unsure: shown again
    script = script_in(monkeypatch, tmp_path, reads)
    answers = [*LIGHT, "",                  # lighting, the hand and the wall laid
               "e",                         # turn 3's feed: the wall's empty
               "S1=9B"]                     # the end check: S1 was a 9B, read as 1C (a silent error)
    folder, prompts = run(script, ["--unattended"], answers)
    assert len(prompts) == len(answers)
    assert "GREEN side up" in prompts[3] and "Feed the wall" in prompts[4] and "Turn the 13 over" in prompts[5]     # no name asked
    (reshow,) = events_of(folder, "fd_reshow")
    assert reshow["first"]["name"] == "5C" and reshow["first"]["confidence"] == 0.4
    shots = events_of(folder, "holdup_shot")
    assert len(shots) == 15 and all(s["filed_by"] == "reader" and s["review_first"] for s in shots)
    s5 = next(s for s in shots if s["spot"] == "S5")
    assert len(s5["files"]) == 4                                            # both holds' photos
    (end,) = events_of(folder, "hand_end")
    assert end["mode"] == "unattended" and end["grade"] == "silent error" and end["silent_errors"] == 1

    # the owner reviews the photos: S1's are a 9B, the rest as filed. Then they train, and --report scores the reads.
    review = photo_review.load_review(base_cam.REVIEW)
    photos = base_cam.review_photos([folder])
    assert not review and len(photos) == 2 * 15
    s1 = [p["key"] for p in photos if p["spot"] == "S1"]
    photo_review.record(s1, "wrong", "9B", base_cam.REVIEW)
    photo_review.record([p["key"] for p in photos if p["spot"] != "S1"], "ok", None, base_cam.REVIEW)
    assert {r["truth"] for r in base_cam.rows(folder) if r["spot"] == "S1"} == {"9B"} and len(base_cam.rows(folder)) == 32
    result = script.main(["--report"])
    wrong = [r for r in result["reads"] if r["guess"] != r["truth"]]
    assert [(r["where"], r["guess"], r["truth"]) for r in wrong] == [("S1", "1C", "9B")]


def test_a_wrong_drawn_read_is_listed_with_the_discard_it_changed(monkeypatch, tmp_path):
    reads = HAND + ["9C"]
    script = script_in(monkeypatch, tmp_path, reads)
    folder, _ = run(script, ["--unattended"], [*LIGHT, "", "q"])      # turns 1 and 2 ask nothing; quit at turn 3's feed
    shot = next(s for s in events_of(folder, "holdup_shot") if s["spot"] == "draw t1")
    keys = [p["key"] for p in base_cam.review_photos([folder]) if p["shot"] == shot["shot"]]
    photo_review.record(keys, "wrong", "4D", base_cam.REVIEW)            # really a 4D
    reads_scored = face_down_play.scored_reads(folder)
    # read as a third 9C the hand threw the 9C; as the 4D it was, the lone 4D would have gone
    assert face_down_play.changed_choices(folder, reads_scored) == [
        {"turn": 1, "read": "9C", "truth": "4D", "discarded": "9C", "would_discard": "4D"}]
    lines = face_down_play.report_lines(folder, reads_scored, face_down_play.changed_choices(folder, reads_scored))
    assert any("draw t1: read 9C (90%), really 4D, the hand used the reader's name" in line for line in lines)
    assert any("turn 1: read 9C (really 4D), threw 9C; would have thrown 4D" in line for line in lines)


def test_a_hand_carries_on_from_its_last_turn_with_resume(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path, HAND + ["4D", "8B"])
    first, _ = run(script, [], [*LIGHT, "", "", "", "q"])          # turn 1, then quit after it
    state = json.loads((first / "checkpoint.json").read_text())
    assert state["phase"] == "play" and state["turn"] == 1
    second, prompts = run(script, ["--resume"], [*LIGHT, "", "end", ""])
    assert not any("GREEN side up" in p for p in prompts)            # the tiles are where they were: nothing laid, nothing read
    (resumed,) = events_of(second, "fd_resumed")
    assert resumed["from_session"] == first.name and resumed["hand"] == state["hand"]
    assert [c["turn"] for c in events_of(second, "discard_chosen")] == [2]
    assert not events_of(second, "fd_register_read")
    with pytest.raises(SystemExit, match="is over"):
        script.main(["--sim", "--resume", second.name], ask=lambda prompt: "")


@pytest.mark.parametrize("argv", [["--station", "X"], ["--wall-stacks", "0"], ["--layout", "even-hand14"]])
def test_a_bad_request_fails_before_any_hardware(monkeypatch, tmp_path, argv):
    script = script_in(monkeypatch, tmp_path, HAND)
    with pytest.raises(SystemExit):
        script.main(["--sim", *argv], ask=lambda prompt: pytest.fail("asked a question"))


def test_unattended_needs_a_reader(monkeypatch, tmp_path):
    script = script_in(monkeypatch, tmp_path, None)
    with pytest.raises(SystemExit, match="no reader"):
        script.main(["--sim", "--unattended"], ask=lambda prompt: pytest.fail("asked a question"))


def test_a_step_done_by_hand_is_a_rescue(monkeypatch, tmp_path):
    from mahjong import face_down_rig
    script = script_in(monkeypatch, tmp_path, HAND + ["4D"])
    real_pick = face_down_rig.Rig.pick

    def pick(self, name, spot, raise_m=0.0, back_m=0.0):
        result = real_pick(self, name, spot, raise_m, back_m)
        return {**result, "holding": False} if name == "reg_S4" else result
    monkeypatch.setattr(face_down_rig.Rig, "pick", pick)
    monkeypatch.setattr(face_down_rig.Rig, "set_down", lambda self, *a, **k: ({"ok": True}, (0.0, 0.0)))
    answers = [*LIGHT, "", "", "",           # lay; S4 missed: try again; missed again, the by-hand prompt (leave it on S4)
               "5C",                         # S4 has no read: its name, from the tile
               "", "", "end", ""]            # hand check; turn 1's read; the wall's empty; end check
    folder, _ = run(script, [], answers)
    assert any(i["kind"] == "fd_by_hand" for i in events_of(folder, "intervention"))
    assert interventions.class_of("fd_by_hand") == interventions.RESCUE
    assert events_of(folder, "hand_end")[0]["grade"] == "rescued"


def test_a_name_a_person_typed_unattended_is_filed_as_theirs_not_the_readers(monkeypatch, tmp_path):
    reads = HAND[:3] + [None] + HAND[4:] + ["4D"]                           # no read for S4
    script = script_in(monkeypatch, tmp_path, reads)
    folder, _ = run(script, ["--unattended"], [*LIGHT, "", "5C", "e", ""])     # lay; S4's name; the wall's empty at turn 3; end check
    shots = {s["spot"]: s for s in events_of(folder, "holdup_shot")}
    assert shots["S4"]["filed_by"] == "person" and shots["S4"]["tile"] == "5C" and shots["S4"]["guess"] is None
    assert shots["S3"]["filed_by"] == "reader"


def test_a_correction_after_a_turn_cant_make_a_fifth_copy(monkeypatch, tmp_path, capsys):
    script = script_in(monkeypatch, tmp_path, HAND + ["4D"])
    # the hand holds 1C 2C 3C 5C 5C 7C 7C 9C 9C 1B 1B 2C 3C and the turn discards the drawn 4D: S1=5C S2=5C S3=5C would be five 5Cs
    answers = [*LIGHT, "", "", "", "S1=5C S2=5C S3=5C", "S1=5C", "end", ""]       # lay; hand check; turn 1's read; the fixes; then end; end check
    folder, _ = run(script, [], answers)
    assert "That would be more than 4 of 5C" in capsys.readouterr().out
    assert json.loads((folder / "checkpoint.json").read_text())["hand"]["S1"] == "5C"      # the one fix that fit was taken
    assert [e["fixes"] for e in events_of(folder, "fd_hand_fixed")] == [{"S1": "5C"}]


def test_a_resume_after_a_quit_mid_turn_has_the_drawn_tile_put_back_first(monkeypatch, tmp_path):
    """A quit at a draw's read leaves the drawn tile on the drop zone and the checkpoint at the turn before: the resume drew
    that turn again from a wall missing it, onto a zone with a tile on it, and said nothing (post-merge review of #125)."""
    script = script_in(monkeypatch, tmp_path, HAND + ["4D", "4D", "8B"])
    first, _ = run(script, [], [*LIGHT, "", "", "q"])                # quit at turn 1's read
    assert json.loads((first / "checkpoint.json").read_text())["turn"] == 0
    second, prompts = run(script, ["--resume"], [*LIGHT, "", "", "", "", "end", ""])
    asked = [p for p in prompts if "Turn 1 stopped before its discard" in p]
    assert asked and "back on the wall, on top of the stack on station F" in asked[0]
    (left,) = events_of(second, "fd_resume_unfinished")
    assert left["turn"] == 1 and left["choice"] is None
    assert [c["turn"] for c in events_of(second, "discard_chosen")] == [1, 2]


def test_a_resumed_hand_s_checkpoint_keeps_its_turn_until_the_next_one_finishes(monkeypatch, tmp_path):
    """The resume writes a checkpoint before its first turn: it said turn 0, so a stop in that turn sent the next --resume
    back to turn 1 (code review of the post-merge fixes)."""
    script = script_in(monkeypatch, tmp_path, HAND + ["4D", "8B", "9B"])
    first, _ = run(script, [], [*LIGHT, "", "", "", "q"])             # turn 1, then quit after it
    second, _ = run(script, ["--resume"], [*LIGHT, "q"])              # quit at turn 2's read
    assert json.loads((second / "checkpoint.json").read_text())["turn"] == 1
    while (tmp_path / f"{datetime.now():%Y%m%d_%H%M%S}_{face_down_play.PLAY_KIND}").exists():
        time.sleep(0.05)            # a session folder is named to the second: three runs can share one
    third, prompts = run(script, ["--resume", second.name], [*LIGHT, "", "", "end", ""])
    assert any("Turn 2 stopped before its discard" in p for p in prompts)
    assert [c["turn"] for c in events_of(third, "discard_chosen")] == [2]


def test_an_unfinished_turn_is_found_from_the_log_and_a_chosen_discard_is_kept():
    start = {"type": "fd_turn_start", "turn": 3}
    pick = {"type": "crude_pick", "turn": 3, "step": "draw", "held": True}
    choice = {"type": "crude_choice", "turn": 3, "drawn": "4D", "discard": "9C", "slot": "S8"}
    assert face_down_play.unfinished_turn([start], 2) is None                   # quit at the feed: nothing moved
    assert face_down_play.unfinished_turn([start, pick], 3) is None             # that turn finished
    missed = {"type": "crude_pick", "turn": 3, "step": "draw", "held": False}
    assert face_down_play.unfinished_turn([start, missed], 2) == {"turn": 3, "layer": "top", "choice": None, "drawn": False}
    assert face_down_play.unfinished_lines({"turn": 3, "layer": "top", "choice": None, "drawn": False}, "L2") == []
    assert face_down_play.unfinished_turn([start, missed, pick], 2) == {"turn": 3, "layer": "top", "choice": None, "drawn": True}
    by_hand = {"type": "crude_read", "turn": 3, "name": "4D"}           # the draw done by hand: the tile reached the zone
    assert face_down_play.unfinished_turn([start, missed, by_hand], 2)["drawn"]
    left = face_down_play.unfinished_turn([start, pick, choice], 2)              # an older log's crude_choice
    assert left["choice"] == {"turn": 3, "drawn": "4D", "discard": "9C", "slot": "S8"}
    chosen = {"type": "discard_chosen", "turn": 3, "drawn": "4D", "tile_name": "9C", "slot": "S8"}
    assert face_down_play.unfinished_turn([start, pick, chosen], 2)["choice"] == left["choice"]
    (line,) = face_down_play.unfinished_lines(left, "L2")
    assert "9C (from S8) in the discard container, the drawn 4D on S8" in line


WINS_ON_5D = ["1C", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "1B", "2B", "3B", "5D"]


def test_a_pickup_stopped_with_esc_counts_against_the_hand_as_on_the_bench(monkeypatch, tmp_path):
    """Face-down picks are logged as `attempt`s (play_log.pickup_fields), so interventions.derive sees an Esc as it does on
    the bench and in face-up play. Before, they were only `crude_pick`s, and the hand graded clean."""
    from mahjong import face_down_rig
    real, calls = face_down_rig.Rig.pick, []

    def pick(self, name, spot, raise_m=0.0, back_m=0.0):
        out = real(self, name, spot, raise_m=raise_m, back_m=back_m)
        calls.append(name)
        if len(calls) == 2:             # the second tile of the hand: a person stopped the policy, and it held anyway
            out["run"]["stop_reason"] = "aborted"
        return out
    monkeypatch.setattr(face_down_rig.Rig, "pick", pick)
    script = script_in(monkeypatch, tmp_path, WINS_ON_5D + ["5D"])
    folder, _ = run(script, ["--unattended"], [*LIGHT, "", ""])
    (end,) = events_of(folder, "hand_end")
    assert end["result"] == "win" and end["grade"] == "rescued" and end["rescues"] == 1
    (aborted,) = [i for i in interventions.derive(read_events(folder)) if i["kind"] == "aborted"]
    assert aborted["attempt_dir"] == events_of(folder, "attempt")[1]["attempt_dir"]


def test_a_win_s_drawn_tile_is_in_the_end_check(monkeypatch, tmp_path):
    """The winning tile lies on the drop zone, not a slot: unattended, a misread there is a false win, and the end check has
    to be able to say so (post-merge review of #125: it only took slots, and graded such a hand clean)."""
    script = script_in(monkeypatch, tmp_path, WINS_ON_5D + ["5D"])
    folder, prompts = run(script, ["--unattended"], [*LIGHT, "", "ZONE=6D"])
    assert any("drop zone" in p for p in prompts if "Turn the" in p)
    (end,) = events_of(folder, "hand_end")
    assert end["result"] == "win" and end["winning"] == "5D"
    assert end["silent_errors"] == 1 and end["grade"] == "silent error"
    (error,) = [i for i in events_of(folder, "intervention") if i["kind"] == "silent_error"]
    assert (error["slot"], error["robot"], error["table"]) == ("zone", "5D", "6D")


def test_the_end_check_s_retry_hint_names_the_zone_after_a_win(monkeypatch, tmp_path, capsys):
    script = script_in(monkeypatch, tmp_path, WINS_ON_5D + ["5D"])
    run(script, ["--unattended"], [*LIGHT, "", "ZONE 6D", ""])
    assert "or ZONE=NAME" in capsys.readouterr().out
