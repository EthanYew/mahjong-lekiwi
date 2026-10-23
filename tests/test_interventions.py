"""Interventions derived from a session's events: classes, marks, and what scans say about the mat."""
import pytest

from mahjong import interventions as iv


def tile(name, x, y, confidence=0.9):
    return {"tile_name": name, "xy": [x, y], "confidence": confidence, "reads": 5}


MAT = [tile("WW", 0.224, 0.021), tile("2D", 0.225, -0.035), tile("9D", 0.216, 0.080)]


def scan(label, tiles, **fields):
    return {"type": "detections", "label": label, "tiles": tiles, **fields}


def attempt(n, success=True, **fields):
    return {"type": "attempt", "attempt_dir": f"{n:03d}_WW", "tile": "WW", "success": success,
            "gripper_says_held": success, "stop_reason": "done", "marks_s": [], **fields}


def session(*events, ended=True):
    return [{"type": "session_start"}, *events] + ([{"type": "session_end", "reason": "quit"}] if ended else [])


def kinds(events):
    return [(i["kind"], i["class"]) for i in iv.derive(events)]


def test_a_clean_session_has_no_interventions():
    events = session(scan("choose", MAT), scan("after_pickup", MAT[1:], attempt_dir="001_WW", target_xy=MAT[0]["xy"]),
                     attempt(1), {"type": "return", "tile": "WW"},
                     scan("after_return", MAT, attempt_dir="001_WW", return_xy=MAT[0]["xy"]), scan("choose", MAT))
    assert iv.derive(events) == []


def test_abort_override_and_reason_from_a_later_review():
    events = session(attempt(1, success=False, stop_reason="aborted"),
                     {"type": "attempt_review", "attempt_dir": "001_WW", "abort_reason": "frozen"},
                     attempt(2, success=True, gripper_says_held=False))
    aborted, overridden = iv.derive(events)
    assert (aborted["kind"], aborted["class"], aborted["reason"], aborted["attempt_dir"]) == \
        ("aborted", "rescue", "frozen", "001_WW")
    assert (overridden["kind"], overridden["class"]) == ("success_overridden", "override")


def test_teaching_the_discard_pose_is_setup_not_an_intervention():
    assert kinds(session(attempt(1), {"type": "discard"}, {"type": "discard_pose_taught"}, attempt(2))) == []


@pytest.mark.parametrize("mark_kind, expected", [("look", "observation"), ("touched_tile", "rescue"), (None, "rescue")])
def test_a_mark_takes_its_class_from_the_answer(mark_kind, expected):
    events = session(attempt(1, marks_s=[6.2, 7.0], **({"mark_kind": mark_kind} if mark_kind else {})))
    (mark,) = iv.derive(events)
    assert mark["kind"] == (mark_kind or "unspecified") and mark["class"] == expected
    assert mark["source"] == "mark" and mark["marks_s"] == [6.2, 7.0]


def test_a_success_with_a_rescue_mark_is_assisted_not_clean():
    attempts = [attempt(1, marks_s=[3.0], mark_kind="touched_tile"), attempt(2, marks_s=[3.0], mark_kind="look"),
                attempt(3)]
    s = iv.summary(iv.derive(session(*attempts)), attempts)
    assert (s["successes"], s["clean_successes"], s["assisted_successes"]) == (3, 2, 1)
    assert s["rescues"] == 1 and s["rescues_per_10"] == pytest.approx(10 / 3) and s["attempts_per_rescue"] == 3
    assert s["observations"] == 1


def test_a_neighbour_moved_during_the_pickup_is_knocked():
    after = [tile("2D", 0.225, -0.027), MAT[2]]          # WW picked up, 2D pushed 8mm
    events = session(scan("choose", MAT), scan("after_pickup", after, attempt_dir="001_WW", target_xy=MAT[0]["xy"]),
                     attempt(1))
    (knock,) = iv.derive(events)
    assert (knock["kind"], knock["class"], knock["during"], knock["attempt_dir"]) == ("knocked", "failure", "pickup", "001_WW")
    assert [m["tile_name"] for m in knock["changes"]["moved"]] == ["2D"]
    assert knock["changes"]["gone"] == []                # the target leaving isn't a knock


def test_the_nudge_seen_on_2026_10_23_counts_as_knocked():
    before = [tile("7B", 0.193, 0.186), tile("8D", 0.207, 0.141), tile("9D", 0.138, 0.123)]
    after = [tile("8D", 0.211, 0.136), tile("9D", 0.138, 0.123)]
    diff = iv.scene_diff(before, after, ignore=[before[0]["xy"]])
    assert [m["tile_name"] for m in diff["moved"]] == ["8D"] and diff["moved"][0]["moved_m"] == pytest.approx(0.0064, abs=1e-4)


def test_jitter_label_flips_flicker_and_phantoms_are_not_changes():
    jittered = [tile("WW", 0.2255, 0.0215), tile("8D", 0.225, -0.034),       # 1.5mm; 2D read as 8D
                tile("9D", 0.216, 0.080), tile("3B", 0.150, 0.004, confidence=0.52),   # low-confidence flicker
                tile("8D", 0.478, 0.232)]                                          # out of reach
    assert iv.scene_diff(MAT, jittered) == {"moved": [], "gone": [], "appeared": []}


def test_a_tile_put_back_while_idle_is_a_reset():
    events = session(scan("choose", MAT), scan("after_pickup", MAT[1:], attempt_dir="001_WW", target_xy=MAT[0]["xy"]),
                     attempt(1), {"type": "discard", "tile": "WW"}, scan("after_discard", MAT[1:], attempt_dir="001_WW"),
                     scan("choose", MAT))
    (reset,) = iv.derive(events)
    assert (reset["kind"], reset["class"], reset["source"], reset["attempt_dir"]) == ("tile_reset", "reset", "scans", "001_WW")
    assert [t["tile_name"] for t in reset["changes"]["appeared"]] == ["WW"]


def test_a_reset_both_noted_and_seen_counts_once():
    events = session(scan("after_discard", MAT[1:], attempt_dir="001_WW"), attempt(1),
                     {"type": "intervention", "kind": "tile_reset", "attempt_dir": "001_WW", "note": "WW back"},
                     scan("choose", MAT))
    (reset,) = iv.derive(events)
    assert (reset["kind"], reset["source"], reset["note"]) == ("tile_reset", "logged", "WW back")


def test_moving_a_knocked_tile_back_is_straightening():
    knocked = [tile("2D", 0.225, -0.027), MAT[2]]
    events = session(scan("choose", MAT), scan("after_pickup", knocked, attempt_dir="001_WW", target_xy=MAT[0]["xy"]),
                     attempt(1), scan("after_return", [MAT[0]] + knocked, attempt_dir="001_WW", return_xy=MAT[0]["xy"]),
                     scan("choose", MAT))
    assert kinds(events) == [("knocked", "failure"), ("straightened", "rescue")]


def test_how_a_session_ended():
    assert kinds(session(attempt(1), ended=False)) == [("killed", "rescue")]
    crashed = session(attempt(1), ended=False) + [{"type": "session_end", "reason": "RuntimeError", "error": "motors"}]
    assert kinds(crashed) == [("crashed", "rescue")]
    interrupted = session(ended=False) + [{"type": "session_end", "reason": "KeyboardInterrupt"}]
    assert kinds(interrupted) == [("interrupted", "rescue")]


def test_logged_interventions_and_skips():
    events = session(attempt(1), {"type": "intervention", "kind": "discard_missed", "attempt_dir": None},
                     {"type": "attempt_skipped", "reason": "unreachable", "tile": "7B"},
                     {"type": "attempt_skipped", "reason": "cancelled"},
                     {"type": "intervention", "kind": "hover_rejected", "tile": "3D"},
                     {"type": "connect", "arm_at_rest": False, "off_rest_deg": 12.0})
    derived = iv.derive(events)
    assert [(i["kind"], i["class"]) for i in derived] == [
        ("discard_missed", "rescue"), ("rescan", "rescue"), ("cancelled", "observation"),
        ("hover_rejected", "rescue"), ("arm_moved_while_off", "observation")]
    assert derived[0]["attempt_dir"] == "001_WW"          # attributed to the most recent attempt
    assert derived[-1]["attempt_dir"] is None


def answering(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_asking_about_a_mark():
    assert iv.ask_mark("attempt 3", ask=answering("1")) == ("touched_tile", "")
    assert iv.ask_mark("attempt 3", ask=answering("")) == ("unspecified", "")
    assert iv.ask_mark("attempt 3", ask=answering("ca")) == ("caught_drop", "")
    assert iv.ask_mark("attempt 3", ask=answering("9", "5", "tile slid off the mat")) == ("other", "tile slid off the mat")
    assert iv.ask_abort("attempt 3", ask=answering("1")) == "frozen"
    assert iv.ask_abort("attempt 3", ask=answering("")) is None
