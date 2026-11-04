"""Play's direct moves: a held tile goes straight from its pickup to the discard or the slot. Everything else goes via rest, a
look with the wrist camera included, and the scan after the release still answers for the pickup."""
import builtins
import math
import sys
from pathlib import Path

import pytest

from mahjong import arm as arm_module, config, hand_truth, interventions, kinematics
from mahjong.arm import Arm
from mahjong.sim import SimRobot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_play import Answers, events_of, player, run


class Reader:
    """Echoes the name the robot's going on, except `wrong`: {that name: what it says instead}."""
    def __init__(self, p, wrong=None):
        self.p, self.wrong = p, wrong or {}

    def read(self, frame):
        name = self.p.picking or self.p.state["drawn"]
        return {"name": self.wrong.get(name, name), "confidence": 0.9}


def trips_to_rest(p):
    """What the arm was doing each time it went to rest from elsewhere (the phase of the turn, whether it held a tile)."""
    trips, rest = [], p.bench.arm.rest

    def counted():
        if not p.bench.arm.is_at_rest():
            trips.append((p.state["phase"], p.bench.arm.holding()[0]))
        rest()
    p.bench.arm.rest = counted
    return trips


def scan_labels(p):
    return [e["label"] for e in events_of(p) if e["type"] == "detections"]


def scans(p, label):
    return [e for e in events_of(p) if e["type"] == "detections" and e["label"] == label]


def test_a_held_tile_goes_straight_on_and_the_arm_rests_after_each_release(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    trips = trips_to_rest(p)
    assert run(p) == "clean"                                # nothing the scans saw looked like a person or a bump
    assert not [t for t in trips if t[1]], "went to rest holding a tile"
    # GD from the zone to the bin; RD from H12 to the bin, then 9B from the zone to H12: a rest after each release, nothing between
    assert len(trips) == 3
    assert scan_labels(p)[3:] == ["drawn", "drawn", "choose", "after_pickup",
                                  "drawn", "drawn", "choose", "after_pickup", "choose", "after_return",
                                  "drawn", "drawn"]


def test_the_scan_after_a_release_answers_for_the_pickup(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    run(p)
    attempts = {e["attempt_dir"]: e for e in events_of(p) if e["type"] == "attempt"}
    discards = scans(p, "after_pickup")
    assert all(s["released"] and s["attempt_dir"] in attempts for s in discards)
    assert all(s["target_xy"] == list(attempts[s["attempt_dir"]]["detection_xy"]) for s in discards)
    back = scans(p, "after_return")[0]
    assert back["handled_xy"] == [list(attempts[back["attempt_dir"]]["detection_xy"])] and "released" not in back
    assert all((p.session.dir / name / "overview_after.jpg").exists() for name in attempts)


def test_via_rest_keeps_a_rest_and_a_scan_after_every_pickup(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers(), argv=("--sim", "--via-rest"))
    trips = trips_to_rest(p)
    assert run(p) == "clean"
    assert len(trips) == 6 and sum(held for _, held in trips) == 3
    assert scan_labels(p).count("after_pickup") == 3 and scan_labels(p).count("after_discard") == 2
    assert not any(s.get("released") for s in scans(p, "after_pickup"))
    assert next(e for e in events_of(p) if e["type"] == "play_setup")["direct"] is False


def test_direct_moves_take_less_arm_time(tmp_path, monkeypatch):
    def arm_time(folder, argv):
        p = player(tmp_path / folder, monkeypatch, Answers(), argv=argv)
        p.bench.arm.rest()
        p.register()
        p.confirm_hand()
        seconds = []
        p.bench.arm.sleep = seconds.append
        while p.state["turn"] < 2:
            p.turn()
        p.close()
        return sum(seconds)
    assert arm_time("direct", ("--sim",)) < arm_time("rest", ("--sim", "--via-rest")) - 10


def test_a_bump_during_a_direct_pickup_is_a_knock_not_a_person(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    monkeypatch.setattr(builtins, "input", lambda prompt: "")     # as the scans say
    discard = p.bench.discard

    def knocks_a_neighbour(*args, **kwargs):
        if p.state["discard_slot"] == "H12":                      # H11 is the next tile along
            neighbour = min(p.mat.tiles, key=lambda t: math.dist(t["xy"], p.home("H11")))
            neighbour["xy"] = [neighbour["xy"][0] + 0.009, neighbour["xy"][1]]
        return discard(*args, **kwargs)
    monkeypatch.setattr(p.bench, "discard", knocks_a_neighbour)
    run(p)
    reviews = [e for e in events_of(p) if e["type"] == "scan_review" and e.get("kinds")]
    assert any("knocked" in r["kinds"] for r in reviews)
    assert not any(set(r["kinds"]) & {"tile_reset", "straightened"} for r in reviews)


def test_a_miss_goes_to_rest_and_aims_again_from_a_new_scan(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    run_policy, misses = p.bench.policy.run, []

    def misses_once(arm, camera, trace_dir, **kwargs):
        result = run_policy(arm, camera, trace_dir, **kwargs)
        if p.state["phase"] == "discard" and not misses:
            misses.append(1)
            arm.robot.holding = False                             # slipped out, back on the mat
            p.mat.tiles.append({"tile_name": p.state["discard"]["tile_name"], "xy": list(p.home(p.state["discard_slot"]))})
        return result
    monkeypatch.setattr(p.bench.policy, "run", misses_once)
    assert run(p) == "clean"
    attempts = [e for e in events_of(p) if e["type"] == "attempt"]
    assert not attempts[0]["gripper_says_held"] and attempts[1]["success"]
    assert scan_labels(p)[3:7] == ["drawn", "drawn", "choose", "after_pickup"]
    assert not scans(p, "after_pickup")[0].get("released")        # taken at once, with no tile let go


def test_a_key_pressed_during_a_pickup_gets_its_scan_straight_away(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    monkeypatch.setattr(builtins, "input", lambda prompt: "")
    run_policy, pressed = p.bench.policy.run, []

    def presses(arm, camera, trace_dir, **kwargs):
        if not pressed:
            pressed.append(1)
            p.bench.keys.press("b")
        return run_policy(arm, camera, trace_dir, **kwargs)
    monkeypatch.setattr(p.bench.policy, "run", presses)
    run(p)
    assert scan_labels(p).count("after_discard") == 1             # that one went via rest; the next turn's didn't


def test_a_corrected_discard_rests_and_scans_before_the_new_one(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers("", "GD"))          # the tile at H12 is a GD
    p.reader = Reader(p, {"RD": "GD"})
    run(p)
    events = events_of(p)
    off = next(i for i, e in enumerate(events) if e["type"] == "pickup_called_off")
    assert next(e for e in events[off:] if e["type"] in ("detections", "attempt"))["type"] == "detections"


def test_a_look_at_the_drawn_tile_goes_back_to_rest_before_anything_else(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers("", "7C"))          # the hand's fine; the drawn tile is the 7C it reads
    p.reader = Reader(p)
    p.draws = ["7C"]
    p.bench.arm.rest()
    p.register()
    p.confirm_hand()
    hover, hovers = p.bench.arm.hover, []
    monkeypatch.setattr(p.bench.arm, "hover", lambda *a, **k: hovers.append(p.bench.arm.is_at_rest()) or hover(*a, **k))
    p.turn()
    p.close()
    assert p.state["discards"] == ["7C"] and hovers == [True, True]   # the look, then the pickup, each setting out from rest


def test_handled_points_are_this_scans_target_and_a_pickup_with_no_scan():
    scan = {"label": "after_return", "target_xy": [1, 2], "handled_xy": [[3, 4]]}
    assert interventions.handled_points(scan) == [[1, 2], [3, 4]]
    assert interventions.handled_points({"label": "choose", "handled_xy": [[3, 4]]}) == []


def test_replaying_a_carry_leaves_both_of_its_ends_out_of_the_changes():
    tile = lambda name, x, y: {"tile_name": name, "xy": [x, y], "confidence": 0.9}
    before = {"label": "choose", "scan": 1, "tiles": [tile("9B", 0.25, 0.0), tile("1D", 0.15, -0.1)]}
    after = {"label": "after_return", "scan": 2, "target_xy": [0.2, 0.1], "handled_xy": [[0.25, 0.0]],
             "tiles": [tile("9B", 0.2, 0.1), tile("1D", 0.15, -0.1)]}
    watch = interventions.MatWatch()
    watch.advance(before)
    assert watch.propose(after) is None


def test_the_detector_still_gets_a_discards_scan_but_not_one_with_the_tile_in_the_jaws(tmp_path, monkeypatch):
    """A direct discard's scan is labelled after_pickup, which hand_truth leaves out because the tile's usually held. Not this one."""
    p = player(tmp_path, monkeypatch, Answers())
    run(p)
    direct = {s["label"] for s in hand_truth.scans(p.session.dir)}
    assert "after_pickup" in direct
    q = player(tmp_path / "via_rest", monkeypatch, Answers(), argv=("--sim", "--via-rest"))
    run(q)
    assert "after_pickup" not in {s["label"] for s in hand_truth.scans(q.session.dir)}


# --- the moves themselves ---

def lifted_over(x, y):
    robot = SimRobot(start=kinematics.hover_command(x, y, 0.08, 11.0)[0], holding=True)
    return Arm(robot, sleep=lambda s: None), robot


def test_a_direct_place_arrives_with_the_pan_turning_up_as_it_does_from_rest():
    arm, robot = lifted_over(0.142, -0.114)                       # pan well above where it's going
    start_pan = arm.commanded()["shoulder_pan"]
    robot.sent.clear()
    assert arm.place(0.192, 0.09, 0.045, direct=True) is not None
    above = kinematics.hover_command(0.192, 0.09, 0.095, 11.0)[0]
    assert above["shoulder_pan"] < start_pan
    pans = [c["shoulder_pan"] for c in robot.sent]
    i = pans.index(min(pans[:len(pans) // 2]))
    assert pans[i] == pytest.approx(above["shoulder_pan"] - arm_module.TRAVEL_PAN_BACK_DEG, abs=0.5)
    assert pans[i + 1] > pans[i]


def test_a_direct_place_is_timed_by_its_distance_and_one_from_rest_isn_t():
    def seconds(arm, **kwargs):
        spent = []
        arm.sleep = spent.append
        arm.place(0.192, 0.039, 0.045, **kwargs)
        return sum(spent)
    assert seconds(lifted_over(0.192, 0.09)[0], direct=True) < seconds(lifted_over(0.192, 0.09)[0]) - 1.5
    rest_start = {**config.REST_POSE, "gripper": 11.0}
    assert seconds(Arm(SimRobot(start=rest_start, holding=True), sleep=lambda s: None), direct=True) \
        == pytest.approx(seconds(Arm(SimRobot(start=rest_start, holding=True), sleep=lambda s: None)))


@pytest.mark.parametrize("a,b", [((0.192, 0.141), (0.142, -0.165)), ((0.142, 0.141), (0.192, -0.165)),
                                 ((0.142, -0.114), (0.192, 0.09))])
def test_a_held_tile_stays_over_the_others_on_the_way_even_measured(a, b):
    arm, robot = lifted_over(*a)
    robot.sent.clear()
    robot.clearances.clear()
    arm.place(*b, 0.045, direct=True)
    arrive = next(i for i, q in enumerate(robot.sent) if math.dist(kinematics.tool_point(q)[:2], b) < 0.005)
    tile_top_mm = (config.TILE_THICKNESS_M - config.GRASP_POINT_ABOVE_TABLE_M) * 1000
    assert min(robot.clearances[:arrive]) - 12 > tile_top_mm + 10     # 12: a held tile's underside below the tip


def test_lift_clear_says_when_a_lift_gave_up_low(monkeypatch):
    arm, _ = lifted_over(0.192, 0.09)
    assert arm.lift_clear()
    monkeypatch.setattr(arm, "tip_clearance", lambda: (0.03, 0.015))
    assert not arm.lift_clear()
