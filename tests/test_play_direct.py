"""Play's direct moves: no trip to rest between a pickup and its discard, a discard and the next pickup, or a look and a
pickup; and the one scan at the end of the turn still answers for everything the arm handled."""
import builtins
import math
import sys
from pathlib import Path

import pytest

from mahjong import arm as arm_module, config, interventions, kinematics
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
    """Count the trips the arm makes to rest (a rest() called away from it)."""
    trips, rest = [], p.bench.arm.rest

    def counted():
        if not p.bench.arm.is_at_rest():
            trips.append(p.state["phase"])
        rest()
    p.bench.arm.rest = counted
    return trips


def stopwatch(p):
    """Add up the seconds the arm's moves and settles take."""
    seconds = []
    p.bench.arm.sleep = seconds.append
    return seconds


def scan_labels(p):
    return [e["label"] for e in events_of(p) if e["type"] == "detections"]


def test_a_turn_goes_straight_from_tile_to_tile_and_scans_at_its_end(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    trips = trips_to_rest(p)
    assert run(p) == "clean"                                # nothing the scans saw looked like a person or a bump
    assert trips == ["discard", "place"]                    # the zone discard's end, the moved tile's end (Arm.place rests)
    labels = scan_labels(p)
    assert labels == ["register"] * 3 + ["drawn", "drawn", "choose", "after_pickup"] \
        + ["drawn", "drawn", "choose", "after_return"] + ["drawn", "drawn"]
    end = next(e for e in events_of(p) if e["type"] == "detections" and e["label"] == "after_return")
    assert len(end["handled_xy"]) == 2 and len(end["handled_attempts"]) == 2


def test_via_rest_keeps_a_scan_and_a_rest_after_every_move(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers(), argv=("--sim", "--via-rest"))
    trips = trips_to_rest(p)
    assert run(p) == "clean"
    assert len(trips) == 6
    assert scan_labels(p).count("after_pickup") == 3 and scan_labels(p).count("after_discard") == 2
    assert next(e for e in events_of(p) if e["type"] == "play_setup")["direct"] is False


def test_direct_moves_take_much_less_arm_time(tmp_path, monkeypatch):
    def arm_time(folder, argv):
        p = player(tmp_path / folder, monkeypatch, Answers(), argv=argv)
        p.bench.arm.rest()
        p.register()
        p.confirm_hand()
        seconds = stopwatch(p)
        while p.state["turn"] < 2:
            p.turn()
        p.close()
        return sum(seconds)
    direct, via_rest = arm_time("direct", ("--sim",)), arm_time("rest", ("--sim", "--via-rest"))
    print(f"arm seconds over two turns: direct {direct:.0f}, via rest {via_rest:.0f}")
    assert direct < 0.75 * via_rest


def test_a_knock_seen_at_the_end_of_the_turn_is_a_knock_not_a_person(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers())
    p.bench.defer = True
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


def test_a_next_tile_near_the_last_one_is_scanned_for_first(tmp_path, monkeypatch):
    """It may have been knocked, and the hover aims from the scan."""
    monkeypatch.setattr(interventions, "NEIGHBOUR_M", 0.2)        # H12, discarded before it, counts as near the zone
    p = player(tmp_path, monkeypatch, Answers())
    run(p)
    labels = scan_labels(p)
    assert "after_discard" not in labels and labels.count("after_pickup") == 2   # the scan carries the pickup it answers for
    assert p.state["hand"]["H12"] == "9B"


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
    i = next(k for k, e in enumerate(events_of(p)) if e["type"] == "attempt")
    assert next(e for e in events_of(p)[i - 1:] if e["type"] == "detections")["label"] == "after_pickup"


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


def test_a_corrected_discard_hovers_over_the_new_one_without_going_to_rest(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers("", "GD"))          # the tile at H12 is a GD
    p.reader = Reader(p, {"RD": "GD"})
    run(p)
    events = events_of(p)
    off = next(i for i, e in enumerate(events) if e["type"] == "pickup_called_off")
    until = next(i for i, e in enumerate(events) if i > off and e["type"] == "attempt")
    assert not [e for e in events[off:until] if e["type"] == "detections"]        # the scan it aims from is still good


def test_a_look_at_the_drawn_tile_is_where_its_pickup_starts(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch, Answers("", "7C"))          # the hand's fine; the drawn tile is the 7C it reads
    p.reader = Reader(p)
    p.draws = ["7C"]
    p.bench.arm.rest()
    p.register()
    p.confirm_hand()
    hover, hovers = p.bench.arm.hover, []
    monkeypatch.setattr(p.bench.arm, "hover", lambda *a, **k: hovers.append(a[:2]) or hover(*a, **k))
    trips = trips_to_rest(p)
    p.turn()
    p.close()
    assert p.state["discards"] == ["7C"] and len(hovers) == 1
    assert trips == ["discard"]                                   # only after the discard


def test_handled_points_are_this_scans_target_and_the_pickups_with_no_scan():
    scan = {"label": "after_return", "target_xy": [1, 2], "handled_xy": [[3, 4]]}
    assert interventions.handled_points(scan) == [[1, 2], [3, 4]]
    assert interventions.handled_points({"label": "choose", "handled_xy": [[3, 4]]}) == []


def test_replaying_a_direct_turn_leaves_its_tiles_out_of_the_changes():
    tile = lambda name, x, y: {"tile_name": name, "xy": [x, y], "confidence": 0.9}
    before = {"label": "choose", "scan": 1, "tiles": [tile("RD", 0.2, 0.1), tile("9B", 0.25, 0.0), tile("1D", 0.15, -0.1)]}
    after = {"label": "after_return", "scan": 2, "target_xy": [0.2, 0.1], "handled_xy": [[0.2, 0.1], [0.25, 0.0]],
             "tiles": [tile("9B", 0.2, 0.1), tile("1D", 0.15, -0.1)]}
    watch = interventions.MatWatch()
    watch.advance(before)
    assert watch.propose(after) is None


# --- the moves themselves ---

def lifted_over(x, y):
    robot = SimRobot(start=kinematics.hover_command(x, y, 0.08, 11.0)[0], holding=True)
    return Arm(robot, sleep=lambda s: None), robot


def test_a_direct_hover_arrives_with_the_pan_turning_up_as_it_does_from_rest():
    arm, robot = lifted_over(0.142, -0.114)                       # pan well above where it's going
    target = kinematics.hover_command(0.192, 0.09, 0.065, 26.7)[0]
    assert target["shoulder_pan"] < arm.commanded()["shoulder_pan"]
    arm.hover(0.192, 0.09, 0.045, 0.04, 26.7, direct=True)
    pans = [c["shoulder_pan"] for c in robot.sent]
    assert min(pans) == pytest.approx(target["shoulder_pan"] - arm_module.TRAVEL_PAN_BACK_DEG, abs=0.5)
    assert pans[-1] > min(pans)


def test_a_direct_hover_is_timed_by_its_distance_and_a_hover_from_rest_isn_t():
    def seconds(arm_robot, **kwargs):
        arm, _ = arm_robot
        spent = []
        arm.sleep = spent.append
        arm.hover(0.192, 0.039, 0.045, 0.04, 26.7, **kwargs)
        return sum(spent)
    near = seconds(lifted_over(0.192, 0.09), direct=True)
    assert near < seconds(lifted_over(0.192, 0.09)) - 1.5         # the same hop, the old 3s move
    from_rest = (Arm(SimRobot(), sleep=lambda s: None), None)
    assert seconds(from_rest, direct=True) > arm_module.MOVE_TIME_S + arm_module.RAISE_TIME_S


@pytest.mark.parametrize("a,b", [((0.192, 0.141), (0.142, -0.165)), ((0.142, 0.141), (0.192, -0.165)),
                                 ((0.142, -0.114), (0.192, 0.09))])
def test_a_direct_hop_stays_over_the_tiles_even_measured(a, b):
    arm, robot = lifted_over(*a)
    arm.hover(*b, 0.045, 0.04, 26.7, direct=True)
    held_low = [c for c, q in zip(robot.clearances, robot.sent)
                if math.dist(kinematics.pincer_tip(q)[:2], b) > 0.03]     # still on its way, not over the target
    assert min(held_low) > (config.TILE_THICKNESS_M - config.GRASP_POINT_ABOVE_TABLE_M) * 1000 + 25


def test_lift_clear_says_when_a_lift_gave_up_low(monkeypatch):
    arm, _ = lifted_over(0.192, 0.09)
    assert arm.lift_clear()
    monkeypatch.setattr(arm, "tip_clearance", lambda: (0.03, 0.015))
    assert not arm.lift_clear()
