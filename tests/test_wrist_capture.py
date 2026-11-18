import random
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import config
from mahjong.session import read_events


def test_sim_set_saves_every_view_under_the_confirmed_names(tmp_path, monkeypatch):
    import wrist_capture
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    answers = iter(["3=2c -2", ""])           # the sim mat, left to right: 9D, 3B, 8D
    wrist_capture.main(["--sim", "--sets", "1", "--views", "4", "--frames", "2", "--seed", "1"],
                       ask=lambda prompt: next(answers))
    session = next(tmp_path.glob("*_wrist"))
    wrist = session / "wrist"
    assert sorted(p.name for p in wrist.iterdir()) == ["2C", "9D"]               # 3B left out, 8D renamed
    assert len(list((wrist / "9D").glob("*.jpg"))) == 4 * 2
    views = [e for e in read_events(session) if e["type"] == "wrist_view"]
    assert len(views) == 8 and {v["tile"] for v in views} == {"9D", "2C"}
    renamed = [v for v in views if v["tile"] == "2C"]
    assert all(v["detected_as"] == "8D" for v in renamed)
    first = next(v for v in views if v["tile"] == "9D" and v["view"] == 0)
    assert first["hover_jitter_xy"] == [0.0, 0.0] and first["roll_offset_deg"] == 0.0
    assert any(abs(v["roll_offset_deg"]) > 0 for v in views)                     # the other views roll
    assert {round(v["target_clearance_m"], 3) for v in views} == {0.04, 0.055, 0.07}
    assert [e for e in read_events(session) if e["type"] == "wrist_set_done"][0]["short_of"] == []
    assert not wrist_capture.views_per_name(tmp_path, tmp_path)["wrist_capture"]     # a sim session never counts


def test_empty_hovers_over_the_layouts_bare_spots_only_and_keeps_them_from_the_name_check(tmp_path, monkeypatch):
    import wrist_capture
    from mahjong import layout, sim, wrist_photos
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    spots = layout.load("even-hand14")["spots"]
    near = {s["id"] for s in spots if any(((s["xy"][0] - x) ** 2 + (s["xy"][1] - y) ** 2) ** 0.5 <= 0.035
                                          for _, x, y in sim.SIM_TILES)}
    bare = [s["id"] for s in spots if s["id"] not in near]
    answers = iter([f"-{bare[1]}", ""])                       # one bare spot left out by hand
    wrist_capture.main(["--sim", "--sets", "1", "--views", "2", "--frames", "2", "--seed", "1", "--empty", "even-hand14"],
                       ask=lambda prompt: next(answers))
    session = next(tmp_path.glob("*_wrist"))
    assert not (session / "wrist").exists()                   # nothing the name check reads
    views = [e for e in read_events(session) if e["type"] == "wrist_view"]
    looked = {v["spot"] for v in views}
    assert looked and looked <= set(bare) - {bare[1]} and not looked & near
    assert {v["tile"] for v in views} == {"empty"} and all(f.startswith("wrist_empty/empty/") for v in views
                                                           for f in v["files"])
    assert len(list((session / "wrist_empty" / "empty").glob("*.jpg"))) == 2 * len(views)
    assert not wrist_photos.photos(tmp_path)


def test_on_the_arm_empty_refuses_weights_that_cant_see_backs():
    wc = wrist_capture_module()

    class Blind:
        names = ["1B", "2B"]

        def detect_tiles(self):
            raise AssertionError("refused before scanning")
    with pytest.raises(SystemExit, match="see no backs"):
        wc.confirm_empty(Blind(), [], ask=lambda p: "", show=False, sim=False)


def test_a_bare_spot_is_one_with_no_tile_near_it():
    wc = wrist_capture_module()
    spots = [{"id": "S1", "xy": [0.19, 0.14]}, {"id": "S2", "xy": [0.14, 0.14]}, {"id": "S3", "xy": [0.09, 0.14]}]
    tiles = [{"xy": (0.19, 0.14)}, {"xy": (0.14 - 0.03, 0.14)}]           # on S1; knocked 30mm off S2, toward S3
    assert [s["id"] for s in wc.empty_spots(tiles, spots)] == []          # S2 and S3 both have it within 35mm
    assert [s["id"] for s in wc.empty_spots(tiles[:1], spots)] == ["S2", "S3"]


def test_plan_views_stay_inside_the_jitter_and_roll():
    views = wrist_capture_module().plan_views(20, 0.008, [0.04, 0.055], 15.0, random.Random(0))
    assert views[0] == (0.0, 0.0, 0.04, 0.0)
    assert all((dx * dx + dy * dy) ** 0.5 <= 0.008 + 1e-9 and abs(roll) <= 15 for dx, dy, _, roll in views)


def test_name_edits_are_checked():
    wc = wrist_capture_module()
    tiles = [{"tile_name": "8D", "detected_as": "8D"}, {"tile_name": "3B", "detected_as": "3B"}]
    assert wc.parse_answer("", tiles)[0] == "accept"
    action, edited = wc.parse_answer("1=9d, -2", tiles)
    assert action == "edit" and edited[0]["tile_name"] == "9D" and edited[0]["renamed_from"] == "8D" and edited[1]["skip"]
    assert wc.parse_answer("1=XX", tiles)[0] == "error" and wc.parse_answer("7=2C", tiles)[0] == "error"


def wrist_capture_module():
    import wrist_capture
    return wrist_capture


def test_windows_keep_answering_through_sleeps_and_questions():
    import time as _time
    wc = wrist_capture_module()
    pumped = []
    wc.pumping_sleep(0.02, pump=lambda: pumped.append(1))
    assert pumped == [1]
    asking = wc.ask_pumping(lambda prompt: (_time.sleep(0.05), prompt.upper())[1], pump=lambda: pumped.append(1) or _time.sleep(0.005))
    assert asking("ok?") == "OK?" and len(pumped) > 2              # pumped while the question waited


def test_a_failed_question_is_never_read_as_enter():
    wc = wrist_capture_module()
    def closed(prompt):
        raise EOFError
    with pytest.raises(EOFError):
        wc.ask_pumping(closed, pump=lambda: None)("Names right? ")


def test_a_lost_wrist_camera_is_reopened_and_the_view_taken_again(tmp_path, monkeypatch):
    import wrist_capture
    from mahjong import camera as camera_module, hardware, sim
    from mahjong.sim import SimCamera

    class Lost(SimCamera):
        simulated = False

        def read(self):
            return False, None

    reopened = []
    def reopen(camera, ask=None, openers=None):
        reopened.append(camera)
        return SimCamera()
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(sim, "SimCamera", Lost)                 # the script's camera, until reopened
    monkeypatch.setattr(hardware, "reopen_wrist_camera", reopen)
    monkeypatch.setattr(camera_module, "read_within", lambda camera: camera.read())   # no 1 s waits
    answers = iter(["-2 -3", ""])                                # just 9D
    wrist_capture.main(["--sim", "--sets", "1", "--views", "2", "--seed", "0"], ask=lambda p: next(answers))
    session = next(tmp_path.glob("*_wrist"))
    assert len(reopened) == 1                                    # once: the reopened camera then works
    assert len([e for e in read_events(session) if e["type"] == "wrist_view"]) == 2
    assert [e["type"] for e in read_events(session)].count("wrist_camera_reopened") == 1


def test_views_count_bench_attempts_and_recorded_episodes_too(tmp_path):
    import wrist_capture
    sessions, recordings = tmp_path / "sessions", tmp_path / "hf"
    (sessions / "20261028_102732_bench" / "013_103524_6C").mkdir(parents=True)
    (sessions / "20261028_102732_bench" / "013_103524_6C" / "wrist_0000.jpg").write_bytes(b"jpg")   # 0 bytes: a power cut's
    (sessions / "20261028_102732_bench" / "014_103600_6C").mkdir()          # no hover frame: not counted
    (recordings / "Yewy" / "mat").mkdir(parents=True)
    (recordings / "Yewy" / "mat" / "episode_log.jsonl").write_text('{"tile_name": "8D"}\n{"tile_name": "8D"}\n')
    sources = wrist_capture.views_per_name(sessions, recordings)
    assert sources["bench"]["6C"] == 1 and sources["recordings"]["8D"] == 2 and not sources["wrist_capture"]
