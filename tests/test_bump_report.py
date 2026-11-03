"""bench.py --unattended end to end in sim, the report it leaves, and verdicts reaching the scorecard's numbers."""
import builtins
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import bump_report, config, interventions, scorecard
from mahjong.session import read_events

NEIGHBOUR = {"tile_name": "5C", "xy": [0.160, -0.220]}   # 55mm from the sim's 8D (as test_bench)


@pytest.fixture
def no_input(monkeypatch):
    """An unattended run must never wait on a person: any question fails the test."""
    def refuse(prompt=""):
        raise AssertionError(f"asked a question: {prompt}")
    monkeypatch.setattr(builtins, "input", refuse)


def run_bench(tmp_path, monkeypatch, *more, setup=None):
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    made = []
    real = bench.Bench

    class Keep(real):
        def __init__(self, args):
            super().__init__(args)
            made.append(self)
            if setup:
                setup(self)
    monkeypatch.setattr(bench, "Bench", Keep)
    bench.main(["--sim", "--unattended", *more])
    return made[0]


def knock_once(bench_, name="5C", dx=0.008, on_run=1):
    """A neighbour nudged as the `on_run`th pickup starts."""
    bench_.realsense.tiles.append(dict(NEIGHBOUR))
    run, count = bench_.policy.run, [0]

    def wrapped(*args, **kwargs):
        count[0] += 1
        if count[0] == on_run:
            bench_.realsense.nudge(name, dx, 0.0)
        return run(*args, **kwargs)
    bench_.policy.run = wrapped


def test_unattended_needs_a_batch_that_asks_nothing():
    import bench
    for argv in (["--unattended"], ["--unattended", "--trials", "2"], ["--unattended", "--tile", "8D"]):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)
    assert bench.parse_args(["--unattended", "--trials", "2", "--layout", "x"]).unattended


def test_a_clean_unattended_run_asks_nothing_and_writes_the_report(tmp_path, monkeypatch, no_input, capsys):
    b = run_bench(tmp_path, monkeypatch, "--trials", "2", "--tile", "8D")
    events = read_events(b.session.dir)
    assert [e["label"] for e in events if e["type"] == "detections"] == ["choose", "after_pickup", "after_return"] * 2 + ["final"]
    assert interventions.derive(events) == []
    assert (b.session.dir / bump_report.PAGE).exists()
    assert "0 self-reported bump(s)" in capsys.readouterr().out
    assert next(e for e in events if e["type"] == "bench_setup")["unattended"] is True


def test_a_neighbour_that_moved_is_self_reported_with_no_question(tmp_path, monkeypatch, no_input):
    b = run_bench(tmp_path, monkeypatch, "--trials", "2", "--tile", "8D", setup=knock_once)
    events = read_events(b.session.dir)
    (knock,) = interventions.derive(events)
    assert (knock["kind"], knock["during"], knock["source"]) == ("knocked", "pickup", "scans")
    assert [m["tile_name"] for m in knock["changes"]["moved"]] == ["5C"]
    assert knock["attempt_dir"] == next(e for e in events if e["type"] == "attempt")["attempt_dir"]
    assert not scorecard.clean_flags(events)[knock["attempt_dir"]]


def test_a_bump_in_the_last_return_is_still_reported(tmp_path, monkeypatch, no_input):
    """Only the closing scan comes after it."""
    def setup(b):
        b.realsense.tiles.append(dict(NEIGHBOUR))
        place = b.arm.place

        def wrapped(*args, **kwargs):
            b.realsense.nudge("5C", 0.008, 0.0)
            return place(*args, **kwargs)
        b.arm.place = wrapped
    b = run_bench(tmp_path, monkeypatch, "--trials", "1", "--tile", "8D", setup=setup)
    (knock,) = interventions.derive(read_events(b.session.dir))
    assert (knock["kind"], knock["during"]) == ("knocked", "return")


def test_a_flicker_that_goes_back_is_a_glitch_not_a_bump(tmp_path, monkeypatch, no_input):
    def setup(b):
        b.realsense.tiles.append(dict(NEIGHBOUR))
        run, count = b.policy.run, [0]

        def wrapped(*args, **kwargs):
            count[0] += 1
            b.realsense.nudge("5C", 0.008 if count[0] == 1 else -0.008, 0.0)
            return run(*args, **kwargs)
        b.policy.run = wrapped
    b = run_bench(tmp_path, monkeypatch, "--trials", "2", "--tile", "8D", setup=setup)
    events = read_events(b.session.dir)
    kinds = [i["kind"] for i in interventions.derive(events)]
    assert "knocked" not in kinds and "scan_glitch" in kinds
    found = bump_report.phases(events)
    assert bump_report.counts(events, found)["flagged"] == 0        # shown, but not a self-report


def test_a_knocked_tile_the_arm_picks_up_next_stays_a_knock(tmp_path, monkeypatch, no_input):
    """Its return would put it home, which the later scans would otherwise read as the knock going back."""
    from mahjong import layout
    spots = [{"id": "A", "xy": [0.212, 0.021]}, {"id": "B", "xy": [0.212, 0.075]}]    # the sim's lift reaches these
    monkeypatch.setattr(layout, "load", lambda name: {"spots": spots})

    def setup(b):
        b.realsense.tiles = [{"tile_name": "8D", "xy": list(spots[0]["xy"])}, {"tile_name": "5C", "xy": list(spots[1]["xy"])}]
        run, count = b.policy.run, [0]

        def wrapped(*args, **kwargs):
            count[0] += 1
            if count[0] == 1:
                b.realsense.nudge("5C", 0.008, 0.0)
            return run(*args, **kwargs)
        b.policy.run = wrapped
    b = run_bench(tmp_path, monkeypatch, "--trials", "1", "--layout", "x", setup=setup)
    events = read_events(b.session.dir)
    (knock,) = interventions.derive(events)
    assert knock["kind"] == "knocked" and [m["tile_name"] for m in knock["changes"]["moved"]] == ["5C"]


def test_a_tile_not_found_is_skipped_without_asking(tmp_path, monkeypatch, no_input):
    def setup(b):
        b.realsense.tiles[:] = [t for t in b.realsense.tiles if t["tile_name"] != "8D"]
    b = run_bench(tmp_path, monkeypatch, "--trials", "1", "--tile", "8D", setup=setup)
    assert [e["reason"] for e in read_events(b.session.dir) if e["type"] == "attempt_skipped"] == ["not_detected"]


def test_a_return_landing_far_off_is_left_there_and_noted(tmp_path, monkeypatch, no_input):
    def setup(b):
        mat = b.realsense
        b.arm.robot.on_release = lambda xy, clearance: mat.release([xy[0] + 0.03, xy[1]], clearance)     # lands 30mm off
    b = run_bench(tmp_path, monkeypatch, "--trials", "1", "--tile", "8D", setup=setup)
    events = read_events(b.session.dir)
    assert [e["tile"] for e in events if e["type"] == "reset_skipped"] == ["8D"]
    assert not [i for i in interventions.derive(events) if i["class"] == interventions.RESET]
    found = bump_report.phases(events)
    assert bump_report.counts(events, found)["left_off_home"] == 1
    assert "left there" in next(p for p in found if p["during"] == "return")["result"]


def test_a_pale_surface_look_with_no_one_to_ask_stays_unknown(tmp_path, monkeypatch, no_input):
    from mahjong import wrist_check

    def setup(b):       # 5C lost in the after-pickup and after-return scans: looked at once the tile is down
        b.realsense.tiles.append(dict(NEIGHBOUR))
        run = b.policy.run

        def wrapped(*args, **kwargs):
            b.realsense.flicker("5C", scans=2)
            return run(*args, **kwargs)
        b.policy.run = wrapped
    monkeypatch.setattr(wrist_check, "check", lambda *a, **k: {"present": None, "pale": True, "fill": None, "frame": None,
                                                               "why": "pale surface"})
    b = run_bench(tmp_path, monkeypatch, "--trials", "1", "--tile", "8D", setup=setup)
    checks = [e for e in read_events(b.session.dir) if e["type"] == "wrist_check"]
    assert checks and all(c["present"] is None for c in checks)


def test_a_camera_that_wont_reopen_ends_an_unattended_run_rather_than_waiting(monkeypatch):
    import bench
    from mahjong import hardware
    asked = []

    def reopen(camera, ask=None, openers=None):
        asked.append(ask("retry? "))
        raise SystemExit("Quit before starting")
    monkeypatch.setattr(hardware, "reopen_wrist_camera", reopen)

    class Args:
        unattended = True

    class Fake:
        args, camera = Args, None
    with pytest.raises(SystemExit, match="wouldn't reopen"):
        bench.Bench.reopen_camera(Fake)
    assert asked == ["q"]


# --- the report ---

def knocked_session(tmp_path, monkeypatch):
    return run_bench(tmp_path, monkeypatch, "--trials", "2", "--tile", "8D", setup=knock_once).session.dir


def test_the_report_has_every_pickup_and_return_and_the_bump_marked(tmp_path, monkeypatch, no_input):
    session = knocked_session(tmp_path, monkeypatch)
    events, found = bump_report.load(session)
    assert [p["during"] for p in found] == ["pickup", "return"] * 2
    pickup = found[0]
    (row,) = [i for i in pickup["items"] if i["flagged"]]
    assert row["what"] == "5C moved 8mm" and row["said"] == "knocked"
    assert all(not i["said"] for i in found[1]["items"])            # the return: 5C listed, nothing said of it
    assert [i["what"].split()[0] for i in found[1]["items"]] == ["5C"]
    for phase in found:
        for path in phase["images"].values():
            assert (session / path).stat().st_size > 0
    page = (session / bump_report.PAGE).read_text()
    assert "5C moved 8mm" in page and "1 bump" in page


def test_the_report_picture_is_cropped_around_the_tiles_and_lines_up(tmp_path, monkeypatch, no_input):
    import cv2
    session = knocked_session(tmp_path, monkeypatch)
    _, found = bump_report.load(session)
    images = found[0]["images"]
    before, after = cv2.imread(str(session / images["before_raw"])), cv2.imread(str(session / images["after_raw"]))
    assert before.shape == after.shape and before.shape[1] <= bump_report.SHOWN_W + 1
    frame = cv2.imread(str(session / found[0]["before"]["frames"] / bump_report.FRAME))
    assert before.shape[:2] != frame.shape[:2]


def test_saying_a_bump_didnt_happen_takes_it_off_the_scorecard(tmp_path, monkeypatch, no_input):
    session = knocked_session(tmp_path, monkeypatch)
    events, found = bump_report.load(session)
    attempt = next(e for e in events if e["type"] == "attempt")["attempt_dir"]
    assert not scorecard.clean_flags(events)[attempt]
    (row,) = [i for p in found for i in p["items"] if i["flagged"]]
    bump_report.record_verdict(session, found, row["id"], "not_moved")
    events = read_events(session)
    assert scorecard.clean_flags(events)[attempt]
    assert [i["kind"] for i in interventions.derive(events)] == ["scan_glitch"]
    assert bump_report.counts(events, bump_report.phases(events))["rejected"] == 1
    bump_report.record_verdict(session, found, row["id"], "moved")     # changed their mind: the latest stands
    events = read_events(session)
    assert not scorecard.clean_flags(events)[attempt]
    assert bump_report.counts(events, bump_report.phases(events))["confirmed"] == 1


def test_saying_a_tile_moved_that_the_scans_missed_is_a_knock(tmp_path, monkeypatch, no_input):
    b = run_bench(tmp_path, monkeypatch, "--trials", "1", "--tile", "8D",
                  setup=lambda b: b.realsense.tiles.append(dict(NEIGHBOUR)))
    session = b.session.dir
    events, found = bump_report.load(session)
    assert interventions.derive(events) == []
    row = found[0]["items"][0]
    assert row["what"].startswith("5C") and not row["said"]
    bump_report.record_verdict(session, found, row["id"], "moved")
    events = read_events(session)
    (knock,) = interventions.derive(events)
    assert (knock["kind"], knock["during"], knock["source"]) == ("knocked", "pickup", "scans")
    assert bump_report.counts(events, found)["missed"] == 1
    assert next(iter(bump_report.phases(events)))["items"][0]["verdict"] == "moved"     # still there on a reload


def test_a_verdict_on_a_tile_the_scans_saw_no_change_in_says_nothing_if_not_moved(tmp_path, monkeypatch, no_input):
    b = run_bench(tmp_path, monkeypatch, "--trials", "1", "--tile", "8D",
                  setup=lambda b: b.realsense.tiles.append(dict(NEIGHBOUR)))
    events, found = bump_report.load(b.session.dir)
    bump_report.record_verdict(b.session.dir, found, found[0]["items"][0]["id"], "not_moved")
    assert not [i for i in interventions.derive(read_events(b.session.dir)) if i["kind"] == "knocked"]


def test_sessions_with_no_frames_or_no_pixels_still_report(tmp_path):
    """Older sessions: the page still lists what the scans said; a scan with no frames shows no pictures."""
    session = tmp_path / "s"
    session.mkdir()
    tile = {"tile_name": "5C", "xy": [0.16, -0.22], "confidence": 0.9}
    events = [{"type": "detections", "label": "choose", "scan": 1, "tiles": [tile, {**tile, "tile_name": "8D", "xy": [0.16, -0.17]}]},
              {"type": "detections", "label": "after_pickup", "scan": 2, "attempt_dir": "a", "target_xy": [0.16, -0.17], "tiles": [tile]}]
    (session / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    assert bump_report.write(session)["flagged"] == 0
    assert "No frames were kept" in (session / bump_report.PAGE).read_text()


def test_the_page_saves_verdicts_and_serves_only_the_session(tmp_path, monkeypatch, no_input):
    import bump_report as script
    session = knocked_session(tmp_path, monkeypatch)
    server = script.make_server(session)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/"

    def post(body, kind="application/json"):
        return urllib.request.urlopen(urllib.request.Request(base + "verdict", json.dumps(body).encode(),
                                                             {"Content-Type": kind}))
    try:
        assert b"Bump report" in urllib.request.urlopen(base).read()
        _, found = bump_report.load(session)
        (row,) = [i for p in found for i in p["items"] if i["flagged"]]
        image = next(iter(found[0]["images"].values()))
        assert urllib.request.urlopen(base + image).status == 200
        assert json.loads(post({"id": row["id"], "verdict": "not_moved"}).read())["rejected"] == 1
        assert [e["verdict"] for e in read_events(session) if e["type"] == "scan_verdict"] == ["not_moved"]
        for bad in ({"id": "nope", "verdict": "moved"}, {"id": row["id"], "verdict": "maybe"}):
            with pytest.raises(urllib.error.HTTPError):
                post(bad)
        with pytest.raises(urllib.error.HTTPError):     # a form on someone else's page can't post here
            post({"id": row["id"], "verdict": "moved"}, "text/plain")
        for path in ("../../etc/passwd", "events.jsonl", "%2e%2e/%2e%2e/secret.jpg"):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(base + path)
    finally:
        server.shutdown()
        server.server_close()


def test_attended_runs_still_ask(tmp_path, monkeypatch):
    """--unattended off changes nothing: the neighbour is asked about, as test_bench's says."""
    import bench
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "1", "--tile", "8D"]))
    knock_once(b)
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt: asked.append(prompt) or "")
    b.arm.rest()
    try:
        b.trials(1, "8D")
    finally:
        b.close()
    assert len(asked) == 1 and "5C moved 8mm" in asked[0]
    assert [e["label"] for e in read_events(Path(b.session.dir)) if e["type"] == "detections"][-1] == "after_return"
