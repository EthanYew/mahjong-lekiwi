"""A person takes the arm from the policy in bench.py and the face-down scripts, and the recovery is a DAgger episode."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from test_record import FakeDataset

from mahjong import config, scorecard, takeover
from mahjong.dataset import EPISODE_LOG, takeover_frame
from mahjong.session import Session, read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class PressOnce:
    """A gamepad whose A reads pressed the first time it's checked in each pickup (the sim checks once, after the descent)."""
    name = "fake pad"

    def __init__(self, press=True):
        self.press, self.starts, self.closed = press, 0, False

    def arm_takeover(self):
        self.pending = self.press

    def takeover_pressed(self):
        pressed, self.pending = self.pending, False
        return pressed

    def start(self, arm):
        self.starts += 1

    def step(self, arm, dt):
        return arm.commanded()

    def close(self):
        self.closed = True


def fake_takeovers(monkeypatch, tmp_path, press=True):
    """takeover.make gives a Takeover on a fake gamepad and dataset whenever --dagger-dataset is given, sim or not."""
    made = []

    def make(repo_id, gamepad, log=print):
        if not repo_id:
            return None
        t = takeover.Takeover(PressOnce(press), repo_id, dataset=FakeDataset(tmp_path / "dataset"), log=log)
        made.append(t)
        return t
    monkeypatch.setattr(takeover, "make", make)
    (tmp_path / "dataset").mkdir()
    return made


def log_rows(tmp_path):
    path = tmp_path / "dataset" / EPISODE_LOG
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# --- saving ---

def session_in(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    return Session("bench", ask_setup=False, sim=True)


def kept(t, n=5):
    for k in range(n):
        t.add(np.zeros((480, 640, 3), np.uint8), np.full(6, float(k)), {j: float(k) + 0.5 for j in config.JOINTS})


def test_a_saved_recovery_is_a_dagger_episode_in_records_format(monkeypatch, tmp_path):
    session = session_in(monkeypatch, tmp_path)
    dataset = FakeDataset(tmp_path)
    t = takeover.Takeover(PressOnce(), "Yewy/x", dataset=dataset, log=lambda *a: None)
    kept(t)
    run = {"takeover_step": 2, "takeover_end": "pressed"}
    assert t.save(run, session, "pickup_v8", tile_name="5C", hover={"offset_m": 0.04}, attempt_dir="a1") == 0
    session.close()
    assert dataset.saved == [5]                     # every frame from the hover on, as record.py --policy keeps them
    (row,) = [json.loads(line) for line in (tmp_path / EPISODE_LOG).read_text().splitlines()]
    assert takeover_frame(row) == (True, 2)         # what dataset.build trains from
    assert row["policy"] == "pickup_v8" and row["attempt_dir"] == "a1" and row["slice"] == "train"
    events = read_events(session.dir)
    (rescue,) = [e for e in events if e["type"] == "intervention"]
    assert rescue["kind"] == "takeover" and rescue["attempt_dir"] == "a1"
    assert any(e["type"] == "takeover_saved" for e in events)


def test_the_frames_are_deltas_from_the_hover(monkeypatch, tmp_path):
    session = session_in(monkeypatch, tmp_path)
    dataset = FakeDataset(tmp_path)
    t = takeover.Takeover(PressOnce(), "Yewy/x", dataset=dataset, log=lambda *a: None)
    kept(t, 3)
    frames = []
    dataset.add_frame = frames.append
    t.save({"takeover_step": 1, "takeover_end": "done"}, session, "p")
    session.close()
    assert np.allclose(frames[0]["observation.state"], 0) and np.allclose(frames[2]["observation.state"], 2)
    assert np.allclose(frames[2]["action"], 2.5) and np.allclose(frames[2]["action_raw"], 2.5)


@pytest.mark.parametrize("end", ["discarded", "camera"])
def test_esc_or_a_camera_gap_saves_nothing_but_the_rescue_still_counts(monkeypatch, tmp_path, end):
    session = session_in(monkeypatch, tmp_path)
    dataset = FakeDataset(tmp_path)
    t = takeover.Takeover(PressOnce(), "Yewy/x", dataset=dataset, log=lambda *a: None)
    kept(t)
    assert t.save({"takeover_step": 2, "takeover_end": end}, session, "p", attempt_dir="a1") is None
    session.close()
    assert dataset.saved == [] and not (tmp_path / EPISODE_LOG).exists()
    kinds = [e.get("kind", e["type"]) for e in read_events(session.dir) if e["type"] in ("intervention", "takeover_discarded")]
    assert kinds == ["takeover", "takeover_discarded"]


def test_a_pickup_nobody_took_over_is_not_saved(monkeypatch, tmp_path):
    session = session_in(monkeypatch, tmp_path)
    dataset = FakeDataset(tmp_path)
    t = takeover.Takeover(PressOnce(), "Yewy/x", dataset=dataset, log=lambda *a: None)
    kept(t)
    assert t.save({"takeover_step": None}, session, "p") is None
    session.close()
    assert dataset.saved == [] and not t.frames
    assert not [e for e in read_events(session.dir) if e["type"] == "intervention"]


def test_one_dataset_holds_one_hover(tmp_path):
    hover = {"offset_m": 0.041, "clearance_m": 0.03, "gripper": 26.8}
    takeover.same_hover([hover, dict(hover)])
    with pytest.raises(SystemExit, match="different hovers"):
        takeover.same_hover([hover, {**hover, "gripper": 21.0}])
    (tmp_path / EPISODE_LOG).write_text(json.dumps({"episode_index": 0, "hover": hover}) + "\n")
    t = takeover.Takeover(PressOnce(), "Yewy/x", dataset=FakeDataset(tmp_path), log=lambda *a: None)
    t.expect_hover(hover)
    with pytest.raises(SystemExit):
        t.expect_hover({**hover, "gripper": 90.0})


def test_an_attempt_logs_the_policys_close_and_the_return_keeps_the_persons():
    run = {"takeover_step": 30, "grasp_xy": [0.2, 0.1], "grasp_clearance_m": 0.004,
           "policy_grasp_xy": [0.21, 0.1], "policy_grasp_clearance_m": 0.024}
    view = takeover.policy_view(run)
    assert view["grasp_clearance_m"] == 0.024 and view["person_grasp_clearance_m"] == 0.004
    assert "policy_grasp_xy" not in view
    assert takeover.policy_view({"takeover_step": None, "grasp_xy": [1, 2]})["grasp_xy"] == [1, 2]


# --- end to end on the sim ---

def test_bench_a_takeover_is_saved_counted_as_a_rescue_and_the_tile_still_returned(monkeypatch, tmp_path):
    import bench
    made = fake_takeovers(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "1", "--tile", "8D", "--dagger-dataset", "Yewy/bench_dagger"]))
    try:
        b.arm.rest()
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    (attempt,) = [e for e in events if e["type"] == "attempt"]
    assert attempt["takeover_step"] == 2 and attempt["success"]
    assert scorecard.clean_flags(events) == {attempt["attempt_dir"]: False}      # picked, but a person did it
    assert len([e for e in events if e["type"] == "return"]) == 1               # the recovered tile goes home as usual
    (row,) = log_rows(tmp_path)
    assert row["policy"] == "sim" and row["takeover_frame"] == 2 and row["attempt_dir"] == attempt["attempt_dir"]
    assert row["hover"] == b.hover_of(b.policy) and row["tile_name"] == "8D"
    assert made[0].gamepad.starts == 1 and made[0].gamepad.closed
    assert not (b.session.dir / "hover_corrections.jsonl").exists()             # a person's grasp isn't a policy sample


def test_bench_without_a_takeover_saves_nothing_and_stays_clean(monkeypatch, tmp_path):
    import bench
    fake_takeovers(monkeypatch, tmp_path, press=False)
    monkeypatch.setattr(config, "SESSIONS", tmp_path / "sessions")
    b = bench.Bench(bench.parse_args(["--sim", "--trials", "1", "--tile", "8D", "--dagger-dataset", "Yewy/bench_dagger"]))
    try:
        b.arm.rest()
        b.trials(1, "8D")
    finally:
        b.close()
    events = read_events(b.session.dir)
    assert list(scorecard.clean_flags(events).values()) == [True]
    assert log_rows(tmp_path) == []


def test_base_cam_collect_a_takeover_in_a_hold_is_saved_and_the_hold_still_shown(monkeypatch, tmp_path):
    from test_base_cam_collect import run_script, script_in
    fake_takeovers(monkeypatch, tmp_path)
    script = script_in(monkeypatch, tmp_path)
    folder = run_script(script, ["--tiles", "5C", "--target-holds", "1", "--rounds", "1", "--frames", "1",
                                 "--dagger-dataset", "Yewy/holdup_dagger"], ["", "", "", "", "d", ""])
    events = read_events(folder)
    assert [e["type"] for e in events].count("holdup_shot") >= 1
    rows = log_rows(tmp_path)
    assert rows and all(r["face_down"] and r["spot"] == "S7" and r["takeover_frame"] == 2 for r in rows)
    assert len([e for e in events if e.get("kind") == "takeover"]) == len(rows)


def test_every_script_that_picks_takes_the_option():
    import base_cam_collect
    import bench
    import crude_run
    import wall_test
    assert bench.parse_args(["--dagger-dataset", "Yewy/d"]).dagger_dataset == "Yewy/d"
    assert base_cam_collect.parse_args(["--dagger-dataset", "Yewy/d"]).dagger_dataset == "Yewy/d"
    assert wall_test.parse_args(["--dagger-dataset", "Yewy/d"]).dagger_dataset == "Yewy/d"
    assert crude_run.parse_args(["--hand", ",".join(["1B"] * 13), "--dagger-dataset", "Yewy/d"]).dagger_dataset == "Yewy/d"
