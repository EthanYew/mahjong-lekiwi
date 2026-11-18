"""The wrist camera's back check: which photos it trains and is examined on, how a look's reads decide, and the look itself
asking it before the back's shape."""
import json

import cv2
import numpy as np
import pytest

from mahjong import back_check, wrist_check

HOVER = {"offset_m": 0.045, "clearance_m": 0.04, "gripper": 21}


def write_session(folder, events=(), sim=False):
    folder.mkdir(parents=True)
    rows = [{"type": "session_start", "sim": sim}, *events]
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in rows))


def jpg(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((48, 64, 3), np.uint8))


def sessions(tmp_path):
    """A base camera collection run, a face-down bench and a face-up one, a wrist capture with bare spots, a sim run."""
    holdup = tmp_path / "20261113_110111_holdup"
    write_session(holdup)
    jpg(holdup / "001_203628_WW" / "wrist_0000.jpg")
    jpg(holdup / "002_203743_6D" / "wrist_0000.jpg")
    looks = [
        {"type": "wrist_check", "tile": "back", "image": "wrist_checks/a.jpg", "by": "person", "present": True},
        {"type": "wrist_check", "tile": "back", "image": "wrist_checks/b.jpg", "by": "back shape", "present": True},
        {"type": "wrist_check", "tile": "back", "image": "wrist_checks/c.jpg", "by": "person", "present": False},
        {"type": "wrist_check", "tile": "back", "image": "wrist_checks/d.jpg", "by": "back shape", "present": False},
        {"type": "wrist_check", "tile": "back", "image": "wrist_checks/e.jpg", "present": False},    # the face disc's misread
        {"type": "wrist_check", "tile": "back", "image": "wrist_checks/f.jpg", "by": "person", "present": None},
    ]
    down = tmp_path / "20261118_221723_bench"
    write_session(down, [{"type": "bench_start", "face_down": True}, *looks])
    jpg(down / "001_221906_back" / "wrist_0000.jpg")
    jpg(down / "002_222009_8D" / "wrist_0000.jpg")          # not a back attempt
    for name in "abcdef":
        jpg(down / "wrist_checks" / f"{name}.jpg")
    up = tmp_path / "20261117_135608_bench"
    write_session(up, [{"type": "bench_start", "face_down": False},
                       {"type": "wrist_check", "tile": "3C", "image": "wrist_checks/g.jpg", "by": "person", "present": False}])
    jpg(up / "001_140000_3C" / "wrist_0000.jpg")             # a face: not a back
    jpg(up / "wrist_checks" / "g.jpg")
    wrist = tmp_path / "20261119_100000_wrist"
    write_session(wrist)
    jpg(wrist / "wrist_empty" / "empty" / "01_S3_v00_f0.jpg")
    jpg(wrist / "wrist_empty" / "empty" / "01_S3_v00_f1.jpg")
    jpg(wrist / "wrist" / "9D" / "01_01_v00_f0.jpg")         # a named tile: the name check's, not this
    sim = tmp_path / "20261119_110000_holdup"
    write_session(sim, sim=True)
    jpg(sim / "001_110000_WW" / "wrist_0000.jpg")


def test_backs_are_face_down_pickups_and_confirmed_looks_and_empties_are_bare_spots_and_a_persons_no(tmp_path):
    sessions(tmp_path)
    found = {p["key"]: p["name"] for p in back_check.photos(tmp_path)}
    assert found == {
        "20261113_110111_holdup/001_203628_WW/wrist_0000.jpg": "back",
        "20261113_110111_holdup/002_203743_6D/wrist_0000.jpg": "back",
        "20261118_221723_bench/001_221906_back/wrist_0000.jpg": "back",
        "20261118_221723_bench/wrist_checks/a.jpg": "back",      # a person said there
        "20261118_221723_bench/wrist_checks/b.jpg": "back",      # the back's shape found one
        "20261118_221723_bench/wrist_checks/c.jpg": "empty",     # a person said gone
        "20261117_135608_bench/wrist_checks/g.jpg": "empty",     # a face-up spot a person said was empty
        "20261119_100000_wrist/wrist_empty/empty/01_S3_v00_f0.jpg": "empty",
        "20261119_100000_wrist/wrist_empty/empty/01_S3_v00_f1.jpg": "empty",
    }


def test_the_exam_sessions_never_train_it(tmp_path):
    sessions(tmp_path)
    training, exam = back_check.split(back_check.photos(tmp_path), {"sessions": {"20261118_221723_bench": {}}})
    assert {p["session"] for p in exam} == {"20261118_221723_bench"}
    assert "20261118_221723_bench" not in {p["session"] for p in training}
    assert back_check.summary(training).startswith("2 back, 3 empty from 3 sessions")


def test_the_committed_exam_holds_the_bench_that_prompted_it():
    assert "20261118_221723_bench" in back_check.load_exam()["sessions"]


def test_it_wont_train_without_empty_photos():
    with pytest.raises(ValueError, match="--empty"):
        back_check.train([{"name": "back", "path": "x", "session": "s"}])


def test_a_look_decides_only_when_every_frame_is_sure_of_the_same_thing():
    sure_back = [{"name": "back", "confidence": 0.99}] * 3
    assert back_check.decide(sure_back) == "back"
    assert back_check.decide([{"name": "empty", "confidence": 0.95}] * 3) == "empty"
    assert back_check.decide(sure_back[:2] + [{"name": "empty", "confidence": 0.99}]) is None     # frames disagree
    assert back_check.decide(sure_back[:2] + [{"name": "back", "confidence": 0.6}]) is None       # one unsure
    assert back_check.decide([]) is None


def test_one_empty_spot_read_as_a_back_fails_the_gate():
    back, empty, unsure = {"name": "back", "confidence": 0.99}, {"name": "empty", "confidence": 0.99}, \
        {"name": "back", "confidence": 0.5}
    labels = ["back"] * 99 + ["empty"] * 100
    passing = back_check.score([back] * 98 + [unsure] + [empty] * 100, labels)
    assert passing["passes"] and passing["backs_seen"] == pytest.approx(98 / 99) and passing["false_backs"] == 0
    one_false = back_check.score([back] * 99 + [back] + [empty] * 99, labels)
    assert not one_false["passes"] and one_false["false_backs"] == 1
    assert not back_check.score([back] * 99, ["back"] * 99)["passes"]       # no empties examined: can't pass
    few = back_check.score([back] * 99 + [unsure] * 2, ["back"] * 99 + ["empty"] * 2)
    assert few["false_backs"] == 0 and not few["passes"]                    # 0 of 2 proves nothing


class Arm:
    def lift(self):
        pass

    def hover(self, *args, **kwargs):
        return {"hover_xy": [0.2, 0.0]}


class Camera:
    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return True, self.frame

    def flush(self, frames=10):
        pass


class Reader:
    def __init__(self, *reads):
        self.reads = list(reads)

    def read_many(self, frames):
        return self.reads[:len(frames)]


def bare():
    return np.full((480, 640, 3), 200, np.uint8)     # pale, no back's green: the shape says "not there"


def test_a_sure_back_check_decides_the_look_before_the_backs_shape():
    sure = {"name": "back", "confidence": 0.98}
    result = wrist_check.check(Arm(), Camera(bare()), 0.2, 0.0, HOVER, back=True, reader=Reader(sure, sure, sure))
    assert result["present"] is True and result["by"] == "back model" and len(result["reads"]) == wrist_check.FRAMES
    gone = {"name": "empty", "confidence": 0.97}
    assert wrist_check.check(Arm(), Camera(bare()), 0.2, 0.0, HOVER, back=True,
                             reader=Reader(gone, gone, gone))["present"] is False


def test_an_unsure_back_check_leaves_it_to_the_backs_shape_and_logs_its_reads():
    reads = [{"name": "back", "confidence": 0.98}, {"name": "empty", "confidence": 0.7}, {"name": "back", "confidence": 0.9}]
    result = wrist_check.check(Arm(), Camera(bare()), 0.2, 0.0, HOVER, back=True, reader=Reader(*reads))
    assert result["by"] == "back shape" and result["present"] is False and result["model_reads"] == reads


def test_a_face_up_look_never_asks_the_back_check():
    class Refuses:
        def read_many(self, frames):
            raise AssertionError("a face isn't the back check's question")
    wrist_check.check(Arm(), Camera(bare()), 0.2, 0.0, HOVER, back=False, reader=Refuses())
