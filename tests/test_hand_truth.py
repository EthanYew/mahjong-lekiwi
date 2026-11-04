"""Labelling detector frames from a person's names (mahjong/hand_truth.py), on a small made-up session. Pixels map to the mat as
xy = pixel / 1000. Each case here was a wrong label or a lost frame on 20261104_183142_play."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from mahjong import detector, hand_truth

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

HOME = {"P1": (200, 100), "P2": (250, 100), "P3": (200, 150), "P4": (250, 150)}
NAMES = {"P1": "5C", "P2": "3C", "P3": "8C", "P4": "4B"}
ZONE = (300, 100)


def box(name, conf, centre):
    x, y = centre
    return {"tile_name": name, "confidence": conf, "xyxy": [x - 12, y - 9, x + 12, y + 9]}


def steady(misread=None):
    """The four tiles of the hand, read right unless `misread` says otherwise."""
    return [box((misread or {}).get(pid, NAMES[pid]), 0.9, xy) for pid, xy in HOME.items()]


class Log:
    def __init__(self, folder):
        self.dir, self.events, self.n = Path(folder), [], 0

    def event(self, event_type, **fields):
        self.events.append({"time": "t", "type": event_type, **fields})

    def scan(self, label, frames):
        self.n += 1
        name = f"{self.n:03d}_{label}"
        folder = self.dir / "scans" / name
        folder.mkdir(parents=True)
        (folder / "reads.json").write_text(json.dumps({"names": detector.TILE_NAMES, "size": [848, 480], "frames": frames}))
        for i in range(len(frames)):
            cv2.imwrite(str(folder / f"f{i}.jpg"), np.zeros((480, 848, 3), np.uint8))
        tiles = {tuple(detector._centre(b["xyxy"])): b for b in frames[0]}
        self.event("detections", scan=self.n, label=label, frames=f"scans/{name}",
                   tiles=[{"tile_name": b["tile_name"], "confidence": b["confidence"], "pixel": list(c),
                           "xy": [c[0] / 1000, c[1] / 1000]} for c, b in tiles.items()])

    def write(self):
        (self.dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in self.events))
        return self.dir


def hand(tmp_path, after=None):
    """Registered, checked by a person, one tile drawn at the zone and discarded."""
    log = Log(tmp_path / "20261104_120000_play")
    log.event("session_start", kind="play")
    log.scan("register", [steady({"P1": "9C"})] * 2)
    log.scan("register", [steady({"P1": "9C"})] * 2)
    for pid, xy in HOME.items():
        log.event("placement", id=pid, spot=pid, xy=[xy[0] / 1000, xy[1] / 1000], truth=None, truth_source="unconfirmed")
    log.event("hand_confirmed", confirmed=NAMES)
    log.scan("drawn", [steady() + [box("9D", 0.9, ZONE)]] * 2)                  # seen before its placement is written
    log.event("placement", id="P5", spot="zone", xy=[ZONE[0] / 1000, ZONE[1] / 1000], truth=None, truth_source="unconfirmed")
    log.event("intervention", kind="name_asked", placement="P5", answer="9D")
    log.scan("after_pickup", [steady()] * 2)
    log.event("attempt", placement="P5", spot="zone")
    log.event("discard")
    log.scan("after_discard", [steady()] * 2)                                   # no longer there, its end not yet written
    log.event("placement_ended", id="P5", spot="zone", why="discarded")
    if after:
        after(log)
    return log.write()


def test_a_misread_tile_takes_the_name_a_person_gave_it(tmp_path):
    frames, refused = hand_truth.labelled(hand(tmp_path))
    first = frames[0]
    assert {t["tile_name"] for t in first["truth"] if t["placement"] == "P1"} == {"5C"}
    assert first["wrong"] == [("5C", "9C")] and not refused


def test_a_drawn_tile_is_labelled_in_the_scans_before_its_placement_and_gone_after_its_discard(tmp_path):
    frames, _ = hand_truth.labelled(hand(tmp_path))
    tiles = {f["label"]: {t["placement"] for t in f["truth"]} for f in frames}
    assert "P5" in tiles["drawn"]
    assert "P5" not in tiles["after_discard"]


def test_scans_during_a_pickup_are_not_used(tmp_path):
    frames, _ = hand_truth.labelled(hand(tmp_path))
    assert "after_pickup" not in {f["label"] for f in frames}


def test_a_tile_missed_in_a_frame_borrows_its_box_and_counts_as_missed(tmp_path):
    def after(log):
        log.scan("choose", [steady()[1:], steady()])                            # P1 unseen in the first frame
    frames, _ = hand_truth.labelled(hand(tmp_path, after))
    frame = next(f for f in frames if f["label"] == "choose" and f["index"] == 0)
    assert frame["missed"] == ["5C"]
    assert next(t for t in frame["truth"] if t["placement"] == "P1")["xyxy"] == box("5C", 0.9, HOME["P1"])["xyxy"]


def test_a_tile_never_seen_in_its_place_refuses_the_frame(tmp_path):
    log = Log(tmp_path / "20261104_120000_play")
    log.event("session_start", kind="play")
    for k in range(2):
        log.scan("register", [steady()[1:] + [box("1D", 0.9, (300, 200))]] * 2)  # enough tiles to fit the mat, none at P1
    for pid, xy in HOME.items():
        log.event("placement", id=pid, spot=pid, xy=[xy[0] / 1000, xy[1] / 1000], truth=None, truth_source="unconfirmed")
    log.event("placement", id="P6", spot="S6", xy=[0.3, 0.2], truth=None, truth_source="unconfirmed")
    log.event("hand_confirmed", confirmed={**NAMES, "P6": "1D"})
    frames, refused = hand_truth.labelled(log.write())
    assert not frames and refused["a tile with no box"] == 4


def test_a_confident_box_on_no_tile_refuses_the_frame_but_the_bin_doesnt(tmp_path):
    def after(log):
        log.scan("choose", [steady() + [box("7B", 0.9, (225, 125))]])           # on the mat, between the tiles
        log.scan("choose", [steady() + [box("7B", 0.3, (225, 125))]])           # a hesitation isn't a tile
        log.scan("choose", [steady() + [box("8B", 0.9, (700, 400))]])           # far from every tile: the bin
    frames, refused = hand_truth.labelled(hand(tmp_path, after))
    assert refused["a box on the mat on no tile"] == 1
    assert sum(f["label"] == "choose" for f in frames) == 2


def test_a_tile_nobody_named_refuses_the_scan(tmp_path):
    log = Log(tmp_path / "20261104_120000_play")
    log.event("session_start", kind="play")
    log.scan("register", [steady()] * 2)
    log.scan("register", [steady()] * 2)
    for pid, xy in HOME.items():
        log.event("placement", id=pid, spot=pid, xy=[xy[0] / 1000, xy[1] / 1000], truth=None, truth_source="unconfirmed")
    frames, refused = hand_truth.labelled(log.write())
    assert not frames and refused["a tile nobody named"] == 4


def test_a_moved_tile_keeps_what_was_said_of_either_placement():
    events = [{"type": "placement", "id": "A"}, {"type": "placement", "id": "B", "from_placement": "A"},
              {"type": "placement", "id": "C"}, {"type": "placement", "id": "D", "from_placement": "C"},
              {"type": "hand_audit", "confirmed": {"B": "1B", "C": "2B", "D": "3B"}}]
    assert hand_truth.stated_names(events) == {"A": "1B", "B": "1B"}     # C and D were given two names: neither is named


def test_a_returned_tile_is_followed_from_where_it_landed_not_the_zone(tmp_path):
    def after(log):
        log.scan("drawn", [steady() + [box("1B", 0.9, ZONE)]])
        log.event("placement", id="P6", spot="zone", xy=[ZONE[0] / 1000, ZONE[1] / 1000], truth=None, truth_source="unconfirmed")
        log.event("attempt", placement="P6", spot="zone")
        log.scan("after_return", [steady() + [box("1B", 0.9, (300, 150))]])
        log.event("placement_ended", id="P6", spot="zone", why="moved")
        log.event("placement", id="P7", spot="S7", xy=[0.3, 0.15], from_placement="P6", truth=None, truth_source="game")
        log.event("hand_audit", confirmed={"P7": "1B"})
        log.scan("choose", [steady() + [box("1B", 0.9, (300, 150))]])
    frames, refused = hand_truth.labelled(hand(tmp_path, after))
    last = [f for f in frames if f["label"] == "choose"][-1]
    assert {t["placement"]: t["tile_name"] for t in last["truth"]}["P7"] == "1B"
    assert not refused


def test_picking_keeps_the_shaky_frames_of_each_arrangement_one_per_scan():
    def frame(scan, index, wrong=(), tiles=("5C", "3C")):
        return {"scan": scan, "index": index, "wrong": list(wrong), "missed": [],
                "truth": [{"tile_name": t, "placement": t} for t in tiles]}
    frames = [frame("a", 0), frame("a", 1, [("5C", "9C")]), frame("b", 0), frame("c", 0)]
    assert [(f["scan"], f["index"]) for f in hand_truth.pick(frames, per_arrangement=2, per_hard_name=1)] == [("a", 1), ("b", 0)]


def test_export_uses_the_persons_names_and_refuses_an_exam_session(tmp_path, monkeypatch):
    frames, _ = hand_truth.labelled(hand(tmp_path))
    monkeypatch.setattr(detector, "load_exam", lambda path=None: {"sessions": {}})
    out = tmp_path / "export"
    hand_truth.export(frames[:1], out)
    label = next((out / "train" / "labels").glob("*.txt")).read_text()
    assert str(detector.TILE_NAMES.index("5C")) in {line.split()[0] for line in label.splitlines()}
    assert str(detector.TILE_NAMES.index("9C")) not in {line.split()[0] for line in label.splitlines()}
    monkeypatch.setattr(detector, "load_exam", lambda path=None: {"sessions": {frames[0]["session"]: {}}})
    with pytest.raises(SystemExit):
        hand_truth.export(frames[:1], tmp_path / "export2")


def test_the_script_reports_exports_and_builds_a_training_set(tmp_path, monkeypatch, capsys):
    import hand_frames
    monkeypatch.setattr(detector, "load_exam", lambda path=None: {"sessions": {}})
    session = hand(tmp_path)
    base = tmp_path / "base"
    for split in ("train", "valid"):
        (base / split / "images").mkdir(parents=True)
        (base / split / "labels").mkdir(parents=True)
    (base / "train" / "images" / "old.jpg").write_bytes(b"x")
    (base / "train" / "labels" / "old.txt").write_text("")
    (base / "data.yaml").write_text(json.dumps({"names": detector.TILE_NAMES}))
    out = tmp_path / "train_new"
    hand_frames.main([session.name, "--sessions-dir", str(tmp_path), "--out", str(out), "--onto", str(base)])
    printed = capsys.readouterr().out
    assert "5C->9C" in printed
    assert (out / "train" / "images" / "old.jpg").exists()
    assert any(p.name.startswith(session.name) for p in (out / "train" / "images").iterdir())
    assert (out / "data.yaml").exists()
    with pytest.raises(SystemExit):                     # never over an existing folder
        hand_frames.main([session.name, "--sessions-dir", str(tmp_path), "--out", str(out)])
