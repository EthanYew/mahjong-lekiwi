"""wrist_review.py and mahjong.wrist_photos on a made-up sessions folder: photos found by physical tile,
verdicts logged (never a file moved), renames counted under the right name."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import config, wrist_photos


def make_sessions(root):
    """A wrist session with two physical 3C tiles (2 views + 1 view, 2 frames each), a bench attempt at
    3C, and a simulated wrist session (never counted)."""
    def jpg(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((480, 640, 3), 128, np.uint8))
    wrist = root / "20261028_150939_wrist" / "wrist" / "3C"
    for stem in ("01_01_v00", "01_01_v01", "01_02_v00"):
        for k in (0, 1):
            jpg(wrist / f"{stem}_f{k}.jpg")
    jpg(root / "20261028_102732_bench" / "013_103524_3C" / "wrist_0000.jpg")
    sim = root / "20261028_160000_wrist"
    jpg(sim / "wrist" / "3C" / "01_01_v00_f0.jpg")
    (sim / "events.jsonl").write_text(json.dumps({"type": "session_start", "sim": True}) + "\n")


def test_photos_are_grouped_by_physical_tile_and_sim_sessions_left_out(tmp_path):
    make_sessions(tmp_path)
    found = wrist_photos.photos(tmp_path)
    assert len(found) == 4 and {p["source"] for p in found} == {"wrist_capture", "bench"}
    groups = [p["group"] for p in found if p["source"] == "wrist_capture"]
    assert groups.count("20261028_150939_wrist/01_01") == 2 and groups.count("20261028_150939_wrist/01_02") == 1
    assert all(len(p["frames"]) == 2 for p in found if p["source"] == "wrist_capture")       # f0 and f1 of each view


def test_review_marks_rows_renames_and_counts_follow(tmp_path, monkeypatch):
    import wrist_review
    make_sessions(tmp_path)
    log = tmp_path / "review.jsonl"
    reviewer = wrist_review.Reviewer(wrist_photos.photos(tmp_path), {}, ["3C"], log=log)
    assert [len(r) for r in reviewer.page_rows()] == [1, 2, 1]           # the bench attempt, tile 01_01, tile 01_02
    row = lambda r: wrist_review.HEADER_H + r * (wrist_review.THUMB_H + wrist_review.GAP) + 5
    reviewer.click(10, row(1))                                           # the label: the whole tile is wrong
    reviewer.rename(wrist_review.LABEL_W + 5, row(2), "2C")              # the other tile was really a 2C
    assert reviewer.next_page() is False                                 # the bench photo becomes ok; the end
    review = wrist_photos.load_review(log)
    assert [review[p["key"]]["verdict"] for p in wrist_photos.photos(tmp_path)].count("wrong") == 3
    counts = wrist_photos.counts(tmp_path, review)
    assert counts["wrist_capture"] == {"2C": 1} and counts["bench"] == {"3C": 1}
    assert all(p.exists() for p in tmp_path.rglob("*_f0.jpg"))          # nothing moved or deleted
    reviewer.click(10, row(1))                                           # clicking again: ok after all
    assert wrist_photos.counts(tmp_path, wrist_photos.load_review(log))["wrist_capture"]["3C"] == 2
    assert reviewer.render().shape[1] == wrist_review.LABEL_W + wrist_review.COLUMNS * (wrist_review.THUMB_W + wrist_review.GAP)
    again = wrist_review.Reviewer(wrist_photos.photos(tmp_path), wrist_photos.load_review(log), ["2C", "3C"], log=log)
    assert again.names == []                                             # everything reviewed: nothing to show


def test_main_reviews_with_the_mouse_and_keys(tmp_path, monkeypatch):
    import wrist_review
    make_sessions(tmp_path)
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(wrist_photos, "REVIEW", tmp_path / "review.jsonl")
    callback, keys = {}, iter([-1, ord(" ")])
    def wait_key(ms):
        key = next(keys)
        if key == -1:              # first frame: right-click the second tile's photo, then answer 2C
            callback["f"](cv2.EVENT_RBUTTONDOWN, wrist_review.LABEL_W + 5,
                          wrist_review.HEADER_H + wrist_review.THUMB_H + wrist_review.GAP + 5, 0, None)
        return key
    monkeypatch.setattr(cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, image: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, f: callback.update(f=f))
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    wrist_review.main(["--name", "3C", "--only"], ask=lambda prompt: "2c")
    review = wrist_photos.load_review()
    assert sorted(r["verdict"] for r in review.values()) == ["ok", "ok", "ok", "wrong"]
    assert [r["name"] for r in review.values() if r["verdict"] == "wrong"] == ["2C"]
