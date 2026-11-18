"""A power cut mid-session (2026-11-18, 20261118_230033_wrist): the end of a log padded with zero bytes or cut mid-line, and
the last photos written as 0-byte files. One damaged session mustn't stop every script that lists sessions, and the next
append to a damaged log mustn't merge with the cut line."""
import json

import cv2
import numpy as np
import pytest

from mahjong import back_check, session, wrist_photos
from mahjong.session import append_jsonl, read_jsonl

ROWS = [{"type": "session_start", "sim": False}, {"type": "wrist_view", "view": 0}]


@pytest.fixture(autouse=True)
def fresh_warnings(monkeypatch):
    monkeypatch.setattr(session, "_DAMAGE_REPORTED", set())


def log(path, tail=b""):
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in ROWS) + tail)
    return path


def test_zero_bytes_after_the_last_line_are_left_out_and_said_once(tmp_path, capsys):
    path = log(tmp_path / "events.jsonl", b"\0" * 566)
    assert read_jsonl(path) == ROWS
    assert "566 zero bytes" in capsys.readouterr().err
    read_jsonl(path)
    assert capsys.readouterr().err == ""


def test_a_last_line_cut_short_is_left_out(tmp_path, capsys):
    assert read_jsonl(log(tmp_path / "events.jsonl", b'{"type": "wrist_vi')) == ROWS
    assert "cut-off last line" in capsys.readouterr().err
    assert read_jsonl(log(tmp_path / "padded.jsonl", b'{"type": "wr' + b"\0" * 9)) == ROWS     # cut, then padded


def test_a_last_line_whole_but_for_its_newline_is_kept(tmp_path):
    assert read_jsonl(log(tmp_path / "events.jsonl", b'{"type": "end"}')) == ROWS + [{"type": "end"}]


def test_damage_before_the_last_whole_line_is_not_a_power_cut(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"type": "a"}\nnot json\n{"type": "b"}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(bad)
    zeros = tmp_path / "zeros.jsonl"
    zeros.write_bytes(b'{"tile": "6\0\0D"}\n{"type": "b"}\n')           # a zeroed block mid-file: no silent "6D"
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(zeros)


def test_a_whole_log_reads_as_before_without_a_word(tmp_path, capsys):
    assert read_jsonl(log(tmp_path / "events.jsonl")) == ROWS and capsys.readouterr().err == ""


def test_the_next_append_after_a_cut_starts_clean_and_keeps_the_cut_bytes(tmp_path, capsys):
    path = log(tmp_path / "episode_log.jsonl", b'{"episode_index": 7, "ti' + b"\0" * 4)
    append_jsonl(path, [{"episode_index": 8}])
    append_jsonl(path, [{"episode_index": 9}])
    assert read_jsonl(path) == ROWS + [{"episode_index": 8}, {"episode_index": 9}]
    assert (tmp_path / "episode_log.jsonl.cut").read_bytes() == b'{"episode_index": 7, "ti' + b"\0" * 4
    assert "moved to episode_log.jsonl.cut" in capsys.readouterr().err


def test_appending_to_a_whole_or_new_log_touches_nothing_else(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    append_jsonl(path, ROWS[:1])
    append_jsonl(path, ROWS[1:])
    assert read_jsonl(path) == ROWS
    assert not (tmp_path / "events.jsonl.cut").exists() and capsys.readouterr().err == ""


def test_session_events_go_through_the_repair(tmp_path, monkeypatch):
    from mahjong import config
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    s = session.Session("bench", ask_setup=False, sim=True)
    with open(s.dir / "events.jsonl", "ab") as f:
        f.write(b'{"type": "cut' + b"\0" * 20)
    s.event("after_the_cut")
    assert [e["type"] for e in session.read_events(s.dir)][-1] == "after_the_cut"
    s.close()


def jpg(path, damage=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))
    if damage == "empty":
        path.write_bytes(b"")
    elif damage == "cut":
        path.write_bytes(path.read_bytes()[:-40])
    elif damage == "padded":
        path.write_bytes(path.read_bytes()[:-40] + b"\0" * 40)


def test_photos_not_written_whole_are_left_out(tmp_path):
    wrist = tmp_path / "20261118_230033_wrist"
    wrist.mkdir()
    log(wrist / "events.jsonl", b"\0" * 100)
    jpg(wrist / "wrist_empty" / "empty" / "02_S2_v06_f0.jpg")
    jpg(wrist / "wrist_empty" / "empty" / "02_S2_v07_f0.jpg", "empty")
    jpg(wrist / "wrist_empty" / "empty" / "02_S3_v00_f0.jpg", "cut")
    jpg(wrist / "wrist_empty" / "empty" / "02_S3_v01_f0.jpg", "padded")
    jpg(wrist / "wrist" / "9D" / "01_01_v00_f0.jpg")
    jpg(wrist / "wrist" / "9D" / "01_01_v00_f1.jpg", "empty")
    jpg(wrist / "wrist" / "9D" / "01_01_v01_f0.jpg", "cut")
    bench = tmp_path / "20261118_221723_bench"
    bench.mkdir()
    log(bench / "events.jsonl")
    jpg(bench / "001_221906_6D" / "wrist_0000.jpg", "empty")
    assert [p["key"] for p in back_check.photos(tmp_path)] == ["20261118_230033_wrist/wrist_empty/empty/02_S2_v06_f0.jpg"]
    named = wrist_photos.photos(tmp_path)
    assert [p["key"] for p in named] == ["20261118_230033_wrist/wrist/9D/01_01_v00_f0.jpg"]
    assert [f.name for f in named[0]["frames"]] == ["01_01_v00_f0.jpg"]


def test_a_photo_gone_mid_listing_is_left_out(tmp_path):
    assert not wrist_photos.whole_jpeg(tmp_path / "gone.jpg")
