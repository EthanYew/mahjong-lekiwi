"""A power cut mid-session (2026-11-18, 20261118_230033_wrist): the end of a log padded with zero bytes or cut mid-line, and
the last photos written as 0-byte files. One damaged session mustn't stop every script that lists sessions."""
import json

import cv2
import numpy as np
import pytest

from mahjong import back_check, wrist_photos
from mahjong.session import read_jsonl

ROWS = [{"type": "session_start", "sim": False}, {"type": "wrist_view", "view": 0}]


def log(path, tail=b""):
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in ROWS) + tail)
    return path


def test_zero_bytes_after_the_last_line_are_dropped_and_said(tmp_path, capsys):
    path = log(tmp_path / "events.jsonl", b"\0" * 566)
    assert read_jsonl(path) == ROWS
    assert "566 zero bytes" in capsys.readouterr().err
    read_jsonl(path)
    assert capsys.readouterr().err == ""                     # said once per file


def test_a_last_line_cut_short_is_skipped(tmp_path, capsys):
    assert read_jsonl(log(tmp_path / "events.jsonl", b'{"type": "wrist_vi')) == ROWS
    assert "cut-off last line" in capsys.readouterr().err
    assert read_jsonl(log(tmp_path / "zeros.jsonl", b'{"type": "wr\0\0\0')) == ROWS      # cut, then padded


def test_a_bad_line_before_the_end_is_not_a_power_cut(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"type": "a"}\nnot json\n{"type": "b"}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_a_whole_log_reads_as_before_without_a_word(tmp_path, capsys):
    assert read_jsonl(log(tmp_path / "events.jsonl")) == ROWS and capsys.readouterr().err == ""


def jpg(path, empty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if empty:
        path.write_bytes(b"")
    else:
        cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))


def test_photos_written_as_0_byte_files_are_left_out(tmp_path):
    wrist = tmp_path / "20261118_230033_wrist"
    wrist.mkdir()
    log(wrist / "events.jsonl", b"\0" * 100)
    jpg(wrist / "wrist_empty" / "empty" / "02_S2_v06_f0.jpg")
    jpg(wrist / "wrist_empty" / "empty" / "02_S2_v07_f0.jpg", empty=True)
    jpg(wrist / "wrist" / "9D" / "01_01_v00_f0.jpg")
    jpg(wrist / "wrist" / "9D" / "01_01_v00_f1.jpg", empty=True)
    jpg(wrist / "wrist" / "9D" / "01_01_v01_f0.jpg", empty=True)
    bench = tmp_path / "20261118_221723_bench"
    bench.mkdir()
    log(bench / "events.jsonl")
    jpg(bench / "001_221906_6D" / "wrist_0000.jpg", empty=True)
    assert [p["key"] for p in back_check.photos(tmp_path)] == ["20261118_230033_wrist/wrist_empty/empty/02_S2_v06_f0.jpg"]
    named = wrist_photos.photos(tmp_path)
    assert [p["key"] for p in named] == ["20261118_230033_wrist/wrist/9D/01_01_v00_f0.jpg"]
    assert [f.name for f in named[0]["frames"]] == ["01_01_v00_f0.jpg"]
