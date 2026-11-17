"""The base camera's photos (base_cam.rows) and the split and scoring a reader is trained and checked on."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from holdup_sessions import write_session
from mahjong import base_cam


def test_rows_are_one_per_photo_and_a_sim_session_says_so(tmp_path):
    out = write_session(tmp_path, tiles=("5C", "3C"), holds=1, frames=2)
    rows = base_cam.rows(out)
    assert len(rows) == 2 * 2 * 2 and base_cam.is_sim(out)
    assert {(r["truth"], r["kind"]) for r in rows} == {(t, k) for t in ("5C", "3C") for k in base_cam.KINDS}
    assert all(r["pose"] is None and r["daylight"] == "none" for r in rows)       # no pose: not taken by the arm
    assert len(list((out / "photos" / "5C").glob("*.jpg"))) == 4


def test_a_discarded_shot_leaves_no_row(tmp_path):
    out = write_session(tmp_path, tiles=("5C",), holds=1, frames=1)
    with (out / "events.jsonl").open("a") as f:
        f.write(json.dumps({"type": "holdup_shot_discarded", "shot": 1}) + "\n")
    assert [r["kind"] for r in base_cam.rows(out)] == ["turned"]


def test_square_around_a_box_stays_inside_the_frame():
    image = np.zeros((480, 640, 3), np.uint8)
    assert base_cam.square_around(image, [600.0, 400.0, 640.0, 480.0]).shape[:2] == (224, 224)


def fake_rows(sessions):
    rows = []
    for session in sessions:
        for truth in ("5C", "3C", "2C"):
            for kind in base_cam.KINDS:
                for hold in range(1, 5):
                    rows.append({"truth": truth, "kind": kind, "hold": hold, "shot": hold, "session": session,
                                 "path": f"{session}/{truth}_{kind}_{hold}.jpg"})
    return rows


def test_split_holds_out_the_newest_session_or_the_last_holds():
    two = base_cam.split(fake_rows(["a_holdup", "b_holdup"]), "auto")
    assert {r["session"] for r in two[0]} == {"a_holdup"} and {r["session"] for r in two[1]} == {"b_holdup"} and two[3] is None
    one = base_cam.split(fake_rows(["a_holdup"]), "auto", test_holds=2)
    assert {r["hold"] for r in one[1]} == {3, 4} and {r["hold"] for r in one[0]} == {1, 2}
    assert "OPTIMISTIC" in one[3]
    assert len(one[0]) + len(one[1]) == 24


def test_a_split_by_hold_keeps_play_holds_in_training():
    """Play's holds are numbered per name from 1 and are a different tile each: 'the last holds' would be all of them."""
    play = [{"truth": t, "kind": base_cam.UNKNOWN_KIND, "hold": h, "shot": h, "session": "c_play_fd", "path": f"c/{t}_{h}.jpg"}
            for t in ("5C", "3C") for h in (1, 2)]
    train, test, _, _ = base_cam.split(fake_rows(["a_holdup"]) + play, "hold", test_holds=2)
    assert len([r for r in train if r["session"] == "c_play_fd"]) == 4 and not [r for r in test if r["session"] == "c_play_fd"]
    assert {r["hold"] for r in test} == {3, 4}


def test_check_summary_compares_with_chance_and_a_constant_answer():
    rows = fake_rows(["a_holdup"])
    s = base_cam.check_summary(rows, [r["truth"] if r["truth"] != "2C" else "5C" for r in rows])
    assert s["tiles"] == 3 and abs(s["right"] - 2 / 3) < 1e-9 and abs(s["chance"] - 1 / 3) < 1e-9
    assert s["per_tile"]["2C"] == 0.0 and s["confusions"][0][0] == "2C->5C"
