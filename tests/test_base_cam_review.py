"""The base camera's photos as a person reviewed them (mahjong.base_cam, data/holdup_review.jsonl) and scripts/base_cam_review.py:
rows by tile laid on a spot, renames counted under the right name, the counts table, the window driven by fake clicks."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import base_cam, base_cam_collect, config, photo_review

POSES = ("base_view_1", "base_view_2")


def write_session(folder, arrangements, frames=2, daylight="bright", order="interleaved"):
    """A hold-up session: [{spot: tile}] per arrangement, each tile held once (2 poses x `frames` frames)."""
    folder.mkdir(parents=True)
    lines = [{"type": "session_start", "sim": False},
             {"type": "holdup_conditions", "conditions": {"daylight": daylight}, "source": "arm", "order": order}]
    holds, shot = {}, 0
    for n, tiles in enumerate(arrangements, 1):
        lines.append({"type": "holdup_arrangement", "n": n, "laying": "as_layout", "rounds": 1, "tiles": tiles})
        for spot, tile in tiles.items():
            shot += 1
            hold = holds[tile] = holds.get(tile, 0) + 1
            files, poses = [], []
            for i in range(len(POSES) * frames):
                photo = folder / "photos" / tile / f"upright_h{hold}_{i}.jpg"
                photo.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(photo), np.full((480, 640, 3), 100, np.uint8))
                files.append(str(photo.relative_to(folder)))
                poses.append(POSES[i // frames])
            lines.append({"type": "holdup_shot", "shot": shot, "tile": tile, "kind": "upright", "hold": hold, "files": files,
                          "frame_poses": poses, "spot": spot, "source": "arm"})
    (folder / "events.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def two_arrangements(root):
    write_session(root / "20261112_100000_holdup", [{"S5": "3B", "S7": "3D"}, {"S5": "3D", "S7": "3B"}])
    return root / "20261112_100000_holdup"


def test_one_review_photo_per_pose_of_each_hold_grouped_by_tile_laid_on_a_spot(tmp_path):
    session = two_arrangements(tmp_path)
    photos = base_cam.review_photos([session])
    assert len(photos) == 4 * len(POSES)                    # 4 holds, a photo per pose; the pose's other frames go with it
    assert all(len(p["frames"]) == 2 for p in photos)
    assert sorted({p["group"] for p in photos}) == [f"{session.name}/a1/S5/3B", f"{session.name}/a1/S7/3D",
                                                     f"{session.name}/a2/S5/3D", f"{session.name}/a2/S7/3B"]


def test_a_rename_moves_a_hold_to_its_right_name_and_a_wrong_photo_drops_out_of_training(tmp_path):
    session = two_arrangements(tmp_path)
    photos = base_cam.review_photos([session])
    laid_wrong = [p["key"] for p in photos if p["group"].endswith("a2/S5/3D")]       # that was really a 3F
    one_blurred = next(p["key"] for p in photos if p["group"].endswith("a1/S5/3B") and p["pose"] == "base_view_2")
    photo_review.record(laid_wrong, "wrong", "3F", base_cam.REVIEW)
    photo_review.record([one_blurred], "wrong", None, base_cam.REVIEW)
    rows = base_cam.rows(session)
    assert len(rows) == 4 * 4 - 2                           # the blurred pose's 2 frames are gone
    renamed = [r for r in rows if r["filed_as"] != r["truth"]]
    assert len(renamed) == 4 and {r["truth"] for r in renamed} == {"3F"}
    counts = base_cam_collect.hold_counts(tmp_path)
    assert counts == {("3B", "upright"): 2, ("3D", "upright"): 1, ("3F", "upright"): 1}


def test_a_renamed_hold_counts_even_when_its_number_is_one_the_right_name_has(tmp_path):
    session = two_arrangements(tmp_path)          # 3B's hold 1 and 3D's hold 1: rename 3D's to 3B and both count
    keys = [p["key"] for p in base_cam.review_photos([session]) if p["group"].endswith("a1/S7/3D")]
    photo_review.record(keys, "wrong", "3B", base_cam.REVIEW)
    assert base_cam_collect.hold_counts(tmp_path)[("3B", "upright")] == 3


def test_tile_counts_have_every_tile_with_its_light_and_how_much_is_reviewed(tmp_path):
    session = two_arrangements(tmp_path)
    write_session(tmp_path / "20261112_200000_holdup", [{"S5": "3B"}], daylight="none")
    first = base_cam.review_photos([session])[0]
    photo_review.record([first["key"]], "ok", None, base_cam.REVIEW)
    counts = base_cam_collect.tile_counts(tmp_path)
    assert counts["3B"] == {"holds": {"upright": 3, "turned": 0}, "play_holds": 0, "waiting": 0, "photos": 12, "reviewed": 2, "sessions": 2,
                            "daylight": {"bright": 2, "none": 1}}
    assert counts["9D"]["holds"] == {"upright": 0, "turned": 0} and counts["9D"]["photos"] == 0


def test_counts_print_playable_tiles_fewest_first_then_flowers_and_the_sessions_left_out(tmp_path, monkeypatch, capsys):
    import base_cam_review
    two_arrangements(tmp_path)
    write_session(tmp_path / "20261106_105954_holdup", [{"S5": "EW"}], order=None)
    write_session(tmp_path / "20261112_200000_holdup", [{"S5": "2F"}])
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    base_cam_review.main(["--counts"])
    out = capsys.readouterr().out
    table = [line.split()[0] for line in out.splitlines() if line[:2] in {"3B", "3D", "9D", "2F"}]
    assert table.index("9D") < table.index("3B") and table[-1] == "2F"          # never collected first; flowers last
    assert "20261106_105954_holdup" in out                                     # not counted, and said so


def test_main_reviews_a_whole_row_with_one_click_on_its_label(tmp_path, monkeypatch):
    import base_cam_review
    two_arrangements(tmp_path)
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    callback, keys = {}, iter([-1, ord("q")])
    R = base_cam_review.Reviewer

    def wait_key(ms):
        key = next(keys, ord("q"))       # ask_pumping pumps waitKey too
        if key == -1:              # right-click the first row's label, answer 3F: every photo of that laying renamed
            callback["f"](cv2.EVENT_RBUTTONDOWN, 5, R.HEADER_H + 5, 0, None)
        return key
    monkeypatch.setattr(cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, image: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, f: callback.update(f=f))
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    base_cam_review.main(["--name", "3B", "--only"], ask=lambda prompt: "3f")
    review = base_cam.load_review()
    assert len(review) == len(POSES) and {r["name"] for r in review.values()} == {"3F"}    # one hold, its 2 poses
    assert base_cam_collect.hold_counts(tmp_path)[("3F", "upright")] == 1


def test_a_cell_shows_the_tile_box_crop_the_reader_sees(tmp_path, monkeypatch):
    import base_cam_review
    session = two_arrangements(tmp_path)
    photo = base_cam.review_photos([session])[0]
    reviewer = base_cam_review.Reviewer([photo], {}, ["3B"], boxes={"base_view_1": [100, 0, 300, 200]})
    image = reviewer.image_of(photo)
    assert image.shape[:2] == (reviewer.THUMB_H, reviewer.THUMB_W)


# --- holds saved in face-down play wait for a person's review (play_face_down.py, the owner's rule of 2026-11-17) ---

def write_play_session(root, tiles, filed_by="reader"):
    """A play_fd session: one hold per tile at both poses, filed `review_first` under the given names."""
    folder = root / "20261118_100000_play_fd"
    folder.mkdir(parents=True)
    lines = [{"type": "session_start", "sim": False}, {"type": "holdup_conditions", "conditions": {"daylight": "dim"}, "order": "play"}]
    for shot, tile in enumerate(tiles, 1):
        files = []
        for i, pose in enumerate(POSES):
            photo = folder / "photos" / tile / f"unknown_h{shot}_{i}.jpg"
            photo.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(photo), np.full((480, 640, 3), 100, np.uint8))
            files.append(str(photo.relative_to(folder)))
        lines.append({"type": "holdup_shot", **base_cam_collect.shot_event(tile, base_cam.UNKNOWN_KIND, shot, files, list(POSES), shot=shot,
                                                                          spot="zone", review_first=True, filed_by=filed_by)})
    (folder / "events.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return folder


def test_a_hold_from_play_is_in_no_row_until_a_person_reviews_it(tmp_path):
    session = write_play_session(tmp_path, ["5C", "9C"])
    assert base_cam.rows(session) == []
    assert base_cam.waiting_for_review(session) == {1: "5C", 2: "9C"}
    photos = base_cam.review_photos([session])
    assert {p["filed_by"] for p in photos} == {"reader"} and all(p["review_first"] for p in photos)
    five = [p["key"] for p in photos if p["name"] == "5C"]
    nine = [p["key"] for p in photos if p["name"] == "9C"]
    photo_review.record(five, "ok", None, base_cam.REVIEW)
    photo_review.record(nine[:1], "wrong", "7C", base_cam.REVIEW)         # the reader read a 7C as 9C
    photo_review.record(nine[1:], "wrong", None, base_cam.REVIEW)         # and one pose showed no tile
    assert sorted((r["truth"], r["pose"]) for r in base_cam.rows(session)) == [("5C", POSES[0]), ("5C", POSES[1]), ("7C", POSES[0])]
    assert base_cam.waiting_for_review(session) == {}


def test_play_sessions_count_and_their_unreviewed_holds_show_as_waiting(tmp_path, monkeypatch, capsys):
    import base_cam_review
    write_play_session(tmp_path, ["5C", "5C", "9C"])
    counted, _ = base_cam_collect.counted_sessions(tmp_path)
    assert [s.name for s in counted] == ["20261118_100000_play_fd"]
    counts = base_cam_collect.tile_counts(tmp_path)
    assert counts["5C"]["waiting"] == 2 and counts["9C"]["waiting"] == 1
    assert counts["5C"]["holds"] == {"upright": 0, "turned": 0} and counts["5C"]["play_holds"] == 0
    photo_review.record([p["key"] for p in base_cam.review_photos(counted) if p["name"] == "9C"], "ok", None, base_cam.REVIEW)
    counts = base_cam_collect.tile_counts(tmp_path)
    assert counts["9C"]["play_holds"] == 1 and counts["9C"]["waiting"] == 0
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    base_cam_review.main(["--counts"])
    assert "2 hold(s) from play wait for review: python scripts/base_cam_review.py --play" in capsys.readouterr().out


def test_training_finds_play_sessions_but_only_their_reviewed_holds(tmp_path, monkeypatch):
    import base_cam_train
    session = write_play_session(tmp_path, ["5C"])
    write_session(tmp_path / "20261112_100000_holdup", [{"S5": "3B"}])
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    assert [f.name for f in base_cam_train.find_sessions(None)] == ["20261112_100000_holdup", session.name]
    assert base_cam.rows(session) == []


def test_a_photo_marked_wrong_with_no_name_can_be_named_in_a_later_run_with_all(tmp_path):
    """status.py --void and the face-down tally ask for such a photo to be named, and no new window showed it (2026-11-26)."""
    import base_cam_review
    photo = {"key": "s/a.jpg", "path": tmp_path / "a.jpg", "name": "3B", "session": "20261125_000000_play_fd", "pose": "base_view_1",
             "spot": "draw t1", "arrangement": None, "shot": 1, "hold": 1, "kind": "unknown", "group": "g"}
    log = tmp_path / "review.jsonl"
    review = {"s/a.jpg": {"photo": "s/a.jpg", "verdict": "wrong", "name": None}}
    assert base_cam_review.Reviewer([photo], dict(review), ["3B"], log=log, boxes={}).names == []
    reviewer = base_cam_review.Reviewer([photo], dict(review), ["3B", "3D"], show_all=True, log=log, boxes={})
    assert reviewer.names == ["3B"] and reviewer.rows == [[photo]]
    reviewer.mark([photo], "wrong", "3D")
    assert photo_review.name_of(photo, photo_review.load_review(log)) == "3D"


@pytest.mark.parametrize("unsaved, unpushed, worktree, said", [
    (["data/holdup_review.jsonl"], [], False, "From the main checkout, run:\n  python scripts/push_session_data.py"),
    (["data/holdup_review.jsonl"], [], True, "this worktree's data/holdup_review.jsonl"),
    ([], ["data/holdup_review.jsonl"], False, "git push origin main"),
    ([], ["data/holdup_review.jsonl"], True, None),                  # a worktree's branch: its commits go by its PR
    (["data/hover_corrections.jsonl"], ["docs/usage.md"], False, None),
    ([], [], False, None)])
def test_a_review_ends_by_reminding_where_verdicts_not_on_github_are(tmp_path, monkeypatch, capsys, unsaved, unpushed,
                                                                     worktree, said):
    import base_cam_review
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(base_cam, "REVIEW", tmp_path / "holdup_review.jsonl")
    two_arrangements(tmp_path)
    checkout = photo_review.checkout
    monkeypatch.setattr(checkout, "unsaved_samples", lambda repo=None: unsaved)
    monkeypatch.setattr(checkout, "unpushed_files", lambda repo=None: unpushed)
    monkeypatch.setattr(checkout, "is_worktree", lambda repo=None: worktree)
    monkeypatch.setattr(photo_review, "run", lambda *args: None)         # the window, quit straight away
    base_cam_review.main([], ask=lambda prompt: "")
    out = capsys.readouterr().out
    assert (said in out) if said else "REMINDER" not in out


def test_a_review_stopped_with_ctrl_c_still_reminds(tmp_path, monkeypatch, capsys):
    import base_cam_review
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    monkeypatch.setattr(base_cam, "REVIEW", tmp_path / "holdup_review.jsonl")
    two_arrangements(tmp_path)
    monkeypatch.setattr(photo_review.checkout, "unsaved_samples", lambda repo=None: ["data/holdup_review.jsonl"])

    def stopped(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(photo_review, "run", stopped)
    with pytest.raises(KeyboardInterrupt):
        base_cam_review.main([], ask=lambda prompt: "")
    assert "REMINDER" in capsys.readouterr().out
