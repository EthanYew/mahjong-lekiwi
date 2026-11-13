"""The base camera's photos as a person reviewed them (mahjong.base_cam, data/holdup_review.jsonl) and scripts/base_cam_review.py:
rows by tile laid on a spot, renames counted under the right name, the counts table, the window driven by fake clicks."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

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
    assert counts["3B"] == {"holds": {"upright": 3, "turned": 0}, "photos": 12, "reviewed": 2, "sessions": 2,
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
