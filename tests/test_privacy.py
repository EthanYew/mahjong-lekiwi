"""Blurring faces in the base camera's photos (mahjong/privacy.py, scripts/blur_faces.py). The pose model itself is driven by
one slow test; the rest use a finder that says where the face is."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mahjong import base_cam, privacy

FACE = (200, 100, 80, 80)


def striped(height=480, width=640):
    image = np.zeros((height, width, 3), np.uint8)
    image[:, ::4] = 255                         # fine stripes: everything a blur removes
    return image


def test_blur_boxes_removes_detail_inside_the_box_and_changes_nothing_outside():
    image = striped()
    out = privacy.blur_boxes(image, [FACE])
    x, y, w, h = FACE
    assert out[y:y + h, x:x + w].std() < 0.2 * image[y:y + h, x:x + w].std()
    outside = np.ones(image.shape[:2], bool)
    pad = int(privacy.PAD * w) + 1
    outside[y - pad:y + h + pad, x - pad:x + w + pad] = False
    assert (out[outside] == image[outside]).all()
    assert out is not image and (image == striped()).all()        # a copy: the camera's frame is untouched


def test_a_box_at_the_edge_of_the_frame_is_clipped_not_an_error():
    out = privacy.blur_boxes(striped(), [(600, 0, 80, 80), (-20, 440, 60, 60), (700, 700, 10, 10)])
    assert out.shape == (480, 640, 3)


def test_no_face_means_the_same_image_back():
    image = striped()
    out, boxes = privacy.blur_faces(image, finder=lambda im: [])
    assert out is image and boxes == []


def test_the_head_is_placed_round_the_visible_face_keypoints_else_the_top_of_the_person():
    keypoints = np.zeros((17, 2))
    keypoints[0], keypoints[1], keypoints[2] = (300, 120), (320, 110), (280, 110)        # nose, eyes
    confidence = np.zeros(17)
    confidence[:3] = 0.9
    x, y, w, h = privacy.head_box((200, 0, 400, 400), keypoints, confidence)
    assert x < 300 < x + w and y < 115 < y + h and w >= privacy.MIN_HEAD_PX
    x, y, w, h = privacy.head_box((500, 0, 640, 380), keypoints, np.zeros(17))          # no keypoints seen: the top of the box
    assert y == 0 and abs(h - privacy.HEAD_OF_PERSON * 380) < 2 and 500 <= x and x + w <= 640


def test_a_person_the_size_of_the_frame_is_the_held_tile_and_is_left_alone():
    # on the first photos: the close-up tile seen as a 0.7-confident person [23,0,633,479]; a real person at the edge [526,44,640,356];
    # a real person close to the camera [302,0,640,478] (0.52 of the frame)
    people = np.array([[23, 0, 633, 479], [526, 44, 640, 356], [302, 0, 640, 478]], float)
    keypoints = np.full((3, 17, 2), 300.0)
    confidence = np.full((3, 17), 0.9)
    heads = privacy.heads(people, keypoints, confidence, (480, 640, 3))
    assert len(heads) == 2


def test_blur_file_only_rewrites_with_apply(tmp_path):
    path = tmp_path / "p.jpg"
    cv2.imwrite(str(path), striped())
    before = path.read_bytes()
    assert privacy.blur_file(path, apply=False, finder=lambda im: [FACE]) == [FACE]
    assert path.read_bytes() == before
    privacy.blur_file(path, apply=True, finder=lambda im: [FACE])
    assert path.read_bytes() != before
    assert privacy.blur_file(tmp_path / "missing.jpg", apply=True, finder=lambda im: [FACE]) == []


def test_saved_hold_up_photos_have_the_face_blurred(tmp_path, monkeypatch):
    monkeypatch.setattr(privacy, "find_faces", lambda image: [FACE])
    files = base_cam.save_shot(tmp_path, {"tile": "5C", "kind": "upright", "hold": 1}, [striped()])
    saved = cv2.imread(str(tmp_path / files[0]))
    x, y, w, h = FACE
    assert saved[y:y + h, x:x + w].std() < 0.3 * striped()[y:y + h, x:x + w].std()


def test_the_blur_script_is_a_dry_run_unless_apply(tmp_path, capsys):
    import blur_faces
    session = tmp_path / "20261105_100000_holdup"
    for name in ("photos/5C/upright_h1_0.jpg", "view_check.jpg", "captures/001_base.jpg", "captures/001_wrist.jpg"):
        path = session / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), striped())
    before = {p: p.read_bytes() for p in session.rglob("*.jpg")}
    hit, total = blur_faces.main([str(session), "--sheet", str(tmp_path / "sheet.jpg")], finder=lambda im: [FACE])
    assert (hit, total) == (3, 3)                       # the wrist photo isn't the base camera's, so it isn't looked at
    assert {p: p.read_bytes() for p in session.rglob("*.jpg")} == before and (tmp_path / "sheet.jpg").exists()
    assert "dry run" in capsys.readouterr().out
    blur_faces.main([str(session), "--apply", "--sheet", str(tmp_path / "sheet.jpg")], finder=lambda im: [FACE])
    changed = {p.name for p in before if p.read_bytes() != before[p]}
    assert changed == {"upright_h1_0.jpg", "view_check.jpg", "001_base.jpg"}


@pytest.mark.slow
def test_the_real_model_finds_a_person_and_not_the_held_tile():
    """Needs assets/yolo11n-pose.pt. A blank frame has no one in it; the check on real photos is the sheet blur_faces.py makes."""
    if not privacy.WEIGHTS.exists():
        pytest.skip("no pose weights")
    assert privacy.find_faces(np.zeros((480, 640, 3), np.uint8)) == []
