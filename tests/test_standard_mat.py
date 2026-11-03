"""The tile finders on real frames of the striped standard mat, where a tile's white face has no edge a colour threshold
can find (fixtures: sessions 20261103_120259_setup and 20261103_121112_bench)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mahjong import layout, sim, wrist_check
from mahjong import setup_check as sc
from mahjong.layout import find_face, find_tile

FIXTURES = Path(__file__).parent / "fixtures" / "standard_mat"
K = sim.SimPerception.intrinsics
T = np.load(Path(__file__).parents[1] / "assets" / "T_cam_to_base.npy")
HOVERED = {"centre_px": [365.0, 240.0], "area_px": 15000.0, "angle_deg": 0.0, "size_px": [142.0, 105.0]}


def image(name):
    return cv2.imread(str(FIXTURES / f"{name}.jpg"))


def boxes():
    return json.loads((FIXTURES / "realsense_boxes.json").read_text())


def box_xy(box):
    return layout.pixel_to_table([((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)], K, T)[0][:2]


def test_the_detectors_boxes_find_every_tile_on_the_stripes_without_a_yaw():
    frame, found = image("realsense"), boxes()
    assert len(found) >= 11
    for box in found:
        tile = find_tile(frame, {"xy": list(box_xy(box)), "yaw_deg": 0.0}, K, T, boxes=lambda _: found)
        assert tile is not None and np.linalg.norm(np.array(tile["xy"]) - box_xy(box)) < 0.01


def test_a_slice_of_stripe_is_not_a_tile():
    frame, found = image("realsense"), boxes()
    for box in found:         # a colour read near a real tile on this mat is a stripe, or nothing
        read = find_face(frame, {"xy": list(box_xy(box)), "yaw_deg": 0.0}, K, T)
        assert read is None or read["size_mm"][0] <= layout.FACE_MAX_ASPECT * read["size_mm"][1]


def test_no_box_no_tile_whatever_the_colour_says():
    frame = layout.render([(0.2, 0.0, 0.0)], K, T)     # a clear white face, but the detector saw nothing
    spot = {"xy": (0.2, 0.0), "yaw_deg": 0.0}
    assert find_face(frame, spot, K, T) is not None
    assert find_tile(frame, spot, K, T, boxes=lambda _: []) is None


def test_a_colour_read_that_agrees_with_the_box_gives_the_yaw():
    frame = layout.render([(0.2, 0.0, 10.0)], K, T)
    spot = {"xy": (0.2, 0.0), "yaw_deg": 0.0}
    u, v = layout.table_to_pixel([(0.2 - layout.BOX_TO_FACE_M[0], 0.0, layout.face_z(0.2, 0.0))], K, T)[0]
    agrees = find_tile(frame, spot, K, T, boxes=lambda _: [(u - 14, v - 14, u + 14, v + 14)])
    assert agrees["yaw_deg"] == pytest.approx(10.0, abs=3)
    u += 12                                            # ~18mm from the face: another blob, so the box stands alone
    alone = find_tile(frame, spot, K, T, boxes=lambda _: [(u - 14, v - 14, u + 14, v + 14)])
    assert alone["yaw_deg"] is None and layout.status(spot, alone) in ("close", "off")


def test_green_on_position_alone_when_no_yaw_was_read():
    spot = {"id": "S1", "xy": (0.2, 0.0), "yaw_deg": 30.0}
    found = {"xy": (0.201, 0.0), "yaw_deg": None, "size_mm": (30.0, 28.0)}
    assert layout.status(spot, found) == "ok"
    smoother = layout.Smoother()
    for _ in range(3):
        smoothed, state = smoother.update(spot, found)
    assert smoothed["yaw_deg"] is None and state == "ok"
    rows = layout.capture(lambda: image("realsense"), [spot], K, T, frames=1, boxes=lambda _: [])
    assert "dx_mm" not in rows[0]
    frame = layout.render([], K, T)
    u, v = layout.table_to_pixel([(0.2, 0.0, layout.face_z(0.2, 0.0))], K, T)[0]
    rows = layout.capture(lambda: frame, [spot], K, T, frames=2, boxes=lambda _: [(u - 14, v - 14, u + 14, v + 14)])
    assert rows[0]["yaw_deg"] is None and rows[0]["yaw_sd"] is None and rows[0]["status"] == "ok"
    assert "S1" in layout.capture_report(rows, layout.yaw_trend(rows))
    assert layout.draw(frame, [spot], K, T, found={"S1": found}, active="S1").shape == frame.shape


def test_colour_finds_nothing_on_the_stripes_in_the_wrist_view():
    assert sc.wrist_tile(image("wrist_bare_at_disc")) is None


def test_template_matching_finds_the_same_tile_again_and_not_an_empty_spot():
    template = sc.wrist_template(image("wrist_tile_a"), HOVERED["centre_px"])
    again = sc.match_tile(image("wrist_tile_a_again"), template, HOVERED)
    assert again["score"] > 0.9
    assert np.linalg.norm(np.array(again["centre_px"]) - HOVERED["centre_px"]) < 15
    assert abs(np.sqrt(again["area_px"] / HOVERED["area_px"]) - 1) < 0.1
    assert sc.match_tile(image("wrist_bare_at_disc"), template, HOVERED) is None


def test_template_matching_reads_a_shift_a_size_and_a_turn():
    frame = image("wrist_tile_a")
    template = sc.wrist_template(frame, HOVERED["centre_px"])
    moved = cv2.warpAffine(frame, np.float32([[1, 0, 20], [0, 1, -12]]), (640, 480))
    found = sc.match_tile(moved, template, HOVERED)
    assert found["centre_px"] == pytest.approx([385, 228], abs=2) and found["score"] > 0.98
    bigger = cv2.warpAffine(frame, cv2.getRotationMatrix2D(HOVERED["centre_px"], 0, 1.1), (640, 480))
    assert np.sqrt(sc.match_tile(bigger, template, HOVERED)["area_px"] / HOVERED["area_px"]) == pytest.approx(1.1, abs=0.03)
    turned = cv2.warpAffine(frame, cv2.getRotationMatrix2D(HOVERED["centre_px"], 6, 1.0), (640, 480))    # anticlockwise
    assert sc.match_tile(turned, template, HOVERED)["angle_deg"] == pytest.approx(-6, abs=3)


def test_a_marked_tile_stays_inside_the_frame_and_keeps_only_a_find_it_agrees_with():
    half = sc.WRIST_PATCH_HALF_PX
    assert sc.patch_centre((5, 470), (480, 640)) == [half, 480 - half]
    marked = sc.marked_tile((300, 200), None, (480, 640))
    assert marked["area_px"] == sc.WRIST_AREA_PX and sc.wrist_template(np.zeros((480, 640, 3), np.uint8), marked["centre_px"]).shape == (2 * half, 2 * half)
    near = {"centre_px": [305.0, 205.0], "area_px": 9000.0, "angle_deg": 12.0, "size_px": [120.0, 90.0]}
    assert sc.marked_tile((300, 200), near, (480, 640))["area_px"] == 9000.0
    assert sc.marked_tile((500, 200), near, (480, 640))["area_px"] == sc.WRIST_AREA_PX


class Arm:
    """A look's arm: lift and hover succeed."""
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


HOVER = {"offset_m": 0.045, "clearance_m": 0.04, "gripper": 21}


def test_the_wrist_look_wont_call_a_white_stripe_a_tile():
    bare = image("wrist_bare_at_disc")
    assert wrist_check.face_fill(bare) >= wrist_check.PRESENT_FILL       # the false "still there" this guards against
    result = wrist_check.check(Arm(), Camera(bare), 0.2, 0.0, HOVER)
    assert result["present"] is None and "pale" in result["why"] and result["frame"] is not None
    assert wrist_check.check(Arm(), Camera(image("wrist_tile_a")), 0.2, 0.0, HOVER)["present"] is None


def test_the_wrist_look_still_answers_on_a_dark_surface():
    dark = np.full((480, 640, 3), 25, np.uint8)
    assert wrist_check.check(Arm(), Camera(dark), 0.2, 0.0, HOVER)["present"] is False
    u, v = wrist_check.EXPECTED_PX
    cv2.rectangle(dark, (u - 70, v - 60), (u + 70, v + 60), (225, 235, 240), -1)
    assert wrist_check.check(Arm(), Camera(dark), 0.2, 0.0, HOVER)["present"] is True


def test_the_detector_runs_once_a_frame_however_many_spots_ask(monkeypatch):
    from mahjong import config, perception
    realsense = object.__new__(perception.RealSense)
    realsense._model, realsense._boxes_of, calls = object(), (None, []), []

    def run(model, frame):
        calls.append(frame)
        return [{"tile_name": "8D", "confidence": 0.9, "xyxy": [1, 2, 3, 4]},
                {"tile_name": "WD", "confidence": config.DETECTION_CONFIDENCE - 0.1, "xyxy": [5, 6, 7, 8]}]
    monkeypatch.setattr(perception, "run_detector", run)
    frame, other = np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8)
    assert realsense.tile_boxes(frame) == [[1, 2, 3, 4]] and realsense.tile_boxes(frame) == [[1, 2, 3, 4]]
    assert len(calls) == 1
    realsense.tile_boxes(other)
    assert len(calls) == 2


def test_marking_a_tile_draws_the_template_where_it_was_clicked():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import setup_check
    frame = image("wrist_bare_at_disc")
    shown = setup_check.mark_view(frame, {"id": "F6"}, (505, 205))
    assert shown.shape == frame.shape and (shown != frame).any() and (frame == image("wrist_bare_at_disc")).all()
