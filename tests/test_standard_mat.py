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


def test_on_a_pale_surface_the_look_only_says_the_view_has_not_changed():
    tile = image("wrist_tile_a")
    template = wrist_check.view_template(tile)
    same = wrist_check.check(Arm(), Camera(tile), 0.2, 0.0, HOVER, template)
    assert same["present"] is True and same["by"] == "same view"
    assert wrist_check.check(Arm(), Camera(image("wrist_bare_at_disc")), 0.2, 0.0, HOVER, template)["present"] is None
    knocked = cv2.warpAffine(tile, np.float32([[1, 0, 45], [0, 1, 0]]), (640, 480))      # ~11mm over
    assert wrist_check.check(Arm(), Camera(knocked), 0.2, 0.0, HOVER, template)["present"] is None


def test_place_views_go_by_place_and_forget_a_tile_that_may_have_moved():
    views = wrist_check.PlaceViews()
    here, beside, far = (0.2, 0.0), (0.205, 0.0), (0.2, 0.1)
    assert views.template(here) is None
    views.keep(here, "picture")
    assert views.template(beside) == "picture" and views.template(far) is None   # a detection moves a few mm
    views.keep(beside, "newer")
    assert views.template(here) == "newer" and len(views.places) == 1
    views.forget(beside)
    assert views.template(here) is None


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

    def run(model, frame, conf):
        assert conf == config.DETECTION_PRESENCE_CONFIDENCE
        calls.append(frame)
        return [{"tile_name": "8D", "confidence": 0.9, "xyxy": [1, 2, 3, 4]}]
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


def test_a_tile_too_unsure_to_name_is_faint_and_its_second_guesses_are_not():
    from mahjong.perception import faint_tiles, merge_tiles

    def read(name, conf, frame, x):
        return {"tile_name": name, "confidence": conf, "frame": frame, "xy": (x, 0.0)}
    reads = [read("4D", 0.9, k, 0.2) for k in range(5)] + [read("7D", 0.12, k, 0.2) for k in range(5)]    # a second guess
    reads += [read("3C", 0.18, 0, 0.1), read("7C", 0.15, 1, 0.1), read("3C", 0.2, 2, 0.1)]                # a hard tile
    reads += [read("9D", 0.11, 3, 0.3)]                                                                   # one frame: a flicker
    tiles = merge_tiles([r for r in reads if r["confidence"] >= 0.25], 5)
    assert [t["tile_name"] for t in tiles] == ["4D"]
    (faint,) = faint_tiles(reads, tiles, 5)
    assert faint["tile_name"] == "3C" and faint["confidence"] == 0.2 and faint["xy"] == pytest.approx((0.1, 0.0))


def test_setup_check_keeps_the_tile_a_person_used_not_the_layouts_label():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import setup_check
    answers = iter(["8D", ""])
    assert setup_check.tile_names([{"id": "F2"}, {"id": "F3"}], lambda prompt: next(answers)) == {"F2": "8D"}


def test_a_back_between_the_jaws_is_found_by_its_shape_where_the_face_disc_reads_nothing():
    """20261116_213816_bench, scan 4: S11's back right there, and the face look said "not there" (as all 36 looks at backs did)."""
    back = image("wrist_back_look")
    assert wrist_check.face_fill(back) < 0.05
    assert wrist_check.back_blob(back) is not None
    result = wrist_check.check(Arm(), Camera(back), 0.2, 0.0, HOVER, back=True)
    assert result["present"] is True and result["by"] == "back shape" and result["backs_seen"] == wrist_check.FRAMES


def test_the_olive_stripe_in_a_dim_room_has_a_backs_green_but_not_its_shape():
    """20261103_214640_bench (face up, no daylight): at (180, 145) the stripe is a back's colour across a whole disc."""
    dim = image("wrist_dim_olive_stripe")
    hsv = cv2.cvtColor(dim, cv2.COLOR_BGR2HSV)
    disc = np.zeros(hsv.shape[:2], np.uint8)
    cv2.circle(disc, (180, 145), wrist_check.RADIUS_PX, 1, -1)
    green = cv2.inRange(hsv, np.array(wrist_check.BACK_HSV_LO, np.uint8), np.array(wrist_check.BACK_HSV_HI, np.uint8))
    assert (green[disc > 0] > 0).mean() > 0.8                            # colour alone would call it a back
    assert wrist_check.back_blob(dim, centre=(180, 145)) is None
    assert wrist_check.check(Arm(), Camera(dim), 0.2, 0.0, HOVER, back=True)["present"] is False    # a face there, not a back
    assert wrist_check.check(Arm(), Camera(image("wrist_bare_at_disc")), 0.2, 0.0, HOVER, back=True)["present"] is False


def test_a_back_look_cant_tell_when_frames_disagree_or_the_surface_is_backs_green():
    back, bare = image("wrist_back_look"), image("wrist_bare_at_disc")

    class Frames(Camera):
        def __init__(self, frames):
            self.frames = iter(frames)

        def read(self):
            return True, next(self.frames)
    split = wrist_check.check(Arm(), Frames([back, bare, bare]), 0.2, 0.0, HOVER, back=True)
    assert split["present"] is None and "some frames" in split["why"] and split["backs_seen"] == 1
    green = np.zeros_like(back)
    green[:] = cv2.cvtColor(np.uint8([[[36, 120, 85]]]), cv2.COLOR_HSV2BGR)[0, 0]
    assert wrist_check.check(Arm(), Camera(green), 0.2, 0.0, HOVER, back=True)["present"] is None


def test_a_back_look_with_the_backs_green_but_not_its_shape_cant_tell():
    """A back under the arm's shadow, or merged with a knocked neighbour, still fills the disc with its green."""
    stripe = np.full((480, 640, 3), 200, np.uint8)
    u, v = wrist_check.EXPECTED_PX
    stripe[:, u - 60:u + 60] = cv2.cvtColor(np.uint8([[[35, 120, 80]]]), cv2.COLOR_HSV2BGR)[0, 0]    # a band, not a block
    assert wrist_check.back_blob(stripe) is None and wrist_check.back_disc(stripe) > 0.9
    result = wrist_check.check(Arm(), Camera(stripe), 0.2, 0.0, HOVER, back=True)
    assert result["present"] is None and "shape" in result["why"]
