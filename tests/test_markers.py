import cv2
import numpy as np
import pytest

from mahjong import config, markers
from mahjong.layout import table_to_pixel
from mahjong.sim import SimPerception


def draw_marker(image, xy, intrinsics, T, side_m=0.04, marker_id=markers.DRAWN_TILE_MARKER_ID):
    """The marker printed flat on the table at xy, as the RealSense would see it."""
    x, y = xy
    h = side_m / 2
    table = [(x + dx, y + dy, config.table_z(x + dx, y + dy)) for dx, dy in ((h, h), (h, -h), (-h, -h), (-h, h))]
    pixels = table_to_pixel(table, intrinsics, T).astype(np.float32)
    art = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(markers.DICTIONARY), marker_id, 120)
    art = cv2.copyMakeBorder(art, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)   # the white margin it's printed with
    s, m = art.shape[0], 20 * 160 / 120
    src = np.float32([[m, m], [s - m, m], [s - m, s - m], [m, s - m]])
    # order the table corners to match the image's (top-left first, clockwise) as the camera sees them
    order = np.argsort(np.arctan2(pixels[:, 1] - pixels[:, 1].mean(), pixels[:, 0] - pixels[:, 0].mean()))
    pixels = pixels[np.roll(order, -int(np.argmin([pixels[i].sum() for i in order])))]
    warp = cv2.warpPerspective(cv2.cvtColor(art, cv2.COLOR_GRAY2BGR), cv2.getPerspectiveTransform(src, pixels),
                               image.shape[1::-1], borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full(art.shape, 255, np.uint8), cv2.getPerspectiveTransform(src, pixels),
                               image.shape[1::-1])
    image[mask > 0] = warp[mask > 0]
    return image


def test_the_marker_is_found_where_it_lies():
    mat = SimPerception(tiles=[])
    image = draw_marker(mat.snapshot(), (0.17, -0.05), mat.intrinsics, mat.T_cam_to_base, side_m=0.06)
    corners = markers.find(image)
    assert corners is not None
    xy = markers.centre_xy(corners, mat.intrinsics, mat.T_cam_to_base)
    assert xy == pytest.approx([0.17, -0.05], abs=0.003)
    assert markers.find(image, marker_id=7) is None


def test_a_zone_from_another_calibration_is_ignored(tmp_path, monkeypatch):
    path = tmp_path / "zone.json"
    markers.save_zone([0.17, -0.05], path, session="test")
    assert markers.load_zone(path)["xy"] == [0.17, -0.05]
    monkeypatch.setattr(markers, "setup_fingerprint", lambda: {"t_cam_to_base_sha": "other"})
    assert markers.load_zone(path) is None
    assert markers.moved({"xy": [0.17, -0.05]}, [0.176, -0.05]) and not markers.moved({"xy": [0.17, -0.05]}, [0.172, -0.05])
