import json
import math

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


HAND = [{"id": f"S{i + 1}", "xy": [0.15 + 0.06 * (i % 2), 0.17 - 0.057 * (i // 2)]} for i in range(13)]
ZONE = [0.163, -0.163]


def bench_session(sessions, name, spot_xy, success=True, landed=True, carried_mm=5.0):
    import json
    folder = sessions / f"{name}_bench"
    folder.mkdir(parents=True)
    events = [{"type": "attempt", "spot": "S14", "spot_xy": spot_xy, "success": success, "attempt_dir": "001_x"}]
    if landed:
        events.append({"type": "return_landed", "attempt_dir": "001_x", "error_m": 0.003})
        for _ in range(markers.CARRY_MIN_LANDINGS):     # and carried to a slot, as a hand does
            events.append({"type": "return_landed", "attempt_dir": "002_carry", "error_m": carried_mm / 1000,
                           "carried": True, "from_xy": spot_xy})
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def carry_session(sessions, name, zone, home, scan):
    """A real carry's log, in the order bench.py writes it: the after-return scan, then a landing logged as not seen."""
    folder = sessions / f"{name}_bench"
    folder.mkdir(parents=True)
    events = [{"type": "session_start"},
              {"type": "attempt", "tile": "2C", "attempt_dir": "001_x", "detection_xy": zone, "hover_xy": zone,
               "spot": "S14", "spot_xy": zone, "success": True},
              {"type": "return", "attempt_dir": "001_x", "xy": home, "home_xy": home},
              {"type": "detections", "label": "after_return", "tiles": scan},
              {"type": "return_landed", "tile": "2C", "attempt_dir": "001_x", "home_xy": home, "landed_xy": None,
               "carried": True, "from_xy": zone}]
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_a_landing_logged_as_not_seen_is_scored_again_from_its_scan(tmp_path):
    # 20261104_195901_bench: logged as not seen, while the scan shows it 3mm from home under another name
    home = [0.192, 0.141]
    scan = [{"tile_name": "3C", "xy": [0.189, 0.139]}, {"tile_name": "2C", "xy": [0.187, 0.037]}]
    carry_session(tmp_path, "20261104_195901", ZONE, home, scan)
    assert markers.carried_from(ZONE, tmp_path) == pytest.approx([math.dist([0.189, 0.139], home)])


def test_a_landing_with_nothing_near_home_stays_not_seen_and_says_so(tmp_path):
    carry_session(tmp_path, "20261104_100000", ZONE, [0.192, 0.141], [{"tile_name": "2C", "xy": [0.187, 0.037]}])
    carry_session(tmp_path, "20261104_110000", ZONE, [0.192, 0.141], [])
    assert markers.carried_from(ZONE, tmp_path) == [math.inf, math.inf]
    result = markers.zone_test(ZONE, HAND, HAND + [{"id": "S14", "xy": ZONE}], sessions_dir=tmp_path)
    problem = next(p for p in result["problems"] if "carried" in p)
    assert "inf" not in problem and "2 not seen" in problem


def test_a_simulated_bench_doesnt_check_the_real_zone(tmp_path):
    folder = tmp_path / "20261104_100000_bench"
    folder.mkdir()
    events = [{"type": "session_start", "sim": True},
              {"type": "attempt", "spot_xy": ZONE, "success": True, "attempt_dir": "001_x"},
              {"type": "return_landed", "attempt_dir": "001_x", "error_m": 0.0}]
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    assert markers.benched_at(ZONE, tmp_path) == (0, 0, 0)


def test_a_voided_bench_doesnt_check_the_real_zone(tmp_path, monkeypatch):
    from mahjong import session
    folder = tmp_path / "20261104_120000_bench"
    folder.mkdir()
    events = [{"type": "session_start"}, {"type": "attempt", "spot_xy": ZONE, "success": True, "attempt_dir": "001_x"}]
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    assert markers.benched_at(ZONE, tmp_path) == (1, 1, 0)
    (tmp_path / "void.json").write_text(json.dumps({folder.name: "torn wrist frames"}))
    monkeypatch.setattr(session, "VOID_SESSIONS", tmp_path / "void.json")
    assert markers.benched_at(ZONE, tmp_path) == (0, 0, 0)


def test_the_zone_counts_as_checked_only_after_a_bench_pickup_and_return_there(tmp_path):
    test_layout = HAND + [{"id": "S14", "xy": [0.1627, -0.1626]}]
    result = markers.zone_test(ZONE, HAND, test_layout, sessions_dir=tmp_path)
    assert not result["ok"] and result["spot"] == "S14" and result["attempts"] == 0
    assert any(f"bench.py --layout {markers.ZONE_TEST_LAYOUT}" in f for f in result["fix"])

    bench_session(tmp_path, "20261029_094543", [0.1627, -0.1626], success=False)      # skipped or missed: not yet
    assert not markers.zone_test(ZONE, HAND, test_layout, sessions_dir=tmp_path)["ok"]
    bench_session(tmp_path, "20261030_100000", [0.1627, -0.1626])
    result = markers.zone_test(ZONE, HAND, test_layout, sessions_dir=tmp_path)
    assert result["ok"] and (result["attempts"], result["picked"], result["returned"]) == (2, 1, 1)


def test_the_zone_isnt_checked_until_tiles_carried_from_it_land_near_their_slot(tmp_path):
    test_layout = HAND + [{"id": "S14", "xy": ZONE}]
    bench_session(tmp_path, "20261104_100000", ZONE, carried_mm=18.0)     # the aim that hit the neighbours
    result = markers.zone_test(ZONE, HAND, test_layout, sessions_dir=tmp_path)
    assert not result["ok"] and "carried from the drop zone" in result["problems"][0]
    assert any("--carry S14,S1" in f for f in result["fix"])
    assert result["carried"] == markers.CARRY_MIN_LANDINGS and result["carried_median_mm"] == pytest.approx(18.0)

    bench_session(tmp_path, "20261104_110000", ZONE, carried_mm=5.0)
    bench_session(tmp_path, "20261104_120000", ZONE, carried_mm=5.0)
    assert markers.zone_test(ZONE, HAND, test_layout, sessions_dir=tmp_path)["ok"]


def test_the_standard_layouts_are_a_hand_plus_the_zone(tmp_path):
    from mahjong import layout, scorecard
    assert scorecard.standard_setup()["layout"] == markers.ZONE_TEST_LAYOUT
    hand, test = layout.load(markers.PLAY_LAYOUT)["spots"], layout.load(markers.ZONE_TEST_LAYOUT)["spots"]
    assert len(hand) == 13 and len(test) == 14
    zone = next(s for s in test if s["id"] == "S13")["xy"]
    bench_session(tmp_path, "20261103_100000", zone)
    result = markers.zone_test(zone, hand, test, sessions_dir=tmp_path)
    assert result["ok"] and result["spot"] == "S13"


def test_a_test_layout_off_the_zone_or_missing_a_slot_is_reported_with_the_commands_to_fix_it(tmp_path):
    bench_session(tmp_path, "20261030_100000", ZONE)
    off_zone = HAND + [{"id": "S14", "xy": [0.25, 0.0]}]
    result = markers.zone_test(ZONE, HAND, off_zone, sessions_dir=tmp_path)
    assert not result["ok"] and result["spot"] is None
    assert "--remove S14" in result["fix"][0] and "--add S14 0.1630 -0.1630" in result["fix"][1]

    retaught = [dict(s, xy=[s["xy"][0] + 0.01, s["xy"][1]]) if s["id"] == "S3" else s for s in HAND]
    result = markers.zone_test(ZONE, retaught, HAND + [{"id": "S14", "xy": ZONE}], sessions_dir=tmp_path)
    assert not result["ok"] and "S3" in result["problems"][0]
