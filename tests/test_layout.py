"""Layout overlay geometry and the tile finder, on synthetic RealSense frames through the real hand-eye
transform (intrinsics fitted to a 2026-10-23 overview frame: detections land within ~3px of the tiles)."""
import numpy as np
import pytest

from mahjong import config, layout

T = np.load(config.T_CAM_TO_BASE)
INTRINSICS = {"fx": 590.0, "fy": 590.0, "ppx": 424.0, "ppy": 244.0}


def test_table_to_pixel_and_back_round_trip():
    points = np.array([(x, y, layout.face_z(x, y)) for x, y in [(0.15, 0.0), (0.22, -0.15), (0.13, 0.19)]])
    pixels = layout.table_to_pixel(points, INTRINSICS, T)
    assert ((pixels > 0) & (pixels < [848, 480])).all()       # the mat is in view
    assert np.abs(layout.pixel_to_table(pixels, INTRINSICS, T) - points).max() < 1e-4


def test_image_up_is_away_from_the_robot():
    near, far = layout.table_to_pixel([(0.15, 0.0, layout.face_z(0.15, 0.0)), (0.22, 0.0, layout.face_z(0.22, 0.0))],
                                      INTRINSICS, T)
    assert far[1] < near[1]


@pytest.mark.parametrize("x,y,yaw", [(0.15, 0.01, 0.0), (0.22, -0.10, 15.0), (0.20, 0.15, -30.0), (0.16, -0.16, 30.0)])
def test_finder_recovers_a_tiles_position_and_yaw(x, y, yaw):
    image = layout.render([(x, y, yaw)], INTRINSICS, T)
    found = layout.find_tile(image, {"xy": (x + 0.006, y - 0.004), "yaw_deg": 0.0}, INTRINSICS, T)
    assert found is not None
    assert np.hypot(found["xy"][0] - x, found["xy"][1] - y) < 0.0015
    assert abs(layout.wrap_yaw(found["yaw_deg"] - yaw)) < 3.0
    assert abs(found["size_mm"][0] - config.TILE_FACE_M[0] * 1000) < 5     # ~1px of rasterisation per side at ~1.5mm/px


def test_finder_takes_the_tile_at_its_spot_not_the_neighbour():
    image = layout.render([(0.15, 0.01, 0.0), (0.15, 0.07, 0.0)], INTRINSICS, T)
    found = layout.find_tile(image, {"xy": (0.15, 0.06), "yaw_deg": 0.0}, INTRINSICS, T)
    assert abs(found["xy"][1] - 0.07) < 0.002


def test_no_tile_no_find():
    image = layout.render([(0.22, -0.10, 0.0)], INTRINSICS, T)
    assert layout.find_tile(image, {"xy": (0.15, 0.10), "yaw_deg": 0.0}, INTRINSICS, T) is None


def test_status_by_position_and_yaw():
    spot = {"xy": (0.2, 0.0), "yaw_deg": 15.0}
    assert layout.status(spot, None) == "missing"
    assert layout.status(spot, {"xy": (0.202, 0.001), "yaw_deg": 17.0}) == "ok"
    assert layout.status(spot, {"xy": (0.207, 0.0), "yaw_deg": 15.0}) == "close"
    assert layout.status(spot, {"xy": (0.2, 0.0), "yaw_deg": 30.0}) == "off"
    assert layout.status({"xy": (0.2, 0.0), "yaw_deg": 88.0}, {"xy": (0.2, 0.0), "yaw_deg": -89.0}) == "ok"   # 180° symmetric


def test_teach_save_load_and_draw(tmp_path):
    tiles = [(0.15, 0.01, 0.0), (0.22, -0.10, 15.0)]
    image = layout.render(tiles, INTRINSICS, T)
    detections = [{"tile_name": "3B", "xy": (0.151, 0.012)}, {"tile_name": "9B", "xy": (0.219, -0.098)}]
    spots = layout.spots_from_tiles(detections, image, INTRINSICS, T)
    assert [s["id"] for s in spots] == ["S1", "S2"] and spots[0]["label"] == "3B"       # left to right
    assert abs(spots[1]["yaw_deg"] - 15.0) < 3.0
    path = tmp_path / "layouts.json"
    layout.save("eval_v1", spots, path=path)
    loaded = layout.load("eval_v1", path=path)
    assert [s["id"] for s in loaded["spots"]] == ["S1", "S2"]
    out = layout.draw(image, loaded["spots"], INTRINSICS, T, active="S2", ghost=image)
    assert out.shape == image.shape and not np.array_equal(out, image)
    with pytest.raises(SystemExit):
        layout.load("missing", path=path)


def keys(*sequence):
    it = iter(sequence)
    return lambda: next(it)


def test_place_loop_accepts_skips_and_quits_with_the_measured_tile():
    spots = [{"id": "S1", "xy": (0.15, 0.01), "yaw_deg": 0.0}, {"id": "S2", "xy": (0.22, -0.10), "yaw_deg": 15.0}]
    image = layout.render([(0.151, 0.011, 1.0), (0.22, -0.10, 30.0)], INTRINSICS, T)
    shown = []
    got = layout.place(lambda: image, spots, "S1", INTRINSICS, T, ghost=image, show=shown.append,
                       wait_key=keys(ord("g"), 255, 32))
    assert got["action"] == "accept" and got["status"] == "ok" and len(shown) == 3
    got = layout.place(lambda: image, spots, "S2", INTRINSICS, T, show=shown.append, wait_key=keys(ord("s")))
    assert got["action"] == "skip" and got["status"] == "off"          # 30° where 15° was wanted
    assert layout.place(lambda: image, spots, "S2", INTRINSICS, T, show=shown.append, wait_key=keys(27))["action"] == "quit"


def test_script_adds_tilted_copies_and_removes_spots(tmp_path, monkeypatch):
    import argparse
    import importlib.util
    spec = importlib.util.spec_from_file_location("layout_script", config.REPO / "scripts" / "layout.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(config, "LAYOUTS", tmp_path / "layouts.json")
    layout.save("eval_v1", [{"id": "S1", "xy": (0.15, 0.01), "yaw_deg": 0.0}], ghost="eval_v1.jpg")

    def run(**kw):
        script.edit(argparse.Namespace(**{"name": "eval_v1", "add": None, "like": None, "yaw": 0.0, "remove": None, **kw}))
    run(add=["T1"], like="S1", yaw=15.0)
    run(add=["N1", "0.24", "0.17"])
    entry = layout.load("eval_v1")
    assert [(s["id"], s["xy"], s["yaw_deg"]) for s in entry["spots"]] == [
        ("S1", [0.15, 0.01], 0.0), ("T1", [0.15, 0.01], 15.0), ("N1", [0.24, 0.17], 0.0)]
    assert entry["ghost"] == "eval_v1.jpg"
    run(remove="N1")
    assert [s["id"] for s in layout.load("eval_v1")["spots"]] == ["S1", "T1"]
    with pytest.raises(SystemExit):
        run(add=["T1"], like="S1")            # ids are unique


def test_nearest_tile_within_the_merge_radius():
    tiles = [{"tile_name": "3B", "xy": (0.151, 0.014)}, {"tile_name": "2B", "xy": (0.151, 0.071)}]
    assert layout.nearest_tile(tiles, (0.153, 0.06))["tile_name"] == "2B"
    assert layout.nearest_tile(tiles, (0.22, -0.10)) is None
    assert layout.nearest_tile([], (0.15, 0.0)) is None


def test_green_holds_through_jitter_at_the_tolerance_edge():
    spot = {"id": "S1", "xy": (0.2, 0.0), "yaw_deg": 0.0}
    assert layout.status(spot, {"xy": (0.2055, 0.0), "yaw_deg": 0.0}) == "close"
    assert layout.status(spot, {"xy": (0.2055, 0.0), "yaw_deg": 0.0}, previous="ok") == "ok"      # hysteresis
    assert layout.status(spot, {"xy": (0.2085, 0.0), "yaw_deg": 0.0}, previous="ok") == "close"   # past 1.5x
    smoother, states = layout.Smoother(), []
    for k in range(30):        # per-frame yaw jumping 4.5 / 7.5 / 5 / 6.5 around a tile at the 6° edge
        _, state = smoother.update(spot, {"xy": (0.2, 0.0), "yaw_deg": [4.5, 7.5, 5.0, 6.5][k % 4], "size_mm": (35, 26)})
        states.append(state)
    assert len(set(states[5:])) == 1                       # settles on one colour instead of flickering


def test_smoother_drops_a_tile_that_is_gone():
    spot = {"id": "S1", "xy": (0.2, 0.0), "yaw_deg": 0.0}
    smoother = layout.Smoother()
    for _ in range(10):
        smoother.update(spot, {"xy": (0.2, 0.0), "yaw_deg": 0.0, "size_mm": (35, 26)})
    for _ in range(6):
        found, state = smoother.update(spot, None)
    assert found is None and state == "missing"


def test_capture_measures_every_spot_and_the_yaw_drift():
    spots = [{"id": f"{r}{i}", "xy": (x, y), "yaw_deg": 0.0}
             for r, x in (("A", 0.15), ("B", 0.22)) for i, y in enumerate([0.15, 0.05, -0.05, -0.12], 1)]
    # tiles turned 5° per 10cm of y (+ to the left), the pattern a perspective bias would give; B4 missing
    image = layout.render([(s["xy"][0], s["xy"][1], 50 * s["xy"][1]) for s in spots if s["id"] != "B4"], INTRINSICS, T)
    rows = layout.capture(lambda: image, spots, INTRINSICS, T, frames=3)
    by_id = {r["id"]: r for r in rows}
    assert by_id["B4"]["found_frames"] == 0 and "yaw_deg" not in by_id["B4"]
    assert abs(by_id["A1"]["yaw_deg"] - 7.5) < 2.5 and by_id["A1"]["found_frames"] == 3
    trend = layout.yaw_trend(rows)
    assert abs(trend["per_10cm_y"] - 5.0) < 1.0 and abs(trend["per_10cm_x"]) < 3.0
    narrow = layout.yaw_trend([r for r in rows if r["id"].startswith("A")])      # one row: no x span
    assert narrow["per_10cm_x"] is None and narrow["per_10cm_y"] is not None
    text = layout.capture_report(rows, trend)
    assert "B4" in text and "no tile found" in text and "per 10cm of y" in text


def test_schedule_repeats_each_spot_in_a_row():
    spots = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
    assert layout.schedule(spots, 2) == ["A", "A", "B", "B", "C", "C"]
    assert layout.schedule(spots, 1, start="B") == ["B", "C"]
    with pytest.raises(SystemExit):
        layout.schedule(spots, 1, start="Z")
