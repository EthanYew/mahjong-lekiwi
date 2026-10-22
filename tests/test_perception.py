from mahjong import config
from mahjong.perception import merge_tiles, choose_tile


def det(name, x, y, conf):
    return {"tile_name": name, "confidence": conf, "xy": (x, y), "pixel": (0, 0)}


def test_merge_by_position_not_label():
    raw = [det("8B", 0.20, 0.05, 0.9), det("WD", 0.201, 0.051, 0.6), det("8B", 0.199, 0.049, 0.8),
           det("3D", 0.15, -0.10, 0.7)]
    tiles = merge_tiles(raw)
    assert [t["tile_name"] for t in tiles] == ["8B", "3D"]          # left to right
    assert tiles[0]["reads"] == 3
    assert abs(tiles[0]["xy"][0] - 0.2) < 1e-3
    assert tiles[0]["z"] == config.grasp_z(*tiles[0]["xy"])


def test_choose_tile_by_name():
    tiles = merge_tiles([det("8D", 0.16, -0.16, 0.9), det("9D", 0.14, 0.2, 0.8)])
    assert choose_tile(tiles, tile_name="8d")["tile_name"] == "8D"
    assert choose_tile(tiles, tile_name="7B") is None
