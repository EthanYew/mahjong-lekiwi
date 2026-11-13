from mahjong import config
from mahjong.perception import merge_tiles, choose_tile, is_back, plausible, sees_backs


def test_a_back_box_at_table_height_is_mat_pattern_and_a_face_box_isnt_judged_by_height():
    assert plausible(config.BACK, 0.019) and not plausible(config.BACK, 0.003)
    assert plausible("2C", 0.003)           # every face result so far was scanned without this check


def test_a_merged_tile_keeps_its_median_height():
    raw = [{**det(config.BACK, 0.20, 0.05, 0.8), "frame": k, "height_m": h} for k, h in enumerate((0.018, 0.020, 0.035))]
    (tile,) = merge_tiles(raw, n_frames=3)
    assert is_back(tile) and tile["height_m"] == 0.020


def test_sees_backs_only_with_the_class():
    class Weights:
        def __init__(self, names):
            self.names = names
    assert sees_backs(Weights(["2C", config.BACK])) and not sees_backs(Weights(["2C"]))


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


def test_a_scan_votes_on_each_tiles_name_across_frames():
    def at(name, conf, frame, x=0.20):
        return {**det(name, x, 0.05, conf), "frame": frame}
    raw = [at("3C", 0.9, 0), at("2C", 0.7, 1), at("2C", 0.75, 2), at("2C", 0.7, 3), at("2C", 0.35, 4),  # 2C outvotes one sure 3C
           at("WD", 0.95, 2, x=0.10),                                     # in one frame only: a flicker, not a tile
           at("5B", 0.3, 0, x=0.30), at("5B", 0.45, 1, x=0.30), at("5B", 0.4, 2, x=0.30)]   # never sure enough
    tiles = merge_tiles(raw, n_frames=5)
    assert [t["tile_name"] for t in tiles] == ["2C"]
    assert tiles[0]["frames_seen"] == 5 and tiles[0]["vote_confidence"] == 0.5  # 2C's 2.5 over the 5 frames it was seen in
    assert tiles[0]["confidence"] == 0.75                                       # its best single 2C read
    assert tiles[0]["votes"] == {"2C": 2.5, "3C": 0.9}
    assert [t["tile_name"] for t in merge_tiles([at("WD", 0.95, 0, x=0.10)], n_frames=1)] == ["WD"]   # a 1-frame scan


def test_choose_tile_by_name():
    tiles = merge_tiles([det("8D", 0.16, -0.16, 0.9), det("9D", 0.14, 0.2, 0.8)])
    assert choose_tile(tiles, tile_name="8d")["tile_name"] == "8D"
    assert choose_tile(tiles, tile_name="7B") is None


class DepthFrame:
    """pyrealsense2's depth frame as far as robust_depth uses it: raises outside the frame, as it does."""
    def __init__(self, w=848, h=480, d=0.5):
        self.w, self.h, self.d = w, h, d

    def get_width(self):
        return self.w

    def get_height(self):
        return self.h

    def get_distance(self, x, y):
        if not (0 <= x < self.w and 0 <= y < self.h):
            raise RuntimeError('out of range value for argument "x"')
        return self.d


def test_depth_at_a_box_on_the_frame_edge_uses_the_samples_inside_it():
    from mahjong.perception import robust_depth
    frame = DepthFrame()
    assert robust_depth(frame, 2, 240) == 0.5 and robust_depth(frame, 847, 479) == 0.5
    assert robust_depth(frame, 424, 240) == 0.5
