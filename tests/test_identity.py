from mahjong.identity import Votes, scan_votes


def tile(name, confidence=0.9, votes=None):
    return {"tile_name": name, "confidence": confidence, **({"votes": votes} if votes else {})}


def test_one_scan_isnt_enough_and_a_split_read_isnt_sure():
    v = Votes()
    v.add(0, tile("8C"))
    assert v.sure(0) is None                    # one scan
    v.add(0, tile("8C"))
    assert v.sure(0) == "8C"
    v.add(1, tile("8C", votes={"8C": 2.46, "9C": 1.53}))
    v.add(1, tile("8C", votes={"8C": 2.4, "9C": 1.6}))
    assert v.leader(1)[0] == "8C" and v.sure(1) is None


def test_votes_build_up_over_flicker_and_reset_on_a_move():
    v = Votes()
    for name in ["5D"] * 9 + ["6D"]:
        v.add(3, tile(name))
    assert v.sure(3) == "5D"
    v.reset(3)
    assert v.leader(3) == (None, 0.0, 0)


def test_a_known_name_outweighs_flicker_but_not_a_real_change():
    v = Votes()
    v.known(2, "RD")
    v.add(2, tile("GD", 0.6))
    assert v.sure(2) == "RD"
    for _ in range(12):                         # the tile really is a GD now: the vote splits, then turns
        v.add(2, tile("GD"))
    assert v.sure(2) != "RD"


def test_scan_votes_fall_back_to_the_best_read():
    assert scan_votes(tile("1B", 0.7)) == {"1B": 0.7}
    assert scan_votes(tile("1B", 0.7, votes={"1B": 3.1, "7B": 0.4})) == {"1B": 3.1, "7B": 0.4}


def test_doubt_is_about_the_name_read_and_its_confidence():
    v = Votes()
    v.add(0, tile("7C", 0.99))
    assert "misread" in v.doubt(0, "7C")        # however confident: an 8C reads 7C at up to 0.91
    v.add(1, tile("1D", 0.9))
    v.add(1, tile("1D", 0.8))
    assert v.doubt(1, "1D") is None
    v.add(2, tile("1D", 0.7))
    v.add(2, tile("1D", 0.8))
    assert "0.75" in v.doubt(2, "1D")
    v.add(3, {"tile_name": "1D", "confidence": 0.0})    # no read of a name: found by the wrist camera
    assert "0.00" in v.doubt(3, "1D")
    v.reset(2)
    assert "0.00" in v.doubt(2, "1D")
