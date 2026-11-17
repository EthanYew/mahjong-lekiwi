"""The game logic: shanten against the `mahjong` library, the discard ranking, the hand checks."""
import json
import random
import subprocess
import sys

import pytest

from mahjong import game

# the library's 34-tile order: man, pin, sou, then E S W N haku hatsu chun. Ours is C D B + the same
# honours, so the arrays line up (suits are interchangeable for shanten anyway)
LIBRARY = """
import json, sys
from mahjong.shanten import Shanten
s = Shanten()
print(json.dumps([s.calculate_shanten(c) for c in json.load(sys.stdin)]))
"""


def library_shanten(hands, tmp_path):
    """The library's answers, run isolated (-I) from outside the repo: this repo's `mahjong` package
    shadows it otherwise. None if the library isn't installed."""
    out = subprocess.run([sys.executable, "-I", "-c", LIBRARY], input=json.dumps(hands), capture_output=True,
                         text=True, cwd=tmp_path)
    return json.loads(out.stdout) if out.returncode == 0 else None


def random_hand(rng, size):
    wall = [k for k in range(34) for _ in range(4)]
    c = [0] * 34
    for k in rng.sample(wall, size):
        c[k] += 1
    return c


def test_shanten_agrees_with_the_library(tmp_path):
    rng = random.Random(7)
    hands = [random_hand(rng, 13) for _ in range(150)] + [random_hand(rng, 14) for _ in range(150)]
    # shapes random hands rarely reach: near-complete, seven pairs, thirteen orphans
    named = [["1C", "2C", "3C", "4D", "5D", "6D", "7B", "8B", "9B", "EW", "EW", "EW", "RD"],
             ["1C", "1C", "4D", "4D", "7B", "7B", "9B", "9B", "EW", "EW", "RD", "RD", "GD"],
             ["1C", "9C", "1D", "9D", "1B", "9B", "EW", "SW", "WW", "NW", "WD", "GD", "RD"],
             ["1C", "1C", "1C", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "9C", "9C", "5C"]]
    hands += [game.counts(h) for h in named]
    expected = library_shanten(hands, tmp_path)
    if expected is None:
        pytest.skip("the mahjong library isn't installed")
    assert [game.shanten(c) for c in hands] == expected


def test_known_hands():
    assert game.shanten(game.counts(["1C", "2C", "3C", "4D", "5D", "6D", "7B", "8B", "9B", "EW", "EW", "EW", "RD"])) == 0
    assert game.is_win(["1C", "2C", "3C", "4D", "5D", "6D", "7B", "8B", "9B", "EW", "EW", "EW", "RD", "RD"])
    assert not game.is_win(["1C", "2C", "3C", "4D", "5D", "6D", "7B", "8B", "9B", "EW", "EW", "EW", "RD", "GD"])
    assert game.is_win(["1C", "1C", "4D", "4D", "7B", "7B", "9B", "9B", "EW", "EW", "RD", "RD", "GD", "GD"])
    assert not game.is_win(["2C"] * 14)                                  # five of a tile: a misread, never a win
    assert not game.is_win(["2C"] * 5 + ["3C", "4C", "5C", "6C", "7C", "8C", "EW", "EW", "EW"])


def test_flowers_and_typos_are_refused():
    with pytest.raises(ValueError):
        game.counts(["1F"])
    assert not game.is_playable("1F") and game.is_playable("RD")


def test_discard_keeps_the_ready_hand_and_prefers_the_drawn_tile():
    hand = ["1C", "2C", "3C", "4D", "5D", "6D", "7B", "8B", "9B", "EW", "EW", "EW", "RD"]
    best, candidates = game.choose_discard(hand, "NW")
    assert best["index"] == 13 and best["tile_name"] == "NW" and best["shanten_after"] == 0
    # discarding RD instead is just as good (wait on NW instead of RD), so the drawn tile wins the tie
    rd = next(c for c in candidates if c["tile_name"] == "RD")
    assert (rd["shanten_after"], rd["ukeire_tiles"]) == (best["shanten_after"], best["ukeire_tiles"])


def test_copies_left_change_the_discard():
    # 13 tiles waiting on RD (single wait) vs NW after swapping: keep the wait with copies left
    hand = ["1C", "2C", "3C", "4D", "5D", "6D", "7B", "8B", "9B", "EW", "EW", "EW", "RD"]
    best, _ = game.choose_discard(hand, "NW", visible=["NW", "NW"])
    # NW has 1 copy left (2 visible + the drawn one), RD 3: discard NW, wait on RD
    assert best["tile_name"] == "NW"
    best, _ = game.choose_discard(hand, "NW", visible=["RD", "RD", "RD"])
    # all RDs gone (3 visible + 1 in hand): waiting on RD can't win, so discard RD and wait on NW
    assert best["tile_name"] == "RD" and best["ukeire_tiles"] == 3     # the other 3 NW


def test_a_ready_hand_names_exactly_the_tiles_that_win():
    # the hand 20261104_183142_play waited on from turn 11: 3456789C 123456B wins on 3C, 6C and 9C
    hand = ["3C", "4C", "5C", "6C", "7C", "8C", "9C", "1B", "2B", "3B", "4B", "5B", "6B"]
    assert game.improving_tiles(hand) == [("3C", 3), ("6C", 3), ("9C", 3)]
    assert game.seeking(hand) == "Ready (shanten 0). Wins on: 3C (3 left), 6C (3 left), 9C (3 left): 9 tiles left"
    assert game.improving_tiles(hand, visible=["6C", "6C"]) == [("3C", 3), ("6C", 1), ("9C", 3)]
    for name in game.NAMES:
        if hand.count(name) < game.COPIES:
            assert game.is_win(hand + [name]) == (name in ("3C", "6C", "9C")), name


def test_a_hand_not_ready_says_how_far_and_which_draws_help():
    hand = ["1C", "2C", "3C", "4D", "5D", "6D", "7B", "8B", "EW", "SW", "WW", "NW", "RD"]
    text = game.seeking(hand)
    away = game.shanten(game.counts(hand))
    assert away > 0 and text.startswith(f"{away} from ready (shanten {away}). Draws that get closer: ")
    assert "9B (4 left)" in text and "6B (4 left)" in text and "RD (3 left)" in text    # none held; one RD held


def test_a_ready_hand_holding_every_copy_of_its_wait_says_so():
    # shanten calls this ready, but its only wait is the 1C it already holds all four of
    hand = ["1C", "1C", "1C", "1C", "4D", "5D", "6D", "7B", "8B", "9B", "EW", "EW", "EW"]
    assert game.shanten(game.counts(hand)) == 0 and game.improving_tiles(hand) == []
    assert "no tile can complete it" in game.seeking(hand)


def test_ukeire_counts_kinds_and_copies():
    hand = ["1C", "2C", "3C", "4D", "5D", "6D", "7B", "8B", "9B", "EW", "EW", "EW", "RD"]
    assert game.ukeire(hand) == (1, 3, 0)               # only RD, 3 copies left
    assert game.ukeire(hand, visible=["RD", "RD"]) == (1, 1, 0)


def test_hand_checks():
    assert game.too_many(["5D"] * 5 + ["1C"] * 4) == ["5D"]
    assert game.hand_mismatches({0: "1C", 1: "2C", 2: "3C"}, {0: "1C", 1: "3C"}) == {1: ("2C", "3C"), 2: ("3C", None)}
