"""The game side of autonomous play (scripts/play.py): which tile to discard, whether the hand has won, and
whether the table still matches what the robot thinks it holds. No hardware here.

Terms: *shanten* is how many tiles a hand is from ready (0 = one tile from winning, "tenpai"; -1 = won).
*Ukeire* are the draws that lower it; counted two ways here, as kinds of tile and as copies still left
(4 minus the ones in the hand and the visible discards), since a kind whose copies are all discarded
can't be drawn (docs/autonomy_plan.md, "Decided").

Our own shanten code, not the `mahjong` library's: this repo's package is also called `mahjong`, so
`from mahjong.shanten import Shanten` finds this package from the repo root (tests/test_game.py checks
the two agree, running the library from outside the repo). Closed hands only: no calls (pon/chi/kan),
no flowers.

Tile names are the detector's (detector.TILE_NAMES): 1-9 + C (characters), D (dots), B (bamboo); winds
EW SW WW NW; dragons WD GD RD; flowers 1F-4F (not playable here).
"""
from collections import Counter
from functools import lru_cache

SUITS = "CDB"                                       # 0-8 characters, 9-17 dots, 18-26 bamboo
HONORS = ["EW", "SW", "WW", "NW", "WD", "GD", "RD"]  # 27-33
NAMES = [f"{n}{s}" for s in SUITS for n in range(1, 10)] + HONORS
INDEX = {name: i for i, name in enumerate(NAMES)}
TERMINALS = [0, 8, 9, 17, 18, 26] + list(range(27, 34))
HAND_SIZE = 13
COPIES = 4


def is_playable(name):
    return name in INDEX


def counts(names):
    """34 counts from tile names; ValueError on a name that isn't a playable tile (a flower, a typo)."""
    out = [0] * 34
    for name in names:
        if name not in INDEX:
            raise ValueError(f"{name!r} isn't a playable tile (flowers aren't handled)")
        out[INDEX[name]] += 1
    return out


@lru_cache(maxsize=None)
def _suit_options(suit_counts):
    """Every useful (melds, partials) split of one suit's 9 counts, as a frozenset. A partial is a pair or
    two tiles one or two apart; isolated tiles are skipped."""
    c = list(suit_counts)
    i = next((k for k, v in enumerate(c) if v), None)
    if i is None:
        return frozenset({(0, 0)})
    out = set()

    def take(shape, melds, partials):
        rest = c[:]
        for k in shape:
            if k > 8 or rest[k] == 0:
                return
            rest[k] -= 1
        out.update((m + melds, p + partials) for m, p in _suit_options(tuple(rest)))

    take((i, i, i), 1, 0)
    take((i, i + 1, i + 2), 1, 0)
    take((i, i), 0, 1)
    take((i, i + 1), 0, 1)
    take((i, i + 2), 0, 1)
    rest = c[:]
    rest[i] -= 1                                    # i as an isolated tile
    out.update(_suit_options(tuple(rest)))
    return frozenset(out)


def _honor_options(honor_counts):
    melds = sum(v >= 3 for v in honor_counts)
    pairs = sum(v == 2 for v in honor_counts)
    return {(melds, pairs)}


def _regular(c):
    """Shanten toward four melds and a pair: 8 - 2 x melds - partials - (pair), with melds + partials
    at most 4, minimised over every split, with and without a pair set aside as the head."""
    best = 8

    def score(c, head):
        combos = {(0, 0)}
        for part in (_suit_options(tuple(c[0:9])), _suit_options(tuple(c[9:18])), _suit_options(tuple(c[18:27])),
                     _honor_options(c[27:34])):
            combos = {(m1 + m2, p1 + p2) for m1, p1 in combos for m2, p2 in part}
        value = 8
        for m, p in combos:
            m = min(m, 4)
            p = min(p, 4 - m)
            value = min(value, 8 - 2 * m - p - head)
        return value

    best = min(best, score(c, 0))
    for k in range(34):
        if c[k] >= 2:
            c[k] -= 2
            best = min(best, score(c, 1))
            c[k] += 2
    return best


def _seven_pairs(c):
    pairs = sum(v >= 2 for v in c)
    kinds = sum(v >= 1 for v in c)
    return 6 - pairs + max(0, 7 - kinds)


def _thirteen_orphans(c):
    kinds = sum(c[k] >= 1 for k in TERMINALS)
    pair = any(c[k] >= 2 for k in TERMINALS)
    return 13 - kinds - pair


def shanten(c):
    """Shanten of a closed hand (34 counts, 13 or 14 tiles): the best of the regular shape, seven pairs
    and thirteen orphans. -1 on a 14-tile win."""
    c = list(c)
    return min(_regular(c), _seven_pairs(c), _thirteen_orphans(c))


def copies_left(hand_counts, visible=()):
    """Copies of each kind that could still be drawn: 4 minus the hand's and the visible tiles' (discards)."""
    seen = counts(visible)
    return [max(0, COPIES - h - s) for h, s in zip(hand_counts, seen)]


def improving_tiles(names13, visible=()):
    """[(name, copies left)] for each kind of draw that lowers a 13-tile hand's shanten. On a ready hand
    (shanten 0) these are exactly the tiles that win."""
    c = counts(names13)
    base = shanten(c)
    left = copies_left(c, visible)
    out = []
    for k in range(34):
        if c[k] >= COPIES:
            continue
        c[k] += 1
        if shanten(c) < base:
            out.append((NAMES[k], left[k]))
        c[k] -= 1
    return out


def ukeire(names13, visible=()):
    """(kinds, tiles, base shanten) for a 13-tile hand: the kinds of draw that lower its shanten, and how
    many copies of them are left."""
    tiles = improving_tiles(names13, visible)
    return len(tiles), sum(left for _, left in tiles), shanten(counts(names13))


def seeking(names13, visible=()):
    """What a 13-tile hand is after, in words a person can check against the table."""
    away = shanten(counts(names13))
    tiles = improving_tiles(names13, visible)
    listed = ", ".join(f"{name} ({left} left)" for name, left in tiles)
    total = sum(left for _, left in tiles)
    if away > 0:
        return f"{away} from ready (shanten {away}). Draws that get closer: {listed}: {total} tiles left"
    if not tiles:
        return "Ready (shanten 0), but no tile can complete it: it holds every copy of what it needs"
    return f"Ready (shanten 0). Wins on: {listed}: {total} tiles left"


def is_win(names14):
    """A complete hand of 14. Never one with five of a tile: there are four, so a name is misread (face-down play unattended
    once read every noise frame of the sim as 2C, and 14 2Cs were a win)."""
    return len(names14) == HAND_SIZE + 1 and not too_many(names14) and shanten(counts(names14)) == -1


def choose_discard(hand, drawn, visible=()):
    """Which of the 14 tiles (the 13 `hand` names in slot order, then `drawn`) to discard. Ranked by the
    shanten left, then the improving tiles still left (copies), then the improving kinds, then the drawn
    tile first (one manipulation instead of two), then the lower slot. `visible`: tiles seen elsewhere
    (the discards so far). Returns (best, every candidate), each {index (13 = the drawn tile), tile_name,
    shanten_after, ukeire_kinds, ukeire_tiles}."""
    fourteen = list(hand) + [drawn]
    candidates, seen = [], {}
    for i, name in enumerate(fourteen):
        if name not in seen:            # the same name leaves the same hand
            kinds, tiles, after = ukeire(fourteen[:i] + fourteen[i + 1:], visible)
            seen[name] = {"shanten_after": after, "ukeire_kinds": kinds, "ukeire_tiles": tiles}
        candidates.append({"index": i, "tile_name": name, **seen[name]})
    candidates.sort(key=lambda c: (c["shanten_after"], -c["ukeire_tiles"], -c["ukeire_kinds"],
                                   c["index"] != HAND_SIZE, c["index"]))
    return candidates[0], candidates


def too_many(names):
    """Names seen more than 4 times (hand + drawn + discards): the robot has misread one of them."""
    return sorted(n for n, k in Counter(names).items() if k > COPIES)


def hand_mismatches(expected, seen):
    """Slots where the table doesn't match the hand: {slot: (expected name, what was seen or None)}.
    `expected` and `seen` map slot -> name; a slot missing from `seen` had no sure read there."""
    return {slot: (name, seen.get(slot)) for slot, name in expected.items() if seen.get(slot) != name}
