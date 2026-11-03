"""What's actually on the mat: a placement is one physical tile put down at one place, until a person or a knock changes
that. It's the unit tile names are scored on (mahjong/name_ledger.py): the name a person gave it once, against every read
the scans made of it. Without it, a scan's read can only be tied to a tile by position, and a knocked 3D sitting on 1B's
spot looks like 60 misreads of 1B (20261030_115706_bench).

Events: `placement` {id, spot, xy, truth, truth_source, read_at_start} and `placement_ended` {id, spot, why}. truth_source
is "person", "game" (what the robot believes; only a person's later word confirms it), "unconfirmed" or "sim". A knock
isn't written here: the ledger works that out from the scan reviews, since a person's verdict can overrule them later.
"""
from . import config, detector, layout


class Placements:
    """The placement at each spot right now, for one session."""

    def __init__(self, session):
        self.session, self.live, self.n = session, {}, 0

    def start(self, spot, xy, truth=None, truth_source="unconfirmed", read=None, **fields):
        """A tile now at `spot`, replacing any before it. Returns its id: unique across sessions, since a resumed hand
        audits discards from an earlier session."""
        self.end(spot, "replaced")
        self.n += 1
        placement = {"id": f"{self.session.dir.name[:15]}-P{self.n}", "spot": spot, "xy": list(xy), "truth": truth,
                     "truth_source": truth_source, "read_at_start": read, **fields}
        self.session.event("placement", **placement)
        self.live[spot] = placement
        return placement["id"]

    def end(self, spot, why):
        """The tile at `spot` left it. Returns its placement's id, or None if there wasn't one."""
        placement = self.live.pop(spot, None)
        if placement is None:
            return None
        self.session.event("placement_ended", id=placement["id"], spot=spot, why=why)
        return placement["id"]

    def id_at(self, spot):
        return self.live[spot]["id"] if spot in self.live else None

    def truth_at(self, spot):
        return self.live[spot]["truth"] if spot in self.live else None


def rows(spots, tiles):
    """One row per spot from a scan's tiles, for a person to check: the read nearest it, or UNNAMED."""
    out = []
    for spot in spots:
        tile = layout.nearest_tile(tiles, spot["xy"])
        out.append({"id": spot["id"], "xy": list(tile["xy"]) if tile else list(spot["xy"]),
                    "tile_name": tile["tile_name"] if tile else config.UNNAMED,
                    "detected_as": tile["tile_name"] if tile else None,
                    "confidence": tile["confidence"] if tile else 0.0})
    return out


def parse_answer(answer, tiles):
    """The names step: "" accept, "r" rescan, "q" quit, else edits like "3=2C -5" (1-based) or "S7=2B -S9" for rows with an
    "id". Returns ("accept"|"rescan"|"quit"|"edit", tiles after the edits) or ("error", message)."""
    answer = answer.strip()
    if answer.lower() in ("", "r", "q"):
        return {"": "accept", "r": "rescan", "q": "quit"}[answer.lower()], tiles
    tiles = [dict(t) for t in tiles]
    ids = {str(t["id"]).upper(): i for i, t in enumerate(tiles) if "id" in t}

    def row(key):
        return ids[key.upper()] if key.upper() in ids else int(key) - 1
    for part in answer.replace(",", " ").split():
        try:
            if part.startswith("-"):
                tiles[row(part[1:])]["skip"] = True
            else:
                key, name = part.split("=")
                name = name.strip().upper()
                if name not in detector.TILE_NAMES:
                    return "error", f"{name} isn't a tile name ({', '.join(detector.TILE_NAMES)})"
                tile = tiles[row(key)]
                tile["renamed_from"] = tile.get("renamed_from") or tile["detected_as"]
                tile["tile_name"] = name
        except (ValueError, IndexError):
            return "error", f"can't read {part!r}: use e.g. 3=2C to rename tile 3, -3 to leave it out"
    return "edit", tiles


def confirm(rows_, ask):
    """A person checks a scan's names, spot by spot. Returns the rows left in, with "confirmed": whether a person vouched for
    them, or None to scan again. q leaves every name they didn't correct unconfirmed."""
    while True:
        print(f"\nWhat's on the mat? {len(rows_)} spots, as the scan reads them:")
        for row in rows_:
            note = " (left out)" if row.get("skip") else f" (was {row['renamed_from'] or 'no name'})" \
                if "renamed_from" in row else ""
            print(f"  {row['id']:>4} {row['tile_name']:<7} {row['confidence']:.2f}{note}")
        action, result = parse_answer(ask("Check them against the table. Enter if all right, S7=2B to correct, -S7 for no "
                                          "tile there, r rescan, q leave them unchecked: "), rows_)
        if action == "error":
            print(result)
        elif action == "edit":
            rows_ = result
        elif action == "rescan":
            return None
        else:
            return [{**row, "confirmed": action == "accept" and row["tile_name"] != config.UNNAMED}
                    for row in rows_ if not row.get("skip")]
