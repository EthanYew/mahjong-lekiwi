"""The crude full run (docs/face_down_plan.md, test 8): draw from the wall, show the tile to the base camera, read it, a person checks, then
discard or keep. Its job is a list of where it breaks, so nothing here tries to be robust: a step that fails is written down and a person does it.
scripts/crude_run.py runs this on the arm.
"""
from . import game

STEPS = ("draw", "show", "set_on_zone", "read", "pick_from_hand", "discard", "pick_from_zone", "place_in_hand", "person")


def parse_hand(text, slots):
    """{slot id: name} from 13 comma-separated names, in the layout's order. SystemExit before any hardware if they can't be a hand."""
    names = [n.strip().upper() for n in text.split(",") if n.strip()]
    if len(names) != len(slots):
        raise SystemExit(f"--hand needs {len(slots)} names, one per slot of the layout; got {len(names)}")
    unknown = [n for n in names if not game.is_playable(n)]
    if unknown:
        raise SystemExit(f"--hand: {', '.join(unknown)} isn't a tile name (for example 5C, 3D, 9B, EW, RD)")
    extra = game.too_many(names)
    if extra:
        raise SystemExit(f"--hand has more than {game.COPIES} of {', '.join(extra)}")
    return dict(zip(slots, names))


def draw_plan(turns, stacked):
    """One draw per turn. Flat: every draw is the next tile, so the wall is fed before each but the first. Stacked: top, bottom, feed."""
    from . import wall_test
    if stacked:
        return [{"turn": d["draw"], "layer": d["layer"], "feed_before": d["feed_before"]} for d in wall_test.cycle_plan(turns)]
    return [{"turn": n + 1, "layer": "flat", "feed_before": n > 0} for n in range(turns)]


def _flat_picture(columns=4):
    return [["[G]", *["G"] * (columns - 1)]]


def flat_wall_step(station_id, tiles):
    return {"lines": [f"[crude run] Clear station {station_id}, then lay the wall, ONE tile high:", f"{tiles} tiles, GREEN side up, flat on the mat, end to end",
                      f"First tile on {station_id}, the rest running away from the robot"],
            "picture": _flat_picture()}


def flat_feed_step(station_id):
    return {"lines": [f"[crude run] Feed the wall at station {station_id}:", "Push the whole wall toward the robot", f"until the next tile sits on {station_id}"],
            "picture": _flat_picture()}


def hand_text(slots, hand):
    return ", ".join(f"{s} {hand[s]}" for s in slots)


def summary(turns_done, breaks, reads):
    """What the run found: turns finished, breaks per step, and the reader's score on the tiles a person named."""
    per_step = {s: sum(b["step"] == s for b in breaks) for s in STEPS if any(b["step"] == s for b in breaks)}
    scored = [r for r in reads if r["guess"] is not None]
    return {"turns_done": turns_done, "breaks": len(breaks), "breaks_per_step": per_step, "by_hand": sum(bool(b["by_hand"]) for b in breaks),
            "reads": len(reads), "reader_guesses": len(scored), "reader_right": sum(r["guess"] == r["name"] for r in scored),
            "faces_not_visible": sum(bool(r["face_hidden"]) for r in reads)}


def summary_lines(result, breaks):
    lines = [f"{result['turns_done']} turn(s) finished, {result['breaks']} break(s), {result['by_hand']} done by hand.",
             f"Reader: right on {result['reader_right']} of {result['reader_guesses']} guesses; a person named all {result['reads']}; face not visible on {result['faces_not_visible']}."]
    lines += ["Where it broke:"] + [f"  turn {b['turn']} {b['step']}: {b['what']}" + (" (done by hand)" if b["by_hand"] else "") for b in breaks] if breaks else ["Nothing broke."]
    return lines
