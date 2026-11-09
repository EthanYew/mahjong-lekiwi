"""Testing draws from a wall (docs/face_down_plan.md, "The wall: one station"): what's tried, in what order, and how it's scored.
The arm goes to one saved spot, a station; a person feeds the next stack onto it. scripts/wall_test.py runs this on the arm.

Nothing here can see a face-down tile, so "clean" is a person's word, given per attempt (OUTCOMES). The jaws' reading can't
replace it: two tiles lifted together are as wide as one.
"""
import math
import statistics
from pathlib import Path

from . import config
from .session import read_events

LAYOUT = "wall-stations"
PUSH_LIMIT_MM = 3.0         # half the jaws' ~6mm tolerance (a judgement, not a measurement): what a push may stray from where it usually lands
PUSH_RADIUS_M = 0.030       # the lead tile is the one read this close to the station
BASELINE_SCANS = 3          # scans of the untouched row before any push: the detector's own wander, which a push's spread can't be told from

CYCLE = "cycle"

# face: which way up the target lies. layer: how it's stacked, which sets the hover. behind: what touches its far end.
CASES = {
    "control": {"face": "up", "layer": "flat", "behind": None,
                "lay": "one tile FACE UP on {id}, nothing touching it"},
    "flat": {"face": "down", "layer": "flat", "behind": None,
             "lay": "one tile FACE DOWN on {id}, nothing touching it"},
    "strip": {"face": "down", "layer": "flat", "behind": "flat",
              "lay": "one tile FACE DOWN on {id}, and another FACE DOWN touching its far end"},
    "top": {"face": "down", "layer": "top", "behind": "stack",
            "lay": "a stack of two FACE DOWN on {id}, square and flush, and another stack of two touching its far end"},
    "bottom": {"face": "down", "layer": "bottom", "behind": "stack",
               "lay": "one tile FACE DOWN on {id} (the stack's top already gone), and a stack of two touching its far end"},
}
HOVER_RAISE_M = {"flat": 0.0, "bottom": 0.0, "top": config.TILE_THICKNESS_M}      # above the policy's own hover
LEVEL_M = HOVER_RAISE_M     # how far the target's own grip level sits above a flat tile's

OUTCOMES = {       # one per attempt: the first that applies, worst first
    "t": "a person touched it",
    "2": "lifted two tiles",
    "k": "knocked the top tile off",
    "n": "moved the next stack",
    "p": "pushed its own tile",
    "d": "dropped it on the way",
    "a": "closed on air",
    "c": "clean",
}
CLEAN = OUTCOMES["c"]


def parse_list(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def schedule(stations, cases, trials):
    """[(round, station id, case)]: every station and case once a round, the order reversed every other round (a b, b a),
    so neither a station nor a case always goes first as the session wears on."""
    combos = [(s, c) for s in stations for c in cases]
    return [(n + 1, s, c) for n in range(trials) for s, c in (combos if n % 2 == 0 else combos[::-1])]


def cycle_plan(draws):
    """The real operation, `draws` draws from a wall: top, bottom, then the next stack is fed onto the station."""
    return [{"draw": i + 1, "layer": "top" if i % 2 == 0 else "bottom", "feed_before": i > 0 and i % 2 == 0,
             "feed_index": i // 2, "draws_since_feed": i % 2} for i in range(draws)]


def case_of(case, layer=None):
    """A case's settings; a cycle's draw is a top or bottom with the next stack behind it."""
    return CASES[{"top": "top", "bottom": "bottom"}[layer]] if case == CYCLE else CASES[case]


def above_own_level_m(grasp_clearance_m, layer):
    """How far above the target's own grip level the jaws closed. grasp_clearance_m is above a flat tile's, so a top tile's
    close would read 20mm high against the 20mm line where misses start without this."""
    return None if grasp_clearance_m is None else grasp_clearance_m - LEVEL_M[layer]


def spread(points):
    """(mean xy, the furthest any point is from it, in mm) of [(x, y)] in metres."""
    mx, my = sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)
    return (mx, my), max(math.hypot(p[0] - mx, p[1] - my) for p in points) * 1000


def push_result(pushes, baseline, station_xy):
    """What a push check found: how far each push strayed from where pushes usually land, against the detector's own wander.
    The offset from the station is separate: a steady offset moves the station, a spread can't be corrected."""
    centre, worst = spread(pushes)
    wander = spread(baseline)[1] if baseline else None
    return {"pushes": len(pushes), "mean_xy": [float(v) for v in centre], "worst_mm": round(worst, 2),
            "baseline_mm": None if wander is None else round(wander, 2),
            "offset_from_station_mm": round(math.dist(centre, station_xy) * 1000, 1), "limit_mm": PUSH_LIMIT_MM,
            "passed": worst <= PUSH_LIMIT_MM}


def verdict(clean, tries, case):
    """The face-down plan's thresholds, decided before the runs, which are for 10 tries."""
    if case == "control":
        if tries < 5:
            return "too few tries to judge"
        return "control fine" if clean / tries >= 0.8 else "control under 80%: fix the setup before reading anything else"
    if tries != 10:
        return f"{tries} tries: the plan's thresholds are for 10"
    return "carry on" if clean >= 8 else "try a change, then decide" if clean >= 4 else "record demos"


def attempts(session_dir):
    """The answered attempts of one session (quit or voided ones have no outcome)."""
    return [e for e in read_events(Path(session_dir)) if e["type"] == "wall_attempt" and e.get("outcome")]


def summarize(rows):
    """One entry per station, case and layer: tries, clean, what went wrong, and the close heights above the target's own level."""
    groups = {}
    for row in rows:
        groups.setdefault((row["station"], row["case"], row["layer"]), []).append(row)
    out = []
    for (station, case, layer), group in sorted(groups.items()):
        closes = [r["above_own_level_m"] * 1000 for r in group if r.get("above_own_level_m") is not None]
        clean = sum(r["outcome"] == CLEAN for r in group)
        not_clean = {}
        for r in group:
            if r["outcome"] != CLEAN:
                not_clean[r["outcome"]] = not_clean.get(r["outcome"], 0) + 1
        out.append({"station": station, "case": case, "layer": layer, "tries": len(group), "clean": clean,
                    "held": sum(bool(r["held"]) for r in group), "not_clean": not_clean,
                    "close_median_mm": round(statistics.median(closes), 1) if closes else None,
                    "close_max_mm": round(max(closes), 1) if closes else None,
                    "verdict": verdict(clean, len(group), case)})
    return out


def table(groups):
    """The summary as lines of text."""
    lines = [f"{'station':8}{'case':9}{'layer':8}{'clean':>7}{'held':>6}{'close mm (median, max)':>25}  verdict"]
    for g in groups:
        close = "-" if g["close_median_mm"] is None else f"{g['close_median_mm']:.1f}, {g['close_max_mm']:.1f}"
        lines.append(f"{g['station']:8}{g['case']:9}{g['layer']:8}{g['clean']:>4}/{g['tries']:<2}{g['held']:>6}{close:>25}  {g['verdict']}")
        lines += [f"{'':25}{count} x {what}" for what, count in sorted(g["not_clean"].items())]
    return lines


def sessions(root=None):
    """Every real wall-test session, oldest first; a simulated one says nothing about the arm."""
    root = Path(root or config.SESSIONS)
    found = []
    for folder in sorted(p for p in root.glob("*_wall*") if p.is_dir()):
        start = next((e for e in read_events(folder) if e["type"] == "session_start"), {})
        if not start.get("sim"):
            found.append(folder)
    return found
