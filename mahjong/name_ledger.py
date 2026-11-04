"""How often each tile gets misread, and how badly it matters, worked out from the session folders: every read tied to a
placement (mahjong/placements.py), scored against what a person said was there.

Placements and sessions are the sample size, not scans: 60 scans of one tile aren't 60 tests. A placement ends up:
- accepted wrong: the votes (mahjong/identity.py) cleared their bar for a wrong name, so the robot would act without asking.
  In play that's what the robot did (`name_decided`); at the bench, the votes replayed over the placement's scans.
- asks: the votes didn't clear it, so a person would be asked.
- accepted right.

Where the truth comes from, strongest first: a person's verdict on the attempt's wrist photo (data/wrist_review.jsonl),
their answer during a hand (`name_asked`, `recount`, `silent_error`, `hand_audit`), then the `placement` event. Sessions
before placements existed have only the photos, so they're `backfilled`. What the robot merely believed ("game", or a
person who skipped the check) isn't truth, and a placement with no truth isn't scored.

A person's or a knock's change at a spot ends what's known there: a knocked 3D on 1B's spot isn't a misread of 1B
(20261030_115706_bench). What follows counts only once a reviewed photo says what's there.

Bench attempts and hovers without a spot (--tile trials, the menu) are grouped by where they were. wrist_capture.py sets count
too, as "read once": one scan and a person's name, which says how the read went but not what the votes would have done. Exam
sessions are never read here: nothing gets tuned on them.

An outcome is what the votes alone would do. `guarded` is the same placement with play's extra checks (identity.doubt):
the bands stay on the first, since the checks are a stopgap that hides how bad a tile is.
"""
import hashlib
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from . import config, detector, identity, interventions, layout, wrist_photos
from .offline_eval import surface as surface_of
from .session import read_events

SCORED_SCANS = ("choose", "after_return", "name_check", "register", "drawn", "final")   # not after_pickup: the tile's held
ENDS_A_PLACEMENT = {"knocked", "moved_elsewhere", "tile_reset", "straightened"}
RIGHT, ASKS, WRONG, READ_ONCE = "accepted right", "asks", "accepted wrong", "read once"
# Bands. First guesses, to be set from data like identity.py's bars
COSTLY_ASK_SHARE = 0.25     # asks on this share of placements
FLAKY_BAD_SHARE = 0.10      # votes right, but this share of scans wrong or unnamed
ENOUGH_PLACEMENTS, ENOUGH_SESSIONS = 5, 3   # below these, "fine" would only mean untested
BANDS = ("severe", "costly", "flaky", "fine", "not enough data")
FLOWERS = ("1F", "2F", "3F", "4F")          # not played


@lru_cache(maxsize=1)
def _weights_history():
    """[(commit time, weights id)] of the live detector's file in git, oldest first; [] without git."""
    path = config.TILE_DETECTOR.relative_to(config.REPO)
    try:
        log = subprocess.run(["git", "-C", str(config.REPO), "log", "--format=%ct %H", "--", str(path)],
                             capture_output=True, text=True, timeout=10).stdout.split("\n")
        out = []
        for time, commit in (line.split() for line in log if line.strip()):
            blob = subprocess.run(["git", "-C", str(config.REPO), "show", f"{commit}:{path}"], capture_output=True,
                                  timeout=60).stdout
            out.append((int(time), f"{path.stem}@{hashlib.sha256(blob).hexdigest()[:8]}"))
        return sorted(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def weights_of(session_name, start):
    """The detector a session ran: logged since this was added; before that, what git had when the session began (a
    good guess: a checkout could lag main)."""
    if start.get("detector"):
        return start["detector"]
    from datetime import datetime
    try:
        began = datetime.strptime(session_name[:15], "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return "unknown"
    history = _weights_history()
    return next((w for time, w in reversed(history) if time <= began), history[0][1] if history else "unknown")


def _points(diff):
    """Where a scene diff's tiles left or arrived. A tile that only turned is still the same tile, in the same place."""
    return ([m["from"] for m in diff.get("moved", [])] + [m["to"] for m in diff.get("moved", [])]
            + [t["xy"] for t in diff.get("gone", [])] + [t["xy"] for t in diff.get("appeared", [])])


def changes_by_scan(events):
    """(ends, glitches): [(event index of the scan, points)] of real changes on the mat, and of changes that turned out to
    be the detector's. Reviews are often logged long after their scan, so they're placed by scan number; older ones don't
    say what changed, so the scans are compared again, leaving out the tile the arm was handling."""
    scans = [(i, e) for i, e in enumerate(events) if e["type"] == "detections"]
    index_of = {e["scan"]: k for k, (_, e) in enumerate(scans) if "scan" in e}
    ends, glitches = [], []
    for scan, reviews in interventions.scan_reviews(events).items():
        if scan not in index_of:
            continue
        k = index_of[scan]
        i, event = scans[k]
        for review in reviews:
            kinds = set(review.get("kinds") or [])
            if kinds & ENDS_A_PLACEMENT:
                points = _points(review.get("changes") or {})
                if not points and k:
                    target = event.get("target_xy")
                    points = _points(interventions.scene_diff(scans[k - 1][1]["tiles"], event["tiles"],
                                                              ignore=[target] if target else ()))
                ends.append((i, points))
            elif "scan_glitch" in kinds:
                glitches.append((i, _points(review.get("changes") or {})))
    return sorted(ends, key=lambda c: c[0]), glitches


def _near(points, xy):
    return any(math.dist(p, xy) <= config.TILE_MERGE_RADIUS_M for p in points)


def segments(events):
    """Stretches of a session in which one tile sat at a spot: [{"spot", "xy", "lo", "hi" (event indexes), "placement" (its
    event, or None), "attempts"}]. A placement event, its end, or a change at the spot closes one."""
    marks, ends = defaultdict(list), changes_by_scan(events)[0]
    home = {}

    def spot_of(e):
        """Its layout spot, or, for --tile trials and the menu, a place named by where the tile first was."""
        if e.get("spot"):
            return e["spot"]
        xy = e["detection_xy"]
        near = [s for s in home if s.startswith("@") and math.dist(home[s], xy) <= config.TILE_MERGE_RADIUS_M]
        return near[0] if near else f"@{xy[0]:.3f},{xy[1]:+.3f}"
    for i, e in enumerate(events):
        if e["type"] == "placement":
            marks[e["spot"]].append((i, "placement", e))
            home[e["spot"]] = e["xy"]
        elif e["type"] == "placement_ended":
            marks[e["spot"]].append((i, "ended", e))
        elif (e["type"] == "attempt" or (e["type"] == "hover" and e.get("attempt_dir"))) and e.get("detection_xy"):
            spot = spot_of(e)
            marks[spot].append((i, "attempt", e))
            home.setdefault(spot, e.get("spot_xy") or e["detection_xy"])
    out = []
    for spot, spot_marks in marks.items():
        cuts = [(i, "change", None) for i, points in ends if _near(points, home[spot])]
        current = {"spot": spot, "xy": home[spot], "lo": 0, "placement": None, "attempts": []}
        for i, kind, e in sorted(spot_marks + cuts, key=lambda m: m[0]):
            if kind == "attempt":
                current["attempts"].append(e)
                continue
            out.append({**current, "hi": i})
            current = {"spot": spot, "xy": e["xy"] if kind == "placement" else home[spot], "lo": i,
                       "placement": e if kind == "placement" else None, "attempts": []}
        out.append({**current, "hi": len(events)})
    return [s for s in out if s["attempts"] or s["placement"]]


def _truth(segment, session_name, reviewed, answers, audited, by_id):
    """(the name, where it came from, whether a review overruled the placement), (None, why not, False) if unscorable."""
    photos = {reviewed[key] for key in (f"{session_name}/{a.get('attempt_dir')}/wrist_0000.jpg" for a in segment["attempts"])
              if key in reviewed}
    if len(photos) > 1:
        return None, "conflicted", False
    placement = segment["placement"]
    stated, source, seen = None, None, set()
    while placement and stated is None and placement["id"] not in seen:     # a tile moved keeps what was said of it
        seen.add(placement["id"])
        pid = placement["id"]
        if pid in answers:
            stated, source = answers[pid], "person"
        elif pid in audited:
            stated, source = audited[pid], "person"
        elif placement.get("truth") and placement.get("truth_source") in ("person", "sim"):
            stated, source = placement["truth"], placement["truth_source"]
        else:
            placement = by_id.get(placement.get("from_placement"))
    if photos:
        name = next(iter(photos))
        return name, "review", stated is not None and stated != name
    return (stated, source, False) if stated else (None, "no truth", False)


def _audited(sessions):
    """{placement id: the name a person gave it, at the hand check before the first draw or at a hand's end (which wins)}, over
    all sessions: a resumed hand audits earlier ones."""
    out = {}
    for _, events in sessions:
        for e in events:
            if e["type"] in ("hand_confirmed", "hand_audit"):
                out.update(e["confirmed"])
    return out


def wrist_placements(meta, events, reviewed):
    """A wrist_capture.py session's tiles: what the one scan read, against the name a person gave it (a review overrules)."""
    views = defaultdict(list)
    for e in events:
        if e["type"] == "wrist_view":
            views[(e.get("set"), e.get("tile_n"))].append(e)
    scored, skipped = [], Counter()
    for (_, tile_n), group in views.items():
        first = group[0]
        photos = {reviewed[key] for key in (f"{meta['session']}/{e['files'][0]}" for e in group if e.get("files"))
                  if key in reviewed}
        if len(photos) > 1:
            skipped["conflicted"] += 1
            continue
        truth = next(iter(photos)) if photos else first["tile"]
        read, right = first.get("detected_as"), first.get("detected_as") == truth
        scored.append({**meta, "spot": f"T{tile_n}", "tile": truth, "truth_source": "review" if photos else "person",
                       "overruled": bool(photos) and truth != first["tile"], "backfilled": False, "scans": 1,
                       "right": int(right), "wrong": int(not right), "unnamed": 0, "faint": 0,
                       "mixups": Counter({read: 1}) if not right else Counter(),
                       "wrong_confidence": [first.get("confidence") or 0.0] if not right else [], "glitches": 0,
                       "outcome": READ_ONCE, "guarded": READ_ONCE, "decided_live": False})
    return scored, skipped


def placements_of(session_name, events, reviewed, audited=None):
    """Scored placements of one session, and a Counter of why the rest weren't."""
    start = next((e for e in events if e["type"] == "session_start"), {})
    conditions = start.get("conditions") or {}
    meta = {"session": session_name, "surface": surface_of(conditions), "lighting": detector.lighting(conditions),
            "weights": weights_of(session_name, start)}
    if session_name.endswith("_wrist"):
        return wrist_placements(meta, events, reviewed)
    by_id = {e["id"]: e for e in events if e["type"] == "placement"}
    answers = {}
    for e in events:
        if e["type"] == "intervention" and e.get("placement"):
            if e.get("kind") in ("name_asked", "recount"):
                answers[e["placement"]] = e["answer"]
            elif e.get("kind") == "silent_error":
                answers[e["placement"]] = e["table"]
    decided = {}
    for e in events:
        if e["type"] == "name_decided":
            decided.setdefault(e["placement"], e)       # the first decision: what the robot went on
    glitches = changes_by_scan(events)[1]
    scored, skipped = [], Counter()
    for segment in segments(events):
        truth, source, overruled = _truth(segment, session_name, reviewed, answers, audited or {}, by_id)
        if truth is None:
            skipped[source] += 1
            continue
        row = {**meta, "spot": segment["spot"], "tile": truth, "truth_source": source, "overruled": overruled,
               "backfilled": segment["placement"] is None, "scans": 0, "right": 0, "wrong": 0, "unnamed": 0, "faint": 0,
               "mixups": Counter(), "wrong_confidence": [], "glitches": 0}
        votes = identity.Votes()
        for i in range(segment["lo"], segment["hi"]):
            e = events[i]
            if e["type"] != "detections" or e.get("label") not in SCORED_SCANS:
                continue
            row["scans"] += 1
            tile = layout.nearest_tile(e["tiles"], segment["xy"])
            if tile is None:
                row["unnamed"] += 1
                row["faint"] += any(math.dist(f["xy"], segment["xy"]) <= config.TILE_MERGE_RADIUS_M
                                    for f in e.get("faint") or [])
                continue
            votes.add(segment["spot"], tile)
            if tile["tile_name"] == truth:
                row["right"] += 1
            else:
                row["wrong"] += 1
                row["mixups"][tile["tile_name"]] += 1
                row["wrong_confidence"].append(tile["confidence"])
        placement_id = segment["placement"]["id"] if segment["placement"] else None
        decision = decided.get(placement_id)
        if decision is None and not row["scans"]:
            skipped["no scans"] += 1
            continue
        # "wrist": the votes and the wrist camera agreed, so the robot went on it unasked, and a wrong one counts as accepted
        said = (decision["name"] if decision["how"] != "person" else None) if decision else votes.sure(segment["spot"])
        row["outcome"] = ASKS if said is None else RIGHT if said == truth else WRONG
        guarded = said if decision or said is None or votes.doubt(segment["spot"], said) is None else None
        row["guarded"] = ASKS if guarded is None else RIGHT if guarded == truth else WRONG
        row["decided_live"] = decision is not None
        row["glitches"] = sum(segment["lo"] <= i < segment["hi"] and _near(points, segment["xy"]) for i, points in glitches)
        scored.append(row)
    return scored, skipped


def reviewed_names(review=None):
    """{photo key: its name} for the wrist photos a person has reviewed and could name."""
    review = wrist_photos.load_review() if review is None else review
    out = {}
    for key, verdict in review.items():
        if verdict["verdict"] == "ok":
            parts = key.split("/")      # <session>/<n>_<time>_<NAME>/wrist_0000.jpg or <session>/wrist/<NAME>/<view>.jpg
            if len(parts) == 3 and parts[2] == "wrist_0000.jpg":
                out[key] = parts[1].rsplit("_", 1)[-1]
            elif len(parts) == 4 and parts[1] == "wrist":
                out[key] = parts[2]
        elif verdict.get("name"):
            out[key] = verdict["name"]
    return {key: name for key, name in out.items() if name in detector.TILE_NAMES}     # not an old S9 folder, or "unnamed"


def load(sessions_dir=None, review=None, include_sim=False, since=None):
    """(scored placements, Counter of why the others weren't, sessions read) over every bench, play and wrist session."""
    sessions_dir = Path(sessions_dir or config.SESSIONS)
    sessions, exam = [], detector.load_exam()["sessions"]
    for folder in sorted(p for pattern in ("*_bench", "*_play", "*_wrist") for p in sessions_dir.glob(pattern)):
        if (since and folder.name < since) or folder.name in exam:
            continue
        if not folder.is_dir() or (detector.is_sim(folder) and not include_sim):
            continue
        sessions.append((folder.name, read_events(folder)))
    reviewed, audited = reviewed_names(review), _audited(sessions)
    scored, skipped = [], Counter()
    for name, events in sessions:
        rows, why = placements_of(name, events, reviewed, audited)
        scored += rows
        skipped += why
    return sorted(scored, key=lambda r: (r["session"], r["spot"])), skipped, len(sessions)


def band(row):
    if row["accepted_wrong"]:
        return "severe"
    if row.get("voted", row["placements"]) and row["asks"] / row.get("voted", row["placements"]) >= COSTLY_ASK_SHARE:
        return "costly"
    if row["wrong_share"] + row["unnamed_share"] >= FLAKY_BAD_SHARE:
        return "flaky"
    return "fine" if row["placements"] >= ENOUGH_PLACEMENTS and row["sessions"] >= ENOUGH_SESSIONS else "not enough data"


def rows(placements, by=()):
    """One row per tile name + detector + surface (+ each field of `by`), worst first. Shares are averaged per placement, so
    a tile sitting through 100 scans counts as much as one through 10."""
    groups = defaultdict(list)
    for p in placements:
        groups[(p["tile"], p["weights"], p["surface"], *(str(p[k]) for k in by))].append(p)
    out = []
    for (tile, weights, surface, *extra), group in groups.items():
        def share(field):
            return sum(p[field] / p["scans"] for p in group if p["scans"]) / max(1, sum(bool(p["scans"]) for p in group))
        mixups = sum((p["mixups"] for p in group), Counter())
        confidences = [c for p in group for c in p["wrong_confidence"]]
        outcomes = Counter(p["outcome"] for p in group)
        voted = len(group) - outcomes[READ_ONCE]
        row = {"tile": tile, "weights": weights, "surface": surface, "extra": extra, "placements": len(group),
               "sessions": len({p["session"] for p in group}), "scans": sum(p["scans"] for p in group),
               "accepted_wrong": outcomes[WRONG], "asks": outcomes[ASKS], "accepted_right": outcomes[RIGHT],
               "voted": voted, "read_once": outcomes[READ_ONCE],
               "read_once_wrong": sum(p["outcome"] == READ_ONCE and p["wrong"] > 0 for p in group),
               "accepted_wrong_bound": detector.wrong_rate_bound(outcomes[WRONG], voted) if voted else None,
               "right_share": share("right"), "wrong_share": share("wrong"), "unnamed_share": share("unnamed"),
               "mixups": mixups.most_common(), "wrong_confidence": statistics.median(confidences) if confidences else None,
               "overruled": sum(p["overruled"] for p in group), "glitches": sum(p["glitches"] for p in group),
               "backfilled": sum(p["backfilled"] for p in group)}
        row["band"] = band(row)
        out.append(row)
    return sorted(out, key=lambda r: (BANDS.index(r["band"]), -r["accepted_wrong"], -r["asks"] / r["placements"],
                                      -(r["wrong_share"] + r["unnamed_share"]), r["tile"]))


def untested(all_rows, weights, surface):
    """Playable names with no scored placement on this detector and surface: not good, just not known."""
    seen = {r["tile"] for r in all_rows if (r["weights"], r["surface"]) == (weights, surface)}
    return [name for name in detector.TILE_NAMES if name not in seen and name not in FLOWERS]


def table(all_rows):
    """The rows as lines of text."""
    lines = [f"  {'tile':<4} {'band':<15} {'plc':>3} {'sess':>4} {'scans':>5} {'wrong acc':>9} {'(up to)':>7} {'asks':>4} "
             f"{'once':>5} {'right':>5} {'wrong':>5} {'unnamed':>7} {'conf wrong':>10}  read as"]
    for r in all_rows:
        extra = f" [{' '.join(r['extra'])}]" if r["extra"] else ""
        conf = "-" if r["wrong_confidence"] is None else f"{r['wrong_confidence']:.2f}"
        lines.append(f"  {r['tile']:<4} {r['band']:<15} {r['placements']:>3} {r['sessions']:>4} {r['scans']:>5} "
                     f"{r['accepted_wrong']:>9} {'-' if r['accepted_wrong_bound'] is None else format(r['accepted_wrong_bound'], '.0%'):>7} "
                     f"{r['asks']:>4} {str(r['read_once_wrong']) + '/' + str(r['read_once']) if r['read_once'] else '-':>5} {r['right_share']:>5.0%} "
                     f"{r['wrong_share']:>5.0%} {r['unnamed_share']:>7.0%} {conf:>10}  "
                     + (" ".join(f"{name} x{n}" for name, n in r["mixups"][:3]) or "-") + extra)
    return lines


def session_lines(folder):
    """The session's tile names, for session_report.py: anything not fine, with what's still unscored."""
    placements, skipped = placements_of(folder.name, read_events(folder), reviewed_names())
    if not placements and not skipped:
        return []
    outcomes = Counter(p["outcome"] for p in placements)
    lines = [f"tile names: {len(placements)} placements scored: {outcomes[RIGHT]} accepted right, {outcomes[ASKS]} ask, "
             f"{outcomes[WRONG]} accepted wrong" + ("; not scored: " + ", ".join(f"{why} {n}" for why, n in skipped.items())
                                                    if skipped else "")]
    bad = [r for r in rows(placements) if r["band"] in ("severe", "costly", "flaky")]
    return lines + (table(bad) if bad else [])


def worst_line(sessions_dir=None, surface=None, limit=5):
    """One line for status.py: the worst tiles on the latest detector, on `surface` (default: the standard setup's)."""
    import json
    surface = surface or json.loads((config.DATA / "standard_setup.json").read_text())["surface"]
    placements = load(sessions_dir)[0]
    if not placements:
        return "Tile names: no scored placements yet"
    weights = placements[-1]["weights"]
    on_it = rows([p for p in placements if p["surface"] == surface and p["weights"] == weights])
    bad = [r for r in on_it if r["band"] in ("severe", "costly", "flaky")]
    return (f"Tile names on {surface} ({sum(r['placements'] for r in on_it)} placements, {weights}): "
            f"{sum(r['accepted_wrong'] for r in on_it)} accepted wrong; worst "
            + (", ".join(f"{r['tile']} ({r['band']})" for r in bad[:limit]) or "none") + "; python scripts/name_report.py")
