"""Human interventions: what a person had to step in and do, as data, so "rescues per 10 attempts" can
go on a scorecard (ROADMAP 3) instead of being remembered.

Classes, because only one of them measures the policy:
  rescue       unplanned: something went wrong and a person fixed it (Esc, a hand on the tile, a crash)
  reset        part of the protocol: tiles put back after a discard, the mat rearranged
  override     a person's verdict differs from the automatic one (the success prompt): the check's error
  failure      not a step-in, but something went wrong (a knocked neighbour, the arm bumping a tile)
  observation  listed, not counted (a mark that only says "look at this", a cancelled choice)

Most are derived (derive()) from what a session logs anyway: attempts, the `detections` scans around
each attempt, session_end. A person is asked only after marking an attempt (→ during ACT), aborting it
(Esc), answering a discard went wide, with the bench's `note` command, or when a scan shows the mat
changed (the bench asks then, so a detection glitch isn't logged as a knock: `scan_review`).
"""
import math
from collections import Counter

from . import config, kinematics, session

RESCUE, RESET, OVERRIDE, FAILURE, OBSERVATION = "rescue", "reset", "override", "failure", "observation"

# kind: (what a person is shown, class)
KINDS = {
    # asked after a mark (→ during ACT)
    "touched_tile": ("touched the tile to help", RESCUE),
    # not a step-in: the attempt can still be a clean success, but the collision is counted
    "collided": ("the arm bumped the tile or a neighbour, no one helped", FAILURE),
    "helped_arm": ("held or steadied the arm or base", RESCUE),
    "caught_drop": ("caught a drop", RESCUE),
    "look": ("nothing touched, just look at this", OBSERVATION),
    "other": ("other", RESCUE),
    "unspecified": ("marked, not explained", RESCUE),       # you pressed it for a reason
    # the bench's note command
    "straightened": ("straightened knocked tiles", RESCUE),
    "tile_reset": ("put tiles back / rearranged the mat", RESET),
    "discard_missed": ("picked up a discard that missed the container", RESCUE),
    "note": ("just a note", OBSERVATION),
    # logged or derived
    "aborted": ("stopped the policy (Esc)", RESCUE),
    "interrupted": ("Ctrl-C during a command", RESCUE),
    "crashed": ("the session ended on an error", RESCUE),
    "killed": ("no session_end: killed, or still running", RESCUE),
    "rescan": ("tile not found or unreachable", RESCUE),
    "cancelled": ("tile choice cancelled", OBSERVATION),
    "spot_empty": ("no tile at a layout spot; the walk skipped it", OBSERVATION),
    "hover_rejected": ("rejected the hover (←)", RESCUE),
    "takeover": ("took the gamepad from the policy (DAgger)", RESCUE),
    "success_overridden": ("answered against the gripper's verdict", OVERRIDE),
    "knocked": ("a tile other than the target moved", FAILURE),
    "scan_glitch": ("the scans changed but nothing moved (a detection glitch)", OBSERVATION),
    "arm_moved_while_off": ("arm not where it settled at the last disconnect (moved while off)", OBSERVATION),
}
MARK_KINDS = ["touched_tile", "collided", "helped_arm", "caught_drop", "look", "other"]
NOTE_KINDS = ["straightened", "tile_reset", "helped_arm", "discard_missed", "other", "note"]
ABORT_REASONS = {"frozen": "frozen at the hover", "wrong_place": "heading for the wrong place",
                 "knocking": "about to knock a neighbour", "pressing": "pressing into the table",
                 "bad_grip": "bad grip", "other": "other"}

# A tile counts as moved beyond this. 2026-10-23: untouched tiles' detections moved 1-1.5mm scan to scan,
# tiles put back by hand 3.2mm at most; the 7B pickup nudged 8D (rotated, visible in overview_after)
# and its detection moved 6.4mm, from (0.207, +0.141) to (0.211, +0.136).
KNOCK_M = 0.005
# A tile appearing or disappearing only counts at this confidence: the one detection that flickered
# between scans on 2026-10-23 read 0.52-0.56; every tile seen in every scan read 0.71 or more.
CHANGE_CONFIDENCE = 0.6
# A tile turned in place (its centre within KNOCK_M) shows as its detection box changing shape. 2026-10-23:
# 8D knocked ~20 deg by a return moved 1.4mm but its box grew +6.1 x +3.3 px; untouched tiles' boxes
# changed 0.7 px at most between scans (two sessions, 40+ tile pairs).
TURN_PX = 2.5


def class_of(kind):
    return KINDS.get(kind, (kind, RESCUE))[1]


def ask_kind(prompt, kinds, labels=None, ask=None, default=None):
    """One numbered menu line; returns the chosen kind, or `default` on Enter."""
    labels = labels or {k: KINDS[k][0] for k in kinds}
    shown = [labels[k] for k in kinds]
    choice = session.ask_one({"ask": prompt, "choices": shown}, labels.get(default), ask or input)
    return None if choice is None else kinds[shown.index(choice)]


def ask_mark(what, ask=None):
    """(mark kind, note) for a marked attempt: Enter -> "unspecified"; "other" asks for a line."""
    ask = ask or input
    kind = ask_kind(f"{what}: what happened", MARK_KINDS, ask=ask) or "unspecified"
    note = ask("  in a few words: ").strip() if kind == "other" else ""
    return kind, note


def ask_abort(what, ask=None):
    return ask_kind(f"{what}: why stop it", list(ABORT_REASONS), labels=ABORT_REASONS, ask=ask)


SCAN_ANSWERS = {"knocked": {"knocked": "yes, a tile moved", "scan_glitch": "no, nothing moved (a detection glitch)"},
                "between": {"straightened": KINDS["straightened"][0], "tile_reset": KINDS["tile_reset"][0],
                            "both": "both", "scan_glitch": "nothing moved (a detection glitch)"}}


def ask_scan(scan, proposal, ask=None):
    """The kinds that stand for a scan's proposed change (MatWatch.propose), from a person who was
    there: Enter accepts the proposal."""
    kind, diff = proposal
    where = scan.get("attempt_dir") or "the mat"
    if kind == "knocked":
        prompt = f"{where}: the scans say {changes_text(diff)} during the {scan['label'].removeprefix('after_')}. Right"
        answers = SCAN_ANSWERS["knocked"]
    else:
        prompt = f"{where}, since the last scan: {changes_text(diff)}. What was it"
        answers = SCAN_ANSWERS["between"]
    answer = ask_kind(prompt, list(answers), labels=answers, ask=ask, default=kind)
    return ["straightened", "tile_reset"] if answer == "both" else [answer]


# --- what changed on the mat between two scans ---

def on_mat(tile):
    """Within the arm's reach. Detections beyond it are phantoms or tiles off the workspace
    (2026-10-23: an 8D at (0.478, +0.232) came and went)."""
    x, y = tile["xy"]
    return kinematics.hover_command(x, y, config.HOVER_OFFSET_M, config.HOVER_GRIPPER)[0] is not None


def scene_diff(before, after, ignore=(), radius_m=config.TILE_MERGE_RADIUS_M, moved_m=KNOCK_M, turn_px=TURN_PX):
    """Tiles that moved, turned in place, appeared or went between two scans (lists of {tile_name, xy,
    confidence, box_px (older sessions: none)}). Matched by position, nearest first, not by label
    (labels flicker between scans). Changes within radius_m of an `ignore` point (the tile the arm was
    handling) are left out."""
    before, after = [t for t in before if on_mat(t)], [t for t in after if on_mat(t)]
    pairs = sorted((math.dist(b["xy"], a["xy"]), i, j) for i, b in enumerate(before) for j, a in enumerate(after))
    matched_before, matched_after, moved, turned = set(), set(), [], []
    for d, i, j in pairs:
        if d > radius_m:
            break
        if i in matched_before or j in matched_after:
            continue
        matched_before.add(i)
        matched_after.add(j)
        if d > moved_m:
            moved.append({"tile_name": after[j]["tile_name"], "from": list(before[i]["xy"]),
                          "to": list(after[j]["xy"]), "moved_m": d})
        elif before[i].get("box_px") and after[j].get("box_px"):
            change = max(abs(a - b) for a, b in zip(after[j]["box_px"], before[i]["box_px"]))
            if change > turn_px:
                turned.append({"tile_name": after[j]["tile_name"], "xy": list(after[j]["xy"]), "box_change_px": change})

    def near(xy):
        return any(math.dist(xy, p) <= radius_m for p in ignore)

    def brief(t):
        return {"tile_name": t["tile_name"], "xy": list(t["xy"])}
    return {"moved": [m for m in moved if not (near(m["from"]) or near(m["to"]))],
            "turned": [t for t in turned if not near(t["xy"])],
            "gone": [brief(t) for i, t in enumerate(before)
                     if i not in matched_before and t["confidence"] >= CHANGE_CONFIDENCE and not near(t["xy"])],
            "appeared": [brief(t) for j, t in enumerate(after)
                         if j not in matched_after and t["confidence"] >= CHANGE_CONFIDENCE and not near(t["xy"])]}


def _changed(diff):
    return any(diff.values())


def changes_text(diff):
    """"9D moved 12mm; 8D turned; 4D appeared" """
    return "; ".join([f"{m['tile_name']} moved {m['moved_m'] * 1000:.0f}mm" for m in diff["moved"]]
                     + [f"{t['tile_name']} turned" for t in diff.get("turned", [])]
                     + [f"{t['tile_name']} appeared" for t in diff["appeared"]]
                     + [f"{t['tile_name']} gone" for t in diff["gone"]])


def _from_points(diff):
    return ([m["from"] for m in diff["moved"]] + [t["xy"] for t in diff.get("turned", [])]
            + [t["xy"] for t in diff["gone"]])


def _to_points(diff):
    return ([m["to"] for m in diff["moved"]] + [t["xy"] for t in diff.get("turned", [])]
            + [t["xy"] for t in diff["appeared"]])


# --- the session's interventions ---

ATTEMPT_SCANS = ("after_pickup", "after_return")      # the arm handled tiles on the mat since the last scan


class MatWatch:
    """What the scans say changed on the mat, scan by scan. derive() replays a session through it; the
    bench feeds it live, so a person can confirm or correct a change (ask_scan) while it's in front of them."""
    def __init__(self):
        self.previous, self.knocked_to = None, []

    def propose(self, scan):
        """(kind, diff) the scans suggest for this `detections` event against the previous one, or None.
        knocked: the arm handled tiles since the previous scan (the tile it handled, at target_xy, is
        left out); otherwise nothing on the arm touched the mat, so a person did: straightened if it
        moved a tile a knock had left, else tile_reset."""
        if self.previous is None:
            return None
        if scan["label"] in ATTEMPT_SCANS:
            ignore = [scan["target_xy"]] if scan.get("target_xy") else []
            diff = scene_diff(self.previous["tiles"], scan["tiles"], ignore)
            return ("knocked", diff) if _changed(diff) else None
        diff = scene_diff(self.previous["tiles"], scan["tiles"])
        if not _changed(diff):
            return None
        fixed = any(math.dist(p, k) <= config.TILE_MERGE_RADIUS_M for p in _from_points(diff) for k in self.knocked_to)
        return ("straightened" if fixed else "tile_reset", diff)

    def advance(self, scan, proposal, kinds):
        """Move on to `scan`, with the kinds that stand for its proposal."""
        if proposal and "knocked" in kinds:
            self.knocked_to += _to_points(proposal[1]) + [t["xy"] for t in proposal[1]["gone"]]
        self.previous = scan


def derive(events):
    """Every intervention in a session's events, logged or derived, in order:
    [{kind, class, attempt_dir, source, time, ...}]. source: logged (an intervention event or a logged
    fact), mark (asked after → during ACT), derived (from attempt fields / session_end), scans (from
    consecutive `detections` events), confirmed (scans a person answered for: `scan_review`). Each is
    attributed to the attempt it belongs to, or the most recent one; session-level ones to None."""
    out = []

    def add(kind, attempt_dir, source, time, **fields):
        out.append({"kind": kind, "class": class_of(kind), "attempt_dir": attempt_dir, "source": source,
                    "time": time, **{k: v for k, v in fields.items() if v not in (None, "")}})

    reviews = {e["attempt_dir"]: e for e in events if e["type"] == "attempt_review"}
    scan_reviews = {e["scan"]: e["kinds"] for e in events if e["type"] == "scan_review"}
    last_attempt, mat, ended = None, MatWatch(), False
    for e in events:
        kind, time = e["type"], e.get("time")
        if kind == "attempt":
            a = {**e, **{k: v for k, v in reviews.get(e["attempt_dir"], {}).items()
                         if k in ("abort_reason", "mark_kind", "mark_note")}}
            last_attempt = a["attempt_dir"]
            if a.get("stop_reason") == "aborted":
                add("aborted", last_attempt, "logged", time, reason=a.get("abort_reason"))
            if a.get("marks_s"):
                add(a.get("mark_kind") or "unspecified", last_attempt, "mark", time, marks_s=a["marks_s"],
                    note=a.get("mark_note"))
            if "gripper_says_held" in a and a["gripper_says_held"] != a["success"]:
                add("success_overridden", last_attempt, "derived", time, gripper_says_held=a["gripper_says_held"],
                    success=a["success"])
        elif kind == "intervention":
            fields = {k: v for k, v in e.items() if k not in ("type", "time", "kind", "attempt_dir")}
            add(e["kind"], e.get("attempt_dir") or last_attempt, "logged", time, **fields)
        elif kind == "attempt_skipped":
            skip = {"cancelled": "cancelled", "not_at_spot": "spot_empty"}.get(e.get("reason"), "rescan")
            add(skip, last_attempt, "logged", time, reason=e.get("reason"), tile=e.get("tile"),
                **({"spot": e["spot"]} if e.get("spot") else {}))
        elif kind == "connect" and e.get("arm_at_rest") is False:
            add("arm_moved_while_off", None, "logged", time, off_rest_deg=e.get("off_rest_deg"),
                off_parked_deg=e.get("off_parked_deg"))
        elif kind == "detections":
            proposal = mat.propose(e)
            confirmed = proposal is not None and e.get("scan") in scan_reviews
            kinds = scan_reviews[e["scan"]] if confirmed else [proposal[0]] if proposal else []
            during = e["label"].removeprefix("after_") if e["label"] in ATTEMPT_SCANS else None
            for k in kinds:
                add(k, (e.get("attempt_dir") if during else None) or last_attempt,
                    "confirmed" if confirmed else "scans", time, during=during, changes=proposal[1])
            mat.advance(e, proposal, kinds)
        elif kind == "session_end":
            ended = True
            reason = e.get("reason", "quit")
            if reason != "quit":
                add("interrupted" if reason == "KeyboardInterrupt" else "crashed", last_attempt, "logged", time,
                    reason=reason, error=e.get("error"))
    if events and not ended:
        add("killed", last_attempt, "derived", events[-1].get("time"))
    # a person who noted what the scans also saw (tiles put back, straightened) did it once
    noted = {(i["kind"], i["attempt_dir"]) for i in out if i["source"] == "logged"}
    return [i for i in out if not (i["source"] in ("scans", "confirmed") and (i["kind"], i["attempt_dir"]) in noted)]


def summary(interventions, attempts):
    """Counts per class, rescues per 10 attempts, attempts per rescue, and clean vs assisted successes
    (assisted: a rescue-class mark on the attempt, so it wasn't autonomous). `attempts`: the attempt
    events (or any per-trial events; ones without `success` count as no success)."""
    n = len(attempts)
    counts = Counter(i["class"] for i in interventions)
    rescues = counts[RESCUE]
    assisted_dirs = {i["attempt_dir"] for i in interventions if i["source"] == "mark" and i["class"] == RESCUE}
    successes = [a for a in attempts if a.get("success")]
    assisted = sum(a.get("attempt_dir") in assisted_dirs for a in successes)
    return {"attempts": n, "rescues": rescues, "rescues_per_10": 10 * rescues / n if n else None,
            "attempts_per_rescue": n / rescues if rescues else None, "resets": counts[RESET],
            "overrides": counts[OVERRIDE], "failures": counts[FAILURE], "observations": counts[OBSERVATION],
            "knocks": sum(i["kind"] == "knocked" for i in interventions),
            "collisions": sum(i["kind"] == "collided" for i in interventions),
            "successes": len(successes), "clean_successes": len(successes) - assisted, "assisted_successes": assisted}
