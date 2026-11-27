"""Human interventions: what a person had to step in and do, as data, so "rescues per 10 attempts" can
go on a scorecard (ROADMAP 3) instead of being remembered.

Classes, because only one of them measures the policy:
  rescue       unplanned: something went wrong and a person fixed it (Esc, a hand on the tile, a crash)
  reset        part of the protocol: tiles put back after a discard, the mat rearranged
  override     a person's verdict differs from the automatic one (the success prompt): the check's error
  failure      not a step-in, but something went wrong (a knocked neighbour, the arm bumping a tile)
  observation  listed, not counted (a mark that only says "look at this", a cancelled choice)
  ask          a person answered a question and touched nothing (autonomous play: a tile name the robot
               wasn't sure of, a recount). Not a rescue, but a hand with one isn't clean (hand_grade)

Most are derived (derive()) from what a session logs anyway: attempts, the `detections` scans around
each attempt, session_end. What a person saw happen they say with a key as it happens (KEYS: b the arm
bumped another tile, ...), during the pickup or the return, instead of answering for it afterwards. They
are asked only after marking an attempt with → and no key, aborting it (Esc), answering a discard went
wide, with the bench's `note` command, or when a scan shows a neighbour of the handled tile moved or
turned and no bump was keyed (asked right then, so a detection glitch isn't logged as a knock:
`scan_review`). Every other change the scans show is settled by the scans after it (ChangeWatch): one
that goes back was the detector, one that stays happened. With no one there (bench.py --unattended) that goes
for the neighbours too; a person checks them afterwards (mahjong/bump_report.py: `scan_verdict` events).
"""
import json
import math
from collections import Counter
from functools import lru_cache

from . import config, kinematics, session

RESCUE, RESET, OVERRIDE, FAILURE, OBSERVATION, ASK = "rescue", "reset", "override", "failure", "observation", "ask"

# kind: (what a person is shown, class)
KINDS = {
    # asked after a mark (→ during ACT)
    "touched_tile": ("touched the tile to help", RESCUE),
    # not a step-in: the attempt can still be a clean success, but the collision is counted
    "collided": ("the arm bumped another tile, no one helped", FAILURE),
    "pushed_target": ("the arm pushed or turned the tile it was picking up", FAILURE),
    "helped_arm": ("held or steadied the arm or base", RESCUE),
    "caught_drop": ("caught a drop", RESCUE),
    "look": ("nothing touched, just look at this", OBSERVATION),
    # a held tile gripped near its bottom or its top: precarious, it may slip out (keys d / u); the attempt stands
    "grip_low": ("gripped low on the tile", OBSERVATION),
    "grip_high": ("gripped high on the tile", OBSERVATION),
    # held, but only by the jaw tips: the tile didn't go deep enough between the jaws and may drop (key s); the attempt stands
    "grip_shallow": ("gripped shallow: only the jaw tips held the tile", OBSERVATION),
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
    "moved_elsewhere": ("a tile away from the arm moved and stayed moved: not the arm", OBSERVATION),
    "unsettled": ("a change the session ended before later scans could settle", OBSERVATION),
    "arm_moved_while_off": ("arm not where it settled at the last disconnect (moved while off)", OBSERVATION),
    # autonomous play (scripts/play.py)
    "name_asked": ("said what a tile is: the robot wasn't sure of its name", ASK),
    "zone_asked": ("put the drawn tile on the drop zone again: none was seen there", ASK),
    "recount": ("checked the hand: the table didn't match what the robot thought it held", ASK),
    "count_unresolved": ("kept every name though more than 4 of a tile were counted: a discard may be misread", ASK),
    "pickup_failed": ("moved a tile by hand: the robot couldn't pick it up after its retries", RESCUE),
    "silent_error": ("the end-of-hand check found a tile the robot had wrong without noticing", FAILURE),
    "zone_unchecked": ("played before the bench checked a pickup on the drop zone (--skip-zone-test)", OBSERVATION),
    # face-down play (scripts/play_face_down.py): pressing Enter on the reader's name is no question, a correction is
    "name_corrected": ("said a tile's name: the base camera's reader had it wrong, or had none", ASK),
    "fd_by_hand": ("did a step by hand that the arm couldn't (a pick, a set-down, a lost tile)", RESCUE),
    "fd_conflict": ("unattended: a read made a fifth copy of a tile, and play went on with it", OBSERVATION),
}
MARK_KINDS = ["touched_tile", "collided", "pushed_target", "helped_arm", "caught_drop", "look", "other"]
# Pressed during a pickup or a return (the bench's KeyMarks): what happened, said as it happens. 2026-10-30: the
# questions at the end of a batch couldn't be answered accurately, nor told apart by attempt. Only what the arm
# does: a person helping (touched the tile, held the arm, caught a drop) never happens on the bench, and is
# still a → and an answer if it does.
# r says what the scan after it will show (a person moved tiles: putting a knocked one back for the next run),
# so it's no intervention by itself (BY_HAND): the scan's changes become straightened / tile_reset.
BY_HAND = "by_hand"
KEYS = {"b": "collided", "p": "pushed_target", "r": BY_HAND, "d": "grip_low", "u": "grip_high",
        "s": "grip_shallow", "l": "look"}
KEY_LABELS = {"b": "bumped another tile", "p": "pushed the tile it's picking up", "r": "I moved a tile by hand",
              "d": "gripped low", "u": "gripped high", "s": "gripped shallow",
              "l": "look at this"}
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
# A change this close to the tile the arm handled is a neighbour's (layouts: nearest neighbours 46-82mm, hand14's
# diagonals ~75mm). Answers to 212 scan changes (23 sessions to 2026-10-30): beyond 90mm, 43 of 47 were glitches;
# neighbours that moved or turned at 40-60mm, 30 of 39 were real; gone / appeared at any distance mostly glitches.
NEIGHBOUR_M = 0.085
FAINT_SAME_M = 2 * KNOCK_M      # a faint box this close to a lost tile is that tile, unmoved (as _state's "before")
# A watched change is real once this many later scans still show it (ChangeWatch: ~one attempt, 3 scans each);
# one a later scan shows undone was a glitch. Replayed on the same answers: 3 matched the person on 43 of 55
# watched changes (40 of their 44 glitches), 2 on 40. Most of the rest were real answers given for a whole
# scan that also held a real knock, or a tile put back before the next scan (which looks like a glitch).
SETTLE_SCANS = 3


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


SCAN_ANSWERS = {"knocked": {"knocked": "the arm (a knock)", BY_HAND: "I did, by hand",
                            "scan_glitch": "no, nothing moved (a detection glitch)"},
                "between": {"straightened": KINDS["straightened"][0], "tile_reset": KINDS["tile_reset"][0],
                            "both": "both", "scan_glitch": "nothing moved (a detection glitch)"}}


def ask_scan(scan, proposal, ask=None):
    """The kinds that stand for a scan's proposed change (MatWatch.propose), from a person who was
    there: Enter accepts the proposal. [BY_HAND]: a person moved it (the caller splits it, MatWatch.by_hand)."""
    kind, diff = proposal
    where = scan.get("attempt_dir") or "the mat"
    if kind == "knocked":
        prompt = (f"{where}: the scans say {changes_text(diff)} during the {scan['label'].removeprefix('after_')}. "
                  f"What moved it")
        answers = SCAN_ANSWERS["knocked"]
    else:
        prompt = f"{where}, since the last scan: {changes_text(diff)}. What was it"
        answers = SCAN_ANSWERS["between"]
    answer = ask_kind(prompt, list(answers), labels=answers, ask=ask, default=kind)
    return ["straightened", "tile_reset"] if answer == "both" else [answer]


# --- what changed on the mat between two scans ---

@lru_cache(maxsize=4096)
def _reachable_mm(x_mm, y_mm):
    return kinematics.hover_command(x_mm / 1000, y_mm / 1000, config.HOVER_OFFSET_M, config.HOVER_GRIPPER)[0] is not None


def on_mat(tile):
    """Within the arm's reach. Detections beyond it are phantoms or tiles off the workspace
    (2026-10-23: an 8D at (0.478, +0.232) came and went). To the mm, cached: the same tiles are checked in every scan, and
    the IK was nearly all of status.py's time."""
    x, y = tile["xy"]
    return _reachable_mm(round(x * 1000), round(y * 1000))


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
                turned.append({"tile_name": after[j]["tile_name"], "xy": list(after[j]["xy"]), "box_change_px": change,
                               "box_from": list(before[i]["box_px"]), "box_to": list(after[j]["box_px"])})

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


def _within(xy, points, radius_m=config.TILE_MERGE_RADIUS_M):
    return any(math.dist(xy, p) <= radius_m for p in points)


def changed(diff):
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
# Logged on a `detections` event (detector.log_scan, record.py) for MatWatch to remember places across missed scans.
# Only scans that logged it are read that way: replayed on older logs, a first version dropped answers people had given.
SCAN_MEMORY = {"scan_memory": True}


def handled_points(scan):
    """Where the arm handled tiles since the previous scan: this scan's target, and a pickup that had no scan of its own
    (`handled_xy`: play goes straight from a pickup to where the tile is set down)."""
    if scan["label"] not in ATTEMPT_SCANS:
        return []
    return ([scan["target_xy"]] if scan.get("target_xy") else []) + list(scan.get("handled_xy", ()))


class MatWatch:
    """What the scans say changed on the mat, scan by scan. derive() replays a session through it; the
    bench feeds it live, so a person can confirm or correct a change (ask_scan) while it's in front of them.

    On scans with SCAN_MEMORY it also lists, in `waiting`, places where a tile is known to be though the detector
    isn't seeing it, so that its first sighting there during an attempt isn't a knock (attempts 012 and 053 of
    20261120_181225_bench, which the owner judged false): where another check saw a tile still in place (restore), and
    where the arm set one down that the scan after didn't see. A place is dropped once a tile is seen or handled
    there, and after SETTLE_SCANS scans: a tile that isn't where it's thought to be must show again."""
    def __init__(self):
        self.previous, self.knocked_to, self.waiting = None, [], []

    def propose(self, scan):
        """(kind, diff) the scans suggest for this `detections` event against the previous one, or None.
        knocked: the arm handled tiles since the previous scan (the tile it handled, at target_xy, is
        left out); otherwise nothing on the arm touched the mat, so a person did: straightened if it
        moved a tile a knock had left, else tile_reset."""
        if self.previous is None:
            return None
        if scan["label"] in ATTEMPT_SCANS:
            diff = scene_diff(self.previous["tiles"], scan["tiles"], handled_points(scan))
            diff = without(diff, appeared=[t for t in diff["appeared"]
                                           if any(math.dist(t["xy"], w["xy"]) <= w["radius_m"] for w in self.waiting)])
            return ("knocked", diff) if changed(diff) else None
        diff = scene_diff(self.previous["tiles"], scan["tiles"])
        if not changed(diff):
            return None
        fixed = any(math.dist(p, k) <= config.TILE_MERGE_RADIUS_M for p in _from_points(diff) for k in self.knocked_to)
        return ("straightened" if fixed else "tile_reset", diff)

    def was_knocked(self, category, change):
        """A change (scene_diff `category`) that starts where a knock left a tile: a person straightening it,
        most likely."""
        if category == "appeared":
            return False
        start = change["from"] if category == "moved" else change["xy"]
        return any(math.dist(start, k) <= config.TILE_MERGE_RADIUS_M for k in self.knocked_to)

    def by_hand(self, diff):
        """A person's changes, split: (straightened: from where a knock left a tile, tile_reset: the rest)."""
        return split(diff, self.was_knocked)

    def advance(self, scan, knocked=None):
        """Move on to `scan`; `knocked`: the part of its changes that stands as a knock (where a person may
        straighten a tile later)."""
        if knocked:
            self.knocked_to += _to_points(knocked) + [t["xy"] for t in knocked["gone"]]
        self.previous = scan
        if not scan.get("scan_memory"):
            return
        seen = [t["xy"] for t in scan["tiles"] if t["confidence"] >= CHANGE_CONFIDENCE]
        over = seen + handled_points(scan)
        self.waiting = [{**w, "scans_left": w["scans_left"] - 1} for w in self.waiting
                        if w["scans_left"] > 1 and not _within(w["xy"], over)]
        target = scan.get("target_xy")
        if scan.get("label") == "after_return" and target and not _within(target, seen):
            self._wait(target, config.TILE_MERGE_RADIUS_M)      # where in there it landed was never seen

    def _wait(self, xy, radius_m):
        self.waiting.append({"xy": list(xy), "radius_m": radius_m, "scans_left": SETTLE_SCANS})

    def restore(self, tile):
        """A tile the latest scan missed that the wrist camera or a faint box saw where it was (`wrist_check` /
        `tile_faint`): the next scan is compared with it there. At confidence 0, so a tile the detector keeps missing
        isn't "gone" again at every scan, while one it sees again still matches it."""
        if self.previous is not None:
            restored = {"tile_name": tile["tile_name"], "xy": list(tile["xy"]), "confidence": 0.0}
            self.previous = {**self.previous, "tiles": self.previous["tiles"] + [restored]}
            if self.previous.get("scan_memory"):
                self._wait(restored["xy"], FAINT_SAME_M)        # unmoved, as far as a detection can tell


def split(diff, keep):
    """(the changes in a scene_diff where keep(category, change) holds, the rest), each a scene_diff."""
    a, b = {k: [] for k in diff}, {k: [] for k in diff}
    for k, changes in diff.items():
        for c in changes:
            (a if keep(k, c) else b)[k].append(c)
    return a, b


def where(change):
    """The points a change touches (a move: both ends)."""
    return [change[k] for k in ("from", "to", "xy") if k in change]


def near_any(change, points, radius_m):
    return any(math.dist(p, q) <= radius_m for p in where(change) for q in points)


def merge(*diffs):
    return {k: [c for d in diffs for c in d.get(k, [])] for k in ("moved", "turned", "gone", "appeared")}


def _state(change, kind, tiles):
    """Whether `tiles` (a later scan) show a change as it was before ("before"), as it was after ("after"), or
    neither (None: something else, no evidence either way)."""
    def nearest(xy):
        d = [(math.dist(t["xy"], xy), t) for t in tiles if on_mat(t)]
        return min(d, key=lambda p: p[0]) if d else (math.inf, None)
    r = config.TILE_MERGE_RADIUS_M
    if kind == "gone":
        # back where it was: an unmoved tile's detection moves 1-1.5mm, while a knocked neighbour can end up
        # within r of a lost tile's spot (5C 18mm from 3B's, test_bench)
        return "before" if nearest(change["xy"])[0] <= 2 * KNOCK_M else "after"
    if kind == "appeared":
        return "after" if nearest(change["xy"])[0] <= r else "before"
    if kind == "moved":
        d_from, d_to = nearest(change["from"])[0], nearest(change["to"])[0]
        if min(d_from, d_to) > r:
            return None
        return "before" if d_from < d_to else "after"
    d, tile = nearest(change["xy"])      # turned: the box back to its old shape, or still the new one
    if d > r or not tile or not tile.get("box_px") or "box_from" not in change:
        return None
    for state, box in (("before", change["box_from"]), ("after", change["box_to"])):
        if max(abs(a - b) for a, b in zip(tile["box_px"], box)) <= TURN_PX:
            return state
    return None


class ChangeWatch:
    """Changes the scans showed that no one was asked about, settled by the scans after them: a change that a
    later scan shows undone was the detector (a glitch: 5C went and came back 8 times in one session,
    2026-10-29); one that SETTLE_SCANS later scans still show happened. The bench feeds it each scan, so far
    tiles flickering aren't asked about, and nothing waits for the end of a batch."""
    def __init__(self, settle_scans=SETTLE_SCANS):
        self.settle_scans, self.pending = settle_scans, []
        self.glitches = Counter()       # tile name -> changes that went back: how flaky each tile's detection is

    def watch(self, scan, diff, kind):
        """Watch the changes in `diff`, from detections event `scan`; `kind`: what they are if they stay."""
        for cat, changes in diff.items():
            for c in changes:
                self.pending.append({"scan": scan["scan"], "attempt_dir": scan.get("attempt_dir"), "kind": kind,
                                     "category": cat, "change": c, "agreed": 0})

    def update(self, scan, diff, by_hand=False):
        """A new scan (and its diff against the previous one, or None). Returns (settled: [(pending, kind)],
        what of `diff` is left once undone changes are taken out, those taken out as glitches going back, those
        taken out as a person putting a knocked tile back): their going back isn't a new change. `by_hand`: a
        person said they moved tiles since the last scan, so a knock undone was them putting it back, not the
        detector (it stays a knock)."""
        settled, still, undone, put_back = [], [], [], []
        for p in self.pending:
            state = _state(p["change"], p["category"], scan["tiles"])
            if state == "before":
                hand = by_hand and p["kind"] == "knocked"
                p["put_back"] = hand
                if not hand:
                    self.glitches[p["change"]["tile_name"]] += 1
                if diff:
                    diff, back = split(diff, lambda k, c, p=p: not near_any(c, where(p["change"]),
                                                                            config.TILE_MERGE_RADIUS_M))
                    (put_back if hand else undone).append(back)
                settled.append((p, p["kind"] if hand else "scan_glitch"))
                continue
            p["agreed"] += state == "after"
            if p["agreed"] >= self.settle_scans:
                settled.append((p, p["kind"]))
            else:
                still.append(p)
        self.pending = still
        return settled, diff, merge(*undone), merge(*put_back)

    def forget(self, keep):
        """Drop pending changes where keep(pending) is False (a person explained them); returns the dropped."""
        dropped = [p for p in self.pending if not keep(p)]
        self.pending = [p for p in self.pending if keep(p)]
        return dropped

    def flush(self):
        """At the end of a session: what's still pending, as (pending, kind): the kind if a later scan agreed
        with it, else "unsettled"."""
        out = [(p, p["kind"] if p["agreed"] else "unsettled") for p in self.pending]
        self.pending = []
        return out


def without(diff, gone=(), appeared=()):
    """A scene_diff with those gone / appeared tiles left out (ones the wrist camera or a later scan explained)."""
    return {**diff, "gone": [t for t in diff["gone"] if t not in gone],
            "appeared": [t for t in diff["appeared"] if t not in appeared]}


def latest_verdicts(events):
    """A person's verdicts on what the scans said (`scan_verdict` events, appended by mahjong/bump_report.py, which may
    be long after the session), the latest per change: {(scan, category, its change as JSON): the event}."""
    return {(e["scan"], e["category"], json.dumps(e["change"], sort_keys=True)): e
            for e in events if e["type"] == "scan_verdict"}


def _apply_verdict(reviews, verdict):
    """A scan's `scan_review`s once a person has said whether one change happened: it's taken out of what they say,
    and stands as verdict["kind"] (not moved: a glitch). by="scans", so it isn't counted as a person's step in the run."""
    category, change = verdict["category"], verdict["change"]
    out = []
    for r in reviews:
        if r["kinds"] and change in r.get("changes", {}).get(category, []):
            changes = {k: [c for c in v if not (k == category and c == change)] for k, v in r["changes"].items()}
            if not changed(changes):
                continue
            r = {**r, "changes": changes}
        out.append(r)
    kind = verdict["kind"] if verdict["verdict"] == "moved" else "scan_glitch"
    return out + [{"type": "scan_review", "scan": verdict["scan"], "kinds": [kind], "changes": merge({category: [change]}),
                   "by": "scans", "verified": True}]


def scan_reviews(events):
    """{scan number: what was said about its changes}, with a person's later verdicts applied."""
    reviews = {}
    for e in events:
        if e["type"] == "scan_review":
            reviews.setdefault(e["scan"], []).append(e)
    for verdict in latest_verdicts(events).values():
        reviews[verdict["scan"]] = _apply_verdict(reviews.get(verdict["scan"], []), verdict)
    return reviews


def derive(events):
    """Every intervention in a session's events, logged or derived, in order:
    [{kind, class, attempt_dir, source, time, ...}]. source: logged (an intervention event or a logged
    fact), mark (asked after → during ACT), key (a KEYS press during a pickup or a return: a `keys` event),
    derived (from attempt fields / session_end), scans (from consecutive `detections` events, or settled by
    the scans after them: `scan_review` by "scans"), confirmed (scans a person answered or keyed for:
    `scan_review`). Each is attributed to the attempt it belongs to, or the most recent one; session-level
    ones to None."""
    out = []

    def add(kind, attempt_dir, source, time, **fields):
        out.append({"kind": kind, "class": class_of(kind), "attempt_dir": attempt_dir, "source": source,
                    "time": time, **{k: v for k, v in fields.items() if v not in (None, "")}})

    reviews = {e["attempt_dir"]: e for e in events if e["type"] == "attempt_review"}
    # presses and marks found afterwards not to be the person's (keys_void, appended later: the bench read
    # keystrokes typed in another terminal, 2026-10-30) say nothing
    void = {(v["attempt_dir"], v["during"]) for e in events if e["type"] == "keys_void" for v in e.get("keys", [])}
    void_marks = {a for e in events if e["type"] == "keys_void" for a in e.get("marks", [])}
    events = [e for e in events if not (e["type"] == "keys" and (e["attempt_dir"], e.get("during")) in void)]
    # a → during a pickup someone also pressed a key for says nothing the key didn't
    keyed = {e["attempt_dir"] for e in events if e["type"] == "keys" and e.get("during") == "pickup" and e["presses"]}
    reviews_of_scan = scan_reviews(events)
    last_attempt, mat, ended = None, MatWatch(), False
    for e in events:
        kind, time = e["type"], e.get("time")
        if kind == "attempt":
            a = {**e, **{k: v for k, v in reviews.get(e["attempt_dir"], {}).items()
                         if k in ("abort_reason", "mark_kind", "mark_note")}}
            last_attempt = a["attempt_dir"]
            if a.get("stop_reason") == "aborted":
                add("aborted", last_attempt, "logged", time, reason=a.get("abort_reason"))
            if a.get("marks_s") and last_attempt not in void_marks and (a.get("mark_kind") or last_attempt not in keyed):
                add(a.get("mark_kind") or "unspecified", last_attempt, "mark", time, marks_s=a["marks_s"],
                    note=a.get("mark_note"))
            if "gripper_says_held" in a and a["gripper_says_held"] != a["success"]:
                add("success_overridden", last_attempt, "derived", time, gripper_says_held=a["gripper_says_held"],
                    success=a["success"])
        elif kind == "keys":
            for k in dict.fromkeys(p["kind"] for p in e["presses"] if p["kind"] != BY_HAND):
                add(k, e["attempt_dir"], "key", time, during=e["during"],
                    at_s=[p["t_s"] for p in e["presses"] if p["kind"] == k])
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
            answered = reviews_of_scan.get(e.get("scan"), [])
            if proposal is None:    # the scans saw nothing: only a person saying something moved all the same stands
                answered = [r for r in answered if r.get("verified")]
            during = e["label"].removeprefix("after_") if e["label"] in ATTEMPT_SCANS else None
            # each review stands for the changes it names (older sessions: one review for the whole scan)
            found = [(k, r.get("changes") or proposal[1], r.get("by")) for r in answered for k in r["kinds"]]
            if proposal and not answered:
                found = [(proposal[0], proposal[1], "scans")]
            for k, changes, by in found:
                add(k, (e.get("attempt_dir") if during else None) or last_attempt,
                    "scans" if by == "scans" else "confirmed", time, during=during, changes=changes,
                    by=None if by == "scans" else by)
            mat.advance(e, merge(*[c for k, c, _ in found if k == "knocked"]))
        elif (kind == "wrist_check" and e.get("present")) or kind == "tile_faint":
            mat.restore({"tile_name": e["tile"], "xy": e["xy"]})
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
            "asks": counts[ASK],
            "knocks": sum(i["kind"] == "knocked" for i in interventions),
            "collisions": sum(i["kind"] == "collided" for i in interventions),
            "target_pushes": sum(i["kind"] == "pushed_target" for i in interventions),
            "low_grips": sum(i["kind"] == "grip_low" for i in interventions),
            "high_grips": sum(i["kind"] == "grip_high" for i in interventions),
            "shallow_grips": sum(i["kind"] == "grip_shallow" for i in interventions),
            "successes": len(successes), "clean_successes": len(successes) - assisted, "assisted_successes": assisted}


def hand_grade(interventions):
    """One hand of autonomous play (docs/autonomy_plan.md, section 3): "silent error" if the end-of-hand
    check found a tile the robot had wrong, else "rescued" if a person touched anything (a rescue or a
    reset), else "attended" if they only answered questions (asks, scan questions, a knock), else
    "clean": a person did nothing but put the drawn tiles down."""
    kinds = {i["kind"] for i in interventions}
    classes = {i["class"] for i in interventions}
    if "silent_error" in kinds:
        return "silent error"
    if classes & {RESCUE, RESET}:
        return "rescued"
    if ASK in classes or FAILURE in classes or any(i["source"] == "confirmed" for i in interventions):
        return "attended"
    return "clean"
