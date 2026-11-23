"""Test bench for isolated pickups: closed-loop hover + an ACT policy, one tile at a time or in batches.

    1 hover          detect, choose a tile, hover over it (corrected, settled from above); saves a wrist photo
    2 pickup         hover, then the policy picks it up; parks at rest holding it
    3 return         put the held tile back at its home (where it was first seen this session), aimed so it
                     lands there (mahjong/return_correction.py; up to 15mm closer if beyond reach), then rest
    4 discard        carry the held tile to the taught discard pose and release it, then rest
    5 teach-discard  drive the arm over the discard container with the gamepad and save that pose
    6 trials         pickup -> success check -> return, N times on one tile; prints the success rate
    7 rest           lift if low, then park (never opens the gripper)
    8 note           log something you had to do (straightened tiles, put tiles back, ...) for the last attempt
    9 setup          the setup changed mid-session (lamp on, battery swapped): answer the setup questions again
   10 quit

Type a number or a name. While the arm picks up or returns a tile, say what you see as it happens with a
key (no Enter; the keys are printed as each pickup and return starts):

    b  the arm bumped another tile    p  the arm pushed the tile it's picking up
    r  I moved a tile by hand (put a knocked one back for the next run)    l  look at this
    d  gripped low on the tile    u  gripped high on the tile    (precarious: it may slip; pickup or return)

During a pickup, Esc stops the policy, and → marks the moment without stopping it (a hand on the tile, a
caught drop); a → with no key is asked about right after the attempt. Clean attempts ask nothing. Everything goes in a session folder
(sessions/<time>_bench/): the terminal log, events.jsonl (with a RealSense scan before and after each
pickup, return and discard, so knocked tiles and tiles put back by hand show up without anyone saying
so; each scan's frames and boxes go in scans/, for the tile detector's data: scripts/mine_frames.py),
and per attempt the wrist frames (wrist.mp4: every frame), act_trace.npz and RealSense overview frames
before and after. A hover alone gets a folder too, with its wrist photo (wrist_0000.jpg, as a pickup's first
frame). `python scripts/session_report.py` summarises the latest one, interventions included
(mahjong/interventions.py).

What a scan sees change is settled without the end-of-batch questions:
  - you pressed r during that pickup or return: what changed was you (straightening a tile a knock left,
    else putting tiles back), with b too, the neighbours are asked about;
  - a neighbour of the handled tile (within 85mm) that moved or turned: a knock if you pressed b during
    that pickup or return, else you're asked right then: the arm (Enter), you by hand, or a glitch;
  - anything else (tiles further away, tiles gone or appearing): watched over the next scans. Undone by
    a later scan: a detection glitch (unless you said you moved tiles: a knocked tile you put back). Still
    so 3 scans later: real, a knock near the arm, or moved by something else further away (by a person, if
    it started where a knock had left it).
A `note` of straightened tiles or tiles put back, between commands, says the same for the next scan. The tiles whose detections flickered are listed at the end of trials.

A tile a scan says went, or a spot's tile not detected, is looked at with the wrist camera before anyone is asked
(mahjong/wrist_check.py): the arm hovers where the tile was last seen and checks for a face (face down: a back) between the jaw
tips. A box the detector is too unsure of to name (down to config.DETECTION_PRESENCE_CONFIDENCE) needs no look: the tile's
there (`tile_faint` events). On a pale surface it can only compare the view with an earlier look at that place, and asks you the first time.
Still there: a detection glitch, logged as one
(`wrist_check` and `scan_review` events, the frame in
wrist_checks/), and the tile is picked up from where it was. Not there: the change is real. While the arm holds a tile
its wrist view is blocked, so the look waits for the next scan (which may see the tile again by itself).

    python scripts/bench.py                         # interactive, default policy
    python scripts/bench.py --trials 5 --tile 8D    # batch, then quit
    python scripts/bench.py --sim --trials 3        # no hardware
    python scripts/bench.py --trials 15 --tile 4D --hover-perturb-mm 0,6x,-6x,6y,-6y
    python scripts/bench.py --trials 2 --tiles 2D,6B,1D,3D    # these tiles in turn, 2 rounds each
    python scripts/bench.py --layout eval_new --trials 2 --spots S2,S5   # only these spots (--start-spot S3: S3 to the end)
    python scripts/bench.py --policy pickup_mat_cl,pickup_mat_cl_v2n --layout eval_new --trials 2   # 2 x spots x 2

Returns aim at the tile's home, not where the jaws closed: returning to each grasp point walked tiles
77-100mm in 10 rounds. Each return's landing goes to data/return_corrections.jsonl (what the return model
is fitted from); one landing more than 12mm from home is printed, and past 25mm (or out of the hover's
reach) you're asked to put the tile back on its home, logged as a reset.

Every successful attempt also saves a hover-correction sample (source "bench": where the jaws closed
minus the detection) to data/hover_corrections.jsonl; it's only fitted once data/hover_model.json
lists "bench". --hover-perturb-mm cycles deliberate hover offsets through the trials, to test whether
the policy follows the tile or replays a fixed move from wherever it starts (session_report.py prints
the result); those attempts' samples are saved as excluded.

--policy a,b compares policies paired (ROADMAP 4, 5): trials run in rounds, each policy once per round
from the same hover offset, in alternating order (a b, b a, ...) so a drift over the session (battery,
daylight) doesn't favour one. Each attempt starts from its own policy's hover (data/policies.json).
--layout NAME (taught with scripts/layout.py) walks the layout's spots: --trials rounds at each spot,
picking the tile detected nearest the spot; a spot it can't attempt is skipped, not the walk.
session_report.py prints success per spot and policy side by side.
--spots S1,S3 visits only those spots (in that order); --start-spot S3 starts there and goes on to the last spot.
--tiles 2D,6B,1D picks each of those tiles in turn (--trials rounds each), without a layout.

--face-down (with --layout) picks tiles lying GREEN side up, and you're asked once to lay a tile on each spot. With detector
weights that have the back class (config.BACK), a back is found and checked like any tile: each pickup aims at the back the
scan sees nearest the spot, a return's landing is checked, a knocked back shows in the scans, and --unattended works. Its
hover and return samples are logged but excluded from the fits until a back's box is checked against a face's. With older
weights nothing sees a back, so each pickup aims at its spot's saved position (layout.saved_target, as record.py --face-down
does), nothing sees where a return lands (no landing check, no samples), and a tile that walked is yours to square up (r).
A face-down session counts as the face-down setup when the scans found the backs, compared only with other face-down
sessions; by saved position it's no counted result (scorecard.off_standard, docs/scorecard.md).

--unattended runs --trials with no one watching: nothing is asked once the trials start (the setup questions still
are, before them). A neighbour the scans see move or turn is a self-reported bump, settled by the later scans like
any other watched change; a tile that isn't found is skipped, a tile that lands far off its home is left there, and a
wrist camera that won't reopen ends the run. Ctrl-C stops it. Afterwards `python scripts/bump_report.py` shows every
pickup and return before / after, to say which tiles really moved (mahjong/bump_report.py).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import (back_check, bump_report, config, detector, interventions, layout, perception, placements, play_log,
                     takeover, wrist_check)
from mahjong.camera import RERUNS as CAMERA_RERUNS, grab_frames
from mahjong.gamepad import warn_off_centre
from mahjong.hover_correction import HoverCorrection, append_record, grasp_record, parse_offsets
from mahjong.perception import choose_tile, drain_stdin
from mahjong.terminal_keys import Abort, TerminalKeys, new_events
from mahjong.return_correction import Homes, ReturnCorrection, check_landing
from mahjong.session import Session, ask_conditions, ask_notes


class Mark:
    """→ during ACT, from the same terminal, as policy.run's `mark`. → sets only exit_early (← also
    sets it, so marks too); Esc sets it as well, but that's an abort."""
    def __init__(self, events):
        self.events = events

    def clear(self):
        self.events["exit_early"] = self.events["rerecord_episode"] = False

    def is_set(self):
        return self.events["exit_early"] and not self.events["stop_recording"]


class KeyMarks:
    """Letter keys (interventions.KEYS) pressed while the arm works: what a person saw happen, said as it happens
    instead of asked afterwards. Read from the bench's own terminal (`terminal`, a TerminalKeys), only between
    start() and stop(), so keys typed at a prompt, the menu or another terminal tab aren't marks. Without a
    terminal (sim, tests, piped input) press() is the way in."""
    def __init__(self, terminal=None):
        self.phase, self.presses, self.t0, self.terminal = None, [], 0.0, terminal

    def start(self, phase):
        """Count keys from now (`phase`: pickup or return), with a reminder of which."""
        self.phase, self.presses, self.t0 = phase, [], time.time()
        if self.terminal:
            self.terminal.start()
        keys = "  ".join(f"{k} {label}" for k, label in interventions.KEY_LABELS.items())
        print(f"  keys: {keys}" + ("  |  → mark  Esc stop" if phase == "pickup" else ""))

    def press(self, char):
        kind = interventions.KEYS.get((char or "").lower())
        if self.phase is None or kind is None:
            return
        self.presses.append({"key": char.lower(), "kind": kind, "t_s": round(time.time() - self.t0, 2)})
        print(f"  [{char.lower()}] {interventions.KEY_LABELS[char.lower()]} (noted)")

    def stop(self):
        if self.terminal:
            self.terminal.stop()
        presses, self.phase, self.presses = self.presses, None, []
        return presses

    def close(self):
        if self.terminal:
            self.terminal.stop()


# why an attempt was voided and rerun (policy.run's camera_problem)
VOID_LOST = "wrist camera stopped giving frames"
VOID_TORN = "wrist camera frames torn (USB data lost)"


BACK_UNCHECKED = ("face down: a back box's centre hasn't been checked against a face box's at the same spots, and the hover "
                  "and return models are fitted from detections")


def back_excluded(tile, checked):
    """Provenance that keeps a hover or return sample from a back (perception.is_back) out of the fits, logged all the same,
    unless the live weights' back centres passed the check against face centres (`checked`: detector.backs_checked,
    worked out once a session)."""
    return {"excluded": BACK_UNCHECKED} if perception.is_back(tile) and not checked else {}


class Bench:
    SESSION_KIND = "bench"          # the session folder's suffix (scripts/play.py builds on the bench as "play")

    def __init__(self, args):
        self.args = args
        ids = parse_policies(args.policy)
        # before connecting anything, so a typo fails fast
        self.spots = layout.load(args.layout)["spots"] if args.layout else None
        unknown = [c for c in args.carry or [] if c not in {s["id"] for s in self.spots}]
        if unknown:
            raise SystemExit(f"--carry: {args.layout} has no spot {', '.join(unknown)}")
        self.pick_spots()       # before connecting anything, so a mistyped spot fails fast
        if args.sim:
            from mahjong import sim
            # face down: a simulated tile green side up on each spot, as a person lays them (boxed as backs, or by older
            # weights not at all: sim.SEES_BACKS)
            tiles = [(config.BACK, *s["xy"]) for s in self.spots] if args.face_down else sim.SIM_TILES
            devices = sim.connect(tiles=tiles, wrist_view=True)
            self.arm, self.camera, self.realsense = devices["arm"], devices["wrist_camera"], devices["realsense"]
            self.policies = [sim.SimPolicy(mat=self.realsense, policy_id=i or "sim") for i in ids]
            self.takeover = takeover.make(args.dagger_dataset, None)     # nothing to hand over in the sim
            self.sees_backs = self.face_down_detector()       # before the session folder: it may refuse to start
            self.weights_name = detector.weights_id()          # once: it hashes the weights file
            self.backs_checked = detector.backs_checked(weights_name=self.weights_name)
            self.session = Session(self.SESSION_KIND, ask_setup=False, policy=args.policy or "sim", sim=True,
                                   detector=self.weights_name)
        else:
            from mahjong import hardware
            from mahjong.policy import Policy
            self.policies = [Policy(i) for i in ids]       # before connecting, so a mistyped id fails fast
            # hardware first: nothing (not even the session folder) is created until it's all there
            if args.dagger_dataset:         # one dataset, one hover: before connecting, so it fails fast
                takeover.same_hover([self.hover_of(p) for p in self.policies])
            devices = hardware.connect(["arm", "wrist_camera", "realsense"] + takeover.devices(args.dagger_dataset),
                                       gamepad_kwargs=takeover.gamepad_kwargs(args.device))
            self.arm, self.camera, self.realsense = devices["arm"], devices["wrist_camera"], devices["realsense"]
            # A during a pickup hands a person the arm, and the recovery is a DAgger episode (mahjong/takeover.py)
            self.takeover = takeover.make(args.dagger_dataset, devices.get("gamepad"))
            if self.takeover:
                self.takeover.expect_hover(self.hover_of(self.policies[0]))
            self.sees_backs = self.face_down_detector()       # before the session folder: it may refuse to start
            self.weights_name = detector.weights_id()          # once: it hashes the weights file
            self.backs_checked = detector.backs_checked(weights_name=self.weights_name)
            self.session = Session(self.SESSION_KIND, policy=args.policy or "default", sim=False,
                                   detector=self.weights_name)
        # →, ←, Esc and the letter keys come from this terminal only, while the arm works (terminal_keys.py)
        self.events = new_events()
        self.terminal = TerminalKeys(self.events, on_char=lambda char: self.keys.press(char))
        self.abort, self.mark = Abort(self.events), Mark(self.events)
        self.keys = KeyMarks(self.terminal)
        self.last_keys = []         # the keys pressed during the latest pickup or return: what its scan is judged by
        self.hands_on = None        # why a person moved tiles since the last scan (note, a missed discard): the next
                                    # scan's changes are theirs, already logged
        if self.arm.at_connect:
            self.session.event("connect", **self.arm.at_connect)
        self.last_attempt = None    # attempt_dir of the latest attempt: what a note or an interruption belongs to
        self.to_review = []         # (attempt_dir, tile, run) marked or aborted during trials, asked at the end
        self.mat, self.scan_n = interventions.MatWatch(), 0
        self.scan_checks = []       # (detections event, proposal): neighbours' changes a person hasn't confirmed yet
        self.changes = interventions.ChangeWatch()      # every other change, until later scans settle it
        self.unlooked = []          # (scan number, tile) a scan said went while the arm held a tile: looked at next
        self.looked_empty = []      # (scan number, xy) the wrist camera saw no tile at
        self.views = wrist_check.PlaceViews()       # what the wrist camera saw where lost tiles turned out to be
        # the wrist's back check, asked first at a look at a back: loaded here so a bad file stops the bench before the arm
        # moves; None without assets/back_check.pt (the back's shape decides, as before)
        self.back_reader = back_check.load_reader()
        print(f"Back check: {self.back_reader.path if self.back_reader else 'not installed (a back is found by its shape)'}")
        self.names = {}             # {spot id or tile name: the tile last picked up there}
        self.placements = placements.Placements(self.session)   # what's on the mat, for the name report
        self.ask_names = None if args.sim else input            # a simulated mat's reads stand as what's there
        self.batch = False          # during trials: a tile not found is put back by a person, not skipped
        self.defer = False          # questions wait for the caller (play: the end of the step)
        self.direct = False         # play: from a pickup straight to the discard or the place it's set down, with no trip
                                    # to rest or scan between. Off here: counted bench results keep one motion
        self.unseen = None          # {xy, attempt_dir} of a pickup that had no scan of its own: the next scan answers for it
        self.laid = set()           # --face-down: the spots a person has said have a tile green side up on them
        # The sim's tiles are where they're detected (SimPerception.arm_offset is None), so the real arm's correction, fitted on
        # the RealSense-to-arm gap (up to ~31mm), would hover between two sim tiles: SimPolicy then took the neighbour, and play's
        # sim hands set tiles down on top of others (found by mahjong/motion_watch.py, 2026-11-13). None in the sim.
        self.correction = HoverCorrection([]) if args.sim else HoverCorrection.load()
        # a simulated grasp says nothing about the real hover: keep its samples in the session folder
        self.samples_path = self.session.dir / "hover_corrections.jsonl" if args.sim else config.HOVER_CORRECTIONS
        # simulated returns land exactly where they're set down: the real return model doesn't apply
        self.returns_path = self.session.dir / "return_corrections.jsonl" if args.sim else config.RETURN_CORRECTIONS
        self.returns = ReturnCorrection.load(self.returns_path)
        # {"tile_name", "detection_xy", "home_xy", "grasp_xy", "approach_offset_m", "attempt_dir", "spot"} while holding a tile
        self.held = None
        self.homes = Homes()        # {spot or tile: where it was first seen this session}: every return aims there
        self.policy = self.policies[0]      # the one single commands use; trials take turns
        for policy in self.policies:
            print(f"Policy: {policy.id}, hover {self.hover_of(policy)}")
        print(f"Hover correction: {self.correction.summary()}")
        print(f"Return correction: {self.returns.summary()}")
        if self.spots:
            print(f"Layout {args.layout}: " + ", ".join(s["id"] for s in self.spots))
        self.session.event("bench_setup", policy=self.policy.id, hover=self.hover_of(self.policy),
                           policies=[{"id": p.id, "hover": self.hover_of(p)} for p in self.policies],
                           layout=args.layout, spots=self.spots, unattended=args.unattended, carry=args.carry,
                           correction=self.correction.summary(), return_correction=self.returns.summary(),
                           hover_perturb_offsets_m=args.hover_perturb_mm, hover_gripper_override=args.hover_gripper,
                           hover_lower_mm=args.hover_lower_mm, face_down=args.face_down,
                           dagger_dataset=args.dagger_dataset if self.takeover else None)
        if args.face_down:
            self.session.event("face_down_setup", sees_backs=self.sees_backs, detector=self.weights_name)
            print("Face down: the detector sees backs, so pickups aim at the back it finds and returns are checked"
                  if self.sees_backs else "Face down: the detector can't see backs, so every pickup aims at its spot's saved "
                                          "position, and nothing can see where a return lands")

    # --- pieces ---

    def face_down_detector(self):
        """--face-down: whether the detector sees backs. Weights with the back class find each tile and its landing as any
        scan does; older ones can't, so every pickup goes by the spot's saved position and someone has to watch."""
        if not self.args.face_down:
            return False
        sees = perception.sees_backs(self.realsense)
        if self.args.unattended and not sees:
            raise SystemExit(f"--face-down --unattended needs detector weights with the {config.BACK!r} class "
                             f"({config.TILE_DETECTOR.name} has none): nothing else can see a back that walked")
        return sees

    def hover_of(self, policy):
        """The policy's hover (data/policies.json), with --hover-gripper's opening if given. The policy
        moves the gripper as deltas from it, so its narrowing on the way down shifts by the difference.
        The wrist look keeps the policy's own opening: its face-fill threshold was measured there.
        --hover-lower-mm lowers the hover's height too (its offset and the clearance the settle aims for, together, so the
        settle may go that far down): an experiment on where the policy's descent ends, never a counted result."""
        hover = dict(policy.hover)
        if self.args.hover_gripper is not None:
            hover["gripper"] = self.args.hover_gripper
        lower = (self.args.hover_lower_mm or 0.0) / 1000
        if lower:
            hover["offset_m"] -= lower
            hover["clearance_m"] -= lower
        return hover

    def scan(self, label, confirmed=None, look=True, **fields):
        """Detect tiles and log them as a `detections` event: consecutive scans show what changed on the
        mat (interventions.MatWatch). A tile that went is looked at with the wrist camera first (look_again;
        not while the arm holds a tile: `look`); changes that undo or repeat earlier watched ones settle those
        (ChangeWatch); what's left is sorted (sort_changes), unless a person already said what it was
        (`confirmed`: its kinds, e.g. ["tile_reset"]). A person who moved tiles since the last scan (r pressed
        during the pickup or return it follows, `hands_on`, or `confirmed`) makes a watched knock that went back
        theirs: put back, not a glitch."""
        unseen, self.unseen = self.unseen, None
        if unseen:                  # what the pickup did is only seen now, so this is an attempt's scan
            self.snapshot(self.session.dir / unseen["attempt_dir"] / "overview_after.jpg")
            if label in interventions.ATTEMPT_SCANS:
                fields["handled_xy"] = [unseen["xy"]]
            else:
                label = "after_pickup"
                fields.update(attempt_dir=unseen["attempt_dir"], target_xy=unseen["xy"], released=not self.held)
        self.scan_n += 1
        tiles, scan = detector.log_scan(self.realsense, self.session, self.scan_n, label, **fields)
        hand, self.hands_on = self.hands_on, None
        keyed = label in interventions.ATTEMPT_SCANS and self.pressed(interventions.BY_HAND)
        proposal, seen = self.mat.propose(scan), []
        if confirmed is None and hand is None and not keyed:
            proposal, seen = self.look_again(scan, proposal, look and not self.held)
        settled, diff, undone, put_back = self.changes.update(
            scan, proposal[1] if proposal else None, by_hand=confirmed is not None or hand is not None or keyed)
        self.log_settled(settled)
        if interventions.changed(undone):
            self.session.event("scan_review", scan=scan["scan"], proposed=proposal[0], kinds=[], changes=undone,
                               by="scans", note="undoes an earlier scan's glitch")
        if interventions.changed(put_back):     # the person's own word for it (a note, a reset) is logged already
            self.session.event("scan_review", scan=scan["scan"], proposed=proposal[0],
                               kinds=["straightened"] if keyed else [], changes=put_back,
                               by="key" if keyed else "person", note="a knocked tile put back by hand")
        proposal = (proposal[0], diff) if proposal and interventions.changed(diff) else None
        knocked = None
        if proposal and confirmed is not None:
            self.session.event("scan_review", scan=scan["scan"], proposed=proposal[0], kinds=confirmed)
        elif proposal and hand is not None:
            self.session.event("scan_review", scan=scan["scan"], proposed=proposal[0], kinds=[],
                               changes=proposal[1], by="person", note=hand)
        elif proposal:
            knocked = self.sort_changes(scan, proposal)
        self.mat.advance(scan, knocked)
        for tile in seen:
            self.mat.restore(tile)
        return tiles + [{"tile_name": t["tile_name"], "xy": tuple(t["xy"]), "confidence": 0.0, "wrist_seen": True}
                        for t in seen]

    def look_again(self, scan, proposal, look):
        """The tiles this scan says went (and ones an earlier scan did while the arm held a tile), looked at again
        before anyone is asked: a later scan that sees one back where it was, or the wrist camera seeing it there
        (wrist_look), makes it a detection glitch. Its change is dropped from the scan's proposal (or from the
        earlier scan's, still waiting in scan_checks), and a proposal left with nothing is logged as a
        "scan_glitch". Returns (the proposal left or None, the tiles the wrist camera saw: put back in this scan)."""
        kind, diff = proposal or (None, None)
        lost = [(scan["scan"], t) for t in (diff["gone"] if diff else [])]
        for point in interventions.handled_points(scan):    # the arm handled this one: it's not where it was
            self.views.forget(point)
        todo, self.unlooked, seen = self.unlooked + lost, [], []
        for scan_n, tile in list(todo):         # a faint box needs no arm, so a held tile doesn't get in the way
            if self.faint_at(tile, scan_n, interventions.FAINT_SAME_M):
                todo.remove((scan_n, tile))
                seen.append(tile)
                if scan_n == scan["scan"]:
                    diff = interventions.without(diff, gone=[tile])
                else:
                    self.drop_change(scan_n, tile, "a faint box still there")
        if not look:                # the held tile blocks the wrist view
            self.unlooked, todo = todo, []
        for scan_n, tile in todo:
            back = next((t for t in (diff["appeared"] if diff else [])
                         if math.dist(t["xy"], tile["xy"]) <= config.TILE_MERGE_RADIUS_M), None)
            if scan_n != scan["scan"] and back is not None:
                diff = interventions.without(diff, appeared=[back])
                self.drop_change(scan_n, tile, "a later scan saw it back")
                continue
            if self.crowded(tile, scan["tiles"], scan_n):
                continue
            present = self.wrist_look(tile, scan_n)["present"]
            if present is False and scan_n != scan["scan"]:
                self.settle_gone(scan_n, tile)
            if not present:
                continue
            seen.append(tile)
            if scan_n == scan["scan"]:
                diff = interventions.without(diff, gone=[tile])
            else:
                self.drop_change(scan_n, tile, "the wrist camera saw it")
        if todo and not self.arm.is_at_rest():
            self.arm.rest()         # out of the RealSense's view again
        if proposal and not interventions.changed(diff):
            self.session.event("scan_review", scan=scan["scan"], proposed=kind, kinds=["scan_glitch"],
                               note="looked again: every change was a detection glitch")
            return None, seen
        return ((kind, diff) if proposal else None), seen

    def sort_changes(self, scan, proposal):
        """A scan's changes, three ways. After a pickup or a return, a neighbour of the handled tile (within
        NEIGHBOUR_M) that changed: a knock if b was pressed during it (logged now, by="key"); one that moved or
        turned with no b: asked (check_scans, right after the attempt). Everything else, and every change between
        attempts: watched (ChangeWatch) as what it would be if it stays: a knock near the arm, moved_elsewhere
        away from it, and between attempts the proposal (a person straightening or putting tiles back). A tile
        the wrist camera saw gone is real already. Returns the changes standing as a knock for now."""
        kind, diff = proposal
        empty = {k: [] for k in diff}
        handled = interventions.handled_points(scan)
        if not handled and self.args.unattended:        # no one here to have straightened or put back anything
            self.watch_changes(scan, diff, "moved_elsewhere")
            return None
        if not handled:
            straightened, rest = self.mat.by_hand(diff)
            self.watch_changes(scan, straightened, "straightened")
            self.watch_changes(scan, rest, "tile_reset")
            return None
        during = scan["label"].removeprefix("after_")
        bumped, hand = self.pressed("collided"), self.pressed(interventions.BY_HAND)
        if hand and not bumped:         # r: what changed was the person
            self.log_by_hand(scan, kind, diff, "key", f"r pressed during the {during}")
            return None
        near, far = interventions.split(diff, lambda k, c: interventions.near_any(c, handled, interventions.NEIGHBOUR_M))
        looked = [xy for n, xy in self.looked_empty if n == self.scan_n]
        absent, near = interventions.split(near, lambda k, c: k == "gone" and interventions.near_any(
            c, looked, config.TILE_MERGE_RADIUS_M))
        absent_far, far = interventions.split(far, lambda k, c: k == "gone" and interventions.near_any(
            c, looked, config.TILE_MERGE_RADIUS_M))
        if hand:                        # b and r: the neighbours could be either, the rest was the person
            asked, keyed, watched, absent = interventions.merge(near, absent), empty, empty, empty
            self.log_by_hand(scan, kind, interventions.merge(far, absent_far), "key", f"r pressed during the {during}")
            far, absent_far = empty, empty
        elif bumped:
            keyed, asked, watched = near, empty, empty
        else:
            keyed = empty
            asked, watched = interventions.split(near, lambda k, c: k in ("moved", "turned"))
        if self.args.unattended and interventions.changed(asked):    # no one to ask: the scans decide, bump_report checks
            print(f"  self-report: {interventions.changes_text(asked)} during the {during} (the next scans settle it)")
            watched, asked = interventions.merge(watched, asked), empty
        knocked = interventions.merge(keyed, absent)
        if interventions.changed(knocked):
            by = "key" if interventions.changed(keyed) else "wrist camera"
            note = f"b pressed during the {during}" if by == "key" else "the wrist camera saw the tile gone"
            self.session.event("scan_review", scan=scan["scan"], proposed=kind, kinds=["knocked"], changes=knocked,
                               by=by, note=note)
            print(f"  {interventions.changes_text(knocked)} during the {during}: a knock ({note})")
        if interventions.changed(absent_far):
            self.session.event("scan_review", scan=scan["scan"], proposed=kind, kinds=["moved_elsewhere"],
                               changes=absent_far, by="wrist camera", note="the wrist camera saw the tile gone")
        if interventions.changed(asked):
            self.scan_checks.append((scan, (kind, asked)))
        self.watch_changes(scan, interventions.merge(watched), "knocked")
        # a tile a knock left, moving again away from the arm: a person straightening it, r or not
        straightened, far = self.mat.by_hand(far)
        self.watch_changes(scan, straightened, "straightened")
        self.watch_changes(scan, far, "moved_elsewhere")
        return interventions.merge(knocked, asked)

    def pressed(self, kind):
        """Whether a key for `kind` was pressed during the latest pickup or return."""
        return any(p["kind"] == kind for p in self.last_keys)

    def log_by_hand(self, scan, kind, diff, by, note):
        """Changes a person made: straightened where they start from a tile a knock left, else tile_reset."""
        for k, part in zip(("straightened", "tile_reset"), self.mat.by_hand(diff)):
            if interventions.changed(part):
                self.session.event("scan_review", scan=scan["scan"], proposed=kind, kinds=[k], changes=part, by=by,
                                   note=note)
                print(f"  {interventions.changes_text(part)}: {interventions.KINDS[k][0]} ({note})")

    def watch_changes(self, scan, diff, kind):
        """Leave changes to the scans after this one (ChangeWatch); logged with no kind yet, so a session that
        stops before they settle doesn't count them as `kind`."""
        if interventions.changed(diff):
            self.changes.watch(scan, diff, kind)
            self.session.event("scan_review", scan=scan["scan"], proposed=kind, kinds=[], changes=diff, by="scans",
                               note="watching: the next scans settle it")

    def log_settled(self, settled, by="scans", note=None):
        """Watched changes the latest scan settled: undone (a glitch) or still there SETTLE_SCANS later (real)."""
        for p, kind in settled:
            change = interventions.merge({p["category"]: [p["change"]]})
            why = note or ("put back by hand" if p.get("put_back") else None) or {
                "scan_glitch": "undone by a later scan", "unsettled": "the session ended first"}.get(
                kind, f"still so {p['agreed']} scans later")
            self.session.event("scan_review", scan=p["scan"], proposed=p["kind"], kinds=[kind], changes=change,
                               by=by, note=why)
            if kind not in ("scan_glitch", "unsettled"):
                print(f"  {interventions.changes_text(change)} (scan {p['scan']}), {why}: "
                      f"{interventions.KINDS[kind][0]}")

    def settle_handled(self, tile):
        """Unattended: a watched change at the tile about to be picked up is settled now. Its return aims at the tile's
        home, so a knocked tile would go back there, and the next scans would call the knock a glitch."""
        near = self.changes.forget(lambda p: not interventions.near_any(p["change"], [tile["xy"]],
                                                                        config.TILE_MERGE_RADIUS_M))
        self.log_settled([(p, p["kind"]) for p in near], note="the arm picked that tile up next, before the scans settled it")

    def explain(self, scan_n, explained, note):
        """Changes of scan `scan_n` whose cause is known (explained(category, change)): neither asked nor watched."""
        dropped = self.changes.forget(lambda p: not (p["scan"] == scan_n and explained(p["category"], p["change"])))
        for i, (scan, (kind, diff)) in reversed(list(enumerate(self.scan_checks))):
            if scan["scan"] == scan_n:
                rest, _ = interventions.split(diff, lambda k, c: not explained(k, c))
                if interventions.changed(rest):
                    self.scan_checks[i] = (scan, (kind, rest))
                else:
                    del self.scan_checks[i]
                    dropped.append(None)
        if dropped:
            self.session.event("scan_review", scan=scan_n, proposed=None, kinds=[], note=note)

    def settle_gone(self, scan_n, tile):
        """An earlier scan's "gone" tile the wrist camera saw isn't there: real, without waiting on more scans."""
        for p in self.changes.forget(lambda p: not (p["scan"] == scan_n and p["category"] == "gone"
                                                    and p["change"] == tile)):
            self.log_settled([(p, p["kind"])], by="wrist camera", note="the wrist camera saw the tile gone")

    def drop_change(self, scan_n, tile, why):
        """An earlier scan's "gone" tile turned out to be there: a glitch, no longer watched."""
        for p in self.changes.forget(lambda p: not (p["scan"] == scan_n and p["category"] == "gone"
                                                    and p["change"] == tile)):
            self.session.event("scan_review", scan=scan_n, proposed=p["kind"], kinds=["scan_glitch"],
                               changes=interventions.merge({"gone": [tile]}), by="scans", note=why)

    def crowded(self, tile, tiles, scan_n):
        """Another tile the scan sees within wrist_check.CROWDED_M of where `tile` was: its face would fill the
        look just as well, so don't look (a person is asked). Logged as a `wrist_check` that couldn't look."""
        near = wrist_check.crowded(tile["xy"], [t for t in tiles if tuple(t["xy"]) != tuple(tile["xy"])])
        if near is None:
            return False
        why = f"{near['tile_name']} is {math.dist(near['xy'], tile['xy']) * 1000:.0f}mm from its spot"
        print(f"  {tile['tile_name']} not seen where it was, but {why}: the wrist camera can't tell them apart")
        self.session.event("wrist_check", tile=tile["tile_name"], xy=list(tile["xy"]), scan=scan_n,
                           at_scan=self.scan_n, present=None, fill=None, why=why, policy=self.policy.id)
        return True

    def faint_at(self, tile, scan_n, radius_m):
        """The latest scan's faint box (perception.faint_tiles) within radius_m of a tile it lost: the tile's there, the
        detector just couldn't name it. Logged as `tile_faint`. The box, or None."""
        near = [f for f in self.realsense.faint if math.dist(f["xy"], tile["xy"]) <= radius_m]
        if not near:
            return None
        box = min(near, key=lambda f: math.dist(f["xy"], tile["xy"]))
        print(f"  {tile['tile_name']} not named at ({tile['xy'][0]:.3f}, {tile['xy'][1]:+.3f}), but a box is still there "
              f"({box['confidence']:.2f})")
        self.session.event("tile_faint", tile=tile["tile_name"], xy=list(tile["xy"]), scan=scan_n, at_scan=self.scan_n,
                           confidence=box["confidence"], box_xy=list(box["xy"]), guess=box["tile_name"])
        return box

    def ask_still_there(self, name, x, y):
        """A look that can't tell (a pale surface and nothing to compare with, or a back it can't make out): a person's
        look. True / False."""
        drain_stdin()
        answer = input(f"  {name} not seen at ({x:.3f}, {y:+.3f}), and the wrist camera can't tell. Is a "
                       "tile still there? [Enter = yes / n = no]: ")
        return not answer.strip().lower().startswith("n")

    def wrist_look(self, tile, scan_n):
        """Hover over a tile the scans lost, where they last saw it (hover-corrected, this policy's hover), and see
        whether a face is between the jaw tips (wrist_check.check; a back by its shape). Logged as a `wrist_check` event, the
        frame in wrist_checks/. Returns the check: present True / False, or None if it couldn't look."""
        name, (x, y) = tile["tile_name"], tile["xy"]
        print(f"  {name} not seen at ({x:.3f}, {y:+.3f}): looking with the wrist camera")
        # the sim's tiles are where they're detected: the real arm's correction would look ~50mm off them
        hx, hy = (x, y) if self.args.sim else self.correction.apply(x, y, log=lambda *a: None)
        template = self.views.template((x, y))
        back = self.args.face_down or perception.is_back(tile)    # face down: a back, whatever a scan named it
        reader = self.back_reader if back else None
        result = wrist_check.check(self.arm, self.camera, hx, hy, self.policy.hover, template, back=back, reader=reader)
        if result["present"] is None and result["why"] == "no wrist frames" and not self.args.sim:
            self.reopen_camera()
            result = wrist_check.check(self.arm, self.camera, hx, hy, self.policy.hover, template, back=back, reader=reader)
        if result["present"] is None and (result.get("pale") or back and result.get("frames")) and not self.args.unattended:
            self.arm.lift()
            result = {**result, "present": self.ask_still_there(name, x, y), "by": "person"}
            result.pop("why", None)
        frame, image = result.pop("frame"), None
        if result["present"]:
            # a place's picture is taken once a person has seen it, or on a surface the look can judge; a same-view look
            # keeps it, so it can't drift off the tile a little at a time
            if result.get("by") not in ("same view", "back shape", "back model") and frame is not None:
                self.views.keep((x, y), wrist_check.view_template(frame))
        elif result["present"] is False:
            self.views.forget((x, y))
        if frame is not None:
            folder = self.session.dir / "wrist_checks"
            folder.mkdir(exist_ok=True)
            where = f"_{x * 1000:.0f}_{y * 1000:+.0f}" if back else ""    # every back shares a name
            image = f"wrist_checks/scan{scan_n:03d}_{self.scan_n:03d}_{name}{where}.jpg"
            cv2.imwrite(str(self.session.dir / image), frame)
        if result["present"] is False:
            self.looked_empty.append((self.scan_n, [x, y]))
        verdict = {True: "still there (a detection glitch)", False: "not there",
                   None: f"couldn't look ({result.get('why')})"}[result["present"]]
        print(f"  {name}: {verdict}" + (f", face fills {result['fill']:.0%} of the spot" if result["fill"] is not None else
                                        f", a back in {result['backs_seen']} of {result['frames']} frames" if "backs_seen" in result
                                        else f", the back check sure on all {result['frames']} frames"
                                        if result.get("by") == "back model" else ""))
        self.session.event("wrist_check", tile=name, xy=[x, y], scan=scan_n, at_scan=self.scan_n, image=image,
                           policy=self.policy.id, **result)
        return result

    def look_at_home(self, tile_name=None, spot=None, tiles=()):
        """A tile not detected at its spot (or by its name), looked at with the wrist camera where it was first
        seen this session (a spot's layout point if never picked up): the tile there if it is, else None. Not
        when the scan's `tiles` have another one that close (crowded)."""
        key = spot["id"] if spot else tile_name
        home = self.homes.get(key) if key else None
        home = home or (spot["xy"] if spot else None)
        if home is None or any(n == self.scan_n and math.dist(xy, home) <= config.TILE_MERGE_RADIUS_M
                               for n, xy in self.looked_empty):
            return None
        name = config.BACK if self.args.face_down else None     # green side up: no name to read, a back to look for
        tile = {"tile_name": name or tile_name or self.placements.truth_at(key) or self.names.get(key) or config.UNNAMED,
                "xy": list(home)}
        box = self.faint_at(tile, self.scan_n, config.TILE_MERGE_RADIUS_M)
        if box:                     # picked up from where the box is, as a named tile would be
            self.mat.restore(tile)
            return {"tile_name": tile["tile_name"], "xy": tuple(box["xy"]), "confidence": 0.0, "faint": True}
        if self.crowded(tile, tiles, self.scan_n):
            return None
        result = self.wrist_look(tile, self.scan_n)
        self.arm.rest()
        if not result["present"]:
            return None
        self.mat.restore(tile)
        return {"tile_name": tile["tile_name"], "xy": tuple(home), "confidence": 0.0, "wrist_seen": True}

    def check_scans(self):
        """Neighbours the scans saw move or turn with no b pressed, one question each (Enter: as they say; or a
        detection glitch)."""
        if self.scan_checks:
            drain_stdin()
        for scan, proposal in self.scan_checks:
            kinds = interventions.ask_scan(scan, proposal)
            if kinds == [interventions.BY_HAND]:
                self.log_by_hand(scan, proposal[0], proposal[1], "person", "answered: moved by hand")
                continue
            self.session.event("scan_review", scan=scan["scan"], proposed=proposal[0], kinds=kinds,
                               changes=proposal[1], by="person")
        self.scan_checks = []

    def choose(self, tile_name=None, spot=None):
        """The tile named `tile_name`, or detected nearest layout `spot`, or one you pick. In trials, a tile
        that isn't found is repositioned by a person and the mat rescanned (s skips it): a flickering
        detection (7D read as 6D, 2026-10-29) otherwise skipped the spot or stopped the run.
        --face-down: the back the scan finds at the spot, or with weights that can't see backs, the spot's saved
        position (choose_saved)."""
        if self.args.face_down and not self.sees_backs:
            return self.choose_saved(spot)
        confirmed = None
        while True:
            if not self.arm.is_at_rest():
                self.arm.rest()
            tiles = self.scan("choose", confirmed=confirmed)
            if not self.defer:
                self.check_scans()          # before the arm moves, while what happened is in front of you
            tile, reason, missing = self.find(tiles, tile_name, spot)
            if tile is None and reason != "cancelled":
                tile = self.look_at_home(tile_name, spot, tiles)
            if tile is not None or reason == "cancelled" or not (self.batch and self.reposition(missing, tile_name, spot)):
                break
            confirmed = ["tile_reset"]
        if tile is None:
            self.session.event("attempt_skipped", reason=reason, tile=tile_name,
                               **({"spot": spot["id"], "spot_xy": spot["xy"]} if spot else {}))
        return tile

    def choose_saved(self, spot):
        """--face-down: the tile at `spot`'s saved position, as record.py --face-down and the face-down scripts aim (the
        detector can't see a green back, so a scan can't find it or say it's gone). The mat is still scanned first, so a
        face-up tile near the arm that moves is caught as on any bench. None (skipped) without a spot."""
        if not self.arm.is_at_rest():
            self.arm.rest()
        self.scan("choose")
        if not self.defer:
            self.check_scans()
        if spot is None:
            self.session.event("attempt_skipped", reason="cancelled", tile=layout.FACE_DOWN)
            return None
        return layout.saved_target(spot)

    def ask_spot(self):
        """--face-down at the menu: which layout spot to pick from (a face-down tile can't be chosen from a scan). None: cancel."""
        ids = [s["id"] for s in self.spots]
        drain_stdin()
        while True:
            answer = input(f"Which spot ({', '.join(ids)}; Enter cancels)? ").strip().upper()
            if not answer:
                return None
            spot = next((s for s in self.spots if s["id"].upper() == answer), None)
            if spot is not None:
                return spot
            print(f"Layout {self.args.layout} has no spot {answer}.")

    def find(self, tiles, tile_name=None, spot=None):
        """(tile, None) or (None, skip reason, what was missing) from one scan's tiles."""
        if not tiles:
            print("No tiles detected.")
            return None, "no_tiles", "No tiles detected"
        if spot is not None:
            tile = layout.nearest_tile(tiles, spot["xy"])
            missing = f"No tile detected within {config.TILE_MERGE_RADIUS_M * 1000:.0f}mm of spot {spot['id']}"
            if tile is None:
                print(missing)
            return tile, "not_at_spot", missing
        if tile_name is None:
            drain_stdin()
        tile = choose_tile(tiles, tile_name=tile_name)
        if tile is None and tile_name is not None:
            missing = f"{tile_name} not detected"
            print(f"{missing} (seen: {', '.join(t['tile_name'] for t in tiles)})")
            return None, "not_detected", missing
        return tile, "cancelled", None

    def reposition(self, missing, tile_name=None, spot=None):
        """Ask a person to put the missing tile where it belongs; True if they did (rescan), False to skip."""
        if self.args.unattended:
            print(f"  {missing}. No one here to put it back: skipping it.")
            return False
        home = spot["xy"] if spot else self.homes.get(tile_name)
        where = f" at ({home[0]:.3f}, {home[1]:+.3f})" if home else ""
        name = tile_name or "the tile"
        drain_stdin()
        answer = input(f"  {missing}. Put {name}{where} squarely, then Enter to rescan (type its name first if it's "
                       f"a different tile), or s to skip: ").strip().upper()
        if answer == "S" or answer == "SKIP":
            return False
        if spot:
            known = answer if answer in detector.TILE_NAMES else self.placements.truth_at(spot["id"])
            self.placements.start(spot["id"], spot["xy"], truth=known, truth_source="person" if known else "unconfirmed")
        self.session.intervention("tile_reset", attempt_dir=self.last_attempt, tile=tile_name,
                                  reason="repositioned: not detected", **({"spot": spot["id"]} if spot else {}))
        return True

    def hover_over(self, tile, perturb=(0.0, 0.0)):
        x, y = tile["xy"]
        hx, hy = self.correction.apply(x, y)
        if any(perturb):
            hx, hy = hx + perturb[0], hy + perturb[1]
            print(f"  hover perturbed by ({perturb[0] * 1000:+.0f}, {perturb[1] * 1000:+.0f})mm on purpose")
        h = self.hover_of(self.policy)
        result = self.arm.hover(hx, hy, h["offset_m"], h["clearance_m"], h["gripper"])
        if result is None:
            print("Unreachable from the closed-loop hover.")
            self.session.event("attempt_skipped", reason="unreachable", tile=tile["tile_name"], hover_xy=[hx, hy])
        return result

    def at_hover(self, spot, attempt):
        """The arm is over the tile and the policy hasn't started: False calls the pickup off (play's name check)."""
        return True

    def snapshot(self, path):
        cv2.imwrite(str(path), self.realsense.snapshot())

    # --- commands ---

    def hover(self):
        if self.held:
            return print(f"Holding {self.held['tile_name']} — return or discard it first.")
        spot = self.ask_spot() if self.args.face_down else None
        tile = self.choose(spot=spot)
        if tile and (result := self.hover_over(tile)):
            self.session.event("hover", tile=tile["tile_name"], detection_xy=tile["xy"],
                               attempt_dir=self.hover_photo(tile["tile_name"]), **result)

    def hover_photo(self, tile_name):
        """Save a current wrist frame from the hover, under the name a pickup's first frame has (policy.run), so
        mahjong.wrist_photos gathers both. Returns its folder's name, or None if the camera gave no frame."""
        frames = grab_frames(self.camera, 1)
        if not frames:
            self.reopen_camera()
            frames = grab_frames(self.camera, 1)
        if not frames:
            print("No wrist camera frame: no photo saved for this hover.")
            return None
        folder = self.session.attempt_dir(tile_name)
        cv2.imwrite(str(folder / "wrist_0000.jpg"), frames[0])
        print(f"Wrist photo: {folder.name}/wrist_0000.jpg")
        return folder.name

    def pickup(self, tile_name=None, ask=True, perturb=(0.0, 0.0), spot=None, camera_retries=CAMERA_RERUNS):
        """One attempt with self.policy, from its hover moved by `perturb` (m), at the named tile or the
        one at layout `spot`. Returns True if the gripper ended up holding the tile. An attempt the wrist
        camera dropped out of is void (not an attempt event): the camera is reopened and it runs again."""
        if self.held:
            print(f"Holding {self.held['tile_name']} — return or discard it first.")
            return None
        if self.args.face_down and spot is None:        # the menu's pickup: a spot, so the tile has a home of its own
            spot = self.ask_spot()
            if spot is not None:
                self.lay_face_down([spot])
        tile = self.choose(tile_name, spot)
        if tile is None:
            return None
        if self.args.unattended:
            self.settle_handled(tile)
        attempt = self.session.attempt_dir(tile["tile_name"])
        self.snapshot(attempt / "overview_before.jpg")
        self.keys.start("pickup")
        hover = self.hover_over(tile, perturb)
        if hover is None:
            self.keys.stop()
            return None
        if not self.at_hover(spot, attempt):
            self.keys.stop()
            self.arm.rest()
            return None
        self.abort.clear()
        run = self.policy.run(self.arm, self.camera, attempt, abort=self.abort, mark=self.mark,
                              **({"takeover": self.takeover} if self.takeover else {}))
        squeezed_from = self.arm.squeeze()     # a policy that stopped closing short of 0 on the tile
        if not (self.direct and not run["aborted"] and not run["marks_s"] and self.arm.holding()[0] and self.arm.lift_clear()):
            self.arm.rest()
        presses = self.keys.stop()
        holding, command, measured = self.arm.holding()
        if presses or not holding or run.get("takeover_step") is not None:    # something happened, or a miss: look at the mat now
            self.arm.rest()
        if self.takeover:               # a recovery a person showed is saved (and logged as a rescue) before anything is asked
            self.takeover.save(run, self.session, self.policy.id, tile_name=tile["tile_name"], detection_xy=list(tile["xy"]),
                               hover=self.hover_of(self.policy), hover_xy=hover["hover_xy"],
                               hover_clearance_m=hover["clearance_m"], hover_jitter_xy=list(perturb),
                               hover_correction=self.correction.name, attempt_dir=attempt.name,
                               **({"layout": self.args.layout, "spot": spot["id"], "spot_xy": spot["xy"]} if spot else {}),
                               **({"face_down": True} if self.args.face_down else {}))
        if run["stop_reason"] == "camera" and not holding:
            torn = run.get("camera_problem") == "torn"
            self.session.event("attempt_void", reason=VOID_TORN if torn else VOID_LOST, tile=tile["tile_name"],
                               policy=self.policy.id, attempt_dir=attempt.name, hover_perturb_xy=list(perturb),
                               **({"spot": spot["id"]} if spot else {}))
            self.reopen_camera()
            if camera_retries:
                print("Wrist camera reopened — running the attempt again.")
                return self.pickup(tile_name, ask, perturb, spot, camera_retries - 1)
            print("The wrist camera " + ("gave torn frames" if torn else "dropped out") + " again: skipping this attempt.")
            return None
        if self.arm.is_at_rest():   # else the scan after it's set down: a frame with the arm out shows it across the tiles
            self.snapshot(attempt / "overview_after.jpg")
        self.names[spot["id"] if spot else tile["tile_name"]] = tile["tile_name"]
        self.log_keys(attempt.name, "pickup", presses)
        if self.arm.is_at_rest():
            self.scan("after_pickup", look=not holding, attempt_dir=attempt.name, target_xy=list(tile["xy"]))
        else:
            self.unseen = {"xy": list(tile["xy"]), "attempt_dir": attempt.name}
        success, review = holding, {}
        if ask:
            drain_stdin()
            answer = input(f"Picked it up? (gripper commanded {command:.0f}, measured {measured:.0f}: looks "
                           f"{'held' if holding else 'empty'}) [{'Y/n' if holding else 'y/N'}]: ").strip().lower()
            success = holding if not answer else answer.startswith("y")
        if self.defer and (run["aborted"] or (run["marks_s"] and not presses)):
            self.to_review.append((attempt.name, tile["tile_name"], run, bool(presses)))
        elif not self.defer:
            review = self.review(attempt.name, tile["tile_name"], run, keyed=bool(presses))
        name_source = "faint" if tile.get("faint") else "wrist_seen" if tile.get("wrist_seen") else \
            "saved_position" if tile.get("saved_position") else "read"
        self.session.event("attempt", **play_log.pickup_fields(
                               policy=self.policy.id, attempt_dir=attempt.name, tile=tile["tile_name"], hover=hover,
                               offset_m=hover["offset_m"], holding=holding, command=command, measured=measured, success=success,
                               correction=self.correction.name, layout=self.args.layout, spot=spot, squeezed_from=squeezed_from),
                           name_source=name_source, placement=self.placements.id_at(spot["id"]) if spot else None,
                           detection_xy=tile["xy"], hover_perturb_xy=list(perturb), hover_sag_xyz_m=hover.get("sag_xyz_m"),
                           hover_gripper=self.hover_of(self.policy)["gripper"], **takeover.policy_view(run), **review)
        self.last_attempt = attempt.name
        # a lowered hover says nothing about the usual one; a saved position isn't a detection, which the correction is fitted from
        # ...and a person's grasp after a takeover isn't the policy's
        if success and run["grasp_xy"] and not self.args.hover_lower_mm and not tile.get("saved_position") \
                and run.get("takeover_step") is None:
            sample = grasp_record("bench", tile["xy"], run["grasp_xy"], tile_name=tile["tile_name"],
                                  hover_xy=hover["hover_xy"], hover_perturb_xy=list(perturb), policy=self.policy.id,
                                  correction=self.correction.name, session=self.session.dir.name,
                                  attempt_dir=attempt.name, **back_excluded(tile, self.backs_checked))
            append_record(sample, self.samples_path)
        if holding:
            self.held = {"tile_name": tile["tile_name"], "detection_xy": list(tile["xy"]),
                         "home_xy": self.homes(tile, spot), "grasp_xy": run["grasp_xy"] or hover["hover_xy"],
                         "approach_offset_m": hover["offset_m"], "attempt_dir": attempt.name,
                         "grasp_clearance_m": run.get("grasp_clearance_m"),
                         "spot": spot["id"] if spot else None, "saved_position": bool(tile.get("saved_position"))}
        print(f"{tile['tile_name']}: {'picked up' if success else 'missed'}")
        if not self.defer:
            self.check_scans()
        return success

    def log_keys(self, attempt_dir, during, presses):
        """The keys pressed during a pickup or a return (KeyMarks), as a `keys` event; what that phase's scan is
        judged by (sort_changes)."""
        self.last_keys = presses
        if presses:
            self.session.event("keys", attempt_dir=attempt_dir, during=during, presses=presses)

    def reopen_camera(self):
        """Release and reopen the wrist camera (hardware.reopen_wrist_camera: retry / quit if it won't open; an
        unattended run stops instead of waiting on the retry prompt)."""
        from mahjong import hardware
        if not self.args.unattended:
            self.camera = hardware.reopen_wrist_camera(self.camera)
            return
        try:
            self.camera = hardware.reopen_wrist_camera(self.camera, ask=lambda prompt: "q")
        except SystemExit:
            raise SystemExit("The wrist camera wouldn't reopen: stopping the unattended run.")

    def review(self, attempt_dir, tile_name, run, keyed=False):
        """Ask about an aborted or marked attempt: {abort_reason, mark_kind, mark_note}; {} (no question)
        for a clean one, or a → when a key during it already said what happened (`keyed`)."""
        if self.args.unattended:    # a key someone pressed in passing still counts; there's no one to ask (derive lists it)
            return {}
        answers, what = {}, f"{attempt_dir} ({tile_name})"
        marked = run["marks_s"] and not keyed
        if run["aborted"] or marked:
            drain_stdin()
        if run["aborted"]:
            answers["abort_reason"] = interventions.ask_abort(what)
        if marked:
            when = ", ".join(f"{t:.1f}s" for t in run["marks_s"])
            answers["mark_kind"], answers["mark_note"] = interventions.ask_mark(f"{what}, marked at {when}")
        return {k: v for k, v in answers.items() if v}

    def return_tile(self):
        """Set the held tile down so it lands at its home, then check where it landed (check_landing)."""
        if not self.held:
            return print("Not holding a tile.")
        held = self.held
        home = held["home_xy"]
        print(f"Returning {held['tile_name']} home, to ({home[0]:.3f}, {home[1]:+.3f}) as the RealSense sees it")
        x, y = self.returns.target(home, held["detection_xy"], held["grasp_xy"])
        self.keys.start("return")
        try:
            placed = self.arm.place(x, y, held["approach_offset_m"], grasp_clearance_m=held.get("grasp_clearance_m"),
                                    direct=self.direct)
        finally:
            presses = self.keys.stop()
        self.log_keys(held["attempt_dir"], "return", presses)
        held["by_hand"] = any(p["kind"] == "by_hand" for p in presses)
        if placed is None:
            print("Still holding it — discard it instead.")
            self.session.event("return_skipped", reason="unreachable", tile=held["tile_name"], xy=[x, y],
                               attempt_dir=held["attempt_dir"])
            return
        self.session.event("return", tile=held["tile_name"], asked_xy=[x, y], home_xy=home,
                           grasp_clearance_m=held.get("grasp_clearance_m"),
                           return_correction=self.returns.name, attempt_dir=held["attempt_dir"], **placed)
        # the returned tile, where the camera saw it (the arm's target is in its own frame, ~40mm from that);
        # a change near its home is the return itself, not a knock
        self.held = None
        tiles = self.scan("after_return", attempt_dir=held["attempt_dir"], target_xy=home)
        if not held.get("saved_position"):     # a green back can't be seen landing: no sample, no reset (a person squares it, r)
            self.check_landing(held, placed, tiles, self.scan_n)
        if not self.defer:
            self.check_scans()

    def check_landing(self, held, placed, tiles, scan_n):
        """Where the returned tile landed (return_correction.check_landing: a sample for the return model, a
        warning past RETURN_DRIFT_WARN_M from home), and a person putting it back past RETURN_DRIFT_RESET_M
        or out of the hover's reach (the next pickup would start from there)."""
        home, name = held["home_xy"], held["tile_name"]
        tile, reset = check_landing(held, placed["xy"], tiles, self.session, self.returns_path, self.correction,
                                    self.hover_of(self.policy), policy=self.policy.id,
                                    return_correction=self.returns.name, **back_excluded(held, self.backs_checked))
        if not reset:
            return
        # landing this far off, the returned tile showed in the after-return scan as one that appeared: that
        # was the landing itself, not a knock (anything else the scan saw change is still settled as usual)
        self.explain(scan_n, lambda k, c: k == "appeared" and list(c["xy"]) == list(tile["xy"]),
                     "the returned tile landing off its home")
        if self.args.unattended:    # picked up from where it landed if the next hover can find it, else its spot is skipped
            print(f"  No one here to put {name} back: leaving it there.")
            self.session.event("reset_skipped", tile=name, attempt_dir=held["attempt_dir"], home_xy=home,
                               landed_xy=list(tile["xy"]))
            return
        drain_stdin()
        input(f"  Put {name} back on its home, ({home[0]:.3f}, {home[1]:+.3f}), then Enter: ")
        # what the next scan sees move was a person putting it back: say so instead of asking again
        self.scan("after_reset", confirmed=["tile_reset"], attempt_dir=held["attempt_dir"])

    def discard(self, ask=True):
        pose = config.discard_pose()
        if pose is None:
            return print("No discard pose yet — use teach-discard first.")
        held = self.held or {}
        still_held = self.arm.discard(pose, direct=self.direct)
        self.session.event("discard", tile=held.get("tile_name"), attempt_dir=held.get("attempt_dir"), held_on_arrival=still_held)
        if held and ask:
            drain_stdin()
            if input("In the container? [Y/n]: ").strip().lower().startswith("n"):
                self.session.intervention("discard_missed", attempt_dir=held["attempt_dir"])
                self.hands_on = "the discard missed the container"
        self.held = None
        self.scan("after_discard", attempt_dir=held.get("attempt_dir"))

    def teach_discard(self):
        from mahjong import hardware
        try:
            gamepad = self.takeover.gamepad if self.takeover else \
                hardware.connect(["gamepad"], gamepad_kwargs={"device": self.args.device, "commit_s": 0.0})["gamepad"]
        except SystemExit:
            return print("teach-discard cancelled")
        self.arm.raise_from_rest()
        gamepad.start(self.arm)
        warn_off_centre(gamepad, self.session, None)
        self.abort.clear()
        print("Drive over the discard container's opening, then → to save (Esc cancels), in this terminal.")
        self.terminal.start()
        try:
            while not (self.events["exit_early"] or self.events["stop_recording"]):
                t0 = time.time()
                gamepad.step(self.arm, 1.0 / config.FPS)
                time.sleep(max(0.0, 1.0 / config.FPS - (time.time() - t0)))
        finally:
            self.terminal.stop()
        if not self.takeover:           # the takeover's gamepad stays connected for the pickups
            gamepad.close()
        if not self.events["stop_recording"]:
            pose = {j: self.arm.commanded()[j] for j in config.ARM_JOINTS}
            config.DISCARD_POSE.write_text(json.dumps({"pose": pose, "session": self.session.dir.name}, indent=2) + "\n")
            self.session.event("discard_pose_taught", pose=pose)
            print(f"Saved {config.DISCARD_POSE}")
        self.arm.rest()

    def name_the_mat(self, spots=None):
        """Before trials on a layout, once: a person checks what the scan reads at each spot, so every read after this
        can be scored against what's really there (mahjong/placements.py, scripts/name_report.py). An unattended run asks
        this too, like the setup questions: nothing is asked once the trials start."""
        if self.args.face_down:     # nothing to read on a green back
            return self.lay_face_down(spots or self.spots)
        todo = [s for s in spots or self.spots or [] if not self.placements.id_at(s["id"])]
        if not todo:
            return
        if not self.arm.is_at_rest():
            self.arm.rest()
        checked = None
        while checked is None:
            rows = placements.rows(todo, self.scan("name_check", look=False))
            if self.ask_names is None:
                checked = [{**row, "sim": True} for row in rows]
            else:
                drain_stdin()
                checked = placements.confirm(rows, self.ask_names)
        for row in checked:
            source = "sim" if row.get("sim") else "person" if row["confirmed"] or "renamed_from" in row else "unconfirmed"
            named = row["tile_name"] != config.UNNAMED and source != "unconfirmed"
            self.placements.start(row["id"], row["xy"], truth=row["tile_name"] if named else None, truth_source=source,
                                  read=row["detected_as"])

    def lay_face_down(self, spots):
        """--face-down, once per spot a session: a person lays a tile GREEN side up on each, square to its outline (the
        RealSense can't check it). Logged as `face_down_laid`. The sim's tiles are already there."""
        todo = [s["id"] for s in spots if s["id"] not in self.laid]
        if not todo:
            return
        if self.ask_names is not None:
            drain_stdin()
            input(f"Lay a tile GREEN side up on {', '.join(todo)}, square to the layout (scripts/layout.py shows the outlines), "
                  "then Enter: ")
        self.laid.update(todo)
        self.session.event("face_down_laid", spots=todo)
        if self.sees_backs:         # the scan can check it: a spot with no back near it is said now, not found mid-trials
            if not self.arm.is_at_rest():
                self.arm.rest()
            tiles = self.scan("face_down_check", look=False)
            unseen = [s["id"] for s in spots if s["id"] in todo
                      and layout.nearest_tile([t for t in tiles if perception.is_back(t)], s["xy"]) is None]
            self.session.event("face_down_seen", spots=[i for i in todo if i not in unseen], unseen=unseen)
            if unseen:
                print(f"  No back seen on {', '.join(unseen)}: "
                      + ("it'll be skipped." if self.args.unattended else "squared up when it's reached, or skipped."))

    @property
    def trial_spots(self):
        return self.pick_spots()

    def pick_spots(self):
        """The layout's spots the trials visit: --spots in the order given, --start-spot to the last one, else all."""
        args = self.args
        if not self.spots:
            return self.spots
        by_id = {s["id"].upper(): s for s in self.spots}
        def known(ids, flag):
            missing = [i for i in ids if i.upper() not in by_id]
            if missing:
                raise SystemExit(f"{flag}: {args.layout} has no spot {', '.join(missing)} (it has {', '.join(s['id'] for s in self.spots)})")
        if args.spots:
            known(args.spots, "--spots")
            if len({i.upper() for i in args.spots}) != len(args.spots):
                raise SystemExit("--spots: a spot is listed twice")
            return [by_id[i.upper()] for i in args.spots]
        if args.start_spot:
            known([args.start_spot], "--start-spot")
            first = [s["id"].upper() for s in self.spots].index(args.start_spot.upper())
            return self.spots[first:]
        return self.spots

    def trials(self, n=None, tile_name=None):
        """n rounds on one tile, or at every spot of the layout. A round is one attempt per policy, from
        the same hover offset, in alternating order (a b, b a, ...)."""
        if n is None:
            drain_stdin()
            n = int(input(f"How many rounds{' per spot' if self.spots else ''}? ").strip() or 0)
            if not self.spots and not self.args.tiles:
                tile_name = input("Which tile (name)? ").strip() or None
        if self.args.carry:
            return self.carry(n)
        self.name_the_mat(self.trial_spots)
        offsets = self.args.hover_perturb_mm or [(0.0, 0.0)]
        # what each stretch of rounds aims at: (spot, tile name, label for the report). --tiles: one stretch per tile
        # (labelled with the tile); a layout: one per spot; else the one tile (label None)
        targets = ([(None, t, t) for t in self.args.tiles] if self.args.tiles
                   else [(s, tile_name, s["id"]) for s in self.trial_spots] if self.spots else [(None, tile_name, None)])
        plan = [(target, i, policy) for target in targets for i in range(n)
                for policy in (self.policies if i % 2 == 0 else self.policies[::-1])]
        done, skipped = [], set()       # done: (label, policy id, success)
        self.batch = True
        try:
            for (spot, aim, where), i, policy in plan:
                if where in skipped:
                    continue
                self.policy = policy
                print(f"\n=== {f'spot {where}, ' if spot else f'{where}, ' if where else ''}round {i + 1}/{n}"
                      f"{f', {policy.id}' if len(self.policies) > 1 else ''} ===")
                success = self.pickup(aim, ask=False, perturb=offsets[i % len(offsets)], spot=spot)
                if success is None and where is not None:
                    print(f"Couldn't attempt {f'at spot {where}' if spot else where} — on to the next {'spot' if spot else 'tile'}.")
                    skipped.add(where)
                    continue
                if success is None:
                    print("Couldn't attempt (tile not found or unreachable) — stopping the trials.")
                    break
                done.append((where, policy.id, success))
                if self.held:
                    self.return_tile()
                if self.held:
                    print("Couldn't return it — stopping the trials.")
                    break
        finally:        # even when interrupted: what you marked is still worth a word
            self.policy = self.policies[0]
            self.session.event("trials_done", tile=tile_name, tiles=self.args.tiles, policy=self.policy.id,
                               results=[d[2] for d in done], policies=[d[1] for d in done],
                               spots=[d[0] if self.spots else None for d in done],
                               layout=self.args.layout, skipped_spots=sorted(skipped) if self.spots else [],
                               skipped_tiles=[] if self.spots else sorted(skipped))
            print()
            for where in dict.fromkeys(d[0] for d in done):
                for policy in self.policies:
                    got = [d[2] for d in done if d[:2] == (where, policy.id)]
                    if got:
                        print(f"{where or tile_name or 'trials'}{f' {policy.id}' if len(self.policies) > 1 else ''}: "
                              f"{sum(got)}/{len(got)} picked up")
            if skipped:
                print(f"skipped {'spots' if self.spots else 'tiles'}: {', '.join(sorted(skipped))}")
            self.batch = False
            self.review_queued()
            self.check_scans()
            self.print_flaky()

    def carry(self, n):
        """n rounds of what a hand's turn does and a walk of the layout doesn't: carry one tile from a spot to another,
        across the mat, and back. Each carry is aimed and its landing checked as a return (return_tile); TO starts empty."""
        start, end = (next(s for s in self.spots if s["id"] == c) for c in self.args.carry)
        self.name_the_mat([start])
        self.homes.homes[end["id"]] = [float(v) for v in end["xy"]]
        done = []
        self.batch = True
        try:
            for i in range(n):
                for frm, to in ((start, end), (end, start)):
                    print(f"\n=== round {i + 1}/{n}: {frm['id']} to {to['id']} ===")
                    success = self.pickup(ask=False, spot=frm)
                    if success is not None:
                        done.append((frm["id"], self.policy.id, success))
                    if not success or not self.held:
                        return print(f"Nothing picked up at {frm['id']} — stopping the carries.")
                    self.held.update(home_xy=self.homes.get(to["id"]), spot=to["id"])
                    name = self.held["tile_name"]
                    self.return_tile()
                    if self.held:
                        return print("Couldn't set it down — stopping the carries.")
                    moved = self.placements.end(frm["id"], "moved")
                    self.placements.start(to["id"], self.homes.get(to["id"]), truth_source="game", read=name,
                                          from_placement=moved)
        finally:
            self.session.event("trials_done", tile=None, policy=self.policy.id, results=[d[2] for d in done],
                               policies=[d[1] for d in done], spots=[d[0] for d in done], layout=self.args.layout,
                               skipped_spots=[], carry=self.args.carry)
            self.batch = False
            self.review_queued()
            self.check_scans()
            self.print_flaky()

    def print_flaky(self):
        """The tiles whose detections a later scan undid this session (ChangeWatch.glitches)."""
        if self.changes.glitches:
            print("Detection glitches settled by later scans: "
                  + ", ".join(f"{n} x{c}" for n, c in self.changes.glitches.most_common()))
        if self.changes.pending:
            print(f"{len(self.changes.pending)} change(s) still waiting on the next scans")

    def review_queued(self):
        """The marked or aborted attempts the caller held over (defer), one question each."""
        if self.to_review:
            print(f"\n{len(self.to_review)} attempt(s) to say a word about:")
        for attempt_dir, tile_name, run, keyed in self.to_review:
            answers = self.review(attempt_dir, tile_name, run, keyed)
            if answers:
                self.session.event("attempt_review", attempt_dir=attempt_dir, **answers)
        self.to_review = []

    def note(self):
        """Log something a person had to do that nothing else caught, against the latest attempt."""
        drain_stdin()
        what = f"What did you do{f' (after {self.last_attempt})' if self.last_attempt else ''}"
        kind = interventions.ask_kind(what, interventions.NOTE_KINDS)
        if kind is None:
            return print("Nothing logged.")
        text = input("  in a few words (Enter to skip): ").strip()
        self.session.intervention(kind, attempt_dir=self.last_attempt, note=text)
        if kind in ("straightened", "tile_reset", "discard_missed"):
            self.hands_on = f"noted: {interventions.KINDS[kind][0]}"     # what the next scan sees move
        print(f"Logged: {interventions.KINDS[kind][0]} ({interventions.class_of(kind)})")

    def setup(self):
        """The setup changed mid-session: ask the setup questions again, the current answers as defaults."""
        drain_stdin()
        conditions = ask_conditions(self.session.conditions, confirm=False)
        notes = ask_notes()
        self.session.conditions = conditions
        self.session.event("conditions_changed", conditions=conditions, notes=notes)

    def run_command(self, name, action):
        """One command; Ctrl-C stops it where it is (logged: a person stepped in) and returns to the menu."""
        self.session.event("command", name=name)
        try:
            action()
        except KeyboardInterrupt:
            print("\ninterrupted — holding position")
            self.session.intervention("interrupted", attempt_dir=self.last_attempt, command=name)
        self.check_scans()

    def close(self):
        try:
            self.arm.rest()
            if self.args.unattended and self.scan_n:    # one more look: a bump in the last return still gets a later scan
                self.scan("final", look=False)
        finally:
            self.log_settled(self.changes.flush())
            self.arm.disconnect()
            self.camera.release()
            self.realsense.close()
            self.keys.close()
            if self.takeover:
                self.takeover.close()
            self.session.close()


def parse_policies(text):
    """"a,b" -> ["a", "b"]; None -> [None] (the registry's default)."""
    if not text:
        return [None]
    ids = [s.strip() for s in text.split(",") if s.strip()]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"--policy {text}: each policy once")
    return ids


def parse_names(text):
    """'2d, 6B' -> ['2D', '6B']: a comma-separated list of tile names or spot ids (spot ids match case-insensitively in
    Bench.pick_spots)."""
    names = [n.strip() for n in text.split(",") if n.strip()]
    if not names:
        raise argparse.ArgumentTypeError("needs at least one name")
    return names


COMMANDS = ["hover", "pickup", "return", "discard", "teach-discard", "trials", "rest", "note", "setup", "quit"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--policy", default=None,
                   help="policy id from data/policies.json (default: its 'default'); a,b compares them paired in trials")
    p.add_argument("--trials", type=int, default=None,
                   help="run this many rounds of pickup/return trials, then quit: a round is one attempt per policy, "
                        "at every spot with --layout")
    p.add_argument("--unattended", action="store_true",
                   help="with --trials and --tile or --layout: run with no one watching. Nothing is asked once the trials "
                        "start; neighbours the scans see move are self-reported, for scripts/bump_report.py to verify")
    p.add_argument("--tile", default=None, help="tile name for --trials")
    p.add_argument("--tiles", type=lambda text: [n.upper() for n in parse_names(text)], default=None, metavar="A,B,...",
                   help="with --trials and no --layout: these tiles in turn, --trials rounds on each (tiles to pick by name)")
    p.add_argument("--spots", type=parse_names, default=None, metavar="S1,S3,...",
                   help="with --layout: trials visit only these spots, in the order given (default: every spot)")
    p.add_argument("--start-spot", default=None, metavar="SPOT",
                   help="with --layout: trials start at this spot and go on to the layout's last one (default: its first)")
    p.add_argument("--layout", default=None, help="trials walk this layout's spots (scripts/layout.py)")
    p.add_argument("--sim", action="store_true", help="simulated arm, camera, perception and policy")
    p.add_argument("--device", default=None, help="gamepad evdev path (teach-discard, --dagger-dataset)")
    takeover.add_argument(p)
    p.add_argument("--hover-perturb-mm", type=parse_offsets, default=None,
                   help="hover offsets the trials cycle through, e.g. 0,6x,-6x,6y,-6y (policy sensitivity test)")
    p.add_argument("--carry", type=lambda text: text.split(","), default=None, metavar="FROM,TO",
                   help="trials carry one tile between these two spots of --layout, as a hand does, instead of returning it "
                        "where it came from: a tile on FROM, TO empty. --trials 2 from the drop zone's spot to a slot "
                        "is what play.py needs before a hand")
    p.add_argument("--hover-gripper", type=float, default=None,
                   help="gripper opening at the hover (%%) for every policy, instead of the one its episodes started "
                        "at (data/policies.json; v6 26.7). Its narrowing on the way down shifts by the same amount")
    p.add_argument("--hover-lower-mm", type=float, default=None,
                   help="hover this many mm lower than the policy's own (its offset and clearance both), e.g. 8 at a spot "
                        "where it closes high (v8 at S8, 2026-11-05: 28mm up, missed). An experiment: its attempts save no "
                        "hover samples and the session is no counted result. At most 10 (the arm's own limit)")
    p.add_argument("--face-down", action="store_true",
                   help="with --layout: tiles lie GREEN side up. With detector weights that have the 'back' class, pickups aim "
                        "at the back the scan finds and returns are checked, so --unattended works; with older weights, at each "
                        "spot's saved position, unseen, and someone watching")
    args = p.parse_args(argv)
    if args.tile and args.tiles:
        p.error("--tile and --tiles both name tiles: use --tiles A,B for several")
    if (args.spots or args.start_spot) and not args.layout:
        p.error("--spots and --start-spot choose among a layout's spots: they need --layout")
    if args.spots and args.start_spot:
        p.error("--spots and --start-spot both choose spots: use one (--spots lists them all)")
    if args.tiles and (args.layout or args.spots or args.start_spot):
        p.error("--tiles picks tiles by name, without a layout: with --layout, choose spots with --spots or --start-spot")
    bad = [t for t in args.tiles or [] if t not in detector.TILE_NAMES]
    if bad:
        p.error(f"--tiles: not a tile name: {', '.join(bad)}")
    if args.face_down and (not args.layout or args.tile or args.tiles):
        p.error("--face-down needs --layout (each tile has a spot, its home) and no --tile: every back has the same name")
    if args.hover_lower_mm is not None and not -15 <= args.hover_lower_mm <= 10:
        p.error("--hover-lower-mm is from -15 (higher) to 10 (lower): the settle only goes 10mm under the hover's offset")
    if args.unattended and not (args.trials and (args.tile or args.tiles or args.layout)):
        p.error("--unattended needs --trials and --tile, --tiles or --layout: at the menu, or asked which tile, it would wait for you")
    if args.carry and (len(args.carry) != 2 or len(set(args.carry)) != 2 or not args.layout or args.unattended
                       or args.spots or args.start_spot
                       or "," in (args.policy or "") or args.hover_perturb_mm):
        p.error("--carry takes FROM,TO, two spots of --layout, with one policy, no --hover-perturb-mm, and someone watching")
    return args


def main(argv=None):
    args = parse_args(argv)
    bench = Bench(args)
    actions = {"hover": bench.hover, "pickup": bench.pickup, "return": bench.return_tile, "discard": bench.discard,
               "teach-discard": bench.teach_discard, "trials": bench.trials, "rest": bench.arm.rest,
               "note": bench.note, "setup": bench.setup}
    try:
        bench.arm.rest()
        if args.trials:
            bench.run_command("trials", lambda: bench.trials(args.trials, args.tile))
            return
        while True:
            print("\n" + "\n".join(f"  {i} {name}" for i, name in enumerate(COMMANDS, start=1)))
            print(f"[{'holding ' + bench.held['tile_name'] if bench.held else 'gripper empty'}]")
            drain_stdin()
            choice = input("> ").strip().lower()
            if choice.isdigit() and 1 <= int(choice) <= len(COMMANDS):
                choice = COMMANDS[int(choice) - 1]
            if choice in ("quit", "q"):
                break
            if choice not in actions:
                print(f"unknown command {choice!r}")
                continue
            bench.run_command(choice, actions[choice])
    except KeyboardInterrupt:
        pass            # Ctrl-C at the menu: quit
    finally:
        bench.close()
        if args.unattended:
            print_bump_report(bench.session.dir)


def print_bump_report(session_dir):
    """Write the session's report for a person to verify the self-reports against the scans (mahjong/bump_report.py).
    A failure here must not hide why the run ended, nor cost the session: it can be made later."""
    try:
        counts = bump_report.write(session_dir)
    except Exception as e:
        print(f"Couldn't write the bump report ({type(e).__name__}: {e}); try: python scripts/bump_report.py {session_dir}")
        return
    print(f"\nUnattended run done: {counts['attempts']} attempts, {counts['picked']} picked up, "
          f"{counts['flagged']} self-reported bump(s).\nGo through them: python scripts/bump_report.py {session_dir}")


if __name__ == "__main__":
    main()
