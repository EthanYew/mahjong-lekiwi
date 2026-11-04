"""Autonomous play: the robot plays one closed hand of mahjong. A person only puts each drawn tile on the
drop zone (an ArUco marker, mahjong/markers.py) and presses Enter.

    python scripts/play.py --layout even-hand13                 # the 13 hand slots (data/layouts.json)
    python scripts/play.py --layout even-hand13 --teach-zone    # find the drop-zone marker again first
    python scripts/play.py --layout even-hand13 --resume        # carry on the latest play session's hand
    python scripts/play.py --sim                           # simulated arm and mat, a fixed hand and wall

A turn: read the drawn tile, check the table still matches the hand, choose the discard (lowest shanten,
then the most improving tiles still left, mahjong/game.py), pick it up with the policy and drop it in the
discard container (the bench's taught discard pose), and if it came from the hand, move the drawn tile
into the slot it left, aimed with the return correction so it lands on the slot's home. Ends on a win, or
when you type `end` at the draw prompt (the wall is empty). Then you check the robot's hand against the
table: anything it had wrong without noticing is a silent error.

It asks instead of guessing (docs/autonomy_plan.md, sections 3-4): a tile name the votes aren't clearly
for (mahjong/identity.py), a slot that doesn't match the hand, a fifth copy of a tile, a pickup that
failed after RETRIES re-scanned retries. Every question is logged as an intervention (class "ask"), so a
hand grades as clean (nothing asked), attended (questions only), rescued (a person touched something) or
silent error; `python scripts/session_report.py` prints it. One session is one hand.

Before a pickup, at the hover, the wrist camera reads the tile too (mahjong/name_check.py). Agreement changes nothing; a
disagreement asks you, and if your answer changes the name the discard is chosen again. It never names a tile by itself.
--read-hand does the same for all 13 tiles after registering them, since a tile that's never discarded is otherwise never read.

A person's word on a tile, the hand check's Enter included, holds against later scans until the tile moves more than 5mm (word_holds):
a detector that keeps misreading a tile sitting still is no news. A recount gets the wrist camera's look first.

A read that other tiles get misread as (identity.ALWAYS_ASK_READS) gets the wrist camera's look before you're asked. Registering
the hand, agreement drops that question, because of the next step: before the first draw you check all 13 against the table, and
can correct any of them (S9=5C), then or at any draw prompt. A drawn tile has no such check to come, so it's always asked about.
Any question about a drawn tile, whatever made it uncertain (split votes, a faint read, the ask list), gets that look first, with the
wrist read in the question and its view in the window. A window (mahjong/hand_view.py; --no-view for none) shows what each question is about.

Built on the bench (scripts/bench.py): the same hover, policy, scans before and after every pickup,
return and discard, homes, return correction and landing check, discard pose and session folder
(sessions/<time>_play/), plus checkpoint.json after every step. Teach the discard pose with
`bench.py` (teach-discard) first.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import config, game, hand_view, interventions, layout, markers, name_check
from mahjong.camera import grab_frames
from mahjong.identity import ALWAYS_ASK_READS, DRAWN_MIN_SHARE, Votes
from mahjong.perception import drain_stdin

import bench

ZONE = "zone"
REGISTER_SCANS = 3          # scans when registering the hand (each 5 frames): votes need 2 scans to count
DRAWN_SCANS = 3             # scans of a drawn tile before asking its name
RETRIES = 2                 # re-scanned pickup retries before a person is asked to move the tile
HAND_SLOTS = game.HAND_SIZE

# --sim: two rows of hand slots 57mm apart (Arm.place assumes tiles sit 57mm+ apart; 13 in one row
# won't fit the reach), the drop zone beyond the second row, and a hand that wins on the 3rd draw
SIM_SLOTS = ([{"id": f"H{i + 1}", "xy": [0.15, round(-0.171 + 0.057 * i, 4)]} for i in range(7)]
             + [{"id": f"H{i + 8}", "xy": [0.21, round(-0.1425 + 0.057 * i, 4)]} for i in range(6)])
SIM_ZONE = [0.25, 0.0]
SIM_HAND = ["1D", "2D", "3D", "4D", "5D", "6D", "7B", "8B", "SW", "SW", "SW", "RD", "NW"]     # none in ALWAYS_ASK_READS
SIM_DRAWS = ["GD", "9B", "NW"]


class PlayBench(bench.Bench):
    """The bench, writing a play session, telling the player about every scan (for the name votes)."""
    SESSION_KIND = "play"

    def __init__(self, args, player):
        self.player = player
        super().__init__(args)

    def scan(self, label, confirmed=None, **fields):
        tiles = super().scan(label, confirmed, **fields)
        self.player.observe(tiles)
        return tiles

    def at_hover(self, spot, attempt):
        return self.player.second_read(spot["id"], attempt) if spot else True


class Player:
    def __init__(self, args, ask=input, view=None):
        self.args, self.ask, self.view = args, ask, view
        if args.sim:
            self.slots, zone_xy = [dict(s) for s in SIM_SLOTS], list(SIM_ZONE)
        else:
            self.slots = layout.load(args.layout)["spots"]
            if len(self.slots) != HAND_SLOTS:
                raise SystemExit(f"layout {args.layout} has {len(self.slots)} spots; a hand needs {HAND_SLOTS}")
            zone_xy = None
        self.votes = Votes()
        self.latest = {}                # slot -> the latest scan's detection there (or absent)
        self.picking = None             # the name the pickup under way goes on
        self.name_changed = False       # a person's answer at the hover changed it: the pickup was called off
        self.person_said = {}           # placement -> the name a person gave it: no need to ask about it again
        self.words = {}                 # placement -> {"name", "xy"}: a person's word (the hand check's Enter too), and where the tile was
        self.wrist_seen = {}            # slot -> the wrist camera's latest {"frame", "read"} of its tile, for the window
        self.looked_at = set()          # hand slots the wrist camera has read since registering began
        self.reader = None if args.sim or args.no_name_check else name_check.load_reader()
        if not args.sim and self.reader is None:
            print("No wrist name check: " + ("--no-name-check" if args.no_name_check else f"no weights at {name_check.WEIGHTS}"))
        self.bench = PlayBench(bench.parse_args(bench_argv(args)), self)
        self.session = self.bench.session
        self.placements = self.bench.placements
        self.bench.batch = self.bench.defer = True      # questions wait for the end of the step
        self.hovering = False           # a look before asking left the arm over the mat
        if args.sim:
            self.mat = self.bench.realsense
            self.mat.tiles = [{"tile_name": n, "xy": list(s["xy"])} for n, s in zip(SIM_HAND, self.slots)]
            self.draws = list(SIM_DRAWS)
        self.zone = {"xy": zone_xy} if zone_xy else self.load_zone(args.teach_zone)
        if not args.sim:
            self.check_zone_benched()
        self.state = {"phase": "register", "turn": 0, "hand": {}, "discards": [], "discard_placements": [], "drawn": None, "history": [],
                      "layout": args.layout, "zone_xy": self.zone["xy"]}
        self.session.event("play_setup", layout=args.layout, slots=self.slots, zone=self.zone,
                           policy=self.bench.policy.id)

    # --- what the scans see ---

    def home(self, slot):
        """Where a slot's tile lives: where the slot's first tile was seen this session (the bench's home),
        else where the layout puts it."""
        if slot == ZONE:
            return self.zone["xy"]
        return self.bench.homes.get(slot) or next(s["xy"] for s in self.slots if s["id"] == slot)

    def spot(self, slot):
        return {"id": slot, "xy": self.home(slot)}

    def seen_at(self, slot):
        """Where the latest scan saw the slot's tile, else where it belongs: for outlining it on screen."""
        return self.latest[slot]["xy"] if slot in self.latest else self.home(slot)

    def box_at(self, slot):
        """The latest scan's box around the slot's tile as (u, v, w, h) in pixels, or None."""
        tile = self.latest.get(slot, {})
        return (*tile["pixel"], *tile["box_px"]) if "pixel" in tile and "box_px" in tile else None

    def observe(self, tiles):
        """Add a scan's reads to each slot's vote: the detection nearest each slot's home, if close enough."""
        self.latest = {}
        for slot in [s["id"] for s in self.slots] + [ZONE]:
            tile = layout.nearest_tile(tiles, self.home(slot), markers.ZONE_RADIUS_M if slot == ZONE else
                                       config.TILE_MERGE_RADIUS_M)
            if tile is not None:
                self.latest[slot] = tile
                self.votes.add(slot, tile)

    def scan(self, label, **fields):
        return self.bench.scan(label, **fields)

    # --- the window, and asking ---

    def name_of(self, slot):
        """The name the robot goes on at `slot`: settled, else what the votes lean to."""
        settled = self.state["drawn"] if slot == ZONE else self.state["hand"].get(slot)
        return settled or self.votes.leader(slot)[0]

    def attention(self, slot, why=None, otherwise=None):
        """What the window says about a tile being reviewed: {"why", "could_be"}, or {} if nothing stands out. `why`: this question's
        reason, first if there is one. `otherwise`: said if nothing else is, for a tile being asked about regardless."""
        name = self.name_of(slot)
        said = self.words.get(self.placements.id_at(slot), {}).get("name") == name
        reasons, could_be = self.votes.worth_a_look(slot, name, vouched=said)
        wrist = self.wrist_seen.get(slot)
        if wrist and wrist["read"]["name"] != name:
            could_be = [wrist["read"]["name"], *(other for other in could_be if other != wrist["read"]["name"])]
            if not why:                 # a question about a disagreement already says this
                reasons.insert(0, f"wrist camera reads {wrist['read']['name']} ({wrist['read']['confidence']:.2f})")
        elif wrist and reasons:
            reasons.append(f"wrist camera agrees ({wrist['read']['confidence']:.2f})")
        if said and reasons:
            reasons.insert(0, f"you said {name}")
        reasons = ([why] if why else []) + (reasons or ([otherwise] if otherwise and not why else []))
        return {"why": "; ".join(reasons), "could_be": could_be} if reasons else {}

    def show(self, caption, focus=None, flagged=None):
        """Put a question on screen: the RealSense view, every slot named as the robot has it, `focus` (the tile asked about now) and
        `flagged` ({slot: attention}, every tile in this review) picked out."""
        if self.view is None:
            return
        flagged = flagged or {}
        tiles = [{"id": slot, "xy": self.seen_at(slot), "name": self.name_of(slot), "box": self.box_at(slot), **flagged.get(slot, {})}
                 for slot in [s["id"] for s in self.slots] + [ZONE]]
        realsense = self.bench.realsense
        wrist = self.wrist_seen.get(focus)
        self.view.show(hand_view.compose(realsense.snapshot(), tiles, caption, realsense.intrinsics, realsense.T_cam_to_base,
                                         focus=focus, wrist=wrist["frame"] if wrist else None))

    def question(self, prompt, focus=None, caption=None, flagged=None):
        """Ask a person, with what it's about on screen."""
        self.show((caption or prompt).strip(), focus, flagged)
        drain_stdin()
        return self.ask(prompt)

    # --- questions (each an intervention) ---

    def placement_id(self, slot):
        """The tile at `slot`'s placement (mahjong/placements.py), started now if it's the first word on it."""
        if self.placements.id_at(slot) is None:
            self.placements.start(slot, self.home(slot), read=self.votes.leader(slot)[0])
        return self.placements.id_at(slot)

    def votes_at(self, slot):
        return {name: round(weight, 4) for name, weight in self.votes.totals.get(slot, {}).items()}

    def decided(self, slot, name, how):
        """Every name the robot goes on, the ones the votes accepted too: without those, asks have no denominator and a
        wrong acceptance has no record of what the votes said (scripts/name_report.py)."""
        _, share, scans = self.votes.leader(slot)
        self.session.event("name_decided", slot=slot, placement=self.placement_id(slot), name=name, how=how,
                           share=round(share, 3), scans=scans, votes=self.votes_at(slot))

    def why_ask(self, slot, name, unsure):
        """Why a person should name this tile (`unsure` if the votes have no name yet), or None if the robot can go on it."""
        if name is None:
            return unsure or "Not sure"
        return f"{name} can't be played" if not game.is_playable(name) else self.votes.doubt(slot, name)

    def ask_name(self, slot, why):
        """A person names the tile at `slot`: Enter keeps the leading read."""
        leader = self.votes.leader(slot)[0]
        while True:
            answer = self.question(f"{why}. What is the tile at {slot} (reads: {self.votes.describe(slot)})? "
                                   f"[Enter: {leader}] ", focus=slot, flagged={slot: self.attention(slot, otherwise=why)}).strip().upper()
            name = answer or leader
            if name and game.is_playable(name):
                break
            print(f"  {name!r} isn't a playable tile name (e.g. 5D, 9B, EW, RD; no flowers)")
        self.decided(slot, name, "person")
        self.session.intervention("name_asked", slot=slot, placement=self.placement_id(slot),
                                  reads=self.votes.describe(slot), votes=self.votes_at(slot), answer=name,
                                  typed=bool(answer), leading_read=leader, why=why)
        self.votes.known(slot, name)
        self.person_said[self.placement_id(slot)] = name
        self.give_word(slot, name)
        return name

    def give_word(self, slot, name):
        """A person has said what the tile at `slot` is: it holds, whatever the scans read, until the tile moves (word_holds)."""
        self.words[self.placement_id(slot)] = {"name": name, "xy": list(self.seen_at(slot))}

    def word_holds(self, slot, name):
        """The scans disagree with `name`: True if a person's word on this tile says to ignore them. A read a person overruled is
        no news while the tile sits where it did (S9 read 9C in 201 of 202 scans, and 53 questions in 20261104_183142_play changed
        no answer); what could change its name is the tile moving, past interventions.KNOCK_M. A tile not seen doesn't hold."""
        word, tile = self.words.get(self.placements.id_at(slot)), self.latest.get(slot)
        if word is None or word["name"] != name or tile is None:
            return False
        if word["xy"] is None:          # the word came with a tile the robot has since set down here: this is where it landed
            word["xy"] = list(tile["xy"])
        moved = math.dist(tile["xy"], word["xy"])
        if moved > interventions.KNOCK_M:
            return False
        self.session.event("name_held", slot=slot, placement=self.placements.id_at(slot), name=name, moved_m=round(moved, 4),
                           reads=self.votes.describe(slot), votes=self.votes_at(slot))
        return True

    # --- the second read, from the wrist camera at the hover (mahjong/name_check.py) ---

    def read_wrist(self, slot, name, folder, keep_as=None):
        """The reader's word on the tile under the jaws, logged whatever it says, or None without a frame. The frame is kept
        in the attempt's folder, so its read can be checked later."""
        frames = grab_frames(self.bench.camera, 1)
        if not frames:
            self.reopen_camera()
            frames = grab_frames(self.bench.camera, 1)
        if not frames:
            print("  no wrist camera frame: no second read of this tile")
            self.session.event("name_check", slot=slot, placement=self.placement_id(slot), name=name, read=None)
            return None
        read = self.reader.read(frames[0])
        self.wrist_seen[slot] = {"frame": frames[0], "read": read}
        if slot != ZONE:
            self.looked_at.add(slot)
        for file in filter(None, ("name_check.jpg", keep_as)):
            cv2.imwrite(str(folder / file), frames[0])
        self.session.event("name_check", slot=slot, placement=self.placement_id(slot), name=name, read=read["name"],
                           confidence=read["confidence"], agrees=read["name"] == name, attempt_dir=folder.name,
                           person_said=self.person_said.get(self.placement_id(slot)))
        return read

    def reopen_camera(self):
        """A read found the camera lost: reopen it (5 of 55 reads in 20261104_183142_play came back empty, and the camera stayed lost
        until a pickup voided). It may ask to retry or quit, so a pickup's key listener lets go of the terminal meanwhile."""
        terminal = self.bench.terminal
        listening = terminal.running.is_set()
        terminal.stop()
        self.bench.reopen_camera()
        if listening:
            terminal.start()

    def settle(self, slot, name, read):
        """The reader and the name the robot's going on disagree: a person says. Returns the tile's name."""
        why = f"The wrist camera reads {read['name']} ({read['confidence']:.2f}), the scans {name}"
        if slot == ZONE:
            return self.ask_name(ZONE, why)
        print(f"  {why}")
        self.recount([slot], why=why)
        return self.state["hand"][slot]

    def second_read(self, slot, folder):
        """At the hover, before the policy starts. False calls the pickup off: a person's answer changed the tile about to
        be discarded, so the discard has to be chosen again. A tile being moved from the drop zone carries on under its new name."""
        name = self.picking
        if self.reader is None or name is None:
            return True
        read = self.read_wrist(slot, name, folder)
        if read is None or read["name"] == name or self.person_said.get(self.placement_id(slot)) == name:
            return True
        presses = self.bench.keys.stop()    # the pickup's key listener would eat the answer
        if presses:
            self.bench.log_keys(folder.name, "pickup", presses)
        now = self.settle(slot, name, read)
        if slot == ZONE:
            self.state["drawn"] = now
        if now == name or self.state["phase"] != "discard":
            self.picking = now
            self.bench.keys.start("pickup")
            return True
        self.name_changed = True
        self.session.event("pickup_called_off", slot=slot, name=name, now=now, attempt_dir=folder.name)
        print(f"  It's {now}, not {name}: choosing the discard again")
        return False

    def look(self, slot, name):
        """Hover over the tile at `slot` and read it with the wrist camera, before a person is asked about it. The read, or None
        if there's no reader, no reach or no frame. Leaves the arm there: the caller rests it once it's done asking."""
        tile = self.latest.get(slot)
        if self.reader is None or tile is None or self.bench.hover_over(tile) is None:
            return None
        self.hovering = True
        return self.read_wrist(slot, name, self.session.attempt_dir(name), keep_as="wrist_0000.jpg")

    def rest_after_looking(self):
        """Back to rest if a look left the arm over the mat: the next scan needs a clear view."""
        if self.hovering:
            self.bench.arm.rest()
            self.hovering = False

    def look_before_asking(self, slot, name, why):
        """Registering the hand, a read on the ask list. The wrist camera agreeing drops the question, since a person still checks
        the whole hand before the first draw (confirm_hand): returns None. Otherwise the reason to ask, with the read if any."""
        read = self.look(slot, name)
        if read is None:
            return why
        if read["name"] == name:
            print(f"  {slot} reads {name} (on the ask list); the wrist camera agrees ({read['confidence']:.2f})")
            return None
        return f"{why}. The wrist camera reads {read['name']} ({read['confidence']:.2f})"

    def read_hand_at_hover(self):
        """Every hand tile read by the wrist camera (--read-hand): the arm hovers over each. A tile that's never discarded
        is otherwise never read. Agreement changes nothing; a disagreement asks."""
        if self.reader is None:
            return
        print("Reading the hand with the wrist camera...")
        for s in self.slots:
            slot, tile = s["id"], self.latest.get(s["id"])
            if slot in self.looked_at or tile is None or self.bench.hover_over(tile) is None:
                continue
            name = self.state["hand"][slot]
            read = self.read_wrist(slot, name, self.session.attempt_dir(name), keep_as="wrist_0000.jpg")
            if read and read["name"] != name and self.person_said.get(self.placement_id(slot)) != name:
                self.settle(slot, name, read)
        self.bench.arm.rest()

    def end_step(self):
        """What happened during the step, asked now: pickups marked with → or stopped with Esc (the bench's
        pickup queues them when it doesn't ask), and what the scans saw change (a knock, a tile put back)."""
        self.bench.review_queued()
        self.bench.check_scans()

    def expect_appeared(self, xy):
        """The latest scan saw a tile appear at xy because a person put it there on purpose (the drawn
        tile): not a change to ask about."""
        self.bench.explain(self.bench.scan_n,
                           lambda k, c: k == "appeared" and math.dist(c["xy"], xy) <= markers.ZONE_RADIUS_M,
                           "the drawn tile put on the drop zone")

    # --- the zone ---

    def load_zone(self, teach):
        zone = None if teach else markers.load_zone()
        if zone is None:
            self.bench.arm.rest()
            print("Finding the drop-zone marker (it must be uncovered)...")
            while (xy := markers.locate(self.bench.realsense)) is None:
                drain_stdin()
                self.ask(f"Marker {markers.DRAWN_TILE_MARKER_ID} not seen: uncover it or put it in view, then Enter: ")
            zone = markers.save_zone(xy, session=self.session.dir.name)
            print(f"Drop zone at ({xy[0]:.3f}, {xy[1]:+.3f}), saved to {markers.ZONE_FILE}")
        return zone

    def check_zone_benched(self):
        """The zone is only reached mid-hand: refuse to start unless the bench has checked a tile there
        and carried tiles from it (markers.zone_test), or --skip-zone-test says to go on anyway (logged)."""
        test = markers.zone_test(self.zone["xy"], self.slots, layout.load(markers.ZONE_TEST_LAYOUT)["spots"],
                                 play_layout=self.args.layout)
        self.session.event("zone_test", **test)
        if test["ok"]:
            print(f"Drop zone checked on the bench: {markers.ZONE_TEST_LAYOUT} {test['spot']}, "
                  f"{test['picked']} picked up, {test['returned']} returned, {test['carried']} carried to a slot "
                  f"(median {test['carried_median_mm']:.0f}mm from home)")
            return
        print("\nThe drop zone isn't checked on the bench yet:\n  " + "\n  ".join(test["problems"])
              + "\nTo check it:\n  " + "\n  ".join(test["fix"]))
        if not self.args.skip_zone_test:
            self.bench.close()
            raise SystemExit("not starting the hand (--skip-zone-test plays anyway)")
        self.session.intervention("zone_unchecked", problems=test["problems"])

    def check_zone(self):
        """With the zone empty, the marker should be where it was taught: if it moved, so did the camera or
        the base, and every position on the mat is off. Not checked in --sim (no marker drawn)."""
        if self.args.sim:
            return
        xy = markers.locate(self.bench.realsense)
        self.session.event("zone_check", xy=xy, taught_xy=self.zone["xy"])
        if xy is not None and markers.moved(self.zone, xy):
            raise SystemExit(f"The drop-zone marker moved {math.dist(xy, self.zone['xy']) * 1000:.0f}mm: the camera or "
                             f"the base moved. Check the setup, teach the zone again (--teach-zone) and --resume.")

    # --- the hand ---

    def register(self):
        """Read the 13 slots: a few scans' votes, a person asked for any slot the votes aren't sure of."""
        print("\nReading the hand...")
        for _ in range(REGISTER_SCANS):
            self.scan("register")
        self.end_step()
        for s in self.slots:                # every slot has a tile before the arm goes near any (scans need it at rest)
            slot = s["id"]
            while slot not in self.latest:
                self.question(f"No tile seen at {slot} ({s['xy'][0]:.3f}, {s['xy'][1]:+.3f}): put one there, then Enter: ", focus=slot)
                self.session.intervention("zone_asked", slot=slot, note="no tile at a hand slot")
                self.bench.hands_on = f"a tile put at {slot} when asked"
                self.scan("register")
                self.scan("register")
        hand = self.state["hand"]
        for s in self.slots:
            slot = s["id"]
            name = self.votes.sure(slot)
            why = self.why_ask(slot, name, "Not sure of a hand tile")
            how = "vote"
            if why and name in ALWAYS_ASK_READS:
                why = self.look_before_asking(slot, name, why)
                how = "wrist"
            if why:
                name = self.ask_name(slot, why)
            else:
                self.decided(slot, name, how)
            hand[slot] = name
        self.rest_after_looking()
        if self.args.read_hand:
            self.read_hand_at_hover()
        self.check_counts()
        self.state["phase"] = "draw"
        self.save()

    def check_counts(self):
        """More than 4 of a tile among the hand, the drawn tile and the discards: one is misread."""
        names = list(self.state["hand"].values()) + self.state["discards"] + \
            ([self.state["drawn"]] if self.state["drawn"] else [])
        while extra := game.too_many(names):
            print(f"More than 4 of {', '.join(extra)}: one is misread.")
            self.recount([slot for slot, n in self.state["hand"].items() if n in extra], why=f"More than 4 of {', '.join(extra)}")
            names = list(self.state["hand"].values()) + self.state["discards"] + \
                ([self.state["drawn"]] if self.state["drawn"] else [])

    def recount(self, slots, seen=None, why=None):
        """A person checks slots the table and the robot disagree on: Enter for each keeps the robot's
        name, or types the right one. The window has them all picked out, so a person sees the whole request at the first."""
        for slot in slots:
            flagged = {s: self.attention(s, why=why, otherwise="the scans don't match the hand") for s in slots}
            expected = self.state["hand"][slot]
            prompt = (f"  {slot}: the robot has {expected}"
                      + (f", the scans read {seen[slot]}" if seen and seen.get(slot) else "")
                      + f" (reads: {self.votes.describe(slot)}). The tile there is [Enter: {expected}]: ")
            answer = self.question(prompt, focus=slot, caption=f"{why}. {prompt.strip()}" if why else None, flagged=flagged)
            name = answer.strip().upper() or expected
            while not game.is_playable(name):
                name = self.ask(f"  {name!r} isn't a playable tile name. The tile at {slot} is: ").strip().upper() or expected
            self.set_name(slot, name, typed=bool(answer.strip()))

    def set_name(self, slot, name, typed):
        """A person's word on a hand tile, logged as a recount (a wrong name found this way is the votes' mistake, in the ledger)."""
        self.session.intervention("recount", slot=slot, placement=self.placement_id(slot), expected=self.state["hand"][slot],
                                  answer=name, typed=typed, reads=self.votes.describe(slot), votes=self.votes_at(slot))
        self.state["hand"][slot] = name
        self.votes.known(slot, name)
        self.person_said[self.placement_id(slot)] = name
        self.give_word(slot, name)

    def hand_text(self):
        return "  ".join(f"{s['id']}={self.state['hand'][s['id']]}" for s in self.slots)

    def corrections(self, answer):
        """The hand tiles a person corrected as SLOT=NAME, applied. False if the answer can't be read as that (it says why)."""
        ids = {slot.upper(): slot for slot in self.state["hand"]}
        fixes = parse_fixes(answer, ids)
        if fixes is None:
            print(f"  slots are {', '.join(ids)}, as SLOT=NAME (e.g. S9=5C)")
            return False
        bad = [name for name in fixes.values() if not game.is_playable(name)]
        if bad:
            print(f"  {', '.join(bad)} isn't a playable tile name (e.g. 5D, 9B, EW, RD; no flowers)")
            return False
        for slot, name in fixes.items():
            self.set_name(slot, name, typed=True)
        return True

    def confirm_hand(self):
        """A person checks all 13 against the table before the first draw: the votes can be sure of a wrong name (a 5C read as 9C,
        20261104_164541_play), and nothing else would ask. Enter isn't graded as a question (a hand has this every time); a
        correction is a recount, so that hand isn't clean. Their word goes in the ledger as the truth for these placements, and
        holds against later scans (word_holds). Not against the wrist camera at the hover: that's a new read, before a tile is binned."""
        while True:
            print("\nThe hand as the robot has it: " + self.hand_text())
            flagged = {slot: self.attention(slot) for slot in self.state["hand"]}
            answer = self.question("Check it against the table. Enter if every tile matches, or the wrong ones as "
                                   "SLOT=NAME (e.g. S9=5C): ", flagged=flagged).strip()
            if not answer:
                break
            self.corrections(answer)
        self.check_counts()
        s = self.state
        for slot, name in s["hand"].items():
            self.give_word(slot, name)
        s["hand_confirmed"] = True
        self.session.event("hand_confirmed", confirmed={self.placement_id(slot): name for slot, name in s["hand"].items()})
        self.save()

    def check_hand(self):
        """Before a discard: every slot still shows its tile. A slot whose votes now lean to another name, unless a person's word
        on it holds, or with no tile in the latest scan, goes to a recount, with the wrist camera's look first."""
        seen = {}
        for slot in self.state["hand"]:
            leader, share, _ = self.votes.leader(slot)
            if slot in self.latest and leader is not None:
                seen[slot] = leader if share >= 0.5 else None
        bad = {slot: mismatch for slot, mismatch in game.hand_mismatches(self.state["hand"], seen).items()
               if not self.word_holds(slot, mismatch[0])}
        if bad:
            print("The table doesn't match the hand: " + ", ".join(
                f"{slot} {want} (seen {got or 'nothing sure'})" for slot, (want, got) in bad.items()))
            for slot, (want, got) in bad.items():
                note = self.look_note(slot, want)
                if note:
                    print(f"  {note}")
                self.recount([slot], {slot: got}, why=note)
            self.rest_after_looking()

    def read_drawn(self):
        """The drawn tile: up to DRAWN_SCANS scans until the vote clears the stricter bar, else ask."""
        for k in range(DRAWN_SCANS):
            self.scan("drawn")
            if k == 0:
                self.expect_appeared(self.zone["xy"])
            if ZONE not in self.latest:
                return None
            name = self.votes.sure(ZONE, min_share=DRAWN_MIN_SHARE)
            if not self.why_ask(ZONE, name, ""):
                self.decided(ZONE, name, "vote")
                return name
        name = self.votes.sure(ZONE, min_share=DRAWN_MIN_SHARE)
        return self.ask_drawn(self.why_ask(ZONE, name, "Not sure of the drawn tile"))

    def look_note(self, slot, name):
        """Look at the tile at `slot`, for a question about it: what the wrist camera read, or None with no reader. Leaves the arm there."""
        if self.reader is None:
            return None
        read = self.look(slot, name)
        return f"The wrist camera reads {read['name']} ({read['confidence']:.2f})" if read else "No read from the wrist camera"

    def ask_drawn(self, why):
        """A person names the drawn tile, with the wrist camera's read of it whatever made it uncertain. `votes.sure` is None when the
        votes split, so a look that depends on the name it returns skips the least certain tiles. No hand check is to come, so agreement
        doesn't drop the question: the read only informs (a read never names a tile, mahjong/name_check.py)."""
        note = self.look_note(ZONE, self.votes.leader(ZONE)[0])
        name = self.ask_name(ZONE, f"{why}. {note}" if note else why)
        self.rest_after_looking()
        return name

    def wait_for_draw(self):
        """The drawn tile's name, or None when the wall is empty."""
        self.wrist_seen.pop(ZONE, None)
        if self.args.sim:
            if not self.draws:
                return None
            self.mat.tiles.append({"tile_name": self.draws.pop(0), "xy": list(self.zone["xy"])})
        elif self.ask_for_draw("Put the drawn tile on the drop zone") == "end":
            return None
        self.votes.reset(ZONE)
        while (name := self.read_drawn()) is None:
            if self.ask_for_draw("No tile seen on the drop zone: put it on the marker") == "end":
                return None
            self.session.intervention("zone_asked")
            self.votes.reset(ZONE)
        return name

    def ask_for_draw(self, what):
        """Wait for the drawn tile: "end" if the wall is empty. A hand tile found to be wrong can be corrected here too."""
        while True:
            hand13 = [self.state["hand"][x["id"]] for x in self.slots]
            answer = self.question(f"\n{what}, then Enter ('end' if the wall is empty, S9=5C to correct a hand tile): ",
                                   focus=ZONE, caption=f"{what}. {game.seeking(hand13, self.state['discards'])}").strip()
            if "=" not in answer:
                return answer.lower()
            if self.corrections(answer):
                print("The hand: " + self.hand_text())

    # --- moving tiles ---

    def pick(self, slot, name):
        """Pick the tile at `slot` (its name confirmed by the votes): True when held. Each retry scans again
        (the bench's pickup does), so a tile nudged by a miss is aimed at where it is now."""
        for attempt in range(1 + RETRIES):
            # the last check before a tile goes in the bin: the votes there must still be for it
            if self.votes.sure(slot, min_share=0.5, min_scans=1) not in (name, None) and not self.word_holds(slot, name):
                if slot == ZONE:
                    now = self.ask_drawn(f"The drawn tile ({name}) reads differently now")
                else:
                    self.recount([slot])
                    now = self.state["hand"][slot]
                if now != name:
                    return False            # it's another tile: the caller decides again
            self.picking = name
            success = self.bench.pickup(ask=False, spot=self.spot(slot))
            name, self.picking = self.picking, None
            if self.name_changed:           # a person's answer at the hover: it's another tile
                self.name_changed = False
                return False
            if success:
                return True
            print(f"  {name} at {slot} not picked up" + (", retrying" if attempt < RETRIES else ""))
        self.question(f"The robot couldn't pick up {name} at {slot} after {1 + RETRIES} tries. "
                      + ("Put it in the discard container yourself, then Enter: " if self.state["phase"] == "discard"
                         else f"Put it on {self.state.get('vacated', 'its slot')} yourself, then Enter: "), focus=slot)
        self.session.intervention("pickup_failed", slot=slot, tile=name, attempt_dir=self.bench.last_attempt)
        self.bench.hands_on = f"{name} moved by hand: the robot couldn't pick it up"
        return None

    def place_drawn(self, slot, name):
        """Move the drawn tile from the zone into `slot`, aimed at the slot's home (the return correction
        and landing check of the bench's return)."""
        picked = self.pick(ZONE, name)
        if picked is False:
            return False
        name = self.state["drawn"]          # a person may have corrected it at the hover
        if picked:
            self.bench.held["home_xy"] = list(self.home(slot))
            self.bench.held["spot"] = slot
            self.bench.return_tile()
            if self.bench.held:
                raise SystemExit(f"Couldn't set {name} down at {slot} (out of reach) and it's still held: take it "
                                 f"from the jaws, put it on {slot}, and --resume.")
        self.votes.reset(ZONE)
        self.votes.known(slot, name)
        self.wrist_seen.pop(slot, None)
        moved = self.placements.end(ZONE, "moved")
        self.placements.start(slot, self.home(slot), truth_source="game", read=name, from_placement=moved)
        if self.words.get(moved, {}).get("name") == name:       # the same tile: the word comes with it, from where it lands
            self.words[self.placements.id_at(slot)] = {"name": name, "xy": None}
        return True

    # --- a turn ---

    def turn(self):
        """One draw and discard. Returns "win", "end" (the wall is empty) or None (play on)."""
        s = self.state
        if s["phase"] == "draw":
            hand13 = [s["hand"][x["id"]] for x in self.slots]
            print(f"\n=== Turn {s['turn'] + 1} === hand: {self.hand_text()}")
            print(f"  {game.seeking(hand13, s['discards'])}")
            drawn = self.wait_for_draw()
            if drawn is None:
                return "end"
            s["drawn"] = drawn
            self.check_hand()
            self.check_counts()
            self.end_step()
            hand13 = [s["hand"][x["id"]] for x in self.slots]       # a person may have just corrected it
            if game.is_win(hand13 + [drawn]):
                print(f"Drew {drawn}: the hand is complete. Win!")
                s["history"].append({"turn": s["turn"] + 1, "drawn": drawn, "win": True})
                s["phase"] = "won"
                self.save()
                return "win"
            best, candidates = game.choose_discard(hand13, drawn, s["discards"])
            slot = ZONE if best["index"] == game.HAND_SIZE else self.slots[best["index"]]["id"]
            kept = (hand13 + [drawn])[:best["index"]] + (hand13 + [drawn])[best["index"] + 1:]
            after = [*s["discards"], best["tile_name"]]
            print(f"Drew {drawn}; discard {best['tile_name']} from {slot}. {game.seeking(kept, after)}")
            self.session.event("discard_chosen", turn=s["turn"] + 1, drawn=drawn, slot=slot, **best,
                               candidates=candidates[:5], seeking=game.improving_tiles(kept, after))
            s.update(phase="discard", discard_slot=slot, discard=best)
            self.save()
        if s["phase"] == "discard":
            slot, name = s["discard_slot"], s["discard"]["tile_name"]
            picked = self.pick(slot, name)
            if picked is False:             # a recount changed the tile there: choose again
                s["phase"] = "draw_known"
                return self.decide_again()
            if picked:
                self.bench.discard(ask=False)
            self.votes.reset(slot)
            s["discards"].append(name)
            s.setdefault("discard_placements", []).append(self.placements.end(slot, "discarded"))
            if slot == ZONE:
                s["drawn"] = None
                return self.finish_turn(name, slot)
            s.update(phase="place", vacated=slot)
            self.save()
        if s["phase"] == "place":
            slot, drawn = s["vacated"], s["drawn"]
            if self.place_drawn(slot, drawn) is False:
                raise SystemExit(f"The tile on the drop zone isn't the {drawn} the robot read: check it and --resume.")
            s["hand"][slot] = s["drawn"]
            s["drawn"] = None
            return self.finish_turn(s["discard"]["tile_name"], s["discard_slot"])
        return None

    def decide_again(self):
        """After a recount changed the hand mid-turn: choose the discard again from the corrected hand."""
        s = self.state
        hand13 = [s["hand"][x["id"]] for x in self.slots]
        best, _ = game.choose_discard(hand13, s["drawn"], s["discards"])
        slot = ZONE if best["index"] == game.HAND_SIZE else self.slots[best["index"]]["id"]
        s.update(phase="discard", discard_slot=slot, discard=best)
        self.save()
        return None

    def finish_turn(self, discarded, slot):
        s = self.state
        s["turn"] += 1
        s["history"].append({"turn": s["turn"], "discarded": discarded, "from": slot,
                             "shanten_after": s["discard"]["shanten_after"]})
        self.session.event("turn_done", turn=s["turn"], discarded=discarded, slot=slot,
                           hand=[s["hand"][x["id"]] for x in self.slots], discards=s["discards"])
        s["phase"] = "draw"
        for key in ("discard_slot", "discard", "vacated"):
            s.pop(key, None)
        self.end_step()
        self.check_zone()
        self.save()
        return None

    # --- the end of the hand ---

    def audit(self, result):
        """A person checks the robot's hand against the table and its discards against the container: anything it had
        wrong without asking is a silent error. Logs hand_audit (what the person vouched for, by placement) and hand_end
        with the hand's grade."""
        s = self.state
        print("\nThe robot's hand: " + "  ".join(f"{x['id']}={s['hand'][x['id']]}" for x in self.slots))
        if s["drawn"]:
            print(f"  and {s['drawn']} on the drop zone")
        if s["discards"]:
            print("  Discarded: " + "  ".join(f"D{i}={name}" for i, name in enumerate(s["discards"], 1)))
        discards = {f"D{i}": name for i, name in enumerate(s["discards"], 1)}
        believed = {**s["hand"], **discards, ZONE: s["drawn"]}
        ids = {slot.upper(): slot for slot in believed}
        while True:
            answer = self.question("Check it against the table and the discard container. Enter if every tile matches, or "
                                   "the wrong ones as SLOT=NAME (e.g. H3=5D D2=2B): ")
            wrong = parse_fixes(answer, ids)
            if wrong is not None:
                break
            print(f"  slots are {', '.join(ids)}, as SLOT=NAME")
        where = {**{slot: self.placements.id_at(slot) for slot in (*s["hand"], ZONE)},
                 **dict(zip(discards, s.get("discard_placements", [])))}
        for slot, name in wrong.items():
            self.session.intervention("silent_error", slot=slot, placement=where.get(slot), robot=believed[slot], table=name)
        self.session.event("hand_audit", confirmed={where[slot]: wrong.get(slot, believed[slot]) for slot in believed
                                                    if where.get(slot) and wrong.get(slot, believed[slot])})
        # still running, so derive() sees no session_end and would count a "killed" rescue
        found = [i for i in interventions.derive(self.session_events()) if i["kind"] != "killed"]
        grade = interventions.hand_grade(found)
        self.session.event("hand_end", result=result, turns=s["turn"], grade=grade, silent_errors=len(wrong),
                           asks=sum(i["class"] == interventions.ASK for i in found),
                           rescues=sum(i["class"] == interventions.RESCUE for i in found),
                           hand=[s["hand"][x["id"]] for x in self.slots], discards=s["discards"])
        print(f"\nHand over ({result}) after {s['turn']} turns: {grade}")
        return grade

    def session_events(self):
        from mahjong.session import read_events
        return read_events(self.session.dir)

    # --- checkpoint ---

    def save(self):
        path = self.session.dir / "checkpoint.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        tmp.replace(path)

    def resume(self, folder):
        state = json.loads((folder / "checkpoint.json").read_text())
        if state["phase"] in ("won", "end"):
            raise SystemExit(f"{folder.name}'s hand is over ({state['phase']})")
        holding = self.bench.arm.holding()[0]
        if holding:
            raise SystemExit("The arm is holding a tile: put it where it belongs (bench.py: return or discard), "
                             "then --resume.")
        self.state = {**state, "resumed_from": folder.name}
        for slot, name in state["hand"].items():
            self.votes.known(slot, name)
            self.placements.start(slot, self.home(slot), truth_source="game", read=name)
        self.session.event("play_resumed", from_session=folder.name, phase=state["phase"], turn=state["turn"])
        print(f"Resuming {folder.name}: turn {state['turn'] + 1}, {state['phase']}")
        self.save()

    def play(self):
        self.bench.arm.rest()
        if self.state["phase"] == "register":
            self.register()
        if not self.state.get("hand_confirmed"):
            self.confirm_hand()
        while (result := self.turn()) is None:
            pass
        if result == "end":
            self.state["phase"] = "end"
            self.save()
        return self.audit(result)

    def close(self):
        self.bench.close()
        if self.view is not None:
            self.view.close()


def parse_fixes(answer, ids):
    """{slot: NAME} from "S9=5C H3=5D" (any case), slots looked up in ids ({SLOT in capitals: slot}); None if it isn't that."""
    parts = [part.split("=", 1) for part in answer.strip().upper().split()]
    return {ids[p[0]]: p[1] for p in parts} if all(len(p) == 2 and p[0] in ids for p in parts) else None


def bench_argv(args):
    return (["--sim"] if args.sim else []) + (["--policy", args.policy] if args.policy else [])


def latest_play_session():
    folders = sorted(p for p in config.SESSIONS.glob("*_play") if (p / "checkpoint.json").exists())
    if not folders:
        raise SystemExit(f"no play session with a checkpoint under {config.SESSIONS}")
    return folders[-1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--layout", default=None, help="the 13 hand slots (a layout taught with scripts/layout.py)")
    p.add_argument("--policy", default=None, help="policy id from data/policies.json (default: its 'default')")
    p.add_argument("--teach-zone", action="store_true", help="find the drop-zone marker again (it must be uncovered)")
    p.add_argument("--resume", nargs="?", const="latest", default=None,
                   help="carry on a play session's hand from its checkpoint (default: the latest)")
    p.add_argument("--skip-zone-test", action="store_true",
                   help=f"play even if the bench hasn't picked up and returned a tile on the drop zone (its spot "
                        f"in {markers.ZONE_TEST_LAYOUT}); logged")
    p.add_argument("--no-name-check", action="store_true",
                   help="play without the wrist camera's second read of a tile before it's picked up (logged reads and questions stop)")
    p.add_argument("--read-hand", action="store_true",
                   help="also hover over all 13 hand tiles after registering them and read each with the wrist camera")
    p.add_argument("--no-view", action="store_true", help="no window showing what each question is about; the terminal only")
    p.add_argument("--sim", action="store_true", help="simulated arm, camera, mat and policy, a fixed hand and wall")
    args = p.parse_args(argv)
    if not args.sim and not args.layout:
        p.error("--layout is required (the hand slots), except with --sim")
    return args


def main(argv=None):
    args = parse_args(argv)
    folder = None
    if args.resume:     # before connecting, so a bad folder fails fast
        folder = latest_play_session() if args.resume == "latest" else Path(args.resume)
    player = Player(args, view=None if args.no_view else hand_view.Window())
    try:
        if folder:
            player.resume(folder)
        player.play()
    except KeyboardInterrupt:
        print("\ninterrupted — the checkpoint has the hand; --resume carries on")
        player.session.intervention("interrupted", command="play")
    finally:
        player.close()


if __name__ == "__main__":
    main()
