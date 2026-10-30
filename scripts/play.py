"""Autonomous play: the robot plays one closed hand of mahjong. A person only puts each drawn tile on the
drop zone (an ArUco marker, mahjong/markers.py) and presses Enter.

    python scripts/play.py --layout hand13                 # the 13 hand slots, taught with scripts/layout.py
    python scripts/play.py --layout hand13 --teach-zone    # find the drop-zone marker again first
    python scripts/play.py --layout hand13 --resume        # carry on the latest play session's hand
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

from mahjong import config, game, interventions, layout, markers
from mahjong.identity import DRAWN_MIN_SHARE, Votes
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
SIM_HAND = ["1C", "2C", "3C", "4D", "5D", "6D", "7B", "8B", "EW", "EW", "EW", "RD", "NW"]
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


class Player:
    def __init__(self, args, ask=input):
        self.args, self.ask = args, ask
        if args.sim:
            self.slots, zone_xy = [dict(s) for s in SIM_SLOTS], list(SIM_ZONE)
        else:
            self.slots = layout.load(args.layout)["spots"]
            if len(self.slots) != HAND_SLOTS:
                raise SystemExit(f"layout {args.layout} has {len(self.slots)} spots; a hand needs {HAND_SLOTS}")
            zone_xy = None
        self.votes = Votes()
        self.latest = {}                # slot -> the latest scan's detection there (or absent)
        self.bench = PlayBench(bench.parse_args(bench_argv(args)), self)
        self.session = self.bench.session
        self.bench.batch = self.bench.defer = True      # questions wait for the end of the step
        if args.sim:
            self.mat = self.bench.realsense
            self.mat.tiles = [{"tile_name": n, "xy": list(s["xy"])} for n, s in zip(SIM_HAND, self.slots)]
            self.draws = list(SIM_DRAWS)
        self.zone = {"xy": zone_xy} if zone_xy else self.load_zone(args.teach_zone)
        if not args.sim:
            self.check_zone_benched()
        self.state = {"phase": "register", "turn": 0, "hand": {}, "discards": [], "drawn": None, "history": [],
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

    # --- questions (each an intervention) ---

    def ask_name(self, slot, why):
        """A person names the tile at `slot`: Enter keeps the leading read."""
        leader = self.votes.leader(slot)[0]
        while True:
            drain_stdin()
            answer = self.ask(f"{why}. What is the tile at {slot} (reads: {self.votes.describe(slot)})? "
                              f"[Enter: {leader}] ").strip().upper()
            name = answer or leader
            if name and game.is_playable(name):
                break
            print(f"  {name!r} isn't a playable tile name (e.g. 5D, 9B, EW, RD; no flowers)")
        self.session.intervention("name_asked", slot=slot, reads=self.votes.describe(slot), answer=name,
                                  leading_read=leader, why=why)
        self.votes.known(slot, name)
        return name

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
        (markers.zone_test), or --skip-zone-test says to go on anyway (logged)."""
        test = markers.zone_test(self.zone["xy"], self.slots, layout.load(markers.ZONE_TEST_LAYOUT)["spots"],
                                 play_layout=self.args.layout)
        self.session.event("zone_test", **test)
        if test["ok"]:
            print(f"Drop zone checked on the bench: {markers.ZONE_TEST_LAYOUT} {test['spot']}, "
                  f"{test['picked']} picked up, {test['returned']} returned")
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
        hand = {}
        for s in self.slots:
            slot = s["id"]
            while slot not in self.latest:
                drain_stdin()
                self.ask(f"No tile seen at {slot} ({s['xy'][0]:.3f}, {s['xy'][1]:+.3f}): put one there, then Enter: ")
                self.session.intervention("zone_asked", slot=slot, note="no tile at a hand slot")
                self.scan("register")
                self.scan("register")
            name = self.votes.sure(slot)
            if name is None or not game.is_playable(name):
                name = self.ask_name(slot, "Not sure of a hand tile" if name is None else f"{name} can't be played")
            hand[slot] = name
        self.state["hand"] = hand
        self.check_counts()
        self.state["phase"] = "draw"
        self.save()

    def check_counts(self):
        """More than 4 of a tile among the hand, the drawn tile and the discards: one is misread."""
        names = list(self.state["hand"].values()) + self.state["discards"] + \
            ([self.state["drawn"]] if self.state["drawn"] else [])
        while extra := game.too_many(names):
            print(f"More than 4 of {', '.join(extra)}: one is misread.")
            self.recount([slot for slot, n in self.state["hand"].items() if n in extra])
            names = list(self.state["hand"].values()) + self.state["discards"] + \
                ([self.state["drawn"]] if self.state["drawn"] else [])

    def recount(self, slots, seen=None):
        """A person checks slots the table and the robot disagree on: Enter for each keeps the robot's
        name, or types the right one."""
        drain_stdin()
        for slot in slots:
            expected = self.state["hand"][slot]
            answer = self.ask(f"  {slot}: the robot has {expected}"
                              + (f", the scans read {seen[slot]}" if seen and seen.get(slot) else "")
                              + f" (reads: {self.votes.describe(slot)}). The tile there is [Enter: {expected}]: ")
            name = answer.strip().upper() or expected
            while not game.is_playable(name):
                name = self.ask(f"  {name!r} isn't a playable tile name. The tile at {slot} is: ").strip().upper() or expected
            self.session.intervention("recount", slot=slot, expected=expected, answer=name,
                                      reads=self.votes.describe(slot))
            self.state["hand"][slot] = name
            self.votes.known(slot, name)

    def check_hand(self):
        """Before a discard: every slot still shows its tile. A slot whose votes now lean to another name,
        or with no tile in the latest scan, goes to a recount."""
        seen = {}
        for slot in self.state["hand"]:
            leader, share, _ = self.votes.leader(slot)
            if slot in self.latest and leader is not None:
                seen[slot] = leader if share >= 0.5 else None
        bad = game.hand_mismatches(self.state["hand"], seen)
        if bad:
            print("The table doesn't match the hand: " + ", ".join(
                f"{slot} {want} (seen {got or 'nothing sure'})" for slot, (want, got) in bad.items()))
            self.recount(list(bad), {slot: got for slot, (_, got) in bad.items()})

    def read_drawn(self):
        """The drawn tile: up to DRAWN_SCANS scans until the vote clears the stricter bar, else ask."""
        for k in range(DRAWN_SCANS):
            self.scan("drawn")
            if k == 0:
                self.expect_appeared(self.zone["xy"])
            if ZONE not in self.latest:
                return None
            name = self.votes.sure(ZONE, min_share=DRAWN_MIN_SHARE)
            if name is not None and game.is_playable(name):
                return name
        name = self.votes.sure(ZONE, min_share=DRAWN_MIN_SHARE)
        return self.ask_name(ZONE, "Not sure of the drawn tile" if name is None else f"{name} can't be played")

    def wait_for_draw(self):
        """The drawn tile's name, or None when the wall is empty."""
        if self.args.sim:
            if not self.draws:
                return None
            self.mat.tiles.append({"tile_name": self.draws.pop(0), "xy": list(self.zone["xy"])})
        else:
            drain_stdin()
            if self.ask("\nPut the drawn tile on the drop zone, then Enter ('end' if the wall is empty): ").strip().lower() == "end":
                return None
        self.votes.reset(ZONE)
        while (name := self.read_drawn()) is None:
            drain_stdin()
            answer = self.ask("No tile seen on the drop zone: put it on the marker, then Enter ('end' if the wall is empty): ")
            if answer.strip().lower() == "end":
                return None
            self.session.intervention("zone_asked")
            self.votes.reset(ZONE)
        return name

    # --- moving tiles ---

    def pick(self, slot, name):
        """Pick the tile at `slot` (its name confirmed by the votes): True when held. Each retry scans again
        (the bench's pickup does), so a tile nudged by a miss is aimed at where it is now."""
        for attempt in range(1 + RETRIES):
            # the last check before a tile goes in the bin: the votes there must still be for it
            if self.votes.sure(slot, min_share=0.5, min_scans=1) not in (name, None):
                if slot == ZONE:
                    now = self.ask_name(ZONE, f"The drawn tile ({name}) reads differently now")
                else:
                    self.recount([slot])
                    now = self.state["hand"][slot]
                if now != name:
                    return False            # it's another tile: the caller decides again
            success = self.bench.pickup(ask=False, spot=self.spot(slot))
            if success:
                return True
            print(f"  {name} at {slot} not picked up" + (", retrying" if attempt < RETRIES else ""))
        drain_stdin()
        self.ask(f"The robot couldn't pick up {name} at {slot} after {1 + RETRIES} tries. "
                 + ("Put it in the discard container yourself, then Enter: " if self.state["phase"] == "discard"
                    else f"Put it on {self.state.get('vacated', 'its slot')} yourself, then Enter: "))
        self.session.intervention("pickup_failed", slot=slot, tile=name, attempt_dir=self.bench.last_attempt)
        return None

    def place_drawn(self, slot, name):
        """Move the drawn tile from the zone into `slot`, aimed at the slot's home (the return correction
        and landing check of the bench's return)."""
        picked = self.pick(ZONE, name)
        if picked is False:
            return False
        if picked:
            self.bench.held["home_xy"] = list(self.home(slot))
            self.bench.held["spot"] = slot
            self.bench.return_tile()
            if self.bench.held:
                raise SystemExit(f"Couldn't set {name} down at {slot} (out of reach) and it's still held: take it "
                                 f"from the jaws, put it on {slot}, and --resume.")
        self.votes.reset(ZONE)
        self.votes.known(slot, name)
        return True

    # --- a turn ---

    def turn(self):
        """One draw and discard. Returns "win", "end" (the wall is empty) or None (play on)."""
        s = self.state
        if s["phase"] == "draw":
            hand13 = [s["hand"][x["id"]] for x in self.slots]
            print(f"\n=== Turn {s['turn'] + 1} === hand: {' '.join(hand13)} "
                  f"(shanten {game.shanten(game.counts(hand13))})")
            drawn = self.wait_for_draw()
            if drawn is None:
                return "end"
            s["drawn"] = drawn
            self.check_hand()
            self.check_counts()
            self.end_step()
            if game.is_win(hand13 + [drawn]):
                print(f"Drew {drawn}: the hand is complete. Win!")
                s["history"].append({"turn": s["turn"] + 1, "drawn": drawn, "win": True})
                s["phase"] = "won"
                self.save()
                return "win"
            best, candidates = game.choose_discard(hand13, drawn, s["discards"])
            slot = ZONE if best["index"] == game.HAND_SIZE else self.slots[best["index"]]["id"]
            print(f"Drew {drawn}; discard {best['tile_name']} from {slot}: shanten {best['shanten_after']}, "
                  f"{best['ukeire_tiles']} improving tiles left ({best['ukeire_kinds']} kinds)")
            self.session.event("discard_chosen", turn=s["turn"] + 1, drawn=drawn, slot=slot, **best,
                               candidates=candidates[:5])
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
            if slot == ZONE:
                s["drawn"] = None
                return self.finish_turn(name, slot)
            s.update(phase="place", vacated=slot)
            self.save()
        if s["phase"] == "place":
            slot, drawn = s["vacated"], s["drawn"]
            if self.place_drawn(slot, drawn) is False:
                raise SystemExit(f"The tile on the drop zone isn't the {drawn} the robot read: check it and --resume.")
            s["hand"][slot] = drawn
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
        """A person checks the robot's hand against the table: anything it had wrong without asking is a
        silent error. Logs hand_end with the hand's grade."""
        s = self.state
        print("\nThe robot's hand: " + "  ".join(f"{x['id']}={s['hand'][x['id']]}" for x in self.slots))
        if s["drawn"]:
            print(f"  and {s['drawn']} on the drop zone")
        ids = {slot.upper(): slot for slot in s["hand"]} | {"ZONE": ZONE}
        while True:
            drain_stdin()
            answer = self.ask("Check it against the table. Enter if every tile matches, or the wrong ones as "
                              "SLOT=NAME (e.g. H3=5D H7=2B): ").strip().upper()
            parts = [part.split("=", 1) for part in answer.split()]
            if all(len(p) == 2 and p[0] in ids for p in parts):
                wrong = {ids[slot]: name for slot, name in parts}
                break
            print(f"  slots are {', '.join(s['hand'])} (or ZONE), as SLOT=NAME")
        for slot, name in wrong.items():
            robot = s["drawn"] if slot == ZONE else s["hand"][slot]
            self.session.intervention("silent_error", slot=slot, robot=robot, table=name)
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
        self.session.event("play_resumed", from_session=folder.name, phase=state["phase"], turn=state["turn"])
        print(f"Resuming {folder.name}: turn {state['turn'] + 1}, {state['phase']}")
        self.save()

    def play(self):
        self.bench.arm.rest()
        if self.state["phase"] == "register":
            self.register()
        while (result := self.turn()) is None:
            pass
        if result == "end":
            self.state["phase"] = "end"
            self.save()
        return self.audit(result)

    def close(self):
        self.bench.close()


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
    player = Player(args)
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
