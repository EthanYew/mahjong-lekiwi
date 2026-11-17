"""Face-down play on the arm: the robot reads its own face-down hand, then plays turns from the wall until a win or the wall
runs out. Built on the crude run's turn (mahjong/crude_run.py), with every tile found by its saved position.

    python scripts/play_face_down.py                         # even-hand13, the wall at L2; you check the hand and every draw
    python scripts/play_face_down.py --unattended            # the reader's name is taken: you only lay the hand and feed the wall
    python scripts/play_face_down.py --resume                # carry on the latest hand from its last finished turn
    python scripts/play_face_down.py --report                # the latest hand's reads against your review of its photos
    python scripts/play_face_down.py --sim                   # simulated arm and cameras, no hardware

1. A RealSense window outlines the 13 slots (green), the wall's station (blue, --station) and the drop zone (orange, kept
   clear). Lay any 13 tiles GREEN side up on the outlines, and the wall: --wall-stacks stacks of 2 from the station outward.
2. The arm picks up each slot's tile, holds it up to the base camera at the saved base_view poses, and puts it back on its slot.
3. You check the 13 names in one window (Enter, or the wrong ones as S3=5C S9=EW). A hand with five of a tile isn't taken.
4. Each turn: you push the wall's next stack onto the station when asked (every other turn; e there = the wall's empty, end
   the hand). The arm draws the top tile, then the bottom one, shows it, sets it on the drop zone, and you confirm its name.
   game.py chooses the discard: the drawn tile goes straight to the container, or a hand tile goes and the drawn tile takes
   its slot. After the turn: Enter, a correction (S3=5C), `end`, or a note of what went wrong.
5. At the end, turn the 13 over and check them against the robot's hand: a tile it had wrong is a silent error.

--unattended: no name is asked. A read under --reshow-below gets a second hold at the poses, both holds' votes added, and
the reader's name is taken. Play goes on with a read that makes a fifth copy (logged). What still asks: the wall's feed, and
a step the arm couldn't do (a pick missed twice, a tile lost), which a person has to put right. It needs a reader. Silent
wrong names are possible by design (the owner's choice, 2026-11-17), so an unattended hand is no play-ready result: it
measures the reader and the arm together.

Every hold's photos are saved as the reader's training photos, filed under the name the hand used, and wait out of training
until you review them (python scripts/base_cam_review.py --play). --report then scores every read against your review,
discards too, and lists the discards a wrong read changed. Results in sessions/<time>_play_fd/: the crude run's events,
fd_* events, holdup_shot events and photos/, checkpoint.json, play_fd_summary.json. Run setup_check.py first, and from the
main checkout.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import (base_cam, base_cam_collect, config, crude_run, face_down_play, face_down_rig, game, hand_view, interventions,
                     layout, markers, privacy, takeover, wall_test)
from mahjong.identity import parse_fixes
from mahjong.perception import drain_stdin
from mahjong.session import read_events

LAY_WINDOW = "lay the hand and the wall (RealSense view)"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--layout", default="even-hand13", help="the hand's 13 slots (default %(default)s)")
    p.add_argument("--station", default="L2", help=f"the wall's station in layout {wall_test.LAYOUT} (default %(default)s)")
    p.add_argument("--wall-stacks", type=int, default=4, help="stacks of 2 to lay the wall with at the start (default %(default)s)")
    p.add_argument("--top-back-mm", type=float, default=10.0, help="a top tile's hover starts this much nearer the robot (default %(default)s)")
    p.add_argument("--unattended", action="store_true", help="take the reader's names: no name is asked, only the wall's feed and steps the arm couldn't do")
    p.add_argument("--reshow-below", type=float, default=face_down_play.RESHOW_BELOW,
                   help="--unattended: a read under this confidence gets a second hold at the poses (default %(default)s)")
    p.add_argument("--resume", nargs="?", const="latest", default=None, metavar="SESSION",
                   help="carry on a hand (default: the latest) from its last finished turn; the tiles stay where they are")
    p.add_argument("--report", nargs="?", const="latest", default=None, metavar="SESSION",
                   help="score a hand's reads (default: the latest) against your review of its photos, and stop: no hardware")
    p.add_argument("--policy", default=None, help="policy id (default: the registry's default)")
    p.add_argument("--no-view", action="store_true", help="no windows: the hand to lay and the hand check in the terminal only")
    p.add_argument("--sim", action="store_true", help="simulated arm, cameras and policy: no hardware")
    takeover.add_argument(p)
    p.set_defaults(empty=False, hover_lower_mm=None, frames=1, settle_s=1.0, gap_s=0.25, jitter_deg=0.0, stacked=True)     # what face_down_rig's Rig and crude_run's Run read
    return p.parse_args(argv)


def play_sessions():
    return sorted(config.SESSIONS.glob(f"*_{face_down_play.PLAY_KIND}"))


def find_session(arg, need_checkpoint=False):
    if arg != "latest":
        folder = Path(arg) if Path(arg).is_dir() else config.SESSIONS / arg
        if not folder.is_dir():
            raise SystemExit(f"no session {arg}")
        return folder
    folders = [f for f in play_sessions() if not need_checkpoint or (f / "checkpoint.json").exists()]
    if not folders:
        raise SystemExit(f"no face-down play session{' with a checkpoint' if need_checkpoint else ''} under {config.SESSIONS}")
    return folders[-1]


def load_checkpoint(arg):
    folder = find_session(arg, need_checkpoint=True)
    state = json.loads((folder / "checkpoint.json").read_text()) if (folder / "checkpoint.json").exists() else None
    if state is None:
        raise SystemExit(f"{folder.name} has no checkpoint.json: it stopped before the hand was checked, so start a new one")
    if state["phase"] != "play":
        raise SystemExit(f"{folder.name}'s hand is over ({state['phase']})")
    return folder, state


class Player(crude_run.Run):
    """One hand: the crude run's turn, a hand the arm reads for itself, no end until a win or the wall runs out."""

    def __init__(self, rig, args, ask, slots, station, zone_spot, reader, boxes, window=None):
        super().__init__(rig, args, ask, slots, {}, station, zone_spot, reader, boxes)
        self.unattended, self.window = args.unattended, window
        self.shots, self.holds, self.corrected, self.registered = 0, {}, 0, 0
        self.phase = "register"

    # --- what a person did, for the grade ---

    def broke(self, step, what, by_hand=False):
        super().broke(step, what, by_hand)
        if by_hand:
            self.rig.session.intervention("fd_by_hand", step=step, what=what, turn=self.turn_no)

    def correct(self, slot, read, name, at):
        """A person gave a name the reader didn't (an `ask`, as play.py's questions are); pressing Enter on its read isn't one."""
        self.corrected += 1
        self.rig.session.intervention("name_corrected", slot=slot, read=read, name=name, at=at, turn=self.turn_no)

    # --- reading ---

    def more_looks(self, frames, from_pose):
        """Unattended, a read under --reshow-below gets one more hold at the poses: both holds' votes go into the read."""
        if not self.unattended:
            return frames, from_pose
        guess = base_cam.read_held(self.reader, frames, from_pose, self.boxes)
        if guess is None or guess["confidence"] >= self.args.reshow_below:
            return frames, from_pose
        more, more_pose, problem = self.rig.show(self.rng, still_held=self.jaws_have_tile)
        self.rig.session.event("fd_reshow", turn=self.turn_no, first=guess, problem=problem or None, frames=len(more))
        print(f"  The reader isn't sure ({guess['name']}, {guess['confidence']:.0%}): showing it again.")
        return frames + more, from_pose + more_pose

    def ask_name(self, why):
        """A name only a person can give (the camera never saw the tile, or there's no reader)."""
        drain_stdin()
        while True:
            name = self.ask(f"{why}. Type its name (q = quit): ").strip().upper()
            if name == "Q":
                return None
            if game.is_playable(name):
                return name
            print(f"  '{name}' isn't a tile name (for example 5C, 3D, 9B, EW, RD).")

    def save_hold(self, frames, from_pose, name, guess, where, filed_by):
        """The hold's photos as the reader's training photos, filed under `name`, waiting for a person's review."""
        if not frames:
            return
        self.shots += 1
        hold = self.holds[name] = self.holds.get(name, 0) + 1
        files = base_cam.save_shot(self.rig.session.dir, {"tile": name, "kind": base_cam.UNKNOWN_KIND, "hold": hold}, frames)
        self.rig.session.event("holdup_shot", **base_cam_collect.shot_event(
            name, base_cam.UNKNOWN_KIND, hold, files, from_pose, shot=self.shots, spot=where, turn=self.turn_no,
            review_first=True, filed_by=filed_by, guess=guess["name"] if guess else None,
            guess_confidence=guess["confidence"] if guess else None))

    def copies_over(self, name=None):
        """Tiles with more than 4 copies in the hand, the robot's discards and `name`."""
        return game.too_many([*self.hand.values(), *self.discards, *([name] if name else [])])

    def read(self, frames, from_pose):
        """The drawn tile's name: a person's word on the reader's (crude_run's window), or unattended the reader's own."""
        guess = self.guess(frames, from_pose)
        where = f"draw t{self.turn_no}"
        if self.unattended and guess is not None:
            name = guess["name"]
            print(f"  Read {name} ({guess['confidence']:.0%})")
            row = {"turn": self.turn_no, "guess": name, "name": name, "face_hidden": False, "guess_confidence": guess["confidence"], "by": "reader"}
            self.reads.append(row)
            self.rig.session.event("crude_read", **row)
            extra = self.copies_over(name)
            if extra:
                self.rig.session.intervention("fd_conflict", turn=self.turn_no, name=name, over=extra)
            self.save_hold(frames, from_pose, name, guess, where, "reader")
            return name
        if self.unattended:     # no read to take: the camera never saw it, or the photos had no tile box
            name = self.ask_name("The reader has no name for the drawn tile on the drop zone: lift it, look, put it back face down")
            if name is not None:
                self.reads.append({"turn": self.turn_no, "guess": None, "name": name, "face_hidden": False, "guess_confidence": None, "by": "person"})
                self.rig.session.event("crude_read", **self.reads[-1])
        else:
            name = self.confirm(frames, guess)
        if name is None:
            return None
        if self.reads[-1]["guess"] != name:
            self.correct(where, guess["name"] if guess else None, name, "draw")
        self.save_hold(frames, from_pose, name, guess, where, "person")
        return name

    # --- the hand ---

    def lay_hand(self):
        """The window that shows where the 13 tiles, the wall and the clear zone go. False if they quit."""
        spots = [self.spot_of[s] for s in self.slots]
        lines = face_down_play.lay_lines(self.station["id"], self.args.wall_stacks, self.slots)
        drain_stdin()
        if self.rig.realsense is None or self.args.no_view:
            return self.ask("\n".join(lines) + "\nEnter when done (q = quit): ").strip().lower() != "q"
        print("\n".join(lines) + "\n  (in the RealSense window: Space/Enter when done, q quit)")
        rs = self.rig.realsense
        try:
            while True:
                cv2.imshow(LAY_WINDOW, face_down_play.lay_hand_frame(rs.snapshot(), spots, self.station, self.zone_spot,
                                                                    rs.intrinsics, rs.T_cam_to_base, lines))
                key = cv2.waitKey(30) & 0xFF
                if key in (13, 10, 32):
                    return True
                if key in (ord("q"), 27):
                    return False
        finally:
            face_down_rig.close_windows()

    def register_slot(self, slot):
        """Pick the slot's tile, show it, put it back. Returns (frames, poses), or None if a person quit."""
        spot = self.spot_of[slot]
        pick = self.pick("register", spot, f"reg_{slot}")
        if pick is None:
            return ([], []) if self.by_hand("register", f"{slot} wasn't picked", f"Leave the tile on {slot}: you'll be asked its name") else None
        frames, from_pose = self.show_held(pick, "register_show")
        self.rig.arm.rest()
        if not self.jaws_have_tile():
            done = self.by_hand("register_show", f"{slot}'s tile came out of the jaws", spot=spot, where=slot)
        elif not self.set_down("register_return", pick, spot, spot):
            done = self.by_hand("register_return", "not set back", spot=spot, where=slot)
        else:
            done = True
        return (frames, from_pose) if done else None

    def register(self):
        """Every slot picked, shown and put back, then read. Returns {slot: (frames, poses, guess)}, or None if a person quit."""
        self.turn_no = 0
        seen = {}
        for n, slot in enumerate(self.slots, 1):
            print(f"\n--- Reading the hand: {slot} ({n} of {len(self.slots)})")
            got = self.register_slot(slot)
            if got is None:
                return None
            frames, from_pose = got
            guess = self.guess(frames, from_pose, slot=slot) if frames else None
            self.rig.session.event("fd_register_read", slot=slot, guess=guess, photos=len(frames))
            if guess is not None:
                print(f"  {slot}: {guess['name']} ({guess['confidence']:.0%})")
            seen[slot] = (frames, from_pose, guess)
            self.registered += 1
        return seen

    def name_the_hand(self, seen):
        """The 13 names: the reader's, and a person's where it has none. Then, attended, the hand check on all 13. False if they quit."""
        for slot in self.slots:
            guess = seen[slot][2]
            if guess is not None:
                self.hand[slot] = guess["name"]
                continue
            name = self.ask_name(f"The reader has no name for {slot}: lift its tile, look, put it back face down on its outline")
            if name is None:
                return False
            self.hand[slot] = name
            self.correct(slot, None, name, "register")
        if self.unattended:
            extra = self.copies_over()
            if extra:
                self.rig.session.intervention("fd_conflict", turn=0, name=None, over=extra)
            fixed = {}
        else:
            fixed = self.confirm_hand(seen)
            if fixed is None:
                return False
        for slot in self.slots:
            frames, from_pose, guess = seen[slot]
            by = "reader" if self.unattended and guess is not None else "person"       # a name a person typed isn't the reader's
            self.save_hold(frames, from_pose, self.hand[slot], guess, slot, by)
        self.rig.session.event("fd_hand_confirmed", hand=dict(self.hand), fixed=fixed, by="reader" if self.unattended else "person")
        return True

    def confirm_hand(self, seen):
        """A person's check of all 13 before the first draw, on one sheet of photos. {slot: name} fixed, or None if they quit."""
        photos = {slot: face_down_play.tile_photo(frames, poses, self.boxes) for slot, (frames, poses, _) in seen.items()}
        ids = {s.upper(): s for s in self.slots}
        fixed = {}
        while True:
            guesses = {slot: seen[slot][2] for slot in self.slots}
            if self.window is not None:
                self.window.show(face_down_play.hand_sheet(self.slots, photos, self.hand, guesses))
            print("\nThe robot's hand: " + "  ".join(
                f"{s}={self.hand[s]}" + (f"({guesses[s]['confidence']:.0%})" if guesses[s] and guesses[s]["name"] == self.hand[s] else "") for s in self.slots))
            drain_stdin()
            answer = self.ask("Check it against the photos. Enter if every name is right, or the wrong ones as SLOT=NAME (S3=5C S9=EW), q = quit: ").strip()
            if answer.lower() == "q":
                return None
            fixes = parse_fixes(answer, ids) if answer else {}
            if fixes is None or face_down_play.bad_names(fixes):
                print(f"  As SLOT=NAME, slots {self.slots[0]}-{self.slots[-1]}, names like 5C, 3D, 9B, EW, RD"
                      + (f" ({', '.join(face_down_play.bad_names(fixes))} isn't one)" if fixes else ""))
                continue
            for slot, name in fixes.items():
                if name != self.hand[slot]:
                    self.correct(slot, self.hand[slot], name, "hand check")
                    self.hand[slot] = fixed[slot] = name
            extra = self.copies_over()
            if extra:
                print(f"  That's more than {game.COPIES} of {', '.join(extra)}: one of them is misread. Correct it to go on.")
                continue
            if not fixes:
                return fixed

    # --- the turns ---

    def checkpoint(self, phase="play"):
        self.phase = phase
        state = {"phase": phase, "hand": self.hand, "discards": self.discards, "turn": self.turn_no, "unattended": self.unattended}
        path = self.rig.session.dir / "checkpoint.json"
        path.with_suffix(".tmp").write_text(json.dumps(state, indent=2) + "\n")
        path.with_suffix(".tmp").replace(path)

    def feed_wall(self):
        """The wall's next stack onto the station. Returns "" (fed), "e" (the wall's empty) or "q"."""
        answer = face_down_rig.prompt(self.rig, self.station, face_down_play.feed_step(self.station["id"]), self.ask, keys=face_down_play.END_KEY)
        return answer if answer in ("e", "q") else ""

    def after_turn(self):
        """Attended: a person's word after the turn. None to go on, or "end" / "quit"."""
        ids = {s.upper(): s for s in self.slots}
        drain_stdin()
        while True:
            answer = self.ask("Enter = next turn; S3=5C corrects a tile; end = the wall's empty; q = quit; or say what went wrong (a bump, a bad landing): ").strip()
            if not answer:
                return None
            if answer.lower() in ("end", "q"):
                return {"end": "end", "q": "quit"}[answer.lower()]
            fixes = parse_fixes(answer, ids) if "=" in answer else None
            if fixes is None:
                self.broke("person", answer)
                return None
            if face_down_play.bad_names(fixes):
                print(f"  {', '.join(face_down_play.bad_names(fixes))} isn't a tile name")
                continue
            tried = game.too_many([*{**self.hand, **fixes}.values(), *self.discards])
            if tried:
                print(f"  That would be more than {game.COPIES} of {', '.join(tried)} with the discards: not taken. Correct the other one too, in one answer.")
                continue
            for slot, name in fixes.items():
                if name != self.hand[slot]:
                    self.correct(slot, self.hand[slot], name, "after turn")
                    self.hand[slot] = name
            self.rig.session.event("fd_hand_fixed", turn=self.turn_no, fixes=fixes)
            self.checkpoint()

    def turn(self, plan):
        """One draw to its discard. Returns None to go on, or "win", "end" or "quit"."""
        self.turn_no = plan["turn"]
        print(f"\n=== Turn {self.turn_no}: {crude_run.hand_text(self.slots, self.hand)}")
        self.rig.session.event("fd_turn_start", turn=self.turn_no, slots=self.slots, hand=dict(self.hand), discards=list(self.discards))
        if plan["feed_before"]:
            answer = self.feed_wall()
            if answer:
                return {"e": "end", "q": "quit"}[answer]
        got = self.draw(plan)
        if got is None:
            return "quit"
        name = self.read(*got)
        if name is None:
            return "quit"
        outcome = self.play_it(name)
        if outcome == "quit":
            return "quit"
        self.done += 1
        if outcome == "win":
            self.checkpoint("won")
            return "win"
        self.checkpoint()
        return None if self.unattended else self.after_turn()

    def play(self, resumed=None):
        """The whole hand. Returns how it ended: "win", "end" (the wall's empty), or "quit"."""
        if resumed is None:
            if not self.lay_hand():
                return "quit"
            seen = self.register()
            if seen is None or not self.name_the_hand(seen):
                return "quit"
            self.checkpoint()
            start = 1
        else:
            folder, state = resumed
            self.hand, self.discards = dict(state["hand"]), list(state["discards"])
            start = state["turn"] + 1
            self.rig.session.event("fd_resumed", from_session=folder.name, turn=state["turn"], hand=self.hand)
            print(f"Resuming {folder.name} at turn {start}: {crude_run.hand_text(self.slots, self.hand)}")
            self.checkpoint()
        for plan in face_down_play.turn_plan(start):
            result = self.turn(plan)
            if result is not None:
                if result == "end":
                    self.checkpoint("end")
                return result

    def audit(self, result):
        """The end check: a person turns the 13 over and says which the robot had wrong (silent errors). Returns (grade, checked, wrong)."""
        if not self.hand:
            return None, False, {}
        print("\nThe robot's hand: " + "  ".join(f"{s}={self.hand[s]}" for s in self.slots))
        ids = {s.upper(): s for s in self.slots}
        drain_stdin()
        while True:
            answer = self.ask("Turn the 13 over and check them. Enter if every one matches, the wrong ones as SLOT=NAME (S3=5C), or s to skip: ").strip()
            if answer.lower() == "s":
                checked, wrong = False, {}
                break
            wrong = parse_fixes(answer, ids) if answer else {}
            if wrong is not None and not face_down_play.bad_names(wrong):
                checked = True
                break
            print(f"  As SLOT=NAME, slots {self.slots[0]}-{self.slots[-1]}")
        wrong = {slot: name for slot, name in wrong.items() if name != self.hand[slot]}
        for slot, name in wrong.items():
            self.rig.session.intervention("silent_error", slot=slot, robot=self.hand[slot], table=name)
        self.rig.session.event("fd_audit", checked=checked, wrong=wrong, hand=dict(self.hand))
        found = [i for i in interventions.derive(read_events(self.rig.session.dir)) if i["kind"] != "killed"]
        grade = interventions.hand_grade(found)
        return grade, checked, wrong


def report(arg):
    folder = find_session(arg)
    reads = face_down_play.scored_reads(folder)
    changes = face_down_play.changed_choices(folder, reads)
    print("\n".join(face_down_play.report_lines(folder, reads, changes)))
    return {"reads": reads, "changes": changes}


def check_zone(zone, spots, layout_name):
    """play.py refuses a hand whose drop zone the bench hasn't checked; face down, scans can't check a carry, so this only warns."""
    test = markers.zone_test(zone["xy"], spots, layout.load(markers.ZONE_TEST_LAYOUT)["spots"], play_layout=layout_name)
    if not test["ok"]:
        print("WARNING: the drop zone isn't checked on the bench (markers.zone_test): " + "; ".join(test["problems"]))
    return test


def main(argv=None, ask=input):
    args = parse_args(argv)
    if args.report is not None:
        return report(args.report)
    privacy.require()           # the weights that blur faces in the base camera's photos: before any hardware
    spots = layout.load(args.layout)["spots"]
    slots = [s["id"] for s in spots]
    if len(slots) != game.HAND_SIZE:
        raise SystemExit(f"layout {args.layout} has {len(slots)} spots: a hand has {game.HAND_SIZE}")
    stations = {s["id"]: s for s in layout.load(wall_test.LAYOUT)["spots"]}
    if args.station not in stations:
        raise SystemExit(f"layout {wall_test.LAYOUT} has no station {args.station} (has: {', '.join(stations)})")
    if args.wall_stacks < 1:
        raise SystemExit("--wall-stacks has to be at least 1")
    poses = base_cam_collect.pose_names()
    if not poses:
        raise SystemExit(f"no {base_cam_collect.POSE_PREFIX}* poses in {config.CAMERA_POSES}: teach some with scripts/teleop_view.py")
    config.discard_pose(why="The discards go in the container")
    zone = markers.zone_spot()
    if zone is None:
        raise SystemExit("No drop zone saved for this RealSense calibration: teach it with play.py --teach-zone")
    reader = base_cam.load_reader()
    if reader is None and args.unattended:
        raise SystemExit(f"--unattended takes the reader's names, and there's no reader ({base_cam.READER}): base_cam_train.py --save-reader")
    resumed = load_checkpoint(args.resume) if args.resume else None
    station = stations[args.station]
    print(f"Reader: {base_cam.READER.name}, knows {len(reader.classes)} names" if reader else f"No reader ({base_cam.READER}): you'll name every tile")
    rig = face_down_rig.Rig(args, [station, zone, *spots], poses, sim_empty=[zone])   # the sim's zone starts clear, as the real one does
    player, window, result = None, None, "quit"
    try:
        conditions = base_cam.ask_lighting(ask)
        rig.open_session(face_down_play.PLAY_KIND, station=args.station, unattended=args.unattended, zone_xy=zone["xy"],
                         reader=base_cam.READER.name if reader else None, wall_stacks=args.wall_stacks,
                         resumed_from=resumed[0].name if resumed else None)
        rig.session.conditions = conditions
        rig.session.event("holdup_conditions", conditions=conditions, source="arm", order="play")     # base_cam._shots reads the light from it
        rig.session.event("zone_test", **check_zone(zone, spots, args.layout))
        if not (args.sim or args.no_view or args.unattended):
            window = hand_view.Window()
        rig.arm.rest()
        player = Player(rig, args, ask, slots, station, zone, reader, base_cam.tile_boxes(), window)
        print("\n" + face_down_play.FLOW + "\n")
        result = player.play(resumed)
    except KeyboardInterrupt:
        print("Interrupted.")
        if rig.session:
            rig.session.intervention("interrupted", command="play_face_down")
    finally:
        if window is not None:
            window.close()
        folder = rig.session.dir if rig.session else None
        rig.close()
        if rig.session and player:
            grade, checked, wrong = player.audit(result) if result in ("win", "end") else (None, False, {})
            waiting = len(base_cam.waiting_for_review(folder))
            s = face_down_play.summary(player.done, result, grade, args.unattended, player.registered + len(player.reads),
                                       player.corrected, sum(bool(b["by_hand"]) for b in player.breaks), len(wrong), checked, waiting)
            print("\n" + "\n".join(face_down_play.summary_lines(s) + face_down_play.break_lines(player.breaks)))
            (folder / "play_fd_summary.json").write_text(json.dumps({**s, "breaks_list": player.breaks, "hand": player.hand,
                                                                     "discards": player.discards}, indent=2) + "\n")
            rig.session.event("hand_end", **s, hand=player.hand, discards=player.discards)
        if rig.session:
            rig.session.close()
    return folder


if __name__ == "__main__":
    main()
