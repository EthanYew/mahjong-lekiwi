"""The crude full run (docs/face_down_plan.md, test 8): draw from the wall, show the tile to the base camera, read it, a person checks, then
discard or keep. Its job is a list of where it breaks, so nothing here tries to be robust: a step that fails is written down and a person does it.
scripts/crude_run.py runs this on the arm.
"""
import cv2
import numpy as np

from . import base_cam, base_cam_collect, camera, config, face_down_rig, game, layout, privacy, takeover, wall_test
from .perception import drain_stdin

SHALLOW_MIN_READING = 2.0   # jaws shut on nothing read 0.6-0.9. A held tile's reading fell from 9.5 to 3.0 over 2026-11-09, under holds_tile's margin (not a shallow grip, as first thought)
FLOW = ("Each turn: you feed the wall, the arm draws a tile, shows it to the base camera and sets it on the drop zone, you confirm its name, then the arm plays the discard.\n"
        "A line starting 'Noted' goes on the end-of-run list. It doesn't stop anything: the next prompt says what to do.")
STEPS = ("draw", "show", "set_on_zone", "read", "pick_from_hand", "discard", "pick_from_zone", "place_in_hand", "person")


def maybe_shallow(command, measured):
    """Jaws shut on something narrower than arm.holds_tile counts, or on nothing much: only a person can say which."""
    return command < config.GRIP_CLOSED_BELOW and SHALLOW_MIN_READING <= measured - command <= config.HOLDING_MARGIN


def parse_slots(text, slots):
    """Slot ids from a comma-separated list; SystemExit before any hardware for one the layout hasn't got."""
    wanted = [s.strip().upper() for s in text.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in slots]
    if unknown:
        raise SystemExit(f"--leave-empty: the layout has no slot {', '.join(unknown)} (has: {', '.join(slots)})")
    return wanted


def choose_allowed(candidates, slots, empty):
    """game.choose_discard's ranking without the slots left empty, whose tile isn't on the mat. The drawn tile is always allowed."""
    return next(c for c in candidates if c["index"] == game.HAND_SIZE or slots[c["index"]] not in empty)


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


def place_step(where):
    """What a person does when the arm couldn't put a tile on `where`; the window outlines that place."""
    return {"lines": [f"[by hand] Put the tile on {where}:", "GREEN side up, inside the outline", "Take it from the jaws first if it's still there"],
            "picture": [["G"]]}


def hand_text(slots, hand, empty=()):
    return ", ".join(f"{s} {hand[s]}" + (" (leave this slot EMPTY, tile aside)" if s in empty else "") for s in slots)


def summary(turns_done, breaks, reads, shallow=0):
    """What the run found: turns finished, breaks per step, shallow grips, and the reader's score on the tiles a person named."""
    per_step = {s: sum(b["step"] == s for b in breaks) for s in STEPS if any(b["step"] == s for b in breaks)}
    scored = [r for r in reads if r["guess"] is not None]
    return {"turns_done": turns_done, "breaks": len(breaks), "breaks_per_step": per_step, "by_hand": sum(bool(b["by_hand"]) for b in breaks), "shallow_grips": shallow,
            "reads": len(reads), "reader_guesses": len(scored), "reader_right": sum(r["guess"] == r["name"] for r in scored),
            "faces_not_visible": sum(bool(r["face_hidden"]) for r in reads)}


def summary_lines(result, breaks):
    lines = [f"{result['turns_done']} turn(s) finished, {result['breaks']} break(s), {result['by_hand']} done by hand, {result['shallow_grips']} shallow grip(s).",
             f"Reader: right on {result['reader_right']} of {result['reader_guesses']} guesses; a person named all {result['reads']}; face not visible on {result['faces_not_visible']}."]
    lines += ["Where it broke:"] + [f"  turn {b['turn']} {b['step']}: {b['what']}" + (" (done by hand)" if b["by_hand"] else "") for b in breaks] if breaks else ["Nothing broke."]
    return lines


# --- the run itself (scripts/crude_run.py; scripts/play_face_down.py builds on it) ---

WINDOW = "what the base camera saw"


def photos_with_guess(frames, guess, typed):
    """The held tile's photos side by side, the reader's guess, and what's been typed so far."""
    row = np.hstack([cv2.resize(f, (320, 240)) for f in frames[:4]])
    said = f"Reader: {guess['name']} ({guess['confidence']:.0%}).  Enter accepts, or type the name" if guess else "No reader guess: type the name"
    cv2.putText(row, f"{said}, X = face not visible", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(row, f"> {typed}", (8, row.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return row


def type_in_window(frames, guess, show=None, wait_key=None):
    """What a person types over the photos (kept in the window: nothing else would pump it). "" = Enter, None = Esc."""
    show = show or (lambda image: cv2.imshow(WINDOW, image))
    wait_key = wait_key or (lambda: cv2.waitKey(50) & 0xFF)
    typed = ""
    while True:
        show(photos_with_guess(frames, guess, typed))
        key = wait_key()
        if key == 255:          # no key this time round
            continue
        if key in (13, 10):
            return typed
        if key == 27:
            return None
        if key in (8, 127):
            typed = typed[:-1]
        elif chr(key).isalnum():
            typed += chr(key).upper()


class Run:
    """One crude run: the hand by memory, the wall's station, the drop zone, and everything that went wrong."""

    def __init__(self, rig, args, ask, slots, hand, station, zone_spot, reader, boxes, empty=()):
        self.rig, self.args, self.ask, self.slots, self.hand = rig, args, ask, slots, hand
        self.station, self.zone_spot, self.reader, self.boxes, self.empty = station, zone_spot, reader, boxes, empty
        self.shallow = 0
        self.rng = base_cam_collect.rng_for(0)
        self.pose = config.discard_pose()
        self.spot_of = {s["id"]: s for s in layout.load(args.layout)["spots"]}
        self.turn_no, self.breaks, self.reads, self.discards, self.done = 0, [], [], [], 0

    def broke(self, step, what, by_hand=False):
        entry = {"turn": self.turn_no, "step": step, "what": what, "by_hand": by_hand}
        self.breaks.append(entry)
        self.rig.session.event("crude_break", **entry)
        print(f"  Noted for the list: {step}: {what}" + (" (by hand)" if by_hand else ""))

    def by_hand(self, step, what, instruction=None, spot=None, where=None):
        """A person does what the arm couldn't: to `spot` (`where`, in words), outlined in the window, or as `instruction` says. False if they quit."""
        self.broke(step, what, by_hand=True)
        drain_stdin()
        if spot is not None:
            return face_down_rig.prompt(self.rig, spot, place_step(where), self.ask) != "q"
        return self.ask(f"{instruction}, then Enter (q = quit): ").strip().lower() != "q"

    def jaws_have_tile(self):
        """Arm.holding, or the shallow reading a person has said can be a tile."""
        holding, command, measured = self.rig.arm.holding()
        return holding or maybe_shallow(command, measured)

    def tile_in_jaws(self, pick):
        """The jaws read shut on something narrower than a tile: a person says whether there's one there."""
        drain_stdin()
        while True:
            answer = self.ask(f"The jaws read {pick['gripper_measured']:.1f}, too low for the held check. Is there a tile in the jaws? y/n: ").strip().lower()
            if answer in ("y", "n", "q"):       # q: treated as no, and the next prompt can quit
                return answer == "y"

    def pick(self, step, spot, label, raise_m=0.0, back_m=0.0):
        """The pick at `spot`, tried twice unless a person says otherwise. Its result with the tile held at rest, else None."""
        tries, camera_reruns = 0, camera.RERUNS
        while tries < 2:
            pick = self.rig.pick(label, spot, raise_m=raise_m, back_m=back_m)
            if pick.get("unreachable"):
                self.broke(step, f"{spot['id']} is out of the hover's reach")
                return None
            shallow = bool(not pick["holding"] and maybe_shallow(pick["gripper_command"], pick["gripper_measured"]) and self.tile_in_jaws(pick))
            if pick["run"]["stop_reason"] == "camera":
                from mahjong import hardware
                self.rig.wrist = hardware.reopen_wrist_camera(self.rig.wrist)
                # says nothing about the pick. A tile in the jaws stays: another hover would open them. Rerun once, as the bench does:
                # a camera that keeps tearing reopens without a prompt, so after that it's a miss and a person is asked
                if not pick["holding"] and not shallow and camera_reruns:
                    camera_reruns -= 1
                    continue
            self.rig.session.event("crude_pick", turn=self.turn_no, step=step, spot=spot["id"], held=bool(pick["holding"] or shallow), shallow=shallow, jaw_reading=pick["gripper_measured"],
                                   stop_reason=pick["run"]["stop_reason"], grasp_clearance_m=takeover.policy_view(pick["run"]).get("grasp_clearance_m"),
                                   takeover_s=pick["run"].get("takeover_s"), hover_raise_m=raise_m, hover_back_m=back_m, attempt_dir=pick["attempt"].name)
            if shallow:
                self.shallow += 1
                print("  A tile in the jaws, on a low reading: carrying on with it.")
            if pick["holding"] or shallow:
                return pick
            tries, camera_reruns = tries + 1, camera.RERUNS      # the next try gets its own rerun
            self.broke(step, f"missed at {spot['id']} (stop: {pick['run']['stop_reason']})")
            drain_stdin()
            if tries < 2 and self.ask("Check the tile is where it should be. Enter = try again, h = do this one by hand: ").strip().lower() == "h":
                return None
        return None

    def set_down(self, step, pick, picked_from, spot, layer="flat"):
        """Set the held tile on `spot`, aimed as a return is. False if it couldn't be."""
        placed, (x, y) = self.rig.set_down(pick, picked_from, spot, layer)
        self.rig.session.event("crude_set_down", turn=self.turn_no, step=step, spot=spot["id"], asked_xy=[x, y], reachable=placed is not None)
        if placed is None:
            self.broke(step, f"couldn't reach {spot['id']} to set it down")
            return False
        return True

    def to_container(self, step, what):
        """The held tile into the discard container."""
        if self.jaws_have_tile():
            self.rig.arm.discard(self.pose)
        else:
            self.broke(step, f"lost {what} before the container")

    def draw(self, plan):
        """Draw, show and set on the zone. Returns the photos and which pose each is from. None if a person quit."""
        layer = plan["layer"]
        raise_m = wall_test.HOVER_RAISE_M[layer]
        back_m = self.args.top_back_mm / 1000 if layer == "top" else 0.0
        pick = self.pick("draw", self.station, f"t{self.turn_no}_draw_{layer}", raise_m, back_m)
        if pick is None:
            return ([], []) if self.by_hand("draw", "the wall's tile wasn't drawn", spot=self.zone_spot, where="the drop zone") else None
        frames, from_pose, problem = self.rig.show(self.rng, still_held=self.jaws_have_tile)
        for i, (frame, pose) in enumerate(zip(frames, from_pose)):
            cv2.imwrite(str(pick["attempt"] / f"base_{pose}_{i}.jpg"), privacy.blur_faces(frame)[0])
        if problem:
            self.broke("show", problem)
        self.rig.arm.rest()
        if not self.jaws_have_tile():
            return (frames, from_pose) if self.by_hand("show", "the tile came out of the jaws", spot=self.zone_spot, where="the drop zone") else None
        if not self.set_down("set_on_zone", pick, self.station, self.zone_spot, layer):
            return (frames, from_pose) if self.by_hand("set_on_zone", "not set down", spot=self.zone_spot, where="the drop zone") else None
        return frames, from_pose

    def read(self, frames, from_pose):
        """The robot's guess, logged first, then a person's word. Returns the name, or None if they quit."""
        return self.confirm(frames, self.guess(frames, from_pose))

    def guess(self, frames, from_pose, **fields):
        """The reader's name for the held tile's photos (base_cam.read_held), logged as `crude_guess`; None without a reader."""
        guess = base_cam.read_held(self.reader, frames, from_pose, self.boxes)
        self.rig.session.event("crude_guess", turn=self.turn_no, guess=guess, **fields)
        return guess

    def confirm(self, frames, guess):
        """A person's word on the reader's guess: Enter accepts, a name replaces it, X says the face was hidden. The name, or None if they quit."""
        hidden = False
        while True:
            if frames and self.rig.realsense is not None:      # a real run: the photos, in a window. The tile itself is lying face down by now
                typed = type_in_window(frames, guess)
                face_down_rig.close_windows()
                if typed is None:
                    return None
            else:
                if not frames:
                    question = "The camera never saw this tile. Look at it, type its name, then Enter (q = quit): "
                elif guess:
                    question = f"Reader says {guess['name']} ({guess['confidence']:.0%}). Enter accepts, or type the name, X = face not visible, q = quit: "
                else:
                    question = "No reader guess: type the tile's name, X = face not visible, q = quit: "
                typed = self.ask(question).strip().upper()
                if typed == "Q":
                    return None
            if typed == "X":
                hidden = True
                self.broke("read", "the tile's face wasn't visible to the base camera")
                guess = None        # a guess at a hidden face says nothing about the reader
                continue
            name = typed or (guess["name"] if guess else "")
            if game.is_playable(name):
                break
            print(f"  '{name}' isn't a tile name (for example 5C, 3D, 9B, EW, RD)." if name else "  There's no guess to accept here: type the tile's name.")
        row = {"turn": self.turn_no, "guess": guess["name"] if guess else None, "name": name, "face_hidden": hidden, "guess_confidence": guess["confidence"] if guess else None}
        self.reads.append(row)
        self.rig.session.event("crude_read", **row)
        return name

    def play_it(self, name):
        """game.py's choice, then the arm's moves. Returns "win", "ok" or "quit"."""
        hand13 = [self.hand[s] for s in self.slots]
        if game.is_win(hand13 + [name]):
            print(f"Drew {name}: the hand is complete. Win!")
            self.rig.session.event("crude_win", turn=self.turn_no, drawn=name)
            return "win"
        _, candidates = game.choose_discard(hand13, name, self.discards)
        best = choose_allowed(candidates, self.slots, self.empty)
        from_hand = best["index"] != game.HAND_SIZE
        slot = self.slots[best["index"]] if from_hand else None
        print(f"Drew {name}; discard {best['tile_name']}" + (f" from {slot}" if from_hand else " (the drawn tile)") + f". {best.get('shanten_after', '?')} away after it.")
        self.rig.session.event("crude_choice", turn=self.turn_no, drawn=name, discard=best["tile_name"], slot=slot, shanten_after=best.get("shanten_after"))
        if from_hand:
            spot = self.spot_of[slot]
            pick = self.pick("pick_from_hand", spot, f"t{self.turn_no}_{slot}")
            if pick is None:
                if not self.by_hand("pick_from_hand", f"{best['tile_name']} at {slot} wasn't picked", f"Put the tile on {slot} in the discard container yourself"):
                    return "quit"
            else:
                self.to_container("discard", best["tile_name"])
        pick = self.pick("pick_from_zone", self.zone_spot, f"t{self.turn_no}_zone")
        if pick is None:
            what = f"the drawn {name} wasn't picked off the zone"
            done = self.by_hand("pick_from_zone", what, spot=self.spot_of[slot], where=slot) if from_hand \
                else self.by_hand("pick_from_zone", what, "Put the tile from the drop zone in the discard container yourself")
            if not done:
                return "quit"
        elif from_hand:
            if not self.set_down("place_in_hand", pick, self.zone_spot, self.spot_of[slot]) and \
                    not self.by_hand("place_in_hand", "not set down", spot=self.spot_of[slot], where=slot):
                return "quit"
        else:
            self.to_container("discard", name)
        if from_hand:
            self.hand[slot] = name
        self.discards.append(best["tile_name"])
        return "ok"

    def wall_stacks(self, turns):
        """The stacks to lay at the start: --wall-stacks where a script has it (play has no turn count), else enough for the turns."""
        return getattr(self.args, "wall_stacks", None) or turns // 2 + 2

    def feed(self, plan, turns):
        station_id = self.station["id"]
        if plan["turn"] == 1:
            step = wall_test.wall_step(station_id, self.wall_stacks(turns)) if self.args.stacked else flat_wall_step(station_id, turns + 1)
        elif plan["feed_before"]:
            step = wall_test.feed_step(station_id) if self.args.stacked else flat_feed_step(station_id)
        else:
            return True
        return face_down_rig.prompt(self.rig, self.station, step, self.ask) != "q"

    def turn(self, plan, turns):
        """One draw to its discard. Returns False to stop."""
        self.turn_no = plan["turn"]
        print(f"\n=== Turn {self.turn_no} of {turns}: {hand_text(self.slots, self.hand, self.empty)}")
        if not self.feed(plan, turns):
            return False
        got = self.draw(plan)
        if got is None:
            return False
        name = self.read(*got)
        if name is None:
            return False
        outcome = self.play_it(name)
        if outcome == "quit":
            return False
        self.done += 1
        drain_stdin()
        note = self.ask("Did anything else go wrong this turn (a bump, a bad landing)? Enter = no, or say what: ").strip()
        if note:
            self.broke("person", note)
        return outcome != "win"
