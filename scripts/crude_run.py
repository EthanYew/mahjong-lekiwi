"""The crude full run on the arm (docs/face_down_plan.md, test 8): draw a face-down tile from the wall, show it to the base camera, read it, you check
the read, then the arm plays the discard. Crude on purpose: the point is a list of where it breaks, so a step that fails is written down and you do it by hand.

    python scripts/crude_run.py --hand 1C,2C,3C,5C,5C,7C,7C,9C,9C,1B,1B,2C,3C --turns 6     # a wall one tile high (where picks have worked)
    python scripts/crude_run.py --hand ... --turns 6 --stacked --top-back-mm 10             # two high: top, bottom, feed
    python scripts/crude_run.py --hand ... --sim                                            # simulated arm and cameras, no hardware

--hand is the 13 names you lay GREEN side up on the layout's slots, in the layout's order: the robot goes by memory after that. The wall's tiles can be anything
you can check, but the reader only knows the tiles it was trained on. A turn: pick from the station, hold the tile up at every saved base_view pose, set it on the drop zone
(so the arm doesn't stand holding it while you read), then a window shows the photos and the reader's guess. Enter accepts it, or type the name; X says the face wasn't visible
(then type the name, from the tile). game.py chooses the discard. Then it picks that tile from its slot into the container and the drawn tile from the zone into the slot,
or the drawn tile straight into the container. Nothing can see a green back, so every pick goes by saved position, and a landing is only as good as the aim.

Without assets/holdup_reader.pt it asks you for every name: train one with `base_cam_train.py --save-reader`. Results are in `sessions/<time>_crude/`: `crude_break` and `crude_read`
events, per attempt `wrist.mp4` and `base_*.jpg`, and `crude_summary.json`. Run `setup_check.py` first, and from the main checkout.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from mahjong import face_down_rig
import wall_test as wall_script
from mahjong import config, crude_run, game, base_cam, base_cam_collect, layout, markers, privacy, takeover, wall_test
from mahjong.perception import drain_stdin

WINDOW = "what the base camera saw"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--hand", required=True, help="the 13 tile names you lay green side up, comma separated, in the layout's slot order")
    p.add_argument("--turns", type=int, default=6, help="draws to try (default %(default)s)")
    p.add_argument("--layout", default="even-hand13", help="the layout of the hand's slots (default %(default)s)")
    p.add_argument("--station", default="F", help=f"the wall's station in layout {wall_test.LAYOUT} (default %(default)s)")
    p.add_argument("--stacked", action="store_true", help="a wall two high: top, bottom, then feed. Default: one tile high")
    p.add_argument("--top-back-mm", type=float, default=10.0, help="with --stacked: a top tile's hover starts this much nearer the robot (default %(default)s)")
    p.add_argument("--leave-empty", default="", metavar="SLOTS", help="hand slots with no tile on the mat, comma separated, e.g. S7 (its tile right behind the wall throws the draw off): still named in --hand, never chosen as a discard")
    p.add_argument("--policy", default=None, help="policy id (default: the registry's default)")
    p.add_argument("--sim", action="store_true", help="simulated arm, cameras and policy: no hardware")
    takeover.add_argument(p)
    p.set_defaults(empty=False, no_view=False, hover_lower_mm=None, frames=1, settle_s=1.0, gap_s=0.25, jitter_deg=0.0)       # what base_cam_collect's Rig reads
    return p.parse_args(argv)


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
            return wall_script.prompt(self.rig, spot, crude_run.place_step(where), self.ask) != "q"
        return self.ask(f"{instruction}, then Enter (q = quit): ").strip().lower() != "q"

    def jaws_have_tile(self):
        """Arm.holding, or the shallow reading a person has said can be a tile."""
        holding, command, measured = self.rig.arm.holding()
        return holding or crude_run.maybe_shallow(command, measured)

    def tile_in_jaws(self, pick):
        """The jaws read shut on something narrower than a tile: a person says whether there's one there."""
        drain_stdin()
        while True:
            answer = self.ask(f"The jaws read {pick['gripper_measured']:.1f}, too low for the held check. Is there a tile in the jaws? y/n: ").strip().lower()
            if answer in ("y", "n", "q"):       # q: treated as no, and the next prompt can quit
                return answer == "y"

    def pick(self, step, spot, label, raise_m=0.0, back_m=0.0):
        """The pick at `spot`, tried twice unless a person says otherwise. Its result with the tile held at rest, else None."""
        tries = 0
        while tries < 2:
            pick = self.rig.pick(label, spot, raise_m=raise_m, back_m=back_m)
            if pick.get("unreachable"):
                self.broke(step, f"{spot['id']} is out of the hover's reach")
                return None
            shallow = bool(not pick["holding"] and crude_run.maybe_shallow(pick["gripper_command"], pick["gripper_measured"]) and self.tile_in_jaws(pick))
            if pick["run"]["stop_reason"] == "camera":
                from mahjong import hardware
                self.rig.wrist = hardware.reopen_wrist_camera(self.rig.wrist)
                if not pick["holding"] and not shallow:      # says nothing about the pick. A tile in the jaws stays: another hover would open them
                    continue
            self.rig.session.event("crude_pick", turn=self.turn_no, step=step, spot=spot["id"], held=bool(pick["holding"] or shallow), shallow=shallow, jaw_reading=pick["gripper_measured"],
                                   stop_reason=pick["run"]["stop_reason"], grasp_clearance_m=takeover.policy_view(pick["run"]).get("grasp_clearance_m"),
                                   takeover_s=pick["run"].get("takeover_s"), hover_raise_m=raise_m, hover_back_m=back_m, attempt_dir=pick["attempt"].name)
            if shallow:
                self.shallow += 1
                print("  A tile in the jaws, on a low reading: carrying on with it.")
            if pick["holding"] or shallow:
                return pick
            tries += 1
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
        guess = base_cam.read_held(self.reader, frames, from_pose, self.boxes)
        self.rig.session.event("crude_guess", turn=self.turn_no, guess=guess)
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
        best = crude_run.choose_allowed(candidates, self.slots, self.empty)
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

    def feed(self, plan, turns):
        station_id = self.station["id"]
        if plan["turn"] == 1:
            step = wall_test.wall_step(station_id, turns // 2 + 2) if self.args.stacked else crude_run.flat_wall_step(station_id, turns + 1)
        elif plan["feed_before"]:
            step = wall_test.feed_step(station_id) if self.args.stacked else crude_run.flat_feed_step(station_id)
        else:
            return True
        return wall_script.prompt(self.rig, self.station, step, self.ask) != "q"

    def turn(self, plan, turns):
        """One draw to its discard. Returns False to stop."""
        self.turn_no = plan["turn"]
        print(f"\n=== Turn {self.turn_no} of {turns}: {crude_run.hand_text(self.slots, self.hand, self.empty)}")
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


def main(argv=None, ask=input):
    args = parse_args(argv)
    privacy.require()           # the weights that blur faces in the base camera's photos: before any hardware
    slots = [s["id"] for s in layout.load(args.layout)["spots"]]
    hand = crude_run.parse_hand(args.hand, slots)
    empty = crude_run.parse_slots(args.leave_empty, slots)
    stations = {s["id"]: s for s in layout.load(wall_test.LAYOUT)["spots"]}
    if args.station not in stations:
        raise SystemExit(f"layout {wall_test.LAYOUT} has no station {args.station} (has: {', '.join(stations)})")
    if args.turns < 1:
        raise SystemExit("--turns has to be at least 1")
    poses = base_cam_collect.pose_names()
    if not poses:
        raise SystemExit(f"no {base_cam_collect.POSE_PREFIX}* poses in {config.CAMERA_POSES}: teach some with scripts/teleop_view.py")
    config.discard_pose(why="The discards go in the container")
    zone = markers.zone_spot()
    if zone is None:
        raise SystemExit("No drop zone saved for this RealSense calibration: teach it with play.py --teach-zone")
    station = stations[args.station]
    all_spots = [station, zone, *layout.load(args.layout)["spots"]]
    reader = base_cam.load_reader()
    print(f"Reader: {base_cam.READER.name}, knows {', '.join(reader.classes)}" if reader else f"No reader ({base_cam.READER}): you'll name every tile. base_cam_train.py --save-reader trains one")
    rig = face_down_rig.Rig(args, all_spots, poses)
    run = None
    try:
        conditions = base_cam.ask_lighting(ask)
        rig.open_session("crude", hand=hand, turns=args.turns, station=args.station, stacked=args.stacked, zone_xy=zone["xy"], reader=base_cam.READER.name if reader else None, left_empty=empty)
        rig.session.conditions = conditions
        rig.session.event("crude_conditions", conditions=conditions)
        rig.arm.rest()
        run = Run(rig, args, ask, slots, hand, station, zone, reader, base_cam.tile_boxes(), empty)
        print("\n" + crude_run.FLOW + "\n")
        drain_stdin()
        if ask(f"Lay the hand GREEN side up: {crude_run.hand_text(slots, hand, empty)}. The drop zone stays clear. Enter when done (q = quit): ").strip().lower() != "q":
            for plan in crude_run.draw_plan(args.turns, args.stacked):
                if not run.turn(plan, args.turns):
                    break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        folder = rig.session.dir if rig.session else None
        rig.close()
        if rig.session and run:
            result = crude_run.summary(run.done, run.breaks, run.reads, run.shallow)
            print("\n" + "\n".join(crude_run.summary_lines(result, run.breaks)))
            (folder / "crude_summary.json").write_text(json.dumps({**result, "breaks_list": run.breaks}, indent=2) + "\n")
            rig.session.event("crude_done", **result)
        if rig.session:
            rig.session.close()
    return folder


if __name__ == "__main__":
    main()
