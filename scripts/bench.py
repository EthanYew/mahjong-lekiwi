"""Test bench for isolated pickups: closed-loop hover + an ACT policy, one tile at a time or in batches.

    1 hover          detect, choose a tile, hover over it (corrected, settled from above)
    2 pickup         hover, then the policy picks it up; parks at rest holding it
    3 return         put the held tile back where the jaws closed on it, then rest
    4 discard        carry the held tile to the taught discard pose and release it, then rest
    5 teach-discard  drive the arm over the discard container with the gamepad and save that pose
    6 trials         pickup -> success check -> return, N times on one tile; prints the success rate
    7 rest           lift if low, then park (never opens the gripper)
    8 note           log something you had to do (straightened tiles, put tiles back, ...) for the last attempt
    9 setup          the setup changed mid-session (lamp on, battery swapped): answer the setup questions again
   10 quit

Type a number or a name. During a pickup, Esc stops the policy, and → marks the moment without
stopping it ("something happened here": a hand on the tile or the base, a caught drop, or just "look
at this"); you're asked what afterwards, or at the end of a batch of trials. Clean attempts ask
nothing extra. Everything goes in a session folder (sessions/<time>_bench/): the terminal log,
events.jsonl (with a RealSense scan before and after each pickup, return and discard, so knocked
tiles and tiles put back by hand show up without anyone saying so; when a scan shows the mat changed
you're asked whether it really did, Enter for yes), and per attempt the wrist
frames, act_trace.npz and RealSense overview frames before and after. `python scripts/session_report.py`
summarises the latest one, interventions included (mahjong/interventions.py).

    python scripts/bench.py                         # interactive, default policy
    python scripts/bench.py --trials 5 --tile 8D    # batch, then quit
    python scripts/bench.py --sim --trials 3        # no hardware
    python scripts/bench.py --trials 15 --tile 4D --hover-perturb-mm 0,6x,-6x,6y,-6y

Every successful attempt also saves a hover-correction sample (source "bench": where the jaws closed
minus the detection) to data/hover_corrections.jsonl; it's only fitted once data/hover_model.json
lists "bench". --hover-perturb-mm cycles deliberate hover offsets through the trials, to test whether
the policy follows the tile or replays a fixed move from wherever it starts (session_report.py prints
the result); those attempts' samples are saved as excluded.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import config, interventions
from mahjong.arm import Arm
from mahjong.hover_correction import HoverCorrection, append_record, grasp_record
from mahjong.perception import choose_tile, drain_stdin
from mahjong.session import Session, ask_conditions


class Abort:
    """Esc, from lerobot's keyboard listener, as an Event-like flag for policy.run."""
    def __init__(self, events):
        self.events = events

    def clear(self):
        self.events["stop_recording"] = self.events["exit_early"] = self.events["rerecord_episode"] = False

    def is_set(self):
        return self.events["stop_recording"]


class Mark:
    """→ during ACT, from the same listener, as policy.run's `mark`. → sets only exit_early (← also
    sets it, so marks too); Esc sets it as well, but that's an abort."""
    def __init__(self, events):
        self.events = events

    def clear(self):
        self.events["exit_early"] = self.events["rerecord_episode"] = False

    def is_set(self):
        return self.events["exit_early"] and not self.events["stop_recording"]


class Bench:
    def __init__(self, args):
        self.args = args
        if args.sim:
            from mahjong.sim import SimCamera, SimPerception, SimPolicy, SimRobot
            mat = SimPerception()
            self.arm = Arm(SimRobot(on_release=mat.release), sleep=lambda s: None)
            self.camera, self.realsense, self.policy = SimCamera(), mat, SimPolicy(mat=mat)
            self.events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
            self.listener = None
            self.session = Session("bench", ask_setup=False, policy="sim", sim=True)
        else:
            from lerobot.common.control_utils import init_keyboard_listener
            from mahjong import hardware
            from mahjong.policy import Policy
            # hardware first: nothing (not even the session folder) is created until it's all there
            devices = hardware.connect(["arm", "wrist_camera", "realsense"])
            self.arm, self.camera, self.realsense = devices["arm"], devices["wrist_camera"], devices["realsense"]
            self.session = Session("bench", policy=args.policy or "default", sim=False)
            self.policy = Policy(args.policy)
            self.listener, self.events = init_keyboard_listener()
        self.abort, self.mark = Abort(self.events), Mark(self.events)
        if self.arm.at_connect:
            self.session.event("connect", **self.arm.at_connect)
        self.last_attempt = None    # attempt_dir of the latest attempt: what a note or an interruption belongs to
        self.to_review = []         # (attempt_dir, tile, run) marked or aborted during trials, asked at the end
        self.mat, self.scan_n = interventions.MatWatch(), 0
        self.scan_checks = []       # (detections event, proposal): changes on the mat a person hasn't confirmed yet
        self.batch = False          # during trials: questions wait for the end
        self.correction = HoverCorrection.load()
        # a simulated grasp says nothing about the real hover: keep its samples in the session folder
        self.samples_path = self.session.dir / "hover_corrections.jsonl" if args.sim else config.HOVER_CORRECTIONS
        # {"tile_name", "detection_xy", "return_xy", "approach_offset_m", "attempt_dir"} while holding a tile
        self.held = None
        print(f"Policy: {self.policy.id}, hover {self.policy.hover}\nHover correction: {self.correction.summary()}")
        self.session.event("bench_setup", policy=self.policy.id, hover=self.policy.hover,
                           correction=self.correction.summary(), hover_perturb_offsets_m=args.hover_perturb_mm)

    # --- pieces ---

    def scan(self, label, **fields):
        """Detect tiles and log them as a `detections` event: consecutive scans show what changed on the
        mat (interventions.MatWatch). A change waits for a person to confirm it (check_scans)."""
        tiles = self.realsense.detect_tiles()
        self.scan_n += 1
        scan = self.session.event("detections", label=label, scan=self.scan_n, **fields,
                                  tiles=[{k: t[k] for k in ("tile_name", "xy", "confidence", "reads")} for t in tiles])
        proposal = self.mat.propose(scan)
        if proposal:
            self.scan_checks.append((scan, proposal))
        self.mat.advance(scan, proposal, [proposal[0]] if proposal else [])
        return tiles

    def check_scans(self):
        """What the scans saw change, one question each (Enter: as they say; or a detection glitch)."""
        if self.scan_checks:
            drain_stdin()
        for scan, proposal in self.scan_checks:
            kinds = interventions.ask_scan(scan, proposal)
            self.session.event("scan_review", scan=scan["scan"], proposed=proposal[0], kinds=kinds)
        self.scan_checks = []

    def choose(self, tile_name=None):
        if not self.arm.is_at_rest():
            self.arm.rest()
        tiles = self.scan("choose")
        if not self.batch:
            self.check_scans()          # before the arm moves, while what happened is in front of you
        if not tiles:
            print("No tiles detected.")
            self.session.event("attempt_skipped", reason="no_tiles", tile=tile_name)
            return None
        if tile_name is None:
            drain_stdin()
        tile = choose_tile(tiles, tile_name=tile_name)
        if tile is None:
            if tile_name is not None:
                print(f"{tile_name} not detected (seen: {', '.join(t['tile_name'] for t in tiles)})")
            self.session.event("attempt_skipped", reason="cancelled" if tile_name is None else "not_detected",
                               tile=tile_name)
        return tile

    def hover_over(self, tile, perturb=(0.0, 0.0)):
        x, y = tile["xy"]
        hx, hy = self.correction.apply(x, y)
        if any(perturb):
            hx, hy = hx + perturb[0], hy + perturb[1]
            print(f"  hover perturbed by ({perturb[0] * 1000:+.0f}, {perturb[1] * 1000:+.0f})mm on purpose")
        h = self.policy.hover
        result = self.arm.hover(hx, hy, h["offset_m"], h["clearance_m"], h["gripper"])
        if result is None:
            print("Unreachable from the closed-loop hover.")
            self.session.event("attempt_skipped", reason="unreachable", tile=tile["tile_name"], hover_xy=[hx, hy])
        return result

    def snapshot(self, path):
        cv2.imwrite(str(path), self.realsense.snapshot())

    # --- commands ---

    def hover(self):
        if self.held:
            return print(f"Holding {self.held['tile_name']} — return or discard it first.")
        tile = self.choose()
        if tile and (result := self.hover_over(tile)):
            self.session.event("hover", tile=tile["tile_name"], detection_xy=tile["xy"], **result)

    def pickup(self, tile_name=None, ask=True, perturb=(0.0, 0.0)):
        """One attempt, from the hover moved by `perturb` (m). Returns True if the gripper ended up
        holding the tile."""
        if self.held:
            print(f"Holding {self.held['tile_name']} — return or discard it first.")
            return None
        tile = self.choose(tile_name)
        if tile is None:
            return None
        attempt = self.session.attempt_dir(tile["tile_name"])
        self.snapshot(attempt / "overview_before.jpg")
        hover = self.hover_over(tile, perturb)
        if hover is None:
            return None
        self.abort.clear()
        run = self.policy.run(self.arm, self.camera, attempt, abort=self.abort, mark=self.mark)
        self.arm.rest()
        holding, command, measured = self.arm.holding()
        self.snapshot(attempt / "overview_after.jpg")
        self.scan("after_pickup", attempt_dir=attempt.name, target_xy=list(tile["xy"]))
        success, review = holding, {}
        if ask:
            drain_stdin()
            answer = input(f"Picked it up? (gripper commanded {command:.0f}, measured {measured:.0f}: looks "
                           f"{'held' if holding else 'empty'}) [{'Y/n' if holding else 'y/N'}]: ").strip().lower()
            success = holding if not answer else answer.startswith("y")
            review = self.review(attempt.name, tile["tile_name"], run)
        elif run["aborted"] or run["marks_s"]:
            self.to_review.append((attempt.name, tile["tile_name"], run))
        self.session.event("attempt", tile=tile["tile_name"], policy=self.policy.id, detection_xy=tile["xy"],
                           hover_xy=hover["hover_xy"], hover_offset_m=hover["offset_m"],
                           hover_clearance_m=hover["clearance_m"], gripper_command=command, gripper_measured=measured,
                           gripper_says_held=holding, success=success, attempt_dir=attempt.name,
                           correction=self.correction.name, hover_perturb_xy=list(perturb),
                           hover_sag_xyz_m=hover.get("sag_xyz_m"), **run, **review)
        self.last_attempt = attempt.name
        if success and run["grasp_xy"]:
            sample = grasp_record("bench", tile["xy"], run["grasp_xy"], tile_name=tile["tile_name"],
                                  hover_xy=hover["hover_xy"], hover_perturb_xy=list(perturb), policy=self.policy.id,
                                  correction=self.correction.name, session=self.session.dir.name,
                                  attempt_dir=attempt.name)
            if any(perturb):
                sample["excluded"] = "hover perturbed on purpose (policy sensitivity test)"
            append_record(sample, self.samples_path)
        if holding:
            self.held = {"tile_name": tile["tile_name"], "detection_xy": list(tile["xy"]),
                         "return_xy": run["grasp_xy"] or hover["hover_xy"],
                         "approach_offset_m": hover["offset_m"], "attempt_dir": attempt.name}
        print(f"{tile['tile_name']}: {'picked up' if success else 'missed'}")
        return success

    def review(self, attempt_dir, tile_name, run):
        """Ask about an aborted or marked attempt: {abort_reason, mark_kind, mark_note}; {} (no question)
        for a clean one."""
        answers, what = {}, f"{attempt_dir} ({tile_name})"
        if run["aborted"] or run["marks_s"]:
            drain_stdin()
        if run["aborted"]:
            answers["abort_reason"] = interventions.ask_abort(what)
        if run["marks_s"]:
            when = ", ".join(f"{t:.1f}s" for t in run["marks_s"])
            answers["mark_kind"], answers["mark_note"] = interventions.ask_mark(f"{what}, marked at {when}")
        return {k: v for k, v in answers.items() if v}

    def return_tile(self):
        if not self.held:
            return print("Not holding a tile.")
        x, y = self.held["return_xy"]
        print(f"Returning {self.held['tile_name']} to ({x:.3f}, {y:+.3f})")
        released_at = self.arm.place(x, y, self.held["approach_offset_m"])
        if released_at is not None:
            self.session.event("return", tile=self.held["tile_name"], xy=[x, y], released_at_m=released_at)
            # the returned tile, where the camera saw it (return_xy is the arm's target, ~40mm from that)
            self.scan("after_return", attempt_dir=self.held["attempt_dir"], target_xy=self.held["detection_xy"])
            self.held = None

    def discard(self, ask=True):
        if not config.DISCARD_POSE.exists():
            return print("No discard pose yet — use teach-discard first.")
        pose = json.loads(config.DISCARD_POSE.read_text())["pose"]
        held = self.held or {}
        self.arm.go_to_pose(pose)
        self.arm.set_gripper(config.GRIPPER_RELEASE)
        self.arm.rest()
        self.session.event("discard", tile=held.get("tile_name"), attempt_dir=held.get("attempt_dir"))
        if held and ask:
            drain_stdin()
            if input("In the container? [Y/n]: ").strip().lower().startswith("n"):
                self.session.intervention("discard_missed", attempt_dir=held["attempt_dir"])
        self.scan("after_discard", attempt_dir=held.get("attempt_dir"))
        self.held = None

    def teach_discard(self):
        from mahjong import hardware
        import time
        try:
            gamepad = hardware.connect(["gamepad"], gamepad_kwargs={"device": self.args.device, "commit_s": 0.0})["gamepad"]
        except SystemExit:
            return print("teach-discard cancelled")
        self.arm.raise_from_rest()
        gamepad.start(self.arm)
        self.abort.clear()
        print("Drive over the discard container's opening, then → to save (Esc cancels).")
        while not (self.events["exit_early"] or self.events["stop_recording"]):
            t0 = time.time()
            gamepad.step(self.arm, 1.0 / config.FPS)
            time.sleep(max(0.0, 1.0 / config.FPS - (time.time() - t0)))
        gamepad.close()
        if not self.events["stop_recording"]:
            pose = {j: self.arm.commanded()[j] for j in config.ARM_JOINTS}
            config.DISCARD_POSE.write_text(json.dumps({"pose": pose, "session": self.session.dir.name}, indent=2) + "\n")
            self.session.event("discard_pose_taught", pose=pose)
            print(f"Saved {config.DISCARD_POSE}")
        self.arm.rest()

    def trials(self, n=None, tile_name=None):
        if n is None:
            drain_stdin()
            n = int(input("How many trials? ").strip() or 0)
            tile_name = input("Which tile (name)? ").strip() or None
        results = []
        offsets = self.args.hover_perturb_mm or [(0.0, 0.0)]
        self.batch = True
        try:
            for i in range(n):
                perturb = offsets[i % len(offsets)]
                print(f"\n=== trial {i + 1}/{n} ===")
                success = self.pickup(tile_name, ask=False, perturb=perturb)
                if success is None:
                    print("Couldn't attempt (tile not found or unreachable) — stopping the trials.")
                    break
                results.append(success)
                if self.held:
                    self.return_tile()
        finally:        # even when interrupted: what you marked is still worth a word
            rate = f"{sum(results)}/{len(results)}"
            self.session.event("trials_done", tile=tile_name, policy=self.policy.id, results=results)
            print(f"\n{tile_name or 'trials'}: {rate} picked up")
            self.batch = False
            self.review_queued()
            self.check_scans()

    def review_queued(self):
        """The marked or aborted trials, one question each (clean ones ask nothing)."""
        if self.to_review:
            print(f"\n{len(self.to_review)} attempt(s) to say a word about:")
        for attempt_dir, tile_name, run in self.to_review:
            answers = self.review(attempt_dir, tile_name, run)
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
        print(f"Logged: {interventions.KINDS[kind][0]} ({interventions.class_of(kind)})")

    def setup(self):
        """The setup changed mid-session: ask the setup questions again, the current answers as defaults."""
        drain_stdin()
        conditions = ask_conditions(self.session.conditions, confirm=False)
        notes = input("Anything else to note (Enter to skip): ").strip()
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
        finally:
            self.arm.robot.disconnect()
            self.camera.release()
            self.realsense.close()
            if self.listener:
                self.listener.stop()
            self.session.close()


def parse_perturb(text):
    """"0,6x,-6x,6y,-6y" -> [(0, 0), (0.006, 0), (-0.006, 0), (0, 0.006), (0, -0.006)] (m)"""
    offsets = []
    for item in filter(None, (s.strip().lower() for s in text.split(","))):
        if item in ("0", "0x", "0y"):
            offsets.append((0.0, 0.0))
        elif item[-1] in "xy":
            mm = float(item[:-1]) / 1000
            offsets.append((mm, 0.0) if item[-1] == "x" else (0.0, mm))
        else:
            raise argparse.ArgumentTypeError(f"{item!r}: use e.g. 6x or -6y")
    return offsets


COMMANDS = ["hover", "pickup", "return", "discard", "teach-discard", "trials", "rest", "note", "setup", "quit"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--policy", default=None, help="policy id from data/policies.json (default: its 'default')")
    p.add_argument("--trials", type=int, default=None, help="run this many pickup/return trials, then quit")
    p.add_argument("--tile", default=None, help="tile name for --trials")
    p.add_argument("--sim", action="store_true", help="simulated arm, camera, perception and policy")
    p.add_argument("--device", default=None, help="gamepad evdev path (teach-discard)")
    p.add_argument("--hover-perturb-mm", type=parse_perturb, default=None,
                   help="hover offsets the trials cycle through, e.g. 0,6x,-6x,6y,-6y (policy sensitivity test)")
    args = p.parse_args()
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


if __name__ == "__main__":
    main()
