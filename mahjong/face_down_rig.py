"""The arm, its cameras, the policy and the corrections for the face-down work: picking a tile at a layout spot by its saved
position (the detector can't see a back), holding it up to the base camera at the taught poses, and setting it back down.
Shared by scripts/base_cam_collect.py (the base camera's training photos), scripts/wall_test.py and scripts/crude_run.py.

With --dagger-dataset (args.dagger_dataset), A on the gamepad during a pick hands a person the arm, and the recovery is saved
as a DAgger episode (mahjong/takeover.py).

`guide` is the live RealSense window that shows a person where to lay a tile; `close_windows` closes it before the arm moves.
"""
import cv2

from . import base_cam_collect, camera_poses, config, takeover, wall_test
from .camera import grab_frames
from .hover_correction import HoverCorrection
from .return_correction import ReturnCorrection
from .session import Session
from .terminal_keys import Abort, TerminalKeys, new_events


class Rig:
    """The arm, its cameras, the policy and the corrections: what picking a tile at a spot and showing it needs."""

    def __init__(self, args, spots, pose_list, sim_empty=()):
        """sim_empty: spots the simulated mat starts without a tile on (the drop zone: on the arm it's empty until a draw)."""
        self.args, self.spots, self.pose_list = args, spots, pose_list
        self.realsense = None           # opened only to show where to lay tiles (not --sim, --no-view or --empty)
        self.desk_camera = None         # the RealSense a desk video records (--record-desk): opened for it if not for the above
        if args.sim:
            from mahjong import sim
            tiles = [("5C", *s["xy"]) for s in spots if s not in sim_empty]
            devices = sim.connect(tiles=tiles, wrist_view=True)
            self.arm, self.wrist, self.mat = devices["arm"], devices["wrist_camera"], devices["realsense"]
            self.desk_camera = self.mat
            self.base = sim.SimBaseCamera()
            self.policy = sim.SimPolicy(mat=self.mat, policy_id=args.policy or "sim")
            self.takeover = takeover.make(getattr(args, "dagger_dataset", None), None)     # nothing to hand over in the sim
        else:
            from mahjong import hardware
            from mahjong.policy import Policy
            self.policy = Policy(args.policy)             # before connecting, so a mistyped id fails fast
            guided = not (args.no_view or args.empty)
            repo = getattr(args, "dagger_dataset", None)
            desk = getattr(args, "record_desk", False)
            devices = hardware.connect(["arm", "wrist_camera", "base_camera"] + (["realsense"] if guided or desk else [])
                                       + takeover.devices(repo), gamepad_kwargs=takeover.gamepad_kwargs())
            self.arm, self.wrist, self.base = devices["arm"], devices["wrist_camera"], devices["base_camera"]
            self.takeover = takeover.make(repo, devices.get("gamepad"))
            if self.takeover:
                self.takeover.expect_hover(self.policy.hover)
            self.realsense = devices.get("realsense") if guided else None
            self.desk_camera = devices.get("realsense")
        self.events = new_events()
        self.terminal = TerminalKeys(self.events)
        self.abort = Abort(self.events)
        self.correction = HoverCorrection.load()
        self.returns = None
        self.session = None
        # face-down play: from one move straight to the next, resting only before a person is asked something (crude_run.Run's
        # ask), the RealSense guide window (prompt), a miss and the end. Off (always via rest) for the collection and wall_test.
        self.direct = False
        self.view_map = None            # {laying: upright / turned}, learned from the person's answer on the session's first photo

    def open_session(self, kind, **setup):
        if self.takeover:
            setup["dagger_dataset"] = self.takeover.repo_id
        self.session = Session(kind, ask_setup=False, sim=self.args.sim, policy=self.policy.id, layout=self.args.layout, **setup)
        path = self.session.dir / "return_corrections.jsonl" if self.args.sim else config.RETURN_CORRECTIONS
        self.returns = ReturnCorrection.load(path)     # simulated returns land where they're set down: the real model doesn't apply
        if self.arm.at_connect:
            self.session.event("connect", **self.arm.at_connect)

    def close(self):
        holding = self.arm.holding()[0]
        try:
            self.arm.rest()
        finally:
            self.arm.disconnect()
            self.wrist.release()
            self.base.release()
            if not self.args.sim and self.desk_camera is not None:     # the same RealSense as self.realsense when both
                self.desk_camera.close()
            if self.takeover:
                self.takeover.close()
        if holding:
            print("WARNING: the jaws still read as holding a tile. Take it from the arm before the next run.")

    # --- one hold ---

    def pick(self, name, spot, raise_m=0.0, back_m=0.0):
        """Hover over the spot (the layout's xy stands for a detection: the detector can't see a back) and run the policy.
        raise_m: a hover this much higher, for a tile on top of another; back_m: this much nearer the robot along x (wall_test.py).
        Returns {"hover", "run", "attempt", "holding", ...} or {"unreachable": True}. The arm is left holding at rest, or with
        `direct` lifted clear over the spot (a miss, an abort or a takeover still rests: a person is asked next)."""
        x, y = spot["xy"]
        hx, hy = self.correction.apply(x, y)
        hx -= back_m
        h = dict(self.policy.hover)
        lower = (self.args.hover_lower_mm or 0.0) / 1000 - raise_m      # bench.py --hover-lower-mm: offset and clearance together
        h["offset_m"] -= lower
        h["clearance_m"] -= lower
        attempt = self.session.attempt_dir(name)
        self.terminal.start()
        try:
            hover = self.arm.hover(hx, hy, h["offset_m"], h["clearance_m"], h["gripper"], direct=self.direct)
            if hover is None:
                return {"unreachable": True, "attempt": attempt}
            self.abort.clear()
            run = self.policy.run(self.arm, self.wrist, attempt, abort=self.abort,
                                  **({"takeover": self.takeover} if self.takeover else {}))
            squeezed = self.arm.squeeze()
        finally:
            self.terminal.stop()
        if not (self.direct and not run.get("aborted") and run.get("takeover_step") is None and self.arm.holding()[0]
                and self.arm.lift_clear()):
            self.arm.rest()
        if self.takeover:       # a recovery a person showed: saved, and logged as a rescue, before anything else is asked
            self.takeover.save(run, self.session, self.policy.id, tile_name=name, detection_xy=[float(v) for v in spot["xy"]],
                               hover=dict(self.policy.hover), hover_xy=hover["hover_xy"], hover_offset_m=h["offset_m"],
                               hover_clearance_m=hover["clearance_m"], hover_raise_m=raise_m, hover_back_m=back_m,
                               hover_lower_mm=self.args.hover_lower_mm, hover_correction=self.correction.name,
                               attempt_dir=attempt.name, layout=self.args.layout, spot=spot["id"], spot_xy=spot["xy"],
                               face_down=True)
        holding, command, measured = self.arm.holding()
        return {"hover": hover, "run": run, "attempt": attempt, "holding": holding, "squeezed_from": squeezed,
                "gripper_command": command, "gripper_measured": measured, "offset_m": h["offset_m"]}

    def show(self, rng, still_held=None):
        """Visit every pose with the tile in the jaws; returns (frames, which pose each is from, "" or why it stopped).
        Between poses the arm is checked: a tile that slipped ends the hold. still_held: that check, for a grip Arm.holding can't see.
        With `direct`, the swing in from the mat and the swing back out both pass RAISED_REST (without folding down to rest):
        the poses are by the base camera, on the LeKiwi base, and a straight swing could cut across it (nothing models the base)."""
        if not self.direct:
            return self._show(rng, still_held)
        self.arm.to_raised_rest()
        shown = self._show(rng, still_held)
        self.arm.to_raised_rest()       # every normal way out, the early ones too ("no frames" goes on to the set-down); an
        return shown                    # exception or Ctrl-C moves nothing more: close() rests the arm

    def _show(self, rng, still_held):
        still_held = still_held or (lambda: self.arm.holding()[0])
        frames, from_pose = [], []
        limits = self.arm.joint_limits()
        for name in self.pose_list:
            pose = base_cam_collect.jitter(camera_poses.get(name)["pose"], rng, self.args.jitter_deg)
            pose = {j: min(max(v, limits[j][0]), limits[j][1]) for j, v in pose.items()}     # a nudge never passes a joint's limit
            self.arm.go_to_pose(pose, direct=self.direct)
            self.arm.sleep(self.args.settle_s)
            if not still_held():
                return frames, from_pose, f"the tile was lost at {name}"
            got = []
            for i in range(self.args.frames):
                if i:
                    self.arm.sleep(self.args.gap_s)
                got += grab_frames(self.base, 1)
            if not got:
                return frames, from_pose, "the base camera gave no frames"
            frames += got
            from_pose += [name] * len(got)
        return frames, from_pose, ""

    def set_down(self, pick, picked_from, to=None, layer="flat"):
        """Set the held tile on spot `to` (default: back on `picked_from`), aimed as a return is: `to` is its home and
        `picked_from`'s saved position stands for its detection. A `layer` "top" pick's raise is taken off, so it's let go at
        the mat and not a tile's height above it. Returns (Arm.place's result, None if out of reach; the (x, y) asked)."""
        to = to or picked_from
        grasp = pick["run"]["grasp_xy"] or pick["hover"]["hover_xy"]
        x, y = self.returns.target([float(v) for v in to["xy"]], [float(v) for v in picked_from["xy"]], grasp)
        if not self.direct:
            self.arm.rest()
        placed = self.arm.place(x, y, pick["hover"]["offset_m"] - wall_test.HOVER_RAISE_M[layer],
                                grasp_clearance_m=wall_test.above_own_level_m(pick["run"].get("grasp_clearance_m"), layer),
                                direct=self.direct, rest_after=not self.direct)
        return placed, (x, y)


GUIDE_WINDOW = "lay the tile (RealSense view)"


def close_windows():
    """Close every OpenCV window now: with the arm moving nothing pumps them, and the desktop marks them 'not responding'."""
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def guide(realsense, placement, spot, sign, show=None, wait_key=None, text=None, overlay=None, keys=""):
    """A live RealSense window until the tile is laid: the spot's outline, the end where the picture's top goes, the tile's name.
    Returns "" (laid, go), "s" or "q", or one of `keys` (letters the caller answers to, e.g. play_face_down's "e": the wall's
    empty). The caller closes the window before the arm moves (close_windows): nothing pumps it then.
    `text` replaces the default title; `overlay(frame)` draws on each frame (wall_test.py's picture of what to lay)."""
    answers = {13: "", 10: "", 32: "", ord("s"): "s", ord("q"): "q", 27: "q", **{ord(k): k for k in keys}}
    show = show or (lambda image: cv2.imshow(GUIDE_WINDOW, image))
    wait_key = wait_key or (lambda: cv2.waitKey(30) & 0xFF)
    text = text or (f"Lay {placement['tile']} FACE DOWN on {spot['id']}"
                    + (", then turn it 180" if placement["laying"] == "rotated" else ""))
    while True:
        out = base_cam_collect.guide_frame(realsense.snapshot(), spot, realsense.intrinsics, realsense.T_cam_to_base, text, sign)
        if overlay is not None:
            overlay(out)
        show(out)
        key = wait_key()
        answer = answers.get(key)
        if answer is not None:
            return answer


def draw_picture(frame, rows):
    """The step's side view in the window's bottom left: white and green tiles, the one the arm goes for outlined in yellow."""
    tile_w, tile_h, gap, left = 46, 24, 3, 76
    h = frame.shape[0]
    top = h - 40 - len(rows) * (tile_h + gap)
    cv2.rectangle(frame, (0, top - 26), (max(left + 4 * (tile_w + gap), 310), h - 30), (0, 0, 0), -1)
    cv2.putText(frame, "side view, yellow = the tile the arm goes for", (6, top - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "robot >", (6, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            x, y = left + c * (tile_w + gap), top + r * (tile_h + gap)
            colour = (255, 255, 255) if "W" in cell else (60, 150, 60)
            cv2.rectangle(frame, (x, y), (x + tile_w, y + tile_h), colour, -1)
            cv2.rectangle(frame, (x, y), (x + tile_w, y + tile_h), (0, 255, 255) if cell.startswith("[") else (130, 130, 130), 3 if cell.startswith("[") else 1)
    return frame


def prompt(rig, station, step, ask, keys=""):
    """Show what to lay, on the live RealSense window (or the terminal). Returns "", "s" or "q", or one of `keys` (guide)."""
    text = wall_test.step_text(step)
    rig.arm.rest()          # a person reaches in, and the RealSense window shows the mat: the arm out of both
    if rig.realsense is None:
        return ask(f"{text}\nEnter when done (s = skip, q = quit): ").strip().lower()
    print(f"{text}\n  (in the RealSense window: Space/Enter when done, s skip, q quit)")
    answer = guide(rig.realsense, {"tile": "", "laying": ""}, station, None, text="\n".join(step["lines"]),
                   overlay=lambda frame: draw_picture(frame, step["picture"]), keys=keys)
    close_windows()
    return answer
