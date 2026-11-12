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

    def __init__(self, args, spots, pose_list):
        self.args, self.spots, self.pose_list = args, spots, pose_list
        self.realsense = None           # opened only to show where to lay tiles (not --sim, --no-view or --empty)
        if args.sim:
            from mahjong import sim
            tiles = [("5C", *s["xy"]) for s in spots]
            devices = sim.connect(tiles=tiles, wrist_view=True)
            self.arm, self.wrist, self.mat = devices["arm"], devices["wrist_camera"], devices["realsense"]
            self.base = sim.SimBaseCamera()
            self.policy = sim.SimPolicy(mat=self.mat, policy_id=args.policy or "sim")
            self.takeover = takeover.make(getattr(args, "dagger_dataset", None), None)     # nothing to hand over in the sim
        else:
            from mahjong import hardware
            from mahjong.policy import Policy
            self.policy = Policy(args.policy)             # before connecting, so a mistyped id fails fast
            guided = not (args.no_view or args.empty)
            repo = getattr(args, "dagger_dataset", None)
            devices = hardware.connect(["arm", "wrist_camera", "base_camera"] + (["realsense"] if guided else [])
                                       + takeover.devices(repo), gamepad_kwargs=takeover.gamepad_kwargs())
            self.arm, self.wrist, self.base = devices["arm"], devices["wrist_camera"], devices["base_camera"]
            self.takeover = takeover.make(repo, devices.get("gamepad"))
            if self.takeover:
                self.takeover.expect_hover(self.policy.hover)
            self.realsense = devices.get("realsense")
        self.events = new_events()
        self.terminal = TerminalKeys(self.events)
        self.abort = Abort(self.events)
        self.correction = HoverCorrection.load()
        self.returns = None
        self.session = None
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
            if self.realsense is not None:
                self.realsense.close()
            if self.takeover:
                self.takeover.close()
        if holding:
            print("WARNING: the jaws still read as holding a tile. Take it from the arm before the next run.")

    # --- one hold ---

    def pick(self, name, spot, raise_m=0.0, back_m=0.0):
        """Hover over the spot (the layout's xy stands for a detection: the detector can't see a back) and run the policy.
        raise_m: a hover this much higher, for a tile on top of another; back_m: this much nearer the robot along x (wall_test.py).
        Returns {"hover", "run", "attempt", "holding", ...} or {"unreachable": True}. The arm is left holding at rest."""
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
            hover = self.arm.hover(hx, hy, h["offset_m"], h["clearance_m"], h["gripper"])
            if hover is None:
                return {"unreachable": True, "attempt": attempt}
            self.abort.clear()
            run = self.policy.run(self.arm, self.wrist, attempt, abort=self.abort,
                                  **({"takeover": self.takeover} if self.takeover else {}))
            squeezed = self.arm.squeeze()
        finally:
            self.terminal.stop()
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
                "gripper_command": command, "gripper_measured": measured}

    def show(self, rng, still_held=None):
        """Visit every pose with the tile in the jaws; returns (frames, which pose each is from, "" or why it stopped).
        Between poses the arm is checked: a tile that slipped ends the hold. still_held: that check, for a grip Arm.holding can't see."""
        still_held = still_held or (lambda: self.arm.holding()[0])
        frames, from_pose = [], []
        limits = self.arm.joint_limits()
        for name in self.pose_list:
            pose = base_cam_collect.jitter(camera_poses.get(name)["pose"], rng, self.args.jitter_deg)
            pose = {j: min(max(v, limits[j][0]), limits[j][1]) for j, v in pose.items()}     # a nudge never passes a joint's limit
            self.arm.go_to_pose(pose)
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
        self.arm.rest()
        placed = self.arm.place(x, y, pick["hover"]["offset_m"] - wall_test.HOVER_RAISE_M[layer],
                                grasp_clearance_m=wall_test.above_own_level_m(pick["run"].get("grasp_clearance_m"), layer))
        return placed, (x, y)


GUIDE_WINDOW = "lay the tile (RealSense view)"


def close_windows():
    """Close every OpenCV window now: with the arm moving nothing pumps them, and the desktop marks them 'not responding'."""
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def guide(realsense, placement, spot, sign, show=None, wait_key=None, text=None, overlay=None):
    """A live RealSense window until the tile is laid: the spot's outline, the end where the picture's top goes, the tile's name.
    Returns "" (laid, go), "s" or "q". The caller closes the window before the arm moves (close_windows): nothing pumps it then.
    `text` replaces the default title; `overlay(frame)` draws on each frame (wall_test.py's picture of what to lay)."""
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
        answer = {13: "", 10: "", 32: "", ord("s"): "s", ord("q"): "q", 27: "q"}.get(key)
        if answer is not None:
            return answer
