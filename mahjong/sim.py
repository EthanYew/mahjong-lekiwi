"""A simulated follower arm for tests and `bench.py --sim`: same bus/action interface as lerobot's
SOFollower, with the sag behaviour seen on this arm.

Sag: shoulder_lift and elbow_flex settle below their command by an amount growing with reach, and
hold anywhere inside a friction band — the smaller sag when the command arrived from above, up to
twice that when it arrived from below — moving only when the command pushes them out of the band.
That reproduces both effects measured on hardware (2026-10-22): ~5-13mm of droop, and a raise that
ends lower than the hover it started from. The gripper stalls at GRIP_ON_TILE while holding a tile
and commanded tighter.
"""
import types

import numpy as np

from . import config, kinematics

SAG_DEG_AT_REACH = 1.6        # lift droop arriving from above, at 0.19m reach (~first row)
SAG_UP_FACTOR = 2.0
GRIP_ON_TILE = 11.0


class SimRobot:
    def __init__(self, start=None, holding=False, on_release=None):
        """on_release(tool xy, measured tip clearance m): called when the jaws open on a held tile."""
        self.command = dict(start or config.REST_POSE)
        self.holding = holding
        self.on_release = on_release
        self._measured_offset = None
        self.sent = []          # every command sent, in order
        self.clearances = []    # measured pincer-tip clearance (mm above the grasp point) after each
        self.bus = types.SimpleNamespace(sync_read=self._read)

    def _band(self):
        tip = kinematics.pincer_tip(self.command)
        lo = SAG_DEG_AT_REACH * (np.hypot(tip[0] - kinematics.SHOULDER_PAN_X, tip[1]) / 0.19) ** 2
        return {"shoulder_lift": (lo, SAG_UP_FACTOR * lo), "elbow_flex": (0.5 * lo, 0.5 * SAG_UP_FACTOR * lo)}

    def _settle(self):
        band = self._band()
        previous = self._measured_offset or {j: lo for j, (lo, _) in band.items()}
        self._measured_offset = {j: min(max(previous[j], lo), hi) for j, (lo, hi) in band.items()}

    def _read(self, register):
        if register == "Goal_Position":
            return dict(self.command)
        if self._measured_offset is None:
            self._settle()
        measured = dict(self.command)
        for joint, offset in self._measured_offset.items():
            measured[joint] += offset
        if self.holding and measured["gripper"] < GRIP_ON_TILE:
            measured["gripper"] = GRIP_ON_TILE
        return measured

    def send_action(self, action):
        previous_measured = self._read("Present_Position")
        for key, value in action.items():
            self.command[key.removesuffix(".pos")] = float(value)
        # The measured pose stays put inside the band; express that as an offset from the new command.
        self._measured_offset = {j: previous_measured[j] - self.command[j] for j in ("shoulder_lift", "elbow_flex")}
        self._settle()
        if self.holding and self.command["gripper"] > GRIP_ON_TILE + 5:
            self.holding = False
            if self.on_release:
                self.on_release([float(v) for v in kinematics.tool_point(self.command)[:2]],
                                self.measured_clearance_mm() / 1000)
        self.sent.append(dict(self.command))
        self.clearances.append(self.measured_clearance_mm())
        return action

    def get_observation(self):
        return {f"{j}.pos": v for j, v in self._read("Present_Position").items()}

    def connect(self, calibrate=False):
        pass

    def disconnect(self):
        pass

    def measured_clearance_mm(self):
        tip = kinematics.pincer_tip(self._read("Present_Position"))
        return (tip[2] - config.grasp_z(tip[0], tip[1])) * 1000


# --- stand-ins for the rest of the hardware, for `bench.py --sim` ---

SIM_TILES = [("9D", 0.140, 0.200), ("3B", 0.151, 0.018), ("8D", 0.160, -0.165)]
LANDS_ON_MAT_BELOW_M = 0.03     # a tile let go lower than this lands where the jaws opened; higher, it's off the mat
GRASP_REACH_M = 0.06            # SimPolicy closes on the nearest tile this close to its hover


class SimPerception:
    """A mat that changes: SimPolicy takes the tile it grasps off it, a release near the table puts it
    back where the jaws opened (one higher up, e.g. over the discard container, doesn't), and tests can
    nudge() tiles as a knock or a person would."""
    intrinsics = {"fx": 590.0, "fy": 590.0, "ppx": 424.0, "ppy": 244.0}   # fitted to a real overview frame

    def __init__(self, tiles=SIM_TILES):
        self.tiles = [{"tile_name": n, "xy": [x, y]} for n, x, y in tiles]
        self.held = None
        self.T_cam_to_base = np.load(config.T_CAM_TO_BASE)
        self.last_frames = []       # the latest scan's frames and boxes, like RealSense's
        from .detector import TILE_NAMES
        self.names = TILE_NAMES

    def detect_tiles(self, n_frames=None):
        """Every tile, read right; last_frames holds one rendered frame with a box per tile (no depth)."""
        from .layout import table_to_pixel, tile_corners
        boxes = []
        for t in self.tiles:
            corners = table_to_pixel(tile_corners(*t["xy"], 0.0), self.intrinsics, self.T_cam_to_base)
            (u0, v0), (u1, v1) = corners.min(axis=0), corners.max(axis=0)
            boxes.append({"tile_name": t["tile_name"], "confidence": 0.9,
                          "xyxy": [round(float(c), 1) for c in (u0, v0, u1, v1)]})
        self.last_frames = [{"color": self.snapshot(), "boxes": boxes}]
        return [{"tile_name": t["tile_name"], "confidence": 0.9, "xy": tuple(t["xy"]), "z": config.grasp_z(*t["xy"]),
                 "reads": 5, "pixel": (0, 0)} for t in self.tiles]

    def nearest(self, x, y, within_m=GRASP_REACH_M):
        def distance(t):
            return np.hypot(t["xy"][0] - x, t["xy"][1] - y)
        return min((t for t in self.tiles if distance(t) <= within_m), key=distance, default=None)

    def grasp(self, tile):
        self.tiles.remove(tile)
        self.held = tile

    def release(self, xy, clearance_m):
        if self.held is not None and clearance_m < LANDS_ON_MAT_BELOW_M:
            self.tiles.append({**self.held, "xy": list(xy)})
        self.held = None

    def nudge(self, name, dx, dy):
        tile = next(t for t in self.tiles if t["tile_name"] == name)
        tile["xy"] = [tile["xy"][0] + dx, tile["xy"][1] + dy]

    def snapshot(self):
        from .layout import render
        return render([(*t["xy"], 0.0) for t in self.tiles], self.intrinsics, self.T_cam_to_base)

    def close(self):
        pass


class SimCamera:
    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def flush(self, frames=10):
        pass

    def release(self):
        pass


class SimPolicy:
    """Stands in for an ACT policy: descend onto the tile, close, lift. Same interface as
    policy.Policy for what bench.py uses. With a SimPerception `mat` it goes to the nearest tile on it
    (as a working policy corrects the hover) and takes it off the mat; a mark set by then is noted at 2s."""
    id = "sim"
    hover = dict(config.DEFAULT_HOVER)

    def __init__(self, mat=None, policy_id="sim"):
        self.mat, self.id = mat, policy_id

    def run(self, arm, camera, trace_dir, abort=None, mark=None, log=print):
        x, y, _ = kinematics.tool_point(arm.commanded())
        tile = self.mat.nearest(x, y) if self.mat is not None else None
        if tile is not None:
            x, y = tile["xy"]
        down, _ = kinematics.hover_command(x, y, 0.007, arm.commanded()["gripper"])
        arm.move_to(down, 2.0)
        marks = []
        if mark is not None and mark.is_set():
            mark.clear()
            marks.append(2.0)
        arm.move_to({"gripper": 0.0}, 0.5)
        arm.robot.holding = True
        if tile is not None:
            self.mat.grasp(tile)
        up, _ = kinematics.hover_command(x, y, 0.13, 0.0)
        arm.move_to(up, 2.0)
        return {"steps": 0, "duration_s": 4.5, "stop_reason": "done", "aborted": False, "held_at_s": 3.0,
                "slipped": False, "stall_warning": False, "descended": True, "floored_steps": 0,
                "grasp_step": 0, "grasp_xy": [float(x), float(y)], "marks_s": marks}
