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
        self.T_render = self.T_cam_to_base      # where the camera really is: a test moves it to fake a knocked tripod
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
        image = render([(*t["xy"], t.get("yaw_deg", 0.0)) for t in self.tiles], self.intrinsics, self.T_render)
        return draw_robot(image, self.intrinsics, self.T_render)

    def depth_snapshot(self):
        """(colour, depth in metres): the table plane seen from where the camera really is (T_render)."""
        h, w = 480, 848
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        rays = np.stack([(u - self.intrinsics["ppx"]) / self.intrinsics["fx"],
                         (v - self.intrinsics["ppy"]) / self.intrinsics["fy"], np.ones((h, w))], axis=-1) @ self.T_render[:3, :3].T
        o = self.T_render[:3, 3]
        # table_z is linear (Z0 + sx x + sy y): the ray's depth t where it meets it, in one step
        t = ((config.TABLE_Z0 + config.TABLE_SLOPE_X * o[0] + config.TABLE_SLOPE_Y * o[1] - o[2])
             / (rays[..., 2] - config.TABLE_SLOPE_X * rays[..., 0] - config.TABLE_SLOPE_Y * rays[..., 1]))
        return self.snapshot(), np.where(t > 0, t, 0).astype(np.float32)

    def close(self):
        pass


ROBOT_TOP = ((-0.12, 0.05), (-0.15, 0.15), 0.06)     # x range, y range, height above the table: the base's top


def draw_robot(image, intrinsics, T_cam_to_base):
    """Paint a fixed texture where the LeKiwi base would be, so the image has features that move with the
    camera (setup_check's camera check matches them)."""
    import cv2
    from .layout import table_to_pixel
    (x0, x1), (y0, y1), height = ROBOT_TOP
    z = config.table_z(0.0, 0.0) + height
    quad = table_to_pixel([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], intrinsics, T_cam_to_base)
    texture = cv2.resize(np.random.default_rng(7).integers(0, 255, (24, 40), dtype=np.uint8), (400, 240),
                         interpolation=cv2.INTER_NEAREST)
    texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    th, tw = texture.shape[:2]
    H = cv2.getPerspectiveTransform(np.float32([(0, 0), (0, th), (tw, th), (tw, 0)]), np.float32(quad))
    h, w = image.shape[:2]
    warped = cv2.warpPerspective(texture, H, (w, h))
    mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), H, (w, h)) > 0
    image[mask] = warped[mask]
    return image


class SimWristCamera:
    """A wrist camera over a SimPerception mat: the tile nearest the jaws as a white face on a dark mat, placed
    by where it really is relative to the tool point and sized by the jaws' height (4px/mm at the 40mm hover)."""
    simulated = True
    SHAPE, PX_PER_MM_AT_HOVER = (480, 640), 4.0

    def __init__(self, arm, mat):
        self.arm, self.mat = arm, mat

    def read(self):
        import cv2
        image = np.full((*self.SHAPE, 3), (25, 25, 25), np.uint8)
        x, y, _ = kinematics.tool_point(self.arm.measured())
        tile = self.mat.nearest(x, y, within_m=0.05)
        if tile is not None:
            scale = self.PX_PER_MM_AT_HOVER * config.HOVER_CLEARANCE_M * 1000 / max(self.arm.robot.measured_clearance_mm(), 5.0)
            centre = (self.SHAPE[1] / 2 - (tile["xy"][1] - y) * 1000 * scale,     # left in the image = +y
                      self.SHAPE[0] / 2 - (tile["xy"][0] - x) * 1000 * scale)     # up = +x (away)
            size = (config.TILE_FACE_M[1] * 1000 * scale, config.TILE_FACE_M[0] * 1000 * scale)
            box = cv2.boxPoints((centre, size, -tile.get("yaw_deg", 0.0)))
            cv2.fillPoly(image, [np.round(box).astype(np.int32)], (225, 235, 240))
        return True, image

    def flush(self, frames=10):
        pass

    def release(self):
        pass


class SimCamera:
    simulated = True        # hardware.reopen_wrist_camera leaves it be

    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def flush(self, frames=10):
        pass

    def release(self):
        pass


def connect(tiles=SIM_TILES, wrist_view=False):
    """The simulated counterpart of hardware.connect(["arm", "wrist_camera", "realsense"]): a mat of `tiles`
    that a release near the table puts the held tile back on, an arm that doesn't wait, and a wrist camera
    that renders the mat under it (wrist_view) or gives blank frames."""
    from .arm import Arm
    mat = SimPerception(tiles)
    arm = Arm(SimRobot(on_release=mat.release), sleep=lambda s: None)
    return {"arm": arm, "wrist_camera": SimWristCamera(arm, mat) if wrist_view else SimCamera(), "realsense": mat}


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
