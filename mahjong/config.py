"""Everything specific to this robot and this table, in one place.

The SO101 follower sits on a LeKiwi base, ~9cm above a green mat; a RealSense on a tripod behind it
detects tiles; a wrist camera feeds the ACT policy. Per-policy hover settings (how the episodes a
policy was trained on began) live in data/policies.json, not here.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SESSIONS = REPO / "sessions"

# --- robot ---
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034124-if00"  # LeKiwi board, motors 1-6
FOLLOWER_ID = "ethan_lekiwi_arm"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]
GRIPPER = JOINTS.index("gripper")      # its column in rows of joint values

# Where the unpowered arm settles (sessions/parked_pose.json, 2026-10-27 09:55 and 10:25, ±0.5°), so
# switching the motors off or on doesn't move it. The old folded rest (lift -106.3, elbow 105.6) sagged
# ~8° onto the stops at every disconnect, and every connect lifted it via RAISED_REST and back.
REST_POSE = {"shoulder_pan": -90.7, "shoulder_lift": -102.2, "elbow_flex": 97.6,
             "wrist_flex": 67.6, "wrist_roll": -99.5, "gripper": 10.609171800136892}
# Rest with the arm raised in place (same pan and roll), pincer tip ~80mm above the arm's mount. Moves
# between rest and the table go through it, so the gripper — and a held tile, ~12mm below the jaws —
# crosses the front of the LeKiwi base 60-68mm above the mount instead of up to 11mm below it.
RAISED_REST = {**REST_POSE, "shoulder_lift": -40.0, "elbow_flex": 40.0}

# --- cameras ---
FPS = 30
WRIST_CAM = "/dev/wristcam"            # udev symlink
BASE_CAM = "/dev/basecam"              # the LeKiwi base's front camera (same model as the wrist's): udev symlink by USB port, 3-9 on the NUC
WRIST_USB_PORT = "3-8"                 # its USB port, for a power cycle if it dropped before this run saw where it was
WRIST_RESOLUTION = (640, 480)          # what every dataset was recorded at (resized to 224 for training)
REALSENSE_SERIAL = "825312070255"
REALSENSE_RESOLUTION = (848, 480)
T_CAM_TO_BASE = REPO / "assets" / "T_cam_to_base.npy"   # hand-eye, rear tripod, max neck (2026-10-20)
TILE_DETECTOR = REPO / "assets" / "tile_detector.pt"     # YOLO, table view
DETECTION_CONFIDENCE = 0.5
DETECTION_SAVE_CONFIDENCE = 0.25       # a scan saves boxes down to this: the ones below the cut are where it hesitated
# Whether a tile is there, not which: a box this sure, in 2 of a scan's frames. The detector sees tiles it can't name: 3C
# read at a median of 0.18 in 20261103_121112_bench, and looked gone in 45 of 46 scans. All 14 tiles in all 46 scans at this
# cutoff; none at 31 spots just emptied by a pickup; on the mat at no tile, 0 in 74 scans and 2 in 42 on the black top.
DETECTION_PRESENCE_CONFIDENCE = 0.1
# A tile lying green side up, as the detector names it once its weights have the class (perception.sees_backs). It says a
# tile is there and which way up, never which tile: nothing reads a back. A back box whose centre sits under this above the
# table is mat pattern, not a tile (a tile's top is ~20mm up; the table plane's reading wanders ~3mm, 2026-11-11).
BACK = "back"
BACK_MIN_HEIGHT_M = 0.008
UNNAMED = "unnamed"                    # a tile's name when no scan or person has given it one: never a spot id or a layout label
DETECTION_FRAMES = 5
TILE_MERGE_RADIUS_M = 0.025            # detections closer than this are one physical tile
# A returned tile aims for its home (where it was first seen this session; mahjong/return_correction.py). A pick
# and return moved tiles 5.7mm (median) with ~6mm scatter, 95 bench returns to 2026-10-28; the jaws tolerate ~6mm.
HOME_RADIUS_M = 0.04                   # a tile detected further than this from its home was moved on purpose: new home
RETURN_DRIFT_WARN_M = 0.012            # a return landing this far from home is logged and printed
RETURN_DRIFT_RESET_M = 0.025           # this far (or out of the hover's reach), a person puts it back home

# --- table: a plane in the arm's base frame (fit to RealSense depth, touch-tested 2026-10-20) ---
TABLE_Z0, TABLE_SLOPE_X, TABLE_SLOPE_Y = -0.0678, -0.143, -0.002
GRASP_POINT_ABOVE_TABLE_M = 0.005      # tile heights and clearances are measured from here
TILE_THICKNESS_M = 0.020               # so the tile's top face is ~15mm above the grasp point
TILE_FACE_M = (0.035, 0.026)           # top face, long x short side (long side along x when square); measured
                                       # 2026-10-23 (the RealSense reads ~38x29.5: side faces + ~1px per edge)


def table_z(x, y):
    return TABLE_Z0 + TABLE_SLOPE_X * x + TABLE_SLOPE_Y * y


def grasp_z(x, y):
    return table_z(x, y) + GRASP_POINT_ABOVE_TABLE_M


# --- motion ---
GRIPPER_RELEASE = 25.0   # p90 of the opening held around a tile in 104 recorded pickups (median 21%)
# A return opens only this far past the jaws' reading on the tile (~11), not to GRIPPER_RELEASE: the
# 25% release turned a neighbour on 2026-10-23. Not yet checked on hardware: raise it if a tile sticks.
GRIPPER_RELEASE_PAST_HELD = 6.0
TIP_FLOOR_BELOW_TABLE_M = 0.005        # ACT steps putting the pincer tip lower than this get scaled back
# The gripper counts as shut when commanded below GRIP_CLOSED_BELOW (policies grip by commanding 0), and as
# holding a tile when shut but measured HOLDING_MARGIN wider: a tile between the jaws reads ~11 above the
# command (104/104 recorded episodes, all 9 bench attempts on 2026-10-23).
GRIP_CLOSED_BELOW = 5.0
HOLDING_MARGIN = 5.0
# Jaws commanded above GRIP_CLOSED_BELOW but reading under this may be stalled on a tile (a person stops
# closing there): Arm.squeeze shuts them before asking holds_tile.
GRIP_SQUEEZE_BELOW = 16.0

# --- defaults for recording and correction collection (a policy's own hover lives in policies.json) ---
HOVER_OFFSET_M = 0.045                 # pincer tip commanded this far above the grasp point
HOVER_CLEARANCE_M = 0.040              # ...then settled until it measures this far above it
HOVER_GRIPPER = 90.0                   # opening at the hover; every dataset so far started here
DEFAULT_HOVER = {"offset_m": HOVER_OFFSET_M, "clearance_m": HOVER_CLEARANCE_M, "gripper": HOVER_GRIPPER}

# --- ACT ---
TEMPORAL_ENSEMBLE_COEFF = 0.001        # lerobot: positive weights OLDER plans more
ACT_DURATION_S = 16.0                  # cap; a pickup normally stops itself once the tile is held (policy.py)

# --- files ---
POLICIES = DATA / "policies.json"
HOVER_CORRECTIONS = DATA / "hover_corrections.jsonl"
HOVER_MODEL = DATA / "hover_model.json"       # which correction sources the live hover is fitted from
RETURN_CORRECTIONS = DATA / "return_corrections.jsonl"   # returns: where set down vs where landed (mahjong/return_correction.py)
LAYOUTS = DATA / "layouts.json"           # named tile layouts for placing tiles (mahjong/layout.py)
EVAL_SPLIT = DATA / "eval_split.json"     # episodes never trained on (mahjong/offline_eval.py)
DETECTOR_EXAM = DATA / "detector_exam.json"   # sessions whose frames are the tile detector's exam (mahjong/detector.py)
DETECTOR_EVAL = DATA / "detector_eval"        # detector scores on the exam
DETECTOR_DATA = DATA / "detector"             # exports for labelling and labelled exports (local; Roboflow keeps the labels)
OFFLINE_EVAL = DATA / "offline_eval"      # offline scores per policy and eval set
DISCARD_POSE = DATA / "discard_pose.json"      # over the discard container (bench.py's teach-discard): discard_pose()
CAMERA_POSES = DATA / "camera_poses.json"       # arm poses that hold a tile up to the base camera (scripts/teleop_view.py)


def discard_pose(why=None):
    """The taught pose over the discard container ({joint: deg}), or None if none is taught. With `why` (what needs it):
    SystemExit instead, saying so, for a script that checks before any hardware connects."""
    if DISCARD_POSE.exists():
        return json.loads(DISCARD_POSE.read_text())["pose"]
    if why:
        raise SystemExit(f"No discard pose: teach one with bench.py's teach-discard. {why}")
    return None


HF_USER = "Yewy"
LEROBOT_REPO = Path.home() / "Projects" / "lerobot"     # lerobot-train runs from here (outputs/train/...)
XLEROBOT_EXAMPLES = Path.home() / "Projects" / "XLeRobot" / "software" / "examples"
XLEROBOT_MODEL = Path.home() / "Projects" / "XLeRobot" / "software" / "plugins" / "xlerobot_model"

# --- session setup questions: asked at the start of every hardware session and stored as `conditions`
# on session_start, so sessions compare (scripts/session_report.py --all). Choices are a list, or "yn".
# sticky: Enter keeps the previous session's answer; otherwise Enter means the first choice.
SURFACE_BEFORE_ASKED = "black table top"    # what every session before the surface question was recorded on
SETUP_QUESTIONS = [
    {"key": "daylight", "ask": "Daylight in the room", "choices": ["bright", "dim", "none"], "sticky": True},
    {"key": "lamp", "ask": "Lamp behind the robot on", "choices": "yn", "sticky": True},
    {"key": "ceiling_lights", "ask": "Ceiling lights on", "choices": "yn", "sticky": True},
    {"key": "moved", "ask": "Moved since the last session",
     "choices": ["nothing", "realsense", "mat", "lekiwi", "several"], "sticky": False},
    {"key": "tiles", "ask": "Tiles on the mat", "choices": ["one", "few spread", "rows"], "sticky": True},
    {"key": "battery", "ask": "LeKiwi battery", "choices": ["full", "mid", "low"], "sticky": True},
    # the surface is part of the policy's input like the light: every recording through v5 was on the black
    # table top, and v5 failed on a green cutting mat (2026-10-29). Earlier sessions: SURFACE_BEFORE_ASKED.
    {"key": "surface", "ask": "Surface under the tiles",
     "choices": ["black table top", "green cutting mat", "grey mat", "paper", "brown table", "brown paper",
                 "standard mat", "other"], "sticky": True},
]
