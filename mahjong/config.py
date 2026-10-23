"""Everything specific to this robot and this table, in one place.

The SO101 follower sits on a LeKiwi base, ~9cm above a green mat; a RealSense on a tripod behind it
detects tiles; a wrist camera feeds the ACT policy. Per-policy hover settings (how the episodes a
policy was trained on began) live in data/policies.json, not here.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SESSIONS = REPO / "sessions"

# --- robot ---
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034124-if00"  # LeKiwi board, motors 1-6
FOLLOWER_ID = "ethan_lekiwi_arm"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]

REST_POSE = {"shoulder_pan": -91.0, "shoulder_lift": -106.32967032967034, "elbow_flex": 105.58241758241758,
             "wrist_flex": 66.81318681318682, "wrist_roll": -99.47252747252747, "gripper": 10.609171800136892}
# Rest with the arm raised in place (same pan and roll), pincer tip ~80mm above the arm's mount. Moves
# between rest and the table go through it, so the gripper — and a held tile, ~12mm below the jaws —
# crosses the front of the LeKiwi base 60-68mm above the mount instead of up to 11mm below it.
RAISED_REST = {**REST_POSE, "shoulder_lift": -40.0, "elbow_flex": 40.0}

# --- cameras ---
FPS = 30
WRIST_CAM = "/dev/wristcam"            # udev symlink
WRIST_RESOLUTION = (640, 480)          # what every dataset was recorded at (resized to 224 for training)
REALSENSE_SERIAL = "825312070255"
REALSENSE_RESOLUTION = (848, 480)
T_CAM_TO_BASE = REPO / "assets" / "T_cam_to_base.npy"   # hand-eye, rear tripod, max neck (2026-10-20)
TILE_DETECTOR = REPO / "assets" / "tile_detector.pt"     # YOLO, table view
DETECTION_CONFIDENCE = 0.5
DETECTION_FRAMES = 5
TILE_MERGE_RADIUS_M = 0.025            # detections closer than this are one physical tile

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
TIP_FLOOR_BELOW_TABLE_M = 0.005        # ACT steps putting the pincer tip lower than this get scaled back

# --- defaults for recording and correction collection (a policy's own hover lives in policies.json) ---
HOVER_OFFSET_M = 0.045                 # pincer tip commanded this far above the grasp point
HOVER_CLEARANCE_M = 0.040              # ...then settled until it measures this far above it
HOVER_GRIPPER = 90.0                   # opening at the hover; every dataset so far started here

# --- ACT ---
TEMPORAL_ENSEMBLE_COEFF = 0.001        # lerobot: positive weights OLDER plans more
ACT_DURATION_S = 16.0                  # cap; a pickup normally stops itself once the tile is held (policy.py)

# --- files ---
POLICIES = DATA / "policies.json"
HOVER_CORRECTIONS = DATA / "hover_corrections.jsonl"
HOVER_MODEL = DATA / "hover_model.json"       # which correction sources the live hover is fitted from
LAYOUTS = DATA / "layouts.json"           # named tile layouts for placing tiles (mahjong/layout.py)
EVAL_SPLIT = DATA / "eval_split.json"     # episodes never trained on (mahjong/offline_eval.py)
OFFLINE_EVAL = DATA / "offline_eval"      # offline scores per policy and eval set
DISCARD_POSE = DATA / "discard_pose.json"
HF_USER = "Yewy"
LEROBOT_REPO = Path.home() / "Projects" / "lerobot"     # lerobot-train runs from here (outputs/train/...)
XLEROBOT_EXAMPLES = Path.home() / "Projects" / "XLeRobot" / "software" / "examples"
XLEROBOT_MODEL = Path.home() / "Projects" / "XLeRobot" / "software" / "plugins" / "xlerobot_model"

# --- session setup questions: asked at the start of every hardware session and stored as `conditions`
# on session_start, so sessions compare (scripts/session_report.py --all). Choices are a list, or "yn".
# sticky: Enter keeps the previous session's answer; otherwise Enter means the first choice.
SETUP_QUESTIONS = [
    {"key": "daylight", "ask": "Daylight in the room", "choices": ["bright", "dim", "none"], "sticky": True},
    {"key": "lamp", "ask": "Lamp behind the robot on", "choices": "yn", "sticky": True},
    {"key": "ceiling_lights", "ask": "Ceiling lights on", "choices": "yn", "sticky": True},
    {"key": "moved", "ask": "Moved since the last session",
     "choices": ["nothing", "realsense", "mat", "lekiwi", "several"], "sticky": False},
    {"key": "tiles", "ask": "Tiles on the mat", "choices": ["one", "few spread", "rows"], "sticky": True},
    {"key": "battery", "ask": "LeKiwi battery", "choices": ["full", "mid", "low"], "sticky": True},
]
