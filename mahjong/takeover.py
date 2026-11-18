"""A person takes the arm from the policy mid-pickup and shows how to recover: DAgger wherever a script runs the policy.

record.py --policy records DAgger on purpose: every episode starts at the hover, the policy drives, A on the gamepad
hands a person the arm. The same failures happen in bench.py, base_cam_collect.py, wall_test.py and crude_run.py, and
setting them up again in record.py loses the moment. With --dagger-dataset REPO those scripts connect the gamepad too,
and policy.run hands the arm over on A at any point of a pickup:

- the attempt's frames (wrist image, measured joints, command) are kept in memory from the hover on, and the
  gamepad drives from the policy's last command (no drop, gripper held: GamepadTeleop.start);
- it ends when A is pressed again, once the tile is held and lifted for policy.HOLD_CONFIRM_S, or after MAX_S;
  Esc throws the recovery away;
- the whole attempt is saved as one episode of REPO, in record.py's format (FEATURES, deltas from the hover, an
  episode_log.jsonl row with `policy` and `takeover_frame`), so `make_policy.py --add REPO` trains on the frames
  from the takeover on, as it does record.py's. An attempt nobody took over isn't saved.

A takeover is a rescue (interventions "takeover", attempt_dir set): the attempt isn't clean (scorecard.NOT_CLEAN).
Frames are written after the pickup, not during it, so the policy's loop runs at the rate it always has; a saved
recovery costs a few seconds while it's written.
"""
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config
from .dataset import EPISODE_LOG

JOINT_NAMES = [f"{j}.pos" for j in config.JOINTS]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},          # delta from episode start
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},                     # delta from episode start
    "observation.state_raw": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action_raw": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["h", "w", "c"]},
}
TASK = "Pick up the mahjong tile from the mat"      # record.py's default --task
MAX_S = 30.0            # a recovery longer than this is stopped (a demo takes ~10s); the frames are in memory until saved
HELP = "--dagger-dataset"


def open_dataset(repo_id):
    """A LeRobot dataset to record into, created or resumed. Refuses one of a different layout."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    root = HF_LEROBOT_HOME / repo_id
    info = root / "meta" / "info.json"
    if info.exists() and json.loads(info.read_text()).get("total_episodes", 0) == 0:
        shutil.rmtree(root)            # stub from a run interrupted before its first save
    if info.exists():
        existing = json.loads(info.read_text())["features"]
        missing = set(FEATURES) - set(existing)
        if missing:
            raise SystemExit(f"{repo_id} has a different layout (missing {sorted(missing)}); record into a new dataset")
        dataset = LeRobotDataset.resume(repo_id, root=root)
        print(f"Appending to {repo_id} ({dataset.meta.total_episodes} episodes)")
        return dataset
    print(f"Creating {repo_id}")
    return LeRobotDataset.create(repo_id=repo_id, fps=config.FPS, features=FEATURES, robot_type="so101_follower",
                                 use_videos=True)


def add_argument(parser):
    parser.add_argument("--dagger-dataset", default=None, metavar="REPO",
                        help="connect the gamepad too: A during a pickup hands you the arm, and your recovery is saved as a "
                             "DAgger episode of this dataset (make_policy.py --add); A again or a held, lifted tile ends it, Esc throws it away")


def same_hover(hovers):
    """Refuse a takeover dataset across hovers that differ: a policy works only from the hover its episodes started at."""
    hovers = [dict(h) for h in hovers]
    if any(abs(h[k] - hovers[0][k]) > 1e-6 for h in hovers for k in ("offset_m", "clearance_m", "gripper")):
        raise SystemExit(f"{HELP}: the policies here start from different hovers ({hovers}); one dataset needs one hover, "
                         "so take over with one policy at a time")


class Takeover:
    """The gamepad and the dataset a takeover goes to. policy.run calls begin(), pressed(), start(), step() and add();
    the script calls save() after the pickup."""

    def __init__(self, gamepad, repo_id, dataset=None, task=TASK, log=print):
        self.gamepad, self.repo_id, self.task, self.log = gamepad, repo_id, task, log
        self.dataset = dataset if dataset is not None else open_dataset(repo_id)
        self.frames, self.taken, self.off_centre, self.times = [], None, {}, []

    def expect_hover(self, hover):
        """Refuse to add to a dataset recorded from another hover (record.py refuses the same): its policy works only from
        the hover its episodes started at, and make_policy.py takes one hover per dataset."""
        rows = episode_log_rows(self.dataset.root)
        theirs = next((r["hover"] for r in rows if r.get("hover")), None)
        if theirs is not None:
            same_hover([theirs, hover])

    # --- during the pickup (policy.run) ---

    def begin(self):
        """A pickup starts: nothing kept yet, and an A already held (since the hover, say) isn't a press."""
        self.frames, self.taken, self.off_centre, self.times = [], None, {}, []
        self.gamepad.arm_takeover()

    def pressed(self):
        return self.gamepad.takeover_pressed()

    def start(self, arm, step):
        """A was pressed at `step`: the gamepad drives from the arm's commanded pose."""
        self.taken = step
        self.gamepad.start(arm)
        self.off_centre = self.gamepad.off_centre() if hasattr(self.gamepad, "off_centre") else {}
        for side, (x, y) in self.off_centre.items():        # a thumb resting on a stick moves the arm (gamepad.warn_off_centre)
            self.log(f"  {side} stick reads ({x:+.2f}, {y:+.2f}) untouched? — let go of it")
        self.log(f"  t={step / config.FPS:4.1f}s YOU have it: show the recovery. A again ends and saves it (so does the tile "
                 "held and lifted a moment), Esc throws it away")

    def step(self, arm, dt):
        return self.gamepad.step(arm, dt)

    def add(self, frame, state, command):
        self.frames.append((frame, np.asarray(state, dtype=np.float32),
                            np.array([command[j] for j in config.JOINTS], dtype=np.float32)))
        self.times.append(time.time())

    def fps(self):
        """The rate the frames came at: the policy and the gamepad both assume config.FPS."""
        if len(self.times) < 2:
            return float(config.FPS)
        return round((len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-6), 1)

    # --- after it ---

    def save(self, run, session, policy, **fields):
        """Save the pickup as a DAgger episode if a person took over and didn't throw it away (Esc), logging the takeover
        as a rescue either way. Call it after every pickup run with this takeover. `fields` go in its episode-log row
        (tile_name, detection_xy, hover, hover_xy, attempt_dir...). Returns the episode index, or None."""
        frames, self.frames = self.frames, []
        fps = self.fps()
        taken = run.get("takeover_step")
        if taken is None:
            return None
        attempt_dir = fields.get("attempt_dir")
        session.intervention("takeover", attempt_dir=attempt_dir, frame=taken, tile=fields.get("tile_name"),
                             ended=run.get("takeover_end"), dataset=self.repo_id,
                             **({"sticks_off_centre": self.off_centre} if self.off_centre else {}))
        why = {"discarded": "Esc", "camera": "the wrist camera dropped out or tore: frames missing or corrupt"}.get(run.get("takeover_end"))
        if why or not frames:
            session.event("takeover_discarded", attempt_dir=attempt_dir, frames=len(frames), dataset=self.repo_id,
                          why=why or "no frames")
            self.log(f"  recovery not saved ({why or 'no frames'})")
            return None
        if fps < 0.9 * config.FPS:
            self.log(f"  recorded at {fps:.0f} fps, not {config.FPS}: the loop ran slow (record.py says the same)")
        self.log(f"  writing the recovery to {self.repo_id} ({len(frames)} frames)...")
        started = time.time()
        anchor = frames[0][1]
        for frame, state, action in frames:
            self.dataset.add_frame({"observation.state": state - anchor, "action": action - anchor,
                                    "observation.state_raw": state, "action_raw": action,
                                    "observation.images.wrist": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "task": self.task})
        episode = self.dataset.meta.total_episodes
        self.dataset.save_episode()
        record = {"episode_index": episode, "timestamp": datetime.now().isoformat(timespec="seconds"),
                  "frames": len(frames), "session": session.dir.name, "slice": "train", "conditions": session.conditions,
                  "policy": policy, "takeover_frame": taken, "takeover_end": run.get("takeover_end"),
                  "fps": fps, "source": session.dir.name.split("_", 2)[-1], **fields}
        with open(Path(self.dataset.root) / EPISODE_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        session.event("takeover_saved", dataset=self.repo_id, **record)
        self.log(f"  saved episode {episode} of {self.repo_id}: yours from {taken / config.FPS:.1f}s "
                 f"({len(frames) - taken} frames trained on; {time.time() - started:.0f}s to write)")
        return episode

    def close(self):
        try:
            if self.dataset.has_pending_frames():
                self.dataset.clear_episode_buffer()
            self.dataset.finalize()
        finally:
            self.gamepad.close()


def policy_view(run):
    """A pickup's summary as an attempt logs it: after a takeover, grasp_xy and grasp_clearance_m are the policy's own close
    (what the close-height and wall reports judge the policy by; None if it hadn't closed), and person_grasp_xy /
    person_grasp_clearance_m the person's (what policy.run returns them as, for the return to aim from)."""
    if run.get("takeover_step") is None:
        return dict(run)
    out = {k: v for k, v in run.items() if k not in ("policy_grasp_xy", "policy_grasp_clearance_m")}
    out.update(person_grasp_xy=run.get("grasp_xy"), person_grasp_clearance_m=run.get("grasp_clearance_m"),
               grasp_xy=run.get("policy_grasp_xy"), grasp_clearance_m=run.get("policy_grasp_clearance_m"))
    return out


def episode_log_rows(root):
    from .session import read_jsonl         # appended to while recording, so a power cut can cut it short too
    return read_jsonl(Path(root) / EPISODE_LOG)


def devices(repo_id):
    """The extra devices hardware.connect opens for --dagger-dataset REPO: the gamepad, as record.py connects it."""
    return ["gamepad"] if repo_id else []


def gamepad_kwargs(device=None):
    """As record.py's defaults: a stick moves the arm once held 0.1s, the 0.35 deadzone."""
    from .gamepad import DEADZONE
    return {"device": device, "commit_s": 0.1, "deadzone": DEADZONE}


def make(repo_id, gamepad, log=print):
    """The Takeover for --dagger-dataset REPO and its connected gamepad, or None without them (--sim: nothing to hand over)."""
    if not repo_id:
        return None
    if gamepad is None:
        log(f"{HELP} does nothing here: no gamepad (--sim has no arm to hand over)")
        return None
    log(f"Takeover: A on {getattr(gamepad, 'name', 'the gamepad')} hands you the arm during a pickup; recoveries go to {repo_id}")
    return Takeover(gamepad, repo_id, log=log)
