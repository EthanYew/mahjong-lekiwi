"""ACT pickup policies: the registry, loading, and the control loop.

Policies are loaded with the pre/post-processors saved next to their weights, so normalisation is
whatever the policy was trained with (MahjongBot normalised by hand from a separately-named
dataset's stats, which broke silently whenever the name drifted).

Dataset convention (every dataset so far): state and action are joint deltas from the pose the
episode started at. So the loop anchors at the measured pose when it takes over, feeds
state - anchor, and commands anchor + predicted delta. The hover a policy starts from has to match
its episodes' starts; that hover is stored with the policy in data/policies.json.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from . import config, kinematics
from .arm import holds_tile

STALL_CHECK_S = 4.0
STALL_MOVE_DEG = 2.0
CAMERA_LOST_S = 1.0     # no wrist frames for this long: stop the attempt, it can't count
IMAGE_SIZE = (224, 224)

# A pickup is done once the tile is held and lifted, and stays so for HOLD_CONFIRM_S while the policy
# keeps going (a slip resets it), instead of always running ACT_DURATION_S. Held: arm.holds_tile.
# Lifted: pincer tip LIFTED_M above the grasp point, ~1.7s after closing on the arm. On the 9 bench
# attempts of 2026-10-23 this stops at ~9-11s, not 20s.
LIFTED_M = 0.060
HOLD_CONFIRM_S = 1.5


def registry():
    return json.loads(config.POLICIES.read_text())


def spec(policy_id=None):
    reg = registry()
    policy_id = policy_id or reg["default"]
    return policy_id, reg["policies"][policy_id]


def register(policy_id, entry, make_default=False):
    reg = registry()
    reg["policies"][policy_id] = entry
    if make_default:
        reg["default"] = policy_id
    config.POLICIES.write_text(json.dumps(reg, indent=2) + "\n")


class Policy:
    def __init__(self, policy_id=None, device="cuda", source=None, hover=None):
        """A registered policy by id (default: the registry's default), or any repo/checkpoint path
        via `source` (e.g. a training checkpoint's pretrained_model folder)."""
        import packaging.version  # noqa: F401 — lerobot's from_pretrained uses it without importing it
        from huggingface_hub import snapshot_download
        from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
        from lerobot.policies.factory import make_pre_post_processors

        if source is not None:
            self.id, self.spec = str(source), {"repo": str(source), "hover": hover or dict(config.DEFAULT_HOVER)}
        else:
            self.id, self.spec = spec(policy_id)
        source = self.spec["repo"]
        path = source if Path(source).is_dir() else snapshot_download(source)
        self.act = ACTPolicy.from_pretrained(path)
        self.act.config.temporal_ensemble_coeff = config.TEMPORAL_ENSEMBLE_COEFF
        self.act.config.n_action_steps = 1
        self.act.temporal_ensembler = ACTTemporalEnsembler(config.TEMPORAL_ENSEMBLE_COEFF, self.act.config.chunk_size)
        self.act.to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(self.act.config, pretrained_path=path)

    @property
    def hover(self):
        return self.spec["hover"]

    def reset(self):
        self.act.reset()

    def observation(self, frame_bgr, state_delta):
        image = torch.from_numpy(np.ascontiguousarray(frame_bgr[:, :, ::-1])).permute(2, 0, 1).float() / 255.0
        image = F.resize(image, list(IMAGE_SIZE), antialias=True)
        return self.preprocess({"observation.images.wrist": image,
                                "observation.state": torch.as_tensor(state_delta, dtype=torch.float32)})

    @torch.no_grad()
    def step(self, frame_bgr, state_delta):
        """The next action, as a delta from the episode anchor (temporal ensembling applied)."""
        action = self.act.select_action(self.observation(frame_bgr, state_delta))
        return self.postprocess(action).squeeze(0).cpu().numpy()

    def run(self, arm, camera, trace_dir, abort=None, mark=None, log=print):
        return run(self, arm, camera, trace_dir, abort=abort, mark=mark, log=log)

    @torch.no_grad()
    def plan(self, frame_bgr, state_delta):
        """The full predicted chunk (chunk_size x 6 deltas), before ensembling — for diagnosis."""
        chunk = self.act.predict_action_chunk(self.observation(frame_bgr, state_delta))
        return np.stack([self.postprocess(chunk[:, k]).squeeze(0).cpu().numpy() for k in range(chunk.shape[1])])


def _tip_below_floor(command):
    tip = kinematics.pincer_tip(command)
    return tip[2] < config.table_z(tip[0], tip[1]) - config.TIP_FLOOR_BELOW_TABLE_M


def tip_floor(command, last_safe):
    """Scale a step that would press the pincer tip into the table back toward the last safe command
    (the LeKiwi base is light enough that pressing lifts the robot). Gripper untouched."""
    if not _tip_below_floor(command):
        return command, False
    lo, hi = 0.0, 1.0
    for _ in range(8):
        mid = (lo + hi) / 2
        trial = {j: last_safe[j] + mid * (command[j] - last_safe[j]) for j in command}
        lo, hi = (lo, mid) if _tip_below_floor(trial) else (mid, hi)
    floored = {j: last_safe[j] + lo * (command[j] - last_safe[j]) for j in command}
    floored["gripper"] = command["gripper"]
    return floored, True


def run(policy, arm, camera, trace_dir, abort=None, mark=None, duration_s=config.ACT_DURATION_S, log=print):
    """Run `policy` from wherever the arm is (the hover) until the tile is held and lifted for
    HOLD_CONFIRM_S, `abort` (an Event) is set, or duration_s runs out. `mark` (an Event with clear())
    is a person saying "something happened here": each time it's set the time is noted and it's
    cleared, and the policy carries on. Saves to trace_dir: act_trace.npz, wrist.mp4 (every frame; frame k
    is the image behind act_trace row k, plus one more when it stops on the done check) and a wrist jpg
    every 100 steps. Returns a summary: stop_reason (done / cap / aborted / camera: no wrist frames for
    CAMERA_LOST_S, so the attempt says nothing about the policy), duration_s, held_at_s, slipped
    (held, then lost during the hold check), stall_warning, descended, floored_steps, grasp_step,
    grasp_xy (the tool point where the gripper was first commanded shut — where a return should put
    the tile), marks_s (when it was marked, s)."""
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    policy.reset()
    camera.flush()
    anchor = arm.measured()
    anchor_vec = np.array([anchor[j] for j in config.JOINTS], dtype=np.float32)
    limits = arm.joint_limits()
    last_safe = arm.commanded()
    states, actions, floored_steps, warned, marks = [], [], 0, False, []
    stop_reason, done_since, held_at, slipped = "cap", None, None, False
    failed_reads = 0
    stall_step = int(STALL_CHECK_S * config.FPS)
    confirm_steps = int(HOLD_CONFIRM_S * config.FPS)
    log(f"ACT ({policy.id}) for up to {duration_s:.0f}s, stopping once the tile is held and lifted — Esc stops it")
    video = None        # every frame the policy saw, for replaying a new policy on this attempt (ROADMAP 2)
    try:
        for step in range(int(duration_s * config.FPS)):
            t0 = time.time()
            if abort is not None and abort.is_set():
                stop_reason = "aborted"
                log(f"  stopped at step {step}")
                break
            if mark is not None and mark.is_set():
                mark.clear()
                marks.append(round(step / config.FPS, 2))
                log(f"  t={step / config.FPS:4.1f}s marked (the policy carries on; you'll be asked what happened)")
            ok, frame = camera.read()
            if not ok:
                # 2026-10-28: the camera stopped giving frames mid-session and 8 attempts ran blind to the
                # cap, scored as misses
                failed_reads += 1
                if failed_reads == 1:
                    log(f"  wrist camera read failed at step {step}")
                if failed_reads >= CAMERA_LOST_S * config.FPS:
                    stop_reason = "camera"
                    log(f"  no wrist camera frames for {CAMERA_LOST_S:.0f}s — attempt stopped")
                    break
                arm.sleep(max(0.0, 1.0 / config.FPS - (time.time() - t0)))
                continue
            failed_reads = 0
            measured = arm.measured()
            state = np.array([measured[j] for j in config.JOINTS], dtype=np.float32)
            if video is None:
                video = cv2.VideoWriter(str(trace_dir / "wrist.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), config.FPS,
                                        (frame.shape[1], frame.shape[0]))
            video.write(frame)
            if step % 100 == 0:
                cv2.imwrite(str(trace_dir / f"wrist_{step:04d}.jpg"), frame)
            # Done check, on the state the previous command produced.
            holding = holds_tile(last_safe["gripper"], measured["gripper"])
            tip = kinematics.pincer_tip(measured)
            lifted = tip[2] - config.grasp_z(tip[0], tip[1]) >= LIFTED_M
            if holding and lifted:
                if done_since is None:
                    done_since = step
                    held_at = held_at if held_at is not None else step / config.FPS
                elif step - done_since >= confirm_steps:
                    stop_reason = "done"
                    log(f"  t={step / config.FPS:4.1f}s tile held and lifted for {HOLD_CONFIRM_S:.1f}s — done")
                    break
            elif done_since is not None:
                done_since, slipped = None, True
                log(f"  t={step / config.FPS:4.1f}s grip or lift lost during the hold check")
            delta = policy.step(frame, state - anchor_vec)
            command = {j: float(np.clip(anchor_vec[i] + delta[i], *limits[j])) for i, j in enumerate(config.JOINTS)}
            command, floored = tip_floor(command, last_safe)
            floored_steps += floored
            arm.send(command)
            last_safe = command
            states.append(state)
            actions.append([command[j] for j in config.JOINTS])
            if step == stall_step and not warned:
                moved = float(np.abs(np.array(actions)[:, 1:3] - anchor_vec[1:3]).max())
                if moved < STALL_MOVE_DEG:
                    warned = True
                    log(f"  WARNING: {STALL_CHECK_S:.0f}s in, lift/elbow have moved {moved:.1f} deg — the policy may be "
                        f"waiting at the hover. Esc to stop.")
            if step % config.FPS == 0:
                log(f"  t={step / config.FPS:4.1f}s lift {command['shoulder_lift']:+6.1f} elbow {command['elbow_flex']:+6.1f} "
                    f"gripper {command['gripper']:5.1f}" + (" [tip floor]" if floored else ""))
            arm.sleep(max(0.0, 1.0 / config.FPS - (time.time() - t0)))
    finally:            # an interrupted attempt keeps a readable video
        if video is not None:
            video.release()
    policy.reset()
    actions, states = np.array(actions), np.array(states)
    np.savez(trace_dir / "act_trace.npz", state=states, action=actions, anchor=anchor_vec, joints=np.array(config.JOINTS))
    grasp, grasp_xy = kinematics.grasp_point(actions)
    descended = bool(len(actions)) and float(np.abs(actions[:, 1:3] - anchor_vec[1:3]).max()) >= STALL_MOVE_DEG
    return {"steps": len(actions), "duration_s": len(actions) / config.FPS, "stop_reason": stop_reason,
            "aborted": stop_reason == "aborted", "held_at_s": held_at, "slipped": slipped, "stall_warning": warned,
            "descended": descended, "floored_steps": int(floored_steps), "grasp_step": grasp, "grasp_xy": grasp_xy,
            "marks_s": marks}
