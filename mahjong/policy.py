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

STALL_CHECK_S = 4.0
STALL_MOVE_DEG = 2.0
GRIP_CLOSED_BELOW = 5.0
IMAGE_SIZE = (224, 224)

# A pickup is done once the tile is held and lifted, and stays so for HOLD_CONFIRM_S while the policy
# keeps going (a slip resets it), instead of always running ACT_DURATION_S. Held: gripper commanded
# shut but measured HOLDING_MARGIN wider — a tile between the jaws reads ~11 above the command
# (104/104 recorded episodes, all 9 bench attempts on 2026-10-23). Lifted: pincer tip LIFTED_M above
# the grasp point, ~1.7s after closing on the arm. On those 9 attempts this stops at ~9-11s, not 20s.
HOLDING_MARGIN = 5.0
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
            default_hover = {"offset_m": config.HOVER_OFFSET_M, "clearance_m": config.HOVER_CLEARANCE_M,
                             "gripper": config.HOVER_GRIPPER}
            self.id, self.spec = str(source), {"repo": str(source), "hover": hover or default_hover}
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

    def run(self, arm, camera, trace_dir, abort=None, log=print):
        return run(self, arm, camera, trace_dir, abort=abort, log=log)

    @torch.no_grad()
    def plan(self, frame_bgr, state_delta):
        """The full predicted chunk (chunk_size x 6 deltas), before ensembling — for diagnosis."""
        chunk = self.act.predict_action_chunk(self.observation(frame_bgr, state_delta))
        return np.stack([self.postprocess(chunk[:, k]).squeeze(0).cpu().numpy() for k in range(chunk.shape[1])])


def grasp_point(actions):
    """(step, tool-point xy) of the first command with the gripper shut — where the jaws closed on the
    tile — or (None, None). actions: steps x JOINTS, absolute."""
    grasp = next((k for k in range(len(actions)) if actions[k][5] < GRIP_CLOSED_BELOW), None)
    if grasp is None:
        return None, None
    return grasp, [float(v) for v in kinematics.tool_point(dict(zip(config.JOINTS, actions[grasp])))[:2]]


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


def run(policy, arm, camera, trace_dir, abort=None, duration_s=config.ACT_DURATION_S, log=print):
    """Run `policy` from wherever the arm is (the hover) until the tile is held and lifted for
    HOLD_CONFIRM_S, `abort` (an Event) is set, or duration_s runs out. Saves act_trace.npz and wrist
    frames every 100 steps to trace_dir. Returns a summary: stop_reason (done / cap / aborted),
    duration_s, held_at_s, slipped (held, then lost during the hold check), stall_warning, descended,
    floored_steps, grasp_step, grasp_xy (the tool point where the gripper was first commanded shut —
    where a return should put the tile)."""
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    policy.reset()
    camera.flush()
    anchor = arm.measured()
    anchor_vec = np.array([anchor[j] for j in config.JOINTS], dtype=np.float32)
    limits = arm.joint_limits()
    last_safe = arm.commanded()
    states, actions, floored_steps, warned = [], [], 0, False
    stop_reason, done_since, held_at, slipped = "cap", None, None, False
    stall_step = int(STALL_CHECK_S * config.FPS)
    confirm_steps = int(HOLD_CONFIRM_S * config.FPS)
    log(f"ACT ({policy.id}) for up to {duration_s:.0f}s, stopping once the tile is held and lifted — Esc stops it")
    for step in range(int(duration_s * config.FPS)):
        t0 = time.time()
        if abort is not None and abort.is_set():
            stop_reason = "aborted"
            log(f"  stopped at step {step}")
            break
        ok, frame = camera.read()
        if not ok:
            log(f"  wrist camera read failed at step {step}")
            continue
        measured = arm.measured()
        state = np.array([measured[j] for j in config.JOINTS], dtype=np.float32)
        if step % 100 == 0:
            cv2.imwrite(str(trace_dir / f"wrist_{step:04d}.jpg"), frame)
        # Done check, on the state the previous command produced.
        holding = last_safe["gripper"] < GRIP_CLOSED_BELOW and measured["gripper"] - last_safe["gripper"] > HOLDING_MARGIN
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
    policy.reset()
    actions, states = np.array(actions), np.array(states)
    np.savez(trace_dir / "act_trace.npz", state=states, action=actions, anchor=anchor_vec, joints=np.array(config.JOINTS))
    grasp, grasp_xy = grasp_point(actions)
    descended = bool(len(actions)) and float(np.abs(actions[:, 1:3] - anchor_vec[1:3]).max()) >= STALL_MOVE_DEG
    return {"steps": len(actions), "duration_s": len(actions) / config.FPS, "stop_reason": stop_reason,
            "aborted": stop_reason == "aborted", "held_at_s": held_at, "slipped": slipped, "stall_warning": warned,
            "descended": descended, "floored_steps": int(floored_steps), "grasp_step": grasp, "grasp_xy": grasp_xy}
