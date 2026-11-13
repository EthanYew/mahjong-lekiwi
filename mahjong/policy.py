"""ACT pickup policies: the registry, loading, and the control loop.

Policies are loaded with the pre/post-processors saved next to their weights, so normalisation is
whatever the policy was trained with (MahjongBot normalised by hand from a separately-named
dataset's stats, which broke silently whenever the name drifted).

Dataset convention (every dataset so far): state and action are joint deltas from the pose the
episode started at. So the loop anchors at the measured pose when it takes over, feeds
state - anchor, and commands anchor + predicted delta. The hover a policy starts from has to match
its episodes' starts; that hover is stored with the policy in data/policies.json.
"""
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from . import config, kinematics
from .arm import holds_tile
from .camera import LOST_S, TORN_STOP_FRAMES, torn

STALL_CHECK_S = 4.0
STALL_MOVE_DEG = 2.0
CAMERA_LOST_S = LOST_S  # no wrist frames for this long: stop the attempt, it can't count
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
    config.POLICIES.write_text(json.dumps(reg, indent=2) + "\n")
    if make_default:
        set_default(policy_id, evidence="none: made the default at registration (make_policy.py --default)")


def set_default(policy_id, evidence, why="", since=None):
    """Make policy_id the default, with what it was decided on (a bench session, usually). The note travels with
    the default (tests/test_docs.py fails if they disagree): 2026-10-29's switch to v4 was a hand edit whose
    reason no file kept. `since`: the day it was decided (default today), for a note written after the fact."""
    reg = registry()
    if policy_id not in reg["policies"]:
        raise SystemExit(f"{policy_id} isn't registered: {', '.join(reg['policies'])}")
    if not evidence:
        raise SystemExit("say what the switch rests on (--evidence, e.g. a bench session folder)")
    reg["default"] = policy_id
    reg["default_note"] = {"policy": policy_id, "since": since or time.strftime("%Y-%m-%d"), "evidence": evidence, "why": why}
    config.POLICIES.write_text(json.dumps(reg, indent=2) + "\n")
    return reg["default_note"]


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

    def run(self, arm, camera, trace_dir, abort=None, mark=None, log=print, takeover=None):
        return run(self, arm, camera, trace_dir, abort=abort, mark=mark, log=log, takeover=takeover)

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


def next_command(policy, frame, state, anchor_vec, limits, last_safe):
    """One policy step from the hover `anchor_vec`: (absolute command within the joint limits and above
    the table floor, whether the floor held it back). state: measured joints, config.JOINTS order."""
    delta = policy.step(frame, state - anchor_vec)
    command = {j: float(np.clip(anchor_vec[i] + delta[i], *limits[j])) for i, j in enumerate(config.JOINTS)}
    return tip_floor(command, last_safe)


def run(policy, arm, camera, trace_dir, abort=None, mark=None, duration_s=config.ACT_DURATION_S, log=print, takeover=None):
    """Run `policy` from wherever the arm is (the hover) until the tile is held and lifted for
    HOLD_CONFIRM_S, `abort` (an Event) is set, or duration_s runs out. `mark` (an Event with clear())
    is a person saying "something happened here": each time it's set the time is noted and it's
    cleared, and the policy carries on. Saves to trace_dir: act_trace.npz, wrist.mp4 (every frame; frame k
    is the image behind act_trace row k, plus one more when it stops on the done check) and a wrist jpg
    every 100 steps. Returns a summary: stop_reason (done / cap / aborted / camera: no wrist frames for
    CAMERA_LOST_S, or TORN_STOP_FRAMES torn frames before the tile was first held: camera_problem "lost" / "torn"; either
    way the attempt says nothing about the policy), torn_frames (torn frames before the first hold), duration_s, held_at_s, slipped
    (held, then lost during the hold check), stall_warning, descended, floored_steps, grasp_step,
    grasp_xy (the tool point where the gripper was first commanded shut — where a return should put
    the tile), marks_s (when it was marked, s).

    takeover (a mahjong.takeover.Takeover): A on its gamepad hands a person the arm at any step, and from then on the
    gamepad drives instead of the policy, for up to takeover.MAX_S more, until A again ("pressed"), the tile held and
    lifted ("done") or Esc ("discarded": the recovery is thrown away). Every frame from the hover on goes to it. The
    summary adds takeover_step (None if nobody took over), takeover_s and takeover_end; grasp_xy and grasp_clearance_m
    are then the person's close (where the tile was gripped, which a return aims from), and policy_grasp_xy /
    policy_grasp_clearance_m the policy's own, if it closed before the takeover."""
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
    previous, torn_frames, ever_held, camera_problem = None, 0, False, None
    stall_step = int(STALL_CHECK_S * config.FPS)
    confirm_steps = int(HOLD_CONFIRM_S * config.FPS)
    log(f"ACT ({policy.id}) for up to {duration_s:.0f}s, stopping once the tile is held and lifted — Esc stops it"
        + ("; A on the gamepad takes over" if takeover is not None else ""))
    taken, taken_end, end = None, None, int(duration_s * config.FPS)
    if takeover is not None:
        from .takeover import MAX_S
        takeover.begin()
    video = None        # every frame the policy saw, for replaying a new policy on this attempt (ROADMAP 2)
    try:
        for step in itertools.count():
            if step >= end:
                break
            t0 = time.time()
            if abort is not None and abort.is_set():
                stop_reason = "aborted"
                taken_end = "discarded" if taken is not None else None
                log(f"  stopped at step {step}")
                break
            if takeover is not None and takeover.pressed():
                if taken is None:       # from the policy's last command: no drop, the gripper held
                    taken, end = step, step + int(MAX_S * config.FPS)
                    takeover.start(arm, step)
                else:
                    stop_reason = taken_end = "pressed"
                    log(f"  t={step / config.FPS:4.1f}s A again: the recovery ends here")
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
                    stop_reason, camera_problem = "camera", "lost"
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
            # torn frames count until the jaws first hold: a fast lift after it blurs 1-3 frames into a tear's numbers
            ever_held = ever_held or holding
            if not ever_held and torn(previous, frame):
                torn_frames += 1
                if torn_frames == 1:
                    log(f"  wrist camera frame torn at step {step}")
                if torn_frames >= TORN_STOP_FRAMES:
                    stop_reason, camera_problem = "camera", "torn"
                    log(f"  {torn_frames} torn wrist camera frames — attempt stopped (the policy can't see the tile)")
                    break
            previous = frame
            tip = kinematics.pincer_tip(measured)
            lifted = tip[2] - config.grasp_z(tip[0], tip[1]) >= LIFTED_M
            if holding and lifted:
                if done_since is None:
                    done_since = step
                    held_at = held_at if held_at is not None else step / config.FPS
                elif step - done_since >= confirm_steps:
                    stop_reason = "done"
                    taken_end = "done" if taken is not None else None
                    log(f"  t={step / config.FPS:4.1f}s tile held and lifted for {HOLD_CONFIRM_S:.1f}s — done")
                    break
            elif done_since is not None:
                done_since, slipped = None, True
                log(f"  t={step / config.FPS:4.1f}s grip or lift lost during the hold check")
            if taken is None:
                command, floored = next_command(policy, frame, state, anchor_vec, limits, last_safe)
                floored_steps += floored
                arm.send(command)
            else:
                command, floored = takeover.step(arm, 1.0 / config.FPS), False      # sends it
            last_safe = command
            if takeover is not None:
                takeover.add(frame, state, command)
            states.append(state)
            actions.append([command[j] for j in config.JOINTS])
            if step == stall_step and not warned and taken is None:
                moved = float(np.abs(np.array(actions)[:, 1:3] - anchor_vec[1:3]).max())
                if moved < STALL_MOVE_DEG:
                    warned = True
                    log(f"  WARNING: {STALL_CHECK_S:.0f}s in, lift/elbow have moved {moved:.1f} deg — the policy may be "
                        f"waiting at the hover. Esc to stop.")
            if step % config.FPS == 0 and taken is None:
                log(f"  t={step / config.FPS:4.1f}s lift {command['shoulder_lift']:+6.1f} elbow {command['elbow_flex']:+6.1f} "
                    f"gripper {command['gripper']:5.1f}" + (" [tip floor]" if floored else ""))
            arm.sleep(max(0.0, 1.0 / config.FPS - (time.time() - t0)))
    finally:            # an interrupted attempt keeps a readable video
        if video is not None:
            video.release()
    policy.reset()
    if taken is not None and taken_end is None:
        taken_end = "camera" if stop_reason == "camera" else "cap"
    actions, states = np.array(actions), np.array(states)
    np.savez(trace_dir / "act_trace.npz", state=states, action=actions, anchor=anchor_vec, joints=np.array(config.JOINTS),
             takeover_step=np.array(-1 if taken is None else taken))
    grasp, grasp_xy = kinematics.grasp_point(actions)
    clearance = kinematics.grasp_clearance(states, actions)
    person = {}
    if taken is not None:       # the policy's own close (if any) before the takeover, and the person's after it
        mine = actions[:taken]
        own_xy = kinematics.grasp_point(mine)[1] if len(mine) else None
        person = {"takeover_step": taken, "takeover_s": round(taken / config.FPS, 2), "takeover_end": taken_end,
                  "policy_grasp_xy": own_xy,
                  "policy_grasp_clearance_m": kinematics.grasp_clearance(states[:taken], mine) if len(mine) else None}
        theirs, theirs_xy = kinematics.grasp_point(actions[taken:])
        if theirs is not None:
            grasp, grasp_xy = taken + theirs, theirs_xy
            clearance = kinematics.grasp_clearance(states[taken:], actions[taken:])
        log(f"  takeover at {taken / config.FPS:.1f}s, ended: {taken_end}")
    elif takeover is not None:
        person = {"takeover_step": None}
    descended = bool(len(actions)) and float(np.abs(actions[:, 1:3] - anchor_vec[1:3]).max()) >= STALL_MOVE_DEG
    return {**person, "steps": len(actions), "duration_s": len(actions) / config.FPS, "stop_reason": stop_reason,
            "aborted": stop_reason == "aborted", "held_at_s": held_at, "slipped": slipped, "stall_warning": warned,
            "descended": descended, "floored_steps": int(floored_steps), "grasp_step": grasp, "grasp_xy": grasp_xy,
            "grasp_clearance_m": clearance, "marks_s": marks, "torn_frames": torn_frames,
            **({"camera_problem": camera_problem} if camera_problem else {})}
