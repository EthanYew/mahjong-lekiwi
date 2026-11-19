---
paths:
  - "mahjong/arm.py"
  - "mahjong/kinematics.py"
  - "mahjong/config.py"
  - "mahjong/hardware.py"
  - "mahjong/sim.py"
  - "mahjong/hover_correction.py"
  - "scripts/bench.py"
  - "scripts/record.py"
  - "scripts/play.py"
  - "scripts/collect_corrections.py"
  - "scripts/hover_samples.py"
  - "tests/test_arm.py"
  - "mahjong/motion_watch.py"
  - "tests/test_motion_watch.py"
---
What must stay true of anything that moves the arm. Each cost a debugging round on hardware; evidence in `docs/invariant_notes.md`.

- **Never start motion from the measured pose.** The servos (lerobot's I=0) sit 5-13mm below their
  command; re-commanding the measured pose makes the arm drop again. Interpolate from `Goal_Position`
  (`Arm.move_to`), seed teleop from `Arm.commanded()`. `tests/test_arm.py` catches a regression.
- **Arrive at hover heights from above** — a P-only servo holding against gravity stops further short
  arriving from below. `Arm._settle` handles it.
- **Hold the gripper's commanded value** until a release. Policies grip by commanding 0 (jaws stall at
  ~11 on a tile); commanding any fixed "closed" value or the measured reading loosens or opens them.
- **Heights are above the grasp point** (`config.grasp_z`, table + 5mm); the tile top is ~15mm higher.
- **Rest is where the unpowered arm settles** (`config.REST_POSE`, from `sessions/parked_pose.json`). The old
  folded rest sagged ~8° onto the stops at every disconnect, and each connect then lifted the arm via
  `RAISED_REST` and folded it back before doing anything. Re-measure it if the arm or its stops change.
- **Moves between rest and the table go via `RAISED_REST`** so the gripper clears the LeKiwi base.
- **In play, a held tile goes straight from its pickup to where it's set down; everything else goes via rest** (`Bench.direct`, `Arm.travel`;
  `play.py --via-rest` for the old way; the bench always rests). The RealSense can't scan with the arm out: at the discard pose it covers S7-S9. So
  that pickup has no scan of its own, and the scan after the release answers for it: `handled_xy` on an after-return scan, or an `after_pickup` scan
  marked `released` for a discard (`interventions.handled_points`), so what it handled isn't taken for a person's doing. A miss or a key pressed goes to
  rest and scans at once. The return model was measured arriving from rest, so the move to a slot turns the pan past and comes back the same way
  (`TRAVEL_PAN_BACK_DEG`, reasoned from the height invariant, not measured). Unconfirmed on the arm.
- **Face-down play goes from move to move and rests only to ask** (`Rig.direct`; `crude_run.py` and `play_face_down.py`, `--via-rest`
  for the old way; `base_cam_collect` and `wall_test` always rest). Nothing scans the mat mid-turn, so the rests were only travel: 5 trips
  a sim turn, now 2. It rests before every question (`crude_run.Run.resting`: a person may reach in, and an arm held out while they type
  only heats the servos), before the RealSense guide window (`face_down_rig.prompt`), after a miss, an abort or a takeover, and at the
  end. `hover(direct)`, `place(direct, rest_after=False)`, `discard(direct, rest_after=False)` and `go_to_pose(direct)` (which lifts
  first) are the pieces. The swings to and from the base camera's poses (by the LeKiwi base, which the watchdog doesn't model) still
  pass `RAISED_REST`, without folding down (`Arm.to_raised_rest`, in `Rig.show` on every way out; `tests/test_crude_run.py` checks no
  move goes straight between the mat and a pose). The container's direct moves are the path face-up play has run since 2026-11-05,
  coming back the same way. ~16s less moving a sim turn. Unconfirmed on the arm (2026-11-20).
- **Every move off the table rises straight up first** (`Arm.lift`, called by `rest`, `hover` and `place`, so
  bench, record and every other script share it): 5mm steps keeping the jaws' tilt and roll
  (`kinematics.straight_up`), up to 80mm. The old lift was one joint-space move to the IK's pose, which curved
  through the tiles, and it was skipped when the wrist had been tilted more than 30° from that pose: a discarded
  teleop then swung to rest from the mat.
- **Wrist roll runs out at the left edge.** The hover turns the roll to keep the jaws square to the mat:
  ~−147° at (13, +19) cm, 23° from the −170° command limit. A tile turned clockwise more than ~20° there
  is only reachable by rolling ~+150° the other way, jaws flipped and the wrist image upside down from
  every other demo (golden set T1, 2026-10-26). Expect any policy to fail it; don't place training tiles so.
- **Computed heights drift with reach: far spots are ~10-15mm lower than the FK says.** The settle and the tip floor
  both use FK, so neither sees it; likely flex under load or a tilted table plane. Evidence: `docs/invariant_notes.md`.
- **The far row is at the arm's full reach, not past what the servos can hold.** Pushing past the gamepad's reach
  limit does nothing, and lowering slides the gripper back along the reach circle, which looks like a sag.
  `gamepad.warn_if_at_reach` says so after the hover. Tilt the jaws forward (L/R), or keep tiles inside base r ~0.27m
  (`layout.py`'s `r` shows it). Evidence: `docs/invariant_notes.md`.
- The hover correction, the return correction and the table plane are tied to `assets/T_cam_to_base.npy`;
  recalibrating the RealSense invalidates both corrections (records are set aside automatically; returns
  then aim at home + the grasp's offset until new returns are logged) and may move the plane.
- **A held tile's jaw reading can drift under `holds_tile`'s margin.** It fell from 9.5 to 3.0 over 2026-11-09 with the command at 0 and empty jaws still at 0.6-0.9. Below 5 a good pick reads as not held, runs to the 16s cap, and a retry opens the jaws. Likely cause: the plastic gripper was found cracked and was superglued (owner, reported 2026-11-11); held tiles read 10.9-11.0 again afterwards (`20261111_161703_crude`). A held reading falling over a session is a sign to check the jaws for cracks. Evidence: `docs/face_down_plan.md`.
- **Every simulated arm command is checked against these rules** (`mahjong/motion_watch.py`, called by `SimRobot.send_action`;
  `tests/conftest.py` fails any test whose sim arm broke one), so every end-to-end sim test is also a trajectory test: joint limits
  (widened to `REST_POSE` and the poses taught on the arm), the wrist roll's ±170°, a jump between two commands (a move not started
  from the commanded pose), jaw tips under the table, open jaws onto a tile that isn't the target, a held tile set onto another.
  It doesn't check clearing the LeKiwi base (no model of it; `tests/test_arm.py` covers the moves from rest). It found, on its first run,
  that `bench.py --sim` hovered with the real arm's correction over tiles laid where they're detected, so play's sim hands picked the
  neighbour and set tiles down on others (2026-11-13). A test that sends a bad command on purpose is `@pytest.mark.motion_unwatched`.
