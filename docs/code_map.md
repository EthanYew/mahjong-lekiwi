# Code map

What each file is for. `tests/test_docs.py` fails on a script not listed here.

| | |
|---|---|
| `.claude/rules/`, `.claude/skills/` | what must stay true of each part of the code, loaded with it; repeated jobs, listed in the README |
| `mahjong/config.py` | the one setup: ports, cameras, table plane, rest poses, defaults |
| `mahjong/kinematics.py` | closed-form FK/IK; `pincer_tip`, `hover_command`, `grasp_point` |
| `mahjong/arm.py` | all motion (`hover`, `place`, `rest`, `go_to_pose`) |
| `mahjong/policy.py` | registry (`data/policies.json`), loading, the ACT control loop |
| `mahjong/recipe.py` | a new version's datasets and options checked against the default policy's (`make_policy.py`, `train_remote.py`) |
| `mahjong/perception.py`, `camera.py`, `gamepad.py` | RealSense tiles, wrist camera, XLeRobot gamepad |
| `mahjong/handoff_trace.py`, `scripts/handoff_report.py` | what the arm does once the gamepad has it (collect_corrections' `handoff/` traces): sag, drift, command moves |
| `mahjong/hover_correction.py` | measured hover correction (`data/hover_corrections.jsonl`; live sources in `data/hover_model.json`) |
| `mahjong/return_correction.py`, `scripts/return_samples.py` | where to set a held tile down so it lands at its home (`data/return_corrections.jsonl`, from each return's after-return scan) |
| `mahjong/dataset.py`, `quality.py` | trims + re-anchoring + resize; episode quality |
| `mahjong/session.py`, `sim.py` | session folders; simulated arm/camera/perception/policy |
| `mahjong/bump_report.py`, `scripts/bump_report.py` | every pickup and return before / after from the scans, for a person to say which tiles really moved (`scan_verdict` events) |
| `mahjong/placements.py`, `name_ledger.py`, `scripts/name_report.py`, `docs/name_data.md` | which tiles get misread and how much it matters: a person's word on what's at each spot (bench, before trials; play's answers and audit), every read scored against it per placement, with reviewed wrist photos for older sessions |
| `mahjong/setup_check.py`, `scripts/setup_check.py` | is the workspace where it was: measurements vs a reference (`data/setup_reference/`) |
| `mahjong/interventions.py` | what a person had to do: classes, questions, derived from session events |
| `mahjong/scorecard.py`, `docs/scorecard.md` | clean attempts (picked, nothing bumped, no one helped); the iteration check, the play-ready check and the strict paired rule |
| `mahjong/hand_truth.py`, `scripts/hand_frames.py` | detector frames labelled from a person's names in a session: scores per tile, an export to train on |
| `mahjong/wrist_check.py` | is a tile the scans lost still there: a face between the jaw tips at the hover (the bench looks before asking) |
| `mahjong/name_check.py`, `scripts/train_name_check.py`, `name_check_eval.py` | a second read of a tile's name from the wrist camera at the hover: in play, agreement changes nothing and disagreement asks; its exam is `data/name_check_exam.json` |
| `scripts/label_sheets.py` | one picture per tile name of every box labelled with it, to check a labelled set by eye |
| `mahjong/hand_view.py` | play's window: the mat zoomed on the tiles, every slot named, the tiles under review picked out with what they could be (`identity.Votes.worth_a_look`; it only marks, never asks), the wrist view of the tile asked about |
| `scripts/play.py`, `mahjong/game.py`, `identity.py`, `markers.py` | autonomous play (`docs/autonomy_plan.md`): the turn loop on the bench's pieces; shanten and the discard choice; name votes per hand slot; the ArUco drop zone (`data/drop_zone.json`) |
| `scripts/` | `record`, `collect_corrections`, `hover_samples`, `bench`, `session_report`, `quality_report`, `make_policy` |
| `scripts/status.py`, `scripts/merge_pr.py`, `mahjong/checkout.py` | where things stand (from `data/`); approve + merge a PR and close its branch; the checkout vs origin/main, merged branches and worktrees |
| `.claude/agents/merge-reviewer.md`, `scripts/review_brief.py`, `review_backtest.py`, `mahjong/code_health.py` | the post-merge review (`/post-merge-review`, `merge_pr.py --review`): the method, what it starts from, its score on past merges with known bugs (`data/review_backtest/`); ruff and copied functions, in the fast tests |
| `mahjong/layout.py`, `scripts/layout.py`, `spot_novelty.py` | tile layouts (`data/layouts.json`, committed once a recording uses one; ghost photos and captures in `data/layouts/` stay local) drawn on the RealSense view; place tiles to green; how far each spot is from a training recording's tiles |
| `mahjong/offline_eval.py`, `scripts/offline_eval.py` | scores per slice on episodes never trained on (`data/eval_split.json`) |
| `mahjong/detector.py`, `scripts/capture_frames.py`, `mine_frames.py`, `detector_eval.py`, `split_by_scene.py`, `train_detector.py`, `wrist_capture.py`, `wrist_review.py`, `mahjong/wrist_photos.py` | the tile detector's loop: scan frames saved in `sessions/*/scans/`, frames it struggled with exported pre-labelled for Roboflow, downloads re-split by scene, wrist-camera photos of each tile from jittered hovers (gathered from the sessions, reviewed in `data/wrist_review.jsonl`), the exam (`data/detector_exam.json`), fine-tuning (ROADMAP 10) |
| `mahjong/takeover.py` | A on the gamepad mid-pickup hands a person the arm in any script that runs the policy (`--dagger-dataset`); the recovery saved in record.py's DAgger format, which record.py shares |
| `mahjong/wrist_usb.py`, `scripts/wrist_usb_watch.py`, `scripts/wrist_usb/` | a wrist camera that fell off USB: wait, port power cycle (root helper, installed once), then a person; a watch of the link to find which part fails |
| `mahjong/base_cam.py`, `scripts/base_cam_train.py` | the base camera's tile reader (`assets/holdup_reader.pt`; not the RealSense's detector nor the wrist's name check, which read ~0% on this view): the hold-up photos, cropped to each pose's tile box (`data/camera_poses.json`), a fresh classifier trained on some holds and scored on others with the tile greyed out as a control (`docs/face_down_plan.md`, test 1b); `--save-reader` installs one, `--history` lists every score |
| `mahjong/photo_review.py`, `scripts/base_cam_review.py` | a person's verdicts on photos filed under a tile name and the window they give them in, shared by `wrist_review.py` and the base camera's photos; `base_cam_review.py --counts` is the holds per tile, fewest first, and its verdicts (`data/holdup_review.jsonl`) go into `base_cam.rows`, so training and the collect counts follow them |
| `mahjong/camera_poses.py`, `scripts/teleop_view.py` | gamepad teleop with the wrist and base cameras in one window; saves poses that hold a tile up to the base camera in `data/camera_poses.json` (test 2) |
| `mahjong/privacy.py`, `scripts/blur_faces.py` | faces blurred in every photo the base camera saves (pose model `assets/yolo11n-pose.pt`; a "person" over 65% of the frame is the held tile, not a person); the script does the same for sessions already on disk, dry run unless `--apply` |
| `mahjong/base_cam_collect.py`, `scripts/base_cam_collect.py` | the arm collects the base camera's training photos, every tile name short of 6 holds each way up (only interleaved sessions count): picks up a face-down tile laid on a layout spot, shows it at the saved `base_view_*` poses, sets it back; a tile on each spot, picked in turn and moved one spot along between arrangements, so holds of one tile never come in a row (the scene named the tile when they did); `--empty` visits the poses with nothing in the jaws first |
| `mahjong/face_down_rig.py` | the arm, cameras, policy and corrections the face-down scripts share (`Rig`: pick at a spot by saved position, show at the `base_view_*` poses, set back down) and the RealSense window that says where to lay a tile |
| `mahjong/crude_run.py`, `scripts/crude_run.py` | the whole face-down turn (`docs/face_down_plan.md`, test 8): draw from the station, show to the base camera, set on the drop zone, read with a person checking, play the discard. By memory and saved positions; what fails is logged as `crude_break` and done by hand; the reader comes from `base_cam_train.py --save-reader` |
| `mahjong/wall_test.py`, `scripts/wall_test.py` | drawing face-down tiles from one saved station a person feeds (`docs/face_down_plan.md`, "The wall: one station"): hover and push checks, flat / touching / top / bottom cases, a real cycle of draws; a person's one key on each attempt, since nothing can see a back; `--report` scores every real session |
| `scripts/train_remote.py` | start `make_policy.py` on a Windows/WSL training machine over SSH, detached via Win32_Process (keepawake suppresses sleep); `--sync --version V` pulls `data/policies.json` and `data/offline_eval/` back; `--status` checks progress; `--shutdown-train`/`--shutdown-local` for overnight runs. Config in `data/train_machine.json` (gitignored; template at `data/train_machine.example.json`) |
| `scripts/save_samples.py`, `mahjong/owed.py` | after a hardware session, the logged samples committed and pushed to `main` in one step; what's owed and overdue (a default's unrun checks, ROADMAP behind the registry, branches not closed, unsaved samples), printed by `status.py` and at each hardware session's start |
| `mahjong/motion_watch.py` | the motion watchdog: every command the simulated arm is sent, checked against the motion invariants; `tests/conftest.py` fails a test whose sim arm broke one |
