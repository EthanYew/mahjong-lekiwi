# Code map

What each file is for. `tests/test_docs.py` fails on a script not listed here.

| | |
|---|---|
| `.claude/rules/`, `.claude/skills/` | what must stay true of each part of the code, loaded with it; repeated jobs, listed in the README |
| `mahjong/config.py` | the one setup: ports, cameras, table plane, rest poses, defaults |
| `mahjong/kinematics.py` | closed-form FK/IK; `pincer_tip`, `hover_command`, `grasp_point` |
| `mahjong/arm.py` | all motion (`hover`, `place`, `rest`, `go_to_pose`) |
| `mahjong/policy.py` | registry (`data/policies.json`), loading, the ACT control loop |
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
