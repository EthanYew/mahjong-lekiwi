---
paths:
  - "mahjong/face_down_rig.py"
  - "mahjong/wall_test.py"
  - "scripts/wall_test.py"
  - "mahjong/crude_run.py"
  - "scripts/crude_run.py"
  - "mahjong/face_down_play.py"
  - "scripts/play_face_down.py"
  - "tests/test_play_face_down.py"
  - "mahjong/base_cam.py"
  - "scripts/base_cam_train.py"
  - "mahjong/base_cam_collect.py"
  - "scripts/base_cam_collect.py"
  - "mahjong/camera_poses.py"
  - "scripts/teleop_view.py"
  - "mahjong/privacy.py"
  - "scripts/blur_faces.py"
  - "mahjong/photo_review.py"
  - "scripts/base_cam_review.py"
  - "data/camera_poses.json"
---
Face-down tiles: drawing from the wall station, the base camera's reader, the crude turn, face-down play. Plan and evidence in
`docs/face_down_plan.md`.

- **Face-down play asks a person to confirm every read, unless `--unattended`** (`scripts/play_face_down.py`). The base camera's reader
  has passed no exam, and its two wrong reads in its first crude run came at 41% and 61% (`20261117_140038_crude`). So a person checks
  the 13 once and each drawn tile. `--unattended` takes the reader's names at the owner's request (2026-11-17): wrong names can get in
  silently, so the gate is checked after the hand instead: **an unattended hand counts toward play-ready** (the owner's decision,
  2026-11-20) only once every hold's photo has a verdict (`base_cam_review.py --play`, then `play_face_down.py --report`) and the
  end check matched. One wrong name found there is a silent error: it stops play, as play.md says. **The face-down target is 8 of 10
  consecutive hands clean, 0 silent errors** (`docs/autonomy_plan.md`, "Face-down hands"). A hand not yet reviewed counts for nothing.
  A low read gets a second hold (`--reshow-below` 0.6, a guess from 6 reads). Unconfirmed on the arm.
- **A face-down turn doesn't go to rest between moves**, only before a question, the RealSense window, after a miss and at the end
  (`Rig.direct`, `--via-rest` for the old way). motion.md has the detail.
- **A hold saved in play trains nothing until a person has reviewed it** (the owner's rule, 2026-11-17). Unattended, its name is the
  reader's own, and training on that teaches the reader its mistakes. Every hold is `review_first`, and `base_cam.rows` is the one check,
  so training, `base_cam_collect`'s counts and `--counts` all leave it out until `data/holdup_review.jsonl` has a verdict
  (`base_cam_review.py --play`).

- **A green hand in the wrist view pulls v8 back at station F.** The jaws moved about 5mm further out on an empty mat and 2-8mm back with the hand laid, closing short (2026-11-09), so a wall result counts only with the hand in place. Evidence: `docs/face_down_plan.md`.
- **A two-high stack behind the target pulls v8 forward, and a top tile gets the next column's** (station F, 2026-11-09; 3 top tries). The jaw tips moved 9.4-9.7mm out from the hover with a stack
  behind, 4.8mm beside flat tiles, and a raised hover starts ~4mm further out. Top tiles 0/3, closing at a fine height: the error is forward, not height. A hover 10mm nearer the robot fixed the top tile (8/8) and broke the bottom one (0/2, closing early and shallow, against 10/10 at the regular hover), so
  only top tiles get it (`wall_test.py --top-back-mm`). Evidence: `docs/face_down_plan.md`.
- **The base camera sees the room, so what it saves is face-blurred.** A pose model (`assets/yolo11n-pose.pt`), not Haar cascades,
  which blurred the tile's printing; the held tile close to the camera is seen as a person, so a "person" over 65% of the frame is
  ignored (`mahjong/privacy.py`). The blur lowers the chance of a face in a photo; it doesn't rule it out.
- **A classifier learns whatever separates the tiles, the scene included.** Each tile's base-camera holds taken in a row, a classifier with the tile
  blacked out still scored 99% (2026-11-06). So tiles are collected interleaved and moved between spots, `base_cam_train.py` runs a greyed-out control,
  and a score counts on a session never trained on. Collected that way and cropped to the tile, a reader trained on either the cluttered room or a blank white
  background read the other at 97% (6 tiles, 2026-11-09). Another room is untested.
