---
paths:
  - "mahjong/face_down_rig.py"
  - "mahjong/wall_test.py"
  - "scripts/wall_test.py"
  - "mahjong/crude_run.py"
  - "scripts/crude_run.py"
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
Face-down tiles: drawing from the wall station, the base camera's reader, the crude turn. Plan and evidence in
`docs/face_down_plan.md`.

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
