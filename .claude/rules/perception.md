---
paths:
  - "mahjong/perception.py"
  - "mahjong/camera.py"
  - "mahjong/detector.py"
  - "mahjong/layout.py"
  - "mahjong/wrist_check.py"
  - "mahjong/wrist_photos.py"
  - "mahjong/setup_check.py"
  - "mahjong/hardware.py"
  - "scripts/setup_check.py"
  - "scripts/layout.py"
  - "scripts/detector_eval.py"
  - "scripts/train_detector.py"
  - "scripts/mine_frames.py"
  - "scripts/capture_frames.py"
  - "scripts/split_by_scene.py"
  - "scripts/wrist_capture.py"
  - "scripts/wrist_review.py"
  - "data/detector_exam.json"
---
Cameras, the detector and telling what's on the mat. Evidence in `docs/invariant_notes.md`.

- After packing up or moving anything: `python scripts/setup_check.py` (camera vs robot, table plane, a tile
  in the wrist view at the hover on 3 spots, parked pose, brightness) against `data/setup_reference/`, taken
  with `--save-reference` on a setup that worked. Nudge the camera back to green rather than recalibrating.
- **The wrist camera can drop out mid-session.** No frames for `camera.LOST_S` means it's lost in every script:
  `policy.run` stops (`stop_reason` "camera"), the bench voids and reruns, record.py discards the episode. Reopen with
  `hardware.reopen_wrist_camera` in any new script. If `/dev/wristcam` is gone, only a replug helps. Evidence:
  `docs/invariant_notes.md`.
- **The wrist look sees a tile face, not which tile.** A knocked tile near a lost one fills the same disc, so the
  bench doesn't look within `wrist_check.CROWDED_M`. A pale surface fills it with no tile at all, so without enough dark
  surface in view (`wrist_check.MIN_DARK_SHARE`) it can only say the view matches an earlier look at that place, and a
  person is asked the first time (`wrist_check.PlaceViews`). Evidence: `docs/invariant_notes.md`.
- **Whether a tile is there and which tile it is are separate questions.** The detector boxes tiles it can't name (3C at a
  median 0.18, so it looked gone in 45 of 46 scans), so a box down to `config.DETECTION_PRESENCE_CONFIDENCE` proves a tile
  is there (`perception.faint_tiles`, `tile_faint` events), and only a name over `DETECTION_CONFIDENCE` is acted on as one.
  A layout's labels are what taught it, not what's on the mat: never use them as a tile's name. Evidence:
  `docs/invariant_notes.md`.
- **A tile's white face can't be found by colour on a pale mat**: 1 good read in 950 on the striped mat, and slices of
  stripe passed as tiles. Where it matters, use what a tile looks like: the detector's box on the RealSense
  (`layout.find_tile`, which can't read yaw there, so green is position only) and the reference's picture of the tile in
  the wrist view (`setup_check.match_tile`, so a spot needs the same tile each time). Evidence: `docs/invariant_notes.md`.
- **The tile detector's exam is never trained on either**: whole capture sessions listed in
  `data/detector_exam.json`, labelled in a Roboflow project of their own (`mine_frames.py` and
  `train_detector.py` keep them out). New weights replace `assets/tile_detector.pt` only if they beat it
  on the exam (`detector_eval.py --against`).
