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
  - "mahjong/wrist_usb.py"
  - "scripts/wrist_usb_watch.py"
---
Cameras, the detector and telling what's on the mat. Evidence in `docs/invariant_notes.md`.

- After packing up or moving anything: `python scripts/setup_check.py` (camera vs robot, table plane, a tile
  in the wrist view at the hover on 3 spots, parked pose, brightness) against `data/setup_reference/`, taken
  with `--save-reference` on a setup that worked. Nudge the camera back to green rather than recalibrating.
- **The wrist camera can drop out mid-session.** No frames for `camera.LOST_S` means it's lost in every script:
  `policy.run` stops (`stop_reason` "camera"), the bench voids and reruns, record.py discards the episode. Reopen with
  `hardware.reopen_wrist_camera` in any new script: it waits, cuts the port's power once, and only then asks a person, and each
  one logs how it ended (`wrist_camera_recovery`; `session_report.py`). The cause is the USB link, not the driver (`error -71`,
  `disabled by hub (EMI?)`; 0 disconnects in 446 picks to 2026-10-27, 89 since): no driver or `uvcvideo` option can fix it, and
  the kernel's own port power cycles haven't. **The 10s wait is what recovers it; our power cycle never has** (94 losses to
  2026-11-13): 85 came back by themselves (median 1.55s), and all 9 that reached the cycle still needed a person (7) or a quit
  (2) — 0 for 9. Once the wait has failed the port needs hands, so the cycle only adds ~10s before the prompt. Which part:
  `scripts/wrist_usb_watch.py --kernel-only`, in a second terminal during a real run; its `still` and `by-hand` watches are
  controls and catch nothing (4 watches, 16 min, 0 drops, against ~19 disconnects/h while a script was running).
  Evidence: `docs/invariant_notes.md`.
  **It can also keep sending torn frames** (rows shifted or smeared by lost USB data; it streams YUYV, so nothing fails to
  decode): 20261113_100710_bench ran 25 minutes on them, both policies closing 35-78mm in the air, 10/34 picked against 18/19
  on clean frames, and nothing stopped it. `camera.torn` (a share of rows jumping between frames, which the arm's own motion
  doesn't do: no clean attempt in 1591 videos reached the stop) makes `policy.run` stop as "camera" after
  `TORN_STOP_FRAMES` before the first hold (`camera_problem` "torn"). Tears had been there since 2026-10-28 in short runs.
  A torn camera reopens at once with no prompt, so a script reruns a camera-stopped pick `camera.RERUNS` times and then
  counts it as a miss: crude_run.py and base_cam_collect.py reran it with no limit until 2026-11-13. record.py doesn't stop on
  them: each episode row says its `torn_frames` before the first hold (none logged before 2026-11-13, the whole episode on 11-13 and 11-14), and saving warns from `TORN_STOP_FRAMES`, for a person to judge before training.
  Unconfirmed on the arm.
  **It can also come back on USB and still be deaf, which looks like a hang, not a drop.** After a replug on 2026-11-13 the
  camera enumerated but every UVC control transfer timed out (`-110`), each taking 5s: `uvcvideo` permanently disabled
  Saturation, auto White Balance, Gamma, Gain and Power Line Frequency on that instance, and streaming never started (a frame
  read still hung 3 minutes later). The readiness check blocks in the kernel per query and a blocking ioctl swallows Ctrl-C, so
  the terminal sat silent ~3 minutes before the interrupt landed (`20261113_110111_holdup`, which then quit cleanly with its 50
  holds). Those controls stay disabled until it re-enumerates, and exposure and white balance are part of the policy's input, so
  replug it and run `setup_check.py` rather than carrying on. Connected but silent is also the one case `usbreset` fits.
  **It can also come back slow.** After a replug on 2026-11-12 it enumerated at 12 Mbit/s (USB 1.1; the kernel: "not running
  at top speed") and gave 160x120 magenta frames. It opened and read, so the reopen took it, and v8 ran on them to the 16s cap
  closing 13mm high: that looked like a policy freeze (`20261112_155318_holdup`, attempts 050-051). So every camera open checks
  the frame size (`hardware.check_resolution`, with the link speed in its message) and a wrong one is not ready. Waiting
  doesn't fix a slow link, so a reopen asks for the replug at once rather than wait 10s saying not to replug.
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
- **The wrist and base cameras can only be told apart by USB port.** They report the same ID and serial, and on 2026-11-05 `/dev/wristcam`
  pointed at the base camera. The udev rule names them by port (wrist **3-5**, base 3-9 on the NUC), so keep each cable in its port —
  and moving one means editing `/etc/udev/rules.d/99-mahjong-cameras.rules` and `config.WRIST_USB_PORT`, or `/dev/wristcam` simply
  isn't there and every script stops at `hardware.connect`. `setup_check.py` doesn't check this yet. Evidence: `docs/face_down_plan.md`.
- **The wrist camera's USB trouble was the NUC's port 3-8, not the camera and not the cable** (2026-11-13). Same camera, same cable,
  a different receptacle on the same root hub: on 3-8 it enumerated fine and then timed out (`-110`) setting the UVC probe control, so it
  never streamed — in Cheese or anywhere; moved to 3-5 it gave 267 frames in 10s at 26.7fps, 0 failed reads, 0 kernel lines. So the 94
  logged losses, the `-71`s, the `EMI?` disables and the position-correlated drop rate were all one marginal receptacle. Don't put the
  wrist camera back on 3-8, and treat a port as a suspect before the camera: the swap takes a minute and the cable swap needs a spare.
  Evidence: `docs/invariant_notes.md`.
- **The wrist camera's USB storm takes Bluetooth audio down with it.** The camera (3-8) and the Bluetooth radio (3-14) are on
  the same root hub, so a flapping camera port starves it: on 2026-11-13 the bluez transport failed mid-storm and Firefox's
  audio stream died with an I/O error, in a session with 72 re-enumerations of 3-8. Desktop audio or Bluetooth going strange
  during a bad session is this fault, not a second one to chase.
- **The detector doesn't see tile backs, even at 1% confidence** (6 face-down tiles, 5 frames, no boxes; 34 on a face-up scan, 2026-11-09). A
  face-down tile goes by saved position, and scans can't confirm it. Evidence: `docs/retrospectives/2026-11-09_face_down_progress.md`.
  The fix is a 39th class, `back` (`config.BACK`, README "Teaching the detector tile backs"), not colour (see the pale-mat invariant). A back box
  proves a tile and which way up, never its name: `identity.scan_votes` gives it no votes, and its hover and return samples are logged
  `excluded` until the live weights pass `detector.back_centres` (tiles flipped in place: median back-minus-face offset within 2mm; `detector_eval.py` saves it, `detector.backs_checked` reads it). Back boxes under `config.BACK_MIN_HEIGHT_M` above the table
  are mat pattern (`perception.plausible`). The scripts ask `perception.sees_backs` and keep the saved-position path for weights without the
  class (where training stands: ROADMAP "Seeing tile backs"). **Every face-down frame of a layout is one scene by its labels**
  (a back on each spot), so a split by labels sent all 1440 backs to valid on 2026-11-16: `split_by_scene.py` deals them by
  capture session, and `train_detector.py` refuses a class with no training boxes.
