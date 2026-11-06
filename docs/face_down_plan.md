# Drawing from the wall: the face-down plan

Written 2026-11-05 from Ethan's `docs/retrospectives/face-down-pivot.md` and a review of it against the code and the
cameras. Nothing here has been tried on the arm yet. Numbers marked "estimate" come from the arm's geometry or one
photo, not from a run. Results go in ROADMAP.md's status, as usual.

## Why

Drawing from the wall is the biggest gap to the vision that hasn't been started. It stalled because every policy was
trained on flat, face-up tiles, and the plan assumed two cameras: the RealSense on the tripod and the wrist camera.

The LeKiwi base has a front camera of its own, the same model as the wrist camera. So the arm can draw a face-down tile
and hold it up to that camera, which reads it up close. Try a crude full run early, before polishing any single part.

**The hand ends up face down.** The arm can't flip a tile, so a drawn tile lands face down. That's forced, and it's also
wanted: it hides the hand from human opponents, which face-up tiles couldn't do.

## Decisions

**Ethan:**
- The base camera reads the tiles, and a person checks every read.
- The wall: two high, face down, short sides together, long sides along the robot's x axis, the same direction as tiles
  on the mat now. So the arm still grips the long sides. Its position comes from a marker or tape guide, with no tray.
- A single pick per take. Face-down tiles are tested flat before the wall.
- v1 is a demo. Discards are a non-issue: tiles can flip on the table or in the container, and a name on the screen is enough.
- v1 runs on the NUC. Later, the code and the cameras move to the Pi, with each camera in a specific USB port.
- The thresholds below, and a turn for v8 in play before any demos are recorded, including a hand with the problem tiles.
- Reads are checked in the moment where that's more efficient (see "Checking reads").

**Suggested by Claude:** a scripted camera pose (now taught by gamepad), one policy with a hover that changes per tile,
code tracking the stack instead of depth sensing, v7 as a baseline first, flat and stacked tested apart, the face-up
control, the 3-tile touching test (4b), the raised-hover test (5), the gamepad wall test (6) and the re-pick test (7b).

## What the review found

**The cameras can swap, and on 2026-11-05 they had.** Both report the same ID and serial (`USB2.0_CAM1`), and
`/etc/udev/rules.d/99-mahjong-cameras.rules` names `/dev/wristcam` on vendor and product ID alone, so the last one plugged
in gets the name. It pointed at the base camera (`video10`, USB port 3-9), not the wrist camera (`video8`, port 3-8).
v7 would have run on a view of the room, and the USB recovery would have power-cycled the wrong port. Only the port can
tell them apart, so a reboot test alone proves little: the names hold as long as each cable stays in its port.

**The detector can't see backs, so v7 can't run through the bench.** The bench only targets tiles it detects (`bench.py`
skips a spot with no detection), and bump and landing checks rely on the same scans. These tests need a small script that
aims at saved layout spots. "Clean" can't be worked out from scans, so a person marks bumps for now.

**A top tile needs the hover raised by one tile (20mm), and that shifts the whole descent.** ACT moves relative to its
hover, so a policy ignoring the image would close 20mm higher anyway. Top and bottom closing at different heights proves
nothing, so the doc's test for a blind policy can't work. Test 5 replaces it. Blind may not even be bad here, since a
replayed descent could still work on a stack. A bottom tile with its top gone is at the normal hover, like a flat one.

**A top tile has little margin below it.** Good picks close 5-15mm above the table (`docs/retrospectives/2026-11-03_eval_validity.md`), which
on a top tile is 25-35mm. Under 20mm the jaws are around both tiles: about 5mm of margin (estimate).

**v7 has never seen tiles touching end to end.** From the wrist camera that's one long green strip. A failed stack test
wouldn't say whether height or the strip caused it, hence test 4b.

**A face-down tile can be either way up.** The name classifier trained on turns of ±20° (`name_check._augment`), so one
upside down is new to it. The exam includes them, and each photo can be read both ways up.

**A kept tile is the hard part of a face-down hand.** The RealSense can't read it or confirm where it landed. Carried
tiles have landed up to 10.5mm off (`docs/retrospectives/2026-11-04_first_hands.md`), and v7's tolerance is about ±6mm. Hence test 7b.

**Answers to the doc's open questions, so far:**
- *Do the jaws hide the face?* Probably not: they close 5-15mm above the table, so the face sticks out past the tips. Test 2 confirms it.
- *How are bumps detected?* Compare RealSense frames before and after each pick with the arm at rest, and a person checks. Still to be tried.

## What the base camera sees

One still, 2026-11-05, cap off, arm at rest:
- It sits almost at table level, looking level or slightly up. Near-row tiles just appear at the bottom, seen edge-on.
- Those tiles are about 110px wide, against about 130px from the wrist camera at a hover. So a tile held over the near row
  should be close enough, which helps a wrist-trained classifier (estimate).
- Bright windows fill the background, and a held tile's face points away from them, so it may be dark.
- The discard tub is in the middle of the view, and may make a useful plain backdrop.

The geometry allows jaws pointing back at the base (poses with the tip at least 30mm above the table), but that calculation
ignores collisions, and nothing records where the camera sits. So teach the pose with the gamepad, as the discard pose
is taught. Jaws back and down, with the tile in the upper part of the frame, looks safer than level, which passes close
over the near row.

## Where the wall fits

Beside the hand, within r ≤ 0.27m at both heights (estimate; clearance from the base at the near ends not checked):

| Column | Stacks | Wrist roll (limit -170°) |
|---|---|---|
| y = +0.192 | 3 | -144° to -165°, tight at the end nearest the base |
| y = -0.223 | 2 | about -20°, comfortable, but next to the drop zone |

That's about 10 tiles. The last hands took 19-34 turns, so a demo win needs a pre-arranged wall and hand, and the writeup says so.

## The tests

Times are guesses. Each one can stop the plan before the next is paid for. Ten tries gives a signal for a decision, not a
reliability claim: 10/10 is still consistent with a true rate near 70%.

| # | Test | Cost | Stop or change course if | Needs |
|---|---|---|---|---|
| 0 | Fix the camera names (below), then a photo from each with the arm parked | 30 min | A camera is dead or can't see in front of it | sudo |
| 1 | **Hold tiles up by hand** wherever the arm can reach (close and at an angle, so no fixed distances), right way up and 180°, by day and evening, including 5C, 3C, 2C, 9C; score the classifier and the detector | 20 min | Nothing reads anywhere | capture script; no arm |
| 1b | **Can the view be learned?** Train a fresh small classifier on some holds, score it on others it never saw (`holdup_check.py`, 3 seeds) | 5 min | Under 40% right: look at the photos before capturing more. 70% or more on 3+ tiles and well above the best constant answer: capture at the real pose. In between: more holds, rerun | test 1's photos |
| 2 | **Teach the camera pose by gamepad** (`teleop_view.py`: both cameras in one window, `s` saves the pose). Put 5 tiles in the jaws; check swing, hidden faces, whether they set down flat | 30 min | Tiles swing or drop | saved pose |
| 3 | Run the detector at low confidence on a face-down tile | 5 min | If it boxes backs, targeting and the zone check may come free | one frame |
| 4 | **Face-down vs face-up on the same slots**, alternating, 10 each, single tiles, logging close height. Also how much green stands out from the mat | 1 hr | The thresholds below | test script |
| 4b | 3 flat tiles touching end to end, 5 tries | 15 min | Fails where 4 passed: the strip is the problem, not height | test script |
| 5 | A flat tile with the hover 20mm too high, 5 tries | 10 min | Closes about 20mm in the air: v7 is blind | test script |
| 6 | **Pick from a short wall by gamepad**, top and bottom, at a familiar mid-mat spot | 15 min | If a person can't, v7 won't. Also shows whether the jaws grab two tiles, and which end to draw from | gamepad |
| 7 | v7 on stacks at that spot: 10 top, 10 bottom, close height logged apart | 1 hr | The thresholds below | test script |
| 7b | Set a tile down face down, pick it up again from memory, 10 times, no scan | 30 min | Misses grow with each cycle: a kept tile needs a way to be found | test script |
| 8 | **The crude full run**, on the real columns: draw, show, read, check, place | 1 hr | It completes, with a list of where it broke | `play.py` changes |

Tiles are placed face up to green with `layout.py` first, taped around, then flipped, since the layout tool can't see backs.

**A hand on v8 runs alongside, and doesn't wait for these.** It's also the roadmap's hand with the hard tiles, so one hand
does both: `setup_check.py`, then `python scripts/play.py --layout even-hand13 --policy pickup_mat_cl_v8` with 5C, 3C, 2C, 9C and
S8 in play. The camera fix comes first.

### Thresholds, decided before the runs

With face-up at 8/10 or better as the control (below that, fix the setup before reading anything):

| Face-down | Then |
|---|---|
| 8-10 of 10 | Carry on |
| 4-7 of 10 | Try v8, then decide. Its 20.6% hover opening makes it a different test, not a repeat |
| 3 of 10 or fewer | Record demos: first try mixing face-down takes in with the face-up demos, before recording from scratch |

Test 7 reads the same way, top and bottom apart. Flat holds up and stacks fail: the problem is height, or the strip if 4b failed too.

### Checking reads

A person checks whether the robot's guess was right. The guess is logged first, so accuracy stays measurable, and the
tile's photo sits next to it, so Enter doesn't become a habit.
- **The full run: in the moment.** A wrong name gets fixed before the discard is chosen, which a check at the end can't do.
- **The exam (test 1): in aggregate**, as a photo sheet.

## The camera fix (test 0)

Replace the first line of `/etc/udev/rules.d/99-mahjong-cameras.rules` with these (not yet run):

```
SUBSYSTEM=="video4linux", KERNEL=="video*", ATTR{index}=="0", KERNELS=="3-8", ATTRS{idVendor}=="05a3", SYMLINK+="wristcam"
SUBSYSTEM=="video4linux", KERNEL=="video*", ATTR{index}=="0", KERNELS=="3-9", ATTRS{idVendor}=="05a3", SYMLINK+="basecam"
```

Then `sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=video4linux`. `ls -l /dev/wristcam` should
show `video8`. Those port numbers are the NUC's, and change on the Pi.

Also: a check in `setup_check.py` that the jaws are in the wrist view and the base camera is present; an invariant in
CLAUDE.md once it's in. Whether a second camera of this model makes the wrist camera's dropouts worse is an open
question, so `session_report.py`'s recovery log gets compared before and after. Both are on the same USB bus.

## What gets built

- A capture script for test 1, needing no arm.
- A test script for tests 4-5 and 7-7b: aims at a saved spot with a chosen hover offset, runs the policy, logs close
  height, logs a person's word on bumps. It goes in `ARM_SCRIPTS`, with a `--sim` test and a README section.
- A saved camera pose, the way the discard pose is saved.
- **Progress, 2026-11-05** (`docs/retrospectives/2026-11-05_arm_collects_base_camera_photos.md`): test 1b ran on hand-held photos
  (81-94% on 4 tiles, optimistic). The arm picked face-down tiles with v8 at S5 and S7, and held them at four taught poses; only
  7C is photographed by the arm so far. v8 fails at S8 whatever the tile (a lone tile closes about 25mm up). Faces in the base
  camera's photos are blurred as they are saved.
- `scripts/holdup_arm.py`: the arm collects the base-camera photos (tests 1b and 2) from face-down tiles laid on layout spots, and
  doubles as a first, small version of tests 4 and 7b's script (every pick logs `grasp_clearance_m`; every tile is set down and
  picked up again without a scan). `--empty` first visits the saved poses with nothing in the jaws. Built offline only: nothing here
  has run on the arm.
- For test 8, `play.py` will stop on face-down tiles in two places: a turn waits for a scan to see a tile on the zone, and
  the table check would take a face-down tile as missing. Test 3 may settle the first.

## Results to report

- **Camera:** both named correctly across a reboot.
- **Reading:** the exam rate as a rough range, each tile's errors, every tile read wrong named.
- **Flat:** clean out of about 10, and close heights against face-up in the same slots.
- **Stacked:** clean out of about 10 for each case, and close heights.
- **Re-pick:** the miss at each cycle.
- **Full run:** it completes, and a list of where it broke.

## Offline vs arm

Scoring the exam is offline once images exist; test 1's capture needs the base camera but no arm. The camera fix, the
pose, the picks and the full run need the hardware.

## Out of scope for v1

New demos and a new policy (unless the thresholds call for them), depth sensing, a detector for tile backs (unless test 7b
calls for one), other surfaces and lighting, opponents.

## Parked

- **Re-aiming the RealSense.** If its main job is no longer naming the hand, it could show more of the table: opponents'
  open hands, discards. Ethan's idea, for later. Moving it means recalibrating, which throws out the hover and return
  corrections, and v1 still needs it for face-up tiles.
- **Moving to the Pi.** A base camera still over the network is easy. Moving the wrist camera would change the live
  control loop v7 was trained on, so it needs its own test.
- **What ends a person checking every read.** The second read's gate is about 1000 reads with no wrong agreement; this would
  need something like it. Until then a writeup says a person checked each read, and the hand is hidden only from people who aren't checking.
