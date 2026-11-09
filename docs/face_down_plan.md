# Drawing from the wall: the face-down plan

Written 2026-11-05 from Ethan's `docs/retrospectives/face-down-pivot.md` and a review of it against the code and the
cameras. Progress so far: `docs/retrospectives/2026-11-09_face_down_progress.md`, and "What gets built" below. Numbers marked
"estimate" come from the arm's geometry or one photo, not from a run. Results go in ROADMAP.md's status, as usual.

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
- The demo is in another room, and the white paper behind the poses stays here (2026-11-09). The reader is trained on this setup and
  has to work there.

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

**The owner chose the left (2026-11-09):** space is being set aside left of the layout. Checked with the IK for a hover at both heights (no collisions, tiles
assumed about 40mm end to end): at y = +0.192, stacks centred at x = 0.20, 0.16 and 0.12 have wrist roll -141°, -149° and -158°, and joint margins at the top
tile's height of 22°, 31° and 35°. Below x = 0.10 the roll is past -164°, too close to the limit; a column further out at y = 0.21 is shorter in reach (x = 0.20 has a
10° margin at the top height). Start at x = 0.20: a lone tile in the near row closed high and missed (S8), and the far row (x = 0.192) held all 90 picks on 2026-11-09. The tiles must lie square to the mat,
because a tile turned clockwise more than about 20° here can't be reached by the roll (CLAUDE.md). v8 hasn't picked at y = +0.192 yet.

That's about 10 tiles. The last hands took 19-34 turns, so a demo win needs a pre-arranged wall and hand, and the writeup says so.

**The figures above are at the RealSense's coordinates, and the arm doesn't go there (found 2026-11-09).** The hover correction moves it about 35-44mm further out, so reach has to be worked out at
the corrected point (`layout.reach_band` does; the check agrees with `even-hand14`'s recorded 19mm at S1 and 21mm at S13). Corrected, x = 0.20 at y = +0.195 is past the gamepad's reach and needs the jaws tilted 5° (7° for a top
tile); stacks at x = 0.213 and beyond are refused by the IK. So a column here can't be drawn along its length, which is what the station below avoids.

## The wall: one station

Decided 2026-11-09. The arm draws from **one saved spot, a station**, and a person pushes the wall along so the next stack sits on it, adding stacks at the far end. Reach stops limiting the wall's length, one position needs
checking instead of ten, and only the stack on the station has to be reachable. Top tile first, then the bottom, then a feed. A person feeding the wall is part of the demo and the writeup says so: feeds are logged as feeds, not rescues.

Two candidates (`wall-stations` in `data/layouts.json`), both with the jaws straight down at both heights. Figures are computed at the corrected point, not run on the arm:

| | F, in front of the hand (0.235, -0.012) | L, on its left (0.14, 0.192) |
|---|---|---|
| Wrist roll (limit -170°) | -94° | -146° |
| Joint margin, flat / top tile | 27° / 21° | 28° / 22° |
| Gamepad reach left | 16mm | 28mm |
| Space | 8mm past the far-row tile at S7 | 20mm beside S1/S2 |

F is where a real wall sits and has far more wrist roll to spare; L has more reach. Test 3 below chooses. If pushes turn out not to repeat, the fallback is three one-stack stations refilled by hand, and if top tiles fail, a wall one layer high.
The positions come from the RealSense's xy, so lay a tile face up to green with `layout.py --name wall-stations`, tape around it, then flip it.

The order, each step able to stop the next (`scripts/wall_test.py`, and README). Thresholds are the ones below:

1. `--hover-check`: does the hover centre on a face-up tile on each station? The correction was measured only inside the hand's area, and L is past it.
2. `--push-check`, no arm: does a pushed row land in the same place? The limit is 3mm from where pushes usually land, half the jaws' tolerance of about 6mm (a judgement, not a measurement). Over it: try a low end stop, which bends the no-tray decision, so tape comes first.
3. `--stations F,L --cases control,flat --trials 5`: which station, by close height and clean picks.
4. `--cases strip`, 10 tries: tiles touching end to end, the plan's test 4b at the station.
5. `--cases top`, 10 tries, the hover a tile higher, the next stack behind (test 7).
6. `--cases bottom`, 10 tries, the next stack behind (test 7).
7. `--cases cycle --trials 12`: the real operation, fed by hand. At 90% per draw, 12 clean in a row happens about 28% of the time, so expect to step in.

Tiles drawn go into the discard container, not back on the wall: putting a tile back on a wall is a different skill and isn't tested here.

### First results, 2026-11-09

v8, bright daylight, lamp and ceiling lights off, on `sessions/20261109_151530_wall_hover`, `20261109_152013_wall_push`, `20261109_154119_wall`, `20261109_160213_wall`, `20261109_161634_wall` and `20261109_162040_wall`.
"Clean" is the owner's word per attempt. Close heights are above the tile's own grip level.

| Station, setup | Clean | Median close |
|---|---|---|
| F, white tile | 5/5 | 15.0mm |
| F, green flat | 5/5 | 17.5mm |
| F, two touching end to end | 10/10 | 14.3mm |
| F, bottom tile, stack beside it | 10/10 | 18.8mm (highest 20.0) |
| F, top tile | **0/3** (stopped at 3) | 10.6mm |
| L, white tile | 5/5 | 4.4mm |
| L, green flat | 4/5 (1 pushed) | 10.9mm |

- **Top tiles took the second column's tile, all 3 times.** The owner saw it, and the wrist video agrees. They were logged as "moved the next stack", since no key for it existed then.
- **It's an aim forward, not height.** At the close the jaw tips were about 9mm further out than on touching flat tiles (top 286.7mm, bottom 282.7mm, flat 277.8mm along x, by FK from measured joints). A two-high stack behind the target adds about 5mm, on bottom and top alike
  (tips move 9.4-9.7mm from the hover, against 4.8mm beside flat tiles); the raised hover starts about 4mm further out as well, probably because the arm sags less there. The commanded forward move varied only 14.3-15.6mm over the 3 top tries.
  Judged by eye from the wrist frames, the place v8 expects its target at the hover falls on or next to the second column's tile. Three tries is a small sample.
- **The raised hover works for height:** the top tries closed at 10.6mm above their own level, lower than any other F case.
- **The bottom's 10/10 can't be checked from the photos,** because tiles had often been re-laid before the after-photo (a hand is in some frames). The script now says to leave the mat alone until the outcome keys.
- **F and L aren't settled.** F has more clean picks (30 of 30 without top tiles, L 9 of 10); L closes 7-10mm lower, and hasn't been tried on touching tiles or stacks. Computed heights read higher with reach, which may flatter L a little.
- **The push check failed:** the worst push was 7.2mm from where they usually land (limit 3mm), and the row crept about 8mm sideways over 10 pushes, averaging 14.6mm off the station. The check's "detector wander" read exactly 0.0mm, which may mean only one baseline scan read the tile. So `cycle` waits for a straight guide along the wall.

### The hover shift, 2026-11-09

`sessions/20261109_165958_wall` (top tiles alone, 10mm nearer the robot, dim daylight) and `20261109_173158_wall` (a cycle, the same shift on every draw, bright daylight).

| Tile at F | Regular hover | Hover 10mm nearer the robot |
|---|---|---|
| Top | 0/3 | **8/8** (5 alone, 3 in a cycle) |
| Bottom | **10/10** | 0/2 |

- **The shift cancels the top tile's error.** The jaw tips closed at 277.5mm along x when the top tiles were alone, against 277.8mm for clean flat picks and 286.7mm unshifted.
- **It breaks the bottom tile.** Starting further back, the jaws came down 14mm instead of 24mm and closed 29mm up instead of 19mm: early, shallow closes, as the owner saw.
- **So top tiles get the shift and bottom tiles don't** (`--top-back-mm 10`). The two together haven't been run in a cycle.
- **L went 0/5 on top tiles with the shift** and has never had a try without it, so that doesn't rule it out. F is the station for now.
- **The wall turned 90°** (`20261109_173835_wall`, 2 tries, both failed, logged as an ordinary cycle). The tiles' long sides ran across the jaws, with the next tile touching along that side, so the jaws would have to span 35mm instead of 26mm at the short ends, with neighbours pressed against both faces.
  Two tries prove little, but it probably needs new demos and a wider hover opening. Parked until the v1 demo works, which uses the end-to-end wall.

### The cycle with top tiles shifted, 2026-11-09

`sessions/20261109_205645_wall` (11 draws; no daylight, lamp and ceiling lights), then `20261109_211132_setup`, then `20261109_211759_wall` (2) and `20261109_212052_wall` (4; no daylight, ceiling light only).

| | Clean | Top | Bottom |
|---|---|---|---|
| First run | 3 of 11 | 2 of 6 | 1 of 5 |
| After the setup check | 4 of 6 | 2 of 3 | 2 of 3 |
| **All** | **7 of 17** | 4 of 9 | 3 of 8 |

- **By the plan's thresholds that calls for a change:** 41% clean with both settings together, against 8/8 and 10/10 tried apart. The separate results were in daylight and these runs were after dark, and light wasn't varied on purpose, so what's due to that can't be told apart.
- **The setup check recalibrated nothing.** It only reads: it read WARN (table tilt, and no tile found in the wrist view at F6), and `T_cam_to_base`, the hover model and the reference weren't changed. The lamp was off for the better runs and on for the worst, but six draws can't say why they went better.
- **Closes are still high.** Above each tile's own level: top 12.6-13.4mm and bottom 14.7-17.2mm in the later runs, up to 25.6mm in the first.
- **A held tile can be useless to the reader.** In the last two picks (`20261109_212052_wall` tries 3 and 4), the tile swung in the jaws and reached the base camera with its face turned away. A high grip is the owner's reading; a grip off-centre along the tile would look the same. The outcome keys don't record this.

**Decided 2026-11-09: DAgger on v8 for wall picks**, because flat, touching and single-layer picks work and what fails is closing high and off-centre beside a stack. `record.py --wall-station F` records them, from the same hovers `wall_test.py` uses. A person's corrections are tied to this station.
After training, the new policy has to match v8 on hand picks before it replaces it (a paired `bench.py`), and pass this cycle.

**Alongside it:** the crude full run (test 8) with a wall one layer high at F, where flat and touching tiles went 15 of 15, to find what else breaks without waiting for training.

## The tests

Times are guesses. Each one can stop the plan before the next is paid for. Ten tries gives a signal for a decision, not a
reliability claim: 10/10 is still consistent with a true rate near 70%.

| # | Test | Cost | Stop or change course if | Needs |
|---|---|---|---|---|
| 0 | Fix the camera names (below), then a photo from each with the arm parked | 30 min | A camera is dead or can't see in front of it | sudo |
| 1 | **Hold tiles up by hand** wherever the arm can reach (close and at an angle, so no fixed distances), right way up and 180°, by day and evening, including 5C, 3C, 2C, 9C; score the classifier and the detector | 20 min | Nothing reads anywhere | capture script; no arm |
| 1b | **Can the view be learned?** Train a fresh small classifier on some holds, score it on others it never saw (`holdup_check.py`, 3 seeds), cropped to the tile, with a control that greys the tile out | 5 min | The control beats the best constant answer by 15 points: the background names the tile, recapture before reading anything. Under 40% right: look at the photos before capturing more. 70% or more on 3+ tiles and well above the best constant answer: capture at the real pose. In between: more holds, rerun | test 1's photos |
| 2 | **Teach the camera pose by gamepad** (`teleop_view.py`: both cameras in one window, `s` saves the pose). Put 5 tiles in the jaws; check swing, hidden faces, whether they set down flat | 30 min | Tiles swing or drop | saved pose |
| 3 | Run the detector at low confidence on a face-down tile. **Done 2026-11-09: no boxes** (below) | 5 min | If it boxes backs, targeting and the zone check may come free | one frame |
| 4 | **Face-down vs face-up on the same slots**, alternating, 10 each, single tiles, logging close height. Also how much green stands out from the mat | 1 hr | The thresholds below | test script |
| 4b | 3 flat tiles touching end to end, 5 tries | 15 min | Fails where 4 passed: the strip is the problem, not height | test script |
| 5 | A flat tile with the hover 20mm too high, 5 tries | 10 min | Closes about 20mm in the air: v7 is blind | test script |
| 6 | **Pick from a short wall by gamepad**, top and bottom, at a familiar mid-mat spot. **Tried 2026-11-09: the owner's view is "a bit challenging, but doable"; no session logged** | 15 min | If a person can't, v7 won't. Also shows whether the jaws grab two tiles, and which end to draw from | gamepad |
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

Done: these lines are in `/etc/udev/rules.d/99-mahjong-cameras.rules` (since 2026-11-05), and after the NUC restarted on 2026-11-09
`/dev/wristcam` was `video8` and `/dev/basecam` `video10`. The names hold as long as each cable stays in its port.

```
SUBSYSTEM=="video4linux", KERNEL=="video*", ATTR{index}=="0", KERNELS=="3-8", ATTRS{idVendor}=="05a3", SYMLINK+="wristcam"
SUBSYSTEM=="video4linux", KERNEL=="video*", ATTR{index}=="0", KERNELS=="3-9", ATTRS{idVendor}=="05a3", SYMLINK+="basecam"
```

Those port numbers are the NUC's, and change on the Pi.

Still to do: a check in `setup_check.py` that the jaws are in the wrist view and the base camera is present. Whether a second camera of this model makes the wrist camera's dropouts worse is an open
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
- **Progress, 2026-11-06:** the arm photographed 1B, EW and 7C (`sessions/20261106_105954_holdup`, 18 holds). The installed
  readers scored 0% (they answer one name to everything: this view is new to them, as expected). Test 1b scored 100%, but
  the same classifier trained with the tile blacked out scored 99%: each tile's holds were taken in a row from one spot, so the
  light, the empty spot and whoever was in view named the tile. `holdup_check.py` now crops to a tile box per pose and runs
  that control every time (BACKGROUND verdict). `holdup_arm.py` now lays a tile on each spot, picks them in turn, and moves each
  tile one spot along between arrangements. One of the 216 photos had the face blur over the tile; the rest that look soft are
  the fixed-focus camera up close. The 2026-11-05 base-camera photos (hand-held and arm, 266 files) were deleted at the owner's
  request (early face-blur attempts, a person in view); their event logs and wrist files stay.
- **Second arm session, 2026-11-06 12:52** (`sessions/20261106_125259_holdup`, 18 more holds of 1B, 7C and EW). It ran on
  `275c05b`, before the tiles moved between spots, so it was still collected a tile at a time: 1B at S5 then S7, 7C at S9 then
  S5, EW at S7 then S9. Each way up also came from a single spot (1B turned only at S5, upright only at S7), so the way up is
  tied to the spot too. Trained on both sessions and scored on this one, the crop check scored 96% (93-100% over 3 seeds; 1B
  read as EW in the misses), but with the tile greyed out it still scored 62% against 44% for the best constant answer: 18
  points above it, over the 15-point line, so the verdict is still BACKGROUND. The blackout control fell from 99% to 62%, so
  the crop helps, but not enough on its own. v8 picked 18 of 18 face down, leaving out 4 tries that stopped on the wrist
  camera. The camera came back each time by waiting (1.5-2.9s), with no power cycle: the first sign on the arm that the wait
  works. Close heights: 7C and EW 7-8mm above the grasp point, 1B 14-17mm at S5 and 19.5-22mm at S7, near the 20mm where
  misses start. The wrist frames show white face-up tiles beside every 7C and EW pick and none near 1B (corrected 2026-11-09), which fits
  1B closing high.
- **The first clean pass, 2026-11-06 evening** (`sessions/20261106_195636_holdup`, 72 holds of 5C, 3C, 2C, 9C, 1B and 7C, 6
  per tile and way up; lamp and ceiling lights, no daylight). Collected the new way: three tiles on S5, S7 and S9, picked in turn,
  each moved one spot along and laid the other way after 2 rounds, in two groups (5C/3C/2C, then 9C/1B/7C). `holdup_check.py
  --by hold`: **100% on held-out holds, every tile and both ways up, all 3 seeds; the control with the tile greyed out 0%**
  (best constant answer 17%). The 0% is the design working: the held-out holds are each tile at the one spot it never had in
  training (1B trained at S5 and S7, tested at S9), so a classifier that learned the scene names the tile that was there and
  is always wrong. The one with the tile visible was right on the same photos, so it read the faces. Still optimistic: one
  session, one room, and each group of three was collected in its own stretch of time, so telling a group-one tile from a
  group-two tile could partly be the time. No two tiles in the set look alike (7C vs 8C isn't in it). The installed readers
  scored 0-3% on these photos (`holdup_scores.json`), as expected.
  **Another light** (`holdup_check.py --test-session`, added for this): trained only on the evening session, tested on the two
  daytime sessions' 1B and 7C (EW left out, never trained on). 12:52, bright daylight: 96% (1B 94%, 6 read as 5C; 7C 100%),
  control 31% against a constant 57%. 10:59, bright daylight: 99%, control 60% against a constant 60%. Daylight against lamp
  light in the same room doesn't throw it, on 2 tiles, which the script calls too few to judge.
  The picks: 72 of 72 completed held the tile (1 more stopped on the wrist camera, back after waiting), every close 5-10mm
  above the grasp point, 1B included (14-22mm at 12:52).
- **A very different background, 2026-11-09** (`sessions/20261109_100829_holdup`, 72 holds of the same 6 tiles, tiles grouped differently from
  the evening; bright daylight, lamp and ceiling lights off, white paper behind the poses, the mat uncovered). Trained on the earlier arm
  sessions and held out: **97.0% (95.8-97.8% over 3 seeds)**, every tile 87.5% or better, 9C the weakest (91-94%, read as 7C). Control 15.5%
  against 16.7%, though on a blank background that's close to automatic. So the reader doesn't need its training room, on these 6 tiles.
  v8 held 72 of 72, but its closes rose to a median 13.7mm above the grasp point (highest 18.3) from 7.6mm (highest 9.5).
- **The reverse check, 2026-11-09** (trained only on that white session, scored on the cluttered evening): **97.5% (94.4-99.3%)**, control 16.2% against 16.7%.
  The weakest tile and the direction of the mistakes both change from one direction to the other, so they look like variation between training runs.
- **The lights check, 2026-11-09** (`sessions/20261109_122910_holdup`, 18 holds, lamp and ceiling lights on, dim daylight, otherwise as that morning): v8 held 18 of 18,
  median close 13.0mm against 14.4mm for the same three tiles with the lights off, though the wrist view was twice as bright. So light isn't the main cause. Across the
  sessions, v8 closes at about 8mm with white face-up tiles near the target and 13-18mm without them (a pattern, not a deliberate test; the table is in
  `docs/retrospectives/2026-11-09_face_down_progress.md`). The wall will be mostly green, so expect 13mm or more. The reader scored 99.5% on these 3 tiles.
- **Test 3, 2026-11-09:** six face-down tiles on the mat, five RealSense frames, the installed detector down to 1% confidence: no boxes.
  The same weights gave 34 boxes on a saved face-up scan. So a face-down tile is targeted by saved position, and a face-down landing needs
  another check.
- **Next** (the reasons: `docs/retrospectives/2026-11-09_face_down_progress.md`): the order in "The wall: one station", with `scripts/wall_test.py`; a held-out session in the demo room, adding backgrounds only if it fails. How much more capture depends on which tiles the demo's wall uses.
- `scripts/holdup_arm.py`: the arm collects the base-camera photos (tests 1b and 2) from face-down tiles laid on layout spots, and
  doubles as a first, small version of tests 4 and 7b's script (every pick logs `grasp_clearance_m`; every tile is set down and
  picked up again without a scan). `--empty` first visits the saved poses with nothing in the jaws.
- For test 8, `play.py` will stop on face-down tiles in two places: a turn waits for a scan to see a tile on the zone, and
  the table check would take a face-down tile as missing. Test 3 means neither comes free: both need another way to know.

## Results to report

- **Camera:** both named correctly across a reboot (2026-11-09).
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
calls for one), other surfaces and lighting apart from the demo room's, opponents.

## Parked

- **Re-aiming the RealSense.** If its main job is no longer naming the hand, it could show more of the table: opponents'
  open hands, discards. Ethan's idea, for later. Moving it means recalibrating, which throws out the hover and return
  corrections, and v1 still needs it for face-up tiles.
- **Moving to the Pi.** A base camera still over the network is easy. Moving the wrist camera would change the live
  control loop v7 was trained on, so it needs its own test.
- **What ends a person checking every read.** The second read's gate is about 1000 reads with no wrong agreement; this would
  need something like it. Until then a writeup says a person checked each read, and the hand is hidden only from people who aren't checking.
