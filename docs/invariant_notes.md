# Invariant notes: the full evidence

The invariants in `.claude/rules/` load with the code they cover, so the longest keep the rule, the reason and the
names to look for, and their full measurements live here (moved verbatim on 2026-10-29). Add to these when a
rule's evidence grows; keep the rule's line to what someone changing the code needs.

## Returns aim at the tile's home, never where the jaws closed

Returning to each grasp point carried every return's error into the next pickup: tiles walked 77mm (4D, 9
rounds) and 100mm (6C, 10 rounds) toward the edge of the reach on 2026-10-27. Over 95 bench returns a pick +
return moved a tile 5.7mm (median), leaning +2.8mm in y; the jaws pull a tile only partway to their centre
(55-65% of its offset in the jaws stays with it), so where it lands depends on that grasp, which the return
model (`mahjong/return_correction.py`) takes as input (landing error 10.5mm → 6.8mm RMS,
leave-one-position-out). On the arm (`20261028_102732_bench`, 15 rounds at 6C, v3c) returns landed 3mm from
home (median; worst 4mm) with no build-up, vs 11mm (worst 17mm) at 6C the day before; other spots unchecked.
Homes last the session, by spot or tile name: a tile moved on purpose less than `HOME_RADIUS_M` (40mm) is put
back at its old home. The fit pools every policy; one that grips differently may need its own
(`docs/retrospectives/2026-10-28_return_homes.md`). record.py kept returning to the grasp point until
2026-10-28 (tiles walked 25-31mm in one recording, `20261028_225113_record`); it now shares the homes and
landing check (`return_correction.Homes`, `check_landing`), and its teleop returns are logged as `policy`
"teleop", `source` "record".

## A tile carried across the mat is aimed from its destination

Full account: `docs/retrospectives/2026-11-04_first_play_return.md`. All 5 tiles placed in `20261104_183142_play` aimed 12-22mm
too far in +y against bench returns at the same spots; the 4 measured landed 15-18mm off in y. With the destination's aim and
the stray's share, those landings are predicted at 6.1mm RMS (from 16.9mm), not yet measured on the arm. The "55-65% stays"
above, and a later "x 0.69, y 0.85", mixed a close's stray with the mat-wide gap; with the position trend removed it's 70% in
x, 52% in y (738 returns).

## A return to the same place needs position, and a landing past 30mm is no sample (2026-11-17)

On even-hand14, returns at S1/S2 landed left of home (+y, toward the wall). Mean landed - home y over S1-S6: face up
-1.8 to +2.5mm (11-10..11-13), face down +2.6mm (`20261116_213816_bench`), face down +7.9mm (`20261117_093921_bench`, S2
+11.7 to +15.2mm). S7-S12 stayed about +1 to +2mm. Not face-down as such: face-up S2 landed +5.7 to +8.6mm on 11-12.

- **What changed was the policy's move after the hover.** Hover - detection at S2 was about +1mm y every session; grasp -
  hover at S2 went from +10mm (11-12, 11-16) to +18.6mm (11-17, v11 and v13), and at S1 from +5 to +15mm.
- **The `grip` model aimed with that offset as if the tile kept it.** Its y coefficient said the tile keeps ~88% of the jaws'
  y offset, but most of the spread in grasp - detection is the gap across the mat (+9mm left, +24mm right), not a stray; a real
  stray keeps ~50% (the carried-tile section above found the same mix-up). On the left, its y residual grew with grasp -
  detection: slope +0.66 face down, +0.36 face up.
- **`grip+position` was held back by neighbours taken for landings.** 22 fitted records had landed 44-50mm from home (9 face
  down, 11-16/11-17), the spot pitch; nothing real had landed past 24mm (1061 under 25mm). With `LANDED_WITHIN_M` at 50mm,
  a return the scan didn't see took the next spot's tile: face up through the misread fallback (any name; e.g. 6C and 5C
  "landed" on 1B, `20261030_155453_bench`), face down by the shared name "back". Without them, leave-one-position-out: grip 5.35mm,
  grip+position 4.72mm (12%, past `MIN_GAIN`). Cutting the search radius to 30mm instead would have left a tile landing
  30-50mm off unreset, so the search stays at 50mm and only the fit leaves them out (`SAMPLE_WITHIN_M`,
  `far_landing`): they stay in the file. Still open: an unseen tile's reset prompt and `session_report`'s landing stats act on
  the neighbour; fixing that needs the matcher to know the other spots' homes (bench.py, record.py).
- **Fitted without each session, landings at S1-S6** (predicted against measured; not a re-run): 20261117 grip +8.1mm y, RMS
  9.2mm, against grip+position +3.1mm, 4.6mm; 20261116 +2.3mm, 4.3mm against -0.2mm, 3.8mm; 20261112 +2.4mm, 4.7mm against
  +1.5mm, 3.8mm. For 11-17's grasps the aim moves 7.4mm -y at S2, 5.6mm at S1, 4.2mm at S3, and under 3mm elsewhere.

The policy still closes 15-19mm +y of the hover at S1/S2, which brings the jaws nearer the wall at the pick; that's for its data.

## Hand corrections record the commanded tool point, not where the jaws were — and that's fine

In the 23 gamepad corrections of 2026-10-22 the measured tool point at → sat ~11mm short in reach (median; up
to 17mm far out), 2-18mm low, and up to ±12mm toward the centre line (−7mm for +y tiles, +6mm for −y: the pan
stopping short of its command). A settled hover sags the same way (41 bench hovers: x −11 to −12, y −7 left /
+8 right), so the commanded fit already carries it. Refit offline on the measured point (2026-10-28):
leave-one-position-out 9.0mm vs 10.6mm, and the hover would move only 0-4mm once its own sag is compensated —
not worth an x/y settle loop. The noise is the lining up: the same tile aligned twice differs by ~9mm (RMS, 8
pairs), commanded or measured. The arm also drops and drifts after the gamepad takes over: collect_corrections
traces every teleop step (`mahjong/handoff_trace.py`, `scripts/handoff_report.py`) to tell sinking servos from
a moving command or XLeRobot's IK easing on the first push. The first trace (`20261028_195346_corrections`,
5D) was none of them: the arm held still for 5s, then a thumb resting on the left stick (0.22-0.44 of its
travel, past XLeRobot's 0.15 deadzone) pulled it 87mm toward the base over 8s, the servos lagging up to 12mm
low while it moved. Every gamepad handoff now uses a 0.35 deadzone (`gamepad.DEADZONE`; collect_corrections,
record incl. DAgger takeovers, bench's teach-discard) and warns of a stick off centre
(`gamepad.warn_off_centre`); collect_corrections and record show the stick readings in the wrist window while
you drive (`gamepad.draw_input`).

## A tile the detector missed and then finds is no knock (2026-11-27)

`20261120_181225_bench` (face down, v14 against v13) read v13 25/28 clean. Two of its three unclean attempts were knocks only
the scans called; the frames show neither tile moving during the attempt blamed (no shift measured by template match, under
about 2mm), and the owner judged both false.

- **012** (the tile at S6): missed at scan 34, where a faint box showed it still there, and again at 35. `MatWatch` had
  remembered it for one scan, so at 36, 0.2mm from where it was, it "appeared".
- **053** (the tile at S13): the return before, 052, set it about 11mm off home. Scans 160 and 161 missed it, and its first
  sighting, at 162, was counted against 053.

The fix is gated on `interventions.SCAN_MEMORY`, logged by scans from 2026-11-27. A first version without the gate, and
applied between attempts too, was replayed over the 211 sessions that had scans: knocks went from 272 to 261, and answers
people had given ("put tiles back") disappeared. With the gate, every older session derives as before. The two verdicts are
`scan_verdict` events in the session. Not yet run on the arm. Still counted as before: a set-down tile first seen between
attempts is a `tile_reset`.

## The wrist camera can drop out mid-session

2026-10-28: from the 3rd of 10 v3c trials every read failed, and the policy ran blind to the 16s cap, so 8
misses were scored that said nothing about v3c. No frames for `camera.LOST_S` (1s) means it's lost, in every
script: `policy.run` stops (`stop_reason` "camera") and the bench records `attempt_void` and runs it again;
record.py discards the episode ("wrist camera lost"); wrist_capture.py and setup_check take the view again;
collect_corrections reopens it and restarts the gamepad from the held pose. Each reopens it with
`hardware.reopen_wrist_camera` (retry / quit if it won't open): use that in any new script.

2026-11-05, why it drops (kernel log of every boot since 2026-10-08, every session's events and logs):

- **It's the USB link, not the driver.** On port 3-8 the log has `error -71`, `disabled by hub (EMI?)`, `Cannot enable. Maybe
  the USB cable is bad?` and fallbacks to full speed. 22 of 24 voided attempts had a USB disconnect in the 30s before them.
  Nothing else on the controller shows this: no `-71` on the arm's board, the LifeCam on the next port or the RealSense, and
  no over-current on 3-8.
- **It started at once**: 0 disconnects in 446 picks and episodes, 10-23 to 10-27, then the first at 08:41 on 10-28, on the
  same kernel. The cause of that step is unknown. Candidates: cable strain from the wrist-roll incident of 10-27 (a guess from
  the date). Ruled out: the LifeCam (first seen 3 hours after the first drop), RealSense resets (1 of 96 within 10s of one).
- **Two kinds of drop**: 82 plain disconnects (median 0.42s back, 64 under 1s), and 14 `EMI?` port disables (25s to hours).
  **The slow times can't show whether waiting helps**: they include how fast a person replugged (the owner's point). Only
  sub-second returns are clearly unaided. That's why each recovery now logs its outcome, and the wait is short.
- **Drops lean to the right** (6.5% of attempts, against 1.3% centre, 2.4% left; 25 events, on the right in 6 of the 10 sessions with
  voids, last known position before each). A lead for cable tension, not proof. A 10-minute stream with no arm script running had 0 drops, but
  the long-run rate predicts only 0.7 there, so that's weak evidence.
- **Online fixes that don't apply**: `modprobe -r uvcvideo` acts above the failure, and the RealSense shares the driver.
  `nodrop=1` is already the value on this kernel. `quirks=0x80` is `UVC_QUIRK_FIX_BANDWIDTH`, not a memory setting, and the logs
  show no bandwidth problem. `usbreset` needs a device still on the bus; no log had one connected but silent **until
  2026-11-13, which did** (below), so that is the one case it fits. Resetting the controller would take the arm and RealSense
  with it.

2026-11-13, `20261113_110111_holdup` (base camera collect; kernel log, session events, pipewire/bluez journal):

- **The wait earns its keep, the power cycle does not: 0 for 9.** Over 94 logged losses, 85 came back by themselves (median
  1.55s, range 1.55-5.23s) and 2 were quit at the prompt. The power cycle ran 9 times — every one still ended with a person
  (7, median 66s) or a quit (2). It only runs after the 10s wait has already failed, i.e. on the hard cases, and on this
  evidence those need hands. So it buys nothing and delays the prompt by `AFTER_CYCLE_WAIT_S` (10s). This answers the open
  question in `mahjong/wrist_usb.py`'s docstring. Whether it does harm is not shown: on 2026-11-13 the port was already
  failing to enumerate before our cycle ran (`unable to enumerate` at 12:14:39, cycle at ~12:14:55), so the 2.5 minutes of
  dead port after it can't be pinned on it.
- **Connected but silent.** The port flapped from 11:51, hard from 12:13:12 (`error -71`, `Device not responding to setup
  address`, ~10 cycles), then `disabled by hub (EMI?)` at 12:14:34 and `Cannot enable. Maybe the USB cable is bad?` at
  12:17:28. A replug at 12:17:29 enumerated it as device 123 — and then every control transfer timed out with `-110` for just
  under 3 minutes (~35 queries at 5s each). `uvcvideo` logged "UVC non compliance: permanently disabling control" for
  Saturation, auto White Balance, Gamma, Gain and Power Line Frequency; `v4l2-ctl --list-ctrls` still showed them missing
  afterwards, with `exposure_time_absolute` and `white_balance_temperature` `inactive`, and a 30s open-and-read hung and had
  to be killed. The script was at the person prompt throughout and the terminal looked hung because each query blocks in the
  kernel; the Ctrl-C landed at 12:20:25 and it exited cleanly with its 50 holds, parking the arm.
- **78 re-enumerations of 3-8 in one boot** (08:33-12:51, ~19/h), accelerating: 21 in the ten minutes from 11:50, 11 more
  from 12:10. The same link had produced the torn frames of `20261113_100710_bench` earlier the same morning. Two failure
  modes from one cause, a few hours apart.
- **It's a one-way regime change, not damage per replug.** Grouping every kernel line by enumeration: 61 of the 78 came up
  clean (high speed, UVC found, no errors), and before 12:17 the failures were all `-71` or a UVC device never found — the
  link dropped, but when it came up it worked. From 12:17:29 on, all three successful enumerations came up deaf (`-110`).
  Their counts of permanently disabled controls were 5, then 1, then 7 — **not monotonic**, so a replug costing more
  controls each time is not supported; which controls fail depends on where the timeouts land. The `-110` totals (45, 54,
  142) mostly track how long each instance sat being probed, including by pipewire and by checks run while debugging, so
  they don't measure damage either. What the data supports is that replugging stopped fixing it after 12:17, not that it
  made it worse.
- **A fallback to low or full speed is a retry, not a separate fault.** At 12:50 the ladder was high (failed) → low
  (failed, `-71`) → full (failed) → high (up, deaf): the kernel stepping down speeds after a failed enumeration. It is a
  symptom of a bad link, as the 2026-11-05 notes already had it, but it doesn't localise the fault to a conductor.
- **Camera board or cable was open on the log alone.** Every symptom here belongs to the whole path (host port, cable, the
  camera's own USB interface), so the log couldn't separate them. What leaned cable/connector: drops track arm position (6.5%
  right vs 1.3% centre), the step change of 2026-10-28 followed a mechanical event with 446 clean picks before it, and the
  slack change of 2026-11-10 moved the rate (0 in 13 runs, though 1 in 7 by chance). The host was ruled out — no `-71` on the
  arm's board, the LifeCam or the RealSense. The port swap below then answered it, and the answer was neither.

### 2026-11-13, the port swap: the fault is the NUC's receptacle 3-8

The owner moved the wrist camera to a different USB socket on the NUC, with the same camera and the same cable. That is a
one-variable test the log couldn't do, and it is decisive.

- **On 3-8 it enumerates and then can't stream.** Two attempts after the deaf regime had lifted — the camera came up clean
  both times, no `-71`, no `EMI?`, high speed, `uvcvideo ... Found UVC 1.00 device`:

  | | 14:31:59 on 3-8 | 14:35:42 on 3-8 | 14:36:34 on 3-5 |
  |---|---|---|---|
  | enumerated | clean | clean | clean |
  | control `-110` | 1 (Exposure Time disabled) | 1 (White Balance disabled) | **0** |
  | `Failed to set UVC probe control` | 3 | 1 | **0** |
  | streamed | no | no | **yes** |

  The probe control is the format negotiation `VIDIOC_STREAMON` does, so a `-110` there is exactly "Cheese opens the camera
  and then errors". It is not a driver state: `uvcvideo` disables a control per device instance and that resets on every
  re-enumeration, and both instances above were fresh.

- **On 3-5 it just works.** 267 frames in 10s at 26.7fps, 0 failed reads, worst gap 40ms (one frame period), and not one
  kernel line about the port during the run. First try, no coaxing.

- **What that rules out.** The camera board: it streamed within seconds of the move. The cable: the same cable carried that
  stream. The controller and the software stack: 3-5 and 3-8 are both ports of the same xHCI root hub (`usb3`, 16 ports,
  `0000:80:14.0`) with the same `uvcvideo`. What is left is the receptacle on 3-8 and its short path to the root hub —
  contacts sprung or dirty, or a cracked joint at the connector. A marginal contact fits the signature: enough margin for the
  short, retried descriptor reads of enumeration, not enough for correct signalling afterwards.

- **It also explains the old evidence better than the cable did.** Arm motion tugs the cable, and the tug lands at the
  connector, so a marginal receptacle gives the same position-correlated drop rate; the mechanical event of 2026-10-27 would
  have levered the plug in that socket; and adding slack reduces the tug, which is why 2026-11-10 looked promising.

- **Replugging over the day probably did make it worse, but not the way first claimed.** The mechanism is insertion-cycle
  wear on a receptacle already marginal, not controls accumulating: the per-enumeration counts of permanently disabled
  controls were 5, 1, 7 — not monotonic (above). Moot either way now: the port is retired.

- **Moving it breaks `/dev/wristcam`.** The udev rule matches `KERNELS=="3-8"`, so after the move the symlink was simply
  absent and every script stopped at `hardware.connect` with "no wrist camera at /dev/wristcam" — which reads like the camera
  is still broken. The rule and `config.WRIST_USB_PORT` both have to move with the cable.

- **Still unmeasured.** Whether another device also fails on 3-8 (that would separate a worn receptacle from a worn plug end
  that happens to seat well in 3-5), and whether 3-5 holds up over a real session with the arm moving. One disconnect at
  14:38:32 on 3-5 was the owner replugging, not a drop.
- **2026-11-25: back on 3-8, on trial.** The owner changed to a shorter cable that can't reach 3-5 without hitting the
  tripod. On 3-8 it streamed 1183 frames in 45s (26.3fps, 0 failed reads, 0 torn, no kernel lines for the port; one 0.5s
  gap), arm still. That differs from 11-13, when it couldn't stream there, but cable and insertion both changed since, and
  still-arm watches never caught a drop. Then with the arm moving: `20261125_214715_play_fd` (21:47-22:15), 46 of 46
  picked, 0 torn frames, no camera events and no kernel line for 3-8 (the old rate was ~19 disconnects an hour). Before
  it, 3-8 logged a disconnect at 21:44:38, 15s after `20261125_214243_play_fd` ended, and the camera came back at
  21:45:23 with no error lines; whether a person replugged it isn't recorded. Go back to 3-5 on any drop, `-71` / `-110`
  or torn frames.
- **Bluetooth audio shares the root hub.** `lsusb -t` puts the camera on bus 3 port 8 and `btusb` on bus 3 port 14. At
  12:19:10 wireplumber logged `Failure in Bluetooth audio transport /org/bluez/hci0/dev_*/sep1/fd0` and the bluez output node
  went `running -> error`; at 12:20:24 `pipewire-pulse` gave client `[Firefox]` `error:25 (Input/output error)`, which is
  what the owner saw as Spotify hanging. bluetoothd recovered at 12:21:43. pipewire was also probing `/dev/video0` and
  blocking on the same timeouts (`spa.v4l2: VIDIOC_QUERYCTRL: Connection timed out`), since it re-probes on every hotplug.
  Ruled out as causes of the desktop symptoms: the `i915_flip` kworker (it alternates between PIDs — normal page-flip work,
  not a stall) and Firefox's NVIDIA apparmor denials (every ~3 minutes all boot, routine).
- **Why the watch script catches nothing.** All 4 watches to date total 16 minutes and are `still` (arm idle) or `by-hand`
  (motors off), both controls, and all show 0 drops. The failures track powered arm motion, and `--kernel-only` — made for a
  second terminal during a real run — has never been used. A 10-minute kernel-only watch during a bench should expect ~3
  disconnects at the rate above.

## The wrist look finds a back by its shape, not a face (2026-11-16)

The look measured how much of a disc between the jaws is pale face. Over a green back that reads ~0, so all 36 looks at
backs in `20261116_200001_bench` and `20261116_213816_bench` said "not there", with the back between the jaws (e.g.
`20261116_213816_bench/wrist_checks/scan004_004_back.jpg`), and `look_at_home` skipped face-down spots outright (4 in the
first trial). Back colour (HSV hue 26-50, saturation 60+, brightness up to 160) alone doesn't work: on 388 face-up first frames
of the standard mat a disc of it shows up in ~1 of 100 places, the olive stripe in a dim room
(`20261103_214640_bench/005_214922_4C`, at (180, 145)). A back is a squarish block of it and a stripe a long band, so
`wrist_check.back_blob` opens the mask (25px), keeps blobs of 6-30k px with aspect up to 2.0 and fill 0.65+, within 70px of
the expected point. On the standard mat's bench frames, 2026-11-12 to 11-16: 36 of 37 looks at backs; 143 of 177 first
frames at the hover (misses mostly `20261112_202418_bench`, no daylight and no lamp, the arm's shadow across the tile; 41 of
48 in bright daylight with the ceiling light, `20261116_200001_bench`); 1 false back in 38,024 places on the face-up frames.

On the arm, 2026-11-18 (`20261118_221723_bench`, face down, standard mat): 5 looks, 1 found by shape, 4 with no back's shape in
any of 3 frames and 86-100% of the disc in the back's green, so each asked a person, who said "there" every time. Re-measured
on the saved frames: in 3 the back between the jaws is a patch of 4,775-6,245px, just under or on the 6,000px floor (the jaw
tips and their shadow cover part of it); in 1 (`scan006_007`) there's no patch near the jaws at all, the shadow taking it out
of the colour range. Over every face-down session to that day (holdup, wall, crude, play_fd and face-down benches' `_back`
attempts), the shape found the back in 859 of 1,092 first frames at the hover (79%): wall 0-60% a session, crude 0-80%,
`20261117_200555_play_fd` 55%, holdup 60-100%. Thresholds tuned on one week's light don't carry over, hence the trained back
check (`mahjong/back_check.py`). Its gap is "empty" photos: the arm hovers only where a tile is, so to 2026-11-18 there were 2
(looks a person answered "gone") against 1,162 backs.

## Moved from CLAUDE.md on 2026-11-02, verbatim

The bullets below were shortened in CLAUDE.md (now `.claude/rules/`), to keep what loads small.

- Interventions (`mahjong/interventions.py` has the classes and why, ROADMAP 3): keys pressed while the arm
  picks up or returns (read from the bench's own terminal, `mahjong/terminal_keys.py`: a global listener logged
  ~70 keystrokes typed in another tab on 2026-10-30; a `keys_void` appended to a session discounts noise found
  afterwards) say what happened as it happens (`b` bumped another tile, `p` pushed its own, `r`
  I moved a tile by hand, `d` / `u` gripped low / high (precarious, counted, the attempt stands), `l` look; `interventions.KEYS`, printed as each pickup and return starts), → marks a moment such as a hand
  on the tile (asked right after, unless a key said it). The RealSense scans around each attempt catch knocked or replaced
  tiles: a neighbour that moved or turned is a knock if `b` was pressed, else asked then; every other change
  (far tiles, gone / appeared) is settled by the next scans (`interventions.ChangeWatch`: undone = YOLO
  flicker, still there 3 scans on = real), not asked. No questions wait for the end of a batch. Whatever a
  person did by hand (`r`, the "I did, by hand" answer, a `note`, a reset the bench asked for) is theirs:
  straightened if it starts where a knock left a tile, else a reset; a watched knock undone then is a knock put
  back, not a glitch (`Bench.hands_on`, `ChangeWatch.update(by_hand=)`). Bumping the camera or the table, which
  moves every tile at once, still reads as many moves. "Touched the tile to help" is a rescue
  (the success becomes assisted); "the arm bumped a tile, no one helped" is a failure that leaves a
  success clean. The scorecard number is rescues per 10 attempts.

- **A return lets go where that tile's jaws closed, not at a fixed height.** Grasps close with the measured tip
  6-8mm above the grasp point near the base and 12-14mm at full reach (flex), v4 5mm lower than v3c, and once
  at -0.2 to 2mm (7D/4D, 2026-10-29). The fixed 10mm release let go of v4's tiles a median 3.3mm above where
  they were held (6-9mm that day): they dropped, tilted, and landed further from home (3.4mm under a 3mm drop,
  6.2mm at 6mm+; 137 returns). `Arm.place` now descends (only down, `_set_down`) to 1mm above the grasp's
  own height (`kinematics.grasp_clearance`, from the pickup's measured joints). Unconfirmed on the arm.

- **So is the surface under the tiles.** Every recording through v5 is on the black table top. On a green cutting mat
  with a white grid (2026-10-29, `20261029_180643_bench`) v5 closed its jaws 45mm up in the air once and bumped
  a neighbour the next time. `make_policy.py --augment --replace-mat 0.7` trains on colour jitter and on frames
  whose mat is swapped for other surfaces (`dataset.surface_keep` keeps faces, green backs and the jaws; faces in
  the dim recordings are found by growing from their lit part). The swap is fixed in the image, while a real mat
  moves with the camera, so record a few real episodes on each new surface as well, and score on ones never
  trained on. Each session records the surface (`config.SETUP_QUESTIONS`, earlier ones are the black top) and
  `offline_eval.py --by surface` scores per surface. Run `setup_check.py` on a new surface first: on the
  green mat (~2mm thick) the table plane read 4.1mm higher and tilted 4.4mm per 10cm in x
  (`20261029_191321_setup`), more than its thickness, so check it lies flat.

- **v2 repeats a memorised x/y move instead of steering to the tile** (measured at 4D on 2026-10-23:
  the grasp followed a deliberately offset hover at ~1 mm per mm). The demos' hover→grasp move is
  systematic per position (affine/23 misses ~12mm RMS in a pattern), and v2 learned it per position. So
  for v2: don't change the live hover (`data/hover_model.json`), and don't fit bench samples (grasp =
  hover + fixed move). Success rate hides this, since the jaws tolerate ±6mm; check a policy with
  `bench.py --hover-perturb-mm`. Compare bench and demo shifts per position, never against a mat-wide median.

- **Computed heights drift with reach: far spots are ~10-15mm lower than the FK says.** Same settled hover
  (tip 40mm above the grasp point by FK): a ruler showed 10-15mm above the tile top at far spots, 20mm+
  near (2026-10-28). The demos agree: their lowest FK tip rises from +10mm (r 0.15-0.18m) to +25mm
  (r 0.24-0.30m) while the jaws grip the tile at the same depth. The settle and the tip floor both use
  FK, so neither sees it; likely link/gear flex under load or a tilted table plane. At (0.19, +0.19) both
  v3b and v3c then went 13mm deeper than the demos there and bumped a neighbour once.

- **The far row is at the arm's full reach, not past what the servos can hold.** Held 10s at the 1D hover
  (base r 0.287m) the shoulder and elbow loaded 8% and 13% (overload trips at 80% for 2s), the tip
  steady (2026-10-28). But the wrist is 3mm from the gamepad's reach limit at 1D's grasp point, and
  past it at WW (0.291m). Pushing out there does nothing, and lowering slides the gripper back toward
  the base along the reach circle, which looks like a sag. `gamepad.warn_if_at_reach` says so after
  the hover (record, collect_corrections). Reach such tiles by tilting the jaws forward (L/R), or keep
  tiles inside base r ~0.27m. `layout.py`'s `r` shows that boundary on the mat (`layout.reach_band`): on
  hand14 1D and WW are at the edge, and both v4 and v5 missed WW with a neighbour beside it (2026-10-29).

- **The wrist look sees a tile face, not which tile.** Between the jaw tips at a policy's hover, the hovered tile's
  face fills a fixed disc of the wrist image (`wrist_check.EXPECTED_PX`; fill >= 0.73 on 103 black-top frames, 0
  without). A tile knocked to within 30mm of a lost one's spot fills it too (3D for 1B, 2026-10-30), so the bench
  doesn't look there (`wrist_check.CROWDED_M`). A pale surface fills it with no tile at all: bare white stripe reads
  0.72 (`20261103_120259_setup`), so with too little dark surface in view (`wrist_check.MIN_DARK_SHARE`: >= 0.31 on
  171 black-top frames, >= 0.29 on the green mat, <= 0.12 on the stripes, <= 0.09 on paper) it asks a person. All 11
  looks in `20261103_121112_bench` did see a tile, but under a name it didn't have: 2B as "4B" and "NW", 1B as "EW", 3C
  as "WW". Of those, "NW" and "WW" are the spots' labels in `even-hand14`, not reads; the detector misread 2B as 4B and
  1B as EW, and misses 3C at S13 in 45 of 46 scans. So a look can't be made safe by matching the lost tile's name. On a pale
  surface it compares the view with an earlier look at that place (`wrist_check.SAME_VIEW_SCORE`: 0.96-1.00 on 8 looks,
  another place at most 0.81, bare stripe at most 0.70; `wrist_check.PlaceViews`). A rule that stopped looking at a place
  after 3 confirmations was removed before it reached the arm: it could hide a real loss, and the faint boxes below
  answer all 11 looks anyway.

- **Whether a tile is there and which tile it is are separate questions** (`20261103_121112_bench`). 11 lost-tile events
  came from 3 of 14 tiles the detector does box, just under its cut. Over a scan's 5 frames, in at least 2, at 0.1: all 14
  tiles in all 46 scans. At 31 spots just emptied by a pickup (17 on the mat, 14 on the black top, `20261030_175235_bench`):
  none. On the mat at no tile: 0 in 74 scans; 2 in 42 on the black top. Offline, with the plane rather than depth for
  position. Names need retraining: 2B reads as 4B in 45 of 46 frames, so no vote fixes it.

- **A tile's white face can't be found by colour on a pale mat.** On the striped mat, face and white stripe overlap in
  brightness (face median ~225, stripe up to ~240) and match in colour (Lab `a` ~133). Of 950 colour reads at real
  tiles (`20261103_121112_bench`), 713 found nothing, 231 were stripe slices, 5 were tile-shaped but ~9mm off, 1 was
  right; with no tile there it reported slices of stripe as F2 and F6, so a hover could follow one. So the RealSense's
  find goes by the detector's box (a colour read only refines it, and gives the only yaw reading, so on this mat there
  is none), and the setup check's wrist view matches the reference's own picture of the tile
  (`setup_check.match_tile`: same tile 0.82-0.96 on 14 pairs, no tile 0.68 on one frame; it can't tell which tile,
  another reached 0.84). The cutoff comes from one session's light and hasn't run on the arm.
  It proved marginal across light: from 2026-11-11 to 11-23 every "no tile" on the standard mat (12 frames, the tile
  plainly in view) matched 0.67-0.77 within 15px of the reference's place, against a reference from 2026-11-03, while
  passes scored 0.78-0.82 at F2 and F3. So a match from `MATCH_ASK_SCORE` (0.6) up is shown to a person rather than
  looped as "no tile". The tile detector is no help in the wrist view: it boxed nothing in the reference frames or any
  of those (best 0.55, a tile cut by the image edge); it was trained on the RealSense's view from 0.87m.
  **Swapped camera cables** (2026-11-25, `20261125_201151_setup`): both cameras replugged at 20:08, the base camera's
  into the wrist's port 3-5, so the tile test hovered F2 five times over a tile the base camera couldn't see (match
  0.36). The udev rule was right; only the port tells the two apart. The base camera's frames across those hovers were
  0.94-1.00 alike (`setup_check.view_similarity`), real wrist frames 0.00-0.52 between spots (61 pairs), so a wrist
  view that stays 0.75 alike from rest to the hover stops the tile test.

- **The RealSense's depth reading drifts** (analysed 2026-11-25 from the `depth.npz` saved with each scan; the setup
  check's own `table_fit` run on every one). Over 1881 bench and play scans from 2026-11-13 to 11-23 (8+ confident
  tiles; `20261118_234045_bench` left out, its tiles read 39mm up) the mat read +3.4 / +5.5 / +10.8mm vs `config.table_z`
  (p5 / median / p95); the 11-03 reference reads ~0. Tile tops moved with it (1.0-1.3mm per mm), though a tile's thickness
  can't change. Against the colour image (ORB features, scale of a similarity fit; a real vertical move rescales it):

  | Compared | Depth moved | Image scale seen | A real move predicts | Features |
  |---|---|---|---|---|
  | `20261116_200001_bench`, start → end | +2.2mm | +0.04% | +0.26% | 1515 |
  | `20261118_105159_bench`, start → end | +3.3mm | +0.03% | +0.39% | 1862 |
  | `20261118_153155_bench`, start → end | +1.0mm | +0.01% | +0.12% | 1828 |
  | `20261123_224453_bench`, start → end | +2.4mm | +0.01% | +0.28% | 2293 |
  | `20261123_203643_bench` → `205508` | +3.0mm | +0.02% | +0.36% | 1876 |
  | `20261123_161326_bench` → `203643` | -6.6mm | -0.02% | -0.78% | 118 |
  | `20261123_085733_bench` → `203643` | -11.2mm | -0.49% | -1.32% | 63 |

  The last row doesn't fit as cleanly: few features match and the image also shifted 1.4px, so something may have moved
  between that morning and evening. A `116_200001` → `123_085733` pair (+5.9mm) was inconclusive (±0.17%, 59 features).
  The cause is unknown: the climb within sessions looks like warm-up, but `20261125_203948_setup` read +9.6mm seconds
  after the RealSense was replugged. Effect on scans, which place tiles from this depth: against the same box on the flat
  table plane, x moves -0.22 to -0.25mm per mm of drift, y ~0.05. The setup check's tile test uses the flat plane, so it
  says nothing about this. Untested: face-down scans drop back boxes under `BACK_MIN_HEIGHT_M` (8mm), and face-up tile
  tops read as low as ~6mm on 11-23 evening.

- **The wrist camera can drop out mid-session** (2026-10-28: 8 blind misses scored against v3c). No frames
  for `camera.LOST_S` (1s) means it's lost in every script: `policy.run` stops (`stop_reason` "camera"), the
  bench voids and reruns the attempt, record.py discards the episode. Reopen with
  `hardware.reopen_wrist_camera` in any new script. Evidence: `docs/invariant_notes.md`. It can also fall off
  USB altogether (`/dev/wristcam` gone, 2026-10-30 `20261030_115706_bench`, 3 dropouts in 5 min first): only a
  replug brought it back, and the bench waits at "Fix that, then Enter".
