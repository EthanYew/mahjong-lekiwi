# Invariant notes: the full evidence

CLAUDE.md's invariants are loaded into every Claude session, so the longest keep the rule, the reason and the
names to look for, and their full measurements live here (moved verbatim on 2026-10-29). Add to these when a
rule's evidence grows; keep the CLAUDE.md line to what someone changing the code needs.

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

## The wrist camera can drop out mid-session

2026-10-28: from the 3rd of 10 v3c trials every read failed, and the policy ran blind to the 16s cap, so 8
misses were scored that said nothing about v3c. No frames for `camera.LOST_S` (1s) means it's lost, in every
script: `policy.run` stops (`stop_reason` "camera") and the bench records `attempt_void` and runs it again;
record.py discards the episode ("wrist camera lost"); wrist_capture.py and setup_check take the view again;
collect_corrections reopens it and restarts the gamepad from the held pose. Each reopens it with
`hardware.reopen_wrist_camera` (retry / quit if it won't open): use that in any new script.

## Moved from CLAUDE.md on 2026-11-02, verbatim

The bullets below were shortened in CLAUDE.md, which loads into every session.

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
  another place at most 0.81, bare stripe at most 0.70). Three tiles of 14 caused all 11 looks, so a place confirmed 3
  times has its next 10 losses ignored (`interventions.GlitchTrust`), but not when the arm was within
  `interventions.NEIGHBOUR_M` or a key was pressed, because that's when a knock is likely. An ignored loss that was real
  isn't counted against the attempt: `glitch_ignored` events show how many went unchecked.

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

- **The wrist camera can drop out mid-session** (2026-10-28: 8 blind misses scored against v3c). No frames
  for `camera.LOST_S` (1s) means it's lost in every script: `policy.run` stops (`stop_reason` "camera"), the
  bench voids and reruns the attempt, record.py discards the episode. Reopen with
  `hardware.reopen_wrist_camera` in any new script. Evidence: `docs/invariant_notes.md`. It can also fall off
  USB altogether (`/dev/wristcam` gone, 2026-10-30 `20261030_115706_bench`, 3 dropouts in 5 min first): only a
  replug brings it back, and the bench waits at "Fix that, then Enter".
