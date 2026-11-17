---
paths:
  - "mahjong/return_correction.py"
  - "mahjong/arm.py"
  - "mahjong/layout.py"
  - "mahjong/markers.py"
  - "scripts/bench.py"
  - "scripts/record.py"
  - "scripts/play.py"
  - "scripts/return_samples.py"
  - "tests/test_return*.py"
---
Setting a held tile back down. Evidence in `docs/invariant_notes.md`.

- **A return opens only just past the jaws' reading on the tile** (~11 + `GRIPPER_RELEASE_PAST_HELD`
  6 ≈ 17%): 25% (`GRIPPER_RELEASE`, the widest opening held around a tile in recordings, still used
  over the discard container) turned a neighbour on 2026-10-23. Unconfirmed on hardware that 17% always
  lets go. The policy's own descent opens to its hover gripper (90% for v1/v2, 27.2 for v2n, 26.7-26.8 for
  v3-v4), which only a new dataset or trimmed episodes change.
- **A return lets go where that tile's jaws closed, not at a fixed height.** Grasp height varies with reach and
  policy, and a release above it drops and tilts the tile, so it lands further from home. `Arm.place` descends (only
  down, `_set_down`) to 1mm above the grasp's own height (`kinematics.grasp_clearance`). Unconfirmed on the arm.
  Evidence: `docs/invariant_notes.md`.
- **Returns aim at the tile's home, never where the jaws closed.** Returning to each grasp point carried
  every return's error into the next (tiles walked 77-100mm, 2026-10-27). The jaws pull a tile only partway
  to their centre, so the return model (`mahjong/return_correction.py`) takes the grasp as input; bench and
  record.py share the homes and landing check (`return_correction.Homes`, `check_landing`). A tile moved on
  purpose less than `HOME_RADIUS_M` (40mm) is put back at its old home. The fit pools every policy.
- **A return back to the same place needs the model to know where it is, and a landing past 30mm is no sample.** Grasp minus
  detection is mostly the gap across the mat (y: +9mm on the left, +24mm on the right), and a model without position (`grip`)
  read that as the tile keeping its offset in the jaws. When v11/v13 closed 15-19mm +y past the hover at S1/S2 face down, its
  aim followed them and tiles landed +8mm (S2 +12 to +15mm), toward the wall (`20261117_093921_bench`). `grip+position` predicts
  those at +3mm, but it lost the fit's own test to 22 neighbours' tiles taken for unseen landings 44-50mm away: the 50mm
  search reaches the next spot, and takes it by the misread fallback (any name) or, face down, the shared name "back". `check_landing` still finds and resets within 50mm and records every
  landing; `load_records` fits only those within `SAMPLE_WITHIN_M` (30mm; no real landing past 24mm), and the report counts the
  rest. The search itself still takes a neighbour when a tile is unseen. Unconfirmed on the arm.
- **A landing is matched by place first, then name.** The name a tile read before it moved says little: a carried 2C read 3C
  3mm from home, and a real 2C 104mm away took the name match, so a good landing was logged as not seen and blocked play
  (`20261104_195901_bench`). `layout.nearest_tile` filters by radius first, and `markers.carried_from` rescores old ones from their scan.
- **A tile carried across the mat is aimed from its destination, not from where it was picked up.** Grasp minus detection
  is mostly the gap between what the RealSense sees and where the arm must go, and that varies (y: +31mm at the zone, +4mm
  at S1). It cancels going back to the same place, so only play showed it: tiles landed 15-18mm off and the first hit its
  neighbours (2026-11-04). `return_correction.fit_across`; `bench.py --carry` tests it, and play needs that from the zone.
  Only a close's stray from the usual place carries over (70% x, 52% y): a steady margin at the close doesn't.
  Confirmed on the arm: 7 carried landings in 2 hands, median 4mm, worst 10.5mm (v7; before the fix, 15-19mm on v6). Evidence:
  `docs/invariant_notes.md`, `docs/retrospectives/2026-11-04_first_hands.md`.
  Evidence: `docs/invariant_notes.md`.
- **A person stops closing once the jaws stall on a tile** (commanded 5-12, reading 11-13: 20 of 35 episodes of
  `mahjong_surfaces_mat_v1`, 2026-10-30). `holds_tile` can't tell that from empty jaws, so the tile wasn't
  returned. `Arm.squeeze` shuts jaws that read under 16 before asking (record after every episode, bench after
  the policy). Teleop episodes like that also have no `grasp_frame` (no command under 5), so their return aims
  from the hover and lets go at the default 10mm.
- **The policy reaches further than the IK will plan.** It closed on 4D 11-12mm beyond the hover,
  ~0.29m from the shoulder, outside the IK's envelope (15° tilt, 10° joint margins), so the return
  was refused. `Arm.place` moves a return up to 15mm toward the shoulder, checking every height down
  to the release (reachable at the approach alone would drop the tile from there).
