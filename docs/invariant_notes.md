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
