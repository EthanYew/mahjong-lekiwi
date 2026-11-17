---
paths:
  - "scripts/play.py"
  - "mahjong/game.py"
  - "mahjong/identity.py"
  - "mahjong/markers.py"
  - "mahjong/hand_view.py"
  - "mahjong/name_check.py"
  - "mahjong/name_ledger.py"
  - "mahjong/placements.py"
  - "mahjong/hand_truth.py"
  - "scripts/name_report.py"
  - "scripts/train_name_check.py"
  - "scripts/name_check_eval.py"
  - "tests/test_play*.py"
---
Autonomous play: names, questions, the window and the drop zone. Evidence in `docs/invariant_notes.md`, `docs/name_data.md`.

- Autonomous play: `python scripts/play.py --layout even-hand13` (`--sim` without hardware). One session is one
  hand; it asks instead of guessing a tile name, and the hand grades clean / attended / rescued / silent
  error (`interventions.hand_grade`). Silent wrong tiles are a gate, not a target: one stops play. Bench the
  hand with `bench.py --layout even-hand14` (even-hand13 + the drop zone's spot): play won't start until the zone is.
  Face-down hands are `scripts/play_face_down.py`, with their own rules in face_down.md (its `--unattended` is the exception to this gate).
- **A stricter rule for a problem tile has to go on the name read, not the tile.** The robot never sees "8C", only the "7C"
  it reads, so real 7Cs pay too (`identity.ALWAYS_ASK_READS`, a stopgap in play). Confidence alone isn't enough: wrong EW and
  7C reads overlap real ones. Evidence: `docs/name_data.md`.
- **A second read may only add questions until it passes its exam.** Agreement changes nothing and disagreement asks
  (`name_check`, `play.py`): the two together can still agree on a wrong name, and the gate is 0 of those towards ~1000 reads.
  One exception, at the owner's request: registering the hand, agreement on a read from `identity.ALWAYS_ASK_READS` moves its
  question into the hand check a person does on all 13 before the first draw. A drawn tile has no such check, so it's still asked.
- **The votes can be sure of a wrong name, and nothing asks.** A 5C read as 9C at 0.91 of the votes (S9, `20261104_164541_play`), and
  the only way to say so was to stop the hand. So a person checks the hand before the first draw (`Player.confirm_hand`) and can
  correct a tile at any draw prompt. Enter isn't graded; a correction is a `recount`, and counts as a wrong name in the ledger.
- **A read a person overruled is no news until the tile moves.** The detector's wrong reads are steady, not flicker: S9 read 9C in 201 of
  202 scans, the name a person gave outvoted within 2-3 scans, and 53 re-asks in one hand changed no answer (`20261104_183142_play`). So a
  person's word, the hand check's Enter too, holds against scans until the tile moves past `interventions.KNOCK_M` (`Player.word_holds`). The most
  a re-asked tile had moved was 2.9mm. It doesn't hold against the wrist camera at the hover. Unconfirmed on the arm.
- **Play's RealSense frames are taken with the arm at rest.** A frame with the arm hovering shows it across the tiles, and its hole
  pattern looks like a dotted tile (the owner saw one in play's window, 2026-11-04; no logged scan has one). `PlayBench.scan` rests the arm first and keeps
  that frame, and the window shows it whenever the arm is off rest (`Player.mat_view`). Unconfirmed on the arm.
- **Play's window has its own process** (`mahjong/hand_view.py`). The main thread sits in the policy loop and `input()`, and an OpenCV
  window nobody pumps gets marked "not responding". It draws on a zoom of the tiles: on the real layout a tile is ~30px of the frame, and
  labels on the whole frame overprint each other (only seen rendering a real frame; the tests pass either way).
  Questions during a pickup need the key listener stopped first (`keys.stop()`): it holds the terminal in cbreak and eats the answer.
- **The drop zone is a saved position; the marker only teaches it.** Once taught, hands carry `data/drop_zone.json` over and nothing
  reads the marker, unless `play.py --teach-zone` or a recalibrated RealSense asks. Reading it every turn stopped `20261104_220329_play` when
  the marker itself was moved 6mm (it had read 0.4-2.1mm off in 37 checks), and a marker can't tell that from a moved camera. So a moved
  camera or base no longer stops a hand: that's `setup_check.py`, before it. A turn goes on only once a scan sees a tile on the zone,
  a faint box included, and a box proves a tile, never its name. Unconfirmed on the arm.
- **The drop zone is benched before play.** Play reaches the drawn tile's spot only mid-hand, at the far end of
  the rows (WW). `even-hand14` is `even-hand13` + that spot (S13); `play.py` refuses to start until a
  bench session has picked up and returned a tile within 25mm of the taught zone (`markers.zone_test`).
