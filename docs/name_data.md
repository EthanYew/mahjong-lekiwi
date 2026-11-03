# Tile name data: what's collected, and the one standard for it

Written 2026-11-04. The numbers live in `python scripts/name_report.py`; the first run is kept in
`docs/retrospectives/evidence/2026-11-04_name_data/`. What's done and next stays in ROADMAP.md's status table.

## Why

The detector's raw reads were plentiful, but nothing tied "what it said" to "what the tile was". Nothing recorded which tile
sat where and when, so each question about misreads took a hand-made analysis, and two reasonable ones of the same data
disagreed (4 and 1 wrong names accepted, from the same 190 placements). Play logged no name it accepted without asking, so
asks had no denominator.

## The record

Placement → reads → decision → outcome, each a session event; everything else is worked out from them.

| Event | Says | Written by |
|---|---|---|
| `placement`, `placement_ended` | a tile at a spot, what it really is if a person said, what ended it | bench before trials and on a reposition; play |
| `detections` | every read, now with its `votes` and `frames_seen` | every scan |
| `session_start.detector` | which weights ran (older sessions: from git, by date) | bench, play |
| `attempt.placement`, `attempt.name_source` | which tile this was, and whether its name is a `read`, or how it was found without one (`faint`, `wrist_seen`) | bench |
| `name_decided` | every name play went on: `vote` or `person`, with the votes | play |
| `name_asked`, `recount`, `silent_error`, `hand_audit` | a person's word on a placement during or after a hand | play |

Not written, on purpose: a knock ending a placement, and what the votes would have done at the bench. The ledger works those
out, because a person's later verdict (`scan_verdict`) can overrule a knock, and a change to the vote bars should re-score
old sessions.

## Where each name comes from

| Source | Checked by a person? | Use it for |
|---|---|---|
| The scan's vote (`perception.merge_tiles`) | No: it's what's being tested | Reads |
| Faint boxes (`faint`, `tile_faint`) | No | "Unnamed" reads |
| `attempt.tile`, and its copies in `hover_corrections.jsonl`, `return_corrections.jsonl`, `return_landed`, `wrist_check` | No | Never as evidence of what a tile is. Attempts before this logged spot ids and labels there (`S9`, `NW`) |
| `wrist_capture.py` (`detected_as` vs the name) | Yes, first | Training photos. Not in the report: one read per set |
| `wrist_review.py` (`data/wrist_review.jsonl`) | Yes, after | The strongest truth; now covers played hands' photos too |
| The bench's check before trials | Yes, first | Truth for every read after it |
| Play's answers and end-of-hand audit (now with discards) | Yes | Truth for that hand |
| Detector exam (`data/detector_exam.json`, Roboflow) | Yes, labelled | Scoring new weights, frame by frame. Separate from this report |
| `capture_frames.py` counts, `record.py` `tile_name` | No | Not evidence: counts are of what YOLO read. `record.py` also saves no scan frames (ROADMAP 10) |
| Layout labels | No | Never: what taught the layout, not what's on the mat |

## What the report can't say yet

- **Play:** no real hand has run, so all of it is backfilled bench data. Votes are replayed over a placement's whole session,
  while play decides from about 3 scans, and older scans hold only each tile's best read.
- **Which physical tile:** copies of a name aren't told apart, so 2C reading right in `20261103_214640_bench` and as 3C in
  `20261103_222652_bench`, at the same spot and lighting answers, can't be pinned on the tile. Mark the copies if this keeps
  coming up.
- **Older knocks:** those reviews don't say what moved, so the scans are compared again. A tile landing within a few mm of
  where another left looks like the same tile, and stays a misread.
- **Attempts by name:** only spots of a layout are scored, not `--tile` trials or menu pickups.
- **The bands' thresholds** are first guesses, and the check before trials could lean towards agreeing with what's on the
  screen. The report counts how often a review overrules it.
