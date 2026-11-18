"""Face-down play (scripts/play_face_down.py): the pieces with no hardware in them. The turn plan with no end, the window
that says where to lay the hand and the wall, the sheet a person checks the 13 reads on, and the score of a hand's reads
once a person has reviewed its photos (base_cam_review.py --play).

The hand is read by the arm, not typed in. Each slot's tile is picked, shown to the base camera at the saved poses, set
back, and named by the reader. A person checks all 13 before the first draw, and every drawn tile, unless --unattended.
Unattended, the reader's name is taken. Its photos are saved for training either way, filed under the name the hand used,
and nothing trains on them until a person has reviewed them (base_cam.not_yet_reviewed).
"""
import cv2
import numpy as np

from . import base_cam, base_cam_collect, game, layout, wall_test
from .session import read_events

PLAY_KIND = base_cam_collect.PLAY_KIND
RESHOW_BELOW = 0.6      # unattended: a read under this gets a second hold at the poses. From 6 reads on 2026-11-17: wrong at 0.41 and 0.61, right at 0.85-0.95
END_KEY = "e"           # in the wall's window: the wall's empty, end the hand (a draw)
STATION_COLOUR, ZONE_COLOUR = (255, 170, 0), (0, 165, 255)
FLOW = ("The hand: you lay 13 tiles green side up and the wall, the arm picks each tile up, shows it to the base camera and puts it back.\n"
        "Then each turn it draws from the wall, shows the tile, sets it on the drop zone and plays the discard. You feed the wall\n"
        "when asked (e in the window: the wall's empty, end the hand). A line starting 'Noted' goes on the end-of-hand list.")


def turn_plan(start=1):
    """Turn after turn from a wall two high, as wall_test.cycle_plan: top, bottom, then the next stack is pushed onto the station."""
    turn = start
    while True:
        d = wall_test.cycle_plan(turn)[-1]
        yield {"turn": d["draw"], "layer": d["layer"], "feed_before": d["feed_before"]}
        turn += 1


def unfinished_turn(events, done):
    """The turn after `done` (the checkpoint's) if it began moving tiles and stopped before its checkpoint: a quit at its
    read, at a step done by hand, or Ctrl-C. {turn, layer, choice (its crude_choice, or None before one)}, or None if it
    never started or only reached the feed (nothing moved). A resume can't redo it as it stands: its drawn tile has left
    the wall, and from the choice on the discard may be half done."""
    turn = done + 1
    if not any(e["type"] == "fd_turn_start" and e.get("turn") == turn for e in events):
        return None
    if not any(e["type"] in ("crude_pick", "crude_break") and e.get("turn") == turn for e in events):
        return None
    choice = next((e for e in reversed(events) if e["type"] == "crude_choice" and e.get("turn") == turn), None)
    return {"turn": turn, "layer": wall_test.cycle_plan(turn)[-1]["layer"], "choice": choice}


def unfinished_lines(left, station_id):
    """What a person does before a resume carries on from an unfinished turn (unfinished_turn)."""
    c = left["choice"]
    if c is None:           # the feed before it, if any, was done: the resume doesn't ask for it again
        where = f"on top of the stack on station {station_id}" if left["layer"] == "top" else f"on station {station_id}"
        return [f"Turn {left['turn']} stopped before its discard. Put the tile it drew back on the wall, {where}, "
                "face down as it was, and clear the drop zone. The turn is drawn again"]
    from_where = f"from {c['slot']}" if c["slot"] else "the drawn tile, from the drop zone"
    return [f"Turn {left['turn']} stopped after choosing its discard. Make the table match it: {c['discard']} ({from_where}) "
            "in the discard container" + (f", the drawn {c['drawn']} on {c['slot']}" if c["slot"] else "") + ", the drop zone clear"]


def lay_lines(station_id, stacks, slots):
    return [f"Lay {len(slots)} tiles GREEN side up, one on each green outline ({slots[0]}-{slots[-1]})",
            f"The wall: {stacks} stacks of 2 from WALL ({station_id}, blue), running away from the robot",
            "Keep the drop zone (orange) clear"]


def feed_step(station_id):
    step = wall_test.feed_step(station_id)
    return {**step, "lines": [*step["lines"], f"{END_KEY} = the wall's empty: end the hand"]}


def _outline(out, spot, intrinsics, T_cam_to_base, colour, label):
    corners = layout.table_to_pixel(layout.tile_corners(*spot["xy"], spot.get("yaw_deg", 0.0)), intrinsics, T_cam_to_base)
    cv2.polylines(out, [np.round(corners).astype(np.int32)], True, colour, 2)
    u = int(corners[:, 0].mean())
    cv2.putText(out, label, (u - 4 * len(label), int(corners[:, 1].max()) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)


def lay_hand_frame(image, hand_spots, station, zone, intrinsics, T_cam_to_base, lines):
    """The RealSense frame with the hand's outlines (green, each slot named), the wall's station (blue, WALL) and the drop zone
    (orange): where a person lays the hand and the wall. Nothing can see a back, so this is by eye."""
    out = layout.draw(image, hand_spots, intrinsics, T_cam_to_base, states={s["id"]: "ok" for s in hand_spots})
    _outline(out, station, intrinsics, T_cam_to_base, STATION_COLOUR, "WALL")
    _outline(out, zone, intrinsics, T_cam_to_base, ZONE_COLOUR, "zone: clear")
    cv2.rectangle(out, (0, 0), (out.shape[1], 10 + 26 * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 25 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(out, "Space/Enter: laid, go    q: quit", (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def tile_photo(frames, from_pose, boxes, size=150):
    """The first photo with a tile box, cropped as the reader sees it; a grey square without one."""
    for frame, pose in zip(frames, from_pose):
        if pose in boxes:
            return cv2.resize(base_cam.tile_frame(frame, boxes[pose]), (size, size))
    return np.full((size, size, 3), 90, np.uint8)


def hand_sheet(slots, photos, hand, guesses, low=RESHOW_BELOW, columns=7, size=150):
    """All 13 reads in one picture: each slot's photo, the name the hand has and the reader's confidence; a read under `low`,
    one a person typed, or none at all is outlined red. photos: {slot: image}; guesses: {slot: read_held's guess or None}."""
    rows = -(-len(slots) // columns)
    cell_h = size + 44
    sheet = np.zeros((40 + rows * cell_h, columns * (size + 8) + 8, 3), np.uint8)
    cv2.putText(sheet, "The robot's hand. Enter in the terminal if every name is right, or type the wrong ones: S3=5C S9=EW",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for n, slot in enumerate(slots):
        x, y = 8 + (n % columns) * (size + 8), 40 + (n // columns) * cell_h
        sheet[y:y + size, x:x + size] = photos.get(slot, np.full((size, size, 3), 90, np.uint8))
        guess = guesses.get(slot)
        sure = guess is not None and guess["name"] == hand[slot] and guess["confidence"] >= low
        colour = (0, 200, 0) if sure else (40, 40, 230)
        cv2.rectangle(sheet, (x - 2, y - 2), (x + size + 1, y + size + 1), colour, 2)
        said = f"{guess['confidence']:.0%}" if guess is not None and guess["name"] == hand[slot] else ("typed" if guess is None else f"read {guess['name']}")
        cv2.putText(sheet, f"{slot} {hand[slot]}", (x, y + size + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        cv2.putText(sheet, said, (x, y + size + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    return sheet


def bad_names(fixes):
    """The corrections whose name isn't a tile's."""
    return [f"{slot}={name}" for slot, name in fixes.items() if not game.is_playable(name)]


def summary(turns, result, grade, unattended, reads, corrections, by_hand, silent_errors, audited, waiting):
    """What one hand did: how it ended and its grade, the reads and how many a person corrected, steps done by hand, the
    end check, and the holds that wait for review."""
    return {"result": result, "turns": turns, "grade": grade, "mode": "unattended" if unattended else "attended",
            "reads": reads, "corrected": corrections, "by_hand": by_hand, "silent_errors": silent_errors,
            "end_check": audited, "holds_waiting_for_review": waiting}


def break_lines(breaks):
    """Where the hand broke, one line each (the crude run's list, without its reader line: here the reader's names aren't a person's)."""
    return ["Where it broke:"] + [f"  turn {b['turn']} {b['step']}: {b['what']}" + (" (done by hand)" if b["by_hand"] else "")
                                  for b in breaks] if breaks else ["Nothing broke."]


def summary_lines(s):
    mode = "Unattended hand" if s["mode"] == "unattended" else "Hand"
    lines = [f"{mode} over ({s['result']}) after {s['turns']} turn(s): {s['grade'] or 'not graded (it stopped before the end check)'}",
             f"{s['reads']} reads, {s['corrected']} corrected by a person, {s['by_hand']} step(s) done by hand",
             f"End check: {s['silent_errors']} tile(s) the robot had wrong" if s["end_check"] else "End check skipped: nothing says the hand was right"]
    if s["holds_waiting_for_review"]:
        lines.append(f"{s['holds_waiting_for_review']} hold(s) wait for review before the reader trains on them: python scripts/base_cam_review.py --play")
    return lines


# --- a hand's reads, scored once a person has reviewed its photos ---

def scored_reads(folder, review=None):
    """Every hold of a play session: [{shot, where, turn, guess, confidence, filed (the name the hand used), filed_by,
    truth (the reviewed name; None until reviewed, or when every photo was marked wrong without a name)}]."""
    review = base_cam.load_review() if review is None else review
    events = read_events(folder)
    truths = {}
    for row in base_cam.rows(folder, review):
        truths.setdefault(row["shot"], []).append(row["truth"])
    out = []
    for e in events:
        if e["type"] != "holdup_shot":
            continue
        names = truths.get(e["shot"], [])
        out.append({"shot": e["shot"], "where": e.get("spot"), "turn": e.get("turn"), "guess": e.get("guess"),
                    "confidence": e.get("guess_confidence"), "filed": e["tile"], "filed_by": e.get("filed_by"),
                    "truth": max(set(names), key=names.count) if names else None,
                    "reviewed": all(k in review for k in _keys(folder, e))})
    return out


def _keys(folder, shot):
    """A shot's review keys: its first photo at each pose (base_cam._shots)."""
    first = {}
    for f, pose in zip(shot["files"], shot.get("frame_poses") or [None] * len(shot["files"])):
        first.setdefault(pose, f"{folder.name}/{f}")
    return list(first.values())


def changed_choices(folder, reads):
    """The draws a wrong read changed the discard of: what the hand, as the robot had it, would have thrown with the
    tile's real name. [{turn, read, truth, discarded, would_discard}]."""
    events = read_events(folder)
    starts = {e["turn"]: e for e in events if e["type"] == "fd_turn_start"}
    chose = {e["turn"]: e for e in events if e["type"] == "crude_choice"}
    out = []
    for r in reads:
        if r["truth"] is None or r["filed"] == r["truth"] or not str(r["where"]).startswith("draw") or r["turn"] not in starts:
            continue
        start = starts[r["turn"]]
        hand13 = [start["hand"][s] for s in start["slots"]]
        _, candidates = game.choose_discard(hand13, r["truth"], start["discards"])
        would = candidates[0]["tile_name"]
        did = chose.get(r["turn"], {}).get("discard")
        if would != did:
            out.append({"turn": r["turn"], "read": r["filed"], "truth": r["truth"], "discarded": did, "would_discard": would})
    return out


def report_lines(folder, reads, changes):
    done = [r for r in reads if r["reviewed"]]
    waiting = len(reads) - len(done)
    named = [r for r in done if r["truth"] is not None]
    right = [r for r in named if r["guess"] == r["truth"]]
    lines = [f"{folder.name}: {len(reads)} hold(s), {len(done)} reviewed, {waiting} waiting (python scripts/base_cam_review.py --play)"]
    if named:
        lines.append(f"Reader: right on {len(right)} of {len(named)} reviewed reads ({len(right) / len(named):.0%})")
    lines += [f"  {r['where']}: read {r['guess'] or 'nothing'}" + (f" ({r['confidence']:.0%})" if r["confidence"] is not None else "")
              + f", really {r['truth']}" + (", the hand used the reader's name" if r["filed"] != r["truth"] else ", a person corrected it")
              for r in named if r["guess"] != r["truth"]]
    if changes:
        lines.append("Discards a wrong read changed:")
        lines += [f"  turn {c['turn']}: read {c['read']} (really {c['truth']}), threw {c['discarded']}; would have thrown {c['would_discard']}"
                  for c in changes]
    return lines
