"""How a played hand is logged and read back, the same for both kinds of play: face up (scripts/play.py, tiles found by the
RealSense) and face down (scripts/play_face_down.py, tiles read by the base camera). Every play script logs its pickups,
turns, win and end through the writers here, and every report reads a hand through `hand()` and `pickups()`, so a reader
written for one kind of play works on the other. 2026-11-19: the game video counted 0 pickups and 0 turns of a 15-turn
face-down win (`20261119_142253_play_fd`), and the session report printed its hand as slot ids, because face-down play
logged its own event names (`crude_pick`, `crude_choice`, `crude_win`, `fd_audit`, a dict hand) that only it knew.

The events:
  attempt         a pickup (pickup_fields: the bench's fields; play adds step / turn). interventions.derive reads it.
  discard_chosen  the turn's draw and what it throws: drawn, tile_name (the discard), slot ("zone" = the drawn tile), read
  turn_done       the turn over: discarded, slot, the hand after it in slot order (`hand`, `slots`), the discards
  win             the drawn tile completed the hand: turn, drawn
  hand_end        result, turns, grade, asks, rescues, silent_errors, hand (a list in slot order), slots, discards,
                  winning (the drawn tile of a win), audit {checked, wrong: {slot: name}, covers: [slots]}
`hand()` also reads the logs from before these were shared, so older sessions keep their videos and reports.
"""
from . import game, interventions

ZONE = "zone"           # a slot: the drop zone, where the drawn tile waits (markers.ZONE_ID; play.ZONE)
REVIEW_COMMAND = "python scripts/base_cam_review.py --play"


# --- writing ---

def pickup_fields(*, policy, attempt_dir, tile, hover, offset_m, holding, command, measured, success, correction, layout,
                  spot, squeezed_from=None):
    """An `attempt` event's fields that every pickup has, bench or play (bench.py adds its own, and both add
    takeover.policy_view(run)). spot: the layout spot it was picked from, or None (bench.py picking a detection)."""
    return {"tile": tile, "policy": policy, "hover_xy": hover["hover_xy"], "hover_offset_m": offset_m,
            "hover_clearance_m": hover["clearance_m"], "gripper_command": command, "gripper_measured": measured,
            "gripper_says_held": holding, "success": success, "attempt_dir": attempt_dir, "correction": correction,
            **({"gripper_squeezed_from": squeezed_from} if squeezed_from is not None else {}),
            **({"layout": layout, "spot": spot["id"], "spot_xy": spot["xy"]} if spot else {})}


def in_order(slots, hand):
    """The hand as a list in slot order, from {slot: name} or a list already."""
    return [hand[s] for s in slots] if isinstance(hand, dict) else list(hand)


def discard_chosen(session, turn, drawn, best, slot, read=None, **extra):
    """best: game.choose_discard's pick. slot: where the discard is, ZONE for the drawn tile. read: {name, confidence, by}
    when a reader named the drawn tile (by "reader" or "person")."""
    return session.event("discard_chosen", turn=turn, drawn=drawn, slot=slot or ZONE, **best,
                         **({"read": read} if read else {}), **extra)


def turn_done(session, turn, discarded, slot, slots, hand, discards):
    return session.event("turn_done", turn=turn, discarded=discarded, slot=slot or ZONE, slots=list(slots),
                         hand=in_order(slots, hand), discards=list(discards))


def won(session, turn, drawn, read=None):
    return session.event("win", turn=turn, drawn=drawn, **({"read": read} if read else {}))


def hand_end(session, *, result, turns, grade, found, slots, hand, discards, wrong, checked, covers, winning=None, **extra):
    """found: interventions.derive's list (without the "killed" a running session shows). wrong: {slot: the table's name}
    from the end check; covers: the slots it checked (the 13, plus ZONE when the drawn tile was checked too)."""
    return session.event("hand_end", result=result, turns=turns, grade=grade, silent_errors=len(wrong),
                         asks=sum(i["class"] == interventions.ASK for i in found),
                         rescues=sum(i["class"] == interventions.RESCUE for i in found),
                         slots=list(slots), hand=in_order(slots, hand), discards=list(discards), winning=winning,
                         audit={"checked": checked, "wrong": dict(wrong), "covers": list(covers) if checked else []},
                         **extra)


# --- reading ---

def _t(e):
    return e.get("time")


def _slot(slot):
    """A slot as logged: #129's face-down end check wrote the drop zone as "ZONE"; everything else as ZONE."""
    return ZONE if slot == ZONE.upper() else slot


CHOICES = ("discard_chosen", "crude_choice")


def choice(e):
    """A discard choice event, either name: {turn, drawn, discard, slot (None = the drawn tile)}."""
    slot = e.get("slot")
    return {"turn": e["turn"], "drawn": e["drawn"], "discard": e.get("tile_name") or e.get("discard"),
            "slot": None if slot in (None, ZONE) else slot}


def pickups(events):
    """Every pickup of a session, in order: [{time, attempt_dir, spot, tile, held, step, turn}]. From `attempt` events,
    and from older face-down sessions' `crude_pick`."""
    out, seen = [], set()
    for e in events:
        if e["type"] == "attempt":
            seen.add(e.get("attempt_dir"))
            out.append({"time": _t(e), "attempt_dir": e.get("attempt_dir"), "spot": e.get("spot"), "tile": e.get("tile"),
                        "held": bool(e.get("success")), "step": e.get("step"), "turn": e.get("turn")})
    chosen = {e["turn"]: choice(e) for e in events if e["type"] in CHOICES}
    for e in events:
        if e["type"] == "crude_pick" and e.get("attempt_dir") not in seen:
            c = chosen.get(e.get("turn"), {})
            tile = {"pick_from_hand": c.get("discard"), "pick_from_zone": c.get("drawn")}.get(e.get("step"))
            out.append({"time": _t(e), "attempt_dir": e.get("attempt_dir"), "spot": e.get("spot"), "tile": tile,
                        "held": bool(e.get("held")), "step": e.get("step"), "turn": e.get("turn")})
    return sorted(out, key=lambda p: p["time"] or "")


def hand(events):
    """A played hand as it went: {kind, unattended, slots, register: [{time, slot, name, confidence}], start: {time, hand},
    turns: [{turn, time, done, drawn, read, discard, slot, before, after, discards}], win: {turn, time, drawn, read} or None,
    end: the hand_end event or None, audit: {checked, wrong, covers} or None}. `before` / `after`: the 13 names in slot order
    (`slots`) around the turn, `seen` / `discards` the discards before and after it; `done`: when it finished (None if the
    log never says). Reads old and new logs alike."""
    start_event = next((e for e in events if e["type"] == "session_start"), {})
    out = {"kind": start_event.get("kind"), "unattended": bool(start_event.get("unattended")), "slots": None,
           "register": [], "start": None, "turns": [], "win": None, "end": None, "audit": None}
    reads, current, discards, wrong_logged, audited = {}, None, [], {}, False
    by_turn = {}

    def turn_row(n):
        if n not in by_turn:
            by_turn[n] = {"turn": n, "time": None, "done": None, "drawn": None, "read": reads.get(n), "discard": None,
                          "slot": None, "before": list(current) if current else None, "seen": list(discards),
                          "after": None, "discards": None}
            out["turns"].append(by_turn[n])
        return by_turn[n]

    for e in events:
        kind = e["type"]
        if kind == "fd_register_read":
            guess = e.get("guess") or {}
            out["register"].append({"time": _t(e), "slot": e["slot"], "name": guess.get("name"),
                                    "confidence": guess.get("confidence")})
        elif kind == "fd_hand_confirmed":
            out["slots"] = out["slots"] or list(e["hand"])
            current = in_order(out["slots"], e["hand"])
            out["start"] = {"time": _t(e), "hand": list(current)}
        elif kind == "hand_confirmed" and out["start"] is None:      # face up: {placement: name}, slots unknown until turn_done
            current = list(e.get("confirmed", {}).values())
            out["start"] = {"time": _t(e), "hand": list(current)}
        elif kind == "fd_turn_start":           # face down: the hand at the start of each turn (all older logs have of it)
            out["slots"] = out["slots"] or list(e["slots"])
            current = in_order(out["slots"], e["hand"])
            out["start"] = out["start"] or {"time": _t(e), "hand": list(current)}      # a resumed hand starts here
            prev = by_turn.get(e["turn"] - 1)
            if prev is not None and prev["done"] is None:
                prev.update(done=_t(e), after=list(current), discards=list(e.get("discards", [])))
            discards = list(e.get("discards", []))
        elif kind == "crude_read" and e.get("turn"):
            reads[e["turn"]] = {"name": e["name"], "confidence": e.get("guess_confidence"), "by": e.get("by", "person")}
            if e["turn"] in by_turn:
                by_turn[e["turn"]]["read"] = reads[e["turn"]]
        elif kind in CHOICES:
            row = turn_row(e["turn"])
            c = choice(e)
            row.update(time=_t(e), drawn=c["drawn"], discard=c["discard"], slot=c["slot"], read=e.get("read") or row["read"])
            if kind == "crude_choice":          # older face-down logs: no turn_done, so the hand after it is worked out here
                after = list(current) if current else None
                if after is not None and row["slot"] is not None and out["slots"]:
                    after[out["slots"].index(row["slot"])] = row["drawn"]
                row.update(after=after, discards=discards + [row["discard"]])
        elif kind == "turn_done":
            row = turn_row(e["turn"])
            out["slots"] = out["slots"] or e.get("slots")
            current = list(e["hand"]) if e.get("hand") else current
            discards = list(e["discards"]) if "discards" in e else discards + [e.get("discarded")]
            row.update(done=_t(e), after=list(current) if current else None, discards=list(discards))
            row["discard"] = row["discard"] or e.get("discarded")
            row["time"] = row["time"] or _t(e)
        elif kind in ("win", "crude_win"):
            out["win"] = {"turn": e["turn"], "time": _t(e), "drawn": e["drawn"], "read": e.get("read") or reads.get(e["turn"]),
                          "before": list(current) if current else None, "seen": list(discards)}
        elif kind == "fd_audit":                # face-down logs before hand_end's audit: the 13, and from #129 the zone's tile as "ZONE"
            covers = e.get("covers") or [*e.get("hand", {}), *([ZONE] if e.get("won_on") else [])]
            out["audit"] = {"checked": bool(e.get("checked")), "wrong": {_slot(s): n for s, n in e.get("wrong", {}).items()},
                            "covers": [_slot(s) for s in covers] if e.get("checked") else []}
        elif kind == "hand_audit":
            audited = True
        elif kind == "intervention" and e.get("kind") == "silent_error":
            wrong_logged[_slot(e.get("slot"))] = e.get("table")
        elif kind == "hand_end":
            out["end"] = e
            if e.get("audit"):
                out["audit"] = e["audit"]
            elif out["audit"] is None and audited:     # older face-up logs: the person checked everything they were shown
                out["audit"] = {"checked": True, "wrong": wrong_logged, "covers": None}
    return out


def sorted_hand(names):
    """Names by suit then number (characters, dots, bamboo, winds, dragons), as a player holds them."""
    return sorted(names, key=lambda n: game.INDEX.get(n, 99))


def hand_line(names, gap="  "):
    """'2B 3B 4B  4D 5D  7C': one group per suit."""
    groups, last = [], None
    for name in sorted_hand(names):
        suit = name[-1] if name[0].isdigit() else "honor"
        if suit != last:
            groups.append([])
            last = suit
        groups[-1].append(name)
    return gap.join(" ".join(g) for g in groups)


def melds_line(names14):
    split = game.melds(names14)
    if split is None:
        return " ".join(sorted_hand(names14))
    return "  ".join(_meld_text(m) for m in split)


def _meld_text(meld):
    if all(n[0].isdigit() for n in meld) and len({n[-1] for n in meld}) == 1:
        return "".join(n[0] for n in meld) + meld[0][-1]
    return " ".join(meld)


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def winning_read(hand_log, reads):
    """The scored hold of the winning draw (face_down_play.scored_reads' row), or None."""
    win = hand_log["win"]
    if not win or not reads:
        return None
    rows = [r for r in reads if r.get("turn") == win["turn"] and str(r.get("where", "")).startswith("draw")]
    return rows[-1] if rows else None


def verdict(hand_log, reads=None):
    """Whether the win stands: {"status": "verified" | "not verified" | "not a win", "lines": [why, ...]}, or None for a
    hand that wasn't won. A person's end check vouches for the tiles it covers; a winning tile it didn't cover (face down,
    before 2026-11-19) needs its photos reviewed (`reads`: face_down_play.scored_reads). The win stands when the 14 a
    person vouched for still make one (game.is_win)."""
    win, end = hand_log["win"], hand_log["end"] or {}
    if end.get("result") != "win":
        return None
    lines, missing = [], []
    if hand_log["unattended"]:
        lines.append("Unattended: the reader named every tile, no person confirmed a read during play")
    audit = hand_log["audit"]
    if end.get("slots") and isinstance(end.get("hand"), list):     # hand_end's own: the hand in slot order, even with no turn_done
        slots, thirteen = list(end["slots"]), list(end["hand"])
    else:
        slots = hand_log["slots"]
        thirteen = list(win["before"]) if win and win.get("before") else None
    drawn = win["drawn"] if win else (end.get("winning") or end.get("won_on"))
    wrong = dict((audit or {}).get("wrong") or {})
    if not audit or not audit["checked"]:
        lines.append("End check skipped: no person turned the hand over")
        missing.append("the end check")
    else:
        in_hand = {s: n for s, n in wrong.items() if s != ZONE and (not slots or s in slots)}
        lines.append("End check: a person checked the 13 in hand: " + (
            "all matched" if not in_hand else ", ".join(f"{s} was {n}" for s, n in in_hand.items())))
        if thirteen and slots:
            for s, n in in_hand.items():
                thirteen[slots.index(s)] = n
    truth_drawn, drawn_checked = drawn, False
    if audit and audit["checked"] and audit.get("covers") and ZONE in audit["covers"]:
        drawn_checked, truth_drawn = True, wrong.get(ZONE, drawn)
        lines.append(f"Winning {drawn}: a person checked it" + (f": it was {truth_drawn}" if truth_drawn != drawn else ""))
    elif audit and audit["checked"] and audit.get("covers") is None:      # older face-up: the person was shown it too
        drawn_checked, truth_drawn = True, wrong.get(ZONE, drawn)
    else:
        row = winning_read(hand_log, reads)
        read = (win or {}).get("read") or {}
        said = f"read at {read['confidence']:.0%}" if read.get("confidence") is not None else "read"
        if row and row.get("reviewed") and row.get("truth"):
            drawn_checked, truth_drawn = True, row["truth"]
            lines.append(f"Winning {drawn}: its photos were reviewed: " + ("it was " + truth_drawn if truth_drawn != drawn else "right"))
        elif drawn:
            by = read.get("by")
            if by == "person":
                drawn_checked = True
                lines.append(f"Winning {drawn}: a person confirmed the read")
            else:
                lines.append(f"Winning {drawn}: {said} by the reader, photos not reviewed yet")
                missing.append(f"a review of the winning {drawn}'s photos ({REVIEW_COMMAND})")
        else:
            missing.append("the winning tile (not logged)")
    if reads:
        draws = [r for r in reads if str(r.get("where", "")).startswith("draw")]
        lines.append(f"Draws reviewed: {sum(bool(r.get('reviewed')) for r in draws)} of {len(draws)}")
    known = thirteen is not None and truth_drawn is not None
    if known and not game.is_win(thirteen + [truth_drawn]) and (wrong or truth_drawn != drawn):
        return {"status": "not a win", "lines": lines + ["Not a win: the checked tiles don't make one"]}
    if missing:
        return {"status": "not verified", "lines": lines + ["Win not verified yet: needs " + "; ".join(missing)]}
    if not drawn_checked:
        return {"status": "not verified", "lines": lines + ["Win not verified: nothing checked the winning tile"]}
    if not known:               # older face-up logs: no winning tile logged, but the robot's 14 won and a person matched them all
        if wrong:
            return {"status": "not verified", "lines": lines + [
                f"Win not verified: the end check found {plural(len(wrong), 'tile')} wrong, and the log can't say if it still wins"]}
        return {"status": "verified", "lines": lines + ["Win verified: a person matched every tile to the robot's winning hand"]}
    return {"status": "verified", "lines": lines + ["Win verified: a person vouched for all 14 tiles"]}
