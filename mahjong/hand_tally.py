"""The face-down hand tally: which played hands count toward the target of 8 clean in 10 consecutive hands, 0 silent errors
(docs/autonomy_plan.md, "Face-down hands"). Unattended hands count, once their holds are reviewed.

A hand (a `*_play_fd` session, real and not voided) is one of:
  clean / attended / rescued   graded by `hand_end` (a person only pressed Enter on reads: clean; corrected a name: attended;
                               did a step by hand: rescued), with its win verified and every read reviewed right
  silent error                 the end check found a tile the robot had wrong, or a reviewed read the hand used was wrong
  waiting for review           a hold has no verdict (`base_cam_review.py --play`): the hand can't be judged yet, and no later
                               hand counts as consecutive until it is
  unfinished                   it started turns but never reached its end check, or a person quit it: a failed hand, not skipped
  not verified                 a win no person vouched for (play_log.verdict)
  voided                       stopped by something other than the robot (data/void_sessions.json, status.py --void): listed,
                               not counted. A wrong name it used is still a silent error. It waits for review like any other while
                               a photo is marked wrong with no name, or if a later session resumed it and one has no verdict
Only hands before the first one waiting are counted, voided ones left out. The window is the last WINDOW of them."""
from pathlib import Path

from mahjong import base_cam, config, face_down_play, play_log
from mahjong.session import is_sim, read_events, void_reason

TARGET, WINDOW = 8, 10
COMPLETE = ("win", "end")       # a win, or the wall ran out ("end"); a "quit" is a hand a person stopped
CLEAN, SILENT, WAITING, UNFINISHED, NOT_VERIFIED = "clean", "silent error", "waiting for review", "unfinished", "not verified"
VOIDED = "voided"


def wrong_reads(reads):
    """The reviewed holds whose name the hand used was wrong."""
    return [r for r in reads if r["reviewed"] and r["truth"] is not None and r["filed"] != r["truth"]]


def judge(hand_log, reads, events):
    """{"status", "why"} for one hand (play_log.hand's dict, face_down_play.scored_reads' rows, the session's events), or None
    for a session that never started a turn and ended nothing (a quit at the laying window: not a hand)."""
    end = hand_log["end"]
    wrong = wrong_reads(reads)
    started = any(e["type"] == "fd_turn_start" for e in events)
    if end is None and not started:         # a quit at the laying window: not a hand, whatever its registration holds said
        return None
    if wrong or (end or {}).get("silent_errors"):      # a known wrong name the hand used is a silent error, finished or not
        found = [f"end check: {end['silent_errors']} tile(s)"] if (end or {}).get("silent_errors") else []
        found += [f"{r['where']} read {r['filed']}, really {r['truth']}" for r in wrong]
        return {"status": SILENT, "why": "; ".join(found)}
    if end is None or not end.get("end_check") or end.get("result") not in COMPLETE:
        if not started:
            return None
        why = ("it never reached its end check" if end is None else "the end check was skipped" if not end.get("end_check")
               else f"a person ended it early ({end.get('result')})")
        return {"status": UNFINISHED, "why": why}
    unnamed = [r for r in reads if r["reviewed"] and r["truth"] is None]
    waiting = [r for r in reads if not r["reviewed"]]
    if waiting or unnamed:
        return {"status": WAITING, "why": f"{len(waiting)} of {len(reads)} hold(s) have no verdict, {len(unnamed)} marked wrong "
                                          "with no name (rename it in base_cam_review.py --play --all)"}
    check = play_log.verdict(hand_log, reads)
    if end.get("result") == "win" and (check or {}).get("status") != "verified":
        return {"status": NOT_VERIFIED, "why": "; ".join((check or {}).get("lines", [])) or "no person vouched for the win"}
    word = end.get("end_check_on_word")
    return {"status": end.get("grade") or "-", "why": f"end check on the owner's word: {word}" if word else ""}


def series(rows):
    """Split chronological rows ({status, ...}) into (counted, behind): the hands before the first one waiting for review (voided
    ones left out), and the ones from it on, which can't be called consecutive until it is reviewed."""
    first = next((i for i, row in enumerate(rows) if row["status"] == WAITING), len(rows))
    return [row for row in rows[:first] if row["status"] != VOIDED], rows[first:]


def window(counted):
    """The last WINDOW counted hands and whether they meet the target: {hands, clean, silent, met}."""
    last = counted[-WINDOW:]
    clean = sum(r["status"] == CLEAN for r in last)
    silent = sum(r["status"] == SILENT for r in last)
    setups = sorted({r["setup"] for r in last})
    return {"hands": len(last), "clean": clean, "silent": silent, "setups": setups,
            "met": len(last) == WINDOW and clean >= TARGET and silent == 0 and len(setups) == 1}


def tally(sessions_dir=None, review=None):
    """Every face-down hand in order: [{name, mode, turns, result, status, why, setup}]. `review`: the review store
    (base_cam.load_review), loaded here when not given."""
    folder_root = Path(sessions_dir or config.SESSIONS)
    review = base_cam.load_review() if review is None else review
    rows = []
    folders = [f for f in sorted(folder_root.glob("*_play_fd")) if (f / "events.jsonl").exists() and not is_sim(f)]
    resumed = {e.get("resumed_from") for f in folders for e in read_events(f) if e["type"] == "session_start"}
    for folder in folders:
        events = read_events(folder)
        hand_log = play_log.hand(events)
        reads = face_down_play.scored_reads(folder, review)
        verdict = judge(hand_log, reads, events)
        if verdict is None:
            continue
        if (why := void_reason(folder)) and verdict["status"] != SILENT:
            unnamed = sum(r["reviewed"] and r["truth"] is None for r in reads)
            carried = sum(not r["reviewed"] for r in reads) if folder.name in resumed else 0
            held = ([f"{unnamed} hold(s) are marked wrong with no name"] if unnamed else []) \
                + ([f"a later session carried its hand on and {carried} hold(s) have no verdict"] if carried else [])
            verdict = {"status": WAITING, "why": "voided, but " + "; ".join(held)} if held else {"status": VOIDED, "why": why}
        start = next((e for e in events if e["type"] == "session_start"), {})
        end = hand_log["end"] or {}
        rows.append({"name": folder.name, "mode": "unattended" if hand_log["unattended"] else "attended",
                     "turns": end.get("turns"), "result": end.get("result") or "-", **verdict,
                     "setup": f"{start.get('policy')}, {start.get('layout')}, station {start.get('station')}"})
    return rows


def lines(rows):
    """The tally as text for status.py."""
    counted, behind = series(rows)
    out = [f"Face-down hands (target {TARGET} of {WINDOW} consecutive clean, 0 silent errors; unattended count once reviewed; mahjong/hand_tally.py)"]
    if not rows:
        return out + ["  no face-down hand played yet"]
    for i, r in enumerate(rows):
        mark = "  (not consecutive yet: behind a hand waiting for review)" \
            if i >= len(rows) - len(behind) and r["status"] not in (WAITING, VOIDED) else ""
        out.append(f"  {r['name']:<26} {r['mode']:<10} {r['result']:<5} {r['turns'] or '-':>3} turns  {r['status']}"
                   + (f": {r['why']}" if r["why"] else "") + mark)
    w = window(counted)
    out.append(f"  counted {len(counted)} hand(s); last {w['hands']} of the {WINDOW} needed: {w['clean']} clean, {w['silent']} silent error(s)"
               + (" -> TARGET MET" if w["met"] else f" -> not met (needs {WINDOW} hands with {TARGET}+ clean)"))
    if any(r["status"] == SILENT for r in counted):
        out.append("  GATE: a silent error is in the record: it stops play until its cause is dealt with (.claude/rules/play.md)")
    if len(w["setups"]) > 1:
        out.append("  WARNING: the window mixes setups, so the target can't be met on it (the plan never pools them): " + " | ".join(w["setups"]))
    return out
