"""Summarise a session folder: setup, pickup success per tile and policy, stalls, clearances, trials,
interventions (what a person had to do; rescues per 10 attempts, clean vs assisted successes); and a
contact sheet (one row per attempt: overview before/after, wrist frames).

    python scripts/session_report.py                 # the latest session
    python scripts/session_report.py sessions/20261024_101500_bench
    python scripts/session_report.py --all           # every bench session side by side, with its setup
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import config, interventions
from mahjong.session import read_events

THUMB = (240, 180)


def latest_session():
    sessions = sorted(p for p in config.SESSIONS.glob("*") if (p / "events.jsonl").exists())
    if not sessions:
        raise SystemExit(f"no sessions in {config.SESSIONS}")
    return sessions[-1]


def contact_sheet(session_dir, attempts, tags=None):
    """tags: {attempt_dir: {"RESCUE", "KNOCK"}}, added to each row's label."""
    rows = []
    for a in attempts:
        folder = session_dir / a["attempt_dir"]
        names = ["overview_before.jpg", "overview_after.jpg"] + sorted(p.name for p in folder.glob("wrist_*.jpg"))[:6]
        tiles = []
        for name in names:
            image = cv2.imread(str(folder / name)) if (folder / name).exists() else None
            image = np.zeros((THUMB[1], THUMB[0], 3), np.uint8) if image is None else cv2.resize(image, THUMB)
            cv2.putText(image, name.removesuffix(".jpg"), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(image)
        while len(tiles) < 8:
            tiles.append(np.zeros((THUMB[1], THUMB[0], 3), np.uint8))
        label = f"{a['tile']} {'OK' if a['success'] else 'MISS'}{' STALL' if a.get('stall_warning') else ''}"
        label += "".join(f" {t}" for t in sorted((tags or {}).get(a["attempt_dir"], ())))
        cv2.putText(tiles[0], label, (5, THUMB[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if a["success"] else (0, 0, 255), 2)
        rows.append(np.hstack(tiles))
    if rows:
        path = session_dir / "contact_sheet.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return path


def conditions_text(start):
    shown = {True: "y", False: "n"}
    text = ", ".join(f"{k}={shown.get(v, v) if isinstance(v, bool) else v}" for k, v in start.get("conditions", {}).items())
    return "; ".join(t for t in (text, start.get("notes", "")) if t)


def act_seconds(attempt):
    """How long ACT ran: duration_s, or steps at 30fps for sessions from before the early stop."""
    return attempt.get("duration_s", attempt.get("steps", 0) / config.FPS)


def perturbation_slopes(attempts):
    """For attempts from a deliberately offset hover (bench.py --hover-perturb-mm): per axis, how far
    the grasp point moved per mm of hover offset. ~0: the policy goes to the tile wherever it starts;
    ~1: it repeats a fixed move from the hover. Measured on successful grasps, against the detection.
    Returns {axis: (slope, attempts used)} for the axes that were offset both ways (or offset and not)."""
    slopes = {}
    for axis, other in ((0, 1), (1, 0)):
        used = [a for a in attempts if a.get("hover_perturb_xy") is not None and a.get("grasp_xy")
                and a["success"] and a["hover_perturb_xy"][other] == 0]
        offsets = np.array([a["hover_perturb_xy"][axis] for a in used])
        if len(set(offsets.tolist())) < 2:
            continue
        moved = np.array([a["grasp_xy"][axis] - a["detection_xy"][axis] for a in used])
        slope = float(np.sum((offsets - offsets.mean()) * (moved - moved.mean())) / np.sum((offsets - offsets.mean()) ** 2))
        slopes["xy"[axis]] = (slope, len(used))
    return slopes


def perturbation_report(attempts):
    tested = sorted({(a["policy"], a["tile"]) for a in attempts if any(a.get("hover_perturb_xy") or ())})
    if not tested:
        return
    print("\nhover perturbation (does the policy go to the tile, or repeat a move from the hover?)")
    for policy, tile in tested:
        group = [a for a in attempts if (a["policy"], a["tile"]) == (policy, tile) and a.get("hover_perturb_xy") is not None]
        per_offset = defaultdict(list)
        for a in group:
            per_offset[tuple(round(v * 1000) for v in a["hover_perturb_xy"])].append(a["success"])
        print(f"  {policy} {tile}: " + ", ".join(f"({dx:+d},{dy:+d})mm {sum(r)}/{len(r)}"
                                                  for (dx, dy), r in sorted(per_offset.items())))
        for axis, (slope, n) in perturbation_slopes(group).items():
            verdict = "follows the tile" if slope < 0.3 else "partly repeats a fixed move" if slope < 0.7 else "repeats a fixed move"
            print(f"    {axis}: grasp moved {slope:+.2f} mm per mm of hover offset ({n} grasps) — {verdict}")


def describe(i):
    """One line of what an intervention involved."""
    parts = []
    if i.get("marks_s"):
        parts.append("marked at " + ", ".join(f"{t:.1f}s" for t in i["marks_s"]))
    if "changes" in i:
        parts.append(interventions.changes_text(i["changes"]) + (f" during the {i['during']}" if i.get("during") else ""))
    if i.get("off_rest_deg"):
        parts.append(f"{i['off_rest_deg']:.0f} deg from rest at connect")
    for key in ("reason", "command", "tile", "note", "error"):
        if i.get(key):
            parts.append(f"{key} {i[key]}" if key in ("command", "tile") else str(i[key]))
    return "; ".join(parts)


def interventions_report(events, trials, unit):
    """Every intervention with its class, then the counts: rescues per 10 attempts is the scorecard
    number (ROADMAP 3); resets, overrides, failures and observations are listed but not counted."""
    found = interventions.derive(events)
    print("\ninterventions" + ("" if found else ": none"))
    if found:
        print(f"  {'attempt':<16} {'kind':<20} {'class':<11} {'source':<9} what")
    for i in found:
        print(f"  {i['attempt_dir'] or 'session':<16} {i['kind']:<20} {i['class']:<11} {i['source']:<9} {describe(i)}")
    s = interventions.summary(found, trials)
    if trials:
        rate = (f"{s['rescues_per_10']:.1f} per 10 {unit}s"
                + (f", one per {s['attempts_per_rescue']:.1f}" if s["attempts_per_rescue"] else ""))
        print(f"  rescues: {s['rescues']} in {len(trials)} {unit}s ({rate}); resets {s['resets']}, "
              f"overrides {s['overrides']}, knocked {s['failures']}, observations {s['observations']}")
        if unit == "attempt":
            print(f"  successes: {s['successes']} = {s['clean_successes']} clean + {s['assisted_successes']} assisted")
    if not any(e["type"] == "detections" for e in events):
        print("  (no scans logged: knocked tiles and tiles put back by hand can't be seen in this session)")
    return found


def all_sessions():
    print(f"{'session':<24} {'picked':>6} {'rescue/10':>9} {'ACT s':>6}  setup")
    for folder in sorted(p for p in config.SESSIONS.glob("*_bench") if (p / "events.jsonl").exists()):
        events = read_events(folder)
        attempts = [e for e in events if e["type"] == "attempt"]
        if not attempts:
            continue
        start = next((e for e in events if e["type"] == "session_start"), {})
        policies = ",".join(sorted({a["policy"] for a in attempts}))
        rescues = interventions.summary(interventions.derive(events), attempts)["rescues_per_10"]
        print(f"{folder.name:<24} {sum(a['success'] for a in attempts):>2}/{len(attempts):<3} {rescues:>9.1f} "
              f"{np.mean([act_seconds(a) for a in attempts]):>6.1f}  [{policies}] {conditions_text(start)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("session", nargs="?", type=Path, default=None)
    p.add_argument("--all", action="store_true", help="one line per bench session, with its setup")
    args = p.parse_args()
    if args.all:
        return all_sessions()
    session_dir = args.session or latest_session()
    events = read_events(session_dir)
    start = next((e for e in events if e["type"] == "session_start"), {})
    print(f"{session_dir.name}: {start.get('kind')} at commit {start.get('commit')}")
    if conditions_text(start):
        print(f"  setup: {conditions_text(start)}")
    attempts = [e for e in events if e["type"] == "attempt"]
    if attempts:
        print(f"\n{len(attempts)} pickup attempt(s), {sum(a['success'] for a in attempts)} picked up")
        by = defaultdict(list)
        for a in attempts:
            by[(a["policy"], a["tile"])].append(a)
        print(f"  {'policy':<20} {'tile':<5} {'picked':>6} {'stalls':>6} {'descended':>9} {'clearance mm':>12} "
              f"{'floored':>7} {'ACT s':>6} {'slips':>5}")
        for (policy, tile), group in sorted(by.items()):
            picked = sum(a["success"] for a in group)
            print(f"  {policy:<20} {tile:<5} {picked:>3}/{len(group):<2} {sum(a.get('stall_warning', False) for a in group):>6} "
                  f"{sum(a.get('descended', False) for a in group):>9} "
                  f"{np.mean([a['hover_clearance_m'] for a in group]) * 1000:>12.0f} "
                  f"{sum(a.get('floored_steps', 0) for a in group):>7} "
                  f"{np.mean([act_seconds(a) for a in group]):>6.1f} {sum(a.get('slipped', False) for a in group):>5}")
        print("  picked: held after the pickup, by your answer; stalls: still at the hover at the stall check\n"
              "  floored: ACT steps held back from pressing the tip into the table; slips: held, then lost")
        capped = sum(a.get("stop_reason") == "cap" for a in attempts)
        if capped:
            print(f"  {capped} attempt(s) ran to the {config.ACT_DURATION_S:.0f}s cap without a confirmed hold")
        perturbation_report(attempts)
    episodes = [e for e in events if e["type"] == "episode_saved"]
    found = interventions_report(events, attempts or episodes, "attempt" if attempts or not episodes else "episode")
    if attempts:
        tags = defaultdict(set)
        for i in found:
            if i["class"] in (interventions.RESCUE, interventions.FAILURE):
                tags[i["attempt_dir"]].add("KNOCK" if i["kind"] == "knocked" else "RESCUE")
        sheet = contact_sheet(session_dir, attempts, tags)
        if sheet:
            print(f"\ncontact sheet: {sheet}")
    for t in (e for e in events if e["type"] == "trials_done"):
        print(f"trials on {t['tile']}: {sum(t['results'])}/{len(t['results'])} ({t['policy']})")
    for kind in ("return", "return_skipped", "discard", "episode_saved", "correction_recorded"):
        n = sum(e["type"] == kind for e in events)
        if n:
            print(f"{kind}: {n}")


if __name__ == "__main__":
    main()
