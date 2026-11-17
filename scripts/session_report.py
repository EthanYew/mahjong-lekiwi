"""Summarise a session folder: setup, pickup success per tile and policy, stalls, clearances, trials,
where returned tiles landed from their home, the scorecard (clean attempts, a paired verdict: docs/scorecard.md), interventions (what a person had to do; rescues per 10 attempts, unassisted vs assisted successes),
where to record more (misses, keyed remarks, high closes and the wrist camera per spot),
a play session's hand (scripts/play.py: result, grade, questions, turn times); and a contact sheet (one
row per attempt: overview before/after, wrist frames).

    python scripts/session_report.py                 # the latest session
    python scripts/session_report.py sessions/20261024_101500_bench
    python scripts/session_report.py --all           # every bench session side by side, then every played hand
"""
import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import bump_report, config, interventions, name_ledger, scorecard, wrist_usb
from mahjong.camera import TORN_STOP_FRAMES
from mahjong.return_correction import landings
from mahjong.session import conditions_text, read_events

THUMB = (240, 180)
CLOSE_LOW_MM, CLOSE_HIGH_MM = 10.0, 20.0     # close_height_report's bands
CLOSE_RISK_MM = 13.0     # where picks fall off since 2026-11-04: 92-96% under it, 66% at 13-16mm (scorecard refresh, 2026-11-17)


def latest_session():
    sessions = sorted(p for p in config.SESSIONS.glob("*") if (p / "events.jsonl").exists())
    if not sessions:
        raise SystemExit(f"no sessions in {config.SESSIONS}")
    return sessions[-1]


def contact_sheet(session_dir, attempts, tags=None):
    """tags: {attempt_dir: {"RESCUE", "KNOCK", "HIT"}}, added to each row's label."""
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


def setup_text(start):
    """A session's setup answers and notes, from its session_start event."""
    return conditions_text(start.get("conditions", {}), start.get("notes", ""))


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


def where(attempt):
    """A layout attempt by its spot (the tile there can change), otherwise by its tile."""
    return attempt.get("spot") or attempt["tile"]


def success_map(attempts):
    """Picked up per spot (or tile), one column per policy: a paired --policy a,b run side by side."""
    policies = sorted({a["policy"] for a in attempts})
    if len(policies) < 2 and not any(a.get("spot") for a in attempts):
        return
    print("\nsuccess by " + ("spot" if any(a.get("spot") for a in attempts) else "tile") + " and policy")
    print(f"  {'':<8} " + " ".join(f"{p:>20}" for p in policies))
    for w in dict.fromkeys(where(a) for a in attempts):
        cells = []
        for p in policies:
            got = [a["success"] for a in attempts if where(a) == w and a["policy"] == p]
            cells.append(f"{sum(got)}/{len(got)}" if got else "-")
        print(f"  {w:<8} " + " ".join(f"{c:>20}" for c in cells))


def close_height_report(attempts):
    """How high the tip was as the jaws closed (grasp_clearance_m: measured, above the grasp point), per policy and
    per spot. Chosen on the sessions up to 2026-11-03 (docs/retrospectives/2026-11-03_eval_validity.md) and held on the 473
    real attempts after: under 13mm 92-96% picked, 13-16mm 66%, 20mm or more 1/8 (mostly face down above 13mm;
    docs/retrospectives/2026-11-17_scorecard_refresh.md). A diagnostic per attempt, not a way to rank policies: v13 closed
    13mm+ 5 times to v11's 1 in one session and picked more. Picks and clean attempts decide."""
    closed = [a for a in attempts if a.get("grasp_clearance_m") is not None]
    if not closed:
        return
    heights = lambda group: [a["grasp_clearance_m"] * 1000 for a in group]
    policies = sorted({a["policy"] for a in attempts})
    print("\nclose height: tip above the grasp point as the jaws closed (mm, measured)")
    print(f"  {'policy':<20} {'closed':>7} {'median':>7} {'picked':>7} {'missed':>7} {f'<{CLOSE_LOW_MM:.0f}mm':>6} "
          f"{f'{CLOSE_RISK_MM:.0f}mm+':>6} {f'{CLOSE_HIGH_MM:.0f}mm+':>6}")
    for p in policies:
        mine = [a for a in closed if a["policy"] == p]
        if not mine:
            continue
        median = lambda group: f"{np.median(heights(group)):.1f}" if group else "-"
        all_mine = sum(a["policy"] == p for a in attempts)
        print(f"  {p:<20} {f'{len(mine)}/{all_mine}':>7} {median(mine):>7} {median([a for a in mine if a['success']]):>7} "
              f"{median([a for a in mine if not a['success']]):>7} {sum(h < CLOSE_LOW_MM for h in heights(mine)):>6} "
              f"{sum(h >= CLOSE_RISK_MM for h in heights(mine)):>6} {sum(h >= CLOSE_HIGH_MM for h in heights(mine)):>6}")
    spots = list(dict.fromkeys(where(a) for a in closed))
    print(f"  {'':<8} " + " ".join(f"{p:>20}" for p in policies) + "   (x: a miss there)")
    by_spot = {}
    for w in spots:
        cells = []
        for p in policies:
            group = [a for a in closed if where(a) == w and a["policy"] == p]
            if group:
                by_spot[(w, p)] = float(np.median(heights(group)))
            cells.append(f"{by_spot[(w, p)]:.1f}" + ("x" if not all(a["success"] for a in group) else "") if group else "-")
        print(f"  {w:<8} " + " ".join(f"{c:>20}" for c in cells))
    if len(policies) == 2:
        shared = [w for w in spots if (w, policies[0]) in by_spot and (w, policies[1]) in by_spot]
        lower = sum(by_spot[(w, policies[1])] < by_spot[(w, policies[0])] for w in shared)
        if shared:
            print(f"  {policies[1]} closed lower than {policies[0]} at {lower} of {len(shared)} spots")


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
    if i.get("off_parked_deg") is not None:
        parts.append(f"{i['off_parked_deg']:.0f} deg from where it settled at the last disconnect")
    elif i.get("off_rest_deg"):
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
              f"overrides {s['overrides']}, knocked {s['knocks']}, collided {s['collisions']}, "
              f"pushed the target {s['target_pushes']}, gripped low {s['low_grips']}, high {s['high_grips']}, "
              f"shallow {s['shallow_grips']}, "
              f"observations {s['observations']}")
        if unit == "attempt":
            print(f"  successes: {s['successes']} = {s['clean_successes']} unassisted + {s['assisted_successes']} assisted")
    if not any(e["type"] == "detections" for e in events):
        print("  (no scans logged: knocked tiles and tiles put back by hand can't be seen in this session)")
    return found


# What a person's keys (or the scans) said about an attempt that points at the policy, in the order the table shows them
REMARKS = {"pushed_target": "pushed", "collided": "bumped", "knocked": "knocked", "grip_low": "low", "grip_high": "high",
           "grip_shallow": "shallow", "aborted": "Esc", "touched_tile": "helped", "takeover": "takeover"}
KEYED = set(interventions.KEYS.values()) & set(REMARKS)     # always a column: a dash says no one pressed it


def recording_focus(events, attempts, found):
    """Where more demonstrations would help: per spot (and policy), the misses, every remark a person keyed or the
    scans found (REMARKS), closes 20mm or more up, and the wrist camera at that spot (attempts it voided and reran,
    or a pickup it stopped). A spot with a miss or a remark is listed under "record more at", most trouble first.
    The camera column is there to say whether a spot's trouble was the camera's: a voided attempt isn't counted,
    but a stop_reason "camera" one is, so its miss may not be the policy's."""
    if not any(a.get("spot") for a in attempts):
        return
    by_dir = defaultdict(Counter)
    for i in found:
        if i["kind"] in REMARKS and i.get("attempt_dir"):
            by_dir[i["attempt_dir"]][i["kind"]] += 1
    voided = Counter((e.get("spot"), e.get("policy")) for e in events if e["type"] == "attempt_void")
    policies = sorted({a["policy"] for a in attempts})
    rows = []
    for w in dict.fromkeys(where(a) for a in attempts):
        for p in policies:
            group = [a for a in attempts if where(a) == w and a["policy"] == p]
            if not group:
                continue
            remarks = Counter()
            for a in group:
                remarks.update({k: 1 for k in by_dir.get(a["attempt_dir"], ())})   # attempts with it, not presses
            misses = sum(not a["success"] for a in group)
            high = sum((a.get("grasp_clearance_m") or 0) * 1000 >= CLOSE_HIGH_MM for a in group)
            stopped = sum(a.get("stop_reason") == "camera" for a in group)
            torn = sum(bool(a.get("torn_frames")) and a.get("stop_reason") != "camera" for a in group)  # under the stop: counted
            unchecked = sum("torn_frames" not in a for a in group)     # logged before policy.run counted torn frames
            trouble = sum(not a["success"] or bool(by_dir.get(a["attempt_dir"]))
                          or (a.get("grasp_clearance_m") or 0) * 1000 >= CLOSE_HIGH_MM for a in group)
            rows.append({"spot": w, "policy": p, "n": len(group), "misses": misses, "remarks": remarks, "high": high,
                         "voided": voided[(w, p)], "stopped": stopped, "torn": torn, "unchecked": unchecked,
                         "trouble": trouble})
    print("\nwhere to record more: misses and remarks per spot (keys during the pickup or return, and the scans); "
          "camera: attempts it voided (rerun, not counted), stopped (counted) or that saw a few torn frames (counted)")
    many = len(policies) > 1
    # remark columns no attempt had are left out, except the keys; misses, 20mm+ and the camera always show
    shown = [k for k in REMARKS if k in KEYED or any(r["remarks"][k] for r in rows)]
    print(f"  {'spot':<6}" + (f" {'policy':<20}" if many else "") + f" {'missed':>7} " + " ".join(f"{REMARKS[k]:>8}" for k in shown)
          + f" {'20mm+':>6} {'camera':>20}")
    for r in rows:
        camera = ", ".join(([f"{r['voided']} voided"] if r["voided"] else []) + ([f"{r['stopped']} stopped"] if r["stopped"] else [])
                           + ([f"{r['torn']} torn"] if r["torn"] else []))
        print(f"  {r['spot']:<6}" + (f" {r['policy']:<20}" if many else "") + f" {f'{r['misses']}/{r['n']}':>7} "
              + " ".join(f"{r['remarks'][k] or '-':>8}" for k in shown) + f" {r['high'] or '-':>6} {camera or '-':>20}")
    focus = sorted((r for r in rows if r["trouble"]), key=lambda r: (-r["trouble"], -r["misses"]))
    if not focus:
        print("  record more at: nowhere stands out (no misses, remarks or high closes)")
        return
    print("  record more at (attempts with trouble, most first):")
    for r in focus:
        why = [f"missed {r['misses']}/{r['n']}"] if r["misses"] else []
        why += [f"{REMARKS[k]} {n}" for k, n in r["remarks"].items()] + ([f"closed 20mm+ up {r['high']}"] if r["high"] else [])
        # every camera remark that applies, not the first: one hid another (a void hid unchecked tears, which on
        # 20261113_100710_bench, logged before the check, were 34 of 53 attempts, and the line said "no camera loss")
        camera = ([f"the camera stopped {r['stopped']} of them: those may not be the policy's"] if r["stopped"] else []) \
            + ([f"{r['torn']} saw a few torn wrist frames (under the stop), worth a look"] if r["torn"] else []) \
            + ([f"torn frames weren't checked in {r['unchecked']} (logged before 2026-11-13): watch wrist.mp4 before "
                "recording for them"] if r["unchecked"] else [])
        if r["voided"]:
            camera.append(f"{r['voided']} voided and rerun" if camera
                          else f"no counted attempt lost the camera ({r['voided']} voided and rerun)")
        camera = "; ".join(camera) or "no camera loss"
        print(f"    {r['spot']}" + (f" {r['policy']}" if many else "") + f": {', '.join(why)}; {camera}")


def all_sessions():
    print(f"{'session':<24} {'clean':>6} {'picked':>6} {'rescue/10':>9} {'ACT s':>6}  setup")
    for folder in sorted(p for p in config.SESSIONS.glob("*_bench") if (p / "events.jsonl").exists()):
        events = read_events(folder)
        attempts = [e for e in events if e["type"] == "attempt"]
        if not attempts:
            continue
        start = next((e for e in events if e["type"] == "session_start"), {})
        policies = ",".join(sorted({a["policy"] for a in attempts}))
        found = interventions.derive(events)
        rescues = interventions.summary(found, attempts)["rescues_per_10"]
        clean = sum(scorecard.clean_flags(events, found).values())
        print(f"{folder.name:<24} {clean:>2}/{len(attempts):<3} {sum(a['success'] for a in attempts):>2}/{len(attempts):<3} {rescues:>9.1f} "
              f"{np.mean([act_seconds(a) for a in attempts]):>6.1f}  [{policies}] {setup_text(start)}"
              + ("  VOIDED" if scorecard.void_reason(folder) else ""))
    hands = [(folder, read_events(folder)) for folder in sorted(config.SESSIONS.glob("*_play"))
             if (folder / "events.jsonl").exists()]
    hands = [(folder, events, next(e for e in events if e["type"] == "hand_end")) for folder, events in hands
             if any(e["type"] == "hand_end" for e in events)]
    if hands:
        grades = Counter(end["grade"] for _, _, end in hands)
        print(f"\n{'hand':<24} {'result':<6} {'turns':>5} {'grade':<12} {'asks':>4} {'rescues':>7}  setup   "
              f"({len(hands)} hands: " + ", ".join(f"{n} {g}" for g, n in grades.most_common()) + ")")
        for folder, events, end in hands:
            start = next((e for e in events if e["type"] == "session_start"), {})
            print(f"{folder.name:<24} {end['result']:<6} {end['turns']:>5} {end['grade']:<12} {end['asks']:>4} "
                  f"{end['rescues']:>7}  {setup_text(start)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session", nargs="?", type=Path, default=None)
    p.add_argument("--all", action="store_true", help="one line per bench session, with its setup")
    args = p.parse_args()
    if args.all:
        return all_sessions()
    session_dir = args.session or latest_session()
    events = read_events(session_dir)
    start = next((e for e in events if e["type"] == "session_start"), {})
    print(f"{session_dir.name}: {start.get('kind')} at commit {start.get('commit')}")
    if (void := scorecard.void_reason(session_dir)):
        print(f"  VOIDED (data/void_sessions.json): {void}. Nothing below counts towards a policy's totals or checks")
    if setup_text(start):
        print(f"  setup: {setup_text(start)}")
    bench_setup = next((e for e in events if e["type"] == "bench_setup"), {})
    if bench_setup.get("hover_gripper_override") is not None:
        print(f"  hover gripper {bench_setup['hover_gripper_override']:g}% (bench --hover-gripper), not the policy's own")
    if bench_setup.get("hover_lower_mm"):
        print(f"  hover {bench_setup['hover_lower_mm']:g}mm lower than the policy's own (bench --hover-lower-mm): an experiment, "
              "not a counted result")
    if bench_setup.get("face_down"):
        aim = scorecard.face_down_aim(events)
        counted = aim == scorecard.SEES_BACKS and scorecard.standard_setup().get("face_down")
        print(f"  face-down tiles (bench --face-down): {aim}"
              + ("; counted as the face-down setup, compared only with other face-down sessions" if counted
                 else ("" if aim == scorecard.SEES_BACKS else ", landings unseen") + "; not a counted result"))
    if bench_setup.get("unattended"):
        c = bump_report.counts(events, bump_report.phases(events))
        print(f"  unattended: {c['flagged']} self-reported bump(s), {c['checked']} checked ({c['rejected']} didn't move). "
              f"Unchecked ones count as knocks below: python scripts/bump_report.py {session_dir}")
    recoveries = wrist_usb.recovery_summary(events)
    if recoveries:
        print(f"  wrist camera: {recoveries}")
    attempts =[e for e in events if e["type"] == "attempt"]
    if attempts:
        print(f"\n{len(attempts)} pickup attempt(s), {sum(a['success'] for a in attempts)} picked up")
        by = defaultdict(list)
        for a in attempts:
            by[(a["policy"], where(a))].append(a)
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
        voids = Counter(e.get("reason", "wrist camera stopped giving frames") for e in events if e["type"] == "attempt_void")
        for reason, n in voids.items():
            print(f"  {n} attempt(s) left out: {reason} (the bench reopened it and ran them again)")
        torn = [a for a in attempts if a.get("torn_frames") and a.get("stop_reason") != "camera"]   # stopped: their own line
        if torn:
            print(f"  {len(torn)} counted attempt(s) saw torn wrist frames before the hold, under the "
                  f"{TORN_STOP_FRAMES} that stop one: " + ", ".join(f"{a['attempt_dir'][:3]} ({a['torn_frames']})" for a in torn))
        unchecked = sum("torn_frames" not in a for a in attempts)
        if unchecked:
            print(f"  {unchecked} attempt(s) logged before torn wrist frames were checked (2026-11-13): a miss among them may be "
                  "the camera's; watch their wrist.mp4")
        success_map(attempts)
        close_height_report(attempts)
        perturbation_report(attempts)
    episodes = [e for e in events if e["type"] == "episode_saved"]
    dagger = [e for e in episodes if e.get("policy")]
    if dagger:
        taken = [e["takeover_frame"] / config.FPS for e in dagger if e.get("takeover_frame") is not None]
        print(f"\nDAgger ({', '.join(sorted({e['policy'] for e in dagger}))}): {len(dagger)} episode(s) saved, "
              f"{len(taken)} taken over" + (f" at {np.median(taken):.1f}s (median; range {min(taken):.1f}-{max(taken):.1f}s)"
                                              if taken else "")
              + (f"; {len(dagger) - len(taken)} the policy's alone (not trained on)" if len(taken) < len(dagger) else ""))
    found = interventions_report(events, attempts or episodes, "attempt" if attempts or not episodes else "episode")
    if attempts:
        recording_focus(events, attempts, found)
        print("\n" + "\n".join(scorecard.report(events, found, folder=session_dir)))
        tags = defaultdict(set)
        for i in found:
            if i["class"] in (interventions.RESCUE, interventions.FAILURE):
                tags[i["attempt_dir"]].add({"knocked": "KNOCK", "collided": "HIT"}.get(i["kind"], "RESCUE"))
        sheet = contact_sheet(session_dir, attempts, tags)
        if sheet:
            print(f"\ncontact sheet: {sheet}")
    for t in (e for e in events if e["type"] == "trials_done"):
        on = f"layout {t['layout']}" if t.get("layout") else t["tile"]
        policies = sorted(set(t.get("policies") or [t["policy"]]))
        print(f"trials on {on}: {sum(t['results'])}/{len(t['results'])} ({', '.join(policies)})"
              + (f", skipped spots {', '.join(t['skipped_spots'])}" if t.get("skipped_spots") else ""))
    for kind in ("return", "return_skipped", "discard", "episode_saved", "correction_recorded"):
        n = sum(e["type"] == kind for e in events)
        if n:
            shifts = [e["shift_m"] * 1000 for e in events if e["type"] == kind and e.get("shift_m")]
            print(f"{kind}: {n}" + (f" ({len(shifts)} set down closer to fit the reach: "
                                    + ", ".join(f"{s:.0f}mm" for s in shifts) + ")" if shifts else ""))
    returns_report(events)
    play_report(events, found)
    for line in name_ledger.session_lines(session_dir):
        print(line)


def play_report(events, found):
    """A played hand (scripts/play.py): how it ended, its grade (clean / attended / rescued / silent error,
    docs/autonomy_plan.md section 3), what a person was asked, and how long turns took."""
    turns = [e for e in events if e["type"] == "turn_done"]
    end = next((e for e in events if e["type"] == "hand_end"), None)
    if not turns and not end:
        return
    print("\nhand: " + (f"{end['result']} after {end['turns']} turns, {end['grade']}" if end
                        else f"{len(turns)} turns, not finished (--resume carries on)"))
    if end:
        print(f"  hand {' '.join(end['hand'])}; discards {' '.join(end['discards']) or '-'}")
    asked = Counter(i["kind"] for i in found if i["class"] == interventions.ASK)
    if asked:
        print("  asked: " + ", ".join(f"{kind} {n}" for kind, n in asked.most_common()))
    if end and end["silent_errors"]:
        print(f"  SILENT ERRORS: {end['silent_errors']} tile(s) the robot had wrong without asking")
    # a turn: from the first scan of its drawn tile to its turn_done
    starts, seconds = None, []
    for e in events:
        if e["type"] == "detections" and e.get("label") == "drawn" and starts is None:
            starts = datetime.fromisoformat(e["time"])
        elif e["type"] == "turn_done" and starts is not None:
            seconds.append((datetime.fromisoformat(e["time"]) - starts).total_seconds())
            starts = None
    if seconds:
        print(f"  turn time: median {np.median(seconds):.0f}s, p90 {np.percentile(seconds, 90):.0f}s "
              f"(from the drawn tile's first scan to the end of the turn)")


def returns_report(events):
    """How far returned tiles landed from their home (sessions from before homes: from where the tile was
    before that pickup), and how many a person put back. Bench and record sessions log each landing
    (`return_landed`); older bench sessions are read from their scans."""
    rise_report(events)
    found = [e for e in events if e["type"] == "return_landed" and e.get("landed_xy")] or \
        [r for r in landings(events) if r["landed_xy"]]
    if not found:
        return
    errors = sorted(r["error_m"] * 1000 for r in found)
    over = sum(e > config.RETURN_DRIFT_WARN_M * 1000 for e in errors)
    put_back = sum(e["type"] == "detections" and e.get("label") == "after_reset" or        # bench
                   e["type"] == "intervention" and e.get("kind") == "tile_reset" and "home_xy" in e   # record
                   for e in events)
    homes = any(e["type"] == "return" and e.get("home_xy") for e in events)
    print(f"returns landed {np.median(errors):.0f}mm from {'home' if homes else 'where the tile was'} (median of "
          f"{len(errors)}; worst {errors[-1]:.0f}mm; {over} over {config.RETURN_DRIFT_WARN_M * 1000:.0f}mm)"
          + (f"; {put_back} put back by a person" if put_back else ""))


def rise_report(events):
    """How the open jaws left a returned tile, measured (Arm.place's `rise`): how far the tip rose of the
    RELEASE_RISE_M commanded, how far it moved sideways doing so, and how high it was when the swing to rest
    began."""
    from mahjong.arm import RELEASE_RISE_M
    rises = [e["rise"] for e in events if e["type"] == "return" and e.get("rise")]
    if not rises:
        return
    up, side, swing = (np.array([r[k] for r in rises]) * 1000 for k in ("up_m", "sideways_m", "swing_from_m"))
    print(f"return rise: the tip rose {np.median(up):.0f}mm of {RELEASE_RISE_M * 1000:.0f} commanded (median of "
          f"{len(rises)}; least {up.min():.0f}), {np.median(side):.1f}mm sideways (worst {side.max():.1f}); "
          f"swing to rest from {np.median(swing):.0f}mm (lowest {swing.min():.0f})")
    drops = np.array([e["released_at_m"] - e["grasp_clearance_m"] for e in events if e["type"] == "return"
                      and e.get("grasp_clearance_m") is not None and "released_at_m" in e]) * 1000
    if len(drops):
        print(f"return release: {np.median(drops):+.1f}mm from where the jaws closed on the tile (median of {len(drops)}; "
              f"highest {drops.max():+.1f}): above it, the tile drops that far and can tilt as it lands")


if __name__ == "__main__":
    main()
