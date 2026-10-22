"""Summarise a session folder: setup, pickup success per tile and policy, stalls, clearances, trials;
and a contact sheet (one row per attempt: overview before/after, wrist frames).

    python scripts/session_report.py                 # the latest session
    python scripts/session_report.py sessions/20261024_101500_bench
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import config
from mahjong.session import read_events

THUMB = (240, 180)


def latest_session():
    sessions = sorted(p for p in config.SESSIONS.glob("*") if (p / "events.jsonl").exists())
    if not sessions:
        raise SystemExit(f"no sessions in {config.SESSIONS}")
    return sessions[-1]


def contact_sheet(session_dir, attempts):
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
        cv2.putText(tiles[0], label, (5, THUMB[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if a["success"] else (0, 0, 255), 2)
        rows.append(np.hstack(tiles))
    if rows:
        path = session_dir / "contact_sheet.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("session", nargs="?", type=Path, default=None)
    session_dir = p.parse_args().session or latest_session()
    events = read_events(session_dir)
    start = next((e for e in events if e["type"] == "session_start"), {})
    print(f"{session_dir.name}: {start.get('kind')} at commit {start.get('commit')}")
    if start.get("notes"):
        print(f"  setup notes: {start['notes']}")
    attempts = [e for e in events if e["type"] == "attempt"]
    if attempts:
        print(f"\n{len(attempts)} pickup attempt(s), {sum(a['success'] for a in attempts)} picked up")
        by = defaultdict(list)
        for a in attempts:
            by[(a["policy"], a["tile"])].append(a)
        print(f"  {'policy':<20} {'tile':<5} {'ok':>5} {'stalls':>6} {'descended':>9} {'clearance mm':>12} {'floored':>7}")
        for (policy, tile), group in sorted(by.items()):
            ok = sum(a["success"] for a in group)
            print(f"  {policy:<20} {tile:<5} {ok:>2}/{len(group):<2} {sum(a.get('stall_warning', False) for a in group):>6} "
                  f"{sum(a.get('descended', False) for a in group):>9} "
                  f"{np.mean([a['hover_clearance_m'] for a in group]) * 1000:>12.0f} "
                  f"{sum(a.get('floored_steps', 0) for a in group):>7}")
        sheet = contact_sheet(session_dir, attempts)
        if sheet:
            print(f"\ncontact sheet: {sheet}")
    for t in (e for e in events if e["type"] == "trials_done"):
        print(f"trials on {t['tile']}: {sum(t['results'])}/{len(t['results'])} ({t['policy']})")
    for kind in ("return", "discard", "episode_saved", "correction_recorded"):
        n = sum(e["type"] == kind for e in events)
        if n:
            print(f"{kind}: {n}")


if __name__ == "__main__":
    main()
