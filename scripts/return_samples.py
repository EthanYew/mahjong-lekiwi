"""Import return-correction samples from bench sessions: each return, with the tile before the pickup,
where the jaws closed, where the arm set it down and where the after-return scan saw it land.

bench.py records new returns itself; this backfills older sessions. Simulated sessions and returns whose
tile the scan didn't find are skipped. Sessions don't store the RealSense calibration, so only returns
from after the current setup's first gamepad correction are imported (as scripts/hover_samples.py).

    python scripts/return_samples.py                  # every bench session
    python scripts/return_samples.py sessions/20261028_090511_bench
    python -m mahjong.return_correction               # the fit
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config
from mahjong.hover_correction import append_record, read_all
from mahjong.return_correction import landings, return_record
from mahjong.session import read_events
from hover_samples import setup_since


def session_records(session_dir):
    """Return samples from a real (not --sim) bench session."""
    events = read_events(session_dir)
    if any(e["type"] == "session_start" and e.get("sim") for e in events):
        return []
    times = {e["attempt_dir"]: e["time"] for e in events if e["type"] == "attempt"}
    return [return_record(r["detection_xy"], r["grasp_xy"], r["target_xy"], r["landed_xy"], timestamp=times[r["attempt_dir"]],
                          tile_name=r["tile"], policy=r["policy"], spot=r["spot"], home_xy=r["home_xy"],
                          session=session_dir.name, attempt_dir=r["attempt_dir"])
            for r in landings(events) if r["landed_xy"]]


def import_records(records, path=None, since=None):
    """Append the records not already in the file and not older than `since`. Returns (added, already, too_old)."""
    path = path or config.RETURN_CORRECTIONS
    have = {(r.get("session"), r.get("attempt_dir")) for r in read_all(path)}
    added = already = too_old = 0
    for record in records:
        if (record["session"], record["attempt_dir"]) in have:
            already += 1
        elif since is None or record["timestamp"] < since:
            too_old += 1
        else:
            append_record(record, path)
            have.add((record["session"], record["attempt_dir"]))
            added += 1
    return added, already, too_old


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sessions", nargs="*", type=Path, help="bench session folders (none given: all)")
    args = p.parse_args(argv)
    since = setup_since()
    if since is None:
        raise SystemExit("no gamepad corrections for this setup — collect_corrections.py first")
    for folder in args.sessions or sorted(config.SESSIONS.glob("*_bench")):
        added, already, too_old = import_records(session_records(folder), since=since)
        if added or already or too_old:
            print(f"{folder.name}: {added} added, {already} already there"
                  + (f", {too_old} from before this setup's first correction ({since}) skipped" if too_old else ""))
    print("\n`python -m mahjong.return_correction` shows the fit")


if __name__ == "__main__":
    main()
