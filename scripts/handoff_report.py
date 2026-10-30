"""What the arm did once the gamepad had it, per tile: sag, drift and command moves from the handoff traces
that collect_corrections.py saves (session handoff/; see mahjong/handoff_trace.py for what each number means).

Each tile gets a plot (handoff/<n>_<tile>.png): the tool point's x, y and z, commanded and measured, relative
to the command at the handoff, with the steps where something was pressed shaded. A measured line leaving a
flat commanded one is the servos; a commanded line moving in an unshaded stretch is the teleop.

    python scripts/handoff_report.py                     # the latest session with handoff traces
    python scripts/handoff_report.py sessions/20261029_101500_corrections
    python scripts/handoff_report.py --no-plot
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, kinematics
from mahjong.handoff_trace import describe, summarize, touched
from mahjong.session import read_events, read_jsonl


def latest_traced_session():
    sessions = sorted(p for p in config.SESSIONS.glob("*") if (p / "handoff").is_dir())
    if not sessions:
        raise SystemExit(f"no session in {config.SESSIONS} has handoff traces (collect_corrections.py saves them)")
    return sessions[-1]


def ik_mismatches(session):
    """{trace path in the session: XLeRobot IK mismatch at the seed}, from the handoff_trace events."""
    return {e["trace"]: e.get("ik_mismatch_deg") for e in read_events(session) if e.get("type") == "handoff_trace"}


def plot(rows, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["t"] for r in rows])
    cmd = np.array([kinematics.tool_point(r["command"]) for r in rows]) * 1000
    meas = np.array([kinematics.tool_point(r["measured"]) for r in rows]) * 1000
    base = cmd[0]
    pressed = [bool(touched(r)) for r in rows]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    for k, (ax, name) in enumerate(zip(axes, ("x (reach, +away)", "y (+robot's left)", "z (height)"))):
        ax.plot(t, cmd[:, k] - base[k], label="commanded", color="#1f6feb")
        ax.plot(t, meas[:, k] - base[k], label="measured", color="#d1242f")
        for a, b in _spans(t, pressed):
            ax.axvspan(a, b, color="#999999", alpha=0.2, lw=0)
        ax.set_ylabel(f"{name}\nmm")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{title}   (grey: input)")
    axes[-1].set_xlabel("s since the gamepad took over")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _spans(t, flags):
    spans, start = [], None
    for k, on in enumerate(flags):
        if on and start is None:
            start = t[k]
        if not on and start is not None:
            spans.append((start, t[k]))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


def report(session, make_plots=True, log=print):
    traces = sorted((session / "handoff").glob("*.jsonl"))
    if not traces:
        raise SystemExit(f"{session} has no handoff traces")
    mismatch = ik_mismatches(session)
    log(f"{session.name}: {len(traces)} handoff trace(s); mm, measured - commanded tool point (x reach, y left, z up)\n")
    log(f"{'tile':>8} {'secs':>5} {'sag at handoff':>20} {'drift before input':>20} {'sag at end':>20} "
        f"{'worst':>6} {'cmd idle':>8} {'IK deg':>6}")
    fmt = lambda v: f"({v[0]:+5.1f},{v[1]:+5.1f},{v[2]:+5.1f})"
    summaries = []
    for path in traces:
        rows = read_jsonl(path)
        s = summarize(rows, mismatch.get(str(path.relative_to(session))))
        summaries.append((path, s))
        ik = "-" if s["ik_mismatch_deg"] is None else f"{s['ik_mismatch_deg']:.0f}"
        log(f"{path.stem:>8} {s['duration_s']:5.1f} {fmt(s['sag_at_handoff_mm']):>20} {fmt(s['idle_drift_mm']):>20} "
            f"{fmt(s['sag_at_end_mm']):>20} {s['worst_sag_mm']:6.1f} {s['command_moved_untouched_mm'] or 0:8.1f} {ik:>6}")
        if make_plots:
            plot(rows, f"{session.name} {path.stem}", path.with_suffix(".png"))
    log("")
    for path, s in summaries:
        for line in describe(s)[1:]:
            log(f"  {path.stem}: {line}")
    if make_plots:
        log(f"\nplots: {session / 'handoff'}/*.png")
    return summaries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session", nargs="?", type=Path, default=None, help="session folder (default: the latest with traces)")
    p.add_argument("--no-plot", action="store_true", help="the table only")
    args = p.parse_args(argv)
    report(args.session or latest_traced_session(), make_plots=not args.no_plot)


if __name__ == "__main__":
    main()
