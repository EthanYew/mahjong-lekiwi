"""How new each layout spot is to a policy: its distance to the nearest tile in the recordings the policy
trained on (episode-log detections), so a "new" eval spot really is new and a "familiar" one is on the grid.

v2's rows sit ~6 cm apart along y, so no reachable spot along a row is more than ~3 cm from a demo; between
the rows (x ≈ 18-19 cm) is the furthest from anything it saw (ROADMAP 2).

    python scripts/spot_novelty.py                                # every layout, against v2's recording
    python scripts/spot_novelty.py eval_familiar eval_new
    python scripts/spot_novelty.py eval_new --against Yewy/mahjong_mat_pickup_v3_jitter
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import layout
from mahjong.dataset import episode_log

V2_RECORDING = "Yewy/mahjong_mat_pickup_lekiwi_cl_hover"
FAMILIAR_MM = 10.0     # within this of a demo: v2 trained there (its spots wander ~5 mm between episodes)
NEW_MM = 25.0          # at least this far: new (the row-midpoints are ~30 mm, between the rows ~37 mm)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("layouts", nargs="*", help="layout names (default: all)")
    p.add_argument("--against", action="append",
                   help=f"training recording(s) to compare with; repeat for several (default {V2_RECORDING})")
    return p.parse_args(argv)


def verdict(mm):
    return "familiar" if mm <= FAMILIAR_MM else "new" if mm >= NEW_MM else "near a demo"


def novelty(spots, demos):
    """[(spot, distance in mm to the nearest demo detection)]."""
    return [(s, min(math.dist(s["xy"], d) for d in demos) * 1000) for s in spots]


def main(argv=None):
    args = parse_args(argv)
    sources = args.against or [V2_RECORDING]
    demos = [r["detection_xy"] for src in sources for r in episode_log(src).values() if r.get("detection_xy")]
    if not demos:
        raise SystemExit(f"no episode-log detections in {sources} (is the recording in the HF cache?)")
    layouts = layout.load_layouts()
    names = args.layouts or sorted(layouts)
    print(f"Against {len(demos)} demos from {', '.join(sources)}")
    for name in names:
        print(f"{name}")
        for spot, mm in novelty(layout.load(name)["spots"], demos):
            x, y = spot["xy"]
            print(f"  {spot['id']:5} ({x * 100:5.1f}, {y * 100:+5.1f}) cm  yaw {spot.get('yaw_deg', 0.0):+4.0f}°  "
                  f"nearest demo {mm:5.1f} mm  {verdict(mm)}")


if __name__ == "__main__":
    main()
