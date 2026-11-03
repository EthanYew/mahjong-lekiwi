"""Tile layouts on the RealSense view: teach one from tiles on the mat, view it live, edit spots.

A layout is a list of target spots (x, y, yaw) in data/layouts.json (mahjong/layout.py). The live view
draws each spot as a tile outline: green when the tile there is within 5mm / 6° (and stays green up to
1.5x that, so it doesn't flicker), amber within twice that, red otherwise. On a pale or patterned mat there's no yaw
reading: green is position only, so line the outline up by eye. `record.py --layout NAME` walks through the spots one
episode at a time.

    python scripts/layout.py --name eval_v1 --teach            # place tiles, t saves them (+ a ghost photo)
    python scripts/layout.py --name eval_v1 --teach --append --prefix N   # more spots, ids N1, N2, ...
    python scripts/layout.py --name eval_v1                    # live view: n/p spot, a all, g ghost, r reach, c capture, q quit
    python scripts/layout.py --name eval_v1 --add T1 --like S2 --yaw 15   # a tilted copy of a spot
    python scripts/layout.py --name eval_v1 --add N7 0.24 0.17
    python scripts/layout.py --name eval_v1 --remove N7
    python scripts/layout.py --list

In the live view and while teaching, r tints where the arm reaches a tile comfortably (green), only at
the edge of its reach (amber: 1D and WW on hand14, where corrections and pickups struggled), or not at
all (untinted), through the hover correction as the bench applies it. Keep taught spots in the green.

In the live view, c captures every spot's reading at once (averaged over 30 frames): offset, yaw,
their frame-to-frame spread, and whether the yaw error drifts across the mat. It's printed and saved
to data/layouts/<name>_capture_<time>.json.

The ghost photo is only valid while the RealSense stays where it was; the spots themselves are in the
arm's base frame and survive a recalibration.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import config, hardware, layout

WINDOW = "layout"


class Reach:
    """r toggles the reach mask over the view (worked out on first use: a few seconds)."""

    def __init__(self, realsense):
        self.realsense, self.masks, self.on, self.correction = realsense, None, False, None

    def toggle(self, shape, spots=()):
        """On or off; turning it on also prints the spots outside the green (the mask's 1cm cells are
        coarse where a spot sits on the boundary, as 1D and WW do)."""
        if self.masks is None:
            from mahjong.hover_correction import HoverCorrection
            print("  working out the reach (a few seconds)...")
            self.correction = HoverCorrection.load()
            self.masks = layout.reach_mask(shape, self.realsense.intrinsics, self.realsense.T_cam_to_base,
                                           self.correction)
        self.on = not self.on
        if self.on and spots:
            bands = {s["id"]: layout.reach_band(*s["xy"], self.correction) for s in spots}
            outside = [f"{i} ({b or 'out of reach'})" for i, b in bands.items() if b != "comfortable"]
            print(f"  outside comfortable reach: {', '.join(outside)}" if outside else
                  f"  all {len(spots)} spots within comfortable reach")

    def draw(self, out):
        return layout.draw_reach(out, self.masks) if self.on else out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", help="layout name, e.g. eval_v1")
    p.add_argument("--teach", action="store_true", help="save the tiles on the mat as the layout's spots")
    p.add_argument("--append", action="store_true", help="with --teach: add to the layout instead of replacing it")
    p.add_argument("--prefix", default="S", help="with --teach: spot id prefix (S1, S2, ...)")
    p.add_argument("--note", default="", help="with --teach: what the layout is for")
    p.add_argument("--add", nargs="+", metavar="ID [X Y]", help="add a spot at X Y (metres, base frame) or --like")
    p.add_argument("--like", help="with --add: copy the position of this spot")
    p.add_argument("--yaw", type=float, default=0.0, help="with --add: yaw in degrees (0 = square to the mat)")
    p.add_argument("--remove", help="remove this spot")
    p.add_argument("--list", action="store_true", help="list the layouts (or one layout's spots with --name)")
    p.add_argument("--sim", action="store_true", help="a rendered RealSense frame instead of the camera")
    return p.parse_args()


def print_layout(name, entry):
    print(f"{name}: {len(entry['spots'])} spots, taught {entry['created']}"
          + (f" — {entry['note']}" if entry.get("note") else "") + (" (ghost photo)" if entry.get("ghost") else ""))
    for s in entry["spots"]:
        print(f"  {s['id']:<5} x {s['xy'][0]:.3f}  y {s['xy'][1]:+.3f}  yaw {s['yaw_deg']:+5.1f}  {s.get('label', '')}")


def edit(args):
    entry = layout.load(args.name)
    spots = entry["spots"]
    if args.remove:
        if args.remove not in [s["id"] for s in spots]:
            raise SystemExit(f"no spot {args.remove!r}")
        spots = [s for s in spots if s["id"] != args.remove]
    if args.add:
        spot_id = args.add[0]
        if spot_id in [s["id"] for s in spots]:
            raise SystemExit(f"spot {spot_id!r} exists — --remove it first")
        if args.like:
            like = next((s for s in spots if s["id"] == args.like), None)
            if like is None:
                raise SystemExit(f"no spot {args.like!r} to copy")
            xy = like["xy"]
        elif len(args.add) == 3:
            xy = [float(args.add[1]), float(args.add[2])]
        else:
            raise SystemExit("--add ID X Y, or --add ID --like SPOT")
        spots = spots + [{"id": spot_id, "xy": xy, "yaw_deg": args.yaw}]
    print_layout(args.name, layout.save(args.name, spots, ghost=entry.get("ghost"), note=entry.get("note", "")))


def teach(realsense, args):
    existing = layout.load_layouts().get(args.name, {}).get("spots", []) if args.append else []
    print("Place the tiles. d: detect, t: save these as the layout, r: reach on/off, q: quit without saving")
    candidates, reach = [], Reach(realsense)
    while True:
        image = realsense.snapshot()
        out = reach.draw(layout.draw(image, existing + candidates, realsense.intrinsics, realsense.T_cam_to_base))
        cv2.putText(out, f"{len(candidates)} new spots — d detect, t save, r reach, q quit", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            reach.toggle(image.shape, existing + candidates)
        elif key == ord("d"):
            start = 1 + sum(s["id"].startswith(args.prefix) for s in existing)
            candidates = layout.spots_from_tiles(realsense.detect_tiles(), image, realsense.intrinsics,
                                                 realsense.T_cam_to_base, prefix=args.prefix, boxes=realsense.tile_boxes)
            for k, s in enumerate(candidates):
                s["id"] = f"{args.prefix}{start + k}"
            print(f"  {len(candidates)} tiles: " + ", ".join(f"{s['id']} {s['label']} yaw {s['yaw_deg']:+.0f}" for s in candidates))
        elif key == ord("t") and candidates:
            path = layout.ghost_path(args.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), image)
            entry = layout.save(args.name, existing + candidates, ghost=path.name, note=args.note)
            print_layout(args.name, entry)
            return
        elif key in (ord("q"), 27):
            print("Nothing saved.")
            return


def view(realsense, args):
    spots = layout.load(args.name)["spots"]
    ghost = layout.load_ghost(args.name)
    ids = [s["id"] for s in spots]
    active, use_ghost, smoother, reach = None, False, layout.Smoother(), Reach(realsense)
    print("n/p: next/previous spot, a: all spots, g: ghost photo on/off, r: reach on/off, c: capture all readings, q: quit")
    while True:
        image = realsense.snapshot()
        found, states = {}, {}
        for s in spots:
            found[s["id"]], states[s["id"]] = smoother.update(
                s, layout.find_tile(image, s, realsense.intrinsics, realsense.T_cam_to_base, boxes=realsense.tile_boxes))
        out = layout.draw(image, spots, realsense.intrinsics, realsense.T_cam_to_base, found=found, active=active,
                          ghost=ghost if use_ghost else None, states=states)
        out = reach.draw(out)
        ok = sum(state == "ok" for state in states.values())
        cv2.putText(out, f"{ok}/{len(spots)} placed", (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("n"), ord("p")):
            k = ids.index(active) if active in ids else (-1 if key == ord("n") else 0)
            active = ids[(k + (1 if key == ord("n") else -1)) % len(ids)]
        elif key == ord("a"):
            active = None
        elif key == ord("r"):
            reach.toggle(image.shape, spots)
        elif key == ord("c"):
            capture(realsense, args.name, spots)
        elif key == ord("g"):
            if ghost is None:
                print("  no ghost photo for this layout (taught with --teach saves one)")
            use_ghost = ghost is not None and not use_ghost
        elif key in (ord("q"), 27):
            return


def capture(realsense, name, spots):
    print(f"Capturing {layout.CAPTURE_FRAMES} frames — keep hands off the mat...")
    rows = layout.capture(realsense.snapshot, spots, realsense.intrinsics, realsense.T_cam_to_base,
                          boxes=realsense.tile_boxes)
    trend = layout.yaw_trend(rows)
    print(layout.capture_report(rows, trend))
    path = config.DATA / "layouts" / f"{name}_capture_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"layout": name, "time": datetime.now().isoformat(timespec="seconds"),
                                "intrinsics": realsense.intrinsics, "tile_face_m": list(config.TILE_FACE_M),
                                "spots": rows, "yaw_trend": trend}, indent=2) + "\n")
    print(f"  saved {path}")


def main():
    args = parse_args()
    if args.list:
        layouts = layout.load_layouts()
        if args.name:
            print_layout(args.name, layout.load(args.name))
        for name, entry in ([] if args.name else layouts.items()):
            print_layout(name, entry)
        if not layouts:
            print(f"No layouts in {config.LAYOUTS} yet.")
        return
    if not args.name:
        raise SystemExit("--name is required (or --list)")
    if args.add or args.remove:
        edit(args)
        return
    if args.sim:
        from mahjong.sim import SimPerception
        realsense = SimPerception()
    else:
        realsense = hardware.connect(["realsense"])["realsense"]
    try:
        (teach if args.teach else view)(realsense, args)
    finally:
        realsense.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
