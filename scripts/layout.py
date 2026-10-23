"""Tile layouts on the RealSense view: teach one from tiles on the mat, view it live, edit spots.

A layout is a list of target spots (x, y, yaw) in data/layouts.json (mahjong/layout.py). The live view
draws each spot as a tile outline: green when the tile there is within 4mm / 5°, amber within twice
that, red otherwise. `record.py --layout NAME` walks through the spots one episode at a time.

    python scripts/layout.py --name eval_v1 --teach            # place tiles, t saves them (+ a ghost photo)
    python scripts/layout.py --name eval_v1 --teach --append --prefix N   # more spots, ids N1, N2, ...
    python scripts/layout.py --name eval_v1                    # live view: n/p spot, a all, g ghost, q quit
    python scripts/layout.py --name eval_v1 --add T1 --like S2 --yaw 15   # a tilted copy of a spot
    python scripts/layout.py --name eval_v1 --add N7 0.24 0.17
    python scripts/layout.py --name eval_v1 --remove N7
    python scripts/layout.py --list

The ghost photo is only valid while the RealSense stays where it was; the spots themselves are in the
arm's base frame and survive a recalibration.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from mahjong import config, hardware, layout

WINDOW = "layout"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
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
    print("Place the tiles. d: detect, t: save these as the layout, q: quit without saving")
    candidates = []
    while True:
        image = realsense.snapshot()
        out = layout.draw(image, existing + candidates, realsense.intrinsics, realsense.T_cam_to_base)
        cv2.putText(out, f"{len(candidates)} new spots — d detect, t save, q quit", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("d"):
            start = 1 + sum(s["id"].startswith(args.prefix) for s in existing)
            candidates = layout.spots_from_tiles(realsense.detect_tiles(), image, realsense.intrinsics,
                                                 realsense.T_cam_to_base, prefix=args.prefix)
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
    active, use_ghost = None, False
    print("n/p: next/previous spot, a: all spots, g: ghost photo on/off, q: quit")
    while True:
        image = realsense.snapshot()
        found = {s["id"]: layout.find_tile(image, s, realsense.intrinsics, realsense.T_cam_to_base) for s in spots}
        out = layout.draw(image, spots, realsense.intrinsics, realsense.T_cam_to_base, found=found, active=active,
                          ghost=ghost if use_ghost else None)
        ok = sum(layout.status(s, found[s["id"]]) == "ok" for s in spots)
        cv2.putText(out, f"{ok}/{len(spots)} placed", (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("n"), ord("p")):
            k = ids.index(active) if active in ids else (-1 if key == ord("n") else 0)
            active = ids[(k + (1 if key == ord("n") else -1)) % len(ids)]
        elif key == ord("a"):
            active = None
        elif key == ord("g"):
            if ghost is None:
                print("  no ghost photo for this layout (taught with --teach saves one)")
            use_ghost = ghost is not None and not use_ghost
        elif key in (ord("q"), 27):
            return


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
