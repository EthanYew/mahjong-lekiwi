"""The crude full run on the arm (docs/face_down_plan.md, test 8): draw a face-down tile from the wall, show it to the base camera, read it, you check
the read, then the arm plays the discard. Crude on purpose: the point is a list of where it breaks, so a step that fails is written down and you do it by hand.

    python scripts/crude_run.py --hand 1C,2C,3C,5C,5C,7C,7C,9C,9C,1B,1B,2C,3C --turns 6     # a wall one tile high (where picks have worked)
    python scripts/crude_run.py --hand ... --turns 6 --stacked --top-back-mm 10             # two high: top, bottom, feed
    python scripts/crude_run.py --hand ... --sim                                            # simulated arm and cameras, no hardware

--hand is the 13 names you lay GREEN side up on the layout's slots, in the layout's order: the robot goes by memory after that. The wall's tiles can be anything
you can check, but the reader only knows the tiles it was trained on. A turn: pick from the station, hold the tile up at every saved base_view pose, set it on the drop zone
(so the arm doesn't stand holding it while you read), then a window shows the photos and the reader's guess. Enter accepts it, or type the name; X says the face wasn't visible
(then type the name, from the tile). game.py chooses the discard. Then it picks that tile from its slot into the container and the drawn tile from the zone into the slot,
or the drawn tile straight into the container. Nothing can see a green back, so every pick goes by saved position, and a landing is only as good as the aim.

Without assets/holdup_reader.pt it asks you for every name: train one with `base_cam_train.py --save-reader`. Results are in `sessions/<time>_crude/`: `crude_break` and `crude_read`
events, per attempt `wrist.mp4` and `base_*.jpg`, and `crude_summary.json`. Run `setup_check.py` first, and from the main checkout.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mahjong import face_down_rig
from mahjong import config, crude_run, base_cam, base_cam_collect, layout, markers, privacy, takeover, wall_test
from mahjong.crude_run import Run, photos_with_guess, type_in_window  # noqa: F401  (the window's helpers, here as they were)
from mahjong.perception import drain_stdin


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--hand", required=True, help="the 13 tile names you lay green side up, comma separated, in the layout's slot order")
    p.add_argument("--turns", type=int, default=6, help="draws to try (default %(default)s)")
    p.add_argument("--layout", default="even-hand13", help="the layout of the hand's slots (default %(default)s)")
    p.add_argument("--station", default="F", help=f"the wall's station in layout {wall_test.LAYOUT} (default %(default)s)")
    p.add_argument("--stacked", action="store_true", help="a wall two high: top, bottom, then feed. Default: one tile high")
    p.add_argument("--top-back-mm", type=float, default=10.0, help="with --stacked: a top tile's hover starts this much nearer the robot (default %(default)s)")
    p.add_argument("--leave-empty", default="", metavar="SLOTS", help="hand slots with no tile on the mat, comma separated, e.g. S7 (its tile right behind the wall throws the draw off): still named in --hand, never chosen as a discard")
    p.add_argument("--policy", default=None, help="policy id (default: the registry's default)")
    p.add_argument("--via-rest", action="store_true", help=crude_run.VIA_REST_HELP)
    p.add_argument("--sim", action="store_true", help="simulated arm, cameras and policy: no hardware")
    takeover.add_argument(p)
    p.set_defaults(empty=False, no_view=False, hover_lower_mm=None, frames=1, settle_s=1.0, gap_s=0.25, jitter_deg=0.0)       # what base_cam_collect's Rig reads
    return p.parse_args(argv)


def main(argv=None, ask=input):
    args = parse_args(argv)
    privacy.require()           # the weights that blur faces in the base camera's photos: before any hardware
    slots = [s["id"] for s in layout.load(args.layout)["spots"]]
    hand = crude_run.parse_hand(args.hand, slots)
    empty = crude_run.parse_slots(args.leave_empty, slots)
    stations = {s["id"]: s for s in layout.load(wall_test.LAYOUT)["spots"]}
    if args.station not in stations:
        raise SystemExit(f"layout {wall_test.LAYOUT} has no station {args.station} (has: {', '.join(stations)})")
    if args.turns < 1:
        raise SystemExit("--turns has to be at least 1")
    poses = base_cam_collect.pose_names()
    if not poses:
        raise SystemExit(f"no {base_cam_collect.POSE_PREFIX}* poses in {config.CAMERA_POSES}: teach some with scripts/teleop_view.py")
    config.discard_pose(why="The discards go in the container")
    zone = markers.zone_spot()
    if zone is None:
        raise SystemExit("No drop zone saved for this RealSense calibration: teach it with play.py --teach-zone")
    station = stations[args.station]
    all_spots = [station, zone, *layout.load(args.layout)["spots"]]
    reader = base_cam.load_reader()
    print(f"Reader: {base_cam.READER.name}, knows {', '.join(reader.classes)}" if reader else f"No reader ({base_cam.READER}): you'll name every tile. base_cam_train.py --save-reader trains one")
    rig = face_down_rig.Rig(args, all_spots, poses, sim_empty=[zone])   # the sim's zone starts clear, as the real one does
    rig.direct = not args.via_rest
    run = None
    try:
        conditions = base_cam.ask_lighting(ask)
        rig.open_session("crude", hand=hand, turns=args.turns, station=args.station, stacked=args.stacked, zone_xy=zone["xy"], reader=base_cam.READER.name if reader else None, left_empty=empty)
        rig.session.conditions = conditions
        rig.session.event("crude_conditions", conditions=conditions)
        rig.arm.rest()
        run = Run(rig, args, ask, slots, hand, station, zone, reader, base_cam.tile_boxes(), empty)
        print("\n" + crude_run.FLOW + "\n")
        drain_stdin()
        if ask(f"Lay the hand GREEN side up: {crude_run.hand_text(slots, hand, empty)}. The drop zone stays clear. Enter when done (q = quit): ").strip().lower() != "q":
            for plan in crude_run.draw_plan(args.turns, args.stacked):
                if not run.turn(plan, args.turns):
                    break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        folder = rig.session.dir if rig.session else None
        rig.close()
        if rig.session and run:
            result = crude_run.summary(run.done, run.breaks, run.reads, run.shallow)
            print("\n" + "\n".join(crude_run.summary_lines(result, run.breaks)))
            (folder / "crude_summary.json").write_text(json.dumps({**result, "breaks_list": run.breaks}, indent=2) + "\n")
            rig.session.event("crude_done", **result)
        if rig.session:
            rig.session.close()
    return folder


if __name__ == "__main__":
    main()
