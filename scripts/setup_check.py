"""Is the workspace set up as it was? After a pack-up, compare the RealSense's view of the robot, the mat's
height, a tile in the wrist view at the hover and the arm's unpowered settle with a reference taken on a
setup you trust (mahjong/setup_check.py has what each measures and why).

    python scripts/setup_check.py --save-reference     # once, on a setup that works (e.g. after a good bench run)
    python scripts/setup_check.py                      # after every setup: pass / warn / fail per check
    python scripts/setup_check.py --skip-tiles         # camera and table only (no tile placing; the arm only rests)
    python scripts/setup_check.py --sim --save-reference && python scripts/setup_check.py --sim   # no hardware

Steps: the arm goes to rest; the RealSense looks at the robot's base and the mat. If the robot isn't where it
was in the image, a window shows the reference blended with the live view and an arrow: nudge the tripod
until the box turns green, then Enter (moving it back keeps T_cam_to_base and both corrections valid;
recalibrating doesn't). Then for each check spot: place one tile until its outline is green, the arm hovers
over it as the bench does and the wrist camera looks; the arm rests before the next spot. At the end the arm
is switched off and settles onto its stops, and that pose is compared too.

The reference lives in data/setup_reference/ (commit it: it's what "set up correctly" means). Each run writes
sessions/<time>_setup/ with the images and a `setup_check` event; exits 1 if any check failed.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import config, hardware, layout
from mahjong import setup_check as sc
from mahjong.camera import grab_frames
from mahjong.hover_correction import HoverCorrection, setup_fingerprint
from mahjong.session import Session

WINDOW = "setup check"
DEFAULT_LAYOUT, DEFAULT_SPOTS = "eval_familiar", "F2,F3,F6"      # left near, middle far, right near


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--save-reference", action="store_true",
                   help="measure this setup and save it as the reference (asks before replacing one)")
    p.add_argument("--layout", default=DEFAULT_LAYOUT,
                   help=f"with --save-reference: the layout the check spots come from (default {DEFAULT_LAYOUT})")
    p.add_argument("--spots", default=DEFAULT_SPOTS,
                   help=f"with --save-reference: the check spots' ids (default {DEFAULT_SPOTS}); a check uses "
                        "the reference's")
    p.add_argument("--skip-tiles", action="store_true", help="no tile test: camera, table, brightness, parked pose")
    p.add_argument("--reference", type=Path, default=None,
                   help="reference folder (default data/setup_reference; with --sim, sessions/sim_setup_reference)")
    p.add_argument("--no-view", action="store_true", help="no alignment window (the tile test still needs one)")
    p.add_argument("--sim", action="store_true", help="simulated arm, cameras and mat")
    return p.parse_args(argv)


def check_spots(args, reference):
    """The spots to place a tile on: the layout's (saving a reference) or the reference's (checking)."""
    if reference is not None:
        return [{k: s[k] for k in ("id", "xy", "yaw_deg", "label") if k in s} for s in reference.get("spots", [])]
    spots = layout.load(args.layout)["spots"]
    wanted = [s.strip() for s in args.spots.split(",") if s.strip()]
    missing = [w for w in wanted if w not in {s["id"] for s in spots}]
    if missing:
        raise SystemExit(f"no spot {', '.join(missing)} in layout {args.layout} (have: {[s['id'] for s in spots]})")
    return [s for w in wanted for s in spots if s["id"] == w]


# --- camera, table, mat brightness ---

def measure_camera(realsense, reference, show, wait_key=lambda: cv2.waitKey(30) & 0xFF):
    """RealSense frame, its measurements, and the camera / table results. With the robot off its reference
    place and a window: the alignment view until Enter, then measure again."""
    K, T = realsense.intrinsics, realsense.T_cam_to_base
    color, depth = realsense.depth_snapshot()
    region = reference["region"] if reference else sc.robot_region(K, T, color.shape)
    camera = None
    if reference is not None:
        camera = sc.camera_result(sc.image_shift(reference["realsense_image"], color, region), sc.mm_per_px(K, T))
        if show and camera["status"] != "pass":
            print(f"Camera: {sc.camera_text(camera)}.\n  Nudge the RealSense (or the LeKiwi) until the box turns "
                  "green, then Enter in the window (q leaves it as it is).")
            while True:
                live = realsense.snapshot()
                camera = sc.camera_result(sc.image_shift(reference["realsense_image"], live, region), sc.mm_per_px(K, T))
                cv2.imshow(WINDOW, sc.alignment_view(reference["realsense_image"], live, region, camera))
                if wait_key() in (13, 10, 32, ord("q"), 27):
                    break
            color, depth = realsense.depth_snapshot()
            camera = sc.camera_result(sc.image_shift(reference["realsense_image"], color, region), sc.mm_per_px(K, T))
    measured = {"region": list(region), "table": sc.table_fit(depth, K, T),
                "mat_brightness": sc.brightness(color, sc.mat_region(K, T, color.shape))}
    return color, measured, camera


# --- tile test ---

def place_sim_tile(realsense, spot):
    """--sim: put the tile where a person would — where its outline is drawn green, i.e. at the spot as the
    (possibly moved) camera really sees it."""
    K, T = realsense.intrinsics, realsense.T_cam_to_base
    pixel = layout.table_to_pixel([(*spot["xy"], layout.face_z(*spot["xy"]))], K, T)
    true_xy = layout.pixel_to_table(pixel, K, realsense.T_render)[0][:2]
    realsense.tiles = [{"tile_name": spot.get("label") or "6B", "xy": [float(v) for v in true_xy]}]


def tile_at(realsense, spot, sim):
    """Place the tile at `spot` and measure where it is: (x, y) or None (skipped / not found), and whether to
    quit."""
    K, T = realsense.intrinsics, realsense.T_cam_to_base
    if sim:
        place_sim_tile(realsense, spot)
    else:
        print(f"  Put a tile on {spot['id']} until its outline is green, then Enter in the window.")
        placed = layout.place(realsense.snapshot, [spot], spot["id"], K, T, window=WINDOW)
        if placed["action"] == "quit":
            return None, True
        if placed["action"] == "skip":
            return None, False
    row = layout.capture(realsense.snapshot, [spot], K, T, frames=3 if sim else layout.CAPTURE_FRAMES)[0]
    if "dx_mm" not in row:
        print(f"  {spot['id']}: no tile found there")
        return None, False
    return (spot["xy"][0] + row["dx_mm"] / 1000, spot["xy"][1] + row["dy_mm"] / 1000), False


def wrist_look(arm, camera, correction, xy):
    """Hover over the tile at xy as the bench does and grab a wrist frame: (frame or None, hover result, the
    camera — reopened if it was lost)."""
    hx, hy = correction.apply(*xy, log=lambda *_: None) if correction else xy
    arm.lift()
    hover = arm.hover(hx, hy, config.HOVER_OFFSET_M, config.HOVER_CLEARANCE_M, config.HOVER_GRIPPER, log=lambda *_: None)
    if hover is None:
        return None, None, camera
    frames = grab_frames(camera, 3)
    if not frames:
        camera = hardware.reopen_wrist_camera(camera)
        frames = grab_frames(camera, 3)
    return (frames[-1] if frames else None), hover, camera


def tile_test(arm, camera, realsense, correction, spots, reference, session, sim):
    """Every spot: place, hover, look. Returns ({id: per-spot result}, {id: wrist frame}, camera)."""
    by_id = {s["id"]: s for s in (reference or {}).get("spots", [])}
    results, frames = {}, {}
    for spot in spots:
        arm.rest()
        xy, quit_ = tile_at(realsense, spot, sim)
        if quit_:
            break
        if xy is None:
            results[spot["id"]] = {"status": "skipped", "why": "no tile placed"}
            continue
        frame, hover, camera = wrist_look(arm, camera, correction, xy)
        if frame is None:
            results[spot["id"]] = {"status": "fail", "why": "out of reach" if hover is None else "no wrist frames"}
        else:
            frames[spot["id"]] = frame
            ref = by_id.get(spot["id"])
            found = sc.wrist_tile(frame)
            result = sc.wrist_result(found, ref["tile"] if ref else None) if reference else \
                {**(found or {}), "status": "pass" if found else "fail", "why": "" if found else "no tile found in the wrist view"}
            results[spot["id"]] = {**result, "measured_xy": list(xy), "hover_xy": hover["hover_xy"],
                                   "brightness": sc.brightness(frame)}
            cv2.imwrite(str(session.dir / f"wrist_{spot['id']}.jpg"), frame)
            if reference and ref and ref.get("wrist_image") is not None:
                cv2.imwrite(str(session.dir / f"wrist_{spot['id']}_vs_reference.jpg"),
                            sc.side_by_side(ref["wrist_image"], frame, found, ref["tile"], spot["id"]))
        r = results[spot["id"]]
        print(f"  {spot['id']}: {r['status']} — " + (sc.wrist_text(r) if reference and "du_px" in r else r.get("why") or
                                                     f"tile at {r['centre_px'][0]:.0f}, {r['centre_px'][1]:.0f}px"))
        session.event("setup_tile", spot=spot["id"], **{k: v for k, v in r.items()})
        arm.lift()
    arm.rest()
    return results, frames, camera


def parked_pose_since(start):
    """The pose the arm settled at when switched off (sessions/parked_pose.json), if written after `start`."""
    from mahjong.arm import parked_path
    try:
        data = json.loads(parked_path().read_text())
    except (OSError, ValueError):
        return None
    return data["pose"] if data.get("time", "") >= start else None


def main(argv=None, ask=None, devices=None):
    """devices: {"arm", "wrist_camera", "realsense"} to use instead of connecting (tests)."""
    args = parse_args(argv)
    ask = ask or input
    folder = args.reference or (config.SESSIONS / "sim_setup_reference" if args.sim else sc.REFERENCE_DIR)
    reference = None if args.save_reference else sc.load_reference(folder)
    if not args.save_reference and reference is None:
        raise SystemExit(f"No reference in {folder}. On a setup you trust, run setup_check.py "
                         f"{'--sim ' if args.sim else ''}--save-reference first.")
    spots = check_spots(args, reference)
    if devices is None and args.sim:
        from mahjong.arm import Arm
        from mahjong.sim import SimPerception, SimRobot, SimWristCamera
        realsense = SimPerception(tiles=[])
        arm = Arm(SimRobot(), sleep=lambda s: None)
        devices = {"arm": arm, "wrist_camera": SimWristCamera(arm, realsense), "realsense": realsense}
    if devices is None:
        devices = hardware.connect(["arm", "wrist_camera", "realsense"])
    arm, camera, realsense = devices["arm"], devices["wrist_camera"], devices["realsense"]
    start = datetime.now().isoformat(timespec="seconds")
    session = Session("setup", ask_setup=not args.sim, sim=args.sim)
    show = not (args.sim or args.no_view)
    overall = "skipped"
    try:
        session.event("setup_check_start", mode="save" if args.save_reference else "check", reference=str(folder),
                      reference_time=reference and reference.get("time"), spots=[s["id"] for s in spots])
        if arm.at_connect:
            session.event("connect", **arm.at_connect)
        if reference and reference.get("calibration") != setup_fingerprint():
            print("Note: assets/T_cam_to_base.npy or the arm's calibration changed since the reference was taken: if "
                  "that's trusted, take a new reference (--save-reference) once the bench confirms it.")
        correction = HoverCorrection.load()
        if reference and reference.get("hover_correction") != (correction.name if correction else None):
            print(f"Note: the hover correction changed since the reference ({reference.get('hover_correction')} → "
                  f"{correction.name if correction else None}); the tile test measures the new one.")
        results, frames, tiles, completed = {}, {}, {}, False
        try:
            arm.rest()
            # exactly REST_POSE, as for the reference: rest() leaves an arm found parked where it is (~1° off),
            # and the folded arm is half of what the camera check matches
            arm.move_to({j: config.REST_POSE[j] for j in config.ARM_JOINTS}, 1.0)
            arm.sleep(1.0)
            color, measured, camera_check = measure_camera(realsense, reference, show)
            cv2.imwrite(str(session.dir / "realsense.png"), color)
            if reference is not None:
                cv2.imwrite(str(session.dir / "camera_vs_reference.jpg"),
                            sc.alignment_view(reference["realsense_image"], color, measured["region"], camera_check))
                results["camera"] = {**camera_check, "text": sc.camera_text(camera_check)}
                table = sc.table_result(measured["table"], reference.get("table"))
                results["table"] = {**table, "text": sc.table_text(table)}
            if not args.skip_tiles:
                print(f"Tile test on {', '.join(s['id'] for s in spots)}:")
                tiles, frames, camera = tile_test(arm, camera, realsense, correction, spots, reference, session, args.sim)
                if reference is not None:
                    shown = [f"{i} {r.get('shift_mm', float('nan')):.1f}mm" if "shift_mm" in r else f"{i} {r['status']}"
                             for i, r in tiles.items()]
                    results["tiles"] = {"status": sc.worst(r["status"] for r in tiles.values()), "spots": tiles,
                                        "text": ", ".join(shown) or "no spots done"}
            wrist_light = [r["brightness"] for r in tiles.values() if "brightness" in r]
            measured["wrist_brightness"] = float(np.median(wrist_light)) if wrist_light else None
            if reference is not None:
                mat = sc.brightness_result(measured["mat_brightness"], reference.get("mat_brightness"))
                wrist = (sc.brightness_result(measured["wrist_brightness"], reference.get("wrist_brightness"))
                         if measured["wrist_brightness"] else {"status": "skipped"})
                ratios = [f"{name} {r['ratio']:.2f}x" for name, r in (("mat", mat), ("wrist", wrist)) if "ratio" in r]
                results["brightness"] = {"status": sc.worst([mat["status"], wrist["status"]]), "mat": mat, "wrist": wrist,
                                         "text": "vs reference: " + ", ".join(ratios) if ratios else "not measured"}
            completed = True
        except KeyboardInterrupt:
            print("Interrupted.")
            session.intervention("interrupted", command="setup_check")
        finally:
            arm.rest()
            if show:
                cv2.destroyAllWindows()
            if not args.sim:
                camera.release()
                realsense.close()
                arm.disconnect()             # torque off: notes where the arm settles (the parked check)
        if not completed:
            return overall
        measured["parked"] = parked_pose_since(start) if not args.sim else None
        if reference is not None:
            parked = sc.parked_result(measured["parked"], reference.get("parked"))
            results["parked"] = {**parked, "text": sc.parked_text(parked)}
            lines, overall = sc.summary(results)
            print("\nSetup check against the reference from " + reference.get("time", "?") + ":\n" + "\n".join(lines))
            print(f"\nOverall: {overall.upper()}" + {"pass": " — set up as in the reference.",
                                                     "warn": " — close; see above before trusting a bench run.",
                                                     "fail": " — fix the above before running the bench or recording."}.get(overall, ""))
        session.event("setup_check", mode="save" if args.save_reference else "check", overall=overall,
                      measured=measured, results=results)
        if args.save_reference:
            overall = save(folder, measured, color, frames, tiles, spots, correction, args, ask)
    finally:
        session.close()
    return overall


def save(folder, measured, color, frames, tiles, spots, correction, args, ask):
    """Show what was measured and, if confirmed, write it as the reference."""
    missing = [s["id"] for s in spots if s["id"] not in frames or not tiles[s["id"]].get("centre_px")]
    print(f"\nMeasured: {sc.table_text({**measured['table'], 'status': ''}) if measured['table'] else 'no table'}; "
          f"mat brightness {measured['mat_brightness']:.0f}"
          + (f"; parked pose {', '.join(f'{v:.1f}' for v in measured['parked'].values())}" if measured["parked"] else "")
          + (f"; no tile in the wrist view at {', '.join(missing)}" if missing else ""))
    existing = sc.load_reference(folder)
    prompt = (f"Replace the reference from {existing['time']} with this setup? [y/N]: " if existing
              else "Save this setup as the reference? [y/N]: ")
    if ask(prompt).strip().lower() not in ("y", "yes"):
        print("Not saved.")
        return "skipped"
    record = {**measured, "hover_correction": correction.name if correction else None, "layout": args.layout,
              "spots": [{**s, "tile": {k: tiles[s["id"]][k] for k in ("centre_px", "area_px", "angle_deg", "size_px")}}
                        for s in spots if s["id"] not in missing]}
    sc.save_reference(record, color, {i: f for i, f in frames.items() if i not in missing}, folder)
    print(f"Saved to {folder}" + ("" if args.sim else " — commit it (git add data/setup_reference)."))
    return "pass"


if __name__ == "__main__":
    raise SystemExit(1 if main() == "fail" else 0)
