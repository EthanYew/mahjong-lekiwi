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
recalibrating doesn't). Then for each check spot: place the spot's tile until its outline is green, the arm hovers
over it as the bench does and the wrist camera looks; the arm rests before the next spot. A tile not found
(by the RealSense at the spot, or in the wrist view at the hover) is asked about: reposition it, chasing green
again, and Enter looks again; s skips the spot, q stops the tile test. At the end the arm
is switched off and settles onto its stops, and that pose is compared too.

The wrist view finds the tile by the reference's picture of it, so each spot needs the same tile every time. Saving a
reference: use any tiles, click the middle of the tile in the window if the cross is off it, and say which tile each spot
got; a check then asks for that one.

The reference lives in data/setup_reference/ (commit it: it's what "set up correctly" means). Each run writes
sessions/<time>_setup/ with the images and a `setup_check` event; exits 1 if any check failed.
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import config, hardware, layout, wrist_check
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
        return [{k: s[k] for k in ("id", "xy", "yaw_deg", "tile_name") if k in s} for s in reference.get("spots", [])]
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
    realsense.tiles = [{"tile_name": "6B", "xy": [float(v) for v in true_xy]}]


def tile_at(realsense, spot, sim, again=False):
    """Place the tile at `spot` (the window: move it until its outline is green) and measure where it is.
    Returns ({"xy", "yaw_deg"} or None, "ok" / "missing" (the RealSense finds no tile there) / "skip" / "quit")."""
    K, T = realsense.intrinsics, realsense.T_cam_to_base
    if sim:
        place_sim_tile(realsense, spot)
    else:
        tile = spot.get("tile_name") or "a tile"     # the reference's, not the layout's: the labels aren't what's on the mat
        print(f"  {'Move' if again else 'Put'} {tile} on {spot['id']} until its outline is green, "
              "then Enter in the window (s skips the spot, q stops the tile test).")
        placed = layout.place(realsense.snapshot, [spot], spot["id"], K, T, window=WINDOW, boxes=realsense.tile_boxes)
        if placed["action"] in ("quit", "skip"):
            return None, placed["action"]
        if placed["status"] != "ok":
            print(f"  (accepted while {placed['status']}, not green)")
    row = layout.capture(realsense.snapshot, [spot], K, T, frames=3 if sim else layout.CAPTURE_FRAMES,
                         boxes=realsense.tile_boxes)[0]
    if "dx_mm" not in row:
        return None, "missing"
    return {"xy": [spot["xy"][0] + row["dx_mm"] / 1000, spot["xy"][1] + row["dy_mm"] / 1000],
            "yaw_deg": row["yaw_deg"]}, "ok"


def ask_again(ask, spot, problem):
    """A tile wasn't found: "retry" (reposition it and look again), "skip" the spot, or "quit" the tile test."""
    answer = ask(f"  {spot['id']}: {problem}. Reposition the tile (chase green in the window), then Enter to try "
                 "again; s skips this spot, q stops the tile test: ").strip().lower()
    return {"s": "skip", "q": "quit"}.get(answer[:1], "retry")


def mark_view(frame, spot, mark):
    """The wrist frame with the cross at `mark` and the template's square around it."""
    image = frame.copy()
    u, v = (int(c) for c in mark)
    half = sc.WRIST_PATCH_HALF_PX
    cv2.drawMarker(image, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.rectangle(image, (u - half, v - half), (u + half, v + half), (0, 0, 255), 1)
    cv2.putText(image, f"{spot['id']}: click the middle of the tile if the cross is off it", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return image


def confirm_on_screen(ask):
    """--save-reference: show the wrist frame with a cross where the tile should be (the one found, else where a hovered
    tile sits) and let a person click it onto the tile, so a reference can't be saved on glare, a neighbour (the
    2026-10-28 F3) or a stripe. Returns confirm(spot, frame, found) -> "retry", or the tile's centre."""
    def confirm(spot, frame, found):
        mark = [list(found["centre_px"]) if found else list(wrist_check.EXPECTED_PX)]

        def clicked(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                mark[0] = sc.patch_centre((x, y), frame.shape)
                cv2.imshow(WINDOW, mark_view(frame, spot, mark[0]))

        cv2.imshow(WINDOW, mark_view(frame, spot, mark[0]))
        cv2.setMouseCallback(WINDOW, clicked)
        cv2.waitKey(1)
        answer = ask(f"  Is the cross on the tile at {spot['id']}? Click it in the window if not. "
                     "[Enter = yes / r = reposition and look again]: ")
        cv2.setMouseCallback(WINDOW, lambda *_: None)
        return "retry" if answer.strip().lower().startswith("r") else mark[0]
    return confirm


def wrist_look(arm, camera, xy, delta):
    """Hover over the tile at xy, moved by the hover correction `delta` (m), and grab 3 wrist frames: (frames,
    hover result or None if out of reach, the camera — reopened if it was lost)."""
    frames, hover = wrist_check.look(arm, camera, xy[0] + delta[0], xy[1] + delta[1], config.DEFAULT_HOVER)
    if hover is None:
        return [], None, camera
    if not frames:
        camera = hardware.reopen_wrist_camera(camera)
        frames = grab_frames(camera, 3)
    return frames, hover, camera


def hover_delta(spot_reference, correction, xy):
    """The hover correction (m) for a tile at xy: the one stored with the reference when checking — so a refit
    correction (it moved the hover 4-8mm at the check spots between 2026-10-28 and 29) doesn't move the tile in
    the wrist view — or the live one when saving."""
    if spot_reference is not None:
        return [v / 1000 for v in spot_reference["hover_delta_mm"]]
    return list(correction.delta_at(*xy)) if correction else [0.0, 0.0]


def tile_test(arm, camera, realsense, correction, spots, reference, session, sim, ask=input, confirm=None):
    """Every spot: place, hover (with the reference's hover correction), look. A tile not found (by the RealSense,
    or in the wrist view at the hover) is repositioned and looked at again, until it's found or the person skips
    the spot / stops. Saving, confirm(spot, frame, found) -> "ok" / "retry" / the tile's centre lets a person reject a
    find or mark the tile themselves, so nothing needs finding by colour. Returns ({id: per-spot result},
    {id: wrist frame}, camera)."""
    by_id = {s["id"]: s for s in (reference or {}).get("spots", [])}
    results, frames = {}, {}
    for spot in spots:
        attempt, stop = 0, False
        ref = by_id.get(spot["id"])
        while True:                     # until the tile is found, or the person skips / quits
            arm.rest()
            placed, action = tile_at(realsense, spot, sim, again=attempt > 0)
            attempt += 1
            if action == "quit":
                stop = True
                break
            if action == "skip":
                results[spot["id"]] = {"status": "skipped", "why": "no tile placed"}
                break
            problem, frame, hover, found = "the RealSense finds no tile there", None, None, None
            if placed is not None:
                delta = hover_delta(ref, correction, placed["xy"])
                shots, hover, camera = wrist_look(arm, camera, placed["xy"], delta)
                frame = shots[-1] if shots else None
                found = sc.median_tile([sc.match_tile(f, ref["template_image"], ref["tile"]) if ref else sc.wrist_tile(f)
                                        for f in shots])
                marking = confirm is not None and ref is None
                problem = None if frame is None or found is not None or marking else "no tile in the wrist view at the hover"
            if problem is None and frame is not None and confirm is not None:
                answer = confirm(spot, frame, found)
                if answer == "retry":
                    arm.lift()
                    continue
                if answer != "ok":
                    found = sc.marked_tile(answer, found, frame.shape)
            if problem is None:
                break
            print(f"  {spot['id']}: {problem}")
            session.event("setup_tile_not_found", spot=spot["id"], attempt=attempt, problem=problem)
            if frame is not None:
                cv2.imwrite(str(session.dir / f"wrist_{spot['id']}_not_found_{attempt}.jpg"), frame)
            arm.lift()
            choice = ask_again(ask, spot, problem)
            if choice != "retry":
                results[spot["id"]] = {"status": "fail" if placed is not None else "skipped", "why": problem,
                                       "attempts": attempt}
                stop = choice == "quit"
                break
        if spot["id"] not in results:
            if stop:                    # quit before anything was measured here
                break
            if frame is None:
                results[spot["id"]] = {"status": "fail", "why": "out of reach" if hover is None else "no wrist frames"}
            else:
                frames[spot["id"]] = frame
                if reference:
                    was = ref["measured_yaw_deg"]
                    turn = None if placed["yaw_deg"] is None or was is None else layout.wrap_yaw(placed["yaw_deg"] - was)
                    result = sc.wrist_result(found, ref["tile"], placed_turn_deg=turn)
                else:
                    result = {**found, "status": "pass"}
                live = list(correction.delta_at(*placed["xy"])) if correction else [0.0, 0.0]
                results[spot["id"]] = {**result, "measured_xy": placed["xy"], "measured_yaw_deg": placed["yaw_deg"],
                                       "hover_delta_mm": [v * 1000 for v in delta], "hover_xy": hover["hover_xy"],
                                       "live_correction_change_mm": math.hypot(*(1000 * (a - b) for a, b in zip(live, delta))),
                                       "brightness": sc.brightness(frame), "attempts": attempt}
                cv2.imwrite(str(session.dir / f"wrist_{spot['id']}.jpg"), frame)
                if reference and ref.get("wrist_image") is not None:
                    cv2.imwrite(str(session.dir / f"wrist_{spot['id']}_vs_reference.jpg"),
                                sc.side_by_side(ref["wrist_image"], frame, found, ref["tile"], spot["id"]))
        r = results[spot["id"]]
        print(f"  {spot['id']}: {r['status']} — " + (sc.wrist_text(r) if reference and "du_px" in r else r.get("why") or
                                                     f"tile at {r['centre_px'][0]:.0f}, {r['centre_px'][1]:.0f}px"))
        if reference and r.get("live_correction_change_mm", 0) >= 1.0:
            print(f"      (the live hover correction now differs here by {r['live_correction_change_mm']:.1f}mm from the "
                  "reference's; not graded)")
        session.event("setup_tile", spot=spot["id"], **{k: v for k, v in r.items()})
        arm.lift()
        if stop:
            break
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
    if reference and not args.skip_tiles and any("hover_delta_mm" not in s or s.get("template_image") is None
                                                 for s in reference.get("spots", [])):
        raise SystemExit(f"The reference in {folder} ({reference.get('time')}) predates the stored hover correction and "
                         "the tile's picture the tile test now uses: run setup_check.py --save-reference again (or "
                         "--skip-tiles).")
    if devices is None and args.sim:
        from mahjong import sim
        devices = sim.connect(tiles=[], wrist_view=True)
    if devices is None:
        devices = hardware.connect(["arm", "wrist_camera", "realsense"])
    arm, camera, realsense = devices["arm"], devices["wrist_camera"], devices["realsense"]
    start = datetime.now().isoformat(timespec="seconds")
    session = Session("setup", ask_setup=not args.sim, sim=args.sim)
    show = not (args.sim or args.no_view)
    if show:              # the windows keep answering while a question waits in the terminal
        from wrist_capture import ask_pumping
        ask = ask_pumping(ask)
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
                  f"{correction.name if correction else None}); the tile test hovers with the reference's, so that "
                  "doesn't count against the setup.")
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
                confirm = confirm_on_screen(ask) if show and reference is None else None
                tiles, frames, camera = tile_test(arm, camera, realsense, correction, spots, reference, session, args.sim,
                                                  ask, confirm)
                if reference is not None:
                    results["tiles"] = sc.tiles_result(tiles)
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


def tile_names(spots, ask):
    """{spot id: the tile a person says they used}, for a check to ask for the same one (its picture is what's matched)."""
    names = {s["id"]: ask(f"  Which tile did you use on {s['id']}? A check needs the same one (e.g. 8D): ").strip() for s in spots}
    return {i: n for i, n in names.items() if n}


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
    names = {} if args.sim else tile_names([s for s in spots if s["id"] not in missing], ask)
    record = {**measured, "hover_correction": correction.name if correction else None, "layout": args.layout,
              "spots": [{**{k: v for k, v in s.items() if k != "label"},
                         **({"tile_name": names[s["id"]]} if s["id"] in names else {}),
                         "tile": {k: tiles[s["id"]][k] for k in ("centre_px", "area_px", "angle_deg", "size_px")},
                         **{k: tiles[s["id"]][k] for k in ("measured_xy", "measured_yaw_deg", "hover_delta_mm")}}
                        for s in spots if s["id"] not in missing]}
    sc.save_reference(record, color, {i: f for i, f in frames.items() if i not in missing}, folder)
    print(f"Saved to {folder}" + ("" if args.sim else " — commit it (git add data/setup_reference)."))
    return "pass"


if __name__ == "__main__":
    raise SystemExit(1 if main() == "fail" else 0)
