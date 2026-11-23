"""Drive the arm over a checkerboard on the mat and photograph it from the wrist camera at many poses, so the wrist
camera's calibration (its lens and where it sits on the arm) can be checked against the RealSense's view of the same board.

    python scripts/grid_capture.py                         # find the board with the RealSense, then 24 poses over it
    python scripts/grid_capture.py --poses 36 --clearances-mm 60,80,100,120
    python scripts/grid_capture.py --grid-xy 0.17,-0.14    # the board's centre (base frame, m), no RealSense look
    python scripts/grid_capture.py --sim                   # simulated arm and cameras, no hardware

Lay the 10x7-square board flat on the mat (nothing else under the arm's path: no tiles near it) and don't move it, or the
RealSense, until the run ends. The arm rests, the RealSense finds the board's 54 inner corners and computes where they lie on the
mat (`grid_located` event). Then for every pose it lifts, hovers over the board (the usual measured hover, settled from above)
moved by an offset that spreads over a disc, at a height and with a wrist roll that cycle so every height meets every roll,
and saves the wrist frame if the whole board is in it. Poses are spread by a fixed rule, not drawn at random, so a run can't
come out short of variety. The jaws never touch anything: the lowest height is --clearances-mm's first value above the grasp
point. Not seen from a low pose is normal (the board is 80mm wide): those are listed and left out. The heights only decide how
much of the board the camera sees; the calibration doesn't depend on them.

Saved to sessions/<time>_gridcal/: frames/NNN.jpg (the board whole), frames_partial/NNN.jpg (it wasn't), and an event per pose
(`grid_pose`: commanded and measured joints, the hover, roll, whether found, the file). `grid_located` holds the RealSense's pixels
and the corners' base-frame positions. Ends at rest. If fewer than --min-found poses saw the board it says so and how to widen.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from mahjong import config, hardware, wrist_grid
from mahjong.camera import grab_frames
from mahjong.session import Session
from wrist_capture import roll_to

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))
SIM_GRID_XY = (0.17, -0.14)
ROLL_STEPS = 3                  # rolls cycled: -roll, 0, +roll (coprime with the usual 4 heights, so all 12 pairs appear)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--poses", type=int, default=24, help="poses to try (default 24; about 12 usually see the whole board)")
    p.add_argument("--clearances-mm", default="60,80,100,120", help="hover heights above the grasp point, cycled (default 60,80,100,120)")
    p.add_argument("--jitter-mm", type=float, default=30.0, help="the hover moves over a disc of this radius round the board (default 30)")
    p.add_argument("--roll-deg", type=float, default=25.0, help="wrist roll cycled through -x, 0, +x from the hover's (default 25)")
    p.add_argument("--frames", type=int, default=3, help="wrist frames tried at each pose, the first with the whole board is kept (default 3)")
    p.add_argument("--grid-xy", default=None, metavar="X,Y", help="the board's centre in the base frame, m (default: found by the RealSense)")
    p.add_argument("--min-found", type=int, default=15, help="poses that must see the board for the run to be enough (default 15)")
    p.add_argument("--sim", action="store_true", help="simulated arm and cameras")
    return p.parse_args(argv)


def plan_poses(n, jitter_m, clearances_m, roll_deg):
    """n poses as (dx, dy, clearance, roll offset): the first centred and unrolled; offsets on a golden-angle spiral over the
    disc (even coverage for any n), heights and rolls cycled so each height meets each roll."""
    poses = []
    for i in range(n):
        r, a = (0.0, 0.0) if i == 0 else (jitter_m * math.sqrt(i / max(n - 1, 1)), i * GOLDEN_ANGLE)
        roll = (-roll_deg, 0.0, roll_deg)[i % ROLL_STEPS] if i else 0.0
        poses.append((r * math.cos(a), r * math.sin(a), clearances_m[i % len(clearances_m)], roll))
    return poses


def locate_grid(realsense, session, args):
    """The board's centre (x, y) on the mat: --grid-xy, or found in a RealSense frame (its corners are saved as the ground truth)."""
    if args.grid_xy:
        x, y = (float(v) for v in args.grid_xy.split(","))
        session.event("grid_located", source="--grid-xy", centre_xy=[x, y])
        return x, y
    if args.sim:
        session.event("grid_located", source="sim", centre_xy=list(SIM_GRID_XY))
        return SIM_GRID_XY
    found = wrist_grid.locate_on_mat(realsense.snapshot(), realsense.intrinsics, realsense.T_cam_to_base)
    if found is None:
        return None
    pixels, points = found
    centre = points[:, :2].mean(axis=0)
    session.event("grid_located", source="realsense", centre_xy=[float(v) for v in centre], pixels=pixels.tolist(),
                  corners_base_xyz=points.tolist(), square_m=wrist_grid.SQUARE_M, grid_corners=list(wrist_grid.GRID_CORNERS))
    return float(centre[0]), float(centre[1])


def capture_pose(arm, camera, session, folder, n, pose, centre, n_frames):
    """One pose: lift, hover, roll, photograph. Returns (found, the camera: reopened if it was lost)."""
    dx, dy, clearance, roll = pose
    hx, hy = centre[0] + dx, centre[1] + dy
    arm.lift()
    result = arm.hover(hx, hy, clearance + (config.HOVER_OFFSET_M - config.HOVER_CLEARANCE_M), clearance, config.HOVER_GRIPPER,
                       log=lambda *_: None)
    if result is None:
        print(f"  pose {n}: out of reach, skipped")
        session.event("grid_pose", n=n, skipped="unreachable", hover_xy=[hx, hy], target_clearance_m=clearance)
        return False, camera
    applied = roll_to(arm, roll)
    frames = grab_frames(camera, n_frames)
    if not frames:
        camera = hardware.reopen_wrist_camera(camera)
        session.event("wrist_camera_reopened", n=n)
        frames = grab_frames(camera, n_frames)
    kept, found = (frames[0] if frames else None), False
    for frame in frames:
        if wrist_grid.grid_found(frame):
            kept, found = frame, True
            break
    name = None
    if kept is not None:
        name = f"{'frames' if found else 'frames_partial'}/{n:03d}.jpg"
        (folder / name).parent.mkdir(exist_ok=True)
        cv2.imwrite(str(folder / name), kept, [cv2.IMWRITE_JPEG_QUALITY, 95])
    session.event("grid_pose", n=n, found=found, file=name, hover_xy=[hx, hy], clearance_m=result["clearance_m"],
                  target_clearance_m=clearance, roll_offset_deg=applied, commanded=arm.commanded(), measured=arm.measured())
    print(f"  pose {n}: {'board whole' if found else 'board not whole' if kept is not None else 'no wrist frame'} "
          f"(height {clearance * 1000:.0f}mm, roll {applied:+.0f}deg)")
    return found, camera


def main(argv=None):
    args = parse_args(argv)
    clearances = [float(c) / 1000 for c in args.clearances_mm.split(",")]
    if args.grid_xy:
        float(args.grid_xy.split(",")[1])            # a malformed value stops before the arm moves
    if args.sim:
        from mahjong import sim
        devices = sim.connect()
    else:
        devices = hardware.connect(["arm", "wrist_camera", "realsense"])
    arm, camera, realsense = devices["arm"], devices["wrist_camera"], devices["realsense"]
    session = Session("gridcal", ask_setup=not args.sim, sim=args.sim)
    session.event("grid_capture_start", poses=args.poses, clearances_m=clearances, jitter_mm=args.jitter_mm, roll_deg=args.roll_deg,
                  frames=args.frames, grid_xy=args.grid_xy)
    if arm.at_connect:
        session.event("connect", **arm.at_connect)
    found_count = 0
    try:
        arm.rest()          # the RealSense must see the board with the arm out of the way
        centre = locate_grid(realsense, session, args)
        if centre is None:
            print("The RealSense can't see the whole board. Lay it flat in view, clear of the arm, or give --grid-xy.")
            return session.dir
        print(f"Board centre ({centre[0] * 1000:.0f}, {centre[1] * 1000:.0f}) mm; {args.poses} poses.")
        for n, pose in enumerate(plan_poses(args.poses, args.jitter_mm / 1000, clearances, args.roll_deg), 1):
            found, camera = capture_pose(arm, camera, session, session.dir, n, pose, centre, args.frames)
            found_count += found
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        arm.rest()
        session.event("grid_capture_done", found=found_count)
        session.close()
        if not args.sim:
            camera.release()
            realsense.close()
            arm.disconnect()
    enough = found_count >= args.min_found
    print(f"{found_count} of {args.poses} poses saw the whole board" + ("." if enough else
          f", fewer than {args.min_found}: lay the board nearer the arm's reach, add --clearances-mm 80,100,120,140, or raise --poses."))
    print(f"Frames in {session.dir}/frames")
    return session.dir


if __name__ == "__main__":
    main()
