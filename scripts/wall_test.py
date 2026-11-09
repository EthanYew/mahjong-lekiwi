"""Test draws from a wall on the arm (docs/face_down_plan.md, "The wall: one station"): v8 at a saved spot, a person says how each went.

    python scripts/wall_test.py --push-check --stations F --trials 10     # no arm: does a pushed row land in the same place?
    python scripts/wall_test.py --hover-check --stations F,L              # the arm hovers over a face-up tile on each station, no policy
    python scripts/wall_test.py --stations F,L --cases control,flat --trials 5     # which station: 5 of each, alternating
    python scripts/wall_test.py --stations F --cases strip --trials 10    # then at that station: strip, top, bottom in turn
    python scripts/wall_test.py --stations F --cases cycle --trials 12    # the real thing: 12 draws, fed by hand every second draw
    python scripts/wall_test.py --report                                  # every real session so far, scored
    python scripts/wall_test.py --sim ...                                 # simulated arm and cameras, no hardware

A station is one saved spot in data/layouts.json (`wall-stations`). Put a tile on it face up to green with `layout.py --name wall-stations`,
tape around it, then flip: the detector can't see a back, so nothing else says a face-down tile is in place. The layout's xy stands in
for a detection; a top tile's hover is one tile higher. Before each attempt a window shows the station and says what to lay.

Each attempt: RealSense frame with the arm at rest, hover, policy, rest, one photo from the base camera if the jaws read held (does it
show two tiles?), into the discard container, RealSense frame again, then your one key on how it went. Enter isn't an answer, so it
can't become a habit. In a cycle anything but clean asks you to put the wall right; that's logged too. A dropped wrist camera voids the
attempt and it runs again. Results go to `sessions/<time>_wall/`: `wall_attempt` events, each attempt's `wrist.mp4`,
`table_before_after.jpg` and `base.jpg`, and `wall_summary.json`. Run `setup_check.py` first, and from the main checkout.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import holdup_arm as holdup_script
from mahjong import camera_poses, config, gamepad, holdup, holdup_arm, layout, privacy, wall_test, wrist_check
from mahjong.camera import grab_frames
from mahjong.perception import drain_stdin
from mahjong.session import Session, read_events

CROP_PX = 150               # table_before_after.jpg: this far around the station


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stations", default="F", help="station ids in the layout, comma separated (default %(default)s; the layout has F, in front of the hand, and L, on its left)")
    p.add_argument("--cases", default="flat", help=f"what to try, comma separated: {', '.join(wall_test.CASES)}; or {wall_test.CYCLE} alone (default %(default)s)")
    p.add_argument("--trials", type=int, default=10, help="tries of each station and case; in a cycle, draws; in a push check, pushes (default %(default)s)")
    p.add_argument("--layout", default=wall_test.LAYOUT, help="the layout holding the stations (default %(default)s)")
    p.add_argument("--push-check", action="store_true", help="no arm: scan a face-up lead tile after each push, and say how far pushes stray")
    p.add_argument("--hover-check", action="store_true", help="no policy: hover over a face-up tile on each station and save the wrist view")
    p.add_argument("--report", action="store_true", help="score every real wall-test session so far; no hardware")
    p.add_argument("--policy", default=None, help="policy id (default: the registry's default)")
    p.add_argument("--sim", action="store_true", help="simulated arm, cameras and policy: no hardware")
    p.set_defaults(empty=False, no_view=False, hover_lower_mm=None, frames=1, settle_s=1.0, gap_s=0.25, jitter_deg=0.0)       # what holdup_arm's Rig reads
    return p.parse_args(argv)


def prompt(rig, station, text, ask):
    """Show what to lay, on the live RealSense window (or the terminal). Returns "", "s" or "q"."""
    if rig.realsense is None:
        return ask(f"{text}. Enter when done (s = skip, q = quit): ").strip().lower()
    print(f"{text}\n  (the RealSense window shows where: Space/Enter when done, s skip, q quit)")
    answer = holdup_script.guide(rig.realsense, {"tile": "", "laying": ""}, station, 1, text=text)
    holdup_script.close_windows()
    return answer


def table_of(rig):
    return rig.realsense if rig.realsense is not None else rig.mat


def save_before_after(table, station, before, after, folder):
    """The two RealSense frames around the station, side by side: nothing else can show a back that moved."""
    u, v = layout.table_to_pixel(layout.tile_corners(*station["xy"], 0.0), table.intrinsics, table.T_cam_to_base).mean(axis=0)
    h, w = before.shape[:2]
    x0, x1, y0, y1 = max(int(u) - CROP_PX, 0), min(int(u) + CROP_PX, w), max(int(v) - CROP_PX, 0), min(int(v) + CROP_PX, h)
    cv2.imwrite(str(Path(folder) / "table_before_after.jpg"), np.hstack([before[y0:y1, x0:x1], after[y0:y1, x0:x1]]))


def ask_outcome(ask, held, close_mm, photo):
    """One key for how the attempt went; None if the person quit. Clean needs the jaws to have read a tile."""
    keys = "  ".join(f"{k} = {v}" for k, v in wall_test.OUTCOMES.items())
    drain_stdin()           # keys pressed during the pickup aren't answers
    close = "no close logged" if close_mm is None else f"closed {close_mm:.0f}mm above its level"
    print(f"Jaws read {'a tile' if held else 'nothing'}, {close}." + (f" Base photo: {photo}" if photo else ""))
    while True:
        answer = ask(f"How did it go? {keys}  (q = quit): ").strip().lower()
        if answer == "q":
            return None
        if answer == "c" and not held:
            print("The jaws read nothing, so that isn't clean: which was it?")
        elif answer in wall_test.OUTCOMES:
            return wall_test.OUTCOMES[answer]


def discard(rig, pose):
    """Into the container, as the bench lets go there: a drawn tile isn't put back on the wall in this test."""
    rig.arm.rest()
    rig.arm.go_to_pose(pose)
    rig.arm.set_gripper(config.GRIPPER_RELEASE)
    rig.arm.rest()


def attempt(rig, rng, station, case, ask, ctx, layer=None, **extra):
    """One pick at the station. layer: a cycle's top or bottom. Returns "ok", "fix" (not clean), "stop" (out of reach) or "quit"."""
    session, table = rig.session, table_of(rig)
    settings = wall_test.case_of(case, layer)
    layer, raise_m = settings["layer"], wall_test.HOVER_RAISE_M[settings["layer"]]
    before = table.snapshot()
    while True:
        pick = rig.pick(f"{station['id']}_{case}_{layer}", station, raise_m=raise_m)
        if pick.get("unreachable"):
            session.event("wall_unreachable", station=station["id"], case=case, layer=layer)
            print(f"{station['id']} is out of the hover's reach at this height: stopping this station.")
            return "stop"
        if pick["run"]["stop_reason"] != "camera" or pick["holding"]:
            break
        session.event("wall_void", station=station["id"], case=case, layer=layer, attempt_dir=pick["attempt"].name, why="wrist camera lost")
        from mahjong import hardware
        rig.wrist = hardware.reopen_wrist_camera(rig.wrist)
        if ask("The wrist camera dropped out and was reopened: this try doesn't count. Check nothing moved. Enter = run it again, q = quit: ").strip().lower() == "q":
            return "quit"
    run, hover = pick["run"], pick["hover"]
    held = bool(pick["holding"])
    photo, problem = None, ""
    if held:
        frames, _, problem = rig.show(rng)
        if frames:
            photo = pick["attempt"] / "base.jpg"
            cv2.imwrite(str(photo), privacy.blur_faces(frames[0])[0])
        rig.arm.rest()
        if rig.arm.holding()[0]:
            discard(rig, ctx["discard_pose"])
    save_before_after(table, station, before, table.snapshot(), pick["attempt"])
    close = wall_test.above_own_level_m(run.get("grasp_clearance_m"), layer)
    if problem:
        print(f"Stopped showing it: {problem}.")
    ctx["n"] += 1
    print(f"[{ctx['n']}] {station['id']} {case} {layer}")
    outcome = ask_outcome(ask, held, None if close is None else close * 1000, photo)
    session.event("wall_attempt", n=ctx["n"], station=station["id"], station_xy=station["xy"], case=case, layer=layer, face=settings["face"],
                  behind=settings["behind"], policy=rig.policy.id, hover_raise_m=raise_m, hover_xy=hover["hover_xy"], hover_clearance_m=hover["clearance_m"],
                  hover_tilt_deg=hover["tilt_deg"], attempt_dir=pick["attempt"].name, stop_reason=run["stop_reason"], duration_s=run["duration_s"], held=held,
                  gripper_command=pick["gripper_command"], gripper_measured=pick["gripper_measured"], grasp_clearance_m=run.get("grasp_clearance_m"),
                  above_own_level_m=close, base_photo=photo.name if photo else None, show_problem=problem or None,
                  white_tiles_near=ctx["whites"][station["id"]], outcome=outcome, **extra)
    if outcome is None:
        return "quit"
    return "ok" if outcome == wall_test.CLEAN else "fix"


def run_cases(rig, rng, spots, args, cases, ask, ctx):
    stopped = set()
    for number, station_id, case in wall_test.schedule(list(spots), cases, args.trials):
        if (station_id, case) in stopped:
            continue
        station = spots[station_id]
        answer = prompt(rig, station, f"[{case}, try {number}/{args.trials}] Lay {wall_test.CASES[case]['lay'].format(id=station_id)}", ask)
        if answer == "q":
            return
        if answer == "s":
            rig.session.event("wall_skipped", station=station_id, case=case, round=number)
            continue
        result = attempt(rig, rng, station, case, ask, ctx, round=number)
        if result == "quit":
            return
        if result == "stop":
            stopped.add((station_id, case))


def run_cycle(rig, rng, station, draws, ask, ctx):
    plan = wall_test.cycle_plan(draws)
    stacks = sum(d["layer"] == "top" for d in plan)
    if prompt(rig, station, f"Lay a wall of {stacks + 1} stacks, two high, FACE DOWN, touching end to end, the first on {station['id']}, running away from the robot", ask) == "q":
        return
    for draw in plan:
        if draw["feed_before"]:
            if prompt(rig, station, f"Push the wall in until the next stack sits on {station['id']} (add one at the far end if you want)", ask) == "q":
                return
            rig.session.event("wall_feed", station=station["id"], feed_index=draw["feed_index"], before_draw=draw["draw"])
        result = attempt(rig, rng, station, wall_test.CYCLE, ask, ctx, layer=draw["layer"], draw=draw["draw"], feed_index=draw["feed_index"],
                         draws_since_feed=draw["draws_since_feed"])
        if result in ("quit", "stop"):
            return
        if result == "fix":
            answer = prompt(rig, station, "Not clean: put the wall right so the next draw is as planned (take a missed tile off, square the stack)", ask)
            rig.session.event("wall_put_right", station=station["id"], draw=draw["draw"])
            if answer == "q":
                return


def push_check(args, station, ask):
    """No arm. How far a pushed row strays, read off a face-up lead tile, against scans of the same row untouched."""
    if args.sim:
        from mahjong import sim
        table = sim.SimPerception(tiles=[("5C", *station["xy"])])
    else:
        from mahjong import hardware
        table = hardware.connect(["realsense"])["realsense"]
    print(f"Lay the wall two high against {station['id']}, the top tile of the lead stack FACE UP: that's the tile the scans can read, and the "
          "one a push is most likely to leave behind. The arm stays parked.")
    try:
        if ask("Enter once it's laid (q = quit): ").strip().lower() == "q":
            return None
        session = Session("wall_push", ask_setup=False, sim=args.sim, station=station["id"], station_xy=station["xy"])

        def read():
            tile = layout.nearest_tile(table.detect_tiles(), station["xy"], radius_m=wall_test.PUSH_RADIUS_M)
            return None if tile is None else [float(v) for v in tile["xy"]]

        baseline = [p for p in (read() for _ in range(wall_test.BASELINE_SCANS)) if p]
        pushes = []
        while len(pushes) < args.trials:
            if ask(f"[{len(pushes) + 1}/{args.trials}] Pull the row back about 3cm, push it in again, Enter (q = stop): ").strip().lower() == "q":
                break
            xy = read()
            session.event("wall_push", n=len(pushes) + 1, xy=xy)
            if xy is None:
                print(f"No tile read within {wall_test.PUSH_RADIUS_M * 1000:.0f}mm of the station: check the lead tile is face up, and try that push again.")
                continue
            pushes.append(xy)
        if pushes:
            result = wall_test.push_result(pushes, baseline, station["xy"])
            session.event("wall_push_summary", station=station["id"], **result)
            print(f"{result['pushes']} pushes: the worst strayed {result['worst_mm']}mm from where they usually land (limit {result['limit_mm']}mm), "
                  f"the detector's own wander {result['baseline_mm']}mm, {result['offset_from_station_mm']}mm off the station on average: "
                  + ("PASS." if result["passed"] else "FAIL: try a low end stop, or one-stack stations."))
        session.close()
        return session.dir
    finally:
        table.close()


def hover_check(rig, spots, ask):
    """The hover over a face-up tile on each station, no policy: the wrist view with the place a tile should be marked, the reach left."""
    table = table_of(rig)
    for station_id, station in spots.items():
        if prompt(rig, station, f"Lay one tile FACE UP on {station_id}", ask) == "q":
            return
        x, y = station["xy"]
        tile = layout.nearest_tile(table.detect_tiles(), station["xy"], radius_m=wall_test.PUSH_RADIUS_M)
        off = None if tile is None else float(np.hypot(tile["xy"][0] - x, tile["xy"][1] - y) * 1000)
        hx, hy = rig.correction.apply(x, y)
        frames, hover = wrist_check.look(rig.arm, rig.wrist, hx, hy, rig.policy.hover)
        fill = None
        if frames:
            shown = frames[-1].copy()
            fill = float(wrist_check.face_fill(shown))
            cv2.circle(shown, wrist_check.EXPECTED_PX, wrist_check.RADIUS_PX, (0, 255, 255), 2)
            cv2.imwrite(str(rig.session.dir / f"hover_{station_id}.jpg"), shown)
        rig.arm.rest()
        left = gamepad.reach_left_m(hx, hy)
        rig.session.event("wall_hover", station=station_id, station_xy=station["xy"], hover_xy=[hx, hy], tile_off_station_mm=off,
                          reachable=hover is not None, hover_tilt_deg=hover and hover["tilt_deg"], gamepad_reach_left_m=left, face_fill=fill)
        print(f"{station_id}: " + ("no reach at the hover" if hover is None else f"hover at ({hx:.3f}, {hy:+.3f}), tilt {hover['tilt_deg']}")
              + ("" if left is None else f", {left * 1000:.0f}mm of gamepad reach left")
              + ("; the detector doesn't read a tile there" if off is None else f"; the tile reads {off:.1f}mm from the station")
              + (f". Is it inside the circle? {rig.session.dir / f'hover_{station_id}.jpg'}" if frames else ". No wrist frame."))


def report(root=None):
    folders = wall_test.sessions(root)
    if not folders:
        return print("No real wall-test sessions yet.")
    everything = []
    for folder in folders:
        for e in (e for e in read_events(folder) if e["type"] == "wall_push_summary"):
            print(f"{folder.name}: push check at {e['station']}: worst {e['worst_mm']}mm of {e['pushes']} pushes "
                  f"(detector wander {e['baseline_mm']}mm), {'pass' if e['passed'] else 'FAIL'}")
        rows = wall_test.attempts(folder)
        if rows:
            everything += rows
            print(f"\n{folder.name}")
            print("\n".join(wall_test.table(wall_test.summarize(rows))))
    if everything:
        print(f"\nAll {len(folders)} session(s)")
        print("\n".join(wall_test.table(wall_test.summarize(everything))))


def check_cases(args, cases, stations):
    unknown = [c for c in cases if c not in wall_test.CASES and c != wall_test.CYCLE]
    if unknown:
        raise SystemExit(f"no case {', '.join(unknown)} (have: {', '.join(wall_test.CASES)}, {wall_test.CYCLE})")
    if wall_test.CYCLE in cases and (len(cases) > 1 or len(stations) > 1):
        raise SystemExit(f"{wall_test.CYCLE} runs alone, at one station")
    if args.trials < 1:
        raise SystemExit("--trials has to be at least 1")


def main(argv=None, ask=input):
    args = parse_args(argv)
    if args.report:
        return report()
    wanted = wall_test.parse_list(args.stations)
    cases = wall_test.parse_list(args.cases)
    known = {s["id"]: s for s in layout.load(args.layout)["spots"]}
    missing = [s for s in wanted if s not in known]
    if missing:
        raise SystemExit(f"layout {args.layout} has no station {', '.join(missing)} (has: {', '.join(known)})")
    spots = {s: known[s] for s in wanted}
    check_cases(args, cases, wanted)
    if args.push_check:
        if len(spots) != 1:
            raise SystemExit("--push-check is for one station at a time")
        return push_check(args, next(iter(spots.values())), ask)
    privacy.require()           # the weights that blur faces in the base camera's photos: before any hardware
    poses = holdup_arm.pose_names()
    if not poses:
        raise SystemExit(f"no {holdup_arm.POSE_PREFIX}* poses in {config.CAMERA_POSES}: teach some with scripts/teleop_view.py")
    if not args.hover_check and not config.DISCARD_POSE.exists():
        raise SystemExit("No discard pose: teach one with bench.py's teach-discard")
    for name in poses[:1]:
        camera_poses.get(name)
    rig = holdup_script.Rig(args, list(spots.values()), poses[:1])
    rng = holdup_arm.rng_for(0)
    ctx = {"n": 0}
    try:
        if args.hover_check:
            rig.open_session("wall_hover", stations=wanted)
            rig.arm.rest()
            hover_check(rig, spots, ask)
            return rig.session.dir
        conditions = holdup.ask_lighting(ask)
        ctx["whites"] = {s: ask(f"White face-up tiles within about 8cm of {s}, not counting the target? [y/N]: ").strip().lower().startswith("y") for s in wanted}
        ctx["discard_pose"] = json.loads(config.DISCARD_POSE.read_text())["pose"]
        rig.open_session("wall", stations=wanted, cases=cases, trials=args.trials)
        rig.session.conditions = conditions
        rig.session.event("wall_conditions", conditions=conditions, white_tiles_near=ctx["whites"], show_pose=poses[0])
        rig.arm.rest()
        if cases == [wall_test.CYCLE]:
            run_cycle(rig, rng, next(iter(spots.values())), args.trials, ask, ctx)
        else:
            run_cases(rig, rng, spots, args, cases, ask, ctx)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        folder = rig.session.dir if rig.session else None
        rig.close()
        if rig.session:
            rows = wall_test.attempts(folder) if ctx["n"] else []
            groups = wall_test.summarize(rows)
            if groups:
                print("\n".join(wall_test.table(groups)))
                (folder / "wall_summary.json").write_text(json.dumps(groups, indent=2) + "\n")
            rig.session.event("wall_done", attempts=len(rows))
            rig.session.close()
    return folder


if __name__ == "__main__":
    main()
