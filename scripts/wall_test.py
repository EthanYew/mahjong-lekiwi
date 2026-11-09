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
for a detection; a top tile's hover is one tile higher. Before each attempt a window shows the station, says what to lay and draws it from
the side; the terminal has the same words and picture.

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
    p.add_argument("--hover-back-mm", type=float, default=0.0,
                   help=f"start every pick this many mm nearer the robot than the station, up to {wall_test.MAX_HOVER_BACK_MM:.0f} (negative: further out). "
                        "To test whether a pick that lands too far out is an aiming error (default %(default)s)")
    p.add_argument("--top-back-mm", type=float, default=0.0,
                   help="the same, for top tiles only: bottom tiles keep the regular hover. At F, 10 worked for top tiles and broke bottom ones, 2026-11-09 (default %(default)s)")
    p.add_argument("--exclude-session", default=None, metavar="SESSION", help="leave this session out of --report, because it wasn't the test it was logged as (needs --why); no hardware")
    p.add_argument("--why", default=None, help="with --exclude-session: the reason, kept with the session")
    p.add_argument("--layout", default=wall_test.LAYOUT, help="the layout holding the stations (default %(default)s)")
    p.add_argument("--push-check", action="store_true", help="no arm: scan a face-up lead tile after each push, and say how far pushes stray")
    p.add_argument("--hover-check", action="store_true", help="no policy: hover over a face-up tile on each station and save the wrist view")
    p.add_argument("--report", action="store_true", help="score every real wall-test session so far; no hardware")
    p.add_argument("--policy", default=None, help="policy id (default: the registry's default)")
    p.add_argument("--sim", action="store_true", help="simulated arm, cameras and policy: no hardware")
    p.set_defaults(empty=False, no_view=False, hover_lower_mm=None, frames=1, settle_s=1.0, gap_s=0.25, jitter_deg=0.0)       # what holdup_arm's Rig reads
    return p.parse_args(argv)


def draw_picture(frame, rows):
    """The step's side view in the window's bottom left: white and green tiles, the one the arm goes for outlined in yellow."""
    tile_w, tile_h, gap, left = 46, 24, 3, 76
    h = frame.shape[0]
    top = h - 40 - len(rows) * (tile_h + gap)
    cv2.rectangle(frame, (0, top - 26), (max(left + 4 * (tile_w + gap), 310), h - 30), (0, 0, 0), -1)
    cv2.putText(frame, "side view, yellow = the tile the arm goes for", (6, top - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "robot >", (6, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            x, y = left + c * (tile_w + gap), top + r * (tile_h + gap)
            colour = (255, 255, 255) if "W" in cell else (60, 150, 60)
            cv2.rectangle(frame, (x, y), (x + tile_w, y + tile_h), colour, -1)
            cv2.rectangle(frame, (x, y), (x + tile_w, y + tile_h), (0, 255, 255) if cell.startswith("[") else (130, 130, 130), 3 if cell.startswith("[") else 1)
    return frame


def prompt(rig, station, step, ask):
    """Show what to lay, on the live RealSense window (or the terminal). Returns "", "s" or "q"."""
    text = wall_test.step_text(step)
    if rig.realsense is None:
        return ask(f"{text}\nEnter when done (s = skip, q = quit): ").strip().lower()
    print(f"{text}\n  (in the RealSense window: Space/Enter when done, s skip, q quit)")
    answer = holdup_script.guide(rig.realsense, {"tile": "", "laying": ""}, station, None, text="\n".join(step["lines"]),
                                 overlay=lambda frame: draw_picture(frame, step["picture"]))
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
    drain_stdin()           # keys pressed during the pickup aren't answers
    close = "no close logged" if close_mm is None else f"closed {close_mm:.0f}mm above its level"
    print(f"Jaws read {'a tile' if held else 'nothing'}, {close}." + (f" Base photo: {photo}" if photo else ""))
    print("Press the key for the WORST thing that happened:")
    for key, what in wall_test.OUTCOMES.items():
        print(f"  {key}  {what}" + ("  (picked the tile, nothing else moved, no help)" if what == wall_test.CLEAN else ""))
    while True:
        answer = ask("Key, then Enter (q = quit): ").strip().lower()
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
    back_m = ctx["back_m"] + (ctx["top_back_m"] if layer == "top" else 0.0)
    before = table.snapshot()
    while True:
        pick = rig.pick(f"{station['id']}_{case}_{layer}", station, raise_m=raise_m, back_m=back_m)
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
    print(f"\nTry {ctx['n']}: station {station['id']}, {case}" + (f" ({layer} tile)" if case == wall_test.CYCLE else ""))
    outcome = ask_outcome(ask, held, None if close is None else close * 1000, photo)
    session.event("wall_attempt", n=ctx["n"], station=station["id"], station_xy=station["xy"], case=case, layer=layer, face=settings["face"],
                  behind=settings["behind"], policy=rig.policy.id, hover_raise_m=raise_m, hover_back_m=back_m, hover_xy=hover["hover_xy"], hover_clearance_m=hover["clearance_m"],
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
        answer = prompt(rig, station, wall_test.lay_step(case, station_id, f"{case}, try {number}/{args.trials}"), ask)
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
    if prompt(rig, station, wall_test.wall_step(station["id"], stacks + 1), ask) == "q":
        return
    already_fed = False
    for draw in plan:
        if draw["feed_before"] and not already_fed:
            if prompt(rig, station, wall_test.feed_step(station["id"]), ask) == "q":
                return
            rig.session.event("wall_feed", station=station["id"], feed_index=draw["feed_index"], before_draw=draw["draw"])
        already_fed = False
        result = attempt(rig, rng, station, wall_test.CYCLE, ask, ctx, layer=draw["layer"], draw=draw["draw"], feed_index=draw["feed_index"],
                         draws_since_feed=draw["draws_since_feed"])
        if result in ("quit", "stop"):
            return
        if result == "fix" and draw is not plan[-1]:          # after the last draw there's nothing to put right for
            step = wall_test.put_right_step(station["id"]) if draw["layer"] == "top" else wall_test.clear_and_feed_step(station["id"])
            if prompt(rig, station, step, ask) == "q":
                return
            rig.session.event("wall_put_right", station=station["id"], draw=draw["draw"])
            if draw["layer"] == "bottom":          # one step for the person: what's left comes off and the next stack comes on
                already_fed = True
                rig.session.event("wall_feed", station=station["id"], feed_index=draw["feed_index"] + 1, before_draw=draw["draw"] + 1)


def push_check(args, station, ask):
    """No arm. How far a pushed row strays, read off a face-up lead tile, against scans of the same row untouched."""
    if args.sim:
        from mahjong import sim
        table = sim.SimPerception(tiles=[("5C", *station["xy"])])
    else:
        from mahjong import hardware
        table = hardware.connect(["realsense"])["realsense"]
    print(wall_test.step_text(wall_test.push_step(station["id"])))
    print("\nThe white tile is the one the camera can read, and the one a push is most likely to leave behind.")
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
            if ask(f"[push {len(pushes) + 1}/{args.trials}] Pull the whole wall back about 3cm, push it in until the white tile is on {station['id']} again, then Enter (q = stop): ").strip().lower() == "q":
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
        if prompt(rig, station, wall_test.hover_step(station_id), ask) == "q":
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
        why = wall_test.excluded(folder)
        if why:
            print(f"{folder.name}: left out ({why})")
            continue
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
    if abs(args.hover_back_mm) > wall_test.MAX_HOVER_BACK_MM:
        raise SystemExit(f"--hover-back-mm is at most {wall_test.MAX_HOVER_BACK_MM:.0f}mm either way")
    if args.hover_back_mm and (args.push_check or args.hover_check):
        raise SystemExit("--hover-back-mm is for runs that pick tiles, not --push-check or --hover-check")
    if abs(args.top_back_mm) > wall_test.MAX_HOVER_BACK_MM:
        raise SystemExit(f"--top-back-mm is at most {wall_test.MAX_HOVER_BACK_MM:.0f}mm either way")
    if args.top_back_mm and args.hover_back_mm:
        raise SystemExit("--top-back-mm or --hover-back-mm, not both: a top tile's shift would be their sum")
    if args.top_back_mm and (args.push_check or args.hover_check or not {"top", wall_test.CYCLE} & set(cases)):
        raise SystemExit("--top-back-mm needs a run that picks top tiles: --cases top or cycle")


def exclude_session(name, why):
    folder = config.SESSIONS / Path(name).name
    if "_wall" not in folder.name or not (folder / "events.jsonl").exists():
        raise SystemExit(f"no wall-test session {folder.name} in {config.SESSIONS}")
    if not why:
        raise SystemExit("--exclude-session needs --why: the reason stays with the session")
    count = len(wall_test.attempts(folder))
    wall_test.exclude(folder, why)
    print(f"{folder.name}: {count} attempt(s) left out of --report ({why}). Its files are untouched.")
    return folder


def main(argv=None, ask=input):
    args = parse_args(argv)
    if args.report:
        return report()
    if args.exclude_session:
        return exclude_session(args.exclude_session, args.why)
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
    ctx = {"n": 0, "back_m": args.hover_back_mm / 1000, "top_back_m": args.top_back_mm / 1000}
    try:
        if args.hover_check:
            rig.open_session("wall_hover", stations=wanted)
            rig.arm.rest()
            hover_check(rig, spots, ask)
            return rig.session.dir
        conditions = holdup.ask_lighting(ask)
        ctx["whites"] = {s: ask(f"Apart from the tile being tested, is any tile lying WHITE side up within about 8cm of station {s}? [y/N]: ").strip().lower().startswith("y") for s in wanted}
        ctx["discard_pose"] = json.loads(config.DISCARD_POSE.read_text())["pose"]
        rig.open_session("wall", stations=wanted, cases=cases, trials=args.trials)
        rig.session.conditions = conditions
        rig.session.event("wall_conditions", conditions=conditions, white_tiles_near=ctx["whites"], show_pose=poses[0], hover_back_m=ctx["back_m"], top_back_m=ctx["top_back_m"])
        if ctx["back_m"]:
            print(f"Every pick starts {args.hover_back_mm:g}mm nearer the robot than the station.")
        if ctx["top_back_m"]:
            print(f"Top tiles start {args.top_back_mm:g}mm nearer the robot than the station; bottom tiles keep the regular hover.")
        print("Leave the mat alone until the outcome keys appear: the photo after each try is taken just before them, and a re-laid tile hides what happened.")
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
