"""Turn a played game's session folder into a video to show it: the desk from above as the hand goes, the gripper camera's
clip of each pickup cut in where it happened, and on every frame what the robot was doing and how often a person stepped in.
A face-down hand also shows, after each draw from the wall, the base camera's photos of the drawn tile with the name the reader
gave it, and before the ending, the hand's 13 and the winning draw as the base camera saw them: a viewer can check each read
instead of taking it on trust.

    python scripts/game_video.py                                   # the latest play session, as a story
    python scripts/game_video.py sessions/20261104_183142_play --crop mat
    python scripts/game_video.py latest --style timelapse --fps 12

The desk comes from the session's own photos: `desk.mp4` if the game ran with `--record-desk` (the arm seen moving), else the
first frame of every scan (the arm is always at rest in those, so the board changes and the arm doesn't). The gripper clips are
each pickup's `wrist.mp4`. The base camera's photos are the `holdup_shot` photos play_face_down.py saves, cut down to each
pose's tile box (data/camera_poses.json), the square the reader reads: the room behind the tile stays out, faces included, since
the photos' own blur has been seen to miss half a face (20261119_142253_play_fd). Faces are blurred on every desk frame (mahjong/privacy.py); `--crop mat` also cuts the frame down to
where tiles were seen, which leaves most of the room out. The blur lowers the chance of a face, it doesn't rule it out: watch the
video before posting it.

The counts on screen are the session report's (mahjong/interventions.py): "questions" a person answered without touching
anything, "hands-on" a person touching a tile, the arm or the mat. The hand, its turns and pickups are read through
mahjong/play_log.py, the same for face-up and face-down play, and the ending says whether the win is verified as the session
report does (play_log.verdict). A video says which session it is from, so every claim made with it traces back to one. Never
connects to the arm.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mahjong import base_cam, config, desk_video, face_down_play, game, interventions, play_log, privacy
from mahjong.session import read_events

OUT_FPS = 30
SCAN_FPS, DESK_FPS = 8, 30          # desk frames a second of video: a scan per 1/8s; 2fps desk video at 15x real time
CLIP_SPEED = 4                      # a gripper clip plays this much faster than it happened
MAX_SIZE = (1280, 1080)             # the video fits in this, keeping the desk's shape (a square crop: 1080x1080)
CROP_MARGIN_PX = 60                 # around every tile the scans saw, in the camera's pixels
TITLE_S, END_S = 3.0, 5.0
END_S_PER_LINE = 0.7                # a long ending card stays up longer
PIP_SHARE = 0.38                    # a gripper clip's width, as a share of the frame's
READ_S = 2.5                        # a drawn tile's base-camera photos stay up this long after its gripper clip
HAND_S = 7.0                        # the hand's photos, before the ending
READ_CELL_PX = 240                  # a tile photo's side in a read's inset
DRAWN_COLOUR = (0, 215, 255)        # the winning draw's outline in the hand's photos (BGR: amber)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FOLDER_TIME = re.compile(r"^\d+_(\d{6})_")


def latest_session():
    folders = sorted(p for p in config.SESSIONS.glob("*_play*") if (p / "events.jsonl").exists())
    if not folders:
        raise SystemExit(f"no play session in {config.SESSIONS}")
    return folders[-1]


def seconds(stamp):
    return desk_video.wall_time(stamp)


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


# --- what the session holds ---

def scan_frames(session_dir, events):
    """[(time, path)] of the first frame of each scan, in order."""
    out = []
    for e in events:
        if e["type"] == "detections" and e.get("frames"):
            path = session_dir / e["frames"] / "f0.jpg"
            if path.exists():
                out.append((seconds(e["time"]), path))
    return out


def desk_frames(session_dir):
    """[(time, frame index)] of the session's desk.mp4, or [] without one."""
    times = session_dir / desk_video.TIMES
    if not (session_dir / desk_video.VIDEO).exists() or not times.exists():
        return []
    return [(seconds(t), i) for i, t in enumerate(json.loads(times.read_text())["times"])]


def clip_start(session_dir, attempt_dir):
    """When a pickup's folder was made (its name has the time; the session's name the day)."""
    m = FOLDER_TIME.match(attempt_dir)
    day = datetime.strptime(session_dir.name[:15], "%Y%m%d_%H%M%S")
    t = datetime.combine(day.date(), datetime.strptime(m.group(1), "%H%M%S").time())
    if t < day - timedelta(minutes=1):      # past midnight
        t += timedelta(days=1)
    return t.timestamp()


def clips(session_dir, events):
    """[{time, path, tile, success, step, spot}] for each pickup with a gripper video, in order (play_log.pickups: face up
    and face down alike)."""
    out = []
    for p in play_log.pickups(events):
        if FOLDER_TIME.match(p.get("attempt_dir") or ""):
            path = session_dir / p["attempt_dir"] / "wrist.mp4"
            if path.exists():
                out.append({"time": clip_start(session_dir, p["attempt_dir"]), "path": path, "tile": p["tile"],
                            "success": p["held"], "step": p["step"], "spot": p["spot"]})
    return sorted(out, key=lambda c: c["time"])


def clip_label(c, speed):
    """What a gripper clip shows: the step of a face-down turn where it has one, the tile where it's known."""
    tile, spot = c.get("tile"), c.get("spot")
    what = {"register": f"reading {spot}", "draw": "drawing from the wall",
            "pick_from_hand": f"taking {tile or 'the discard'} out of {spot}",
            "pick_from_zone": f"{tile or 'the drawn tile'} off the drop zone"}.get(c.get("step"), tile or spot or "a tile")
    return f"gripper camera, {speed}x speed: {what}, " + ("held" if c["success"] else "missed")


# --- the base camera's reads (face-down play) ---

def tile_crops(session_dir, shot, boxes, size=READ_CELL_PX):
    """A hold's photos cut down to their pose's tile box (base_cam.tile_frame: what the reader reads), one per pose: a second
    look at the same poses adds nothing new to show. A photo at a pose without a box is left out, since the whole frame shows
    the room. [] if none is left."""
    out, seen = [], set()
    for f, pose in zip(shot["files"], shot.get("frame_poses") or [None] * len(shot["files"])):
        if pose in seen or pose not in boxes:
            continue
        image = cv2.imread(str(session_dir / f))
        if image is not None:
            seen.add(pose)
            out.append(base_cam.tile_frame(image, boxes[pose], size=size))
    return out


def base_reads(session_dir, events, reads=None, boxes=None):
    """[{time, shot, turn, where, name, guess, confidence, by, truth, crops}] for every hold of a face-down hand with photos to
    show, in order. where: the slot read registering the hand, "draw t<N>" for a draw. name: what the hand used (the reader's,
    or a person's); truth: the name a person's review gave the photos since (reads: face_down_play.scored_reads), or None."""
    boxes = base_cam.tile_boxes() if boxes is None else boxes
    truths = {r["shot"]: r["truth"] for r in reads or []}
    discarded = {e["shot"] for e in events if e["type"] == "holdup_shot_discarded"}
    out = []
    for e in events:
        if e["type"] != "holdup_shot" or e.get("shot") in discarded:
            continue
        crops = tile_crops(Path(session_dir), e, boxes)
        if crops:
            out.append({"time": seconds(e["time"]), "shot": e.get("shot"), "turn": e.get("turn"),
                        "where": str(e.get("spot") or ""), "name": e["tile"], "guess": e.get("guess"),
                        "confidence": e.get("guess_confidence"), "by": e.get("filed_by", "person"),
                        "truth": truths.get(e.get("shot")), "crops": crops})
    return out


def is_draw(read):
    return read["where"].startswith("draw")


def read_label(r):
    """What a drawn tile's photos show: the name the hand went on and who gave it, and a review's name where it differs."""
    sure = f" ({r['confidence']:.0%} sure)" if r.get("confidence") is not None else ""
    if r["by"] == "reader":
        said = f"base camera: the reader read {r['name']}{sure}"
    elif r["guess"] == r["name"]:
        said = f"base camera: the reader read {r['name']}{sure}, a person confirmed it"
    elif r["guess"]:
        said = f"base camera: the reader read {r['guess']}{sure}, a person named it {r['name']}"
    else:
        said = f"base camera: a person named it {r['name']}"
    if r.get("truth") and r["truth"] != r["name"]:
        said += f"; reviewed since: {r['truth']}"
    return said


def cut_ins(shown, drawn):
    """The gripper clips and the draws' base-camera photos in the order they go in: a draw's photos after its clip (the clip's
    folder is made before the pickup, the photos after it) and before the next clip, the photos first on the same second."""
    return sorted([{**c, "kind": "clip"} for c in shown] + [{**r, "kind": "read"} for r in drawn],
                  key=lambda item: (item["time"], item["kind"] == "clip"))


def tile_grid(crops):
    """A hold's tile photos side by side, two to a row from three on (black where a row is short)."""
    side = crops[0].shape[0]
    cols = len(crops) if len(crops) <= 2 else 2
    grid = np.zeros((-(-len(crops) // cols) * side, cols * side, 3), np.uint8)
    for k, crop in enumerate(crops):
        r, c = divmod(k, cols)
        grid[r * side:(r + 1) * side, c * side:(c + 1) * side] = cv2.resize(crop, (side, side))
    return grid


def final_hand(events, reads):
    """The hand at the end, each tile with the hold that read it: [{name, read (a base_reads row, or None without photos),
    drawn}], in slot order, the winning draw last. A slot holds the tile read registering the hand until a turn throws it,
    then that turn's draw. [] when the log has no hand at its end."""
    log = play_log.hand(events)
    end = log["end"] or {}
    hand = end.get("hand")
    if not hand:
        return []
    if isinstance(hand, dict):
        slots, names = list(hand), list(hand.values())
    else:
        slots, names = list(end.get("slots") or log["slots"] or []), list(hand)
    by_shot = {r["shot"]: r for r in reads}
    draws, in_slot = {}, {}
    for e in events:            # in order: a draw's hold comes before its turn's choice
        if e["type"] == "holdup_shot" and e.get("shot") in by_shot:
            r = by_shot[e["shot"]]
            if is_draw(r):
                draws[r["turn"]] = r
            else:
                in_slot[r["where"]] = r
        elif e["type"] in play_log.CHOICES:
            c = play_log.choice(e)
            if c["slot"] is not None:
                in_slot[c["slot"]] = draws.get(c["turn"])
    if len(slots) != len(names):
        slots = [None] * len(names)
    out = [{"name": n, "read": in_slot.get(s), "drawn": False} for s, n in zip(slots, names)]
    win = log["win"]
    if win:
        out.append({"name": win["drawn"], "read": draws.get(win["turn"]), "drawn": True})
    return out


def checked_reads(events):
    """Face-down play: the reader's names a person checked (the 13 in the hand check, each draw confirmed). Not questions in
    the grade (pressing Enter on a read isn't one), but a person looked at every one."""
    hand = sum(len(e.get("hand", {})) for e in events if e["type"] == "fd_hand_confirmed" and e.get("by") == "person")
    return hand + sum(1 for e in events if e["type"] == "crude_read" and e.get("turn") and e.get("by") != "reader")


def crop_box(events, shape, margin=CROP_MARGIN_PX):
    """(x0, y0, x1, y1) around every tile the scans saw, down to the frame's bottom edge (where the arm comes in), widened
    to a square where the frame allows (what LinkedIn shows best), or None without any tiles."""
    pixels = [t["pixel"] for e in events if e["type"] == "detections" for t in e.get("tiles", []) if t.get("pixel")]
    if not pixels:
        return None
    xs, ys = [p[0] for p in pixels], [p[1] for p in pixels]
    height, width = shape[:2]
    x0, x1 = max(0, int(min(xs)) - margin), min(width, int(max(xs)) + margin)
    y0 = max(0, int(min(ys)) - margin)
    side = height - y0
    if x1 - x0 < side:                          # wider, about the tiles' middle, kept inside the frame
        middle = (x0 + x1) // 2
        x0 = min(max(0, middle - side // 2), max(0, width - side))
        x1 = min(width, x0 + side)
    elif x1 - x0 > side:                        # taller, upwards
        y0 = max(0, height - (x1 - x0))
    return (x0, y0, x1, height)


class Story:
    """What to write on a frame at a given time: the turn, the hand, and the counts so far (play_log.hand: face up and face
    down alike). The ending says whether the win stands (play_log.verdict)."""

    def __init__(self, session_dir, events, reads=None):
        self.name = session_dir.name
        self.found = interventions.derive(events)
        self.attempts = play_log.pickups(events)
        self.hand = play_log.hand(events)
        self.checked = checked_reads(events)
        self.end = self.hand["end"]
        if reads is None and self.hand["kind"] == face_down_play.PLAY_KIND:
            reads = face_down_play.scored_reads(session_dir)        # whether a person has reviewed the reader's names since
        self.reads = reads or []
        self.verdict = play_log.verdict(self.hand, reads)
        self.policy = next((e.get("policy") for e in events if e["type"] == "bench_setup"), None) \
            or next((e.get("policy") for e in events if e["type"] == "session_start"), None)
        times = [seconds(e["time"]) for e in events if e.get("time")]
        self.start, self.stop = (min(times), max(times)) if times else (0, 0)

    def counts(self, t=None):
        """{picked, attempts, turns, questions, rescues, resets, hands_on} up to time t (the whole game without one): turns
        whose draw was named and played, the win's included; the session report's classes (a reset is tiles put back, a
        rescue a person fixing what went wrong)."""
        def by(time):
            return t is None or (time is not None and seconds(time) <= t)
        found = [i for i in self.found if i.get("time") and by(i["time"])]
        rescues = sum(i["class"] == interventions.RESCUE for i in found)
        resets = sum(i["class"] == interventions.RESET for i in found)
        win = self.hand["win"]
        return {"picked": sum(a["held"] for a in self.attempts if by(a["time"])),
                "attempts": sum(1 for a in self.attempts if by(a["time"])),
                "turns": sum(1 for r in self.hand["turns"] if by(r["time"])) + bool(win and by(win["time"])),
                "questions": sum(i["class"] == interventions.ASK for i in found),
                "rescues": rescues, "resets": resets, "hands_on": rescues + resets}

    def _turn_at(self, t):
        """The latest turn (or the win) whose draw was named by time t, and whether its hand has changed yet."""
        rows = [r for r in self.hand["turns"] if r["time"] and seconds(r["time"]) <= t]
        win = self.hand["win"]
        if win and seconds(win["time"]) <= t:
            return {**win, "win": True}, True
        if not rows:
            return None, False
        done = rows[-1]["done"]
        return rows[-1], done is not None and seconds(done) <= t

    def doing(self, t):
        row, done = self._turn_at(t)
        if row is None:
            read = [r for r in self.hand["register"] if seconds(r["time"]) <= t]
            if read and len(read) < len(self.hand["slots"] or read):
                return f"Reading the hand: {len(read)} of {len(self.hand['slots'] or read)} tiles"
            return "Reading the hand"
        drew = f"drew {row['drawn']}{read_text(row.get('read'))}" if row.get("drawn") else ""
        if row.get("win"):
            return f"Turn {row['turn']}: {drew}: the hand is complete"
        if done:
            return f"Turn {row['turn']}: discarded {row['discard']}"
        where = f" from {row['slot']}" if row.get("slot") else ""
        return f"Turn {row['turn']}: " + (f"{drew}, " if drew else "") + f"discarding {row['discard']}{where}"

    def hand_lines(self, t):
        """['Hand: ...', 'Discards: ...'] as they stood at time t: the hand as read, then after each turn."""
        row, done = self._turn_at(t)
        if row is None:
            read = [r["name"] for r in self.hand["register"] if seconds(r["time"]) <= t and r["name"]]
            start = self.hand["start"]
            names = start["hand"] if start and seconds(start["time"]) <= t else read
            return [f"Hand: {play_log.hand_line(names)}"] if names else []
        after = done and not row.get("win")
        names = row.get("after") if after else row.get("before")
        discards = (row.get("discards") if after else row.get("seen")) or []
        out =[f"Hand: {play_log.hand_line(names)}" + (f"  + {row['drawn']}" if row.get("win") else "")] if names else []
        return out + ([f"Discards: {' '.join(discards)}"] if discards else [])

    def lines(self, t):
        c = self.counts(t)
        clock = datetime.fromtimestamp(t).strftime("%H:%M:%S")
        total = (self.end or {}).get("turns")
        turns = f"turn {c['turns']}" + (f" of {total}" if total else "")
        return (f"{self.name}   {clock}", self.doing(t), *self.hand_lines(t),
                f"picked {c['picked']} of {c['attempts']}   {turns}   person: {c['questions']} questions answered, "
                f"{c['hands_on']} hands-on")

    def title(self, base_camera=False):
        day = datetime.fromtimestamp(self.start).strftime("%Y-%m-%d")
        down = self.hand["kind"] == face_down_play.PLAY_KIND
        return ["Robot arm playing a hand of mahjong" + (", tiles face down" if down else ""),
                f"session {self.name}  ({day})", f"policy {self.policy}" if self.policy else "",
                "desk camera from above, gripper camera inset",
                "then the base camera's photos of each drawn tile, as the reader saw it" if base_camera else ""]

    def ending(self):
        c, minutes = self.counts(), (self.stop - self.start) / 60
        out = []
        win = self.hand["win"]
        if self.end:
            result = self.end.get("result", "?")
            out.append(f"Result: {result}" + (f" in {self.end['turns']} turns" if self.end.get("turns") is not None else ""))
            if win and win.get("before"):
                out.append(f"Winning hand: {play_log.melds_line(win['before'] + [win['drawn']])}  (drew {win['drawn']})")
            if self.end.get("grade"):
                out.append(f"Graded: {self.end['grade']}")
        else:
            out.append("The hand didn't finish in this session")
        out += [f"{minutes:.0f} minutes, {c['picked']} of {c['attempts']} pickups held the tile",
                f"A person answered {c['questions']} questions",
                f"and stepped in by hand {plural(c['hands_on'], 'time')} ({plural(c['rescues'], 'rescue')}, "
                f"{plural(c['resets'], 'reset')} of the mat)"]
        if self.checked:
            out.append(f"A person checked {self.checked} of the reader's names")
        if self.verdict:
            out += self.verdict["lines"]
        out.append(f"session {self.name}")
        return out


def read_text(read):
    """' (reader 95%)' for a name the reader gave unchecked, ' (checked)' for one a person confirmed or gave."""
    if not read:
        return ""
    if read.get("by") == "reader":
        return f" (reader {read['confidence']:.0%})" if read.get("confidence") is not None else " (reader)"
    return " (checked)"


# --- drawing ---

def text_band(frame, lines, top):
    """Lines of white text on a dark band across the top or bottom of the frame (in place)."""
    lines = [s for s in lines if s]
    scale = frame.shape[1] / 1280
    line_h = int(34 * scale)
    band = line_h * len(lines) + int(14 * scale)
    y0 = 0 if top else frame.shape[0] - band
    frame[y0:y0 + band] = (frame[y0:y0 + band] * 0.35).astype(np.uint8)
    for k, s in enumerate(lines):
        y = y0 + int(10 * scale) + line_h * (k + 1) - int(8 * scale)
        size = fitted(s, 0.85 * scale, frame.shape[1] - int(32 * scale))
        cv2.putText(frame, s, (int(16 * scale), y), FONT, size, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)


def fitted(text, scale, width):
    """The font scale, at most `scale`, at which `text` fits in `width` pixels (a long hand or discard row on a square crop)."""
    (w, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    return scale * min(1.0, width / max(w, 1))


def card(size, lines):
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    scale = size[0] / 1280
    lines = [s for s in lines if s]
    step = min(int(56 * scale), (size[1] - int(40 * scale)) // max(len(lines), 1))      # a long ending still fits the height
    y = (size[1] - step * (len(lines) - 1)) // 2
    for k, s in enumerate(lines):
        big = fitted(s, (1.3 if k == 0 else 0.9) * scale * min(1.0, step / (56 * scale)), size[0] - int(40 * scale))
        (w, _), _ = cv2.getTextSize(s, FONT, big, 2)
        cv2.putText(frame, s, ((size[0] - w) // 2, y), FONT, big, (255, 255, 255), 2, cv2.LINE_AA)
        y += step
    return frame


def inset(frame, clip_frame, label):
    """The gripper frame in the top right corner of `frame` (a copy), with a label under it."""
    out = frame.copy()
    width = int(frame.shape[1] * PIP_SHARE)
    height = int(clip_frame.shape[0] * width / clip_frame.shape[1])
    small = cv2.resize(clip_frame, (width, height), interpolation=cv2.INTER_AREA)
    x, y = frame.shape[1] - width - 12, 70
    cv2.rectangle(out, (x - 3, y - 3), (x + width + 2, y + height + 2), (255, 255, 255), 3)
    out[y:y + height, x:x + width] = small
    scale = 0.7 * frame.shape[1] / 1280
    (w, h), _ = cv2.getTextSize(label, FONT, scale, 2)
    scale *= min(1.0, (width - 8) / w)          # it fits under the inset
    under = (y + height + 4, y + height + int(h * 2.2))
    out[under[0]:under[1], x - 3:x + width + 3] = (out[under[0]:under[1], x - 3:x + width + 3] * 0.35).astype(np.uint8)
    cv2.putText(out, label, (x + 4, under[1] - int(h * 0.5)), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def hand_groups(tiles, won):
    """The hand's tiles in the groups a viewer checks the win by (game.melds) where the 14 make a win, else one by one in slot
    order. The winning draw goes in the first group its name fits, so its outline sits in its meld."""
    split = game.melds([t["name"] for t in tiles]) if won else None
    if not split:
        return [[t] for t in tiles]
    pool, groups = list(tiles), []
    for meld in split:
        group = []
        for name in meld:
            pick = next((t for t in pool if t["name"] == name and t["drawn"]), None) \
                or next(t for t in pool if t["name"] == name)
            pool.remove(pick)
            group.append(pick)
        groups.append(group)
    return groups


def hand_card(size, tiles, won, session):
    """The hand's photos on one frame, grouped (hand_groups), each with its name under it, the winning draw outlined."""
    width, height = size
    scale = width / 1280
    frame = np.zeros((height, width, 3), np.uint8)
    groups = hand_groups(tiles, won)
    title = ("The winning hand" if won else "The hand at the end") + ", each tile as the base camera saw it"
    footer = ("outlined: the winning draw.  " if any(t["drawn"] for t in tiles) else "") + f"Photos from session {session}"
    top, bottom, margin = int(100 * scale), int(80 * scale), int(30 * scale)

    def layout(cell):
        pad, gap = max(2, cell // 14), max(6, cell // 3)
        rows, row, used = [], [], 0
        for g in groups:
            w = len(g) * cell + (len(g) - 1) * pad
            if row and used + gap + w > width - 2 * margin:
                rows.append(row)
                row, used = [], 0
            used += (gap if row else 0) + w
            row.append(g)
        return rows + [row], pad, gap

    cell = width // 5
    while True:                 # the biggest photos that fit
        rows, pad, gap = layout(cell)
        line = cell + int(cell * 0.32) + pad
        if cell <= 30 or len(rows) * line <= height - top - bottom:
            break
        cell -= 4
    y = top + (height - top - bottom - len(rows) * line) // 2
    for row in rows:
        x = (width - sum(len(g) * cell + (len(g) - 1) * pad for g in row) - gap * (len(row) - 1)) // 2
        for g in row:
            for t in g:
                if t["read"]:
                    frame[y:y + cell, x:x + cell] = cv2.resize(t["read"]["crops"][0], (cell, cell), interpolation=cv2.INTER_AREA)
                else:
                    frame[y:y + cell, x:x + cell] = 60
                    cv2.putText(frame, "no photo", (x + 4, y + cell // 2), FONT, fitted("no photo", 0.5 * scale, cell - 8),
                                (200, 200, 200), 1, cv2.LINE_AA)
                colour = DRAWN_COLOUR if t["drawn"] else (255, 255, 255)
                if t["drawn"]:
                    cv2.rectangle(frame, (x - 3, y - 3), (x + cell + 2, y + cell + 2), colour, max(3, int(4 * scale)))
                big = fitted(t["name"], 0.9 * scale, cell)
                (w, h), _ = cv2.getTextSize(t["name"], FONT, big, 2)
                cv2.putText(frame, t["name"], (x + (cell - w) // 2, y + cell + h + int(10 * scale)), FONT, big, colour, 2,
                            cv2.LINE_AA)
                x += cell + pad
            x += gap - pad
        y += line
    for text, at, base in ((title, top // 2 + int(14 * scale), 1.0), (footer, height - bottom // 2, 0.6)):
        big = fitted(text, base * scale, width - 2 * margin)
        (w, _), _ = cv2.getTextSize(text, FONT, big, 2)
        cv2.putText(frame, text, ((width - w) // 2, at), FONT, big, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


class Desk:
    """Desk frames by time, blurred and cropped, from scans or desk.mp4."""

    def __init__(self, session_dir, events, crop):
        self.desk, self.capture, self.first = desk_frames(session_dir), None, None
        if self.desk:
            self.capture = cv2.VideoCapture(str(session_dir / desk_video.VIDEO))
            ok, self.first = self.capture.read()
            if not ok:              # a game killed mid-recording leaves desk.mp4 unfinished: the scans still show the hand
                print(f"{desk_video.VIDEO} can't be read (the game stopped before it was finished?): using the scans")
                self.capture.release()
                self.desk, self.capture, self.first = [], None, None
        self.frames = self.desk or scan_frames(session_dir, events)
        self.crop = crop
        self.box, self.size = None, None

    def read(self, item):
        if self.first is not None:
            frame, self.first = self.first, None
            return frame
        if self.capture is not None:
            ok, frame = self.capture.read()     # in order, one at a time
            return frame if ok else None
        return cv2.imread(str(item))

    def prepare(self, frame, events):
        frame = privacy.blur_faces(frame)[0]
        if self.crop and self.box is None:
            self.box = crop_box(events, frame.shape)
            if self.box is None:        # face-down play has no scans: nothing says where the tiles are
                print("--crop mat: no scan saw a tile in this session, so the whole frame stays in, the room too. "
                      "Watch for faces before posting it")
                self.box = (0, 0, frame.shape[1], frame.shape[0])
        if self.box:
            x0, y0, x1, y1 = self.box
            frame = frame[y0:y1, x0:x1]
        if self.size is None:
            scale = min(MAX_SIZE[0] / frame.shape[1], MAX_SIZE[1] / frame.shape[0])
            self.size = tuple(int(round(n * scale / 2) * 2) for n in (frame.shape[1], frame.shape[0]))
        return cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)


# --- writing ---

class Writer:
    """H.264 through ffmpeg when it's installed (what LinkedIn and phones play), else OpenCV's mp4v."""

    def __init__(self, path, size, fps=OUT_FPS):
        self.path, self.size, self.frames = path, size, 0
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg:
            self.proc = subprocess.Popen(
                [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart",
                 str(path)], stdin=subprocess.PIPE)
        else:
            self.video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame, times=1):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        for _ in range(times):
            if self.ffmpeg:
                try:
                    self.proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    raise SystemExit(f"ffmpeg stopped while writing {self.path} (its error is above)") from None
            else:
                self.video.write(frame)
            self.frames += 1

    def close(self):
        if self.ffmpeg:
            self.proc.stdin.close()
            if self.proc.wait():
                raise SystemExit(f"ffmpeg failed writing {self.path}")
        else:
            self.video.release()


def clip_frames(path, speed):
    capture = cv2.VideoCapture(str(path))
    k = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if k % speed == 0:
            yield k, frame
        k += 1
    capture.release()


def build(session_dir, style="story", fps=None, crop=False, out=None, clip_speed=CLIP_SPEED, clip_every=1):
    """Writes the video; returns {path, frames, seconds, desk, clips}."""
    session_dir = Path(session_dir)
    events = read_events(session_dir)
    story = Story(session_dir, events)
    desk = Desk(session_dir, events, crop)
    shown = clips(session_dir, events)[::clip_every] if style == "story" else []
    reads = base_reads(session_dir, events, story.reads) if style == "story" else []
    if not desk.frames and not shown:
        raise SystemExit(f"{session_dir.name} has no scans, desk video or gripper clips to make a video of")
    fps = fps or (DESK_FPS if desk.desk else SCAN_FPS)
    hold = max(1, round(OUT_FPS / fps))
    out = Path(out) if out else session_dir / "video" / f"game_{style}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    first = None
    if desk.frames:
        raw = desk.read(desk.frames[0][1])
        if raw is None:
            raise SystemExit(f"can't read {desk.frames[0][1]}")
        first = desk.prepare(raw, events)
    size = desk.size or (MAX_SIZE[0], int(MAX_SIZE[0] * 480 / 848) // 2 * 2)
    writer = Writer(out, size)
    drawn = [r for r in reads if is_draw(r)]
    writer.write(card(size, story.title(bool(drawn))), int(TITLE_S * OUT_FPS))
    last = first if first is not None else np.zeros((size[1], size[0], 3), np.uint8)
    queue = cut_ins(shown, drawn)
    read_frames = int(READ_S * OUT_FPS)

    def play_clips(until):
        while queue and queue[0]["time"] <= until:
            c = queue.pop(0)
            if c["kind"] == "read":
                shot = inset(last, tile_grid(c["crops"]), read_label(c))
                top, *bottom = story.lines(c["time"])
                text_band(shot, [top], top=True)
                text_band(shot, bottom, top=False)
                writer.write(shot, read_frames)
                continue
            label = clip_label(c, clip_speed)
            for k, frame in clip_frames(c["path"], clip_speed):
                t = c["time"] + k / config.FPS
                shot = inset(last, privacy.blur_faces(frame)[0], label)     # the gripper camera can see the room too
                top, *bottom = story.lines(t)
                text_band(shot, [top], top=True)
                text_band(shot, bottom, top=False)
                writer.write(shot)

    for n, (t, item) in enumerate(desk.frames):
        frame = first if n == 0 else desk.read(item)
        if frame is None:
            break
        if n:
            frame = desk.prepare(frame, events)
        play_clips(t)
        last = frame
        shot = frame.copy()
        top, *bottom = story.lines(t)
        text_band(shot, [top], top=True)
        text_band(shot, bottom, top=False)
        writer.write(shot, hold)
    play_clips(float("inf"))
    hand = final_hand(events, reads) if reads else []
    hand_frames = int(HAND_S * OUT_FPS) if hand else 0
    if hand:
        writer.write(hand_card(size, hand, (story.end or {}).get("result") == "win", story.name), hand_frames)
    ending = story.ending()
    end_frames = int(max(END_S, END_S_PER_LINE * len(ending)) * OUT_FPS)        # a long ending stays up long enough to read
    writer.write(card(size, ending), end_frames)
    cards = int(TITLE_S * OUT_FPS) + end_frames + hand_frames
    writer.close()
    return {"path": out, "frames": writer.frames, "seconds": writer.frames / OUT_FPS,
            "desk": "desk.mp4" if desk.desk else f"{len(desk.frames)} scans", "clips": len(shown), "card_frames": cards,
            "reads": len(drawn), "read_frames": len(drawn) * read_frames, "hand": len(hand)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session", nargs="?", default="latest", help="a session folder (default: the latest play session)")
    p.add_argument("--style", choices=("story", "timelapse"), default="story",
                   help="story: the desk with each pickup's gripper clip cut in (face down: and each draw's base-camera photos, "
                        "the hand's at the end); timelapse: the desk only (default %(default)s)")
    p.add_argument("--fps", type=float, default=None,
                   help=f"desk frames per second of video (default {SCAN_FPS} from scans, {DESK_FPS} from desk.mp4)")
    p.add_argument("--crop", choices=("none", "mat"), default="none",
                   help="mat: cut the frame down to where tiles were seen, leaving most of the room out (default %(default)s)")
    p.add_argument("--clip-speed", type=int, default=CLIP_SPEED, help="how much faster a gripper clip plays (default %(default)s)")
    p.add_argument("--clip-every", type=int, default=1, help="cut in every Nth pickup only, for a shorter video (default %(default)s)")
    p.add_argument("--out", default=None, help="where to write it (default <session>/video/game_<style>.mp4)")
    args = p.parse_args(argv)
    if args.clip_speed < 1 or args.clip_every < 1:
        p.error("--clip-speed and --clip-every are at least 1")
    if args.fps is not None and not 0 < args.fps <= OUT_FPS:
        p.error(f"--fps is over 0 and at most {OUT_FPS}")
    return args


def main(argv=None):
    args = parse_args(argv)
    session_dir = latest_session() if args.session == "latest" else Path(args.session)
    if not (session_dir / "events.jsonl").exists():
        raise SystemExit(f"{session_dir} is not a session folder (no events.jsonl)")
    privacy.require()
    result = build(session_dir, args.style, args.fps, args.crop == "mat", args.out, args.clip_speed, args.clip_every)
    reads = f", {result['reads']} base-camera reads and the hand's {result['hand']} photos" if result["reads"] else ""
    print(f"{result['path']}: {result['seconds']:.0f}s from {result['desk']} and {result['clips']} gripper clips{reads}")
    print("Faces are blurred where the model finds them: watch it through before posting.")
    return result


if __name__ == "__main__":
    main()
