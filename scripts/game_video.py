"""Turn a played game's session folder into a video to show it: the desk from above as the hand goes, the gripper camera's
clip of each pickup cut in where it happened, and on every frame what the robot was doing and how often a person stepped in.

    python scripts/game_video.py                                   # the latest play session, as a story
    python scripts/game_video.py sessions/20261104_183142_play --crop mat
    python scripts/game_video.py latest --style timelapse --fps 12

The desk comes from the session's own photos: `desk.mp4` if the game ran with `--record-desk` (the arm seen moving), else the
first frame of every scan (the arm is always at rest in those, so the board changes and the arm doesn't). The gripper clips are
each pickup's `wrist.mp4`. Faces are blurred on every desk frame (mahjong/privacy.py); `--crop mat` also cuts the frame down to
where tiles were seen, which leaves most of the room out. The blur lowers the chance of a face, it doesn't rule it out: watch the
video before posting it.

The counts on screen are the session report's (mahjong/interventions.py): "questions" a person answered without touching
anything, "hands-on" a person touching a tile, the arm or the mat. A video says which session it is from, so every claim made
with it traces back to one. Never connects to the arm.
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

from mahjong import config, desk_video, interventions, privacy
from mahjong.session import read_events

OUT_FPS = 30
SCAN_FPS, DESK_FPS = 8, 30          # desk frames a second of video: a scan per 1/8s; 2fps desk video at 15x real time
CLIP_SPEED = 4                      # a gripper clip plays this much faster than it happened
MAX_SIZE = (1280, 1080)             # the video fits in this, keeping the desk's shape (a square crop: 1080x1080)
CROP_MARGIN_PX = 60                 # around every tile the scans saw, in the camera's pixels
TITLE_S, END_S = 3.0, 5.0
PIP_SHARE = 0.38                    # a gripper clip's width, as a share of the frame's
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
    """[{time, path, tile, success}] for each pickup with a gripper video, in order."""
    out = []
    for e in pickups(events):
        if FOLDER_TIME.match(e.get("attempt_dir") or ""):
            path = session_dir / e["attempt_dir"] / "wrist.mp4"
            if path.exists():
                out.append({"time": clip_start(session_dir, e["attempt_dir"]), "path": path, "tile": e.get("tile"),
                            "success": e.get("success")})
    return sorted(out, key=lambda c: c["time"])


def pickups(events):
    """Every pickup, as play.py's `attempt` events: face-down play (play_face_down.py, the crude run) logs `crude_pick`,
    with the spot it picked from for a tile (a face-down tile's name isn't known when it's picked) and `held` for success."""
    return [e if e["type"] == "attempt" else {**e, "tile": e.get("spot"), "success": e.get("held")}
            for e in events if e["type"] in ("attempt", "crude_pick")]


def checked_reads(events):
    """Face-down play: the reader's names a person checked (the 13 in the hand check, each draw confirmed). Not questions in
    the grade (pressing Enter on a read isn't one), but a person looked at every one."""
    hand = sum(len(e.get("hand", {})) for e in events if e["type"] == "fd_hand_confirmed" and e.get("by") == "person")
    return hand + sum(1 for e in events if e["type"] == "crude_read" and e.get("by") != "reader")


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
    """What to write on a frame at a given time: the turn, and the counts so far."""

    def __init__(self, session_dir, events):
        self.name = session_dir.name
        self.found = interventions.derive(events)
        self.attempts = pickups(events)
        self.turns = [e for e in events if e["type"] in ("discard_chosen", "turn_done", "crude_choice")]
        self.checked = checked_reads(events)
        self.end = next((e for e in reversed(events) if e["type"] == "hand_end"), None)
        self.policy = next((e.get("policy") for e in events if e["type"] == "bench_setup"), None) \
            or next((e.get("policy") for e in events if e["type"] == "session_start"), None)
        times = [seconds(e["time"]) for e in events if e.get("time")]
        self.start, self.stop = (min(times), max(times)) if times else (0, 0)

    def counts(self, t=None):
        """{picked, attempts, questions, rescues, resets, hands_on} up to time t (the whole game without one): the session
        report's classes (a reset is tiles put back, a rescue a person fixing what went wrong)."""
        def by(e):
            return t is None or seconds(e["time"]) <= t
        found = [i for i in self.found if i.get("time") and by(i)]
        rescues = sum(i["class"] == interventions.RESCUE for i in found)
        resets = sum(i["class"] == interventions.RESET for i in found)
        return {"picked": sum(bool(a.get("success")) for a in self.attempts if by(a)),
                "attempts": sum(1 for a in self.attempts if by(a)),
                "questions": sum(i["class"] == interventions.ASK for i in found),
                "rescues": rescues, "resets": resets, "hands_on": rescues + resets}

    def doing(self, t):
        done = [e for e in self.turns if seconds(e["time"]) <= t]
        if not done:
            return "Reading the hand"
        e = done[-1]
        if e["type"] == "crude_choice":         # face-down play
            return f"Turn {e['turn']}: drew {e['drawn']}, discarding {e['discard']}"
        if e["type"] == "discard_chosen":
            drew = f"drew {e['drawn']}, " if e.get("drawn") else ""
            return f"Turn {e['turn']}: {drew}discarding {e['tile_name']}"
        return f"Turn {e['turn']}: discarded {e['discarded']}"

    def lines(self, t):
        c = self.counts(t)
        clock = datetime.fromtimestamp(t).strftime("%H:%M:%S")
        return (f"{self.name}   {clock}", self.doing(t),
                f"picked {c['picked']}   person: {c['questions']} questions answered, {c['hands_on']} hands-on")

    def title(self):
        day = datetime.fromtimestamp(self.start).strftime("%Y-%m-%d")
        return ["Robot arm playing a hand of mahjong", f"session {self.name}  ({day})",
                f"policy {self.policy}" if self.policy else "", "desk camera from above, gripper camera inset"]

    def ending(self):
        c, minutes = self.counts(), (self.stop - self.start) / 60
        out = []
        if self.end:
            result = self.end.get("result", "?")
            out.append(f"Result: {result}" + (f" in {self.end['turns']} turns" if self.end.get("turns") is not None else ""))
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
        out.append(f"session {self.name}")
        return out


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
        cv2.putText(frame, s, (int(16 * scale), y), FONT, 0.85 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)


def card(size, lines):
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    scale = size[0] / 1280
    lines = [s for s in lines if s]
    y = size[1] // 2 - int(28 * scale) * len(lines)
    for k, s in enumerate(lines):
        big = 1.3 if k == 0 else 0.9
        (w, _), _ = cv2.getTextSize(s, FONT, big * scale, 2)
        cv2.putText(frame, s, ((size[0] - w) // 2, y), FONT, big * scale, (255, 255, 255), 2, cv2.LINE_AA)
        y += int(56 * scale)
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
            self.box = crop_box(events, frame.shape) or (0, 0, frame.shape[1], frame.shape[0])
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
    writer.write(card(size, story.title()), int(TITLE_S * OUT_FPS))
    last = first if first is not None else np.zeros((size[1], size[0], 3), np.uint8)
    queue = list(shown)

    def play_clips(until):
        while queue and queue[0]["time"] <= until:
            c = queue.pop(0)
            label = f"gripper camera, {clip_speed}x speed: {c['tile']} " + ("picked up" if c["success"] else "missed")
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
    writer.write(card(size, story.ending()), int(END_S * OUT_FPS))
    writer.close()
    return {"path": out, "frames": writer.frames, "seconds": writer.frames / OUT_FPS,
            "desk": "desk.mp4" if desk.desk else f"{len(desk.frames)} scans", "clips": len(shown)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session", nargs="?", default="latest", help="a session folder (default: the latest play session)")
    p.add_argument("--style", choices=("story", "timelapse"), default="story",
                   help="story: the desk with each pickup's gripper clip cut in; timelapse: the desk only (default %(default)s)")
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
    print(f"{result['path']}: {result['seconds']:.0f}s from {result['desk']} and {result['clips']} gripper clips")
    print("Faces are blurred where the model finds them: watch it through before posting.")
    return result


if __name__ == "__main__":
    main()
