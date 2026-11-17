"""A slow video of the desk from the RealSense while a game is played (`--record-desk` on play.py and play_face_down.py), so
the arm is seen moving: the scans are all taken with the arm at rest. scripts/game_video.py turns it into the game's video.

A background thread keeps about FPS frames a second in `<session>/desk.mp4`, with the wall-clock time of each frame in
`desk_times.json` (to line it up with events.jsonl). A frame comes from the scans' own pulls when there are any
(`RealSense.on_colour`), else from `RealSense.idle_colour`, which never runs while a scan pulls frames and never passes a
frame through the depth filters: those keep state, and an extra frame would change the depth the scans read.

Nothing here can stop a game: a frame that fails is skipped and counted. Faces are not blurred here (the pose model would
share the GPU with the policy mid-pick): the video stays in the session folder like the scans, and game_video.py blurs what
it publishes.
"""
import json
import threading
from datetime import datetime

import cv2

FPS = 2.0
VIDEO, TIMES = "desk.mp4", "desk_times.json"


class DeskRecorder:
    def __init__(self, folder, grab, fps=FPS, clock=None):
        """grab(): a BGR frame or None, called from the recorder's thread when no frame was offered since the last one."""
        self.folder, self.grab, self.fps, self.period = folder, grab, fps, 1.0 / fps
        self.clock = clock or (lambda: datetime.now().isoformat(timespec="milliseconds"))
        self.times, self.failures, self.last_error = [], 0, None
        self._offered, self._lock = None, threading.Lock()
        self._writer, self._stop = None, threading.Event()
        self._thread = threading.Thread(target=self._run, name="desk_video", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def offer(self, frame):
        """A frame something else pulled (a scan): kept, so the recorder doesn't pull one of its own."""
        with self._lock:
            self._offered = frame.copy()

    def _take(self):
        with self._lock:
            frame, self._offered = self._offered, None
        return frame if frame is not None else self.grab()

    def _run(self):
        try:
            while not self._stop.wait(self.period):
                try:
                    frame = self._take()
                    if frame is None:
                        continue
                    if self._writer is None:
                        height, width = frame.shape[:2]
                        self._writer = cv2.VideoWriter(str(self.folder / VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                                       (width, height))
                    self._writer.write(frame)
                    self.times.append(self.clock())
                except Exception as e:          # a camera hiccup must not end the game
                    self.failures += 1
                    self.last_error = repr(e)
        finally:
            if self._writer is not None:        # this thread's own writer: closed here, so nothing writes after it
                self._writer.release()

    def stop(self, wait_s=10):
        """Stops the thread and closes the file; returns what the session log keeps: {frames, failures, ...}. A thread that
        hasn't stopped by `wait_s` (a grab hung) is left to finish on its own, and its file may be unreadable."""
        self._stop.set()
        self._thread.join(timeout=wait_s)
        hung = self._thread.is_alive()
        times = list(self.times)
        (self.folder / TIMES).write_text(json.dumps({"fps": self.fps, "times": times}) + "\n")
        return {"video": VIDEO if times else None, "frames": len(times), "failures": self.failures,
                **({"last_error": self.last_error} if self.last_error else {}), **({"hung": True} if hung else {})}


def start(session, camera, fps=None):
    """Records `camera` (a RealSense, or the sim's mat) into the session folder; logs `desk_video_start`. None for no camera."""
    if camera is None:
        print("--record-desk: no RealSense open, so no desk video")
        return None
    fps = fps or FPS
    grab = getattr(camera, "idle_colour", None) or camera.snapshot     # the sim's mat has no stream to share
    recorder = DeskRecorder(session.dir, grab, fps)
    if hasattr(camera, "on_colour"):
        camera.on_colour = recorder.offer
    session.event("desk_video_start", fps=fps, video=VIDEO)
    print(f"Recording the desk: {session.dir.name}/{VIDEO} ({fps:g} fps)")
    return recorder.start()


def stop(session, camera, recorder):
    """Stops recording before the camera closes; logs `desk_video_end`. Never raises: it runs before the arm's own cleanup."""
    if recorder is None:
        return
    try:
        if hasattr(camera, "on_colour"):
            camera.on_colour = None
        result = recorder.stop()
        session.event("desk_video_end", **result)
        print(f"Desk video: {result['frames']} frames" + (f", {result['failures']} failed" if result["failures"] else ""))
    except Exception as e:
        print(f"Desk video: couldn't finish it ({e!r}); the game is unaffected")


def wall_time(stamp):
    """A desk_times.json or events.jsonl time as seconds since the epoch."""
    return datetime.fromisoformat(stamp).timestamp()
