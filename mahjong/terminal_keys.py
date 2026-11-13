"""Keys from the bench's own terminal, not the whole desktop.

lerobot's keyboard listener and the bench's letter keys used pynput, which hears every key on the desktop: on
2026-10-30 (`20261030_163108_bench`) ~70 keystrokes typed in another terminal tab were logged as collisions,
pushes and grip calls, and arrow keys there as five → marks. Tabs of one terminal window are one window to the
desktop, so focus can't tell them apart; reading this terminal's own input can, since only the tab being typed
in receives it.

While started (the arm working: a pickup, a return, teach-discard) the terminal is in cbreak mode (keys arrive
one at a time, unechoed; Ctrl-C still interrupts) and a thread reads them: → and ← and Esc set the same
`events` flags lerobot's listener did (→ exit_early; ← rerecord_episode + exit_early; Esc stop_recording +
exit_early), anything else goes to `on_char`. Stopped, the terminal is back to normal line input for prompts.
Without a terminal (tests, piped input) it does nothing: tests call the handlers directly.
"""
import atexit
import os
import select
import sys
import threading

ESC_WAIT_S = 0.03       # an arrow key arrives as ESC [ C in one go; a lone ESC with nothing after it is Esc


def new_events():
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


def parse(data, more=lambda: b""):
    """Keys in a chunk of terminal input: "right", "left", "esc", or single characters. `more()` returns what
    arrives within ESC_WAIT_S after a lone ESC (the rest of an escape sequence, or nothing)."""
    keys, i = [], 0
    while i < len(data):
        if data[i:i + 1] == b"\x1b":
            if i + 1 == len(data):
                data += more()
            seq = data[i + 1:i + 3]
            if seq in (b"[C", b"OC"):
                keys.append("right")
                i += 3
            elif seq in (b"[D", b"OD"):
                keys.append("left")
                i += 3
            elif seq[:1] in (b"[", b"O"):          # another escape sequence (up/down, F-keys...): not ours
                j = i + 2
                while j < len(data) and not (0x40 <= data[j] <= 0x7E):
                    j += 1
                i = j + 1
            else:
                keys.append("esc")
                i += 1
        else:
            keys.append(data[i:i + 1].decode(errors="ignore"))
            i += 1
    return [k for k in keys if k]


class Abort:
    """Esc, from the bench's terminal (mahjong/terminal_keys.py), as an Event-like flag for policy.run."""
    def __init__(self, events):
        self.events = events

    def clear(self):
        self.events["stop_recording"] = self.events["exit_early"] = self.events["rerecord_episode"] = False

    def is_set(self):
        return self.events["stop_recording"]


class TerminalKeys:
    def __init__(self, events, on_char=lambda char: None, stream=None):
        self.events, self.on_char = events, on_char
        stream = stream or sys.stdin
        try:
            self.fd = stream.fileno() if stream.isatty() else None
        except (AttributeError, ValueError, OSError):
            self.fd = None
        self.saved, self.thread, self.running = None, None, threading.Event()
        atexit.register(self.stop)

    @property
    def available(self):
        return self.fd is not None

    def handle(self, key):
        if key == "right":
            self.events["exit_early"] = True
        elif key == "left":
            self.events["rerecord_episode"] = self.events["exit_early"] = True
        elif key == "esc":
            self.events["stop_recording"] = self.events["exit_early"] = True
        else:
            self.on_char(key)

    def start(self):
        if not self.available or self.running.is_set():
            return
        import termios
        import tty
        self.saved = termios.tcgetattr(self.fd)
        termios.tcflush(self.fd, termios.TCIFLUSH)        # nothing typed before the arm moved counts
        tty.setcbreak(self.fd)
        self.running.set()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        def more():
            ready, _, _ = select.select([self.fd], [], [], ESC_WAIT_S)
            return os.read(self.fd, 32) if ready else b""
        while self.running.is_set():
            ready, _, _ = select.select([self.fd], [], [], 0.05)
            if ready:
                for key in parse(os.read(self.fd, 64), more):
                    self.handle(key)

    def stop(self):
        if not self.running.is_set():
            return
        self.running.clear()
        if self.thread:
            self.thread.join(timeout=1.0)
        import termios
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        termios.tcflush(self.fd, termios.TCIFLUSH)
