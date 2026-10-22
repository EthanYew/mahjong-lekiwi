"""Every hardware run writes a session folder that answers "what happened" without anyone having to
describe it: sessions/<timestamp>_<kind>/ with session.log (all terminal output), events.jsonl (one
JSON object per event: setup, hovers, attempts, results), and per-attempt folders (wrist frames,
act_trace.npz, overview frames). scripts/session_report.py summarises one.
"""
import json
import subprocess
import sys
from datetime import datetime

from . import config


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return self.streams[0].isatty()


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(config.REPO), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


class Session:
    def __init__(self, kind, ask_notes=True, **setup):
        self.dir = config.SESSIONS / f"{datetime.now():%Y%m%d_%H%M%S}_{kind}"
        self.dir.mkdir(parents=True)
        self._log = open(self.dir / "session.log", "a")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(self._stdout, self._log), _Tee(self._stderr, self._log)
        notes = ""
        if ask_notes:
            notes = input("Setup notes for this session (lighting / lamp, anything moved; Enter to skip): ").strip()
        self.event("session_start", kind=kind, commit=git_commit(), notes=notes, **setup)
        print(f"Session: {self.dir}")

    def event(self, type, **fields):
        record = {"time": datetime.now().isoformat(timespec="seconds"), "type": type, **fields}
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(record, default=_jsonable) + "\n")
        return record

    def attempt_dir(self, label):
        self._attempts = getattr(self, "_attempts", 0) + 1
        path = self.dir / f"{self._attempts:03d}_{datetime.now():%H%M%S}_{label}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def close(self):
        self.event("session_end")
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._log.close()


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def read_events(session_dir):
    path = session_dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
