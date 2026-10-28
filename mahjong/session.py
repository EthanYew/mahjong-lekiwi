"""Every hardware run writes a session folder that answers "what happened" without anyone having to
describe it: sessions/<timestamp>_<kind>/ with session.log (all terminal output), events.jsonl (one
JSON object per event: setup, hovers, attempts, results), and per-attempt folders (wrist frames,
act_trace.npz, overview frames). scripts/session_report.py summarises one.
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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


def previous_conditions():
    """The conditions recorded by the most recent session that has them, or {}."""
    for folder in sorted(config.SESSIONS.glob("*"), reverse=True):
        start = next((e for e in read_events(folder) if e["type"] == "session_start"), {})
        if start.get("conditions"):
            return start["conditions"]
    return {}


def ask_one(question, default, ask):
    """One question ({"ask", "choices": [...] or "yn"}): a number, a unique prefix or y/n; Enter takes
    `default` (None: skip). Unclear answers are asked again."""
    choices = question["choices"]
    if choices == "yn":
        shown = {True: "y", False: "n", None: "skip"}.get(default, default)
        while True:
            answer = ask(f"  {question['ask']}? [y/n] (Enter = {shown}): ").strip().lower()
            if not answer:
                return default
            if answer[0] in "yn":
                return answer[0] == "y"
    menu = " / ".join(f"{i} {c}" for i, c in enumerate(choices, 1))
    while True:
        answer = ask(f"  {question['ask']} [{menu}] (Enter = {'skip' if default is None else default}): ").strip().lower()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        matches = [c for c in choices if c.startswith(answer)]
        if len(matches) == 1:
            return matches[0]


def ask_conditions(previous, questions=None, ask=None, confirm=True):
    """Ask config.SETUP_QUESTIONS; returns {key: answer}. With a previous session's answers, Enter keeps
    them all (non-sticky questions reset to their first choice); confirm=False goes straight to the
    questions, with those answers as defaults (a change mid-session)."""
    questions = questions or config.SETUP_QUESTIONS
    ask = ask or input
    fresh = {q["key"]: False if q["choices"] == "yn" else q["choices"][0] for q in questions}
    defaults = {q["key"]: previous.get(q["key"], fresh[q["key"]]) if q["sticky"] else fresh[q["key"]]
                for q in questions}
    if previous and confirm:
        print(f"Setup: {conditions_text(defaults)}")
        if ask("Same as that? [Enter = yes / c = change]: ").strip().lower() not in ("c", "change", "n", "no"):
            return defaults
    else:
        print("Setup (Enter takes the default):")
    return {q["key"]: ask_one(q, defaults[q["key"]], ask) for q in questions}


def ask_notes(ask=None):
    return (ask or input)("Anything else to note (Enter to skip): ").strip()


def _shown(value):
    return {True: "y", False: "n"}.get(value, value) if isinstance(value, bool) else value


def conditions_text(conditions, notes=""):
    """Setup answers (and notes) as one line: "daylight=bright, lamp=y, ...; notes"."""
    text = ", ".join(f"{k}={_shown(v)}" for k, v in conditions.items())
    return "; ".join(t for t in (text, notes) if t)


class Session:
    def __init__(self, kind, ask_setup=True, **setup):
        conditions, notes = {}, ""
        if ask_setup:          # before the folder exists, so quitting here leaves nothing behind
            conditions = ask_conditions(previous_conditions())
            notes = ask_notes()
        self.conditions, self.notes = conditions, notes
        self.dir = config.SESSIONS / f"{datetime.now():%Y%m%d_%H%M%S}_{kind}"
        self.dir.mkdir(parents=True)
        self._log = open(self.dir / "session.log", "a")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(self._stdout, self._log), _Tee(self._stderr, self._log)
        self.event("session_start", kind=kind, commit=git_commit(), conditions=conditions, notes=notes, **setup)
        print(f"Session: {self.dir}" + (f"\nSetup: {conditions_text(conditions)}" if conditions else ""))

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

    def intervention(self, kind, attempt_dir=None, **fields):
        """A person stepped in (mahjong.interventions.KINDS); attempt_dir: the attempt it belongs to."""
        return self.event("intervention", kind=kind, attempt_dir=attempt_dir, **fields)

    def close(self, reason=None):
        """reason: why the session ended. By default "quit", or the exception on its way out when close()
        runs in a `finally` during one (sys.exc_info() still sees it there), so a crash records itself."""
        error = sys.exc_info()[1]
        if error is not None and isinstance(error, SystemExit) and not error.code:
            error = None
        if reason is None:
            reason = "quit" if error is None else type(error).__name__
        self.event("session_end", reason=reason, **({"error": str(error)} if error is not None else {}))
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._log.close()


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def read_jsonl(path):
    """The records of a .jsonl file (events, traces, sample logs, episode logs), or [] if there's none."""
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def read_events(session_dir):
    return read_jsonl(session_dir / "events.jsonl")
