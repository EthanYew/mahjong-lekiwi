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

from . import checkout, config


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


def added_choices(key):
    """Answers a person added at run time to the setup question `key` (data/added_choices.json)."""
    try:
        return list(json.loads(config.ADDED_CHOICES.read_text()).get(key, []))
    except (OSError, ValueError):
        return []


def add_choice(key, answer):
    try:
        saved = json.loads(config.ADDED_CHOICES.read_text())
    except (OSError, ValueError):
        saved = {}
    saved[key] = [*dict.fromkeys([*saved.get(key, []), answer])]
    config.ADDED_CHOICES.parent.mkdir(parents=True, exist_ok=True)
    config.ADDED_CHOICES.write_text(json.dumps(saved, indent=2) + "\n")


def ask_one(question, default, ask):
    """One question ({"ask", "choices": [...] or "yn"}): a number, a unique prefix or y/n; Enter takes
    `default` (None: skip). Unclear answers are asked again. On a question with "addable" (and a "key"), text
    that matches no choice can be added as a new one, kept in data/added_choices.json for every later session."""
    choices = question["choices"]
    addable = choices != "yn" and question.get("addable") and question.get("key")
    if addable:
        choices = [*choices, *(c for c in added_choices(question["key"]) if c not in choices)]
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
        if answer in choices:
            return answer
        matches = [c for c in choices if c.startswith(answer)]
        if len(matches) == 1:
            return matches[0]
        if addable and not answer.isdigit() and not matches:
            if ask(f"  '{answer}' isn't listed. Add it as a new {question['key']}? [y/n]: ").strip().lower().startswith("y"):
                add_choice(question["key"], answer)
                return answer


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
    current = None      # the open session, for code that logs without being handed one (hardware's camera recovery)

    def __init__(self, kind, ask_setup=True, **setup):
        conditions, notes = {}, ""
        if ask_setup:          # before the folder exists, so quitting here leaves nothing behind
            conditions = ask_conditions(previous_conditions())
            notes = ask_notes()
        self.conditions, self.notes = conditions, notes
        self.dir = config.SESSIONS / f"{datetime.now():%Y%m%d_%H%M%S}_{kind}"
        self.dir.mkdir(parents=True)
        self._open()
        self.event("session_start", kind=kind, commit=git_commit(), checkout=checkout.state(), conditions=conditions,
                   notes=notes, **setup)
        print(f"Session: {self.dir}" + (f"\nSetup: {conditions_text(conditions)}" if conditions else ""))
        for warning in checkout.warnings():
            print(f"WARNING: {warning}")
        if not setup.get("sim"):
            from . import owed      # here, not at the top: owed reads sessions with this module
            for line in owed.items(quick=True):
                print(f"OWED: {line}")

    @classmethod
    def reopen(cls, folder):
        """A session that stopped (or crashed) before its last step, open again to add to its folder: output goes on in its
        session.log. Logs `session_reopened` with today's code; close() as usual."""
        self = cls.__new__(cls)
        self.conditions, self.notes = {}, ""
        self.dir = Path(folder)
        self._open()
        self.event("session_reopened", commit=git_commit(), checkout=checkout.state())
        return self

    def _open(self):
        self._log = open(self.dir / "session.log", "a")
        self._outer, Session.current = Session.current, self
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(self._stdout, self._log), _Tee(self._stderr, self._log)

    def event(self, type, **fields):
        record = {"time": datetime.now().isoformat(timespec="seconds"), "type": type, **fields}
        append_jsonl(self.dir / "events.jsonl", [record], default=_jsonable)
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
        if Session.current is self:
            Session.current = self._outer


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


_DAMAGE_REPORTED = set()


def _tail(raw):
    """(the whole lines of a .jsonl file's bytes, what follows them: a cut-off line and/or zero padding, b"" if none)."""
    end = raw.rfind(b"\n") + 1
    return raw[:end], raw[end:]


def read_jsonl(path):
    """The records of a .jsonl file (events, traces, sample logs, episode logs), or [] if there's none.

    A power cut mid-write damages only the end of a file: on 2026-11-18 one left 566 zero bytes after an events.jsonl's last
    line (the filesystem had reserved the space, the data never came), and every script that lists sessions stopped on it.
    So what follows the last whole line, zero padding or a line cut short, is left out and said once per file on stderr (a
    reader can also catch a line another process is still writing). A bad line before that, zero bytes in it included,
    isn't a power cut, and still raises."""
    path = Path(path)
    if not path.exists():
        return []
    whole, tail = _tail(path.read_bytes())
    rows = [json.loads(line) for line in whole.decode().splitlines() if line.strip()]
    fragment = tail.rstrip(b"\0")
    if fragment.strip():
        try:
            rows.append(json.loads(fragment.decode()))      # whole but for its newline: the cut came just before it
            fragment = b""
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    damage = [f"{len(tail) - len(tail.rstrip(b'\0'))} zero bytes"] if tail.endswith(b"\0") else []
    damage += ["a cut-off last line"] if fragment.strip() else []
    if damage and str(path) not in _DAMAGE_REPORTED:
        _DAMAGE_REPORTED.add(str(path))
        print(f"WARNING: {path}: {' and '.join(damage)} after its last whole line left out (a write cut short, e.g. a "
              "power cut, or one still being written); the rest is read", file=sys.stderr)
    return rows


def append_jsonl(path, records, default=None):
    """Append records to a .jsonl file, one a line. A file whose end a power cut damaged (read_jsonl) is first cut back to its
    last whole line, the bytes after it kept in <name>.cut beside it: a new record written onto a cut-off line would merge
    with it and be lost, and the next append would leave a bad line before the end, which read_jsonl refuses."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            clean = f.read(1) == b"\n"
        if not clean:
            whole, tail = _tail(path.read_bytes())
            kept = tail.rstrip(b"\0")
            try:
                json.loads(kept.decode())       # a record whole but for its newline, which read_jsonl reads: keep it
            except (json.JSONDecodeError, UnicodeDecodeError):
                kept = b""
            with open(path.with_name(path.name + ".cut"), "ab") as f:
                f.write(tail[len(kept):])
            with open(path, "r+b") as f:
                f.truncate(len(whole))
                f.seek(len(whole))
                f.write(kept + b"\n" if kept else b"")
            print(f"WARNING: {path}: {len(tail) - len(kept)} bytes after its last whole line (a write cut short) moved to "
                  f"{path.name}.cut before appending", file=sys.stderr)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record, default=default) + "\n")


def read_events(session_dir):
    return read_jsonl(session_dir / "events.jsonl")


def is_sim(session_dir):
    """A run of the simulated arm (`--sim`), whatever its folder is called: sim sessions are named like real ones."""
    path = Path(session_dir) / "events.jsonl"
    start = next((e for e in read_jsonl(path) if e.get("type") == "session_start"), {}) if path.exists() else {}
    return bool(start.get("sim"))


VOID_SESSIONS = config.DATA / "void_sessions.json"    # session folder name -> why it counts for nothing
_voided_cache = {}


def voided(path=None):
    """{session folder name: why it counts for nothing} (data/void_sessions.json), {} without the file. For a fault found
    afterwards that the log can't show, e.g. 20261113_100710_bench ran 25 minutes on torn wrist frames before the bench
    could see them. Read once per change of the file: it's asked about every session folder."""
    path = Path(path or VOID_SESSIONS)
    if not path.exists():
        return {}
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _voided_cache:
        _voided_cache.clear()
        _voided_cache[key] = json.loads(path.read_text())
    return _voided_cache[key]


def void(name, why, path=None):
    """Add a session to data/void_sessions.json (scripts/status.py --void says which ones may be)."""
    path = Path(path or VOID_SESSIONS)
    saved = {**voided(path), name: why}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n")
    _voided_cache.clear()


def void_reason(folder, path=None):
    """Why a session folder (or its name) counts for nothing, or None."""
    return voided(path).get(folder.name if hasattr(folder, "name") else Path(str(folder)).name)


def counted_real(session_dir):
    """A real session that counts: not simulated and not voided. Every total over bench sessions goes through this."""
    return not is_sim(session_dir) and not void_reason(session_dir)
