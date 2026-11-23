#!/usr/bin/env python3
"""Checks for the parts of the piq setup that live on this machine (Ollama, pi), and the model's own regression questions.

What the repo can prove about piq's inputs (paths, headings, option lists, question premises) is in tests/test_piq.py, so a branch that
breaks them can't merge. This script covers what a test can't: whether Ollama really runs the limits the Modelfiles ask for, whether the
base model or pi changed since the setup was last verified, and whether the model still answers the regression questions well.

  piq-check --quick   run by `piq` at every launch: fast, no model, prints only problems, never blocks
  piq-check           the same, printing every result
  piq-check --ask     also run the regression questions through piq; if all pass, stamp the setup as verified (about 3 minutes)

Installed as ~/.local/bin/piq-check (a symlink to this file). Exit status is 1 if anything FAILs; the launcher ignores it.
"""
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

HOME = Path.home()
AGENT = HOME / ".pi" / "agent"                 # pi's own config: model sizes, default model, compaction, sessions
STAMP = AGENT / "piq-stamp.json"
BASE_MODEL = "gemma4:12b"
DERIVED = {"gemma4-12b-64k": HOME / ".ollama" / "Modelfile.gemma4-64k", "gemma4-12b-32k": HOME / ".ollama" / "Modelfile.gemma4-32k"}
MAX_AGE_DAYS = 14

results = []   # (level, check, message)


def add(level, check, msg):
    results.append((level, check, msg))


def run(cmd, timeout=20, cwd=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout, r.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return 1, "", str(e)


def repo_root():
    """The checkout piq is being used in (a worktree's own piq/ is what its branch changed), else the one this file lives in."""
    rc, out, _ = run(["git", "rev-parse", "--show-toplevel"])
    here = Path(__file__).resolve().parent.parent
    return Path(out.strip()) if rc == 0 and (Path(out.strip()) / "piq").is_dir() else here


ROOT = repo_root()
PIQ = ROOT / "piq"
QUESTIONS = PIQ / "regression-questions.json"


def setup_files():
    files = [PIQ / "prompt.md", PIQ / "piq", PIQ / "index.py", PIQ / "check.py", PIQ / "regression-questions.json",
             AGENT / "settings.json", AGENT / "models.json", *DERIVED.values(), *sorted((PIQ / "skills").glob("*/SKILL.md"))]
    return [f for f in files if f.exists()]


def setup_hash():
    h = hashlib.sha256()
    for f in setup_files():
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def pi_version():
    rc, out, err = run(["pi", "--version"])
    text = (out or err).strip()
    return text.splitlines()[-1] if text else "?"


def ollama_ids():
    rc, out, _ = run(["ollama", "list"])
    return {line.split()[0].removesuffix(":latest"): line.split()[1] for line in out.splitlines()[1:] if line.strip()} if rc == 0 else None


def ollama_params(model):
    rc, out, _ = run(["ollama", "show", model, "--parameters"])
    return dict(line.split(None, 1) for line in out.splitlines() if line.strip()) if rc == 0 else {}


# ---------------------------------------------------------------- checks

def check_config():
    """Ollama's real limits vs the Modelfile vs what pi believes (failures 001 and 002)."""
    try:
        models = json.loads((AGENT / "models.json").read_text())["providers"]["ollama"]["models"]
        settings = json.loads((AGENT / "settings.json").read_text())
        pi_window = {m["id"]: m["contextWindow"] for m in models}
    except (OSError, ValueError, KeyError, TypeError) as e:
        return add("FAIL", "config", f"can't read pi's models.json/settings.json ({AGENT}): {e!r}")
    ids = ollama_ids()
    if ids is None:
        return add("WARN", "config", "ollama isn't answering; skipped the model checks")
    default = settings.get("defaultModel")
    if default not in ids:
        add("FAIL", "config", f"default model {default} (settings.json) isn't in `ollama list`")
    if default not in pi_window:
        add("FAIL", "config", f"default model {default} isn't in models.json, so pi has no window size for it")
    for name, mf in DERIVED.items():
        if name not in ids:
            add("FAIL", "config", f"{name} is missing from ollama; rebuild: ollama create {name} -f {mf}")
            continue
        real = ollama_params(name)
        file_text = mf.read_text() if mf.exists() else ""
        for key in ("num_ctx", "num_predict", "repeat_penalty"):
            m = re.search(rf"PARAMETER {key} (\S+)", file_text)
            if not m:
                add("FAIL", "config", f"{mf.name} has no {key}" + (" (output is uncapped: failure 002)" if key == "num_predict" else ""))
            elif real.get(key) != m.group(1):
                add("FAIL", "config", f"{name} runs {key}={real.get(key)} but {mf.name} says {m.group(1)}: you edited the file but didn't `ollama create`")
        if str(pi_window.get(name)) != real.get("num_ctx"):
            add("FAIL", "config", f"pi thinks {name} has window {pi_window.get(name)} (models.json) but Ollama runs num_ctx {real.get('num_ctx')}: truncation or late compaction (failure 001)")
    add("OK", "config", f"default {default}; Modelfiles, running models and models.json agree")


def check_versions():
    """Base model re-pulled, derived model rebuilt, pi upgraded, or the setup edited since the last verified run."""
    if not STAMP.exists():
        return add("WARN", "stamp", "never verified: run `piq-check --ask` once to record a baseline")
    st = json.loads(STAMP.read_text())
    ids = ollama_ids() or {}
    ver = pi_version()
    for label, now, then in (("base model " + BASE_MODEL, ids.get(BASE_MODEL), st.get("base_id")),
                             ("pi version", ver, st.get("pi_version")),
                             *[(f"derived model {n}", ids.get(n), st.get("derived", {}).get(n)) for n in DERIVED]):
        if now and then and now != then:
            add("WARN", "versions", f"{label} changed since the last verified run ({then} -> {now}): re-run `piq-check --ask`")
    age = (date.today() - date.fromisoformat(st["date"])).days
    if age > MAX_AGE_DAYS:
        add("WARN", "stamp", f"last verified {age} days ago (limit {MAX_AGE_DAYS}): re-run `piq-check --ask`")
    if st.get("setup_hash") != setup_hash():
        add("WARN", "stamp", "piq's prompt, skills, launcher or model config changed since the last verified run, so the change is untested here: run `piq-check --ask`")


# ---------------------------------------------------------------- regression questions

def session_for(question, since):
    """The newest session file written since `since` whose first user message is this question (another pi window may be open)."""
    for f in sorted((AGENT / "sessions").glob("*/*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True):
        if f.stat().st_mtime < since:
            break
        for line in f.read_text().splitlines():
            try:
                m = json.loads(line).get("message") or {}
            except ValueError:
                continue
            if m.get("role") == "user":
                content = m.get("content", "")
                text = content if isinstance(content, str) else " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                if question in text:
                    return f
                break
    return None


def attempt(q, lim):
    """Ask one regression question once. Returns (problems, seconds, tool calls, peak context tokens)."""
    t0 = time.time()
    rc, out, err = run([str(PIQ / "piq"), "-p", q["q"]], timeout=lim["seconds"], cwd=ROOT)
    secs = time.time() - t0
    problems = []
    if rc != 0:
        problems.append(f"piq exited {rc} or timed out after {secs:.0f}s")
    sess = session_for(q["q"], t0)
    calls = peak = 0
    stops = []
    if sess:
        for line in sess.read_text().splitlines():
            try:
                m = json.loads(line).get("message") or {}
            except ValueError:      # a line still being written
                continue
            if m.get("role") == "assistant":
                stops.append(m.get("stopReason"))
                peak = max(peak, (m.get("usage") or {}).get("totalTokens", 0))
                calls += sum(1 for c in m["content"] if isinstance(c, dict) and c.get("type") == "toolCall")
    else:
        problems.append("no session file found for this question, so tool calls and context size are unchecked")
    if "length" in stops:
        problems.append("a turn ended with stopReason length (cap hit: loop, or num_predict too small)")
    if calls > lim["tool_calls"]:
        problems.append(f"{calls} tool calls (limit {lim['tool_calls']})")
    if peak > lim["peak_tokens"]:
        problems.append(f"peak context {peak} tokens (limit {lim['peak_tokens']})")
    for pat in q.get("must", []):
        if not re.search(pat, out, re.I | re.S):
            problems.append(f"answer lacks /{pat}/")
    for pat in q.get("forbid", []):
        # "there is no --video flag" names the flag to deny it; only a line that doesn't deny it counts as inventing it
        hits = [ln for ln in out.splitlines() if re.search(pat, ln, re.I) and not re.search(r"\b(no|not|never|without|nor)\b|n't", ln, re.I)]
        if hits:
            problems.append(f"answer says /{pat}/ without denying it (invented?): {hits[0].strip()[:80]}")
    return problems, secs, calls, peak


def ask():
    qs = json.loads(QUESTIONS.read_text())
    lim = qs["limits"]
    failed = 0
    for q in qs["questions"]:
        # tests/test_piq.py fails a question whose premise flipped; here it is only skipped, so one stale question can't hide the rest.
        f = ROOT / q["premise"]["file"]
        found = bool(re.search(q["premise"]["regex"], f.read_text(errors="replace"))) if f.exists() else False
        if found != (q["premise"]["expect"] == "present"):
            add("FAIL", "ask", f"STALE QUESTION '{q['id']}': premise is now {'true' if found else 'false'} ({q['premise']['why']}). Update it in piq/regression-questions.json")
            failed += 1
            continue
        problems, secs, calls, peak = attempt(q, lim)
        retried = False
        if problems:   # a 12B model sometimes answers differently on the same question: one retry tells a real regression from bad luck
            retried = True
            first = problems
            problems, secs, calls, peak = attempt(q, lim)
        if problems:
            failed += 1
            add("FAIL", "ask", f"{q['id']}: failed twice. " + "; ".join(problems))
        elif retried:
            add("WARN", "ask", f"{q['id']}: FLAKY, failed once ({'; '.join(first)}) and passed on retry ({secs:.0f}s, {calls} tool calls, peak {peak} tokens)")
        else:
            add("OK", "ask", f"{q['id']}: {secs:.0f}s, {calls} tool calls, peak {peak} tokens")
    return failed


def write_stamp():
    ids = ollama_ids() or {}
    STAMP.write_text(json.dumps({"date": date.today().isoformat(), "at": datetime.now().isoformat(timespec="seconds"),
                                 "pi_version": pi_version(), "base_id": ids.get(BASE_MODEL),
                                 "derived": {n: ids.get(n) for n in DERIVED}, "setup_hash": setup_hash()}, indent=1) + "\n")


def main():
    args = set(sys.argv[1:])
    quick, do_ask = "--quick" in args, "--ask" in args
    check_config()
    check_versions()
    if do_ask:
        if any(r[0] == "FAIL" for r in results):
            add("WARN", "ask", "skipped the questions: fix the FAILs above first")
        elif ask() == 0:
            if pi_version() == "?":      # a stamp with an unknown version would hide the next upgrade
                add("WARN", "stamp", "all questions passed but `pi --version` printed nothing, so the version can't be recorded: not stamped")
            else:
                write_stamp()
                add("OK", "stamp", f"all regression questions passed; stamped {STAMP.name}")
        else:
            add("WARN", "stamp", "not stamped: fix the failures above and re-run")
    shown = [r for r in results if not quick or r[0] != "OK"]
    if quick and shown:
        print(f"piq: {len(shown)} setup warning(s). Details: piq-check (what each guards: piq/README.md)")
    for level, check, msg in shown:
        print(f"  {level:<4} [{check}] {msg}")
    sys.exit(1 if any(r[0] == "FAIL" for r in results) else 0)


if __name__ == "__main__":
    main()
