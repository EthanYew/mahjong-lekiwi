#!/usr/bin/env python3
"""Ask a person before a command runs an arm script; stay silent for everything else.

Claude Code calls this before every Bash and Monitor command (`.claude/settings.json`). Permission rules can't do
this job: `Bash(*scripts/bench.py*)` also prompted on `grep`, and no rule can exempt `--sim` (18 of 128 Bash calls
needed a manual approval after those rules landed, 0 of 3,048 before them).

It fails towards asking: a command that mentions an arm script is silent only if every part of it clearly can't
start one. Python (a heredoc, `-c`, a helper file) is read as code: it asks if it imports an arm script, connects the
hardware, or makes a run call (subprocess, runpy, exec...) naming one, passing sys.argv or running a program it can't see.
Shell variables and `for` loops are expanded, `bash -c` is judged by its string, and xargs by what it runs. Until
2026-11-13 it asked on 26 commands in 14 days of transcripts, 22 of them PR bodies, edit scripts and loops that only
mentioned a script; after, on 4: three real runs and a helper it couldn't read. It guards against an agent starting the arm by accident, not against a command built to hide from it:
a name it can't see (`scripts/$s.py`) or a program that runs one itself isn't caught. A new script that moves the
arm goes in ARM_SCRIPTS (`tests/test_arm_run_gate.py` fails until it does).
"""
import ast
import json
import os
import re
import shlex
import sys
from pathlib import Path

ARM_SCRIPTS = ("bench", "record", "play", "collect_corrections", "setup_check", "wrist_capture", "teleop_view", "base_cam_collect", "wall_test", "crude_run",
               "play_face_down")
SIM_SCRIPTS = ("bench", "play", "setup_check", "wrist_capture", "teleop_view", "base_cam_collect", "wall_test", "crude_run",
               "play_face_down")   # `record` and `collect_corrections` have no --sim
SAFE_FLAGS = {"--help", "-h"}
NO_HARDWARE_FLAGS = {"wall_test": {"--report", "--exclude-session"}, "play_face_down": {"--report"}}   # these return before anything connects

MENTION = re.compile(r"scripts/(%s)\.py" % "|".join(ARM_SCRIPTS))
IMPORTS_ARM = re.compile(r"\b(?:from\s+scripts\s+import\s+(?:%s)\b|(?:from|import)\s+(?:scripts\.)?(?:%s)\b)|['\"](?:scripts\.)?(?:%s)['\"]"
                         % ("|".join(ARM_SCRIPTS), "|".join(ARM_SCRIPTS), "|".join(ARM_SCRIPTS)))
# `from scripts import bench` and runpy.run_module('scripts.bench') name no path
# These read, print or track files; none of them runs a file it is given.
CANT_RUN = {
    "basename", "cat", "cp", "cut", "diff", "dirname", "du", "echo", "egrep", "fgrep", "file", "gh", "git", "grep", "head",
    "less", "ls", "mv", "printf", "pwd", "readlink", "realpath", "rg", "ruff", "sed", "sort", "stat", "tail", "tr", "uniq", "wc",
    "pgrep", "pkill", "kill", "sleep", "true", "false", "test", "[", "jq", "nl", "tee", "ping", "mkdir", "touch", "mkfifo",
}
# These run whatever reaches them, possibly through a pipe from another part of the command.
RUNS_ITS_INPUT = {"bash", "dash", "eval", "exec", "parallel", "sh", "source", ".", "xargs", "zsh"}
PREFIXES = {"!", "{", "builtin", "command", "do", "elif", "else", "env", "if", "nice", "nohup", "stdbuf", "then",
            "time", "timeout", "until", "while"}
PREFIX_FLAGS_WITH_VALUE = {"-f", "-o", "-n", "-s", "-k", "--signal", "--kill-after", "--format", "--output"}   # time -f FMT, nice -n 5
SEPARATORS = {";", ";;", "&", "&&", "|", "|&", "||", "(", ")", "\n"}
ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
HEREDOC = re.compile(r"<<-?\s*(?:'(\w+)'|\"(\w+)\"|(\w+))")
LAST_SEGMENT = re.compile(r"&&|\|\||;|\||\$\(|`")
# Inline Python that edits a script's text mentions its path too; only this vocabulary can start a program.
STARTS_PROGRAMS = re.compile(r"subprocess|runpy|\bpty\b|importlib|__import__|\bos\.(system|exec|spawn|popen)|\bexec\(|\beval\(")
CHECK_ONLY_MODULES = {"py_compile", "pyflakes", "ruff", "flake8", "pycodestyle", "mypy"}     # read a file without running it
# Python code (`-c`, or a helper script such as a patch that edits bench.py's text) with none of this can't start an arm script.
STARTS_ARM = re.compile(STARTS_PROGRAMS.pattern + r"|hardware\.connect|^\s*(?:from|import)\s+(?:scripts\.)?(?:%s)\b"
                        % "|".join(ARM_SCRIPTS), re.M)
SUBSTITUTED = "SUBSTITUTED"
RUN_CALLS = re.compile(r"^(subprocess\.\w+|os\.(system|popen|exec\w*|spawn\w*)|runpy\.\w+|importlib\.\w+|exec|eval|__import__|pty\.spawn)$")
SAFE_FIRST = CANT_RUN | {"gh", "git", "ruff", "pytest"}      # a run call whose program is one of these can't start an arm script


def _dotted(node):
    """"subprocess.run" for subprocess.run(...), "" when it isn't a plain name or attribute chain."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner else ""
    return ""


def _python_could_start_arm(code):
    """Could this Python start an arm script? It imports one or connects the hardware, or a run call (subprocess, os.system,
    runpy, exec...) names one, passes sys.argv, or runs a program it can't see. A run of a literal git/gh/grep... can't.
    Text it can't parse falls back to the vocabulary test (STARTS_ARM)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return bool(STARTS_ARM.search(code))
    aliases = {}                                # `import subprocess as s` -> s: subprocess; `from subprocess import run` -> run: subprocess.run
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({a.asname: a.name for a in node.names if a.asname})
        elif isinstance(node, ast.ImportFrom) and node.module:
            aliases.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names})

    def called(func):
        name = _dotted(func)
        head, _, rest = name.partition(".")
        return f"{aliases[head]}.{rest}".rstrip(".") if head in aliases else name
    arm_module = re.compile(r"^(?:scripts\.)?(?:%s)$" % "|".join(ARM_SCRIPTS))
    for node in ast.walk(tree):                 # code, not text: a string that says "import record" imports nothing
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
            [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names] if isinstance(node, ast.ImportFrom) else []
        if any(n.removeprefix("scripts.") in ARM_SCRIPTS for n in names):
            return True
        if isinstance(node, ast.Call) and _dotted(node.func).endswith("hardware.connect"):
            return True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not RUN_CALLS.match(called(node.func)):
            continue
        if any(isinstance(n, ast.Constant) and isinstance(n.value, str) and arm_module.match(n.value) for n in ast.walk(node)):
            return True                         # runpy.run_module('scripts.bench'), importlib.import_module('bench')
        args = list(node.args) + [k.value for k in node.keywords if k.arg in (None, "args")]
        if not args:
            continue
        first = args[0]
        program = first.elts[0] if isinstance(first, (ast.List, ast.Tuple)) and first.elts else first
        if isinstance(program, ast.Constant) and isinstance(program.value, str):
            name = program.value.split()[0].rsplit("/", 1)[-1] if program.value.split() else ""
            if name in SAFE_FIRST and not MENTION.search(ast.unparse(node)):
                continue                        # git, gh, grep...: whatever else it's given
        text = ast.unparse(node)
        if MENTION.search(text) or "argv" in text:
            return True
        if not all(isinstance(n, ast.Constant) for a in args for n in ast.walk(a) if isinstance(n, ast.expr)
                   and not isinstance(n, (ast.List, ast.Tuple, ast.Starred, ast.Load))):
            if called(node.func) not in ("exec", "eval"):       # exec(compile(src...)) of a file that isn't an arm script
                return True
    return False     # stands in for a `$(...)` once what's inside it has been judged on its own


def _head(words):
    """The program a subcommand runs, and its arguments: past `FOO=1`, `timeout 170`, `time -f FMT`, `then` and the like."""
    i, after_prefix = 0, False
    while i < len(words):
        word = words[i]
        if after_prefix and word in PREFIX_FLAGS_WITH_VALUE:
            i += 2
            continue
        if word.rsplit("/", 1)[-1] in PREFIXES:
            after_prefix = True
            i += 1
            continue
        if ASSIGNMENT.match(word) or word.startswith("-") or word.isdigit() or re.fullmatch(r"\d+[smhd]", word):
            i += 1
            continue
        return word.rsplit("/", 1)[-1], words[i + 1:]
    return "", []


def _heredoc_bodies(command):
    """Split a command into the part a shell parses and the text it feeds to heredocs, as (rest, [(program, body)])."""
    lines = command.split("\n")
    rest, bodies = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        rest.append(line)
        i += 1
        found = HEREDOC.search(line)
        if not found:
            continue
        end = next(g for g in found.groups() if g)
        body = []
        while i < len(lines) and lines[i].strip() != end:
            body.append(lines[i])
            i += 1
        i += 1
        words = []
        try:                                # quote-aware first: `--title "a; b" --body-file - <<EOF` feeds gh
            parts = _subcommands(line[:found.start()])
            words = parts[-1] if parts else []
        except ValueError:                  # an open quote: `--body "$(cat <<EOF` feeds what follows the last $( or ;
            try:
                words = shlex.split(LAST_SEGMENT.split(line[:found.start()])[-1])
            except ValueError:
                pass
        bodies.append((_head(words)[0], "\n".join(body)))
    return "\n".join(rest), bodies


def _substitutions(text):
    """Each outermost `$(...)` and backtick span in text, as (start, end, inner); ValueError if one is unclosed."""
    spans, i, quoted = [], 0, False
    while i < len(text):
        if text[i] == "\\":                      # \` and \$( are literal text
            i += 2
            continue
        if text[i] == "'" and not quoted:          # nothing inside single quotes is substituted
            j = text.find("'", i + 1)
            if j < 0:
                raise ValueError("unclosed quote")
            i = j + 1
            continue
        if text[i] == '"':
            quoted = not quoted
            i += 1
            continue
        if text.startswith("$(", i):
            depth, j = 1, i + 2
            while j < len(text) and depth:
                depth += {"(": 1, ")": -1}.get(text[j], 0)
                j += 1
            if depth:
                raise ValueError("unclosed $(")
            spans.append((i, j, text[i + 2:j - 1]))
            i = j
        elif text[i] == "`":
            j = i + 1
            while j < len(text) and (text[j] != "`" or text[j - 1] == "\\"):
                j += 1
            j = j if j < len(text) else -1
            if j < 0:
                raise ValueError("unclosed backtick")
            spans.append((i, j + 1, text[i + 1:j]))
            i = j + 1
        else:
            i += 1
    return spans


def _subcommands(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars="();<>|&\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    parts, words = [], []
    for token in lexer:
        if token in SEPARATORS:
            parts.append(words)
            words = []
        else:
            words.append(token)
    parts.append(words)
    return [p for p in parts if p]


def _harmless_helper(target, cwd):
    """A .py file python runs that isn't an arm script and has nothing in it that could start one."""
    if MENTION.search(target) or not target.endswith(".py") or "$" in target or SUBSTITUTED in target:
        return False
    path = Path(os.path.expanduser(target))
    if not path.is_absolute():
        path = Path(cwd or os.getcwd()) / path
    try:
        return not _python_could_start_arm(path.read_text())
    except (OSError, UnicodeDecodeError):
        return False


def _is_exempt_run(program, args, cwd=None):
    """`python [flags] scripts/x.py ... --sim` or `--help` (x.py the program python actually runs), a flag of x.py that
    never connects, a linter, `python -c` code that can't start a program, or a helper script that can't either."""
    if not program.startswith("python"):
        return False
    if len(args) > 1 and args[0] == "-m" and args[1] in CHECK_ONLY_MODULES:
        return True
    if args and args[0] == "-c":
        return len(args) > 1 and not _python_could_start_arm(args[1])
    target = next((a for a in args if not a.startswith("-")), "")
    found = MENTION.search(target)
    if not found or not target.endswith(".py") or SUBSTITUTED in target:
        return _harmless_helper(target, cwd)
    after = args[args.index(target) + 1:]
    flags = {a.split("=", 1)[0] for a in after}
    return (bool(SAFE_FLAGS & flags) or (found.group(1) in SIM_SCRIPTS and "--sim" in flags)
            or bool(NO_HARDWARE_FLAGS.get(found.group(1), set()) & flags))


def reason_to_ask(command, cwd=None):
    """Why this command needs a person, or None if it can't start an arm script."""
    if not MENTION.search(command) and not IMPORTS_ARM.search(command):
        return None
    shell_text, bodies = _heredoc_bodies(command)
    piped_to_python = bool(re.search(r"\|\s*(?:\S*/)?python[\d.]*\b", shell_text))
    for program, body in bodies:
        if not (MENTION.search(body) or IMPORTS_ARM.search(body)):
            continue
        if program in CANT_RUN:
            if not piped_to_python:             # a helper written and run in one command can't be read yet: that asks anyway
                continue
            program = "python"                  # `cat <<EOF | python`
        if not program.startswith("python") or _python_could_start_arm(body):
            return f"{program or 'a program'} is given text that mentions an arm script"
    if not MENTION.search(shell_text) and not IMPORTS_ARM.search(shell_text):
        return None
    try:
        spans = _substitutions(shell_text)
    except ValueError:
        return "mentions an arm script and couldn't be parsed"
    for start, end, inner in reversed(spans):
        why = reason_to_ask(inner, cwd)
        if why:
            return f"in a command substitution: {why}"
        found = MENTION.search(inner)       # keep the name, so `python $(echo scripts/bench.py)` still asks
        shell_text = shell_text[:start] + SUBSTITUTED + (f"_{found.group(0)}" if found else "") + shell_text[end:]
    try:
        parts = _subcommands(shell_text)
    except ValueError:
        return "mentions an arm script and couldn't be parsed"
    parts = _expanded(parts)
    for words in parts:
        program, args = _head(words)
        if program in ("bash", "sh", "dash", "zsh") and len(args) > 1 and args[0] == "-c":
            why = reason_to_ask(args[1], cwd)           # a shell's -c string is a command of its own
            if why:
                return f"in {program} -c: {why}"
            continue
        if program == "xargs":
            runs = next((a for a in args if not a.startswith("-") and not re.fullmatch(r"\d+|\{\}", a)), "")
            if runs.rsplit("/", 1)[-1] in CANT_RUN:
                continue
            return "mentions an arm script and xargs runs what it's piped"
        if program in RUNS_ITS_INPUT:
            return "mentions an arm script and runs a shell"
    for words in parts:
        program, args = _head(words)
        text = " ".join(words)
        if not (MENTION.search(text) or IMPORTS_ARM.search(text)) or not program:   # nothing but assignments (expanded below)
            continue
        if program in CANT_RUN or program.startswith("#") or program in ("bash", "sh", "dash", "zsh", "xargs", "for", "done"):
            continue
        if not _is_exempt_run(program, args, cwd):
            return f"may run an arm script: {text[:120]}"
    return None


def _expanded(parts):
    """The parts with each `$VAR` / `${VAR}` replaced by what an assignment or a `for VAR in ...` gave it, and `for` headers
    dropped (the loop's body is its own part): `for f in scripts/bench.py; do python $f; done` is still a run of bench.py."""
    variables, out = {}, []
    for words in parts:
        expanded = []
        for w in words:
            new = re.sub(r"\$\{?(\w+)\}?", lambda m: variables.get(m.group(1), m.group(0)), w)
            bare = re.fullmatch(r"\$\{?\w+\}?", w)        # only a bare $L splits: $L holding several files is several words
            expanded += new.split() if bare and new != w else [new]
        if len(expanded) >= 3 and expanded[0] == "for" and expanded[2] == "in":
            variables[expanded[1]] = " ".join(expanded[3:])
            continue
        if expanded and all(ASSIGNMENT.match(w) for w in expanded):
            variables.update(w.split("=", 1) for w in expanded)
        out.append(expanded)
    return out


def decision(payload):
    """The hook's output for one tool call, or None for no opinion (the normal permission flow decides)."""
    why = reason_to_ask(str((payload.get("tool_input") or {}).get("command") or ""), payload.get("cwd"))
    if why is None:
        return None
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "ask",
        "permissionDecisionReason": f"Real arm run? {why}",
    }}


def main():
    try:
        out = decision(json.load(sys.stdin))
    except Exception as e:   # never let a bug here wave an arm run through
        out = {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": f"arm_run_gate failed ({type(e).__name__}), so a person decides",
        }}
    if out:
        print(json.dumps(out))


if __name__ == "__main__":
    main()
