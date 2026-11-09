#!/usr/bin/env python3
"""Ask a person before a command runs an arm script; stay silent for everything else.

Claude Code calls this before every Bash and Monitor command (`.claude/settings.json`). Permission rules can't do
this job: `Bash(*scripts/bench.py*)` also prompted on `grep`, and no rule can exempt `--sim` (18 of 128 Bash calls
needed a manual approval after those rules landed, 0 of 3,048 before them).

It fails towards asking: a command that mentions an arm script is silent only if every part of it clearly can't
start one. It guards against an agent starting the arm by accident, not against a command built to hide from it:
a name it can't see (`scripts/$s.py`) or a program that runs one itself isn't caught. A new script that moves the
arm goes in ARM_SCRIPTS (`tests/test_arm_run_gate.py` fails until it does).
"""
import json
import re
import shlex
import sys

ARM_SCRIPTS = ("bench", "record", "play", "collect_corrections", "setup_check", "wrist_capture", "teleop_view", "holdup_arm", "wall_test", "crude_run")
SIM_SCRIPTS = ("bench", "play", "setup_check", "wrist_capture", "teleop_view", "holdup_arm", "wall_test", "crude_run")   # `record` and `collect_corrections` have no --sim
SAFE_FLAGS = {"--help", "-h"}

MENTION = re.compile(r"scripts/(%s)\.py" % "|".join(ARM_SCRIPTS))
# These read, print or track files; none of them runs a file it is given.
CANT_RUN = {
    "basename", "cat", "cp", "cut", "diff", "dirname", "du", "echo", "egrep", "fgrep", "file", "gh", "git", "grep", "head",
    "less", "ls", "mv", "printf", "pwd", "readlink", "realpath", "rg", "sed", "sort", "stat", "tail", "tr", "uniq", "wc",
}
# These run whatever reaches them, possibly through a pipe from another part of the command.
RUNS_ITS_INPUT = {"bash", "dash", "eval", "exec", "parallel", "sh", "source", ".", "xargs", "zsh"}
PREFIXES = {"!", "{", "builtin", "command", "do", "elif", "else", "env", "if", "nice", "nohup", "stdbuf", "then",
            "time", "timeout", "until", "while"}
SEPARATORS = {";", ";;", "&", "&&", "|", "|&", "||", "(", ")", "\n"}
ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
HEREDOC = re.compile(r"<<-?\s*(?:'(\w+)'|\"(\w+)\"|(\w+))")
LAST_SEGMENT = re.compile(r"&&|\|\||;|\||\$\(|`")
# Inline Python that edits a script's text mentions its path too; only this vocabulary can start a program.
STARTS_PROGRAMS = re.compile(r"subprocess|runpy|\bpty\b|importlib|__import__|\bos\.(system|exec|spawn|popen)|\bexec\(|\beval\(")
CHECK_ONLY_MODULES = {"py_compile", "pyflakes"}     # parse a file without running it


def _head(words):
    """The program a subcommand runs, and its arguments: past `FOO=1`, `timeout 170`, `then` and the like."""
    i = 0
    while i < len(words):
        word = words[i]
        if ASSIGNMENT.match(word) or word in PREFIXES or word.startswith("-") or word.isdigit() or re.fullmatch(r"\d+[smhd]", word):
            i += 1
            continue
        return word.rsplit("/", 1)[-1], words[i + 1:]
    return "", []


def _heads(parts):
    """Each subcommand's program and arguments, following `P=~/env/bin/python; $P scripts/x.py`."""
    variables, heads = {}, []
    for words in parts:
        program, args = _head(words)
        if program.startswith("$"):
            program = variables.get(program.strip("${}"), program).rsplit("/", 1)[-1]
        elif not program:     # nothing but assignments
            variables.update(w.split("=", 1) for w in words if ASSIGNMENT.match(w))
        heads.append((program, args))
    return heads


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
        try:
            words = shlex.split(LAST_SEGMENT.split(line[:found.start()])[-1])
        except ValueError:
            words = []
        bodies.append((_head(words)[0], "\n".join(body)))
    return "\n".join(rest), bodies


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


def _is_exempt_run(program, args):
    """`python [flags] scripts/x.py ... --sim` or `--help`, with x.py the program python actually runs."""
    if not program.startswith("python"):
        return False
    if len(args) > 1 and args[0] == "-m" and args[1] in CHECK_ONLY_MODULES:
        return True
    target = next((a for a in args if not a.startswith("-")), "")
    found = MENTION.search(target)
    if not found or not target.endswith(".py"):
        return False
    after = args[args.index(target) + 1:]
    return bool(SAFE_FLAGS & set(after)) or (found.group(1) in SIM_SCRIPTS and "--sim" in after)


def reason_to_ask(command):
    """Why this command needs a person, or None if it can't start an arm script."""
    if not MENTION.search(command):
        return None
    shell_text, bodies = _heredoc_bodies(command)
    for program, body in bodies:
        if not MENTION.search(body) or program in CANT_RUN:
            continue
        if not program.startswith("python") or STARTS_PROGRAMS.search(body):
            return f"{program or 'a program'} is given text that mentions an arm script"
    if not MENTION.search(shell_text):
        return None
    try:
        parts = _subcommands(shell_text)
    except ValueError:
        return "mentions an arm script and couldn't be parsed"
    heads = _heads(parts)
    if any(program in RUNS_ITS_INPUT for program, _ in heads):
        return "mentions an arm script and runs a shell or xargs"
    for words, (program, args) in zip(parts, heads):
        text = " ".join(words)
        if not MENTION.search(text):
            continue
        if any(MENTION.search(w[w.index("$("):]) for w in words if "$(" in w) or "`" in text:
            return "runs a command substitution next to an arm script"
        if program in CANT_RUN or program.startswith("#"):
            continue
        if not _is_exempt_run(program, args):
            return f"may run an arm script: {text[:120]}"
    return None


def decision(payload):
    """The hook's output for one tool call, or None for no opinion (the normal permission flow decides)."""
    why = reason_to_ask(str((payload.get("tool_input") or {}).get("command") or ""))
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
