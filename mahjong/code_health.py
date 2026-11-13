"""Checks on the code itself that take no judgement: ruff, and functions copied whole between files.

PR #76's tidying (a helper copied between two scripts, two unused imports) took a review session of 162 tool calls to
find. Run on the code as it stood before that PR, these two checks find exactly those three. `tests/test_code_health.py`
runs them on every commit, and merge_pr.py's test of the merged tree catches a copy made on two branches at once.
"""
import ast
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from . import config

CODE_DIRS = ("mahjong", "scripts", "tests")
COPY_DIRS = ("mahjong", "scripts")      # tests repeat their small fakes on purpose
MIN_STATEMENTS = 3                      # below this, two bodies matching says little


def ruff(repo=None):
    """[{"file", "line", "code", "message"}] from ruff, with the repo's ruff.toml. Raises if ruff isn't installed."""
    repo = Path(repo or config.REPO)
    dirs = [d for d in CODE_DIRS if (repo / d).exists()]
    result = subprocess.run([sys.executable, "-m", "ruff", "check", *dirs, "--no-cache", "--output-format", "json"],
                            cwd=repo, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"ruff didn't run: {result.stderr.strip()}\nInstall it in the lerobot env: pip install ruff")
    return [{"file": str(Path(f["filename"]).relative_to(repo.resolve())), "line": f["location"]["row"], "code": f["code"],
             "message": f["message"]} for f in json.loads(result.stdout or "[]")]


def _fingerprint(function):
    """A function's statements without its docstring, or None if there are too few to mean anything. Names inside
    count, so a copy with its variables renamed is not caught: this finds a paste, not a similar idea."""
    body = function.body[1:] if ast.get_docstring(function) else function.body
    if len(body) < MIN_STATEMENTS:
        return None
    return hashlib.sha1("".join(ast.dump(s) for s in body).encode()).hexdigest()


def copied_functions(repo=None):
    """Groups of "file:name:line", one per function body that appears in more than one file."""
    repo = Path(repo or config.REPO)
    seen = defaultdict(list)
    for folder in COPY_DIRS:
        for path in sorted((repo / folder).glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (key := _fingerprint(node)):
                    seen[key].append((str(path.relative_to(repo)), node.name, node.lineno))
    return [[f"{f}:{name}:{line}" for f, name, line in group] for group in seen.values()
            if len({f for f, _, _ in group}) > 1]
