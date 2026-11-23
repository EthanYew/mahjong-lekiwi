#!/usr/bin/env python3
"""Write a grep-able symbol index of the git repo in the current directory, and print its path.

One line per symbol, every line starting with the file path, so a model can `grep -n "scripts/bench.py" INDEX` and get
that file's outline (a few hundred tokens) instead of reading the file (13k tokens):

    scripts/bench.py:769 def run_pair(...)  first line of the docstring
    docs/usage.md:210 ## Training on a remote machine

Only files over MIN_BYTES are indexed (small files are cheap to read whole); each gets one `FILE` line giving its size.
The index lives outside the repo (~/.cache/piq/<repo>/), so it never goes stale in a commit or conflicts between branches;
`piq` rebuilds it at every launch from the working tree, uncommitted edits included.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

MIN_BYTES = 6000
DOC_LEN = 90


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def first_line(node):
    doc = ast.get_docstring(node)
    return " " + doc.strip().splitlines()[0][:DOC_LEN] if doc else ""


def python_symbols(path, text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    def walk(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
                if node.name.startswith("__") and not ast.get_docstring(node):
                    continue   # bare dunders only add noise
                yield f"{path}:{node.lineno} {kind} {prefix}{node.name}{' ' if first_line(node) else ''}{first_line(node)}"
                if isinstance(node, ast.ClassDef):
                    yield from walk(node.body, f"{prefix}{node.name}.")
    yield from walk(tree.body, "")


def markdown_symbols(path, text):
    fenced = False
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,4} \S", line):
            yield f"{path}:{n} {line.strip()[:120]}"


def parser_options(tree, helpers):
    """Option strings from add_argument calls, plus those added by helper functions (e.g. takeover.add_argument(parser)), which a plain scan misses."""
    opts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "add_argument":
            opts += [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-")]
        key = (f.value.id, f.attr) if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else (None, getattr(f, "id", None))
        if key in helpers:
            opts += helpers[key]
    return opts


def helper_functions(root):
    """(module, name) -> options, for functions in mahjong/ and scripts/ that take a parser and add arguments to it."""
    helpers = {}
    for p in [*sorted((root / "mahjong").glob("*.py")), *sorted((root / "scripts").glob("*.py"))]:
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.args.args:
                inner = []
                for n in ast.walk(node):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument":
                        inner += [a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-")]
                builds_parser = any(isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "ArgumentParser" for n in ast.walk(node))
                if inner and not builds_parser:      # a function that makes its own parser (bench.parse_args) adds nothing to its caller's
                    helpers[(p.stem, node.name)] = inner
    return helpers


def script_facts(root):
    """Facts read straight from the code, so they cannot go stale: each script's argparse options, and any COMMANDS menu list."""
    head = git("log", "-1", "--format=%h %cs").strip()
    out = [f"GENERATED FACTS (read from the working tree at launch, HEAD {head}; current by construction, unlike anything written by hand).",
           "Script options, from each script's add_argument calls. An option not listed for a script does not exist for that script:"]
    menus = []
    helpers = helper_functions(root)
    for p in sorted((root / "scripts").glob("*.py")):
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError:
            continue
        opts = parser_options(tree, helpers)
        if opts:
            out.append(f"scripts/{p.name}: " + " ".join(dict.fromkeys(opts)))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "COMMANDS" for t in node.targets) \
                    and isinstance(node.value, (ast.List, ast.Tuple)):
                names = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
                menus.append(f"Menu commands of scripts/{p.name} (its COMMANDS list; 'q' is also accepted where the script says so): " + " ".join(names))
    return "\n".join(out + menus) + "\n"


def main():
    root = Path(git("rev-parse", "--show-toplevel").strip())
    out_dir = Path.home() / ".cache" / "piq" / str(root).strip("/").replace("/", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "symbol_index.md"
    lines = [f"# Symbol index of {root} (built from the working tree; line numbers can drift if files change after this)"]
    files = git("ls-files", "--cached", "--others", "--exclude-standard", "*.py", "*.md").split()
    for rel in sorted(files):
        p = root / rel
        if not p.is_file() or p.stat().st_size < MIN_BYTES:
            continue
        text = p.read_text(errors="replace")
        lines.append(f"{rel}:1 FILE {text.count(chr(10)) + 1} lines, {p.stat().st_size // 1000} KB")
        symbols = python_symbols if rel.endswith(".py") else markdown_symbols
        lines.extend(symbols(rel, text))
    out.write_text("\n".join(lines) + "\n")
    (out_dir / "facts.md").write_text(script_facts(root))
    print(out)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError:
        sys.exit(1)   # not a git repo: piq runs without an index
