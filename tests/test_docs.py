"""The docs against the files that decide what they describe. On 2026-10-29 the default policy was switched to v4
by hand while README.md still named v3c and ROADMAP.md still had v3 "not yet recorded": the docs restated facts
that live in data/, and nothing compared them. Each test here is one such fact."""
import json
import re
from pathlib import Path

from mahjong import config

REPO = Path(__file__).resolve().parents[1]
README, ROADMAP, CLAUDE, CODE_MAP = ((REPO / f).read_text() for f in ("README.md", "ROADMAP.md", "CLAUDE.md", "docs/code_map.md"))
RULE_FILES = sorted((REPO / ".claude" / "rules").glob("*.md"))
SKILL_FILES = sorted((REPO / ".claude" / "skills").glob("*/SKILL.md"))
RULES = "\n".join(f.read_text() for f in RULE_FILES)
SKILLS = "\n".join(f.read_text() for f in SKILL_FILES)
FOR_CLAUDE = (("CLAUDE.md", CLAUDE), (".claude/rules", RULES), (".claude/skills", SKILLS))
CLAUDE_MD_LIMIT = 6000      # bytes. It loads into every session and had grown to 34.9KB by 2026-11-05: detail goes in .claude/rules/
REGISTRY = json.loads(config.POLICIES.read_text())
DEFAULT = REGISTRY["default"]
# argparse's own, and options of other programs the docs mention (git, gh)
OTHER_TOOLS = {"--help", "--no-verify", "--draft", "--ff-only", "--no-ff", "--force", "--delete", "--fill", "--hard", "--rebase", "--autostash", "--base", "--body-file", "--title", "--match-head-commit", "--short", "--merges", "--query-gpu", "--format"}


def options_in_code():
    """Every --option a script or module defines (the string literals in add_argument calls)."""
    found = set()
    for path in [*(REPO / "scripts").glob("*.py"), *(REPO / "mahjong").glob("*.py")]:
        for call in re.findall(r"add_argument\((.*?)(?:help=|\)\n)", path.read_text(), re.S):
            found |= set(re.findall(r"\"(--[\w-]+)\"", call))
    return found


def test_the_default_policy_carries_its_evidence():
    note = REGISTRY.get("default_note", {})
    assert note.get("policy") == DEFAULT, ("data/policies.json's default changed without its default_note: use "
                                           "python scripts/status.py --set-default ID --evidence ...")
    assert note.get("evidence")


def test_the_readme_names_the_default_and_every_registered_policy():
    assert f"(default: `{DEFAULT}`" in README, f"README.md's policy list should say (default: `{DEFAULT}`"
    missing = [p for p in REGISTRY["policies"] if f"`{p}`" not in README]
    assert not missing, f"README.md doesn't name {missing}"


def test_dagger_examples_correct_the_default_policy():
    """DAgger corrects the deployed policy (ROADMAP 9): an example driving with an older one is stale."""
    for name, text in (("README.md", README), *FOR_CLAUDE):
        for line in text.splitlines():
            if "record.py" in line and (m := re.search(r"--policy (pickup_mat_cl\w*)", line)):
                assert m.group(1) == DEFAULT, f"{name}: DAgger example drives {m.group(1)}, the default is {DEFAULT}:\n{line}"


def test_the_roadmap_status_names_the_default_policy():
    assert "\n## Status\n" in ROADMAP, "ROADMAP.md should open with a ## Status table"
    status = ROADMAP.split("\n## Status\n", 1)[1].split("\n## ", 1)[0]
    assert DEFAULT in status, f"ROADMAP.md's status table doesn't mention the default policy {DEFAULT}"


def test_every_layout_says_what_it_is_for():
    layouts = json.loads((config.DATA / "layouts.json").read_text())
    assert not [name for name, entry in layouts.items() if not entry.get("note")]


def test_every_script_is_in_the_readme_and_the_code_map():
    for script in sorted((REPO / "scripts").glob("*.py")):
        assert script.name in README, f"README.md has no section or row for scripts/{script.name}"
        assert script.stem in CODE_MAP, f"docs/code_map.md doesn't list {script.stem}"


def test_every_option_the_docs_mention_exists():
    known = options_in_code() | OTHER_TOOLS
    for name, text in (("README.md", README), *FOR_CLAUDE):
        mentioned = set(re.findall(r"(?<![\w-])(--[a-z][\w-]*[a-z0-9])(?![\w-])", text))
        assert not mentioned - known, f"{name} mentions options no script defines: {sorted(mentioned - known)}"


def test_claude_md_stays_small():
    size = len(CLAUDE.encode())
    assert size <= CLAUDE_MD_LIMIT, f"CLAUDE.md is {size} bytes, over {CLAUDE_MD_LIMIT}: it loads into every session, so move detail to .claude/rules/"


def rule_paths(path):
    """The globs in a rule's frontmatter."""
    head = path.read_text().split("---\n")[1]
    return re.findall(r'^\s*- "(.+)"$', head, re.M)


def test_every_rule_loads_for_files_that_exist_and_is_in_the_index():
    """A rule whose globs match nothing silently never loads, and one missing from CLAUDE.md's index is hard to find."""
    assert RULE_FILES
    for rule in RULE_FILES:
        globs = rule_paths(rule)
        assert globs, f"{rule.name} has no paths, so it would load into every session"
        for pattern in globs:
            assert list(REPO.glob(pattern)), f"{rule.name}: no file matches {pattern}"
        assert f"`{rule.name}`" in CLAUDE, f"CLAUDE.md's table doesn't list {rule.name}"


def test_skills_name_real_scripts_and_no_machine():
    """A skill's recipe is followed as written, and the repo may go public: the training machine's address lives in memory, not here."""
    for skill in SKILL_FILES:
        text = skill.read_text()
        head = text.split("---\n")[1]
        assert f"name: {skill.parent.name}\n" in head and "description: " in head, f"{skill} needs a name and a description"
        for script in re.findall(r"scripts/(\w+\.py)", text):
            assert (REPO / "scripts" / script).exists(), f"{skill}: scripts/{script} doesn't exist"
        assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b|/home/(?!<)\w+", text), f"{skill} names a machine or a home folder"
