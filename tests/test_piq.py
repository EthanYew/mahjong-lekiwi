"""piq's inputs against the repo they describe (piq/README.md).

piq is the local model (pi + Ollama) that answers read-only questions about this repo. Its prompt, skills and regression questions are
written by hand, and on 2026-11-22/23 a stale or missing fact in them cost two real failures: a nine-minute reasoning loop, then an
invented `--start-at` flag. A branch that renames a heading, moves a file or adds a flag the model is told about now fails here, so the
fix lands in the same branch instead of surfacing at the next launch. What a test can't check (the model's answers, Ollama's limits)
is `piq/check.py`, and `/hand-back` runs it when a branch touches piq/."""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PIQ = REPO / "piq"
SKILLS = sorted((PIQ / "skills").glob("*/SKILL.md"))
PROSE_FILES = [PIQ / "prompt.md", *SKILLS]
QUESTIONS = json.loads((PIQ / "regression-questions.json").read_text())
# Quoted on purpose though they are not in the repo: stale text a skill warns about, and a section label in a skill's output template.
NOT_IN_REPO = {"No face-up bench yet", "Observed", "Wait,"}


def load_index():
    spec = importlib.util.spec_from_file_location("piq_index", PIQ / "index.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prose(path):
    """What the model is told to rely on. The dated 'reviewed bad answer' section is history and may name things since removed."""
    return re.sub(r"^## A reviewed bad answer.*", "", path.read_text(), flags=re.S | re.M)


def repo_text():
    found = []
    for pattern in ("docs/*.md", "*.md", ".claude/rules/*.md", "scripts/*.py", "mahjong/*.py"):
        found += [f.read_text(errors="replace").lower() for f in REPO.glob(pattern)]
    return "\n".join(found)


def test_every_skill_is_named_and_described():
    assert SKILLS
    for skill in SKILLS:
        head = skill.read_text().split("---\n")[1]
        assert f"name: {skill.parent.name}\n" in head and "description: " in head, f"{skill} needs a name matching its folder and a description"


def test_the_prompt_keeps_its_hard_rules():
    """These four lines were each broken by the model in a real session; a tidy-up that drops them brings the loop back."""
    text = (PIQ / "prompt.md").read_text()
    for needle in ("HARD RULES", "first tool call is `grep`", "offset", "Never write a flag", "GENERATED FACTS"):
        assert needle in text, f"piq/prompt.md lost '{needle}' (see the end of that file for why it's there)"


def test_piq_names_no_machine_or_home_folder():
    for path in [*PROSE_FILES, PIQ / "regression-questions.json"]:
        assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b|/home/(?!<)\w+", path.read_text()), f"{path.relative_to(REPO)} names a machine or a home folder (the repo is public)"


def test_every_path_the_prompt_and_skills_cite_exists():
    for path in PROSE_FILES:
        for cited in sorted(set(re.findall(r"\b((?:docs|scripts|data|mahjong|tests|piq|\.claude)/[\w./*-]*[\w*/])", prose(path)))):
            exists = bool(list(REPO.glob(cited))) if "*" in cited else (REPO / cited).exists()
            assert exists, f"{path.relative_to(REPO)} cites {cited}, which doesn't exist: update it in piq/"


def test_size_claims_are_still_roughly_true():
    """'docs/usage.md is 140 KB' tells the model which files not to read whole; a file that doubled or shrank makes that advice wrong."""
    for path in PROSE_FILES:
        for name, kb in re.findall(r"`?([\w./-]+\.(?:md|json|py))`?[^.\n]{0,24}?\(?~?(\d+) KB", prose(path)):
            file = REPO / name
            if file.exists():
                assert abs(file.stat().st_size / 1000 - int(kb)) <= 0.4 * int(kb), f"{path.relative_to(REPO)} says {name} is {kb} KB; it is {file.stat().st_size // 1000} KB"


def test_quoted_headings_still_exist_somewhere():
    corpus = repo_text()
    for path in PROSE_FILES:
        for quoted in sorted(set(re.findall(r'"([A-Z][^"\n]{7,60})"', prose(path)))):
            assert quoted in NOT_IN_REPO or quoted.lower() in corpus, f'{path.relative_to(REPO)} quotes "{quoted}", which is in no doc, rule or script (renamed?)'


def facts_by_script():
    facts = load_index().script_facts(REPO)
    return {m.group(1): set(m.group(2).split()) for m in re.finditer(r"^(scripts/\w+\.py): (.+)$", facts, re.M)}


def test_generated_facts_name_the_scripts_people_run():
    listed = facts_by_script()
    for script in ("scripts/bench.py", "scripts/play.py", "scripts/record.py"):
        assert listed.get(script), f"piq/index.py found no options for {script}"
    assert "--dagger-dataset" in listed["scripts/bench.py"], "options a helper adds to the parser (takeover.add_argument) are missing from the facts"


@pytest.mark.parametrize("script", sorted(facts_by_script()))
def test_generated_option_list_matches_the_scripts_own_help(script):
    """The model is told 'an option not listed does not exist'. If the scanner misses one it denies a real option; if it lists one that
    isn't there it invents one. --help is the ground truth. Scripts with subcommands keep their options on subparsers: not checkable here."""
    run = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, timeout=90, cwd=REPO, stdin=subprocess.DEVNULL)
    assert run.returncode == 0, f"{script} --help failed:\n{run.stderr[-400:]}"
    if re.search(r"^positional arguments:\n\s+\{", run.stdout, re.M):
        pytest.skip(f"{script} has subcommands")
    real = set()
    for line in run.stdout.splitlines():
        if re.match(r"^ {2,4}-", line):      # option declarations only, not flags mentioned in help prose
            real |= set(re.findall(r"(?<![\w-])-{1,2}[a-zA-Z][\w-]*", re.split(r"\s{2,}", line.strip())[0]))
    listed = facts_by_script()[script] | {"--help", "-h"}
    assert real == listed, f"{script}: piq's facts lack {sorted(real - listed)} and list {sorted(listed - real)}: fix piq/index.py"


@pytest.mark.parametrize("question", QUESTIONS["questions"], ids=lambda q: q["id"])
def test_regression_question_premise_still_holds(question):
    """A false-premise question ('how do I pause a bench?') turns wrong the day someone adds the pause option. Whoever does it updates
    the question in the same branch: retire it, or turn it into a question about the real option."""
    premise = question["premise"]
    text = (REPO / premise["file"]).read_text()
    found = bool(re.search(premise["regex"], text))
    assert found == (premise["expect"] == "present"), (
        f"'{question['id']}' assumed: {premise['why']}. That is no longer true of {premise['file']}: update piq/regression-questions.json")
    for pattern in [*question.get("must", []), *question.get("forbid", [])]:
        re.compile(pattern)
