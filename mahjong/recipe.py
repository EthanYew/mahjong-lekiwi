"""What a new policy version trains on, checked against the default policy's: its recipe.

A new version is the default's datasets and options plus what was recorded since. v12 (2026-11-17) was started from
a one-dataset example command: 15k frames and 21k steps where v11 had 88k and 120k, so it couldn't be compared with
v11 and was retrained as v13. Nothing had checked the list against the default's.

Everything here reads a registry dict (data/policies.json's shape) and returns values, so make_policy.py and
train_remote.py share one check and the tests pass registries in. A registration records `datasets`, `options`,
`frames` and `steps` (since 2026-11-17); older entries are read from their notes ("make_policy.py from A + B; N frames,
M steps; mat swapped in 70% ...; colour jitter ..."), which never noted --trim-narrowing.

list_in_usage() is the one writer: a new registration's line in docs/usage.md's policy list, which tests/test_docs.py
requires. v16's sync (2026-11-28) only printed the line to add; nobody did, and the fast tests failed on main.
"""
import json
import re
import time
from dataclasses import dataclass, field

from . import checkout, config

OPTIONS = ("augment", "replace_mat", "trim_narrowing")
# Ends a line of docs/usage.md's policy list; new registrations go on the line after it (tests/test_recipe.py checks the
# guide still has it, once)
USAGE_MARKER = "<!-- new policies are added below this line (mahjong/recipe.py) -->"
# Said after a registration in the main checkout, whatever happened to the guide's line: the registry has changed there
MAIN_CHECKOUT_NOTE = ("This is the main checkout: take what registering changed (data/policies.json, data/offline_eval/, "
                      "docs/usage.md; git status shows which) to a branch and open its PR, then restore them here (git "
                      "checkout -- the changed files, and remove the new eval files). Until then a pull can stop, and with "
                      "docs/usage.md changed merge_pr.py won't run from here. Next time, register from a worktree.")


def registry():
    """data/policies.json (without mahjong.policy, which loads torch)."""
    return json.loads(config.POLICIES.read_text())


def bare(repo):
    """A dataset's name without its Hub user: the registry mixes `name` and `Yewy/name`."""
    return repo.split("/")[-1]


def registered_datasets(entry):
    """The datasets a registered policy was built from, as written (source first: a local recording has no Hub user,
    so `Yewy/` can't be put back on); None when the entry doesn't say."""
    names = entry.get("datasets")
    if names is None:
        found = re.search(r"make_policy\.py from ([^;]+)", entry.get("notes", ""))
        if not found:
            return None
        names = found.group(1).split(" + ")
    return names


def trained_on(entry):
    """registered_datasets() as bare names, to compare with a run's."""
    names = registered_datasets(entry)
    return None if names is None else [bare(name) for name in names]


def options_of(entry):
    """The options a registered policy was trained with (only those it records: old notes never noted trimming)."""
    if "options" in entry:
        return entry["options"]
    notes = entry.get("notes", "")
    found = re.search(r"mat swapped in (\d+)%", notes)
    return {"augment": "colour jitter" in notes, "replace_mat": float(found.group(1)) / 100 if found else 0.0}


def options(augment=False, replace_mat=0.0, trim_narrowing=False):
    """A run's options as they're registered and compared."""
    return {"augment": bool(augment), "replace_mat": float(replace_mat or 0.0), "trim_narrowing": bool(trim_narrowing)}


def scale_of(entry):
    """(frames, steps) a registered policy trained on, or None for each it doesn't record."""
    notes = entry.get("notes", "")
    frames, steps = re.search(r"(\d+) frames", notes), re.search(r"(\d+) steps", notes)
    return (entry.get("frames", int(frames.group(1)) if frames else None),
            entry.get("steps", int(steps.group(1)) if steps else None))


@dataclass
class Review:
    default: str
    missing: list | None            # the default's datasets this run leaves out, as registered; None: can't be read
    warnings: list = field(default_factory=list)

    def stop_message(self):
        if not self.missing:
            return None
        flags = " ".join(f"--add {name}" for name in self.missing)
        return (f"The default, {self.default}, also trained on {', '.join(map(bare, self.missing))}, which this run "
                f"leaves out.\n"
                f"  Keep them (a new version is the default's recipe plus the new recordings): {flags}\n"
                f"  Or drop one on purpose, and say why in the plan: --leave-out REPO")


def review(reg, source, add=(), leave_out=(), run_options=None):
    """This run (source + add, minus leave_out on purpose, with run_options) against the registry's default."""
    default = reg["default"]
    entry = reg["policies"][default]
    registered = registered_datasets(entry)
    theirs = None if registered is None else [bare(name) for name in registered]
    named = {bare(repo) for repo in [source, *add, *leave_out]}
    result = Review(default, None if registered is None else [name for name in registered if bare(name) not in named])
    if theirs and bare(source) != theirs[0]:
        result.warnings.append(f"--source {source}: the default's was {registered[0]}, and the source sets the hover "
                               f"policies start from and the dataset's name: keep it, and --add the new recording")
    result.warnings += [f"{repo} is in both --add and --leave-out: --leave-out only acknowledges a dataset left out, "
                        f"so it's trained on" for repo in leave_out if bare(repo) in {bare(r) for r in add}]
    if theirs is None:
        result.warnings.append(f"can't tell which datasets the default, {default}, trained on (no `datasets` in "
                               f"data/policies.json, and its notes don't start 'make_policy.py from'): check this "
                               f"run's list against it by hand")
    else:
        result.warnings += [f"--leave-out {repo}: the default, {default}, didn't train on it, so it changes nothing"
                            for repo in leave_out if bare(repo) not in theirs]
    used = options_of(entry)
    for name, value in (run_options or {}).items():
        if name not in used:
            continue
        flag = "--" + name.replace("_", "-")
        if name == "replace_mat":
            if abs(value - (used[name] or 0.0)) > 0.005:
                result.warnings.append(f"{flag} {value:g} (the default had {used[name] or 0:g}): say why in the plan, or match it")
        elif bool(value) != bool(used[name]):
            result.warnings.append(f"{flag} {'on' if value else 'off'} (the default had it {'on' if used[name] else 'off'}): "
                                   f"say why in the plan, or match it")
    return result


def check(reg, source, add=(), leave_out=(), run_options=None):
    """review(), said: print its warnings, and stop with the flags to paste if a dataset of the default's is missing."""
    result = review(reg, source, add, leave_out, run_options)
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if (message := result.stop_message()):
        raise SystemExit(message)
    return result


def scale(reg):
    """'<id>: N frames, M steps' for the default, to set a dry run's estimate against."""
    frames, steps = scale_of(reg["policies"][reg["default"]])
    return f"{reg['default']}: {frames or '?'} frames, {steps or '?'} steps"


def usage_line(policy_id, entry, date):
    """A new policy's line for docs/usage.md's list: what its registry entry says, and `date`, the day it was listed (a
    sync can come days after the training). What it did on the bench is for a person to write once there's a session to
    cite."""
    datasets, (frames, steps) = registered_datasets(entry), scale_of(entry)
    facts = []
    if datasets:
        facts.append(f"data: {' + '.join(f'`{name}`' for name in datasets)}")
    if frames:
        facts.append(f"{frames} frames")
    if steps:
        facts.append(f"{steps} steps")
    facts.append(f"listed {date}")
    return f"`{policy_id}` ({'; '.join(facts)}),"


def list_in_usage(policy_id, entry, path=None, date=None):
    """Add `policy_id` to docs/usage.md's policy list unless it's named there already; returns what to tell the person
    after registering it, MAIN_CHECKOUT_NOTE included when that's where the guide is. Never raises: it runs after hours of
    training, and before a sync that may shut machines down. `path`: the usage guide beside the registry
    (config.POLICIES), so a test that moves the registry can't write the real guide."""
    path = path or config.POLICIES.parent.parent / "docs" / "usage.md"
    said = _add_to_list(policy_id, usage_line(policy_id, entry, date or time.strftime("%Y-%m-%d")), path)
    return said if checkout.is_worktree(path.parent.parent) else f"{said}\n{MAIN_CHECKOUT_NOTE}"


def _add_to_list(policy_id, line, path):
    by_hand = f"Add this to the policies list in docs/usage.md by hand (the fast tests fail without it):\n  {line}"
    try:
        text = path.read_text()
        if f"`{policy_id}`" in text:
            return f"docs/usage.md already names `{policy_id}`."
        lines = text.split("\n")
        at = next((i for i, current in enumerate(lines) if current.endswith(USAGE_MARKER)), None)
        if at is None:
            return f"docs/usage.md has lost the marker new policies go under ({USAGE_MARKER}). {by_hand}"
        lines.insert(at + 1, line)
        path.write_text("\n".join(lines))
    except (OSError, UnicodeError) as e:
        return f"Couldn't update docs/usage.md ({e}). {by_hand}"
    return f"Added `{policy_id}` to docs/usage.md's policy list:\n  {line}"


def default_mismatch(here, mains):
    """Why a run shouldn't start when this checkout's registry names a different default from origin/main's, which
    the training machine checks against (it trains from main); None when they agree."""
    if here["default"] == mains["default"]:
        return None
    return (f"This checkout's default is {here['default']} but origin/main's is {mains['default']}, and the PC trains "
            f"from origin/main: pull main (or merge the default switch) first.")
