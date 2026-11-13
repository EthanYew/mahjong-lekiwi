"""The code checks that take no judgement (mahjong/code_health.py). They run in the fast tests rather than a git hook of
their own, so merge_pr.py's test of main + a branch runs them too: a helper pasted on two branches only shows up there."""
import textwrap

from mahjong import code_health


def write(repo, relative, source):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))


BODY = """\
def {name}(weights):
    {doc}
    model = load(weights)
    model.eval()
    return model
"""


def test_ruff_finds_nothing():
    found = code_health.ruff()
    assert not found, "ruff (ruff.toml):\n" + "\n".join(f"{f['file']}:{f['line']} {f['code']} {f['message']}" for f in found)


def test_no_function_is_copied_between_files():
    assert not code_health.copied_functions(), "the same function in two files: move it somewhere both can import"


def test_a_pasted_function_is_found_with_its_name_and_docstring_changed(tmp_path):
    write(tmp_path, "mahjong/a.py", BODY.format(name="predictor", doc='"""One."""'))
    write(tmp_path, "scripts/b.py", BODY.format(name="make_model", doc='"""Another."""'))
    groups = code_health.copied_functions(tmp_path)
    assert len(groups) == 1 and {g.split(":")[1] for g in groups[0]} == {"predictor", "make_model"}


def test_short_functions_and_a_copy_in_one_file_arent_reported(tmp_path):
    write(tmp_path, "mahjong/a.py", "def f(x):\n    return x + 1\n")
    write(tmp_path, "scripts/b.py", "def g(x):\n    return x + 1\n")
    write(tmp_path, "scripts/c.py", BODY.format(name="one", doc="pass") + BODY.format(name="two", doc="pass"))
    assert code_health.copied_functions(tmp_path) == []


def test_a_copy_with_a_changed_statement_isnt_a_copy(tmp_path):
    write(tmp_path, "mahjong/a.py", BODY.format(name="f", doc="pass"))
    write(tmp_path, "scripts/b.py", BODY.format(name="f", doc="pass").replace("model.eval()", "model.train()"))
    assert code_health.copied_functions(tmp_path) == []
