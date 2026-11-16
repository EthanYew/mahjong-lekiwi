"""mahjong/recipe.py: a new version is the default's datasets and options plus the new recording (v12, 2026-11-17)."""
import importlib.util
import sys

import pytest

from mahjong import config, recipe

# v11 as registered: no `datasets` field, its list in the notes
V11_NOTES = ("make_policy.py from mahjong_surfaces_standard_mat_v1 + Yewy/mahjong_surfaces_mat_v1 + "
             "Yewy/mahjong_facedown_dagger_v10; 88075 frames, 120000 steps; mat swapped in 70% of episodes; "
             "colour jitter (AUGMENT); held out: 14 episodes")


def registry(entry):
    return {"default": "old", "policies": {"old": entry}}


def test_v12s_command_misses_the_datasets_v11_was_registered_with():
    result = recipe.review(registry({"notes": V11_NOTES}), "mahjong_surfaces_standard_mat_v1",
                           ["Yewy/mahjong_facedown_dagger_v11"])
    assert result.missing == ["mahjong_surfaces_mat_v1", "mahjong_facedown_dagger_v10"]
    assert "--add Yewy/mahjong_surfaces_mat_v1 --add Yewy/mahjong_facedown_dagger_v10" in result.stop_message()


def test_the_defaults_datasets_plus_a_new_one_pass_with_or_without_the_hub_user():
    add = ["mahjong_surfaces_mat_v1", "Yewy/mahjong_facedown_dagger_v10", "Yewy/mahjong_facedown_dagger_v11"]
    result = recipe.review(registry({"notes": V11_NOTES}), "Yewy/mahjong_surfaces_standard_mat_v1", add)
    assert result.missing == [] and result.stop_message() is None


def test_the_datasets_field_wins_over_the_notes_and_leave_out_drops_one():
    reg = registry({"datasets": ["src", "Yewy/a", "Yewy/b"], "notes": "make_policy.py from src; 1 frames"})
    assert recipe.review(reg, "src").missing == ["a", "b"]
    assert recipe.review(reg, "src", leave_out=["Yewy/a"]).missing == ["b"]


def test_notes_without_a_semicolon_after_the_list_still_read():
    assert recipe.trained_on({"notes": "make_policy.py from src + Yewy/a"}) == ["src", "a"]


def test_an_unreadable_recipe_warns_and_doesnt_stop_or_judge_a_leave_out():
    result = recipe.review(registry({"notes": "hand-registered"}), "src", leave_out=["Yewy/x"])
    assert result.missing is None and result.stop_message() is None
    assert len(result.warnings) == 1 and "by hand" in result.warnings[0]


def test_a_leave_out_the_default_never_trained_on_warns():
    result = recipe.review(registry({"datasets": ["src"]}), "src", leave_out=["Yewy/typo_v09"])
    assert any("--leave-out Yewy/typo_v09" in w for w in result.warnings)


def test_options_are_compared_with_the_notes_for_older_entries():
    reg = registry({"notes": V11_NOTES})
    add = ["mahjong_surfaces_mat_v1", "mahjong_facedown_dagger_v10"]
    same = recipe.review(reg, "mahjong_surfaces_standard_mat_v1", add, run_options=recipe.options(True, 0.7))
    assert same.warnings == []
    differ = recipe.review(reg, "mahjong_surfaces_standard_mat_v1", add, run_options=recipe.options(False, 0.0, True))
    # trimming was never noted, so an older default can't be compared on it
    assert [w.split(" (")[0] for w in differ.warnings] == ["--augment off", "--replace-mat 0"]


def test_registered_options_include_trim_narrowing():
    reg = registry({"datasets": ["src"], "options": recipe.options(True, 0.7, False)})
    assert recipe.review(reg, "src", run_options=recipe.options(True, 0.7, False)).warnings == []
    assert recipe.review(reg, "src", run_options=recipe.options(True, 0.7, True)).warnings[0].startswith(
        "--trim-narrowing on (the default had it off)")


def test_check_prints_warnings_and_stops_on_a_missing_dataset(capsys):
    reg = registry({"datasets": ["src", "Yewy/a"]})
    with pytest.raises(SystemExit, match="--add Yewy/a"):
        recipe.check(reg, "src", leave_out=["Yewy/zzz"])
    assert "Warning: --leave-out Yewy/zzz" in capsys.readouterr().out


def test_scale_prefers_the_registered_fields_and_reads_older_notes():
    assert recipe.scale(registry({"notes": V11_NOTES})) == "old: 88075 frames, 120000 steps"
    assert recipe.scale(registry({"frames": 97446, "steps": 133000, "notes": "1 frames, 2 steps"})) == \
        "old: 97446 frames, 133000 steps"
    assert recipe.scale(registry({})) == "old: ? frames, ? steps"


def test_default_mismatch():
    assert recipe.default_mismatch({"default": "a"}, {"default": "a"}) is None
    assert "origin/main's is b" in recipe.default_mismatch({"default": "a"}, {"default": "b"})


def test_the_live_defaults_recipe_reads():
    """So the check stops a short run against it, rather than only warning."""
    reg = recipe.registry()
    assert recipe.trained_on(reg["policies"][reg["default"]])


@pytest.mark.parametrize("skip_train", [False, True])
def test_make_policy_checks_the_recipe_unless_it_only_registers(monkeypatch, skip_train):
    """--skip-train registers a run already trained: its recipe was settled when it started."""
    spec = importlib.util.spec_from_file_location("make_policy", config.REPO / "scripts" / "make_policy.py")
    make_policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_policy)
    import mahjong.quality

    class Checked(Exception):
        pass

    class PastTheCheck(Exception):
        pass

    def raising(error):
        def stub(*args, **kwargs):
            raise error()
        return stub

    monkeypatch.setattr(sys, "argv", ["make_policy.py", "--source", "s", "--version", "v99"] + ["--skip-train"] * skip_train)
    monkeypatch.setattr(recipe, "check", raising(Checked))
    monkeypatch.setattr(mahjong.quality, "quality_table", raising(PastTheCheck))
    with pytest.raises(PastTheCheck if skip_train else Checked):
        make_policy.main()
