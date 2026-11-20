"""The site's reading hand (site/_includes/hand-rack.html) against the pages it indexes. Each tile links to a
heading id that kramdown makes from the heading's text, so renaming a heading in index.md silently breaks
its tile; and the five acts are the sets of a mahjong hand, so they must run 3, 3, 3, 3 and 2 long."""
import re
from itertools import groupby
from pathlib import Path

import pytest
import yaml

SITE = Path(__file__).resolve().parents[1] / "site"
HAND = yaml.safe_load((SITE / "_data/hand.yml").read_text())
TOCS = {name: yaml.safe_load((SITE / f"_data/{name}.yml").read_text()) for name in ("story", "article_toc")}


def kramdown_id(text):
    """kramdown's auto_ids: lower case, drop everything but ASCII letters, digits, spaces and hyphens, spaces to hyphens."""
    text = re.sub(r"[*_`]", "", text).strip().lower()
    return re.sub(r"[^a-z0-9 -]", "", text).replace(" ", "-").lstrip("0123456789-")


def article_ids():
    md = re.sub(r"<!--.*?-->", "", (SITE / "index.md").read_text(), flags=re.S)
    ids = {kramdown_id(h) for h in re.findall(r"^#{2,3} (.+)$", md, flags=re.M)}
    for inc in re.findall(r"{% include (\S+)", md):
        path = SITE / "_includes" / inc
        if path.exists():
            ids |= set(re.findall(r'\bid="([^"{]+)"', path.read_text()))
    return ids


def test_article_tiles_link_to_headings_that_exist():
    ids = article_ids()
    missing = [(e["n"], e["anchor"]) for e in TOCS["article_toc"] if e["anchor"] not in ids]
    assert not missing, f"article_toc.yml anchors not found in index.md (renamed a heading?): {missing}"


@pytest.mark.parametrize("name", TOCS)
def test_acts_are_the_sets_of_a_hand(name):
    toc = TOCS[name]
    assert [e["n"] for e in toc] == list(range(1, 15))
    assert [len(list(g)) for _, g in groupby(e["act"] for e in toc)] == [3, 3, 3, 3, 2]


def test_tiles_one_to_nine_show_their_own_number():
    assert [t["n"] for t in HAND] == list(range(1, 15))
    assert [t["face"] for t in HAND[:9]] == [str(n) for n in range(1, 10)]
    runs = [HAND[i:i + 3] for i in (0, 3, 6)]
    assert all(len({t["suit"] for t in run}) == 1 for run in runs)            # each run is one suit
    assert len({t["face"] for t in HAND[9:12]}) == 1 and HAND[12]["face"] == HAND[13]["face"]  # a triplet and a pair
