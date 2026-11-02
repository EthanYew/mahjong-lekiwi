# The scorecard: when a policy ships

**Status: the rule below is proposed (2026-11-02) and waits on your sign-off.** The numbers are built:
`session_report.py` prints them for any bench session, and `status.py` totals them per policy. Code:
`mahjong/scorecard.py`. This covers policy changes only. The fuller draft, with cards for hover and control-code
changes, is `~/.claude/plans/reviewing-item-4-in-soft-blanket.md`; add those cards when such a decision comes up.

## Why "picked" isn't enough

On the black table top, v6 picked up 99 of 108 tiles (92%), but only 72 of 108 attempts (67%) were clean. The goal
is a clean hand, and a knocked neighbour damages a hand even when the tile in the jaws comes up fine. Since v3b every
policy has picked up about 92-100%, so that number has mostly stopped telling them apart.

## The headline number: clean attempts

A **clean attempt** picked up the tile, knocked or bumped nothing, and needed no help from a person.

| Makes an attempt not clean | Doesn't |
|---|---|
| Not picked up | Gripped low or high (`d` / `u`: precarious, but the attempt stands) |
| A tile knocked (scans) or bumped (`b`), during the pickup or the return | Straightening a knock afterwards (the knock already counted) |
| The target pushed or turned (`p`) | Tiles reset as part of the protocol |
| Esc, a hand on the tile or arm, a caught drop, an unexplained mark | Key presses later voided as noise (`keys_void`) |
| A discard that missed the container | Crashes and rescans, which the log can't pin on one attempt |

Alongside it:
- **95% range:** the true rate that k of n is compatible with (exact binomial). 28/28 only says "at least 88%".
- **Picked:** kept next to clean, so older numbers still compare.
- **Person steps per 10 attempts:** rescues, resets and questions a person answered. This stands in for
  person-minutes, which aren't logged: session time includes unattended stretches and idle time.
- **Median seconds per pickup.**

## The ship rule

Compare the candidate with what's shipped, **in one session**, with `bench.py --policy baseline,candidate`. Rounds
alternate, so light and battery drift fall on both. Attempts pair by spot and order. Only pairs where exactly one
side was clean carry information, and the exact McNemar test asks how lopsided those are.

| Result | Meaning | What to do |
|---|---|---|
| **No verdict** | Under 20 pairs | Run more. Totals across sessions decide nothing |
| **Revert** | Clean pairs split against it, p ≤ 0.1 (e.g. 0-5) | Keep the baseline |
| **Ship** | Clean pairs split for it, p ≤ 0.1 (e.g. 5-0, 7-1) | Make it the default |
| **Extend** | Anything else | Run more, at any spot it names as 2 or more clean attempts worse |

A person may still ship on an **Extend**, for a benefit the score can't show, such as a surface only that policy
works on. Name the benefit in `status.py --set-default --evidence`, so the exception stays visible.

**What the rule can't do.** Twenty pairs catch large differences, not small ones. p ≤ 0.1 is deliberately loose: a
wrong ship costs one re-run, while demanding more would rule out nearly every decision at these session sizes.
Clean also depends on people pressing `b`, and they marked bumps more closely as the week went on, so **never
compare clean rates between sessions**, only within one.

## Past decisions, scored this way

Worked out from the session logs on 2026-11-02. These decisions came before the rule, so this isn't a criticism of
them. It shows what the rule would have said.

| Decision | Sessions | Picked | Clean | Clean pairs | Rule says | What happened |
|---|---|---|---|---|---|---|
| v4 over v3c | `20261029_094543` | 37/38 vs 36/37 | 34/38 vs 28/37 | 6-1, p 0.12 | Extend | v4 shipped. Picks looked tied; clean points v4's way |
| v5 over v4 | `20261029_163730`, `164226`, `165113`, `165537`, pooled | 20/21 vs 19/21 | 14/21 vs 10/21 | 5-1, p 0.22 | Extend (no one session reaches 20 pairs) | v5 shipped, on bumps |
| v6 over v5, green | `20261029_234437` alone | 28/28 (v5 2/3 separately) | 28/28 | Not paired | No verdict | v6 shipped, a named benefit: the only policy that works on a second surface |
| v6 vs v5, black top | `20261030_142320` | 27/28 vs 22/28 | 14/28 vs 9/28 | 10-5, p 0.30 | Extend | No decision needed: v6 was already the default |
| v7 vs v6 | Not paired | v7 14/14 (`20261030_175235`) | v7 8/14 | None | No verdict | Not the default. **Pair them next** |

The retrospective `2026-10-30_v4_to_v6_surfaces.md` gives v4 vs v5 as 19/20 each; the same four session logs give 21
attempts each, as above. I haven't traced why.

## Where it runs

- `python scripts/session_report.py [folder]`: the table and the verdict for a session.
- `python scripts/session_report.py --all`: clean and picked per session.
- `python scripts/status.py`: clean and picked per policy across all sessions. The first run takes a few minutes,
  because the knock check works out arm reach per tile per scan. It then saves each session's count in
  `clean_counts.json` beside its events.

Totals in `status.py` mix layouts and surfaces, so they're for orientation. Decide from a paired session.
