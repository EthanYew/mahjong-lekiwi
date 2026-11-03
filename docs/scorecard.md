# The scorecard: when a policy ships

**Status: adopted 2026-11-03.** The numbers are built: `session_report.py` prints them for any bench session, and
`status.py` totals them per policy. Code: `mahjong/scorecard.py`. This covers policy changes only. The fuller draft,
with cards for hover and control-code changes, is `~/.claude/plans/reviewing-item-4-in-soft-blanket.md`; add those
cards when such a decision comes up.

## Which check answers which question

| | Iteration check | Play-ready check | Strict paired rule |
|---|---|---|---|
| **Question** | Is this a better base to keep building on? | Is it worth spending scored hands on? | Can we claim X is better, or replace a play-ready policy? |
| **Compared with** | The current default, paired | A fixed bar, no comparison | The baseline, paired |
| **Size** | 14 pairs, a second round if unclear | 28 attempts in one session | 28 pairs or more |
| **Time on the arm** | About 20 minutes a round | About 20 minutes | About 40 minutes |
| **Cost of being wrong** | One more run | Wasted hands | A false public claim, or losing a policy that works |

Offline scores come before all three as a screen. They can't clear a policy: v5's first-action error barely moved
(1.0-1.2° vs 0.9°) while it closed its jaws 45mm in the air. Times assume about 40 seconds an attempt, from 75 attempts
in about 50 minutes (`20261029_094543_bench`).

They also can't rank policies that basically work: over 53 standard-mat demos neither had trained on, v6 and v7 tied on
arm, grasp and timing error, while this bench had them at 6/14 vs 11/14 picked. Use them to catch gross failures such as
freezes or jaws that never close. `session_report.py`'s close height helps explain a run (under 10mm: 127/130 picked; 20mm
or more: 3/25), but it's a diagnostic and doesn't decide anything here. Evidence:
`docs/retrospectives/2026-11-03_eval_validity.md`.

## The standard setup

`data/standard_setup.json` names the mat and layout. Only sessions on it count towards the checks, and
`session_report.py` says when one doesn't. Conditions kept changing during weeks 1-3, which taught real lessons but
broke comparisons: three surfaces and eight layouts appeared while policies were being retrained.

A counted result needs:
- **Mat and layout:** the standard ones, read from the session's log.
- **The setup check:** `setup_check.py` passing against the mat's reference. Sessions don't log this, so it's your step.
- **One hover opening within a comparison:** `session_report.py` warns if a pair's openings differ. Set both with
  `--hover-gripper`.

**Lighting can't be locked if the setup travels.** It's still recorded on every session. Pairing shares it between both
policies, and a play-ready run should happen in the room where you'll play.

**Generalising stays a goal but never blocks.** Each play-ready policy gets one walk on the black top and one on the
green mat, reported on their own lines. Training keeps `--replace-mat`.

## The iteration check

Before the run, name the failure this version is meant to fix, such as narrow jaws, a spot, or steering. Then run one
round with `bench.py --policy baseline,candidate --layout even-hand14 --trials 1`:

| Result in `session_report.py` | What to do |
|---|---|
| **Ahead**: more clean wins than losses | Keep it if the named failure improved, with no new failure type and no freezes. Record it with `status.py --set-default` |
| **Drop**: 3 or more pairs behind | Keep the baseline |
| **Another round**: anything else | Run a second round and add the two sessions' wins and losses. Ahead keeps it; level or behind stays with the baseline |

Only the clean count is computed. The named failure, new failure types and freezes are your call, from the session report.

A wrong call here is cheap, so the check accepts a real chance of one. Chance the clean count keeps a candidate, across
both rounds (`scorecard.iteration_chance`):

| True clean rate of the candidate, against an 80% default | 95% | 90% | 80% | 70% | 65% |
|---|---|---|---|---|---|
| Kept | 96% | 85% | 54% | 25% | 16% |

So about one in four candidates that are 10 points worse gets through on the count alone, before the named-failure test.
That's acceptable because the play-ready bar is fixed, so it catches a slide across several iterations.

**The default means the working best.** It is what you bench and correct from next. Being the default says nothing
about being ready to play.

## The play-ready check

In one session on the standard setup: **at least 26 of 28 attempts clean**, no spot missing twice, and no silent wrong
tile name. Run `bench.py --policy ID --layout even-hand14 --trials 2`. The steering test
(`bench.py --hover-perturb-mm`) is reported alongside, but it doesn't gate: no policy has a slope near 0, so a gate would
block everything, and we don't yet know how slope relates to clean attempts.

Chance of reaching the count (`scorecard.play_ready_chance`):

| True clean rate | 97% | 95% | 90% | 85% | 80% |
|---|---|---|---|---|---|
| Reaches 26/28 | 95% | 84% | 46% | 19% | 6% |

**A pass isn't proof.** At 95% clean, only about 31% of 23-pickup hands are clean, and 99% gives about 79%. The proof
is measured on hands: 8 of 10 clean, from `docs/autonomy_plan.md`.

**The first attended hand has no scorecard gate.** It needs a policy picking at least 12/14 on the mat and a passing
setup check. Its output is the failure list, not a score. Play waiting on a gate is how three weeks passed with no real
hand, so don't let this check do that again.

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

## The strict paired rule

Use this for a claim that X is better, as in the writeup, or to replace a policy that has passed the play-ready check.
For routine iteration, use the iteration check.

Compare the candidate with what's shipped, **in one session**, with `bench.py --policy baseline,candidate`. Rounds
alternate, so light and battery drift fall on both. Attempts pair by spot and order. Only pairs where exactly one
side was clean carry information, and the exact McNemar test asks how lopsided those are. On the standard layout,
`--trials 2` gives 28 pairs, with every spot covered twice.

| Result | Meaning | What to do |
|---|---|---|
| **No verdict** | Under 20 pairs | Run more. Totals across sessions decide nothing |
| **Revert** | Clean pairs split against it, p ≤ 0.1 (e.g. 0-5) | Keep the baseline |
| **Ship** | Clean pairs split for it, p ≤ 0.1 (e.g. 5-0, 7-1) | Make it the default |
| **Extend** | Anything else | Run more, at any spot it names as 2 or more clean attempts worse |

A person may still ship on an **Extend**, for a benefit the score can't show, such as a surface only that policy
works on. Name the benefit in `status.py --set-default --evidence`, so the exception stays visible.

### Where 20 pairs came from

**It was inherited, not calculated.** The 2026-10-23 draft protocol was "4 positions × 5 paired (~20 min)", and that
number became `MIN_PAIRS`. Its first code comment credited ROADMAP 4's power figures (power is the chance of detecting a
real difference), but those are for unpaired comparisons. I found no power calculation for this paired test in the
draft, the code or the session log matches I searched.

**A reason found afterwards.** The test only looks at pairs where exactly one policy was clean, and it needs at least 5
of those to reach p ≤ 0.1 (5-0). Past sessions differed in 19-54% of their pairs: v3c vs v4 7 of 37, v4 vs v5 6 of 21,
v5 vs v6 15 of 28. So 20 pairs yields about 4-11 differing pairs, making it about the smallest session where a verdict
is usually reachable. The floor is a sensible minimum and nothing more.

**What it can't do.** An earlier version of this page said twenty pairs catch large differences, and called p ≤ 0.1
loose. Both were wrong. Even large differences are mostly missed at these sizes. Chance of **Ship** when the candidate
really is better (`scorecard.ship_chance`):

| True clean rate, baseline → candidate | 20 pairs | 28 | 40 | 60 |
|---|---|---|---|---|
| 67% → 87% | 27% | 42% | 60% | 79% |
| 80% → 95% | 18% | 33% | 52% | 74% |
| 80% → 90% | 9% | 15% | 23% | 37% |
| 90% → 95% | 2% | 4% | 9% | 15% |

Two policies that are truly equal get a false Ship only 1-3% of the time at these sizes, so the rule is cautious. These
odds treat the two attempts in a pair as independent, which is an approximation: both are at one spot, which probably
makes differing pairs rarer and these chances lower still.

So **Extend** is the usual answer, and it doesn't mean "no difference". A session of 28 pairs gets a real 15-point
gain over the line about a third of the time, which is why routine iteration has its own check.

**Clean depends on people pressing `b`**, and they marked bumps more closely as the week went on, so **never compare
clean rates between sessions**, only within one.

## Past decisions, scored this way

Worked out from the session logs on 2026-11-02. These decisions came before the rules, so this isn't a criticism of
them. It shows what the rules would have said. These were iteration decisions, made when you weren't yet sure any
version was ready for play. None of them came near the play-ready bar, and the iteration column reads the clean count only.

| Decision | Sessions | Picked | Clean | Clean pairs | Strict rule | Iteration check | What happened |
|---|---|---|---|---|---|---|---|
| v4 over v3c | `20261029_094543` | 37/38 vs 36/37 | 34/38 vs 28/37 | 6-1, p 0.12 | Extend | Ahead | v4 shipped. Picks looked tied; clean points v4's way |
| v5 over v4 | `20261029_163730`, `164226`, `165113`, `165537`, pooled | 20/21 vs 19/21 | 14/21 vs 10/21 | 5-1, p 0.22 | Extend (no one session reaches 20 pairs) | Ahead, pooled | v5 shipped, on bumps |
| v6 over v5, green | `20261029_234437` alone | 28/28 (v5 2/3 separately) | 28/28 | Not paired | No verdict | Too few | v6 shipped, a named benefit: the only policy that works on a second surface |
| v6 vs v5, black top | `20261030_142320` | 27/28 vs 22/28 | 14/28 vs 9/28 | 10-5, p 0.30 | Extend | Ahead | No decision needed: v6 was already the default |
| v7 vs v6 | Not paired | v7 14/14 (`20261030_175235`) | v7 8/14 | None | No verdict | Too few | Not the default. To be paired on the standard mat |

**v6 and v7 were not like for like.** v6's best clean numbers (12/14 `20261030_171411`, 11/14 `20261030_174042`) ran
with `--hover-gripper 21`. v7's 8/14 ran at the registry's 26.7%, and so did v6's 20/28 on `spaced14`
(`20261030_163108`). Small sessions on different layouts can't settle this, so pair them at one opening. v7 is untested at 21%.

The retrospective `2026-10-30_v4_to_v6_surfaces.md` gives v4 vs v5 as 19/20 each; the same four session logs give 21
attempts each, as above. I haven't traced why.

## Where it runs

- `python scripts/session_report.py [folder]`: the table, the strict verdict and, on the standard setup, both checks.
- `python scripts/session_report.py --all`: clean and picked per session.
- `python scripts/status.py`: clean and picked per policy across all sessions. The first run takes a few minutes,
  because the knock check works out arm reach per tile per scan. It then saves each session's count in
  `clean_counts.json` beside its events.

Totals in `status.py` mix layouts and surfaces, so they're for orientation. It also prints a line for the standard setup
alone, still for orientation, because each check reads one session (two for a second iteration round).
