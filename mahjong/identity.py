"""Tile names that never go silently wrong (docs/autonomy_plan.md, section 4): each hand slot keeps a
running vote over every scan of the session, a name counts only when the vote is clearly for it, and a
tile's vote starts again only when it was moved (picked, placed, or seen to change).

A scan's tile carries `votes` ({name: confidence summed over the scan's frames}) once scans vote across
their frames (the scan-voting branch); before that, its single best read stands in ({tile_name:
confidence}). Either way the slot's total adds every scan's votes.

The bars below are first guesses, not measured: set them from the detector exam (agree-and-wrong must
stay 0, the ask rate as low as that allows) and say so here.
"""
from collections import defaultdict

MIN_SHARE = 0.8             # the leading name's share of all the votes a slot has collected
MIN_SCANS = 2               # scans a slot needs to have been seen in before its name counts
DRAWN_MIN_SHARE = 0.9       # the drawn tile's name is new information: a stricter bar

# A stopgap until a second read from the wrist camera passes its exam (docs/name_check_plan.md): play asks about these
# even when the votes are sure. Keyed on the name READ, because that's all the robot knows: an 8C reads "7C", so no rule on
# "8C" could ever fire. So real 7Cs get asked about too. Each is a name another tile was misread as, at a confidence
# overlapping or close to real ones (scripts/name_report.py, docs/name_data.md). Delete an entry once its evidence says so.
ALWAYS_ASK_READS = {
    "7C": "8C, 6C and 3C have been misread as it",
    "4B": "2B has been misread as it",
    "3C": "2C has been misread as it",
    "EW": "1B has been misread as it",
    "6C": "7C and 5C have been misread as it",
    "5C": "9C has been misread as it",
    "2C": "3C has been misread as it, by detector v7x3 at up to 0.96",
}
CONFIDENCE_FLOOR = 0.80     # a name's mean best read per scan. On 42 mat placements in dim or no daylight, every 3-scan window:
                            # wrong names 5.3% -> 1.1%, questions 10.7% -> 21.9%. Lowers the risk, doesn't remove it


def scan_votes(tile):
    """{name: weight} that one scan's detection of a tile gives."""
    return dict(tile.get("votes") or {tile["tile_name"]: tile["confidence"]})


class Votes:
    """Running name votes per slot (a hand slot id, or "zone" for the drop zone)."""

    def __init__(self):
        self.totals = defaultdict(lambda: defaultdict(float))
        self.scans = defaultdict(int)
        self.confidences = defaultdict(lambda: defaultdict(list))

    def add(self, slot, tile):
        for name, weight in scan_votes(tile).items():
            self.totals[slot][name] += weight
        self.scans[slot] += 1
        if tile.get("confidence"):      # 0.0: a tile the wrist camera or a faint box found, with no read of its name
            self.confidences[slot][tile["tile_name"]].append(tile["confidence"])

    def known(self, slot, name, weight=10.0):
        """A name a person (or the game) vouches for: a strong vote, so later flicker can't outvote it,
        while a tile that really changed still shows as a split."""
        self.reset(slot)
        self.totals[slot][name] = weight
        self.scans[slot] = MIN_SCANS

    def reset(self, slot):
        self.totals.pop(slot, None)
        self.scans.pop(slot, None)
        self.confidences.pop(slot, None)

    def leader(self, slot):
        """(name, share of the votes, scans seen in), or (None, 0, 0) for a slot never seen."""
        votes = self.totals.get(slot)
        if not votes:
            return None, 0.0, 0
        name = max(votes, key=votes.get)
        return name, votes[name] / sum(votes.values()), self.scans[slot]

    def sure(self, slot, min_share=MIN_SHARE, min_scans=MIN_SCANS):
        """The slot's name if the vote is clearly for it, else None (ask, or scan again)."""
        name, share, scans = self.leader(slot)
        return name if name is not None and share >= min_share and scans >= min_scans else None

    def doubt(self, slot, name, floor=CONFIDENCE_FLOOR):
        """Why a name the votes are sure of should still go to a person, or None. Play asks this before going on a name;
        later checks of a name already settled don't."""
        if name in ALWAYS_ASK_READS:
            return f"It reads {name}, and {ALWAYS_ASK_READS[name]}"
        reads = self.confidences.get(slot, {}).get(name)
        mean = sum(reads) / len(reads) if reads else 0.0
        return None if mean >= floor else f"It reads {name} at only {mean:.2f} confidence"

    def describe(self, slot):
        votes = self.totals.get(slot) or {}
        return ", ".join(f"{n} {v:.2f}" for n, v in sorted(votes.items(), key=lambda nv: -nv[1])) or "no reads"
