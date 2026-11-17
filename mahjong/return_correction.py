"""Return correction: where to set a held tile down so it lands back at its home (where it was first
seen this session), not wherever the last grasp happened to be.

Returning a tile to where the jaws closed on it lets every return's error carry into the next pickup:
tiles walked 77mm in 9 rounds (4D, 2026-10-27) and 100mm in 10 (6C), toward the edge of the reach.
Aiming every return at one fixed home stops the walk; this model also aims for it more accurately.

What a return does to a tile, measured on 95 bench returns (2026-10-23..28, 24 positions): the jaws
pull a tile only partway to their centre (~55-65% of its offset in the jaws stays with it), so where
it lands depends on how off-centre this grasp was. Predicting the landing (leave-one-position-out):

    the tile keeps its offset in the jaws (no fit)       10.5mm RMS
    a map by position only                                7.7mm
    + how off-centre this grasp was ("grip")              6.8mm
    + both ("grip+position")                              6.1mm

Over 1061 returns to 2026-11-17: none 8.0mm, grip 5.3mm, grip+position 4.7mm. Position matters because detection - grasp
is mostly the gap across the mat, not how off-centre the grasp was: without it, a policy closing 15-19mm +y at S1/S2
dragged the aim along and tiles landed +8mm toward the wall (docs/invariant_notes.md, 2026-11-17).

Each record is one return:
    detection_xy  the tile before the pickup (RealSense, table frame)
    grasp_xy      where the jaws closed on it (the commanded tool point)
    target_xy     where the arm set it down (tool point, after any reach shift: Arm.place's xy)
    landed_xy     the tile after the return (RealSense)
The baseline assumes the tile keeps its offset in the jaws: landed = target + (detection - grasp). The
model fits what the baseline misses, from (detection - grasp) and, with enough positions, detection.

That only works for a tile going back where it came from: detection - grasp is mostly the gap between what the
RealSense sees and where the arm has to go, and that gap changes across the mat (y: +31mm at the drop zone, +4mm
at S1). A tile carried across the mat (play: zone to slot) took the zone's gap to the slot and landed 15-18mm off,
with the aim 22mm off at S4, into the neighbours (20261104_183142_play). So a carried tile is aimed from what returns
measured about the destination (`fit_across`): home + the aim that lands a tile at home + the share of this grasp's
stray that stays with the tile. The stray is how far the jaws closed from where they usually close at that place:
a steady margin left at the close (demos leave one on purpose) is part of "usually" and moves nothing. Over 738
returns 70% of the stray stays in x and 52% in y; `python -m mahjong.return_correction` prints the fit's own numbers.
Like the hover correction (mahjong/hover_correction.py, whose setup fingerprint and file handling it
reuses): kinds chosen by leave-one-position-out error, each must beat the simpler by MIN_GAIN, the
correction is capped and held at its edge value past the measured tiles, records from another setup
are set aside, and a record with an "excluded" reason stays unfitted.

    python -m mahjong.return_correction     # the fit and what it would change
    python scripts/return_samples.py        # add returns from sessions/ (bench.py adds its own)
"""
import argparse
import math

import numpy as np

from . import config, kinematics
from .hover_correction import (EXTRAPOLATION_MARGIN_M, POSITION_CELL_M, append_record, position_cells, read_all,
                               setup_fingerprint)

MIN_GAIN = 0.10                 # a kind is used only if it predicts this much better than the simpler one
MIN_RECORDS = {"grip": 8, "grip+position": 20}
MIN_POSITIONS = {"grip": 3, "grip+position": 5}
MAX_CORRECTION_M = 0.02         # from the baseline target
LANDED_WITHIN_M = 0.05          # the returned tile is the same-name detection nearest its home, this close
SAMPLE_WITHIN_M = 0.03          # a landing further from home is still reset and recorded, but left out of the fit
# Under the layouts' 50mm spot pitch: every face-down tile is "back", so the name can't tell a tile from its neighbour, and an
# unseen landing took the next spot's tile 40-50mm away (22 of 1083 fitted returns to 2026-11-17, 8 face down). No real
# landing was over 24mm (RETURN_DRIFT_RESET_M is 25). Those samples kept grip+position from being chosen.
# Samples whose landing a person's hands may be in: `r` was pressed during the return. New ones aren't recorded.
HAND_MOVED = {("20261030_163108_bench", "015_164126_2B"), ("20261030_163108_bench", "016_164209_2B"),
              ("20261030_163108_bench", "017_164250_NW"), ("20261030_163108_bench", "018_164334_NW"),
              ("20261104_183142_play", "007_183441_1B")}


def carried(detection_xy, home_xy):
    """Is this tile going somewhere else, not back where it came from? (The radius past which Homes calls it moved.)"""
    return math.dist(detection_xy, home_xy) > config.HOME_RADIUS_M


def return_record(detection_xy, grasp_xy, target_xy, landed_xy, **provenance):
    return {"detection_xy": [float(v) for v in detection_xy], "grasp_xy": [float(v) for v in grasp_xy],
            "target_xy": [float(v) for v in target_xy], "landed_xy": [float(v) for v in landed_xy], **provenance}


def landed_tile(tiles, tile_name, home_xy):
    """The returned tile in a scan: the detection named `tile_name` nearest its home (any name if none is),
    within LANDED_WITHIN_M; None if there's none."""
    from .layout import nearest_tile
    return nearest_tile(tiles, home_xy, LANDED_WITHIN_M, tile_name)


def landings(events):
    """Every return in a session's events with where its tile landed: [{tile, attempt_dir, detection_xy,
    grasp_xy, target_xy, home_xy, landed_xy (None if not found), error_m (landed - home), by_hand (`r` was pressed
    during the return)}]. Sessions from before homes: the home is the detection before the pickup."""
    out, attempt, ret, by_hand = [], None, None, False
    for e in events:
        if e["type"] == "attempt":
            attempt, ret, by_hand = e, None, False
        elif e["type"] == "keys" and e.get("during") == "return":
            by_hand = any(p.get("kind") == "by_hand" for p in e.get("presses", []))
        elif e["type"] == "return":
            ret = e
        elif e["type"] == "detections" and e.get("label") == "after_return" and attempt and ret:
            home = ret.get("home_xy") or attempt["detection_xy"]
            tile = landed_tile(e["tiles"], attempt["tile"], home)
            landed = list(tile["xy"]) if tile else None
            out.append({"tile": attempt["tile"], "attempt_dir": attempt["attempt_dir"], "policy": attempt.get("policy"),
                        "spot": attempt.get("spot"), "detection_xy": attempt["detection_xy"],
                        "grasp_xy": attempt.get("grasp_xy") or attempt["hover_xy"], "target_xy": ret["xy"],
                        "home_xy": home, "landed_xy": landed, "error_m": math.dist(landed, home) if landed else None,
                        "by_hand": by_hand})
            attempt = ret = None
    return out


class Homes:
    """Where each tile picked up this session is returned to: where its spot's (or its name's) tile was first
    seen, unless it's now more than HOME_RADIUS_M from there (moved on purpose: a new home). Returning to
    each round's own grasp point walked tiles ~1cm a round (8cm in 10, 2026-10-27, v3b at 6C; 25-31mm in
    one recording, 2026-10-28): any off-centre grip carried into the next round."""

    def __init__(self):
        self.homes = {}

    def __call__(self, tile, spot=None):
        key = spot["id"] if spot else tile["tile_name"]
        home = self.homes.get(key)
        if home is None or math.dist(tile["xy"], home) > config.HOME_RADIUS_M:
            home = self.homes[key] = [float(v) for v in tile["xy"]]
        return home

    def get(self, key):
        return self.homes.get(key)


def check_landing(held, placed_xy, tiles, session, path, correction, hover, **provenance):
    """Where a returned tile landed in the after-return scan `tiles`: a `return_landed` event, a sample for
    the return model in `path`, and a print past RETURN_DRIFT_WARN_M from home. held: {tile_name,
    detection_xy, grasp_xy, home_xy, attempt_dir, spot}; hover: the policy's (whether the next hover can
    reach the tile). Returns the landed detection (None if not seen) and whether a person should put it
    back home (past RETURN_DRIFT_RESET_M, or out of the hover's reach: the next pickup would start there).
    Where a person's hands may be in the landing (held["by_hand"]: `r` during the return) it's no sample."""
    home, name = held["home_xy"], held["tile_name"]
    flags = {**({"carried": True, "from_xy": held["detection_xy"]} if carried(held["detection_xy"], home) else {}),
             **({"by_hand": True} if held.get("by_hand") else {})}
    tile = landed_tile(tiles, name, home)
    if tile is None:
        print(f"  {name} not seen within {LANDED_WITHIN_M * 1000:.0f}mm of its home after the return")
        session.event("return_landed", tile=name, attempt_dir=held["attempt_dir"], home_xy=home, landed_xy=None, **flags)
        return None, False
    landed = list(tile["xy"])
    error = math.dist(landed, home)
    session.event("return_landed", tile=name, attempt_dir=held["attempt_dir"], home_xy=home,
                  landed_xy=landed, error_m=error, **flags)
    if not held.get("by_hand"):
        append_record(return_record(held["detection_xy"], held["grasp_xy"], placed_xy, landed, tile_name=name,
                                    spot=held["spot"], home_xy=home, session=session.dir.name,
                                    attempt_dir=held["attempt_dir"], **provenance), path)
    hx, hy = correction.apply(*landed, log=lambda *a: None) if correction else landed
    reachable = kinematics.hover_command(hx, hy, hover["offset_m"], hover["gripper"])[0] is not None
    if error <= config.RETURN_DRIFT_WARN_M and reachable:
        return tile, False
    print(f"  {name} landed {error * 1000:.0f}mm from its home "
          f"({(landed[0] - home[0]) * 1000:+.0f}, {(landed[1] - home[1]) * 1000:+.0f})mm"
          + ("" if reachable else ", out of the hover's reach"))
    return tile, not (error <= config.RETURN_DRIFT_RESET_M and reachable)


def _arrays(records):
    det = np.array([r["detection_xy"] for r in records], dtype=float)
    grip = det - np.array([r["grasp_xy"] for r in records], dtype=float)
    miss = np.array([r["landed_xy"] for r in records], dtype=float) - np.array([r["target_xy"] for r in records]) - grip
    return det, grip, miss


def _design(det, grip, kind):
    columns = [np.ones((len(det), 1))]
    if kind in ("grip", "grip+position"):
        columns.append(grip)
    if kind == "grip+position":
        columns.append(det)
    return np.hstack(columns)


def _loo_rms(det, grip, miss, kind):
    """RMS error predicting each position cell's landings from the others' ("none": the baseline's)."""
    cells = position_cells(det)
    errors = []
    for cell in set(cells):
        out = np.array([c == cell for c in cells])
        if kind == "none":
            predicted = np.zeros((out.sum(), 2))
        else:
            coef = np.linalg.lstsq(_design(det[~out], grip[~out], kind), miss[~out], rcond=None)[0]
            predicted = _design(det[out], grip[out], kind) @ coef
        errors.append(np.linalg.norm(miss[out] - predicted, axis=1))
    errors = np.concatenate(errors)
    return float(np.sqrt(np.mean(errors ** 2)))


def _affine(xy):
    xy = np.atleast_2d(xy)
    return np.hstack([np.ones((len(xy), 1)), xy])


def fit_across(records):
    """What a tile carried across the mat needs, from returns that went back where they came from:
    usual: where the jaws close relative to the tile's detection, by position (detection -> grasp - detection);
    aim: where to set down relative to where a tile should land, by position (landed -> target - landed);
    kept: how much of a grasp's stray from `usual` stays with the tile (stray @ kept: x, y)."""
    det, grasp, target, landed = (np.array([r[k] for r in records], dtype=float)
                                  for k in ("detection_xy", "grasp_xy", "target_xy", "landed_xy"))
    usual = np.linalg.lstsq(_affine(det), grasp - det, rcond=None)[0]
    stray = grasp - det - _affine(det) @ usual
    coef = np.linalg.lstsq(np.hstack([_affine(landed), stray]), target - landed, rcond=None)[0]
    return {"usual": usual, "aim": coef[:3], "kept": coef[3:], "bounds": (landed.min(axis=0), landed.max(axis=0))}


def far_landing(record):
    """Did this return land further from home than SAMPLE_WITHIN_M? Likely the next spot's tile taken for it (all backs read
    "back"); it's still in the file, so a real run of big misses would show in the report's count."""
    return math.dist(record["landed_xy"], record.get("home_xy") or record["detection_xy"]) > SAMPLE_WITHIN_M


def load_records(path=None):
    """(fitted, set_aside, excluded) records for the current setup. Excluded: marked so, a person's hands may be
    in the landing, carried across the mat (their detection - grasp is another place's: see fit_across), or landed
    further from home than SAMPLE_WITHIN_M (likely a neighbour's tile: see far_landing)."""
    fingerprint = setup_fingerprint()
    fitted, set_aside, excluded = [], [], []
    for record in read_all(path or config.RETURN_CORRECTIONS):
        if any(record.get(k) != v for k, v in fingerprint.items()):
            set_aside.append(record)
        elif (record.get("excluded") or (record.get("session"), record.get("attempt_dir")) in HAND_MOVED
              or carried(record["detection_xy"], record.get("home_xy") or record["detection_xy"]) or far_landing(record)):
            excluded.append(record)
        else:
            fitted.append(record)
    return fitted, set_aside, excluded


class ReturnCorrection:
    def __init__(self, records, set_aside=(), excluded=()):
        self.records, self.set_aside, self.excluded = list(records), list(set_aside), list(excluded)
        self.n = len(self.records)
        self.kind, self.coef, self.loo_rms = "none", None, {}
        self.across = None
        if not self.records:
            return
        self.det, self.grip, self.miss = _arrays(self.records)
        positions = len(set(position_cells(self.det)))
        if positions < 2:
            return
        if self.n >= MIN_RECORDS["grip+position"] and positions >= MIN_POSITIONS["grip+position"]:
            self.across = fit_across(self.records)
        self.loo_rms["none"] = _loo_rms(self.det, self.grip, self.miss, "none")
        for kind in ("grip", "grip+position"):
            if self.n >= MIN_RECORDS[kind] and positions >= MIN_POSITIONS[kind]:
                self.loo_rms[kind] = _loo_rms(self.det, self.grip, self.miss, kind)
                if self.loo_rms[kind] < (1 - MIN_GAIN) * self.loo_rms[self.kind]:
                    self.kind = kind
        if self.kind != "none":
            self.coef = np.linalg.lstsq(_design(self.det, self.grip, self.kind), self.miss, rcond=None)[0]
            self.bounds = (self.det.min(axis=0), self.det.max(axis=0))

    @classmethod
    def load(cls, path=None):
        return cls(*load_records(path))

    @property
    def name(self):
        """e.g. "grip/95": which return model a return was aimed with."""
        return f"{self.kind}/{self.n}"

    def correction_at(self, detection_xy, grasp_xy):
        """What the baseline misses for this grasp (m): the landing minus the baseline's, capped."""
        if self.kind == "none":
            return 0.0, 0.0
        lo, hi = self.bounds
        det = np.clip(np.asarray(detection_xy, dtype=float), lo - EXTRAPOLATION_MARGIN_M, hi + EXTRAPOLATION_MARGIN_M)
        grip = np.subtract(detection_xy, grasp_xy)
        d = (_design(det[None], grip[None], self.kind) @ self.coef)[0]
        size = math.hypot(*d)
        if size > MAX_CORRECTION_M:
            d = d * MAX_CORRECTION_M / size
        return float(d[0]), float(d[1])

    def target_across(self, home_xy, detection_xy, grasp_xy, log=print):
        """Where to set down a tile carried across the mat so it lands at home_xy (fit_across). The stray's share is
        capped like the correction, and positions are held at the edge of what was measured."""
        fit, margin = self.across, EXTRAPOLATION_MARGIN_M
        lo, hi = fit["bounds"]
        home = np.clip(np.asarray(home_xy, dtype=float), lo - margin, hi + margin)
        pickup = np.clip(np.asarray(detection_xy, dtype=float), lo - margin, hi + margin)
        stray = np.subtract(grasp_xy, detection_xy) - (_affine(pickup) @ fit["usual"])[0]
        kept = stray @ fit["kept"]
        size = math.hypot(*kept)
        if size > MAX_CORRECTION_M:
            kept = kept * MAX_CORRECTION_M / size
        aim = (_affine(home) @ fit["aim"])[0]
        log(f"  carried across the mat: aiming ({aim[0] * 1000:+.1f}, {aim[1] * 1000:+.1f})mm from its home, plus "
            f"({kept[0] * 1000:+.1f}, {kept[1] * 1000:+.1f})mm for a close {math.hypot(*stray) * 1000:.0f}mm off the usual")
        return float(home_xy[0] + aim[0] + kept[0]), float(home_xy[1] + aim[1] + kept[1])

    def target(self, home_xy, detection_xy, grasp_xy, log=print):
        """Where to set the tile down so it lands at home_xy. Back where it came from: the baseline (home + grasp -
        detection: the tile keeps its offset in the jaws), moved by what the model says the baseline misses. Carried
        across the mat: target_across, as the baseline would bring the pickup spot's offset with it."""
        if carried(detection_xy, home_xy):
            if self.across is not None:
                return self.target_across(home_xy, detection_xy, grasp_xy, log)
            log("  carried across the mat with too few returns measured for this setup to aim it: "
                "it may land 20mm+ off, into its neighbours")
        bx, by = (home_xy[0] + grasp_xy[0] - detection_xy[0], home_xy[1] + grasp_xy[1] - detection_xy[1])
        dx, dy = self.correction_at(detection_xy, grasp_xy)
        if self.kind != "none":
            log(f"  return correction ({self.kind}, {self.n} returns): ({-dx * 1000:+.1f}, {-dy * 1000:+.1f})mm")
        return bx - dx, by - dy

    def summary(self):
        if not self.loo_rms:
            aside = f"; {len(self.set_aside)} from another setup set aside" if self.set_aside else ""
            return (f"{self.n} return(s) for this setup{aside}: too few positions to fit — returns aim at "
                    "home + (grasp - detection)")
        loo = ", ".join(f"{k} {v * 1000:.1f}mm" for k, v in self.loo_rms.items())
        return f"{self.kind} from {self.n} returns: landing error, leave-one-position-out: {loo}"


def report():
    model = ReturnCorrection.load()
    print(f"{config.RETURN_CORRECTIONS}  {setup_fingerprint()}")
    if model.set_aside:
        print(f"{len(model.set_aside)} record(s) from a different setup set aside.")
    if model.excluded:
        far = sum(map(far_landing, model.excluded))
        print(f"{len(model.excluded)} record(s) excluded"
              + (f", {far} landed over {SAMPLE_WITHIN_M * 1000:.0f}mm from home (likely a neighbour's tile)." if far else "."))
    print(model.summary())
    if model.kind == "none":
        return
    miss_after = model.miss - _design(model.det, model.grip, model.kind) @ model.coef
    print(f"\n{'position x,y':>16} {'returns':>7} {'baseline miss (mm, mean)':>26} {'after (mm, RMS)':>16}")
    cells = position_cells(model.det)
    for cell in sorted(set(cells)):
        rows = np.array([c == cell for c in cells])
        mean = model.miss[rows].mean(axis=0) * 1000
        after = np.sqrt(np.mean(np.sum(miss_after[rows] ** 2, axis=1))) * 1000
        print(f"  ({cell[0] * POSITION_CELL_M:.2f},{cell[1] * POSITION_CELL_M:+.2f}) {rows.sum():>7} "
              f"{f'({mean[0]:+5.1f},{mean[1]:+5.1f})':>26} {after:>16.1f}")
    if model.across is not None:
        kept = model.across["kept"]
        print(f"\nhow far the jaws closed from their usual place, the share that stays with the tile: x {kept[0, 0]:.2f}, "
              f"y {kept[1, 1]:.2f} (1: the jaws don't move it; 0: they centre it). A steady margin at the close isn't a stray.")


if __name__ == "__main__":
    argparse.ArgumentParser(description="The return correction fit").parse_args()
    report()
