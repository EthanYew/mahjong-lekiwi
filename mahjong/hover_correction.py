"""Measured hover correction: maps a RealSense tile detection (x, y) to the hover target that actually
puts the jaws over that tile.

Samples are where the jaws really were over a tile (the tool point of the commanded joints) minus the
detection. The difference absorbs every hover error source at once — the RealSense calibration
(fitted in a small patch above the table, so far-right tiles are extrapolation), servo sag, the tool
point — without modelling any of them. Each record has a `source`:
- gamepad: scripts/collect_corrections.py, the operator lining the jaws up at the hover (records from
  before sources existed are gamepad);
- demo:    where the operator closed the gripper in a recorded episode (scripts/hover_samples.py);
- bench:   where a policy closed the gripper on a successful bench attempt (bench.py, hover_samples.py).
The live hover is fitted only from the sources data/hover_model.json lists, with their weights
(gamepad only without it), so new kinds of samples can pile up and be compared
(`--sources`) before they're trusted.

Model, chosen by leave-one-position-out error so it never claims more than the data supports: a
constant offset, or an affine map (>= MIN_POINTS_AFFINE points, used only if it beats the offset by
AFFINE_MIN_GAIN). Leaving out a whole POSITION_CELL_M cell at a time stops repeated attempts at one
tile from vouching for each other. Corrections are capped, and held at their edge value past the
measured tiles. Records belong to the setup they were measured on (servo profile + T_cam_to_base
contents); recalibrating either sets them aside. A record with an "excluded" reason stays in the file
unfitted.

    python -m mahjong.hover_correction                              # the live fit, per-tile errors, a preview grid
    python -m mahjong.hover_correction --sources gamepad,demo:0.5   # what the hover would become with these
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from . import config

MIN_POINTS_AFFINE = 6
AFFINE_MIN_GAIN = 0.10
MAX_CORRECTION_M = 0.05
EXTRAPOLATION_MARGIN_M = 0.03
POSITION_CELL_M = 0.03
DEFAULT_SOURCES = {"gamepad": 1.0}


def setup_fingerprint():
    sha = hashlib.sha256(config.T_CAM_TO_BASE.read_bytes()).hexdigest()[:12]
    return {"follower_id": config.FOLLOWER_ID, "t_cam_to_base_sha": sha}


def source_of(record):
    return record.get("source", "gamepad")


def active_sources(path=None):
    """{source: weight} the live hover is fitted from."""
    path = path or config.HOVER_MODEL
    return dict(json.loads(path.read_text())["sources"]) if path.exists() else dict(DEFAULT_SOURCES)


def parse_sources(text):
    """"gamepad,demo:0.5" -> {"gamepad": 1.0, "demo": 0.5}"""
    sources = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        name, _, weight = item.partition(":")
        sources[name] = float(weight or 1.0)
    return sources


def grasp_record(source, detection_xy, grasp_xy, **provenance):
    """A sample from where the gripper closed (demo or bench): the grasp tool point minus the detection."""
    return {"source": source, "detection_xy": [float(v) for v in detection_xy],
            "grasp_xy": [float(v) for v in grasp_xy],
            "delta_xy": [float(grasp_xy[0] - detection_xy[0]), float(grasp_xy[1] - detection_xy[1])], **provenance}


def append_record(record, path=None):
    path = path or config.HOVER_CORRECTIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **setup_fingerprint(), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_all(path=None):
    path = path or config.HOVER_CORRECTIONS
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_records(path=None, sources=None):
    """(fitted, set_aside, excluded, unused) records for the current setup; unused are from sources
    not in `sources` (default: the active ones)."""
    sources = active_sources() if sources is None else sources
    fingerprint = setup_fingerprint()
    fitted, set_aside, excluded, unused = [], [], [], []
    for record in read_all(path):
        if any(record.get(k) != v for k, v in fingerprint.items()):
            set_aside.append(record)
        elif record.get("excluded"):
            excluded.append(record)
        elif source_of(record) not in sources:
            unused.append(record)
        else:
            fitted.append(record)
    return fitted, set_aside, excluded, unused


def position_cells(xy):
    return [tuple(c) for c in np.round(np.asarray(xy) / POSITION_CELL_M).astype(int)]


def _design(xy, kind):
    return np.ones((len(xy), 1)) if kind == "offset" else np.column_stack([xy, np.ones(len(xy))])


def _fit(xy, delta, kind, w):
    sw = np.sqrt(w)[:, None]
    return np.linalg.lstsq(_design(xy, kind) * sw, delta * sw, rcond=None)[0]


def _loo_rms(xy, delta, kind, w):
    """Weighted RMS error predicting each position cell from the others."""
    cells = position_cells(xy)
    errors, weights = [], []
    for cell in set(cells):
        out = np.array([c == cell for c in cells])
        coef = _fit(xy[~out], delta[~out], kind, w[~out])
        errors.append(np.linalg.norm(_design(xy[out], kind) @ coef - delta[out], axis=1))
        weights.append(w[out])
    errors, weights = np.concatenate(errors), np.concatenate(weights)
    return float(np.sqrt(np.sum(weights * errors ** 2) / np.sum(weights)))


class HoverCorrection:
    def __init__(self, records, set_aside=(), excluded=(), unused=(), weights=None):
        """weights: {source: weight}; a source missing from it weighs 1."""
        self.records, self.set_aside, self.excluded, self.unused = list(records), list(set_aside), list(excluded), list(unused)
        self.n = len(self.records)
        self.sources = Counter(source_of(r) for r in self.records)
        self.kind, self.coef, self.loo_rms = "none", None, {}
        if not self.records:
            return
        self.xy = np.array([r["detection_xy"] for r in self.records], dtype=float)
        self.delta = np.array([r["delta_xy"] for r in self.records], dtype=float)
        self.w = np.array([(weights or {}).get(source_of(r), 1.0) for r in self.records], dtype=float)
        self.raw_rms = float(np.sqrt(np.mean(np.sum(self.delta ** 2, axis=1))))
        positions = len(set(position_cells(self.xy)))
        if positions >= 2:
            self.loo_rms["offset"] = _loo_rms(self.xy, self.delta, "offset", self.w)
        if self.n >= MIN_POINTS_AFFINE and positions >= 3:
            self.loo_rms["affine"] = _loo_rms(self.xy, self.delta, "affine", self.w)
        affine = "affine" in self.loo_rms and self.loo_rms["affine"] < (1 - AFFINE_MIN_GAIN) * self.loo_rms["offset"]
        self.kind = "affine" if affine else "offset"
        self.coef = _fit(self.xy, self.delta, self.kind, self.w)
        self.bounds = (self.xy.min(axis=0), self.xy.max(axis=0))

    @classmethod
    def load(cls, path=None, sources=None):
        """The fit from `sources` ({source: weight}; default: data/hover_model.json's)."""
        sources = active_sources() if sources is None else sources
        return cls(*load_records(path, sources), weights=sources)

    @property
    def name(self):
        """e.g. "gamepad/affine/23" — which hover an attempt was made from."""
        return f"{'+'.join(sorted(self.sources)) or 'none'}/{self.kind}/{self.n}"

    def delta_at(self, x, y):
        if self.kind == "none":
            return 0.0, 0.0
        lo, hi = self.bounds
        x = min(max(x, lo[0] - EXTRAPOLATION_MARGIN_M), hi[0] + EXTRAPOLATION_MARGIN_M)
        y = min(max(y, lo[1] - EXTRAPOLATION_MARGIN_M), hi[1] + EXTRAPOLATION_MARGIN_M)
        d = (_design(np.array([[x, y]]), self.kind) @ self.coef)[0]
        size = math.hypot(*d)
        if size > MAX_CORRECTION_M:
            d = d * MAX_CORRECTION_M / size
        return float(d[0]), float(d[1])

    def apply(self, x, y, log=print):
        dx, dy = self.delta_at(x, y)
        if self.kind != "none":
            lo, hi = self.bounds
            outside = max(lo[0] - x, x - hi[0], lo[1] - y, y - hi[1])
            note = (f" (tile is {outside * 1000:.0f}mm past the measured tiles; edge value used)"
                    if outside > EXTRAPOLATION_MARGIN_M else "")
            log(f"  hover correction ({self.kind}, {self.n} points): ({dx * 1000:+.1f}, {dy * 1000:+.1f})mm{note}")
        return x + dx, y + dy

    def summary(self):
        if self.kind == "none":
            aside = f"; {len(self.set_aside)} from another setup set aside" if self.set_aside else ""
            return f"no hover corrections for this setup{aside} — run scripts/collect_corrections.py"
        loo = ", ".join(f"{k} {v * 1000:.1f}mm" for k, v in self.loo_rms.items())
        sources = ", ".join(f"{k} {v}" for k, v in sorted(self.sources.items()))
        return (f"{self.kind} from {self.n} points ({sources}): raw RMS {self.raw_rms * 1000:.1f}mm, "
                f"leave-one-position-out {loo}")


GRID_YS = [0.18, 0.09, 0.0, -0.09, -0.17]
GRID_XS = (0.15, 0.20, 0.25)


def _grid(title, value_at):
    print(f"\n{title}, y left -> right, x near -> far:\n        " + "".join(f"{f'y={y:+.2f}':>16}" for y in GRID_YS))
    for x in GRID_XS:
        print(f"x={x:.2f}  " + "".join(f"{'({:+5.1f},{:+5.1f})'.format(*(v * 1000 for v in value_at(x, y))):>16}"
                                       for y in GRID_YS))


def residuals_by_position(model, records):
    """What `model` gets wrong, per position cell and source: {(cell, source): [residual xy, ...]} (m).
    Bench and demo residuals in the same cell disagreeing means the setup changed since recording."""
    out = defaultdict(list)
    for r, cell in zip(records, position_cells([r["detection_xy"] for r in records])):
        x, y = r["detection_xy"]
        out[(cell, source_of(r))].append(np.subtract(r["delta_xy"], model.delta_at(x, y)))
    return out


def report(sources=None):
    active = HoverCorrection.load()
    print(f"{config.HOVER_CORRECTIONS}  {setup_fingerprint()}")
    print(f"live hover fitted from {active_sources()} ({config.HOVER_MODEL.name})")
    if active.set_aside:
        print(f"{len(active.set_aside)} record(s) from a different setup set aside.")
    for r in active.excluded:
        if source_of(r) == "gamepad":
            print(f"excluded: {r.get('tile_name', '')} at ({r['detection_xy'][0]:.3f}, {r['detection_xy'][1]:+.3f}) — {r['excluded']}")
    reasons = Counter(r["excluded"] for r in active.excluded if source_of(r) != "gamepad")
    for reason, n in reasons.items():
        print(f"excluded: {n} sample(s) — {reason}")
    if active.unused:
        print(f"not fitted (source not listed): {dict(Counter(source_of(r) for r in active.unused))}")
    print(active.summary())
    model = HoverCorrection.load(sources=sources) if sources else active
    if sources:
        print(f"\ncandidate from {sources}: {model.summary()}")
    if model.kind == "none":
        return
    if not sources:
        print(f"\n{'tile':>5} {'detected x,y':>16} {'jaws - detection (mm)':>24} {'after correction':>17}")
        for r, (x, y), (dx, dy) in zip(model.records, model.xy, model.delta):
            cx, cy = model.delta_at(x, y)
            print(f"{r.get('tile_name', ''):>5} ({x:6.3f},{y:+6.3f})   ({dx * 1000:+6.1f},{dy * 1000:+6.1f})"
                  f"   {math.hypot(dx - cx, dy - cy) * 1000:5.1f}mm")
    if model.kind == "affine":
        J = np.eye(2) + model.coef[:2].T
        print(f"\naffine: rotation {math.degrees(math.atan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1])):+.2f} deg, "
              f"scale {math.sqrt(abs(np.linalg.det(J))):.3f}")
    _grid("correction (mm)", model.delta_at)
    if sources and active.kind != "none":
        _grid("change from the live hover (mm)",
              lambda x, y: np.subtract(model.delta_at(x, y), active.delta_at(x, y)))
    samples = active.records + active.unused
    if active.kind != "none" and len(Counter(map(source_of, samples))) > 1:
        print("\nwhat the live hover misses (mm, median per position; columns are sources):")
        table = residuals_by_position(active, samples)
        names = sorted({s for _, s in table})
        print(f"  {'cell x,y':>14} " + "".join(f"{n:>22}" for n in names))
        for cell in sorted({c for c, _ in table}):
            cols = []
            for n in names:
                res = table.get((cell, n))
                cols.append(f"{len(res):>3} ({np.median([v[0] for v in res]) * 1000:+5.1f},"
                            f"{np.median([v[1] for v in res]) * 1000:+5.1f})" if res else "")
            print(f"  ({cell[0] * POSITION_CELL_M:.2f},{cell[1] * POSITION_CELL_M:+.2f}) " + "".join(f"{c:>22}" for c in cols))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="The hover correction fit")
    p.add_argument("--sources", type=parse_sources, default=None,
                   help="fit a candidate from these sources, e.g. gamepad,demo:0.5,bench:0.5, and compare it with the live one")
    report(p.parse_args().sources)
