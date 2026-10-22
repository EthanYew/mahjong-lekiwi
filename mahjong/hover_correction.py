"""Measured hover correction: maps a RealSense tile detection (x, y) to the hover target that actually
puts the jaws over that tile.

scripts/collect_corrections.py records, per tile, where the operator lined the jaws up with the
gamepad (the tool point of the commanded joints) against the detection. The difference absorbs every
hover error source at once — the RealSense calibration (fitted in a small patch above the table, so
far-right tiles are extrapolation), servo sag, the tool point — without modelling any of them.

Model, chosen by leave-one-out error so it never claims more than the data supports: a constant
offset, or an affine map (>= MIN_POINTS_AFFINE points, used only if it beats the offset by
AFFINE_MIN_GAIN). Corrections are capped, and held at their edge value past the measured tiles.
Records belong to the setup they were measured on (servo profile + T_cam_to_base contents);
recalibrating either sets them aside. A record with an "excluded" reason stays in the file unfitted.

    python -m mahjong.hover_correction      # the fit, per-tile errors, and a preview grid
"""
import hashlib
import json
import math
from datetime import datetime

import numpy as np

from . import config

MIN_POINTS_AFFINE = 6
AFFINE_MIN_GAIN = 0.10
MAX_CORRECTION_M = 0.05
EXTRAPOLATION_MARGIN_M = 0.03


def setup_fingerprint():
    sha = hashlib.sha256(config.T_CAM_TO_BASE.read_bytes()).hexdigest()[:12]
    return {"follower_id": config.FOLLOWER_ID, "t_cam_to_base_sha": sha}


def append_record(record, path=None):
    path = path or config.HOVER_CORRECTIONS
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **setup_fingerprint(), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_records(path=None):
    """(fitted, set_aside, excluded) records for the current setup."""
    path = path or config.HOVER_CORRECTIONS
    if not path.exists():
        return [], [], []
    fingerprint = setup_fingerprint()
    fitted, set_aside, excluded = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if any(record.get(k) != v for k, v in fingerprint.items()):
            set_aside.append(record)
        elif record.get("excluded"):
            excluded.append(record)
        else:
            fitted.append(record)
    return fitted, set_aside, excluded


def _design(xy, kind):
    return np.ones((len(xy), 1)) if kind == "offset" else np.column_stack([xy, np.ones(len(xy))])


def _fit(xy, delta, kind):
    return np.linalg.lstsq(_design(xy, kind), delta, rcond=None)[0]


def _loo_rms(xy, delta, kind):
    errors = []
    for i in range(len(xy)):
        keep = np.arange(len(xy)) != i
        errors.append(np.linalg.norm(_design(xy[i:i + 1], kind) @ _fit(xy[keep], delta[keep], kind) - delta[i]))
    return float(np.sqrt(np.mean(np.square(errors))))


class HoverCorrection:
    def __init__(self, records, set_aside=(), excluded=()):
        self.records, self.set_aside, self.excluded = list(records), list(set_aside), list(excluded)
        self.n = len(self.records)
        self.kind, self.coef, self.loo_rms = "none", None, {}
        if not self.records:
            return
        self.xy = np.array([r["detection_xy"] for r in self.records], dtype=float)
        self.delta = np.array([r["delta_xy"] for r in self.records], dtype=float)
        self.raw_rms = float(np.sqrt(np.mean(np.sum(self.delta ** 2, axis=1))))
        if self.n >= 2:
            self.loo_rms["offset"] = _loo_rms(self.xy, self.delta, "offset")
        if self.n >= MIN_POINTS_AFFINE:
            self.loo_rms["affine"] = _loo_rms(self.xy, self.delta, "affine")
        affine = "affine" in self.loo_rms and self.loo_rms["affine"] < (1 - AFFINE_MIN_GAIN) * self.loo_rms["offset"]
        self.kind = "affine" if affine else "offset"
        self.coef = _fit(self.xy, self.delta, self.kind)
        self.bounds = (self.xy.min(axis=0), self.xy.max(axis=0))

    @classmethod
    def load(cls, path=None):
        return cls(*load_records(path))

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
        return f"{self.kind} from {self.n} points: raw RMS {self.raw_rms * 1000:.1f}mm, leave-one-out {loo}"


def report():
    model = HoverCorrection.load()
    print(f"{config.HOVER_CORRECTIONS}  {setup_fingerprint()}")
    if model.set_aside:
        print(f"{len(model.set_aside)} record(s) from a different setup set aside.")
    for r in model.excluded:
        print(f"excluded: {r.get('tile_name', '')} at ({r['detection_xy'][0]:.3f}, {r['detection_xy'][1]:+.3f}) — {r['excluded']}")
    print(model.summary())
    if model.kind == "none":
        return
    print(f"\n{'tile':>5} {'detected x,y':>16} {'jaws - detection (mm)':>24} {'after correction':>17}")
    for r, (x, y), (dx, dy) in zip(model.records, model.xy, model.delta):
        cx, cy = model.delta_at(x, y)
        print(f"{r.get('tile_name', ''):>5} ({x:6.3f},{y:+6.3f})   ({dx * 1000:+6.1f},{dy * 1000:+6.1f})"
              f"   {math.hypot(dx - cx, dy - cy) * 1000:5.1f}mm")
    if model.kind == "affine":
        J = np.eye(2) + model.coef[:2].T
        print(f"\naffine: rotation {math.degrees(math.atan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1])):+.2f} deg, "
              f"scale {math.sqrt(abs(np.linalg.det(J))):.3f}")
    ys = [0.18, 0.09, 0.0, -0.09, -0.17]
    print("\ncorrection (mm), y left -> right, x near -> far:\n        " + "".join(f"{f'y={y:+.2f}':>16}" for y in ys))
    for x in (0.15, 0.20, 0.25):
        print(f"x={x:.2f}  " + "".join(f"{'({:+5.1f},{:+5.1f})'.format(*(v * 1000 for v in model.delta_at(x, y))):>16}" for y in ys))


if __name__ == "__main__":
    report()
