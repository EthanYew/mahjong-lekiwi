"""A second read of a tile's name, from the wrist camera at the hover (docs/name_check_plan.md).

The RealSense sees a tile at about 24x18 px, so some of its wrong names are confident ones (v7x3 read a clear 3C as 2C at
0.96). The wrist camera sees the same tile from a few cm away. A small classifier reads that frame, from all 34 names, and is
never told what the detector said. The rule play applies (scripts/play.py) until this passes an exam:

    agreement changes nothing; disagreement asks a person. A read never names a tile on its own.

So a poor reader costs questions, not wrong tiles. Removing a question because the two agree is a decision for after the
exam (the plan's gate: 0 wrong names accepted, towards ~1000 reads), not something this code does.

Trains on the wrist photos (mahjong/wrist_photos.py) whose names a person gave: wrist_capture.py views and reviewed photos.
A bench photo no one reviewed is named by the detector, and on the mat those were often wrong, so it's left out. Exam
sessions (data/name_check_exam.json) never train it.
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config, detector, game, wrist_photos
from .session import read_jsonl
from .wrist_check import EXPECTED_PX

WEIGHTS = config.TILE_DETECTOR.parent / "name_check.pt"
EXAM = config.DATA / "name_check_exam.json"
EVAL = config.DATA / "name_check_eval"
CROP_PX = 240               # around where a hovered tile sits: 240 beat 360 on all 3 seeds of the first test
INPUT_PX = 224
EPOCHS, BATCH, LR = 40, 64, 1e-3


def crop(image, side=CROP_PX, size=INPUT_PX):
    """The square of the frame around where a hovered tile sits, resized for the classifier. A wider crop lets neighbours in."""
    x, y, half = EXPECTED_PX[0], EXPECTED_PX[1], side // 2
    h, w = image.shape[:2]
    x0, y0 = min(max(x - half, 0), w - side), min(max(y - half, 0), h - side)
    return cv2.resize(image[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)


# --- which photos ---

def load_exam(path=None):
    path = Path(path or EXAM)
    return json.loads(path.read_text()) if path.exists() else {"sessions": {}}


def training_photos(sessions_dir=None, review=None, exam=None):
    """The photos to train on: a person gave the name, the name can be played, and the session isn't in the exam."""
    review = wrist_photos.load_review() if review is None else review
    sessions = (load_exam() if exam is None else exam)["sessions"]
    return [p for p in wrist_photos.labelled(wrist_photos.photos(sessions_dir), review)
            if game.is_playable(p["name"]) and p["session"] not in sessions
            and (p["source"] == "wrist_capture" or p["reviewed"])]


def _events(folder):
    return read_jsonl(folder / "events.jsonl") if (folder / "events.jsonl").exists() else []


def _detected(folder):
    """{photo key: what the detector read}, from a session's attempts and wrist_capture views. {placement id: a person's name}."""
    reads, truths = {}, {}
    for event in _events(folder):
        if event["type"] == "placement" and event.get("truth"):
            truths[event["id"]] = event["truth"]
        elif event["type"] == "attempt":
            reads[f"{folder.name}/{event['attempt_dir']}/wrist_0000.jpg"] = (event["tile"], event.get("placement"))
        elif event["type"] == "wrist_view":
            for file in event["files"]:
                reads[f"{folder.name}/{file}"] = (event.get("detected_as"), None)
    return reads, truths


def exam_rows(sessions_dir=None, exam=None, review=None):
    """The exam's photos, with a person's word on each: [{"key", "path", "truth", "yolo", "session", "group"}]. A bench photo's
    truth is its placement's (the check before trials) or a review's; one with neither isn't exam material and is left out."""
    sessions_dir = Path(sessions_dir or config.SESSIONS)
    review = wrist_photos.load_review() if review is None else review
    sessions = (load_exam() if exam is None else exam)["sessions"]
    rows = []
    for name in sessions:
        folder = sessions_dir / name
        if not folder.is_dir():
            raise SystemExit(f"No session {folder}: hardware sessions are per checkout; run this from the one that recorded it")
        reads, truths = _detected(folder)
        for photo in wrist_photos.photos(sessions_dir):
            if photo["session"] != name:
                continue
            yolo, placement = reads.get(photo["key"], (photo["name"], None))
            verdict = review.get(photo["key"])
            if verdict is not None:
                truth = wrist_photos.name_of(photo, review)
            elif photo["source"] == "wrist_capture":
                truth = photo["name"]
            else:
                truth = truths.get(placement)
            if truth is not None:
                rows.append({"key": photo["key"], "path": photo["path"], "truth": truth, "yolo": yolo, "session": name,
                             "group": placement or photo["group"]})
    return rows


def score(rows):
    """rows with "truth", "yolo" and "said". The gate is `accepted_wrong`: the two agreed on a wrong name, which nothing would
    catch. `asks` is what the check costs. Photos of one tile aren't independent, so `tiles` counts the tiles behind them."""
    n = len(rows)
    right = sum(r["said"] == r["truth"] for r in rows)
    accepted_wrong = [r for r in rows if r["said"] == r["yolo"] and r["yolo"] != r["truth"]]
    yolo_wrong = [r for r in rows if r["yolo"] != r["truth"]]
    asks = sum(r["said"] != r["yolo"] for r in rows)
    agreed = n - asks
    return {"photos": n, "tiles": len({r["group"] for r in rows}), "names": len({r["truth"] for r in rows}),
            "right": right, "right_rate": right / n if n else 0.0,
            "yolo_right": n - len(yolo_wrong), "yolo_wrong": len(yolo_wrong),
            "yolo_wrong_caught": sum(r["said"] != r["yolo"] for r in yolo_wrong),
            "accepted_wrong": len(accepted_wrong), "agreed": agreed,
            "accepted_wrong_bound": detector.wrong_rate_bound(len(accepted_wrong), agreed) if agreed else 1.0,
            "asks": asks, "ask_rate": asks / n if n else 0.0,
            "misreads": Counter(f"{r['truth']}->{r['said']}" for r in rows if r["said"] != r["truth"]).most_common()}


# --- the classifier (torch is imported here, so play and the tests start fast without it) ---

def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(n_classes):
    """YOLO11n-cls from its ImageNet weights, with a head for the tile names."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import ClassificationModel
    model = YOLO("yolo11n-cls.pt").model
    ClassificationModel.reshape_outputs(model, n_classes)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def _tensor(images, device):
    import torch
    array = np.stack([crop(image)[:, :, ::-1] for image in images])       # RGB
    return torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div(255).to(device)


def _augment(x):
    """Turned tiles (the wrist rolls, tiles sit askew), off-centre tiles, other light. No flips: they change a face."""
    import torch
    import torch.nn.functional as F
    n, device = len(x), x.device
    uniform = lambda low, high, *shape: torch.empty(n, *shape, device=device).uniform_(low, high)
    angle, scale, tx, ty = uniform(-0.35, 0.35), uniform(0.85, 1.15), uniform(-0.12, 0.12), uniform(-0.12, 0.12)
    theta = torch.stack([torch.stack([angle.cos() * scale, -angle.sin() * scale, tx], 1),
                         torch.stack([angle.sin() * scale, angle.cos() * scale, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)
    return (x * uniform(0.75, 1.25, 3, 1, 1) + uniform(-0.1, 0.1, 1, 1, 1)).clamp(0, 1)


def train(photos, seed=0, epochs=EPOCHS, batch=BATCH, device=None, log=print):
    """A checkpoint trained on `photos`: {"classes", "model", ...}. Thin names are drawn as often as common ones."""
    import random

    import torch
    import torch.nn.functional as F
    device = device or _device()
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    classes = sorted({p["name"] for p in photos})
    index = {name: i for i, name in enumerate(classes)}
    x = _tensor([cv2.imread(str(p["path"])) for p in photos], device)
    y = torch.tensor([index[p["name"]] for p in photos], device=device)
    counts = Counter(p["name"] for p in photos)
    weights = torch.tensor([1 / counts[p["name"]] for p in photos], device=device)
    model = build_model(len(classes)).to(device)
    steps = max(1, len(photos) // batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps)
    model.train()
    for epoch in range(epochs):
        for _ in range(steps):
            pick = torch.multinomial(weights, batch, replacement=True)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = F.cross_entropy(model(_augment(x[pick])), y[pick], label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
        if (epoch + 1) % 10 == 0:
            log(f"  epoch {epoch + 1}/{epochs}: loss {loss.item():.3f}")
    model.eval()
    return {"classes": classes, "model": model.cpu(), "crop_px": CROP_PX, "input_px": INPUT_PX, "seed": seed, "epochs": epochs,
            "photos": len(photos), "sessions": sorted({p["session"] for p in photos}),
            "trained": datetime.now().isoformat(timespec="seconds")}


def save(checkpoint, path=None):
    import torch
    path = Path(path or WEIGHTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


class Reader:
    """Reads a tile's name from a wrist frame: {"name", "confidence"}."""

    def __init__(self, path=None, device=None):
        import torch
        self.path = Path(path or WEIGHTS)
        self.device = device or _device()
        checkpoint = torch.load(self.path, map_location=self.device, weights_only=False)      # our own file, in assets/
        self.classes = checkpoint["classes"]
        self.model = checkpoint["model"].to(self.device).float().eval()

    def read_many(self, images, batch=64):
        import torch
        out = []
        with torch.no_grad():
            for part in _tensor(images, self.device).split(batch):
                result = self.model(part)
                probs = result[0] if isinstance(result, (tuple, list)) else result.softmax(1)
                for p in probs.float().cpu().numpy():
                    best = int(p.argmax())
                    out.append({"name": self.classes[best], "confidence": round(float(p[best]), 4)})
        return out

    def read(self, image):
        return self.read_many([image])[0]


def load_reader(path=None):
    """The installed reader, or None if there are no weights (play then runs without the second read, and says so)."""
    path = Path(path or WEIGHTS)
    return Reader(path) if path.exists() else None
