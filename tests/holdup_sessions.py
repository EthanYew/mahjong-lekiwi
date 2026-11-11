"""A simulated hold-up session on disk, as base_cam_collect.py leaves one: `holdup_shot` events and their photos, for the
tests of what reads them (base_cam.rows, base_cam_train.py). Photos have no pose until a test gives them one."""
import json

import numpy as np

from mahjong import base_cam


def frame(rng):
    """Noise with a tile-sized lighter patch: enough for the readers' framing."""
    image = rng.integers(0, 60, (480, 640, 3), dtype=np.uint8)
    image[160:320, 240:400] = 220
    return image


def write_session(root, tiles=("5C", "3C", "2C"), holds=4, frames=1, name="20261101_120000_holdup", sim=True):
    """Every tile, each way up, `holds` holds of `frames` frames. Returns the session folder."""
    folder = root / name
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    events = [{"type": "session_start", "kind": "holdup", "sim": sim},
              {"type": "holdup_conditions", "conditions": {"daylight": "none", "lamp": True, "ceiling_lights": True}}]
    n = 0
    for tile in tiles:
        for kind in base_cam.KINDS:
            for hold in range(1, holds + 1):
                n += 1
                shot = {"tile": tile, "kind": kind, "hold": hold}
                files = base_cam.save_shot(folder, shot, [frame(rng) for _ in range(frames)])
                events.append({"type": "holdup_shot", "shot": n, **shot, "files": files})
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return folder
