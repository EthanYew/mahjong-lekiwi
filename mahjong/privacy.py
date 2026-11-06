"""Blur faces in the base camera's photos before they're kept.

The base camera looks out over the room, so a person near it ends up in the training photos. Faces are found with a pose
model (`assets/yolo11n-pose.pt`, Ultralytics' small public COCO pose weights, run on the CPU): it finds each person and the
nose, eyes and ears on them, which keeps working for a face turned away, tilted, small or cut off by the frame's edge. OpenCV's
Haar face cascades did not: on the first 262 photos they blurred the printing on the held tile (the tile's red and black marks
look like faces to them) and still left clear faces. Where a person is found but too little of the head shows to place it, the
top of their box is blurred instead.

It lowers the chance of a face in a photo, and doesn't rule it out: look through a sheet of the photos before sharing any
(`scripts/blur_faces.py` makes one). The weights are not a repo secret or training asset, just a tool: to get them again,
`curl -L -o assets/yolo11n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt`.
"""
import cv2
import numpy as np

from . import config

WEIGHTS = config.REPO / "assets" / "yolo11n-pose.pt"
PERSON_CONF = 0.25
KEYPOINT_CONF = 0.35
HEAD_KEYPOINTS = (0, 1, 2, 3, 4)        # nose, eyes, ears (COCO order)
HEAD_PER_KEYPOINT_SPAN = 1.8            # the head is this much wider than the span of its visible keypoints (2.4 on the first try,
                                        # which with PAD covered ~3x the face and took the held tile's printing with it)
HEAD_OF_PERSON = 0.20                   # no head keypoints: the head is the top of the person's box, this share of its height
HEAD_MIN_OF_PERSON = 0.09               # a head is at least this share of the person's height, and
MIN_HEAD_PX = 24                        # at least this wide
MAX_PERSON_AREA = 0.65                  # a "person" bigger than this share of the frame is the tile: on the first photos the tile's
                                        # false people covered 0.79 and 0.95 of it, and real people up to 0.52 (close to the camera)
PAD = 0.10                              # the blur covers the head box widened by this share on every side (was 0.25)

_model = None


def require():
    """Fail now, with how to get the weights, rather than part way through an arm run."""
    if not WEIGHTS.exists():
        raise SystemExit(f"{WEIGHTS} is missing: it blurs faces in the base camera's photos. Get it with\n"
                         "  curl -L -o assets/yolo11n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt")


def _pose():
    global _model
    if _model is None:
        require()
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
    return _model


def head_box(person_box, keypoints, confidence):
    """(x, y, w, h) of one person's head: around the visible nose, eyes and ears, else the top of the person's box."""
    visible = [k for k, c in zip(keypoints[list(HEAD_KEYPOINTS)], confidence[list(HEAD_KEYPOINTS)]) if c >= KEYPOINT_CONF]
    x0, y0, x1, y1 = person_box
    if visible:
        points = np.array(visible)
        cx, cy = points.mean(axis=0)
        span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 0.0)
        size = max(span * HEAD_PER_KEYPOINT_SPAN, MIN_HEAD_PX, HEAD_MIN_OF_PERSON * (y1 - y0))
        return (int(cx - size / 2), int(cy - size / 2), int(size), int(size))
    height = (y1 - y0) * HEAD_OF_PERSON
    width = min(x1 - x0, height * 1.1)
    return (int((x0 + x1) / 2 - width / 2), int(y0), int(width), int(height))


def find_faces(image):
    """Head boxes (x, y, w, h) of every person in a BGR image."""
    result = _pose()(image, conf=PERSON_CONF, verbose=False)[0]
    if result.boxes is None or not len(result.boxes):
        return []
    people = result.boxes.xyxy.cpu().numpy()
    has_keypoints = result.keypoints is not None
    keypoints = result.keypoints.xy.cpu().numpy() if has_keypoints else np.zeros((len(people), 17, 2))
    confidence = result.keypoints.conf.cpu().numpy() if has_keypoints and result.keypoints.conf is not None \
        else np.zeros((len(people), 17))
    return heads(people, keypoints, confidence, image.shape)


def heads(people, keypoints, confidence, shape):
    """The head box of each person (arrays from the pose model: boxes xyxy, 17 keypoints and their confidences each), leaving
    out a 'person' bigger than MAX_PERSON_AREA of the frame."""
    frame_area = shape[0] * shape[1]
    out = []
    for box, kp, conf in zip(people, keypoints, confidence):
        if (box[2] - box[0]) * (box[3] - box[1]) > MAX_PERSON_AREA * frame_area:
            continue                # the held tile close to the camera: its marks pass for a face, 0.7 confident, on a box the size of the frame
        out.append(head_box(box, kp, conf))
    return out


def blur_boxes(image, boxes, pad=PAD):
    """A copy of `image` with each box (widened by `pad`) blurred beyond recognition."""
    out = image.copy()
    height, width = out.shape[:2]
    for x, y, w, h in boxes:
        x0, y0 = max(0, int(x - pad * w)), max(0, int(y - pad * h))
        x1, y1 = min(width, int(x + w + pad * w)), min(height, int(y + h + pad * h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        region = out[y0:y1, x0:x1]
        # a Gaussian alone leaves a face recognisable at a small radius: shrink it to a few pixels and blow it back up first
        small = cv2.resize(region, (max(2, (x1 - x0) // 12), max(2, (y1 - y0) // 12)), interpolation=cv2.INTER_AREA)
        coarse = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        k = (max(3, ((x1 - x0) // 4) | 1), max(3, ((y1 - y0) // 4) | 1))
        out[y0:y1, x0:x1] = cv2.GaussianBlur(coarse, k, 0)
    return out


def blur_faces(image, finder=None):
    """(the image with faces blurred, the boxes found). `finder` replaces find_faces (tests)."""
    boxes = (finder or find_faces)(image)
    return (blur_boxes(image, boxes) if boxes else image), boxes


def blur_file(path, apply=False, finder=None):
    """The head boxes found in the photo at `path`; with `apply` the file is rewritten blurred (same name, same JPEG)."""
    image = cv2.imread(str(path))
    if image is None:
        return []
    out, boxes = blur_faces(image, finder)
    if boxes and apply:
        cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return boxes
