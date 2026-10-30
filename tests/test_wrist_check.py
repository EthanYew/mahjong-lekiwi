"""The wrist camera's close look: is a tile's face between the jaw tips at the hover."""
import cv2
import numpy as np

from mahjong import wrist_check


def frame(*centres):
    image = np.full((480, 640, 3), 25, np.uint8)             # the black table top
    for u, v in centres:
        box = cv2.boxPoints(((u, v), (140, 120), 0.0))
        cv2.fillPoly(image, [np.round(box).astype(np.int32)], (225, 235, 240))
        cv2.circle(image, (int(u), int(v)), 6, (40, 40, 40), -1)    # a printed symbol: closed over
    return image


def test_a_face_between_the_jaw_tips_is_present_and_a_neighbour_is_not():
    u, v = wrist_check.EXPECTED_PX
    assert wrist_check.face_fill(frame((u, v))) > 0.95
    assert wrist_check.face_fill(frame((u + 20, v - 20))) > 0.95           # a hover ~5mm off
    assert wrist_check.face_fill(frame()) == 0.0
    assert wrist_check.face_fill(frame((u + 150, v), (u - 150, v), (u, v - 135))) < 0.05   # neighbours only


def test_the_look_says_why_it_could_not_look():
    class Unreachable:
        def lift(self):
            pass

        def hover(self, *a, **k):
            return None
    result = wrist_check.check(Unreachable(), None, 0.5, 0.0, {"offset_m": 0.045, "clearance_m": 0.04, "gripper": 27})
    assert result["present"] is None and result["why"] == "out of reach"
