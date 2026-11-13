"""camera.torn: a frame whose rows jumped from the one before it (USB data lost), told from the arm's own motion."""
import numpy as np

from mahjong.camera import torn


def scene(shift=0):
    """A textured view of the mat, moved `shift` px down and right (the arm's motion between two frames at 30fps)."""
    y, x = np.mgrid[0:480, 0:640]
    grey = (127 + 60 * np.sin((x + shift) / 9.0) + 40 * np.cos((y + shift) / 13.0)).astype(np.uint8)
    return np.dstack([grey] * 3)


def test_a_band_of_garbage_rows_is_torn():
    tear = scene().copy()
    tear[200:380] = np.random.default_rng(0).integers(0, 255, (180, 640, 3), dtype=np.uint8)
    assert torn(scene(), tear)


def test_rows_shifted_sideways_are_torn():
    tear = scene().copy()
    tear[150:450] = np.roll(tear[150:450], 256, axis=1)     # a lost USB packet shifts the rest of the frame along
    assert torn(scene(), tear)


def test_motion_and_a_change_in_light_are_not():
    assert not torn(scene(), scene(shift=3))
    assert not torn(scene(), np.clip(scene().astype(int) + 25, 0, 255).astype(np.uint8))


def test_nothing_to_compare_with_is_not_torn():
    assert not torn(None, scene())
    assert not torn(scene()[:240], scene())
