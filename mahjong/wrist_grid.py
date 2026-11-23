"""The checkerboard a wrist-camera calibration check uses: finding it in a frame, and where it lies on the mat.

The board is 10x7 squares of about 8mm, so 9x6 inner corners. The RealSense sees it from above the mat, which gives the
corners' base-frame positions (the ground truth a wrist view is compared with); the wrist camera sees it from the arm's poses.
"""
import cv2
import numpy as np

from . import config, layout

GRID_CORNERS = (9, 6)           # inner corners (columns, rows)
SQUARE_M = 0.008                # printed square side, measured with a ruler
_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def find_grid(frame_bgr):
    """The 54 inner corners' pixels (54 x 2, refined) if the whole board is in the frame, else None."""
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(grey, GRID_CORNERS, _FLAGS)
    if not found:
        found, corners = cv2.findChessboardCornersSB(grey, GRID_CORNERS)       # sturdier on a small, blurred board
        if not found:
            return None
    else:
        corners = cv2.cornerSubPix(grey, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return np.asarray(corners, float).reshape(-1, 2)


def grid_found(frame_bgr):
    return find_grid(frame_bgr) is not None


def locate_on_mat(color_bgr, intrinsics, T_cam_to_base):
    """(pixels 54 x 2, base-frame points 54 x 3) of the board's corners seen by the RealSense, or None: the board lies on the
    mat, so each corner's ray meets the table plane."""
    pixels = find_grid(color_bgr)
    if pixels is None:
        return None
    points = layout.pixel_to_table(pixels, intrinsics, T_cam_to_base, z_of_xy=config.table_z)
    return pixels, points
