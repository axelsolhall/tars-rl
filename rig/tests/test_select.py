"""Calibration frame selection: the pieces it rests on (OpenCV's Jacobian layout, pauses, the smear
model, Max-logDet preferring new information over repeats)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from tars_scan import select  # noqa: E402

K = np.array([[780.0, 0, 820], [0, 782, 610], [0, 0, 1]])
D = np.array([0.03, -0.01, 0.004, -0.001])


def board():
    g = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
    return np.c_[g, np.zeros(len(g))]


def test_jacobian_layout_is_intrinsics_then_pose():
    """select.view_information slices projectPoints' Jacobian as f, c, k1-k4, rvec, tvec."""
    obj = board().reshape(1, -1, 3)
    r, t = np.array([0.1, -0.2, 0.05]), np.array([-0.1, -0.05, 0.5])

    def proj(p):
        K2 = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
        return cv2.fisheye.projectPoints(obj, p[8:11], p[11:14], K2, p[4:8].reshape(4, 1))[0]

    p0 = np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], D, r, t]
    _, J = cv2.fisheye.projectPoints(obj, r, t, K, D.reshape(4, 1))
    num = np.stack([(proj(p0 + e) - proj(p0 - e)).ravel() / 2e-6 for e in np.eye(14) * 1e-6], 1)
    assert np.abs(J[:, :14] - num).max() < 1e-2


def test_motion_px_and_pauses():
    t = np.arange(10) * 1e8
    gyro_t = np.arange(0, 1e9, 5e6)
    gyro = np.zeros((len(gyro_t), 3))
    gyro[(gyro_t > 3e8) & (gyro_t < 6e8), 2] = 1.0  # 1 rad/s between 0.3 and 0.6 s
    px = select.motion_px(t, np.full(10, 6000.0), 23_000.0, gyro_t, gyro, 780.0)
    assert px[0] == 0 and px[4] == pytest.approx(1.0 * 29e-3 * 780)
    runs = select.pauses(px, still_px=0.5, min_frames=2)
    assert [list(r) for r in runs] == [[0, 1, 2, 3], [6, 7, 8, 9]]
    assert list(select.pause_representatives(px, 0.5)) in ([0, 6], [6, 0])


def test_greedy_logdet_skips_repeats():
    """Three copies of a view that constrains parameter 0, and two views that constrain 1 and 2: a
    budget of 3 takes one of each, not the copies."""

    def info(axis):
        m = np.zeros((3, 3))
        m[axis, axis] = 1.0
        return m

    infos = [info(0), info(0), info(0), info(1), info(2)]
    pick = select.greedy_logdet(infos, 3)
    assert sorted(pick)[1:] == [3, 4] and sum(i < 3 for i in pick) == 1


def test_candidates_keep_moving_frames_in_value_order():
    value = np.array([3.0, 0.1, 1.5, 0.4, np.nan, 1.9])
    assert list(select.candidates(value, 2.0, 1)) == [1, 3, 2, 5]
