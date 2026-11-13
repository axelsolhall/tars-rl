"""Rolling-shutter correction: warp each frame so every row looks as if it was captured at the
frame's timestamp (mid-exposure of the middle row, A154).

The IMX219 reads its rows top to bottom over readout_us (~12 ms at 1640x1232). A turning rig
therefore sees each row from a slightly different orientation. With the gyro (rotated into the
camera frame, bias removed, tars_scan.camimu) and the sensor's line time, each row's orientation
offset is exp([w] dt_row); the warp undoes it through the fisheye model. Translation during the
readout is ignored (mm at walking speed). None of the SLAM systems compared here model rolling
shutter, which is why this exists; its value is measured with tars_scan.slam_eval.

Row order: image row 0 is the first row read (the frames are stored without libcamera's 180 deg
transform).
"""

from __future__ import annotations

import numpy as np


class FisheyeRemap:
    """Bearings of every output pixel for one camera, computed once."""

    def __init__(self, K: np.ndarray, D: np.ndarray, size: tuple[int, int]):
        import cv2

        w, h = size
        self.K, self.D, self.size = K, D.reshape(4, 1), size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        pts = np.stack([u, v], -1).reshape(1, -1, 2)
        und = cv2.fisheye.undistortPoints(pts, K, self.D).reshape(h, w, 2)
        b = np.concatenate([und, np.ones((h, w, 1))], -1)
        self.bearings = (b / np.linalg.norm(b, axis=-1, keepdims=True)).astype(np.float32)
        self.row_offset = np.arange(h, dtype=np.float32) - (h - 1) / 2  # rows from the middle

    def maps(self, omega_cam: np.ndarray, line_time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """cv2.remap maps: for each corrected pixel, where it was in the captured frame. Small
        angle: the capture of row r is rotated by theta_r = w * (r - mid) * line_time from the
        reference, so a reference bearing b was seen at b - theta_r x b."""
        import cv2

        h, w = self.size[1], self.size[0]
        theta = (self.row_offset * line_time_s)[:, None] * omega_cam[None, :].astype(np.float32)
        src = self.bearings - np.cross(theta[:, None, :], self.bearings)
        # Project with the fisheye model (z > 0 here: the warp is a fraction of a degree).
        p, _ = cv2.fisheye.projectPoints(
            src.reshape(1, -1, 3).astype(np.float64), np.zeros(3), np.zeros(3), self.K, self.D
        )
        p = p.reshape(h, w, 2).astype(np.float32)
        return p[..., 0], p[..., 1]

    def correct(self, img: np.ndarray, omega_cam: np.ndarray, line_time_s: float) -> np.ndarray:
        import cv2

        mx, my = self.maps(omega_cam, line_time_s)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class LinearRemap:
    """FisheyeRemap.correct at a fixed line time, linearised in the angular rate: the remap is
    map0 + J @ omega, with J from central differences once. A rolling-shutter warp is a fraction of
    a degree, so the second-order error is tiny (test_rs_correct), and a frame costs a few
    multiply-adds per pixel instead of a fisheye projection of every pixel (98 ms per pair on the
    Mac with the exact remap: too slow for 30 fps live)."""

    def __init__(self, exact: FisheyeRemap, line_time_s: float, eps: float = 0.5):
        self.line_time_s = line_time_s
        self.m0 = [m.astype(np.float32) for m in exact.maps(np.zeros(3), line_time_s)]
        self.J = []  # per axis: (d map_x / d omega, d map_y / d omega)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            plus, minus = exact.maps(e, line_time_s), exact.maps(-e, line_time_s)
            self.J.append(
                [((a - b) / (2 * eps)).astype(np.float32) for a, b in zip(plus, minus, strict=True)]
            )

    def maps(self, omega_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.asarray(omega_cam, np.float32)
        mx = self.m0[0] + w[0] * self.J[0][0] + w[1] * self.J[1][0] + w[2] * self.J[2][0]
        my = self.m0[1] + w[0] * self.J[0][1] + w[1] * self.J[1][1] + w[2] * self.J[2][1]
        return mx, my

    def correct(self, img: np.ndarray, omega_cam: np.ndarray) -> np.ndarray:
        import cv2

        mx, my = self.maps(omega_cam)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def shift_px(K: np.ndarray, omega_cam: np.ndarray, readout_s: float) -> float:
    """Rough top-to-bottom skew in pixels at the image centre: what the correction removes."""
    return float(np.linalg.norm(omega_cam[:2]) * readout_s * K[0, 0])


def gyro_cam0(session, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """(IMU-clock t_ns, angular rate in camera 0's frame, bias removed) for the whole session."""
    import math

    from tars_rig.check import imu_clock_segments, imu_sample_times, read_csv
    from tars_rig.contracts import read_json

    meta = read_json(session / "meta.json")
    imu, reads = read_csv(session / "imu" / "imu.csv"), read_csv(session / "imu" / "reads.csv")
    t = imu_sample_times(imu["k"], imu_clock_segments(reads))
    ok = np.isfinite(t)
    w = np.c_[imu["gx"], imu["gy"], imu["gz"]][ok] * (
        math.pi / 180 / meta["imu"]["gyro_lsb_per_dps"]
    )
    R_c0_i = np.linalg.inv(model["cams"][0]["T_i_c"])[:3, :3]
    return t[ok], (w - model["gyro_bias"]) @ R_c0_i.T


def mean_rate(t: np.ndarray, w: np.ndarray, t0: float, t1: float) -> np.ndarray:
    m = (t >= t0) & (t <= t1)
    if m.sum() == 0:
        return np.array([np.interp((t0 + t1) / 2, t, w[:, j]) for j in range(3)])
    return w[m].mean(axis=0)
