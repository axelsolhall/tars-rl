"""Camera-IMU calibration on a synthetic rig: known IMU rotation, gyro bias and time offset."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tars_scan import camimu


def test_recovers_rotation_bias_and_time_offset():
    rng = np.random.default_rng(1)
    R_ci = Rotation.from_euler("xyz", [5, 135, -3], degrees=True).as_matrix()
    bias = np.array([-0.025, 0.036, 0.009])
    offset_ns = 6.5e6  # IMU clock = camera clock + offset

    def w_cam(t):  # smooth, varied rotation rate in the camera frame (rad/s)
        return np.stack([np.sin(1.3 * t), 0.7 * np.cos(0.9 * t + 1), 0.5 * np.sin(2.1 * t + 2)], -1)

    # Integrate camera orientations at 30 fps; jitter them like PnP (0.1 deg).
    t_cam = np.arange(0, 60, 1 / 30)
    fine = np.arange(0, 60, 1e-4)
    w_f = w_cam(fine)
    R, Rs, k = np.eye(3), [], 0
    for i, tf in enumerate(fine):
        if k < len(t_cam) and tf >= t_cam[k] - 1e-9:
            jit = Rotation.from_rotvec(rng.normal(0, np.radians(0.1), 3)).as_matrix()
            Rs.append(R @ jit)
            k += 1
        R = R @ Rotation.from_rotvec(w_f[i] * 1e-4).as_matrix()
    poses = {i: (Rs[i], np.zeros(3)) for i in range(len(Rs))}
    t_ns = {i: t_cam[i] * 1e9 for i in range(len(Rs))}

    t_imu = np.arange(-1, 61, 1 / 200) * 1e9 + 0.0
    w_imu = w_cam((t_imu - offset_ns) / 1e9) @ R_ci + bias  # w_i = R_ci^T w_c + b
    w_imu += rng.normal(0, 0.003, w_imu.shape)

    _, spans, wc = camimu.camera_rates(poses, t_ns)
    d, peak = camimu.time_offset_ns(spans, wc, t_imu, w_imu)
    d = camimu.refine_offset(spans, wc, t_imu, w_imu, d)
    R_est, b_est, _ = camimu.rotation_and_bias(wc, camimu.window_means(spans, t_imu, w_imu, d))
    assert d / 1e6 == pytest.approx(6.5, abs=0.3) and peak > 0.95
    angle = np.degrees(np.linalg.norm(Rotation.from_matrix(R_est.T @ R_ci).as_rotvec()))
    assert angle < 0.5
    assert b_est == pytest.approx(bias, abs=0.003)
