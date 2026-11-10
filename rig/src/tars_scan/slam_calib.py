"""Our calibration (calib.json + camimu.json) in the formats SLAM systems read.

One rig model, several writers: each SLAM gets the same numbers, so a difference in results is the
algorithm, not the calibration.

Frames: cam0, cam1 (OpenCV camera frames: x right, y down, z forward), imu. Stored as
  T_c1_c0 (calib.json stereo: X1 = R X0 + T), R_c0_i and p_i_in_c0 (camimu.json: w_c0 = R w_i).

IMU noise is continuous-time density (per sqrt(Hz)), as Basalt, Kalibr and OpenVINS expect. Until
the still IMU session gives Allan-variance numbers, it is the desk measurement inflated by
NOISE_INFLATE (A161), a common margin for unmodelled vibration and temperature.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Desk measurement 2026-11-09 (rig still, 200 Hz, DLPF 3): gyro y/z 0.12 deg/s, accel 2-3 mg rms.
# Effective noise bandwidth of the ~42 Hz low-pass ~ 1.57 * 42 Hz. Gyro x is excluded: it carries
# a chip-internal 46 Hz tone that integrates to ~0.005 deg.
NOISE_BW_HZ = 1.57 * 42
GYRO_RMS_RAD_S = math.radians(0.13)
ACCEL_RMS_M_S2 = 0.003 * 9.80665
NOISE_INFLATE = 5.0  # A161
GYRO_BIAS_RW = 2e-4  # rad/s/sqrt(s), cheap-MEMS order of magnitude (A161)
ACCEL_BIAS_RW = 3e-3  # m/s^2/sqrt(s) (A161)


def rig_model(session: Path) -> dict:
    """Everything a SLAM config needs, as numpy, from <session>/calib/."""
    cal = json.loads((session / "calib" / "calib.json").read_text())
    ci = json.loads((session / "calib" / "camimu.json").read_text())
    R10, t10 = np.array(cal["stereo"]["R"]), np.array(cal["stereo"]["T_m"])
    T_c1_c0 = _T(R10, t10)
    R_c0_i = np.array(ci["R_cam0_imu"])
    T_c0_i = _T(R_c0_i, np.array(ci["p_imu_in_cam0_m_used"]))
    T_i_c0 = np.linalg.inv(T_c0_i)
    T_i_c1 = T_i_c0 @ np.linalg.inv(T_c1_c0)
    gyro_d = GYRO_RMS_RAD_S / math.sqrt(NOISE_BW_HZ) * NOISE_INFLATE
    acc_d = ACCEL_RMS_M_S2 / math.sqrt(NOISE_BW_HZ) * NOISE_INFLATE
    return {
        "cams": [
            {"K": np.array(c["K"]), "D": np.array(c["D"]), "T_i_c": T}
            for c, T in zip(cal["cameras"], (T_i_c0, T_i_c1), strict=True)
        ],
        "time_offset_ns": int(round(ci["time_offset_ms"] * 1e6)),
        "gyro_bias": np.array(ci["gyro_bias_rad_s"]),
        "gyro_noise_density": gyro_d,
        "accel_noise_density": acc_d,
        "gyro_bias_rw": GYRO_BIAS_RW,
        "accel_bias_rw": ACCEL_BIAS_RW,
        "imu_rate_hz": 200.0,
    }


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _quat_xyzw(R: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat().tolist()


def basalt_calib(model: dict, resolution: tuple[int, int], scale: float = 1.0) -> dict:
    """Basalt's calibration JSON (kb4 = OpenCV fisheye / Kannala-Brandt). Image timestamps are
    already moved to the IMU clock by the export, so cam_time_offset_ns is 0."""
    t_ic, intr = [], []
    for c in model["cams"]:
        T = c["T_i_c"]
        q = _quat_xyzw(T[:3, :3])
        t_ic.append(
            {
                "px": T[0, 3],
                "py": T[1, 3],
                "pz": T[2, 3],
                "qx": q[0],
                "qy": q[1],
                "qz": q[2],
                "qw": q[3],
            }
        )
        K, D = c["K"], c["D"]
        intr.append(
            {
                "camera_type": "kb4",
                "intrinsics": {
                    # Pixel scaling: (x + 0.5) * s - 0.5 for the centre, s for focal.
                    "fx": K[0, 0] * scale,
                    "fy": K[1, 1] * scale,
                    "cx": (K[0, 2] + 0.5) * scale - 0.5,
                    "cy": (K[1, 2] + 0.5) * scale - 0.5,
                    "k1": D[0],
                    "k2": D[1],
                    "k3": D[2],
                    "k4": D[3],
                },
            }
        )
    w, h = round(resolution[0] * scale), round(resolution[1] * scale)
    g = model["gyro_bias"].tolist()
    return {
        "value0": {
            "T_imu_cam": t_ic,
            "intrinsics": intr,
            "resolution": [[w, h], [w, h]],
            "calib_accel_bias": [0.0] * 9,
            "calib_gyro_bias": [*g, *([0.0] * 9)],
            "imu_update_rate": model["imu_rate_hz"],
            "accel_noise_std": [model["accel_noise_density"]] * 3,
            "gyro_noise_std": [model["gyro_noise_density"]] * 3,
            "accel_bias_std": [model["accel_bias_rw"]] * 3,
            "gyro_bias_std": [model["gyro_bias_rw"]] * 3,
            "cam_time_offset_ns": 0,
            "vignette": [],
        }
    }


def write_basalt(session: Path, out: Path, resolution=(1640, 1232), scale: float = 1.0) -> Path:
    path = out / "basalt_calib.json"
    path.write_text(json.dumps(basalt_calib(rig_model(session), resolution, scale), indent=2))
    return path


def _cv_matrix(T: np.ndarray) -> str:
    vals = ", ".join(f"{v:.9f}" for v in T.ravel())
    return f"!!opencv-matrix\n  rows: 4\n  cols: 4\n  dt: f\n  data: [{vals}]"


def orbslam3_yaml(
    model: dict, resolution: tuple[int, int], scale: float = 1.0, fps: float = 30
) -> str:
    """ORB-SLAM3 stereo-inertial settings (File.version 1.0, KannalaBrandt8 = OpenCV fisheye)."""
    w, h = round(resolution[0] * scale), round(resolution[1] * scale)
    (c0, c1) = model["cams"]
    T_c0_c1 = np.linalg.inv(c0["T_i_c"]) @ c1["T_i_c"]
    lines = ["%YAML:1.0", 'File.version: "1.0"', 'Camera.type: "KannalaBrandt8"']
    for n, c in ((1, c0), (2, c1)):
        K, D = c["K"], c["D"]
        lines += [
            f"Camera{n}.fx: {K[0, 0] * scale}",
            f"Camera{n}.fy: {K[1, 1] * scale}",
            f"Camera{n}.cx: {(K[0, 2] + 0.5) * scale - 0.5}",
            f"Camera{n}.cy: {(K[1, 2] + 0.5) * scale - 0.5}",
            *(f"Camera{n}.k{i + 1}: {D[i]}" for i in range(4)),
            f"Camera{n}.overlappingBegin: 0",
            f"Camera{n}.overlappingEnd: {w - 1}",
        ]
    lines += [
        f"Stereo.T_c1_c2: {_cv_matrix(T_c0_c1)}",
        f"Camera.width: {w}",
        f"Camera.height: {h}",
        f"Camera.fps: {fps}",
        "Camera.RGB: 0",
        "Stereo.ThDepth: 40.0",
        f"IMU.T_b_c1: {_cv_matrix(c0['T_i_c'])}",
        f"IMU.NoiseGyro: {model['gyro_noise_density']}",
        f"IMU.NoiseAcc: {model['accel_noise_density']}",
        f"IMU.GyroWalk: {model['gyro_bias_rw']}",
        f"IMU.AccWalk: {model['accel_bias_rw']}",
        f"IMU.Frequency: {model['imu_rate_hz']}",
        "ORBextractor.nFeatures: 1250",
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8",
        "ORBextractor.iniThFAST: 20",
        "ORBextractor.minThFAST: 7",
        "Viewer.KeyFrameSize: 0.05",
        "Viewer.KeyFrameLineWidth: 1.0",
        "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08",
        "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -3.5",
        "Viewer.ViewpointF: 500.0",
    ]
    return "\n".join(lines) + "\n"


# Sensor ranges of the MPU6500 as configured (meta.json: 8192 LSB/g, 65.5 LSB/(deg/s)).
ACCEL_RANGE_M_S2 = 4 * 9.80665
GYRO_RANGE_RAD_S = math.radians(500)


def _rows(T: np.ndarray, indent: str) -> str:
    return "\n".join(f"{indent}- [{', '.join(f'{v:.9f}' for v in r)}]" for r in T)


def openvins_files(model: dict, resolution: tuple[int, int], scale: float = 1.0) -> dict:
    """OpenVINS's two Kalibr-style chain files (equidistant = OpenCV fisheye). T_imu_cam is
    camera to IMU (R_CtoI, p_CinI), i.e. our T_i_c. The estimator config is the caller's."""
    w, h = round(resolution[0] * scale), round(resolution[1] * scale)
    cams = []
    for n, c in enumerate(model["cams"]):
        K, D = c["K"], c["D"]
        intr = [K[0, 0] * scale, K[1, 1] * scale, (K[0, 2] + 0.5) * scale - 0.5]
        intr.append((K[1, 2] + 0.5) * scale - 0.5)
        cams.append(
            f"cam{n}:\n  T_imu_cam:\n{_rows(c['T_i_c'], '    ')}\n  cam_overlaps: [{1 - n}]\n"
            f"  camera_model: pinhole\n  distortion_model: equidistant\n"
            f"  distortion_coeffs: [{', '.join(f'{v:.9f}' for v in D)}]\n"
            f"  intrinsics: [{', '.join(f'{v:.6f}' for v in intr)}]\n"
            f"  resolution: [{w}, {h}]\n  rostopic: /cam{n}/image_raw\n  timeshift_cam_imu: 0.0\n"
        )
    eye = "\n".join(
        f"    - [{', '.join('1.0' if i == j else '0.0' for j in range(3))}]" for i in range(3)
    )
    zero = "\n".join("    - [0.0, 0.0, 0.0]" for _ in range(3))
    imu = (
        "imu0:\n  T_i_b:\n" + _rows(np.eye(4), "    ") + "\n"
        f"  accelerometer_noise_density: {model['accel_noise_density']:.6e}\n"
        f"  accelerometer_random_walk: {model['accel_bias_rw']:.6e}\n"
        f"  gyroscope_noise_density: {model['gyro_noise_density']:.6e}\n"
        f"  gyroscope_random_walk: {model['gyro_bias_rw']:.6e}\n"
        f"  rostopic: /imu0\n  time_offset: 0.0\n  update_rate: {model['imu_rate_hz']}\n"
        f'  model: "kalibr"\n  Tw:\n{eye}\n  R_IMUtoGYRO:\n{eye}\n  Ta:\n{eye}\n'
        f"  R_IMUtoACC:\n{eye}\n  Tg:\n{zero}\n"
    )
    return {
        "kalibr_imucam_chain.yaml": "%YAML:1.0\n\n" + "".join(cams),
        "kalibr_imu_chain.yaml": "%YAML:1.0\n\n" + imu,
    }


def okvis2_patch(model: dict, resolution: tuple[int, int], scale: float = 1.0) -> dict:
    """The rig-specific parts of an OKVIS2 config (merged into a base yaml by the runner):
    T_SC is camera to IMU (our T_i_c), equidistant = OpenCV fisheye, noise as densities."""
    w, h = round(resolution[0] * scale), round(resolution[1] * scale)
    cams = []
    for c in model["cams"]:
        K, D = c["K"], c["D"]
        cams.append(
            {
                "T_SC": [float(v) for v in c["T_i_c"].ravel()],
                "image_dimension": [w, h],
                "distortion_coefficients": [float(v) for v in D],
                "distortion_type": "equidistant",
                "focal_length": [float(K[0, 0] * scale), float(K[1, 1] * scale)],
                "principal_point": [
                    float((K[0, 2] + 0.5) * scale - 0.5),
                    float((K[1, 2] + 0.5) * scale - 0.5),
                ],
                "cam_model": "pinhole",
                "camera_type": "gray",
                "slam_use": "okvis",
            }
        )
    return {
        "cameras": cams,
        "imu_parameters": {
            "a_max": ACCEL_RANGE_M_S2,
            "g_max": GYRO_RANGE_RAD_S,
            "sigma_g_c": model["gyro_noise_density"],
            "sigma_a_c": model["accel_noise_density"],
            "sigma_gw_c": model["gyro_bias_rw"],
            "sigma_aw_c": model["accel_bias_rw"],
            "g0": [float(v) for v in model["gyro_bias"]],
            "a0": [0.0, 0.0, 0.0],
        },
    }
