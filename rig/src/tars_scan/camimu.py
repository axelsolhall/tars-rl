"""Camera-IMU calibration from a checkerboard film: how the IMU is turned relative to camera 0, its
time offset, the gyro bias, and (if the data supports it) where it sits.

  python -m tars_scan.camimu <session>    # needs <session>/calib/calib.json and detections_all.npz

1. Camera 0's pose in every frame where it saw the board (fisheye PnP at the calibrated lens), so
   its angular velocity between consecutive frames is known in the camera frame.
2. Time offset: the shift that best lines up |w_camera| with |w_gyro|. Norms do not depend on the
   unknown rotation, so this needs nothing else.
3. Rotation and gyro bias: least squares on w_cam = R (w_gyro - b), R projected onto a rotation.
4. Lever arm (IMU position in camera 0's frame): least squares on the accelerometer against the
   board trajectory's second derivative, with gravity and accelerometer bias as unknowns. Twice
   differentiated PnP positions are noisy, so it is reported with its residual and held to a
   plausibility bound; out of bounds, 0 is written and flagged.

Accuracy for SLAM is assumption A159. Writes <session>/calib/camimu.json. Its eval is its own
numbers: the gyro fit residual, the correlation peak, and the same estimates from the two halves
of the film.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tars_rig.check import imu_clock_segments, imu_sample_times, read_csv
from tars_rig.contracts import read_json

from . import export
from .calib import board_points

G = 9.80665
MAX_LEVER_M = 0.15  # the IMU is on the same small rig; further than this means the fit failed


def so3_log(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / 2 if th < 1e-9 else w * th / (2 * math.sin(th))


def project_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def board_poses(
    corners: dict[int, np.ndarray], obj, K, D
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{frame index: (R_bc, p_bc)}: camera orientation and position in the board frame.

    The detector sometimes numbers the corners from the opposite end, which flips the board pose
    by 180 deg and moves its origin to the far corner (12 jumps of up to 32 cm in the first real
    film). Both numberings are solved and the one closest in rotation to the previous accepted
    frame is kept; the chain starts from the numbering most frames agree with."""
    import cv2

    o = obj.reshape(1, -1, 3).astype(np.float64)
    cand: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for k, c in corners.items():
        options = []
        for cc in (c, c[::-1]):
            ok, r, t = cv2.fisheye.solvePnP(o, cc.reshape(1, -1, 2), K, D.reshape(4, 1))
            if ok:
                R_cb = cv2.Rodrigues(r)[0]
                options.append((R_cb.T, -R_cb.T @ np.ravel(t)))
        if options:
            cand[k] = options
    if not cand:
        return {}
    # Seed: the board x axis seen from the camera; the majority direction picks the numbering.
    xs = np.array([opts[0][0][:, 0] for opts in cand.values()])  # R_bc column 0 = board x in ...
    ref = np.median(xs, axis=0)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    prev = None
    for k in sorted(cand):
        opts = cand[k]
        if prev is None:
            best = max(opts, key=lambda o_: float(o_[0][:, 0] @ ref))
        else:
            best = min(opts, key=lambda o_: np.linalg.norm(so3_log(prev[0].T @ o_[0])))
        out[k] = best
        prev = best
    return out


def camera_rates(
    poses: dict, t_ns: dict, half: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean angular velocity in the camera frame over windows of 2*half+1 consecutive detected
    frames: (mid time, window start, window end in ns) and the rates.

    Frame-to-frame differences of PnP rotations were noise-dominated on the first real film (dark
    frames, small board): a 0.3 deg pose wobble is ~0.2 rad/s over 33 ms. A window of 4 frame
    intervals cuts that 4x; the gyro is averaged over the same window, so both see the same
    bandwidth."""
    spans, ws = [], []
    for k in sorted(poses):
        a, b = k - half, k + half
        if all(j in poses for j in range(a, b + 1)):
            dt = (t_ns[b] - t_ns[a]) / 1e9
            ws.append(so3_log(poses[a][0].T @ poses[b][0]) / dt)
            spans.append((t_ns[a], t_ns[b]))
    sp = np.array(spans)
    return sp.mean(axis=1), sp, np.array(ws)


def window_means(spans: np.ndarray, t: np.ndarray, v: np.ndarray, shift_ns: float = 0.0):
    """Mean of v over each [start, end] + shift (trapezoid integral of the samples)."""
    cum = np.concatenate(
        [np.zeros((1, v.shape[1])), np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t)[:, None], axis=0)]
    )
    a, b = spans[:, 0] + shift_ns, spans[:, 1] + shift_ns
    ia = np.stack([np.interp(a, t, cum[:, j]) for j in range(v.shape[1])], axis=1)
    ib = np.stack([np.interp(b, t, cum[:, j]) for j in range(v.shape[1])], axis=1)
    return (ib - ia) / (b - a)[:, None]


def interp_rows(t_query: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_query, t, v[:, j]) for j in range(v.shape[1])], axis=1)


def time_offset_ns(spans, w_cam, t_imu, w_imu, search_ms: float = 150, step_ms: float = 0.5):
    """Shift d (ns) maximising corr(|w_cam|, |mean w_imu over the window + d|): IMU time =
    camera time + d. Norms do not depend on the unknown rotation."""
    n_cam = np.linalg.norm(w_cam, axis=1)
    shifts = np.arange(-search_ms, search_ms + step_ms, step_ms) * 1e6
    corr = np.array(
        [
            np.corrcoef(n_cam, np.linalg.norm(window_means(spans, t_imu, w_imu, d), axis=1))[0, 1]
            for d in shifts
        ]
    )
    i = int(np.argmax(corr))
    frac = 0.0
    if 0 < i < len(corr) - 1:  # parabola through the peak
        a, b, c = corr[i - 1], corr[i], corr[i + 1]
        if a - 2 * b + c != 0:
            frac = 0.5 * (a - c) / (a - 2 * b + c)
    return float(shifts[i] + frac * step_ms * 1e6), float(corr[i])


def refine_offset(spans, w_cam, t_imu, w_imu, d0: float, half_ms: float = 6, step_ms: float = 0.1):
    """Refine the time offset on the full vector fit: the |w| correlation is rotation-free but a
    gyro bias skews it (by ~1 ms in the synthetic test)."""
    best = None
    for d in d0 + np.arange(-half_ms, half_ms + step_ms, step_ms) * 1e6:
        _, _, resid = rotation_and_bias(w_cam, window_means(spans, t_imu, w_imu, d), iters=2)
        cost = float(np.mean(np.minimum(resid, 0.1) ** 2))
        if best is None or cost < best[0]:
            best = (cost, d)
    return best[1]


def rotation_and_bias(w_cam, w_imu, iters: int = 5, huber: float = 0.05):
    """w_cam = R (w_imu - b). Linear LS for [M | c] with w_cam = M w_imu + c, reweighted (Huber,
    rad/s) against PnP glitches; R = nearest rotation to M; b = -R^T c."""
    X = np.c_[w_imu, np.ones(len(w_imu))]
    wts = np.ones(len(w_imu))
    for _ in range(iters):
        A = (X * wts[:, None]).T @ X
        B = (X * wts[:, None]).T @ w_cam
        sol = np.linalg.solve(A, B)  # 4x3
        R = project_so3(sol[:3].T)
        c = sol[3]
        r = np.linalg.norm(w_cam - (w_imu @ R.T + c), axis=1)
        wts = np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-12))
    b = -R.T @ c
    resid = np.linalg.norm(w_cam - (w_imu - b) @ R.T, axis=1)
    return R, b, resid


def lever_arm(poses, t_ns, t_imu, acc, w_imu, R_ci, b_g, d_ns, half_window: int = 3):
    """IMU position in camera 0's frame from the accelerometer (see the module docstring).

    Uses runs of consecutive detections; positions are fitted with a local quadratic over
    2*half_window+1 frames to get acceleration."""
    rows_A, rows_y = [], []
    keys = sorted(poses)
    for k in keys:
        win = list(range(k - half_window, k + half_window + 1))
        if not all(j in poses for j in win):
            continue
        tt = np.array([(t_ns[j] - t_ns[k]) / 1e9 for j in win])
        P = np.array([poses[j][1] for j in win])
        V = np.vander(tt, 3)  # [t^2, t, 1]
        coef = np.linalg.lstsq(V, P, rcond=None)[0]
        a_b = 2 * coef[0]  # camera acceleration, board frame
        R_bc = poses[k][0]
        # Angular velocity / acceleration of the camera, camera frame, from the gyro (cleaner
        # than differentiating PnP rotations twice).
        tq = t_ns[k] + d_ns
        w_i = interp_rows(np.array([tq]), t_imu, w_imu)[0] - b_g
        w_i2 = interp_rows(np.array([tq + 5e6]), t_imu, w_imu)[0] - b_g
        w_i0 = interp_rows(np.array([tq - 5e6]), t_imu, w_imu)[0] - b_g
        w_c, al_c = R_ci @ w_i, R_ci @ (w_i2 - w_i0) / 0.01
        f_i = interp_rows(np.array([tq]), t_imu, acc)[0]
        # R_bc R_ci f = a_b + R_bc ([al]x + [w]x^2) p_ci - g_b + R_bc R_ci b_a
        Wx, Ax = _skew(w_c), _skew(al_c)
        lhs_p = R_bc @ (Ax + Wx @ Wx)
        lhs_ba = R_bc @ R_ci
        A = np.c_[-lhs_p, np.eye(3), -lhs_ba]  # unknowns: p_ci, g_b, b_a
        rows_A.append(A)
        rows_y.append(a_b - R_bc @ R_ci @ f_i)
    if len(rows_A) < 30:
        return None
    A, y = np.concatenate(rows_A), np.concatenate(rows_y)
    # y = -A_p p + g - A_ba b_a  ->  solve A x = -y with x = [p, g, b_a] (signs folded in above)
    x, *_ = np.linalg.lstsq(A, -y, rcond=None)
    resid = A @ x + y
    return {
        "p_ci_m": x[:3],
        "gravity_board": x[3:6],
        "accel_bias": x[6:9],
        "resid_rms_m_s2": float(np.sqrt(np.mean(resid**2))),
        "rows": len(rows_A),
    }


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def calibrate_camimu(session: Path, detections: Path | None = None) -> dict:
    meta = read_json(session / "meta.json")
    calib = json.loads((session / "calib" / "calib.json").read_text())
    det = np.load(detections or session / "calib" / "detections_all.npz")
    corners = {int(k): c for k, c in zip(det["keys0"], det["corners0"], strict=True)}
    cam = calib["cameras"][0]
    K, D = np.array(cam["K"]), np.array(cam["D"])
    obj = board_points(tuple(calib["board"]), calib["square_m"])

    frames = read_csv(session / "cam0" / "frames.csv")
    t_all = export.frame_times_ns(frames, meta["cameras"][0].get("readout_us"))
    t_ns = {int(i): float(t) for i, t in zip(frames["index"], t_all, strict=True)}
    poses = board_poses(corners, obj, K, D)
    t_cam, spans, w_cam = camera_rates(poses, t_ns)

    imu, reads = read_csv(session / "imu" / "imu.csv"), read_csv(session / "imu" / "reads.csv")
    t_imu = imu_sample_times(imu["k"], imu_clock_segments(reads))
    ok = np.isfinite(t_imu)
    g_scale = math.pi / 180 / meta["imu"]["gyro_lsb_per_dps"]
    a_scale = G / meta["imu"]["accel_lsb_per_g"]
    t_imu = t_imu[ok]
    w_imu = np.c_[imu["gx"], imu["gy"], imu["gz"]][ok] * g_scale
    acc = np.c_[imu["ax"], imu["ay"], imu["az"]][ok] * a_scale

    def solve(sel):
        d, peak = time_offset_ns(spans[sel], w_cam[sel], t_imu, w_imu)
        d = refine_offset(spans[sel], w_cam[sel], t_imu, w_imu, d)
        w_i = window_means(spans[sel], t_imu, w_imu, d)
        R, b, resid = rotation_and_bias(w_cam[sel], w_i)
        return d, peak, R, b, resid

    n = len(t_cam)
    d, peak, R, b, resid = solve(np.arange(n))
    halves = [solve(np.arange(n)[: n // 2]), solve(np.arange(n)[n // 2 :])]
    lever = lever_arm(poses, t_ns, t_imu, acc, w_imu, R, b, d)
    lever_ok = (
        lever is not None
        and np.linalg.norm(lever["p_ci_m"]) <= MAX_LEVER_M
        and abs(np.linalg.norm(lever["gravity_board"]) - G) < 0.5
    )
    p_ci = lever["p_ci_m"] if lever_ok else np.zeros(3)

    def angle_deg(Ra, Rb):
        return math.degrees(np.linalg.norm(so3_log(Ra.T @ Rb)))

    report = {
        "session": session.name,
        "rate_pairs": n,
        "board_frames": len(poses),
        "time_offset_ms": round(d / 1e6, 3),  # IMU clock = camera clock + offset
        "correlation_peak": round(peak, 4),
        "R_cam0_imu": np.round(R, 6).tolist(),  # w_cam0 = R (w_imu - bias)
        "gyro_bias_rad_s": np.round(b, 6).tolist(),
        "gyro_fit_resid_rad_s_median": round(float(np.median(resid)), 4),
        "gyro_fit_resid_rad_s_p90": round(float(np.percentile(resid, 90)), 4),
        "halves": {
            "time_offset_ms": [round(h[0] / 1e6, 3) for h in halves],
            "rotation_diff_deg": round(angle_deg(halves[0][2], halves[1][2]), 3),
            "rotation_vs_full_deg": [round(angle_deg(h[2], R), 3) for h in halves],
        },
        "lever_arm": None
        if lever is None
        else {
            "p_imu_in_cam0_m": np.round(lever["p_ci_m"], 4).tolist(),
            "gravity_norm": round(float(np.linalg.norm(lever["gravity_board"])), 3),
            "accel_bias_m_s2": np.round(lever["accel_bias"], 3).tolist(),
            "resid_rms_m_s2": round(lever["resid_rms_m_s2"], 3),
            "rows": lever["rows"],
            "used": bool(lever_ok),
        },
        "p_imu_in_cam0_m_used": np.round(p_ci, 4).tolist(),
    }
    (session / "calib" / "camimu.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tars_scan.camimu", description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path)
    p.add_argument("--detections", type=Path, default=None)
    a = p.parse_args(argv)
    print(json.dumps(calibrate_camimu(a.session, a.detections), indent=2))


if __name__ == "__main__":
    main()
