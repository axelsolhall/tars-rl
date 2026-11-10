"""Calibrate a rig session's stereo pair from a checkerboard film, most valuable frames first.

  python -m tars_scan.calib <session> [--board 10x7 --square-mm 25.03] [--budget 160] [--max-px 2]

Steps (tars_scan.select has the reasoning and select_eval the measured trade-offs):
1. Value each left frame by its predicted smear from the gyro (no decoding). Without IMU data,
   by sharpness at 1/8 size.
2. Search the board, in value order, in the left frame and its paired right frame.
3. Per camera, greedy Max-logDet picks `budget` views; fit the fisheye model (Kannala-Brandt);
   score on the held-out views.
4. Stereo extrinsics from frames where both cameras saw the board, intrinsics fixed.

Writes <session>/calib/calib.json (everything above, with numbers) and camchain.yaml (Kalibr's
format, pinhole-equidistant, which Basalt and OpenVINS tools convert from).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from tars_rig.check import imu_clock_segments, imu_sample_times, read_csv
from tars_rig.contracts import iter_jpegs, read_json

from . import export, select

HALF_FOV_DEG = 60.0  # the lens spec: 120 degrees across the width, for the initial focal guess


def gyro_rad_s(session: Path, meta: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if not (session / "imu" / "imu.csv").exists():
        return None
    imu, reads = read_csv(session / "imu" / "imu.csv"), read_csv(session / "imu" / "reads.csv")
    t = imu_sample_times(imu["k"], imu_clock_segments(reads))
    g = np.c_[imu["gx"], imu["gy"], imu["gz"]] * (math.pi / 180 / meta["imu"]["gyro_lsb_per_dps"])
    ok = np.isfinite(t)
    t, g = t[ok], g[ok]
    norm = np.linalg.norm(g, axis=1)
    bias = np.median(g[norm <= np.percentile(norm, 20)], axis=0)  # from the stillest fifth
    return t, g - bias


def frame_values(session: Path, meta: dict, frames: dict, focal_px: float) -> np.ndarray:
    cam = meta["cameras"][0]
    t = export.frame_times_ns(frames, cam.get("readout_us"))
    gyro = gyro_rad_s(session, meta)
    if gyro is not None:
        return select.motion_px(
            t, frames["exposure_us"], cam.get("readout_us") or 0.0, gyro[0], gyro[1], focal_px
        )
    import simplejpeg

    rows = zip(
        frames["index"].astype(int),
        frames["offset"].astype(int),
        frames["size"].astype(int),
        strict=True,
    )
    sharp = np.full(len(frames["index"]), np.nan)
    pos = {int(i): n for n, i in enumerate(frames["index"])}
    for index, jpg in iter_jpegs(session / "cam0", rows):
        h, w = simplejpeg.decode_jpeg_header(jpg)[:2]
        small = simplejpeg.decode_jpeg(jpg, colorspace="GRAY", min_height=h // 8, min_width=w // 8)
        sharp[pos[index]] = select.sharpness(small[..., 0])
    # No gyro: rank only. Map sharpness to a pseudo-smear so the most valuable is still lowest.
    return np.nanmax(sharp) / np.maximum(sharp, 1e-9) - 1


def detect(jpeg: bytes, board: tuple[int, int]) -> np.ndarray | None:
    import cv2

    from .export import _decode

    img = _decode(jpeg, 1.0)
    # Stretch to the full range: the first real session was ~4x underexposed (max ~50 of 255) and
    # the detector missed every board. A linear stretch keeps the corner positions unchanged.
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    ok, corners = cv2.findChessboardCornersSB(img, board, flags=cv2.CALIB_CB_ACCURACY)
    return corners.reshape(-1, 2).astype(np.float64) if ok else None


def board_points(board: tuple[int, int], square_m: float) -> np.ndarray:
    g = np.mgrid[0 : board[0], 0 : board[1]].T.reshape(-1, 2) * square_m
    return np.c_[g, np.zeros(len(g))]


def calibrate_session(
    session: Path, board: tuple[int, int], square_m: float, budget: int, max_px: float, stride: int
) -> dict:

    meta = read_json(session / "meta.json")
    cams = meta["cameras"]
    w, h = cams[0]["width"], cams[0]["height"]
    focal = (w / 2) / math.radians(HALF_FOV_DEG)  # equidistant: r = f * theta
    K0 = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    frames = [read_csv(session / f"cam{i}" / "frames.csv") for i in (0, 1)]
    stored = [np.flatnonzero(f["offset"] >= 0) for f in frames]
    times = [
        export.frame_times_ns(f, c.get("readout_us")) for f, c in zip(frames, cams, strict=True)
    ]
    pairs = export.pair_frames(
        times[0][stored[0]], times[1][stored[1]], 0.5e9 / cams[0]["fps"]
    )  # positions into stored[0], stored[1]
    left = stored[0][[a for a, _ in pairs]]
    right = stored[1][[b for _, b in pairs]]
    value = frame_values(session, meta, {k: v[left] for k, v in frames[0].items()}, focal)
    order = select.candidates(value, max_px, stride)

    t0 = time.perf_counter()
    obj = board_points(board, square_m)
    out = session / "calib"
    out.mkdir(exist_ok=True)
    cache = out / "detections.npz"
    found = load_detections(cache, board, max_px, stride)
    searching = found is None
    found = found or {0: {}, 1: {}}
    for cam, rows in ((0, left), (1, right)) if searching else ():
        f = frames[cam]
        wanted = [
            (int(f["index"][rows[n]]), int(f["offset"][rows[n]]), int(f["size"][rows[n]]))
            for n in order
        ]
        index_to_pair = {wanted[k][0]: int(order[k]) for k in range(len(order))}
        for index, jpg in iter_jpegs(session / f"cam{cam}", wanted):
            corners = detect(jpg, board)
            if corners is not None:
                found[cam][index_to_pair[index]] = corners
    if searching:
        save_detections(cache, found, board, max_px, stride)
    search_s = time.perf_counter() - t0

    report: dict = {
        "session": session.name,
        "board": list(board),
        "square_m": square_m,
        "frames_paired": len(pairs),
        "searched": len(order),
        "search_seconds": round(search_s, 1),
        "cameras": [],
    }
    results = []
    for cam in (0, 1):
        # Views in value order (stillest first), as select_and_calibrate expects.
        keys = [int(n) for n in order if int(n) in found[cam]]
        views = [select.View(k, obj.copy(), found[cam][k]) for k in keys]
        weights = 1 / select.noise_px(value[keys]) ** 2
        r = select.select_and_calibrate(views, (w, h), budget, K0, np.zeros(4), weights=weights)
        results.append(r)
        held = r["heldout_rms_px"]
        report["cameras"].append(
            {
                "detected": len(views),
                "used": len(r["frames"]),
                "rms_px": round(r["rms_px"], 4),
                "heldout_rms_px": None if held is None else round(held, 4),
                "K": np.round(r["K"], 4).tolist(),
                "D": np.round(r["D"], 7).tolist(),
                "fit_seconds": round(r["seconds"], 2),
            }
        )
    both = sorted(set(found[0]) & set(found[1]))
    if len(both) >= 5:
        report["stereo"] = stereo_extrinsics(
            obj,
            [found[0][k] for k in both],
            [found[1][k] for k in both],
            (results[0]["K"], results[0]["D"]),
            (results[1]["K"], results[1]["D"]),
        )
    (out / "calib.json").write_text(json.dumps(report, indent=2))
    (out / "camchain.yaml").write_text(camchain_yaml(report, (w, h)))
    return report


def save_detections(path: Path, found: dict, board, max_px: float, stride: int) -> None:
    """The board search is the slow part (minutes); keep its result so refits are instant."""
    arrays = {"params": np.array([*board, max_px, stride], dtype=float)}
    for cam in (0, 1):
        keys = sorted(found[cam])
        arrays[f"keys{cam}"] = np.array(keys, dtype=int)
        arrays[f"corners{cam}"] = np.array([found[cam][k] for k in keys]).reshape(len(keys), -1, 2)
    np.savez_compressed(path, **arrays)


def load_detections(path: Path, board, max_px: float, stride: int) -> dict | None:
    if not path.exists():
        return None
    z = np.load(path)
    if not np.allclose(z["params"], [*board, max_px, stride]):
        return None
    return {
        cam: {int(k): c for k, c in zip(z[f"keys{cam}"], z[f"corners{cam}"], strict=True)}
        for cam in (0, 1)
    }


def stereo_extrinsics(obj, img0: list, img1: list, cam0, cam1, max_views: int = 80) -> dict:
    """The cam0 -> cam1 pose (X1 = R X0 + T) at fixed intrinsics.

    Each view's board pose in each camera gives one relative pose; their median is a robust start
    and drops views that disagree (cv2.fisheye.stereoCalibrate aborted on the first real session
    with an ill-conditioned-view assertion). Then one joint least squares over both cameras'
    reprojection refines it, the same objective stereoCalibrate minimises."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    (K0, D0), (K1, D1) = cam0, cam1
    D0, D1 = D0.reshape(4, 1), D1.reshape(4, 1)
    o = obj.reshape(1, -1, 3).astype(np.float64)
    rel, start = [], []
    for c0, c1 in zip(img0, img1, strict=True):
        ok0, r0, t0 = cv2.fisheye.solvePnP(o, c0.reshape(1, -1, 2), K0, D0)
        ok1, r1, t1 = cv2.fisheye.solvePnP(o, c1.reshape(1, -1, 2), K1, D1)
        if not (ok0 and ok1):
            rel.append(None)
            continue
        R0, R1 = cv2.Rodrigues(r0)[0], cv2.Rodrigues(r1)[0]
        R = R1 @ R0.T
        rel.append((R, np.ravel(t1) - R @ np.ravel(t0)))
        start.append((np.ravel(r0), np.ravel(t0)))
    good = [k for k, r in enumerate(rel) if r is not None]
    Ts = np.array([rel[k][1] for k in good])
    T_med = np.median(Ts, axis=0)
    dev = np.linalg.norm(Ts - T_med, axis=1)
    inl = [good[i] for i in np.flatnonzero(dev <= max(3 * np.median(dev), 0.002))]
    U, _, Vt = np.linalg.svd(sum(rel[k][0] for k in inl))
    R_start = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    use = [inl[i] for i in np.linspace(0, len(inl) - 1, min(max_views, len(inl))).astype(int)]
    pose0 = {k: p for k, p in zip(good, start, strict=True)}

    n_pts = o.shape[1]

    def resid(x):
        R = cv2.Rodrigues(x[:3])[0]
        out = []
        for v, k in enumerate(use):
            rk, tk = x[6 + 6 * v : 9 + 6 * v], x[9 + 6 * v : 12 + 6 * v]
            p0 = cv2.fisheye.projectPoints(o, rk, tk, K0, D0)[0].reshape(-1, 2)
            R1 = R @ cv2.Rodrigues(rk)[0]
            p1 = cv2.fisheye.projectPoints(o, cv2.Rodrigues(R1)[0], R @ tk + x[3:6], K1, D1)[
                0
            ].reshape(-1, 2)
            out += [(p0 - img0[k]).ravel(), (p1 - img1[k]).ravel()]
        return np.concatenate(out)

    x0 = np.concatenate([cv2.Rodrigues(R_start)[0].ravel(), T_med, *[np.r_[pose0[k]] for k in use]])
    sparsity = lil_matrix((4 * n_pts * len(use), len(x0)), dtype=int)
    for v in range(len(use)):
        rows = slice(4 * n_pts * v, 4 * n_pts * (v + 1))
        sparsity[rows, :6] = 1
        sparsity[rows, 6 + 6 * v : 12 + 6 * v] = 1
    fit = least_squares(
        resid, x0, jac_sparsity=sparsity, loss="huber", f_scale=1.0, x_scale="jac", max_nfev=200
    )
    R, T = cv2.Rodrigues(fit.x[:3])[0], fit.x[3:6]
    err = fit.fun.reshape(len(use), 2, n_pts, 2)
    c1 = -R.T @ T  # where the second lens sits, in camera 0's frame (x right, y down, z forward)
    return {
        "views_common": len(img0),
        "views_agreeing": len(inl),
        "views_used": len(use),
        "rms_px": [round(float(np.sqrt(np.mean(err[:, c] ** 2))), 4) for c in (0, 1)],
        "R": np.round(R, 6).tolist(),
        "T_m": np.round(T, 5).tolist(),
        "baseline_m": round(float(np.linalg.norm(T)), 5),
        "cam1_in_cam0_mm": np.round(c1 * 1000, 2).tolist(),
        "rotation_deg": round(float(np.degrees(np.linalg.norm(fit.x[:3]))), 3),
        # How much single views disagree on the baseline: a sanity check, not the error of the fit.
        "per_view_baseline_mm_std": round(
            float(np.std([np.linalg.norm(rel[k][1]) for k in inl]) * 1000), 2
        ),
    }


def camchain_yaml(report: dict, size: tuple[int, int]) -> str:
    """Kalibr camchain (pinhole + equidistant = OpenCV fisheye). T_cn_cnm1 maps cam0 to cam1."""
    lines = []
    for i, c in enumerate(report["cameras"]):
        K = c["K"]
        lines += [
            f"cam{i}:",
            "  camera_model: pinhole",
            f"  intrinsics: [{K[0][0]}, {K[1][1]}, {K[0][2]}, {K[1][2]}]",
            "  distortion_model: equidistant",
            f"  distortion_coeffs: {c['D']}",
            f"  resolution: [{size[0]}, {size[1]}]",
            f"  rostopic: /cam{i}/image_raw",
        ]
        if i == 1 and "stereo" in report:
            R, T = report["stereo"]["R"], report["stereo"]["T_m"]
            lines.append("  T_cn_cnm1:")
            for r in range(3):
                lines.append(f"  - [{R[r][0]}, {R[r][1]}, {R[r][2]}, {T[r]}]")
            lines.append("  - [0.0, 0.0, 0.0, 1.0]")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tars_scan.calib", description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path)
    # Axel's board, measured 2026-11-09: 11x8 squares, 11 squares = 275.3 mm (8 = ~200 mm).
    p.add_argument("--board", default="10x7", help="inner corners: 10x7 for 11x8 squares")
    p.add_argument("--square-mm", type=float, default=25.03, help="measured square size")
    # Defaults from select_eval (A157).
    p.add_argument("--budget", type=int, default=160, help="views per camera to calibrate on")
    p.add_argument("--max-px", type=float, default=2.0, help="largest predicted smear to search")
    p.add_argument("--stride", type=int, default=3, help="search every n-th candidate frame")
    a = p.parse_args(argv)
    board = tuple(int(x) for x in a.board.lower().split("x"))
    r = calibrate_session(a.session, board, a.square_mm / 1000, a.budget, a.max_px, a.stride)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
