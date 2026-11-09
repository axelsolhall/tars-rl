"""Calibration frames, most valuable first: rank every frame by how much the camera moved while it
was taken, keep one frame per pause, look for the board only in those, then pick the subset that
pins the lens model down best and calibrate on it.

1. Value (no image decoding): from the gyro, the predicted smear of each frame in pixels,
   motion_px = |omega| * (exposure + rolling-shutter readout) * focal_px. Exposure gives blur;
   readout gives rolling-shutter skew, which bends the board and biases the lens model.
2. Pauses: runs of consecutive frames below `still_px`; each pause is one board pose, so one frame
   per pause (its stillest) carries all its information. A 2-minute move-pause film of ~3600
   frames shrinks to a few dozen candidates.
3. Board detection on candidates in value order, so the expensive step runs only where it can pay.
4. Greedy Max-logDet (Zhao & Vela, "Good Feature Selection", 2019; the same criterion as "Good
   Feature Matching", T-RO 2020): add the view that most increases log det of the lens-parameter
   information (the board pose of each view marginalised out), so views that constrain the same
   parameters as earlier ones are skipped.
5. Calibrate on the pick (OpenCV fisheye: Kannala-Brandt, 4 radial terms); the score that counts is
   the reprojection error on detected views that were not picked (held out).

Without IMU data the value falls back to image sharpness (variance of the Laplacian at 1/4 size).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

INTRINSICS = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")
_K_COLS, _POSE_COLS = slice(0, 8), slice(8, 14)  # cv2.fisheye.projectPoints jacobian layout


def angular_rate_at(t_ns: np.ndarray, gyro_t_ns: np.ndarray, gyro_rad_s: np.ndarray) -> np.ndarray:
    """|omega| (rad/s) at each time, linearly interpolated from the gyro samples."""
    norm = np.linalg.norm(gyro_rad_s, axis=1)
    return np.interp(t_ns, gyro_t_ns, norm, left=np.nan, right=np.nan)


def motion_px(
    t_ns: np.ndarray,
    exposure_us: np.ndarray,
    readout_us: float,
    gyro_t_ns: np.ndarray,
    gyro_rad_s: np.ndarray,
    focal_px: float,
    gyro_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Predicted smear per frame in pixels (lower is better). t_ns is mid-frame."""
    g = gyro_rad_s - (gyro_bias if gyro_bias is not None else 0.0)
    w = angular_rate_at(t_ns, gyro_t_ns, g)
    return w * (exposure_us + readout_us) * 1e-6 * focal_px


def sharpness(img: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(img, cv2.CV_32F).var())


def pauses(value_px: np.ndarray, still_px: float, min_frames: int = 3) -> list[np.ndarray]:
    """Runs of consecutive frames with value_px < still_px (at least min_frames long)."""
    still = np.nan_to_num(value_px, nan=np.inf) < still_px
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0]))
    runs = [np.arange(a, b) for a, b in zip(edges[::2], edges[1::2], strict=True)]
    return [r for r in runs if len(r) >= min_frames]


def pause_representatives(value_px: np.ndarray, still_px: float) -> np.ndarray:
    """The stillest frame of each pause, ordered from stillest to least still."""
    reps = [r[np.argmin(value_px[r])] for r in pauses(value_px, still_px)]
    return np.array(sorted(reps, key=lambda i: value_px[i]), dtype=int)


@dataclass
class View:
    """One frame where the board was found: board points and their image positions."""

    frame: int
    obj: np.ndarray  # (N, 3) board points in metres
    img: np.ndarray  # (N, 2) pixels
    rvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tvec: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _cv(obj: np.ndarray, img: np.ndarray | None = None):
    # (1, N, 3) / (1, N, 2): OpenCV 5's fisheye.calibrate rejects the (N, 1, 3) layout.
    o = np.ascontiguousarray(obj.reshape(1, -1, 3), dtype=np.float64)
    if img is None:
        return o
    return o, np.ascontiguousarray(img.reshape(1, -1, 2), dtype=np.float64)


def calibrate(views: list[View], size: tuple[int, int], K0=None, D0=None):
    """OpenCV fisheye calibration. Returns (rms_px, K, D) and fills each view's pose."""
    import cv2

    objs, imgs = zip(*(_cv(v.obj, v.img) for v in views), strict=True)
    K = np.array(K0, dtype=np.float64) if K0 is not None else np.eye(3)
    D = np.array(D0, dtype=np.float64).reshape(4, 1) if D0 is not None else np.zeros((4, 1))
    flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW
    if K0 is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-9)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        list(objs), list(imgs), size, K, D, flags=flags, criteria=crit
    )
    for v, r, t in zip(views, rvecs, tvecs, strict=True):
        v.rvec, v.tvec = np.ravel(r), np.ravel(t)
    return float(rms), K, D.ravel()


def fit_pose(view: View, K, D) -> None:
    """Board pose for a view at fixed intrinsics (for held-out views and information)."""
    import cv2

    o, i = _cv(view.obj, view.img)
    ok, r, t = cv2.fisheye.solvePnP(o, i, K, D.reshape(4, 1))
    if ok:
        view.rvec, view.tvec = np.ravel(r), np.ravel(t)


def reprojection_rms(views: list[View], K, D, refit_pose: bool) -> float:
    import cv2

    errs = []
    for v in views:
        if refit_pose:
            fit_pose(v, K, D)
        p, _ = cv2.fisheye.projectPoints(_cv(v.obj), v.rvec, v.tvec, K, D.reshape(4, 1))
        errs.append(np.sum((p.reshape(-1, 2) - v.img) ** 2, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(errs)))) if errs else float("nan")


def view_information(view: View, K, D) -> np.ndarray:
    """8x8 information on the intrinsics from one view, its board pose marginalised out (Schur
    complement), per unit pixel noise."""
    import cv2

    _, J = cv2.fisheye.projectPoints(_cv(view.obj), view.rvec, view.tvec, K, D.reshape(4, 1))
    Jk, Jp = J[:, _K_COLS], J[:, _POSE_COLS]
    Hkk, Hkp, Hpp = Jk.T @ Jk, Jk.T @ Jp, Jp.T @ Jp
    return Hkk - Hkp @ np.linalg.solve(Hpp + 1e-9 * np.eye(6), Hkp.T)


def noise_px(value_px: np.ndarray, base_px: float = 0.15, per_smear_px: float = 0.5) -> np.ndarray:
    """Expected corner noise for a frame from its predicted smear (A156); a view's
    information is scaled by 1 / noise^2, so a still view counts more than a smeared one with the
    same geometry. The two constants are a prior until real calibrations fit them (residual vs
    predicted smear)."""
    return base_px + per_smear_px * np.nan_to_num(value_px, nan=10.0)


def candidates(value_px: np.ndarray, max_px: float, stride: int) -> np.ndarray:
    """Frames worth a board search: predicted smear below max_px, every stride-th, most valuable
    first. Slightly moving frames are kept on purpose: they add board poses, which pins the
    principal point and distortion better than a few still poses (select_eval)."""
    ok = np.flatnonzero(np.nan_to_num(value_px, nan=np.inf) < max_px)[::stride]
    return ok[np.argsort(value_px[ok])]


def greedy_logdet(infos: list[np.ndarray], budget: int, prior: float = 1e-6) -> list[int]:
    """Indices of up to `budget` views chosen greedily by the largest log-det gain."""
    total = prior * np.eye(infos[0].shape[0])
    chosen: list[int] = []
    left = set(range(len(infos)))
    while left and len(chosen) < budget:
        base = np.linalg.slogdet(total)[1]
        gains = {i: np.linalg.slogdet(total + infos[i])[1] - base for i in left}
        best = max(gains, key=gains.get)
        chosen.append(best)
        total = total + infos[best]
        left.discard(best)
    return chosen


def select_and_calibrate(
    views: list[View],
    size: tuple[int, int],
    budget: int,
    K0,
    D0,
    weights: np.ndarray | None = None,
    seed_views: int = 8,
) -> dict:
    """Initial fit on the first `seed_views` (already in value order), then greedy Max-logDet up to
    `budget` (each view's information times its weight, 1 / noise^2), refit, and score on the
    held-out rest."""
    t0 = time.perf_counter()
    seed = views[: min(seed_views, len(views))]
    _, K, D = calibrate(seed, size, K0, D0)
    for v in views:
        fit_pose(v, K, D)
    w = np.ones(len(views)) if weights is None else np.asarray(weights, float)
    infos = [wi * view_information(v, K, D) for v, wi in zip(views, w, strict=True)]
    pick = greedy_logdet(infos, budget)
    chosen = [views[i] for i in pick]
    rms, K, D = calibrate(chosen, size, K, D)
    held = [v for i, v in enumerate(views) if i not in set(pick)]
    return {
        "K": K,
        "D": D,
        "frames": [v.frame for v in chosen],
        "rms_px": rms,
        "heldout_rms_px": reprojection_rms(held, K, D, refit_pose=True) if held else None,
        "heldout_views": len(held),
        "seconds": time.perf_counter() - t0,
    }
