"""Eval of tars_scan.select on a simulated move-pause calibration film with a known lens.

The simulation follows the real rig: 1640x1232 at 30 fps, 6 ms exposure, a rolling shutter that
reads the rows over `readout_us`, a 200 Hz gyro with noise and bias, and a 9x6-corner board that
the camera visits in pauses, moving in between. Corners of moving frames are both noisier (blur)
and bent (each row sees a different pose); the board is only "detected" when the smear is small
enough and the whole board is in view.

Score: lens error in pixels. Over a grid of image points within 60 degrees of the axis, how far
the estimated lens model puts each ray from where the true lens puts it (RMS). Baselines: every
detected frame, and random subsets; against those, the still-frame pick and the still + Max-logDet
pick. Also counted: how many frames need a board search (the costly step).

  python -m tars_scan.select_eval [--seeds 3]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import select

W, H, FPS, RATE = 1640, 1232, 30, 200
TRUE_K = np.array([[781.0, 0, 823.0], [0, 783.5, 612.0], [0, 0, 1]])
TRUE_D = np.array([0.031, -0.012, 0.004, -0.0015])
BOARD = (9, 6)
SQUARE_M = 0.025


def board_points() -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(BOARD[0]), np.arange(BOARD[1]))
    return np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)] * SQUARE_M


def random_pose(rng) -> tuple[Rotation, np.ndarray]:
    """A camera pose (orientation in the board frame, position) that puts the board somewhere in
    the field of view: centre, edges or corners, tilted, at 0.3-0.9 m."""
    centre = board_points().mean(0)
    dist = rng.uniform(0.3, 0.9)
    tilt, az = np.deg2rad(rng.uniform(0, 50)), rng.uniform(0, 2 * np.pi)
    # Camera sits on the board's -z side (board faces it) along a tilted direction.
    d = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)])
    pos = centre + dist * d
    z = (centre - pos) / np.linalg.norm(centre - pos)  # optical axis at the board centre
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    R = Rotation.from_matrix(np.c_[x, np.cross(z, x), z])
    # Then turn the camera so the board lands off-axis (up to 50 deg) and roll it.
    off = rng.uniform(0, np.deg2rad(50))
    a = rng.uniform(0, 2 * np.pi)
    roll = Rotation.from_rotvec([0, 0, rng.uniform(-np.pi, np.pi)])
    turn = Rotation.from_rotvec([off * np.sin(a), off * np.cos(a), 0])
    return R * roll * turn, pos


def simulate(seed: int, pauses: int = 25, pause_s: float = 2.0, move_s: float = 2.0,
             exposure_us: float = 6000.0, readout_us: float = 23_300.0) -> dict:  # fmt: skip
    rng = np.random.default_rng(seed)
    poses = [random_pose(rng) for _ in range(pauses)]
    seg = pause_s + move_s
    total = pauses * seg - move_s
    rots = Rotation.concatenate([p[0] for p in poses])
    key_t = np.arange(pauses) * seg

    def pose_at(t: np.ndarray):
        t = np.clip(t, 0, total)
        i = np.minimum((t // seg).astype(int), pauses - 1)
        local = t - i * seg
        frac = np.clip((local - pause_s) / move_s, 0, 1)
        s = frac * frac * (3 - 2 * frac)  # smoothstep: still, accelerate, decelerate, still
        j = np.minimum(i + 1, pauses - 1)
        slerp = Slerp(key_t, rots)
        R = slerp(np.minimum(key_t[i] + s * (key_t[j] - key_t[i]), key_t[-1]))
        P = (
            np.array([poses[a][1] for a in i]) * (1 - s)[:, None]
            + np.array([poses[b][1] for b in j]) * s[:, None]
        )
        return R, P

    # Gyro: body rates from the orientation's finite difference, plus noise and a small bias.
    gt = np.arange(0, total, 1 / RATE)
    Ra, _ = pose_at(gt)
    Rb, _ = pose_at(gt + 1e-3)
    omega = (Ra.inv() * Rb).as_rotvec() / 1e-3
    bias = rng.normal(0, 0.003, 3)
    gyro = omega + bias + rng.normal(0, 0.004, omega.shape)

    import cv2

    obj = board_points()
    lt = readout_us / H
    n_frames = int(total * FPS)
    start = np.arange(n_frames) / FPS
    mid_t = start + (H / 2 * lt + exposure_us / 2) * 1e-6
    true_px = select.motion_px(
        mid_t * 1e9, np.full(n_frames, exposure_us), readout_us, gt * 1e9, omega, TRUE_K[0, 0]
    )
    views, n_seen = [], 0
    for k in range(n_frames):
        rows = np.full(len(obj), H / 2)
        for _ in range(2):  # each corner is seen at the time its row is exposed
            t = start[k] + (rows * lt + exposure_us / 2) * 1e-6
            R, P = pose_at(t)
            Xc = np.einsum("nij,nj->ni", R.inv().as_matrix(), obj - P)
            if np.any(Xc[:, 2] <= 0.05):
                break
            img, _ = cv2.fisheye.projectPoints(
                Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), TRUE_K, TRUE_D.reshape(4, 1)
            )
            img = img.reshape(-1, 2)
            rows = img[:, 1]
        else:
            theta = np.arctan2(np.linalg.norm(Xc[:, :2], axis=1), Xc[:, 2])
            inside = (img[:, 0] > 5) & (img[:, 0] < W - 5) & (img[:, 1] > 5) & (img[:, 1] < H - 5)
            if inside.all() and theta.max() < np.deg2rad(75):
                n_seen += 1
                if true_px[k] < 4.0:  # findChessboardCornersSB fails on heavy smear
                    sigma = 0.12 + 0.25 * true_px[k]
                    noisy = img + rng.normal(0, sigma, img.shape)
                    views.append(select.View(k, obj.copy(), noisy))
    value = select.motion_px(
        mid_t * 1e9, np.full(n_frames, exposure_us), readout_us, gt * 1e9, gyro, 780.0
    )
    return {"views": views, "value_px": value, "frames": n_frames, "board_in_view": n_seen}


def lens_error_px(K, D) -> float:
    import cv2

    u, v = np.meshgrid(np.arange(20, W, 40.0), np.arange(20, H, 40.0))
    pts = np.c_[u.ravel(), v.ravel()].reshape(-1, 1, 2)
    norm = cv2.fisheye.undistortPoints(pts, TRUE_K, TRUE_D.reshape(4, 1)).reshape(-1, 2)
    keep = np.arctan(np.linalg.norm(norm, axis=1)) < np.deg2rad(60)
    rays = np.c_[norm[keep], np.ones(keep.sum())].reshape(-1, 1, 3)
    est, _ = cv2.fisheye.projectPoints(
        rays, np.zeros(3), np.zeros(3), np.asarray(K, float), np.asarray(D, float).reshape(4, 1)
    )
    return float(np.sqrt(np.mean(np.sum((est.reshape(-1, 2) - pts.reshape(-1, 2)[keep]) ** 2, 1))))


K0 = np.array([[780.0, 0, W / 2], [0, 780.0, H / 2], [0, 0, 1]])  # from the lens spec alone


def run(seed: int, budgets=(20, 40, 80, 160)) -> list[dict]:
    sim = simulate(seed)
    views, value = sim["views"], sim["value_px"]
    by_frame = {v.frame: v for v in views}
    reps = [by_frame[f] for f in select.pause_representatives(value, 0.5) if f in by_frame]
    rng = np.random.default_rng(seed + 1000)
    out = []

    def score(name, chosen, searched, extra=None):
        t = time.perf_counter()
        try:
            _, K, D = select.calibrate([select.View(v.frame, v.obj, v.img) for v in chosen],
                                       (W, H), K0, np.zeros(4))  # fmt: skip
            err = lens_error_px(K, D)
        except Exception as e:  # noqa: BLE001 - a failed fit is a result too
            err = float("nan")
            name += f" (failed: {type(e).__name__})"
        secs = (extra or {}).get("seconds", 0) + time.perf_counter() - t
        out.append({"seed": seed, "method": name, "views": len(chosen), "searched": searched,
                    "lens_err_px": err, "seconds": secs})  # fmt: skip

    score("all detected frames", views, sim["frames"])
    for b in budgets:
        for _ in range(3):
            pick = rng.choice(len(views), size=min(b, len(views)), replace=False)
            score(f"random {b}", [views[i] for i in pick], sim["frames"])
        score(f"stillest {b}", reps[:b], len(reps))
        r = select.select_and_calibrate(reps, (W, H), b, K0, np.zeros(4))
        chosen = [by_frame[f] for f in r["frames"]]
        score(f"still + logdet {b}", chosen, len(reps), {"seconds": r["seconds"]})
        # Value-ranked candidates (moving frames too), weighted Max-logDet. The selector's noise
        # model is deliberately not the simulator's (0.5 px per smear px vs 0.25).
        cand = [by_frame[f] for f in select.candidates(value, 2.0, 3) if f in by_frame]
        n_searched = len(select.candidates(value, 2.0, 3))
        wts = 1 / select.noise_px(value[[v.frame for v in cand]]) ** 2
        r = select.select_and_calibrate(cand, (W, H), b, K0, np.zeros(4), weights=wts)
        chosen = [by_frame[f] for f in r["frames"]]
        score(f"value + logdet {b}", chosen, n_searched, {"seconds": r["seconds"]})
    cand = [by_frame[f] for f in select.candidates(value, 2.0, 3) if f in by_frame]
    score("all value candidates", cand, len(select.candidates(value, 2.0, 3)))
    for row in out:
        row["pauses_found"] = len(reps)
        row["detected"] = len(views)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tars_scan.select_eval", description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args(argv)
    rows = [r for s in range(args.seeds) for r in run(s)]
    first = rows[0]
    print(
        f"simulated film: {first['detected']} frames with a detectable board, "
        f"{first['pauses_found']} pauses found (seed 0)\n"
    )
    print(f"{'method':<24}{'views':>6}{'searched':>10}{'lens err px':>14}{'fit s':>8}")
    for m in dict.fromkeys(r["method"].split(" (")[0] for r in rows):
        rs = [r for r in rows if r["method"].split(" (")[0] == m]
        e = np.array([r["lens_err_px"] for r in rs])
        print(
            f"{m:<24}{np.mean([r['views'] for r in rs]):>6.0f}"
            f"{np.mean([r['searched'] for r in rs]):>10.0f}"
            f"{np.nanmedian(e):>10.3f} med {np.nanmax(e):>6.2f} max"
            f"{np.mean([r['seconds'] for r in rs]):>6.2f}"
            + (f"  ({np.isnan(e).sum()} failed)" if np.isnan(e).any() else "")
        )


if __name__ == "__main__":
    main()
