"""Dense stereo depth from the two fisheye cameras, scored against the calibration board.

  python -m tars_scan.stereo_depth <session> <export dir> [--step 10] [--bandpass 8,1] [--small]

Obstacle mapping needs depth per pixel, not only the SLAM's sparse points. This rectifies the pair
(OpenCV fisheye model, the calibrated stereo extrinsics), runs semi-global block matching, and
compares the depth inside the board with the board plane from ground truth (board PnP pose, see
slam_eval): the board is a flat target at a known place in every scored frame.

Metrics, over pixels inside the board (inner-corner rectangle, shrunk by one square):
- valid: share of board pixels with a disparity.
- abs error: median and p90 of |depth - plane depth|, mm; relative: median of that / plane depth.
- by distance: the same split by plane depth.
- ms per frame (band-pass + rectify + match, this machine).
--small rectifies to half the size (410x308 from the half-res export) with half the disparity
range: ~20x faster on the Pi 5 (12 vs 248 ms). The band-pass runs on the rectified images.
Writes <export>/stereo_depth.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .calib import board_points
from .slam_calib import rig_model
from .slam_eval import ground_truth
from .track_eval import bandpass

FULL_RES = (1640, 1232)
NUM_DISP = 128  # half res: 0.3 m at the rectified focal length still fits
PINHOLE_FOCAL = 0.6  # rectified focal / fisheye focal: ~68 deg x 54 deg of view at half res


def rectifier(session: Path, scale: float, size: tuple[int, int], out: float = 1.0) -> dict:
    """Rectification maps and projection matrices for the export's resolution."""
    import cv2

    cal = json.loads((session / "calib" / "calib.json").read_text())
    model = rig_model(session)
    Ks, Ds = [], []
    for c in model["cams"]:
        K = c["K"].copy()
        K[:2, :2] *= scale
        K[:2, 2] = (K[:2, 2] + 0.5) * scale - 0.5
        Ks.append(K)
        Ds.append(c["D"].reshape(4, 1))
    R, T = np.array(cal["stereo"]["R"]), np.array(cal["stereo"]["T_m"]).reshape(3, 1)
    # cv2.fisheye.stereoRectify returns a degenerate focal length (~0.002 px) for these wide lenses,
    # so: a virtual pinhole of our choosing, rotations from the distortion-free stereoRectify.
    size_out = (round(size[0] * out), round(size[1] * out))
    f = Ks[0][0, 0] * PINHOLE_FOCAL * out
    c = ((size_out[0] - 1) / 2, (size_out[1] - 1) / 2)
    K_new = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
    R1, R2, P1, P2, *_ = cv2.stereoRectify(
        K_new,
        np.zeros(5),
        K_new,
        np.zeros(5),
        size_out,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    maps = [
        cv2.fisheye.initUndistortRectifyMap(K, D, Rr, P, size_out, cv2.CV_16SC2)
        for K, D, Rr, P in ((Ks[0], Ds[0], R1, P1), (Ks[1], Ds[1], R2, P2))
    ]
    return {"maps": maps, "R1": R1, "P1": P1, "f": P1[0, 0], "baseline": -P2[0, 3] / P2[0, 0]}


def matcher(num_disp: int = NUM_DISP, block: int = 5):
    import cv2

    bs = block
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=bs,
        P1=8 * bs * bs,
        P2=32 * bs * bs,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def depth(rect: dict, sgbm, left: np.ndarray, right: np.ndarray, band=None) -> np.ndarray:
    """Depth in metres along rectified cam0's z, NaN where there is no match."""
    import cv2

    l_r = cv2.remap(left, *rect["maps"][0], cv2.INTER_LINEAR)
    r_r = cv2.remap(right, *rect["maps"][1], cv2.INTER_LINEAR)
    if band:  # after rectification: cheaper on the small images, same band in rectified pixels
        l_r, r_r = bandpass(l_r, *band), bandpass(r_r, *band)
    if rect["baseline"] < 0:  # cam0 is the right-hand camera: mirror so the matcher sees L/R
        d = sgbm.compute(l_r[:, ::-1].copy(), r_r[:, ::-1].copy())[:, ::-1]
    else:
        d = sgbm.compute(l_r, r_r)
    d = d.astype(np.float32) / 16
    with np.errstate(divide="ignore"):
        return np.where(d > 0, rect["f"] * abs(rect["baseline"]) / d, np.nan)


def board_truth(rect: dict, R_bc: np.ndarray, p_bc: np.ndarray, corners: np.ndarray, shape):
    """(mask of board pixels, plane depth per pixel) in the rectified cam0 image."""
    import cv2

    R1, K = rect["R1"], rect["P1"][:, :3]
    to_rect = lambda X: (R1 @ (R_bc.T @ (X - p_bc).T)).T  # noqa: E731  board frame -> rectified
    quad = to_rect(corners)
    if (quad[:, 2] <= 0.05).any():
        return None, None
    px = (K @ quad.T).T
    px = px[:, :2] / px[:, 2:]
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(px).astype(np.int32), 1)
    n = R1 @ R_bc.T @ np.array([0.0, 0.0, 1.0])  # plane normal, rectified frame
    o = to_rect(np.zeros((1, 3)))[0]  # board origin, rectified frame
    v, u = np.mgrid[0 : shape[0], 0 : shape[1]]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(shape)], -1)
    with np.errstate(divide="ignore"):
        z = (n @ o) / (rays @ n)
    return mask.astype(bool), z


def evaluate(
    session: Path,
    export_dir: Path,
    step: int,
    band: tuple[float, float] | None,
    small: bool,
    depth_fn=None,
) -> dict:
    """depth_fn: another depth method with depth()'s signature (phase_stereo plugs in here)."""
    import cv2

    cal = json.loads((session / "calib" / "calib.json").read_text())
    exp = json.loads((export_dir / "export.json").read_text())
    scale = exp.get("scale", 0.5)
    sq = cal["square_m"]
    obj = board_points(tuple(cal["board"]), sq)
    lo, hi = obj[:, :2].min(0) + sq, obj[:, :2].max(0) - sq  # shrink: edge pixels mix in the wall
    corners = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]])
    t_gt, R_gt, p_gt = ground_truth(session)
    files = {int(p.stem): p for p in (export_dir / "mav0" / "cam0" / "data").glob("*.png")}
    first = cv2.imread(str(next(iter(files.values()))), cv2.IMREAD_GRAYSCALE)
    size = (first.shape[1], first.shape[0])
    out = 0.5 if small else 1.0
    rect, sgbm = rectifier(session, scale, size, out), matcher(NUM_DISP // 2 if small else NUM_DISP)
    errs, rels, zs, valid, total, ms = [], [], [], 0, 0, []
    # The export and the ground truth may use slightly different camera-IMU offsets: nearest frame.
    stamps = np.array(sorted(files))
    for i in range(0, len(t_gt), step):
        j = int(np.clip(np.searchsorted(stamps, t_gt[i]), 1, len(stamps) - 1))
        t = min(stamps[j - 1], stamps[j], key=lambda s: abs(s - t_gt[i]))
        if abs(t - t_gt[i]) > 2e6:
            continue
        left = cv2.imread(str(files[t]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(export_dir / "mav0" / "cam1" / "data" / files[t].name), 0)
        t0 = time.perf_counter()
        z = (depth_fn or depth)(rect, sgbm, left, right, band)
        ms.append((time.perf_counter() - t0) * 1000)
        mask, z_true = board_truth(rect, R_gt[i], p_gt[i], corners, z.shape)
        if mask is None:
            continue
        total += int(mask.sum())
        ok = mask & np.isfinite(z)
        valid += int(ok.sum())
        e = np.abs(z[ok] - z_true[ok])
        errs.append(e)
        rels.append(e / z_true[ok])
        zs.append(z_true[ok])
    e, r, zt = np.concatenate(errs), np.concatenate(rels), np.concatenate(zs)
    bins = {}
    for a, b in ((0.0, 0.45), (0.45, 0.55), (0.55, 10.0)):
        m = (zt >= a) & (zt < b)
        if m.any():
            bins[f"{a:g}-{b:g} m"] = {
                "pixels": int(m.sum()),
                "abs_median_mm": round(float(np.median(e[m])) * 1000, 2),
                "rel_median_pct": round(float(np.median(r[m])) * 100, 2),
            }
    return {
        "frames": len(ms),
        "bandpass": list(band) if band else None,
        "small": small,
        "valid": round(valid / max(total, 1), 3),
        "abs_median_mm": round(float(np.median(e)) * 1000, 2),
        "abs_p90_mm": round(float(np.percentile(e, 90)) * 1000, 2),
        "rel_median_pct": round(float(np.median(r)) * 100, 2),
        "by_distance": bins,
        "ms_per_frame": round(float(np.median(ms)), 1),
        "rect_focal_px": round(float(rect["f"]), 1),
        "baseline_m": round(float(rect["baseline"]), 4),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tars_scan.stereo_depth", description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path)
    p.add_argument("export", type=Path)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--bandpass", help="lo,hi sigma in rectified px, e.g. 8,1 (track_eval.bandpass)")
    p.add_argument("--small", action="store_true", help="rectify to half size, half disparities")
    a = p.parse_args(argv)
    band = tuple(float(x) for x in a.bandpass.split(",")) if a.bandpass else None
    r = evaluate(a.session, a.export, a.step, band, a.small)
    name = "stereo_depth" + ("_small" if a.small else "")
    name += (f"_bp{a.bandpass.replace(',', '-')}" if band else "") + ".json"
    (a.export / name).write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
