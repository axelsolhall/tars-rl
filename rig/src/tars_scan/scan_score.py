"""Score a scan against a reference scan of the same place (Axel, 2026-11-13: one very thorough pass
with board resets is the reference; quick everyday passes are scored against it, automatically).

  python -m tars_scan.scan_score <reference dir> <test dir> [--tau 0.05] [--view]

Both dirs are scan_mesh outputs (fuse run). Each is drawn at its own draw_level (or --level).
1. Register the test scan to the reference. Both worlds are gravity-aligned (z up: Basalt, and the
   board-anchored "up" trajectories), so only yaw and a translation are unknown: a yaw sweep, each
   yaw's xy shift from the cross-correlation of top-down wall images, z from the floor heights; then
   trimmed point-to-plane ICP (all 6 DoF). Kept only if enough points end up within 2 tau.
2. Metrics on mesh vertices (~2 cm apart), the standard reconstruction scores (Knapitsch et al.
   2017, Tanks and Temples): precision = share of the test surface within tau of the reference
   (wrong surfaces, e.g. through walls, lower it); recall = share of the reference surface within
   tau of the test, counted only where the test walk looked (within its max depth of its path);
   F = 2PR / (P + R). Also median distance test -> reference.
-> <test dir>/score_<reference name>.json; --view: test coloured by distance over the reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .scan_mesh import BANDS, drawn, load_snapshots, load_volume

CELL = 0.05  # top-down image cell (m) for the coarse registration


def surface(d: Path, level: float | None = None, with_z: bool = False):
    """(vertices, vertex normals, camera path, params[, mean viewing distance per vertex]) of a
    scan's final drawn mesh."""
    _, vol, prm = load_volume(d)
    m = drawn(load_snapshots(d / "snapshots.npz")[-1], level or prm["draw_level"], prm["min_faces"])
    out = (np.asarray(m.vertices), np.asarray(m.vertex_normals), vol["path"], prm)
    if with_z:
        z = m.vertex_attributes.get("zview")
        out += (np.asarray(z) if z is not None else np.full(len(m.vertices), np.nan),)
    return out


def _topdown(P: np.ndarray, lo: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = np.zeros(shape, np.float32)
    ij = np.floor((P[:, :2] - lo) / CELL).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1)
    return np.log1p(img)


def _walls(P: np.ndarray, N: np.ndarray) -> np.ndarray:
    return P[np.abs(N[:, 2]) < 0.4]  # vertical surfaces: what pins yaw and xy


def coarse(ref: np.ndarray, ref_n: np.ndarray, test: np.ndarray, test_n: np.ndarray) -> np.ndarray:
    """4x4 test -> reference from a yaw sweep + top-down cross-correlation (z by floor height)."""
    R_w, T_w = _walls(ref, ref_n), _walls(test, test_n)
    span = max(np.ptp(R_w[:, :2], 0).max(), np.ptp(T_w[:, :2], 0).max()) + 1.0
    n = int(2 ** np.ceil(np.log2(2 * span / CELL)))
    lo_r = R_w[:, :2].mean(0) - n * CELL / 2
    F_r = np.fft.rfft2(_topdown(R_w, lo_r, (n, n)))
    best = (-1.0, None)
    c_t = T_w[:, :2].mean(0)
    for yaw in np.radians(np.arange(0, 360, 3)):
        Rz = Rotation.from_euler("z", yaw).as_matrix()
        Q = (T_w - [*c_t, 0]) @ Rz.T
        lo_t = np.array([-n * CELL / 2, -n * CELL / 2])
        F_t = np.fft.rfft2(_topdown(Q, lo_t, (n, n)))
        cc = np.fft.irfft2(F_r * np.conj(F_t), s=(n, n))
        k = int(np.argmax(cc))
        if cc.flat[k] > best[0]:
            i, j = np.unravel_index(k, cc.shape)
            i, j = (i + n // 2) % n - n // 2, (j + n // 2) % n - n // 2
            shift = lo_r - lo_t + np.array([i, j]) * CELL
            best = (cc.flat[k], (Rz, shift))
    Rz, shift = best[1]
    M = np.eye(4)
    M[:3, :3] = Rz
    M[:2, 3] = shift - (Rz[:2, :2] @ c_t)

    def floor(P, N):
        return np.percentile(P[N[:, 2] > 0.8, 2], 10) if (N[:, 2] > 0.8).sum() > 50 else 0

    M[2, 3] = floor(ref, ref_n) - floor((test @ Rz.T), test_n @ Rz.T)
    return M


def icp(ref, ref_n, test, M, max_d=(0.3, 0.15, 0.08, 0.05), iters=10) -> tuple[np.ndarray, float]:
    """Trimmed point-to-plane ICP, coarse to fine; returns (4x4, share of test within 0.1 m)."""
    tree = cKDTree(ref)
    rng = np.random.default_rng(0)
    T = test[rng.choice(len(test), min(len(test), 40000), replace=False)]
    for md in max_d:
        for _ in range(iters):
            P = T @ M[:3, :3].T + M[:3, 3]
            d, j = tree.query(P, distance_upper_bound=md)
            ok = np.isfinite(d)
            if ok.sum() < 100:
                break
            p, q, nn = P[ok], ref[j[ok]], ref_n[j[ok]]
            A = np.c_[np.cross(p, nn), nn]
            b = np.einsum("ij,ij->i", q - p, nn)
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            D = np.eye(4)
            D[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
            D[:3, 3] = x[3:]
            M = D @ M
    P = test @ M[:3, :3].T + M[:3, 3]
    d, _ = tree.query(P)
    return M, float((d < 0.1).mean())


def score(ref_dir: Path, test_dir: Path, tau: float = 0.05, level: float | None = None) -> dict:
    ref, ref_n, _, _ = surface(ref_dir)
    test, test_n, path, prm, zview = surface(test_dir, level, with_z=True)
    if len(test) < 100:
        return {"f": 0.0, "precision": 0.0, "recall": 0.0, "registered": False, "points": len(test)}
    M, inl = icp(ref, ref_n, test, coarse(ref, ref_n, test, test_n))
    P = test @ M[:3, :3].T + M[:3, 3]
    path_r = path @ M[:3, :3].T + M[:3, 3]
    d_tr, _ = cKDTree(ref).query(P)
    # the reference surface the test walk could have seen: within its max depth of its path
    near, _ = cKDTree(path_r).query(ref, distance_upper_bound=prm["max_depth_m"])
    looked = np.isfinite(near)
    d_rt, _ = cKDTree(P).query(ref[looked])
    prec, rec = float((d_tr < tau).mean()), float((d_rt < tau).mean()) if looked.any() else 0.0
    # the distance curve: how right the drawn surface is, by the distance it was seen from
    by_dist = []
    for a, b in zip(BANDS[:-1], BANDS[1:], strict=True):
        sel = (zview >= a) & (zview < b)
        by_dist.append(
            {
                "band_m": f"{a}-{min(b, 3):g}",
                "points": int(sel.sum()),
                "precision": round(float((d_tr[sel] < tau).mean()), 3) if sel.any() else None,
                "median_m": round(float(np.median(d_tr[sel])), 3) if sel.any() else None,
            }
        )
    out = {
        "f": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "median_m": round(float(np.median(d_tr)), 4),
        "tau_m": tau,
        "registered": inl > 0.5,
        "inliers_10cm": round(inl, 3),
        "test_points": int(len(P)),
        "ref_points_looked": int(looked.sum()),
        "precision_by_view_distance": by_dist,
        "T_ref_test": M.round(5).tolist(),
        "draw_level": level or prm["draw_level"],
    }
    if not out["registered"]:
        out["f"] = 0.0  # a scan that cannot be placed scores nothing rather than a random number
    return out


def view(ref_dir: Path, test_dir: Path, sc: dict, tau: float) -> Path:
    import rerun as rr

    ref = surface(ref_dir)[0]
    test = surface(test_dir, sc["draw_level"])[0]
    M = np.array(sc["T_ref_test"])
    P = test @ M[:3, :3].T + M[:3, 3]
    d, _ = cKDTree(ref).query(P)
    t = np.clip(d / (3 * tau), 0, 1)[:, None]
    col = ((1 - t) * [40, 200, 80] + t * [230, 40, 40]).astype(np.uint8)
    rrd = test_dir / f"score_{ref_dir.name}.rrd"
    rr.init("tars_scan_score", recording_id=f"{test_dir.name}-vs-{ref_dir.name}")
    rr.save(str(rrd))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("reference", rr.Points3D(ref, colors=[(150, 150, 150)], radii=0.006), static=True)
    rr.log("test", rr.Points3D(P, colors=col, radii=0.008), static=True)
    return rrd


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("test")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--level", type=float, default=None)
    ap.add_argument("--view", action="store_true")
    a = ap.parse_args(argv)
    ref, test = Path(a.reference), Path(a.test)
    sc = score(ref, test, a.tau, a.level)
    (test / f"score_{ref.name}.json").write_text(json.dumps(sc, indent=2))
    print(json.dumps({k: v for k, v in sc.items() if k != "T_ref_test"}))
    if a.view:
        print(view(ref, test, sc, a.tau))


if __name__ == "__main__":
    main()
