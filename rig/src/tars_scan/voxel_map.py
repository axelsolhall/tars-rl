"""Obstacle map: stereo depth placed with SLAM poses, binned into voxels, scored on the board.

  python -m tars_scan.voxel_map <session> <export dir> <trajectory.txt | gt> [--step 6]

Each scored frame: dense depth (stereo_depth, small rectification + band-pass), points moved to
the world with the SLAM pose at that frame (TUM file, IMU body in world, as slam_eval reads it),
binned into VOXEL_M cubes; a voxel counts as occupied once MIN_HITS frames put a point in it, and
is kept as the mean of its points (a voxel centre would quantise the plane distance to 1 cm).
`gt` places the points with the board poses instead: the best map this depth can give.

Score, in the board's frame (SLAM world aligned to the board path rigidly, as slam_eval's ATE):
- board RMS mm: distance of occupied voxels over the board area to the board plane (pose drift
  and depth error both smear the board into a thicker slab).
- board coverage: share of 2 cm cells of the board area that got an occupied voxel.
- thick share: occupied voxels over the board area that sit 2-10 cm off the plane (ghost copies).
Writes <export>/voxel_map_<name>.json and the occupied voxels' mean points as .npy (world).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calib import board_points
from .slam_calib import rig_model
from .slam_eval import ground_truth, read_tum, umeyama_se3
from .stereo_depth import NUM_DISP, depth, matcher, rectifier

VOXEL_M = 0.02
MIN_HITS = 3
MAX_DEPTH_M = 3.0
BAND = (4.0, 0.5)  # rectified px, the best small-rectification setting in stereo_depth
TIME_TOL_NS = 2e6


def frame_points(rect: dict, z: np.ndarray, every: int = 2) -> np.ndarray:
    """Points [n, 3] in camera-0 coordinates (not rectified) from a rectified depth map."""
    K = rect["P1"][:, :3]
    v, u = np.mgrid[0 : z.shape[0] : every, 0 : z.shape[1] : every]
    d = z[::every, ::every]
    ok = np.isfinite(d) & (d < MAX_DEPTH_M)
    x = (u[ok] - K[0, 2]) / K[0, 0] * d[ok]
    y = (v[ok] - K[1, 2]) / K[1, 1] * d[ok]
    return np.c_[x, y, d[ok]] @ rect["R1"]  # rectified -> camera 0: R1^T applied from the right


def poses_world_cam0(session: Path, traj: str, stamps: np.ndarray) -> dict[int, np.ndarray]:
    """4x4 camera-0 pose per frame stamp: from a SLAM trajectory, or the board (`gt`)."""
    out = {}
    if traj == "gt":
        t, R, p = ground_truth(session)
    else:
        t, R_wi, p_wi = read_tum(Path(traj))
        T = rig_model(session)["cams"][0]["T_i_c"]
        R, p = R_wi @ T[:3, :3], p_wi + R_wi @ T[:3, 3]
    j = np.clip(np.searchsorted(t, stamps), 1, len(t) - 1)
    j = np.where(np.abs(t[j - 1] - stamps) < np.abs(t[j] - stamps), j - 1, j)
    for s, k in zip(stamps, j, strict=True):
        if abs(t[k] - s) <= TIME_TOL_NS:
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = R[k], p[k]
            out[int(s)] = M
    return out


def align_to_board(session: Path, traj: str) -> np.ndarray:
    """4x4 world -> board transform: the rigid fit slam_eval uses for ATE (identity for gt)."""
    if traj == "gt":
        return np.eye(4)
    t_gt, _, p_gt = ground_truth(session)
    poses = poses_world_cam0(session, traj, t_gt)
    keep = [i for i, s in enumerate(t_gt) if int(s) in poses]
    src = np.array([poses[int(t_gt[i])][:3, 3] for i in keep])
    R, t = umeyama_se3(src, p_gt[keep])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def build(session: Path, export_dir: Path, traj: str, step: int) -> np.ndarray:
    """Occupied voxels' mean points [n, 3] in the SLAM world frame."""
    import cv2

    exp = json.loads((export_dir / "export.json").read_text())
    files = sorted((export_dir / "mav0" / "cam0" / "data").glob("*.png"))[::step]
    stamps = np.array([int(f.stem) for f in files])
    poses = poses_world_cam0(session, traj, stamps)
    first = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    size = (first.shape[1], first.shape[0])
    rect = rectifier(session, exp.get("scale", 0.5), size, 0.5)
    sgbm = matcher(NUM_DISP // 2)
    keys, means = [], []
    for f, s in zip(files, stamps, strict=True):
        if int(s) not in poses:
            continue
        left = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(export_dir / "mav0" / "cam1" / "data" / f.name), 0)
        pts = frame_points(rect, depth(rect, sgbm, left, right, BAND))
        M = poses[int(s)]
        w = pts @ M[:3, :3].T + M[:3, 3]
        k, inv = np.unique(np.floor(w / VOXEL_M).astype(np.int32), axis=0, return_inverse=True)
        inv = inv.ravel()
        mean = np.zeros((len(k), 3))
        np.add.at(mean, inv, w)
        keys.append(k)  # one hit per frame and voxel
        means.append(mean / np.bincount(inv)[:, None])
    allk, inv, hits = np.unique(
        np.concatenate(keys), axis=0, return_inverse=True, return_counts=True
    )
    inv = inv.ravel()
    mean = np.zeros((len(allk), 3))
    np.add.at(mean, inv, np.concatenate(means))
    return (mean / hits[:, None])[hits >= MIN_HITS]


def score(session: Path, vox_world: np.ndarray, traj: str) -> dict:
    cal = json.loads((session / "calib" / "calib.json").read_text())
    sq = cal["square_m"]
    obj = board_points(tuple(cal["board"]), sq)
    lo, hi = obj[:, :2].min(0) + sq, obj[:, :2].max(0) - sq
    M = align_to_board(session, traj)
    b = vox_world @ M[:3, :3].T + M[:3, 3]
    over = (b[:, 0] >= lo[0]) & (b[:, 0] <= hi[0]) & (b[:, 1] >= lo[1]) & (b[:, 1] <= hi[1])
    near = over & (np.abs(b[:, 2]) < 0.10)
    z = b[near, 2]
    on = over & (np.abs(b[:, 2]) < VOXEL_M)
    cells = {tuple(c) for c in np.floor(b[on, :2] / VOXEL_M).astype(int)}
    n_cells = int(np.prod(np.ceil((hi - lo) / VOXEL_M)))
    return {
        "voxels": len(vox_world),
        "board_voxels": int(near.sum()),
        "board_rms_mm": round(float(np.sqrt(np.mean(z**2))) * 1000, 1) if len(z) else None,
        "board_coverage": round(min(len(cells) / n_cells, 1.0), 3),
        "thick_share": round(float(np.mean(np.abs(z) >= VOXEL_M)), 3) if len(z) else None,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tars_scan.voxel_map", description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path)
    p.add_argument("export", type=Path)
    p.add_argument("trajectory", help="TUM trajectory file, or gt for the board poses")
    p.add_argument("--step", type=int, default=6, help="use every n-th frame (6 = 5 Hz)")
    a = p.parse_args(argv)
    vox = build(a.session, a.export, a.trajectory, a.step)
    r = {"trajectory": a.trajectory, "step": a.step, **score(a.session, vox, a.trajectory)}
    name = "gt" if a.trajectory == "gt" else Path(a.trajectory).stem
    np.save(a.export / f"voxel_map_{name}.npy", vox.astype(np.float32))
    (a.export / f"voxel_map_{name}.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
