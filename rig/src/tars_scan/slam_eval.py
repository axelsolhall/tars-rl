"""Score a SLAM trajectory against the checkerboard: the true camera path in a calibration film.

  python -m tars_scan.slam_eval <session> <trajectory.txt> [--name basalt]

Ground truth: camera 0's pose relative to the board in every frame that saw it (fisheye PnP at
the calibrated lens, from <session>/calib/detections_all.npz), at the same timestamps the export
wrote (IMU clock). The SLAM's poses (TUM format: t[s] x y z qx qy qz qw, IMU body in its world
frame) are moved to camera 0 with the calibrated T_imu_cam0, then rigidly aligned to the board
path (no scale: stereo + IMU has metric scale, and a scale error should count).

Metrics:
- ATE: RMS position error after alignment, mm, over all board frames the SLAM also tracked.
- RPE 1 s: RMS error of the 1-second relative motion, mm and deg (drift rate, alignment-free).
- coverage: share of board frames for which the SLAM output a pose.
The board poses themselves are good to about 1-3 mm (PnP noise, A160), so differences below that
are not resolved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tars_rig.check import read_csv
from tars_rig.contracts import read_json

from . import export
from .calib import board_points
from .camimu import board_poses
from .slam_calib import rig_model


def ground_truth(session: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t_ns in the IMU clock, R_board_cam0 [n,3,3], p_board_cam0 [n,3])."""
    meta = read_json(session / "meta.json")
    cal = json.loads((session / "calib" / "calib.json").read_text())
    model = rig_model(session)
    det = np.load(session / "calib" / "detections_all.npz")
    corners = {int(k): c for k, c in zip(det["keys0"], det["corners0"], strict=True)}
    obj = board_points(tuple(cal["board"]), cal["square_m"])
    c0 = model["cams"][0]
    poses = board_poses(corners, obj, c0["K"], c0["D"])
    frames = read_csv(session / "cam0" / "frames.csv")
    t_all = export.frame_times_ns(frames, meta["cameras"][0].get("readout_us"))
    t_of = {int(i): t for i, t in zip(frames["index"], t_all, strict=True)}
    keys = sorted(poses)
    t = np.array([t_of[k] + model["time_offset_ns"] for k in keys])
    return t, np.array([poses[k][0] for k in keys]), np.array([poses[k][1] for k in keys])


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.loadtxt(path, comments="#")
    # Seconds or ns: CLOCK_MONOTONIC seconds stay far below 1e8 (3 years of uptime); ns are above.
    t = d[:, 0] * (1e9 if d[0, 0] < 1e8 else 1.0)
    return t, Rotation.from_quat(d[:, 4:8]).as_matrix(), d[:, 1:4]


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising |dst - (R src + t)|."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.diag([1, 1, np.linalg.det(U @ Vt)])
    R = U @ S @ Vt
    return R, md - R @ ms


def score(session: Path, traj: Path, max_dt_ms: float = 2.0) -> dict:
    t_gt, R_gt, p_gt = ground_truth(session)
    t_es, R_wi, p_wi = read_tum(traj)
    T_i_c0 = rig_model(session)["cams"][0]["T_i_c"]
    R_wc = R_wi @ T_i_c0[:3, :3]
    p_wc = p_wi + R_wi @ T_i_c0[:3, 3]
    j = np.clip(np.searchsorted(t_es, t_gt), 1, len(t_es) - 1)
    j = np.where(np.abs(t_es[j - 1] - t_gt) < np.abs(t_es[j] - t_gt), j - 1, j)
    ok = np.abs(t_es[j] - t_gt) <= max_dt_ms * 1e6
    if ok.sum() < 10:
        return {"matched": int(ok.sum()), "error": "too few matching timestamps"}
    gi, ei = np.flatnonzero(ok), j[ok]
    R, t = umeyama_se3(p_wc[ei], p_gt[gi])
    p_al = p_wc[ei] @ R.T + t
    err = np.linalg.norm(p_al - p_gt[gi], axis=1)
    # RPE over ~1 s: pairs (a, b) of matched frames 1 s apart. Nearest frame, not first-after: at
    # 10 fps the first frame after 1 s can sit ~100 ms late and every pair would be dropped.
    tg = t_gt[gi]
    b = np.clip(np.searchsorted(tg, tg + 1e9), 1, len(tg) - 1)
    b = np.where(np.abs(tg[b - 1] - tg - 1e9) < np.abs(tg[b] - tg - 1e9), b - 1, b)
    sel = np.flatnonzero((b > np.arange(len(tg))) & (np.abs(tg[b] - tg - 1e9) < 50e6))
    rpe_t, rpe_r = [], []
    for a in sel:
        bb = b[a]
        ga, gb, ea, eb = gi[a], gi[bb], ei[a], ei[bb]
        d_gt = R_gt[ga].T @ (p_gt[gb] - p_gt[ga])
        d_es = R_wc[ea].T @ (p_wc[eb] - p_wc[ea])
        rpe_t.append(np.linalg.norm(d_gt - d_es))
        dR = (R_gt[ga].T @ R_gt[gb]).T @ (R_wc[ea].T @ R_wc[eb])
        rpe_r.append(np.degrees(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec())))
    path_len = float(np.sum(np.linalg.norm(np.diff(p_gt[gi], axis=0), axis=1)))
    return {
        "board_frames": len(t_gt),
        "matched": int(ok.sum()),
        "coverage": round(float(ok.mean()), 3),
        "ate_rmse_mm": round(float(np.sqrt(np.mean(err**2)) * 1000), 2),
        "ate_max_mm": round(float(err.max() * 1000), 2),
        "rpe_1s_mm": round(float(np.sqrt(np.mean(np.square(rpe_t))) * 1000), 2) if rpe_t else None,
        "rpe_1s_deg": round(float(np.sqrt(np.mean(np.square(rpe_r)))), 3) if rpe_r else None,
        "board_path_m": round(path_len, 2),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tars_scan.slam_eval", description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path)
    p.add_argument("trajectory", type=Path)
    a = p.parse_args(argv)
    print(json.dumps(score(a.session, a.trajectory), indent=2))


if __name__ == "__main__":
    main()
