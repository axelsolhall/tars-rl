"""Anchor a SLAM trajectory to the board at every stop: offline drift correction on a landmark.

  python -m tars_scan.anchor_poses <session> <trajectory.txt> <out.txt> [--by path|time] [--causal]

Axel (2026-11-10): the walk's returns to the board anchor the mapping. Without loop closure the
SLAM's drift piles up (walk2: 114 -> 517 mm against the first stop), so a map fused from it is only
locally consistent. A robot gets the same anchor from a fiducial at its dock.

- At each stop k (slam_legs.stops, >= MIN_STOP_FRAMES tracked frames) the correction
  W_k = T_board_cam0(PnP) * T_world_cam0(SLAM)^-1 maps the SLAM world into the board frame.
- Between stops W is interpolated (rotation slerp, translation linear), weighted by the SLAM's own
  path length (`--by path`, drift grows with motion) or by time; held constant outside the stops.
- `--causal`: only past anchors (W of the last stop, held): what a robot knows while it drives.
- Output: the IMU trajectory in the board frame, TUM format, same time column as the input.

Eval (leave-one-out): for each stop k, anchor on all other stops (or only earlier ones when causal)
and measure the camera's error at stop k against the board: how far off the map is where no anchor
was used. Compared with no anchoring beyond stop 0 (slam_legs' returns).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .slam_calib import rig_model
from .slam_eval import ground_truth, read_tum
from .slam_legs import MAX_DT_NS, MIN_STOP_FRAMES, stops


def anchors(t_gt, R_gt, p_gt, t_es, R_wc, p_wc) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(slam index, R_W, t_W) per usable stop: the board-from-world correction, middle frame."""
    j = np.clip(np.searchsorted(t_es, t_gt), 1, len(t_es) - 1)
    j = np.where(np.abs(t_es[j - 1] - t_gt) < np.abs(t_es[j] - t_gt), j - 1, j)
    ok = np.abs(t_es[j] - t_gt) <= MAX_DT_NS
    out = []
    for run in stops(t_gt):
        run = run[ok[run]]
        if len(run) < MIN_STOP_FRAMES:
            continue
        g = run[len(run) // 2]
        e = j[g]
        Rw = R_gt[g] @ R_wc[e].T
        out.append((int(e), Rw, p_gt[g] - Rw @ p_wc[e]))
    return out


def correction(s: np.ndarray, anc: list, s_anc: np.ndarray, causal: bool):
    """Per-pose (R_W [n,3,3], t_W [n,3]) for poses at coordinate s (path or time)."""
    Rs = Rotation.from_matrix(np.stack([a[1] for a in anc]))
    ts = np.stack([a[2] for a in anc])
    if causal or len(anc) == 1:
        k = np.clip(np.searchsorted(s_anc, s, side="right") - 1, 0, len(anc) - 1)
        return Rs[k].as_matrix(), ts[k]
    sc = np.clip(s, s_anc[0], s_anc[-1])
    R = Slerp(s_anc, Rs)(sc).as_matrix()
    t = np.stack([np.interp(sc, s_anc, ts[:, i]) for i in range(3)], axis=1)
    return R, t


def coordinate(t_es, R_wc, p_wc, by: str) -> np.ndarray:
    """Where each pose sits along the walk: what the drift correction is interpolated over."""
    if by == "time":
        return (t_es - t_es[0]) / 1e9
    if (
        by == "turn"
    ):  # accumulated rotation (rad): drift grows with turning (slam_legs turn-rate finding)
        rel = np.einsum("nji,njk->nik", R_wc[:-1], R_wc[1:])
        return np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))]
        )
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p_wc, axis=0), axis=1))])


def leave_one_out(t_gt, R_gt, p_gt, t_es, R_wc, p_wc, by="path", causal=False) -> dict:
    """Error at each stop k >= 1 with stop k left out of the anchors."""
    anc = anchors(t_gt, R_gt, p_gt, t_es, R_wc, p_wc)
    s = coordinate(t_es, R_wc, p_wc, by)
    rows = []
    for k in range(1, len(anc)):
        use = [a for i, a in enumerate(anc) if i != k and (not causal or i < k)]
        e = anc[k][0]
        s_use = s[[a[0] for a in use]]
        Rw, tw = correction(s[e : e + 1], use, s_use, causal)
        Rw_true, tw_true = anc[k][1], anc[k][2]
        # corrected camera position at stop k vs the board's
        p_fix = Rw[0] @ p_wc[e] + tw[0]
        p_true = Rw_true @ p_wc[e] + tw_true
        dr = np.linalg.norm(Rotation.from_matrix(Rw_true.T @ Rw[0]).as_rotvec())
        rows.append(
            {
                "stop": k,
                "err_mm": round(float(np.linalg.norm(p_fix - p_true)) * 1000, 1),
                "err_deg": round(float(np.degrees(dr)), 2),
            }
        )
    e = np.array([r["err_mm"] for r in rows])
    return {
        "by": by,
        "causal": causal,
        "stops": len(anc),
        "rows": rows,
        "err_mm_median": round(float(np.median(e)), 1) if len(e) else None,
        "err_mm_max": round(float(e.max()), 1) if len(e) else None,
    }


def _cam0(session: Path, traj: Path):
    t_es, R_wi, p_wi = read_tum(traj)
    T = rig_model(session)["cams"][0]["T_i_c"]
    return t_es, R_wi, p_wi, R_wi @ T[:3, :3], p_wi + R_wi @ T[:3, 3]


def _rotation_onto_z(u: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector u onto +z."""
    axis = np.cross(u, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(axis), float(u[2])
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    return Rotation.from_rotvec(axis / s * np.arctan2(s, c)).as_matrix()


def anchor_trajectory(
    session: Path, traj: Path, out: Path, by="path", causal=False, skip=()
) -> dict:
    """Write the anchored trajectory; `skip` = stop numbers not used as anchors (held out)."""
    t_gt, R_gt, p_gt = ground_truth(session)
    t_es, R_wi, p_wi, R_wc, p_wc = _cam0(session, traj)
    anc = [a for i, a in enumerate(anchors(t_gt, R_gt, p_gt, t_es, R_wc, p_wc)) if i not in skip]
    s = coordinate(t_es, R_wc, p_wc, by)
    Rw, tw = correction(s, anc, s[[a[0] for a in anc]], causal)
    R_new = np.einsum("nij,njk->nik", Rw, R_wi)
    p_new = np.einsum("nij,nj->ni", Rw, p_wi) + tw
    # The board frame's z is the board normal, not up. The SLAM world's z is up (VIO aligns it with
    # gravity), so turn the output until up is z again: height bands (obstacle grid, viewer).
    up = np.mean([a[1][:, 2] for a in anc], axis=0)
    R_up = _rotation_onto_z(up / np.linalg.norm(up))
    R_new = np.einsum("ij,njk->nik", R_up, R_new)
    p_new = p_new @ R_up.T
    raw = np.loadtxt(traj, comments="#")
    q = Rotation.from_matrix(R_new).as_quat()
    np.savetxt(
        out,
        np.column_stack([raw[:, 0], p_new, q]),
        fmt="%.9f",
        header="anchored to the board at each stop, z up (tars_scan.anchor_poses)",
    )
    return {"anchors": len(anc), "skipped": list(skip), "out": str(out)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tars_scan.anchor_poses", description=__doc__.split("\n")[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("trajectory", type=Path)
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--by", choices=("path", "time", "turn"), default="path")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--skip", default="", help="stop numbers not used as anchors, e.g. 1,3,5,7")
    a = ap.parse_args(argv)
    t_gt, R_gt, p_gt = ground_truth(a.session)
    t_es, _, _, R_wc, p_wc = _cam0(a.session, a.trajectory)
    r = {"loo": leave_one_out(t_gt, R_gt, p_gt, t_es, R_wc, p_wc, a.by, a.causal)}
    if a.out:
        r["written"] = anchor_trajectory(
            a.session,
            a.trajectory,
            a.out,
            a.by,
            a.causal,
            tuple(int(x) for x in a.skip.split(",") if x),
        )
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
