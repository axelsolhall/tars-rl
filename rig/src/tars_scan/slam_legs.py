"""Drift between board stops: score a SLAM run on a walk that keeps coming back to the board.

  python -m tars_scan.slam_legs <session> <trajectory.txt> [--name basalt]

A walk film has the board only at its stops (board, room, board, ...), so slam_eval's ATE over all
board frames mostly scores the stops themselves. This scores what happens between them instead.

- Stops: runs of board frames (slam_eval.ground_truth), split where more than STOP_GAP_S passes
  without the board. A stop's pose is its middle frame that the SLAM also tracked.
- Leg k (stop k -> stop k+1): the camera's motion between the two stops as the board measures it
  vs as the SLAM measures it (both in the frame of stop k, so no alignment is needed). Error =
  |translation difference| in mm and the rotation difference in degrees; leg length is the SLAM's
  own path between the stops.
- Return k (stop 0 -> stop k): the same against the first stop, i.e. the drift a map without loop
  closure has piled up by the time it is back at the board.
Board poses are good to ~1-3 mm (A160), so leg errors below that are not resolved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .slam_calib import rig_model
from .slam_eval import ground_truth, read_tum

STOP_GAP_S = 2.0
MAX_DT_NS = 2e6


def stops(t_gt: np.ndarray) -> list[np.ndarray]:
    """Index runs of board frames, split at gaps longer than STOP_GAP_S."""
    cut = np.flatnonzero(np.diff(t_gt) > STOP_GAP_S * 1e9) + 1
    return np.split(np.arange(len(t_gt)), cut)


def _relative(Ra, pa, Rb, pb):
    return Ra.T @ (pb - pa), Ra.T @ Rb


def _error(gt, es) -> tuple[float, float]:
    dt = float(np.linalg.norm(gt[0] - es[0]))
    dr = float(np.degrees(np.linalg.norm(Rotation.from_matrix(gt[1].T @ es[1]).as_rotvec())))
    return dt, dr


def legs(t_gt, R_gt, p_gt, t_es, R_wc, p_wc) -> dict:
    """Per-leg and return errors; inputs are camera-0 poses (board frame / SLAM world)."""
    j = np.clip(np.searchsorted(t_es, t_gt), 1, len(t_es) - 1)
    j = np.where(np.abs(t_es[j - 1] - t_gt) < np.abs(t_es[j] - t_gt), j - 1, j)
    ok = np.abs(t_es[j] - t_gt) <= MAX_DT_NS
    picked = []  # (gt index, slam index, start s, frames) per stop the SLAM tracked
    for run in stops(t_gt):
        run = run[ok[run]]
        if len(run):
            g = run[len(run) // 2]
            picked.append((g, j[g], (t_gt[g] - t_gt[0]) / 1e9, len(run)))
    out_legs, out_ret = [], []
    for k in range(1, len(picked)):
        for ref, dest in ((picked[k - 1], out_legs), (picked[0], out_ret)):
            (ga, ea, ta, _), (gb, eb, tb, _) = ref, picked[k]
            dt, dr = _error(
                _relative(R_gt[ga], p_gt[ga], R_gt[gb], p_gt[gb]),
                _relative(R_wc[ea], p_wc[ea], R_wc[eb], p_wc[eb]),
            )
            seg = p_wc[min(ea, eb) : max(ea, eb) + 1]
            path = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
            dest.append(
                {
                    "from_s": round(ta, 1),
                    "to_s": round(tb, 1),
                    "path_m": round(path, 2),
                    "err_mm": round(dt * 1000, 1),
                    "err_deg": round(dr, 2),
                }
            )
    e = np.array([x["err_mm"] for x in out_legs])
    m = np.array([x["path_m"] for x in out_legs])
    return {
        "stops": len(picked),
        "stop_frames": [p[3] for p in picked],
        "legs": out_legs,
        "returns": out_ret,
        "leg_err_mm_median": round(float(np.median(e)), 1) if len(e) else None,
        "leg_err_pct_of_path": round(float(e.sum() / 10 / m.sum()), 2) if m.sum() > 0 else None,
        "leg_err_deg_median": round(float(np.median([x["err_deg"] for x in out_legs])), 2)
        if out_legs
        else None,
    }


def score(session: Path, traj: Path) -> dict:
    t_gt, R_gt, p_gt = ground_truth(session)
    t_es, R_wi, p_wi = read_tum(traj)
    T_i_c0 = rig_model(session)["cams"][0]["T_i_c"]
    return legs(t_gt, R_gt, p_gt, t_es, R_wi @ T_i_c0[:3, :3], p_wi + R_wi @ T_i_c0[:3, 3])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tars_scan.slam_legs", description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path)
    p.add_argument("trajectory", type=Path)
    a = p.parse_args(argv)
    print(json.dumps(score(a.session, a.trajectory), indent=2))


if __name__ == "__main__":
    main()
