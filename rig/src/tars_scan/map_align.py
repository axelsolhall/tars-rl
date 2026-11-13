"""Scan-to-map alignment: nudge a frame's pose so its depth points sit on the map's surfaces.

The SLAM's drift piles up between visits (walk2: 114 -> 517 mm against the first board stop), so a
place seen again lands a little beside where the map has it. Before a frame is fused, a few
point-to-plane ICP steps against the map's surface points (occupancy_map.Grid.surface) estimate a
small correction; it is kept as a world-frame offset applied to all later SLAM poses, so drift
that the map can see is taken out as it happens (no board, no loop-closure database).

Guards, because a wrong pull is worse than none:
- only map points seen many times (hits >= min_hits) are targets; matches farther than max_dist_m
  are ignored;
- a step is accepted only with enough matches (min_matches) and a small residual, and it is clipped
  to max_step_m / max_step_deg per frame;
- the offset is applied with gain < 1, so single frames cannot yank the trajectory.

Eval (the board only scores, it never touches the poses): run occupancy_map --test on a walk with
board stops and raw SLAM poses, then score <export>/occupancy_<name>_aligned.txt with
`python -m tars_scan.slam_legs <session> <it>` and `python -m tars_scan.anchor_poses <session> <it>`
(leave-one-out at the stops; --causal for what a robot knows live).
walk2 2026-11-11: legs 185 -> 106 mm, returns 295 -> 90 mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class AlignParams:
    every: int = 1  # align every k-th fused frame (0 = off)
    points: int = 1500  # depth points used per alignment
    min_hits: int = 8  # map voxels hit at least this often are targets
    max_dist_m: float = 0.10  # ignore pairs farther apart
    min_matches: int = 300
    iters: int = 5
    max_step_m: float = 0.02  # per aligned frame
    max_step_deg: float = 0.5
    gain: float = 0.5  # share of the estimated correction applied
    normal_k: int = 8  # neighbours for the map normals
    # targets = voxels solid when last seen (log-odds before decay), not only those still occupied
    # now: the obstacle map forgets, alignment should not (long gaps between visits)
    memory: bool = False
    # which corrections to estimate: "6dof", "yaw" (position + yaw: VIO pins roll and pitch to
    # gravity, so those only add noise; walk2 2026-11-11: 6dof made leg rotation 1.9 -> 2.3 deg) or
    # "t" (position only)
    dof: str = "6dof"


def _normals(pts: np.ndarray, tree, k: int) -> np.ndarray:
    _, nn = tree.query(pts, k=k)
    q = pts[nn] - pts[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", q, q)
    _, vec = np.linalg.eigh(cov)
    return vec[:, :, 0]  # smallest eigenvalue's vector


class MapAligner:
    """Holds the running world-frame correction (R, t): corrected = R @ slam + t."""

    def __init__(self, p: AlignParams):
        self.p = p
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.stats = {"tried": 0, "accepted": 0, "step_mm": []}
        self._tree = None
        self._tree_pts = None
        self._tree_n = None
        self._built_at = -1

    def correct(self, M: np.ndarray) -> np.ndarray:
        """A SLAM camera pose (4x4, world <- cam) with the running correction applied."""
        out = M.copy()
        out[:3, :3] = self.R @ M[:3, :3]
        out[:3, 3] = self.R @ M[:3, 3] + self.t
        return out

    def _targets(self, grid, now: float, frame_no: int):
        if self._tree is not None and frame_no - self._built_at < 10:
            return self._tree, self._tree_pts, self._tree_n
        from scipy.spatial import cKDTree

        if self.p.memory:
            occ = np.flatnonzero((grid.L > grid.p.l_occ) & (grid.hits >= self.p.min_hits))
        else:
            occ = grid.occupied(now)
            occ = occ[grid.hits[occ] >= self.p.min_hits]
        if len(occ) < 50:
            return None, None, None
        pts = grid.surface(occ)
        tree = cKDTree(pts)
        self._tree, self._tree_pts = tree, pts
        self._tree_n = _normals(pts, tree, min(self.p.normal_k, len(pts)))
        self._built_at = frame_no
        return tree, pts, self._tree_n

    def step(self, grid, now: float, frame_no: int, pts_world: np.ndarray) -> None:
        """Estimate a correction from this frame's (already corrected) world points and fold a
        share of it into the running offset."""
        p = self.p
        if not p.every or frame_no % p.every or len(pts_world) < p.min_matches:
            return
        tree, tgt, nrm = self._targets(grid, now, frame_no)
        if tree is None:
            return
        self.stats["tried"] += 1
        rng = np.random.default_rng(frame_no)
        src = pts_world[rng.choice(len(pts_world), min(p.points, len(pts_world)), replace=False)]
        c = src.mean(0)
        R_acc, t_acc = np.eye(3), np.zeros(3)
        for _ in range(p.iters):
            cur = (src - c) @ R_acc.T + c + t_acc
            d, j = tree.query(cur, distance_upper_bound=p.max_dist_m)
            ok = np.isfinite(d)
            if ok.sum() < p.min_matches:
                return
            s, q, n = cur[ok], tgt[j[ok]], nrm[j[ok]]
            r = np.einsum("ni,ni->n", s - q, n)  # point-to-plane residual
            A = np.c_[np.cross(s - c, n), n]  # small rotation about c, then translation
            cols = {"6dof": [0, 1, 2, 3, 4, 5], "yaw": [2, 3, 4, 5], "t": [3, 4, 5]}[p.dof]
            x = np.zeros(6)  # world z is up (gravity-aligned SLAM world): yaw = rotation about z
            x[cols], *_ = np.linalg.lstsq(A[:, cols], -r, rcond=None)
            dR = Rotation.from_rotvec(x[:3]).as_matrix()
            R_acc, t_acc = dR @ R_acc, dR @ t_acc + x[3:]
        ang = np.degrees(np.linalg.norm(Rotation.from_matrix(R_acc).as_rotvec()))
        mv = np.linalg.norm(t_acc)
        if mv > 3 * p.max_step_m or ang > 3 * p.max_step_deg:  # implausible jump: distrust
            return
        scale = min(1.0, p.max_step_m / max(mv, 1e-9), p.max_step_deg / max(ang, 1e-9)) * p.gain
        rv = Rotation.from_matrix(R_acc).as_rotvec() * scale
        dR = Rotation.from_rotvec(rv).as_matrix()
        dt = t_acc * scale + c - dR @ c  # rotation about the scan's centre, as estimated
        self.R, self.t = dR @ self.R, dR @ self.t + dt
        self.stats["accepted"] += 1
        self.stats["step_mm"].append(float(np.linalg.norm(t_acc * scale)) * 1000)
