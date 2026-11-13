"""Scan mesh: a recorded walk -> a coloured 3D mesh that looks like a scan and keeps up with change.

  python -m tars_scan.scan_mesh fuse <session> <export dir> <trajectory.txt> --out <dir> [--set k=v]
  python -m tars_scan.scan_mesh mesh --out <dir> [--levels 0.5,1]   # .ply per certainty level
  python -m tars_scan.scan_mesh view --out <dir>                     # Rerun, with a time slider
  python -m tars_scan.scan_mesh preview --out <dir> [--levels 0.5]   # shaded PNG, no viewer

Certainty (Axel, 2026-11-13): certainty decays with distance and a surface needs a certain level of
it to be drawn. Each keyframe that sees a surface adds w(z) = min(1, (z_ref/z)^power) seconds-worth
of looking (scaled by the keyframe rate): 1 = one second seen from z_ref or closer. power 4 is the
inverse variance of stereo depth (sigma_z ~ z^2), the maximum-likelihood TSDF weight; 2 is the
other common choice. The TSDF mean uses the same weights, so near looks also decide where it sits.

Change (Axel, 2026-11-13): move a box and scan again: where it stood, the surface behind must
appear; where it stands now, the surface behind it must stay; a person walking through is detected,
and removed with high certainty once the place is seen empty. Judged per keyframe against the map
as it stood then (ChangeGrid, 4 cm):
- seen through: a ray that passes a solid voxel (certain, at or behind its surface) before its
  own surface was measured through it. It is not fused (a wrong match behind a wall is the usual
  cause), but it counts against the blocker: free_since += its look weight.
- removal: a voxel whose free_since (looking through it since its last hit) reaches remove_level
  is reset, here and in the mesh volume. A hit resets free_since, and a hit wins over a pass in the
  same keyframe, so a wall that keeps being seen survives stray wrong matches through it.
- hidden is not gone: a surface behind something new gets no rays at all, so nothing changes.
- moving: a hit in a voxel that was empty with high confidence (free_total >= ever_free_level)
  and is not yet solid is a moving thing (Dynablox's ever-free cue): logged as dynamic points.

Pipeline (each stage persisted, re-run alone):
1. fuse -> volume.npz [ScanVolume v2], snapshots.npz (mesh every snapshot_s, per-vertex
   certainty, so any drawing level is a face filter), dynamic.npz, fuse.json
2. mesh -> mesh_c<level>.ply + mesh.json (per distance band: surface seen, share drawn)
3. view -> scan.rrd: the snapshots on a time slider, moving points in red, the camera path
Colour: sessions with [cameras] color_cams = [0] (from 2026-11-13) have colour cam0 frames in the
export; older walks are grey. colour_images() is the one place colour enters.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

VOLUME_VERSION = 2
# a running fuse saves its state this often (minutes) and resumes from it
CHECKPOINT_MIN = float(os.environ.get("SCAN_CHECKPOINT_MIN", 10))
FPS = 30.0  # recorded pairs per second: certainty is in seconds of looking
COLS = ("wsdf", "w", "r", "g", "b", "cw", "cert", "zsum", "nlook")  # ScanVolume v2 value columns
BANDS = (0, 0.5, 1, 1.5, 2, 2.5, 3.01)


@dataclass(frozen=True)
class ScanParams:
    # geometry (Voxblox: truncation 4 voxels; 2 cm voxels resolve the rig's ~1 cm depth at 1 m)
    voxel_m: float = 0.02
    trunc_m: float = 0.08  # TSDF truncation: > 1.5 sigma of the depth noise at 2 m (~5 cm)
    dropoff: bool = True  # weight falls linearly to 0 behind the surface (Bylow et al.)
    # depth
    every: int = 8  # keyframe every n-th exported pair (30 fps -> 3.75 Hz)
    rect_out: float = 1.0  # rectified size vs export (1.0: focal ~266 px)
    ray_every: int = 3  # rectified pixels between rays (3 px = 11 mm at 1 m < one voxel)
    max_depth_m: float = 3.0  # hard cut; the certainty weight does the rest
    edge_rel: float = 0.04  # occupancy_map.filter_depth: drop mixed near/far pixels
    lr_max_diff: float = 1.0  # occupancy_map.depth_lr
    # certainty
    z_ref_m: float = 1.0  # a look from z_ref or closer counts 1
    power: float = 4.0  # w(z) = min(1, (z_ref/z)^power)
    draw_level: float = 0.5  # seconds of close looking for a surface to be drawn (mesh-time)
    min_faces: int = 200  # loose pieces smaller than this are not drawn (mesh-time)
    # change (ChangeGrid)
    block_level: float = 0.5  # certainty (s) above which a voxel blocks sight (0 = no check)
    # rays are traced up to range - max(margin, 3 sigma); 0.12 -> 0.25 on static walk2 kept the
    # drawn area 45 -> 64.5 m2 (drift makes a re-seen wall look see-through)
    sight_margin_m: float = 0.25
    remove_level: float = 1.0  # s of close looking through a voxel since its last hit: removed
    ever_free_level: float = 2.0  # s of looking through: a hit there now is something moving
    # looking-through evidence fades with this time constant (s; 0 = never): removal needs
    # recent, consistent see-through, not stray rays summed over minutes
    remove_tau_s: float = 10.0
    # moving needs clearance: no solid surface within the 3x3x3 cells of this size around the hit
    # (Dynablox's high-confidence free space). Drift puts a re-seen wall beside its old copy, in
    # space once seen empty: walk2 (nothing moved) flagged 3.9 % of all depth points before this
    free_clear_m: float = 0.2
    # drift: scan-to-map alignment (map_align.MapAligner against this scan's solid surface);
    # every k-th keyframe nudges a running pose correction (0 = off). The occupancy map's setting
    # (walk2 2026-11-11, raw Basalt: leg drift 185 -> 106 mm): every 1, gain 0.3, 5 cm, position
    align_every: int = 0
    align_gain: float = 0.3
    align_max_dist_m: float = 0.05
    # run
    chunk: int = 6  # keyframes per volume merge (memory vs speed)
    snapshot_s: float = 30.0  # walk seconds between mesh snapshots (the time slider)

    @property
    def check_voxel_m(self) -> float:
        return 2 * self.voxel_m  # ChangeGrid voxel: exactly 2x2x2 mesh voxels


MESH_ONLY = ("draw_level", "min_faces")  # changing these needs no re-fuse
# Axel, 2026-11-13: the reference pass is about quality only (compute is free): every second frame,
# every pixel, 1.5 cm voxels (truncation still 4 voxels, Voxblox), drawn only where looked at for a
# full second up close. Everyday scans keep the defaults and are tuned for quality per compute.
PRESETS = {
    "reference": {
        "every": 2,
        "ray_every": 1,
        "voxel_m": 0.015,
        "trunc_m": 0.06,
        "draw_level": 1.0,
        "snapshot_s": 60.0,
    }
}


def parse_params(kv: dict[str, str], base: ScanParams | None = None) -> ScanParams:
    base = base or ScanParams()
    types = {f.name: f.type for f in fields(ScanParams)}
    conv = {"int": int, "float": float, "bool": lambda s: str(s).lower() in ("1", "true", "yes")}
    return ScanParams(**{**asdict(base), **{k: conv[types[k]](v) for k, v in kv.items()}})


def look_weight(z: np.ndarray, p: ScanParams) -> np.ndarray:
    """Certainty one keyframe adds for a surface seen at depth z (m), before keyframe scaling."""
    return np.minimum(1.0, (p.z_ref_m / np.maximum(z, 1e-3)) ** p.power)


def colour_images(rect: dict, left: np.ndarray, colour_file: Path) -> np.ndarray:
    """RGB per rectified cam0 pixel: from the export's colour frame (mav0/cam0/color/<stamp>.jpg,
    colour sessions), else the luma three times."""
    import cv2

    if colour_file.exists():
        bgr = cv2.imread(str(colour_file), cv2.IMREAD_COLOR)
        return cv2.remap(bgr, *rect["maps"][0], cv2.INTER_LINEAR)[..., ::-1]
    g = cv2.remap(left, *rect["maps"][0], cv2.INTER_LINEAR)
    return np.repeat(g[..., None], 3, axis=2)


def rect_valid(rect: dict, src_shape: tuple[int, int], margin: int) -> np.ndarray:
    """Rectified pixels where both cameras have image, eroded by margin px. Outside the fisheyes'
    view the rectified images are black, and the band-pass + matcher turn that black edge into
    near-looking false depth (streaks along the top, bars along the bottom: 2026-11-13, walk2)."""
    import cv2

    ones = np.full(src_shape, 255, np.uint8)
    m = np.ones(rect["maps"][0][0].shape[:2], bool)
    for k in (0, 1):
        m &= cv2.remap(ones, *rect["maps"][k], cv2.INTER_NEAREST, borderValue=0) == 255
    k = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    return cv2.erode(m.astype(np.uint8), k, borderValue=0).astype(bool)


class ChangeGrid:
    """Dense 4 cm grid over the walk: what was solid, seen through, or empty, as of now."""

    def __init__(self, lo: np.ndarray, dims_fine: np.ndarray, p: ScanParams, f_b: float):
        self.lo, self.v, self.p, self.f_b = lo, p.check_voxel_m, p, f_b
        self.dims = (dims_fine + 1) // 2
        n = int(np.prod(self.dims))
        z = lambda: np.zeros(n, np.float32)  # noqa: E731
        self.cert, self.wsdf, self.w, self.free_since, self.free_total = z(), z(), z(), z(), z()
        self.t_pass = z()  # walk time of the last pass (for the fade)
        # solid surface per clearance cell (free_clear_m), for the moving test
        self.cdims = np.ceil(self.dims * self.v / p.free_clear_m).astype(np.int64) + 2
        self.solid_cell = np.zeros(int(np.prod(self.cdims)), bool)

    def idx(self, X: np.ndarray):
        i = np.floor((X - self.lo) / self.v).astype(np.int64)
        ok = ((i >= 0) & (i < self.dims)).all(-1)
        i = np.where(ok[..., None], i, 0)
        return i[..., 0] + self.dims[0] * (i[..., 1] + self.dims[1] * i[..., 2]), ok

    def trace(self, o: np.ndarray, dw: np.ndarray, end: np.ndarray):
        """Grid indices along each ray from 0.1 m to its own end (ragged, half-voxel steps):
        (voxel per sample, ray per sample, inside the grid)."""
        h = self.v * 0.5
        n = np.maximum(np.ceil((end - 0.1) / h), 0).astype(np.int64)
        ray = np.repeat(np.arange(len(end)), n)
        k = np.arange(len(ray)) - np.repeat(np.cumsum(n) - n, n)
        # integer voxel coordinates in float32: 3x less memory traffic than float64 (profiled
        # 2026-11-13: idx() was the top cost of the reference preset)
        f = ((o - self.lo) / self.v).astype(np.float32)
        step = (dw * (h / self.v)).astype(np.float32)
        c = f + step[ray] * (k.astype(np.float32) + np.float32(0.1 / h))[:, None]
        ci = np.floor(c).astype(np.int64)
        ok = ((ci >= 0) & (ci < self.dims)).all(1)
        ci[~ok] = 0
        return ci[:, 0] + self.dims[0] * (ci[:, 1] + self.dims[1] * ci[:, 2]), ray, ok

    def solid(self, i: np.ndarray) -> np.ndarray:
        """Certain enough, and at or behind its surface (a ray grazing a floor stays in front)."""
        tsdf = self.wsdf[i] / np.maximum(self.w[i], 1e-9)
        return (self.cert[i] >= self.p.block_level) & (tsdf < 0.25)

    def step(self, o, dw, r, z, w, t: float):
        """One keyframe's rays. Returns (blocked rays, voxels to remove)."""
        p = self.p
        sigma = z**2 / self.f_b * 0.5  # 0.5 px disparity noise, as occupancy_map
        end = r - np.maximum(p.sight_margin_m, 3 * sigma)
        vi, ray, ok = self.trace(o, dw, end)
        vi, ray = vi[ok], ray[ok]
        blocked = np.zeros(len(r), bool)
        if p.block_level > 0:
            blocked[ray[self.solid(vi)]] = True
        # looking through: per voxel the strongest look this keyframe (once per keyframe)
        kw = p.every / FPS
        u, inv = np.unique(vi, return_inverse=True)
        best = np.zeros(len(u), np.float32)
        np.maximum.at(best, inv, w[ray].astype(np.float32))
        thru = best * kw
        if p.remove_tau_s > 0:
            self.free_since[u] *= np.exp(-(t - self.t_pass[u]) / p.remove_tau_s)
        self.t_pass[u] = t
        self.free_since[u] += thru
        self.free_total[u] += thru
        return blocked, u

    def hits(self, X_hits, w_hits, X_samples, sdf, ww) -> np.ndarray:
        """Fuse accepted rays; returns per hit: was it in space known to be empty (moving)."""
        i, ok = self.idx(X_samples)
        u, inv = np.unique(i[ok], return_inverse=True)
        self.wsdf[u] += np.bincount(inv, (ww * sdf)[ok], minlength=len(u))
        self.w[u] += np.bincount(inv, ww[ok], minlength=len(u))
        i, ok = self.idx(X_hits)
        moving = (
            ok
            & (self.free_total[i] >= self.p.ever_free_level)
            & (self.cert[i] < self.p.block_level)
            & self.clear(X_hits)
        )
        u, j = np.unique(i[ok], return_index=True)
        self.cert[u] += w_hits[ok][j] * self.p.every / FPS
        self.free_since[u] = 0  # a hit wins over a pass in the same keyframe
        solid = u[self.cert[u] >= self.p.block_level]
        if len(solid):
            self.solid_cell[self._cell(self._centres(solid))] = True
        return moving

    def _centres(self, coarse: np.ndarray) -> np.ndarray:
        d = self.dims
        ijk = np.stack([coarse % d[0], (coarse // d[0]) % d[1], coarse // (d[0] * d[1])], 1)
        return self.lo + (ijk + 0.5) * self.v

    def _cell(self, X: np.ndarray, off=(0, 0, 0)) -> np.ndarray:
        c = np.floor((X - self.lo) / self.p.free_clear_m).astype(np.int64) + 1 + np.array(off)
        c = np.clip(c, 0, self.cdims - 1)
        return c[:, 0] + self.cdims[0] * (c[:, 1] + self.cdims[1] * c[:, 2])

    def clear(self, X: np.ndarray) -> np.ndarray:
        """No solid surface in the 27 clearance cells around each point."""
        near = np.zeros(len(X), bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    near |= self.solid_cell[self._cell(X, (dx, dy, dz))]
        return ~near

    def removals(self, passed: np.ndarray) -> np.ndarray:
        """Voxels looked through for remove_level since their last hit, holding a surface (at or
        behind it): reset. Returns them."""
        if self.p.remove_level <= 0:
            return np.zeros(0, np.int64)
        tsdf = self.wsdf[passed] / np.maximum(self.w[passed], 1e-9)
        gone = passed[
            (self.free_since[passed] >= self.p.remove_level) & (self.w[passed] > 0) & (tsdf < 0.25)
        ]
        for a in (self.cert, self.wsdf, self.w, self.free_since):
            a[gone] = 0
        return gone

    def fine_keys(self, coarse: np.ndarray, dims_fine: np.ndarray) -> np.ndarray:
        """The 8 mesh-volume keys inside each coarse voxel."""
        cx = coarse % self.dims[0]
        cy = (coarse // self.dims[0]) % self.dims[1]
        cz = coarse // (self.dims[0] * self.dims[1])
        out = []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    x, y, z = 2 * cx + dx, 2 * cy + dy, 2 * cz + dz
                    ok = (x < dims_fine[0]) & (y < dims_fine[1]) & (z < dims_fine[2])
                    out.append((x + dims_fine[0] * (y + dims_fine[1] * z))[ok])
        return np.concatenate(out) if out else np.zeros(0, np.int64)


class ScanAligner:
    """map_align.MapAligner with this scan's solid mesh voxels (certainty >= block_level, voxel
    centres) as the targets instead of the occupancy grid's."""

    def __init__(self, vol: Volume, p: ScanParams):
        from .map_align import AlignParams, MapAligner

        self.vol, self.p = vol, p
        self.a = MapAligner(
            AlignParams(
                every=p.align_every, gain=p.align_gain, max_dist_m=p.align_max_dist_m, dof="t"
            )
        )
        self.a._targets = self._targets  # the aligner asks this object for its map points

    def _targets(self, _grid, _now, frame_no):
        a = self.a
        if a._tree is not None and frame_no - a._built_at < 10:
            return a._tree, a._tree_pts, a._tree_n
        from scipy.spatial import cKDTree

        from .map_align import _normals

        v = self.vol
        sel = v.keys[v.vals[:, 6] >= max(self.p.block_level, 1e-3)]
        if len(sel) < 50:
            return None, None, None
        d = v.dims
        ijk = np.stack([sel % d[0], (sel // d[0]) % d[1], sel // (d[0] * d[1])], 1)
        pts = v.lo + (ijk + 0.5) * v.voxel
        tree = cKDTree(pts)
        a._tree, a._tree_pts, a._tree_n = tree, pts, _normals(pts, tree, min(8, len(pts)))
        a._built_at = frame_no
        return a._tree, a._tree_pts, a._tree_n

    def correct(self, M: np.ndarray) -> np.ndarray:
        return self.a.correct(M)

    def step(self, frame_no: int, now: float, pts_world: np.ndarray) -> None:
        self.a.step(None, now, frame_no, pts_world)


class Volume:
    """Mesh voxels (2 cm): sorted keys -> value columns (float32), updated in place per chunk;
    removed voxels are zeroed."""

    def __init__(self, lo: np.ndarray, dims: np.ndarray, voxel: float):
        self.lo, self.dims, self.voxel = lo, dims, voxel
        self.keys = np.zeros(0, np.int64)
        self.vals = np.zeros((0, len(COLS)), np.float32)

    def key(self, X: np.ndarray):
        i = np.floor((X - self.lo) / self.voxel).astype(np.int64)
        ok = ((i >= 0) & (i < self.dims)).all(1)
        i = i[ok]
        return i[:, 0] + self.dims[0] * (i[:, 1] + self.dims[1] * i[:, 2]), ok

    def add(self, keys: np.ndarray, vals: np.ndarray) -> None:
        u, inv = np.unique(keys, return_inverse=True)
        v = np.stack(
            [np.bincount(inv, vals[:, c], minlength=len(u)) for c in range(vals.shape[1])], 1
        ).astype(np.float32)
        pos = np.searchsorted(self.keys, u)
        have = pos < len(self.keys)
        have[have] = self.keys[pos[have]] == u[have]
        self.vals[pos[have]] += v[have]
        if (~have).any():
            self.keys = np.insert(self.keys, pos[~have], u[~have])
            self.vals = np.insert(self.vals, pos[~have], v[~have], axis=0)

    def reset(self, keys: np.ndarray) -> int:
        if not len(keys) or not len(self.keys):
            return 0
        pos = np.clip(np.searchsorted(self.keys, keys), 0, len(self.keys) - 1)
        hit = pos[self.keys[pos] == keys]
        self.vals[hit] = 0
        return int(len(hit))

    def ijk(self) -> np.ndarray:
        k, d = self.keys, self.dims
        return np.stack([k % d[0], (k // d[0]) % d[1], k // (d[0] * d[1])], 1)


def fuse(session: Path, export_dir: Path, traj: str, p: ScanParams, out: Path) -> dict:
    import cv2

    from .occupancy_map import Params, depth_lr, filter_depth, rect_and_matcher
    from .voxel_map import BAND, poses_world_cam0

    exp = json.loads((export_dir / "export.json").read_text())
    files = sorted((export_dir / "mav0" / "cam0" / "data").glob("*.png"))[:: p.every]
    stamps = np.array([int(f.stem) for f in files])
    poses = poses_world_cam0(session, traj, stamps)
    files = [f for f in files if int(f.stem) in poses]
    t_first = int(files[0].stem)
    first = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    occ = Params(rect_out=p.rect_out, max_depth_m=p.max_depth_m, edge_rel=p.edge_rel)
    rect, sgbm = rect_and_matcher(session, exp.get("scale", 0.5), first.shape[::-1], occ)
    band = tuple(b * p.rect_out / 0.5 for b in BAND)  # BAND is in small-rectification pixels
    K = rect["P1"][:, :3]
    cen = np.array([M[:3, 3] for M in poses.values()])
    pad = p.max_depth_m + p.trunc_m + 0.1
    lo = cen.min(0) - pad
    dims = np.ceil((cen.max(0) + pad - lo) / p.voxel_m).astype(np.int64)
    dims += dims % 2  # whole ChangeGrid voxels
    vol = Volume(lo, dims, p.voxel_m)
    grid = ChangeGrid(lo, dims, p, rect["f"] * abs(rect["baseline"]))
    offs = np.arange(-p.trunc_m, p.trunc_m + 1e-9, p.voxel_m)  # signed distance of each sample
    valid = rect_valid(rect, first.shape, margin=int(2 * band[0]) + 5)
    kw = p.every / FPS

    import threading

    from .stereo_depth import matcher

    local = threading.local()

    def one(f):
        # one SGBM per thread: a shared cv2 matcher is not thread-safe (2026-11-13: two identical
        # runs differed by 1-2 % of depth points until each thread got its own)
        if not hasattr(local, "sgbm"):
            local.sgbm = matcher(sgbm.getNumDisparities(), sgbm.getBlockSize())
        left = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(export_dir / "mav0" / "cam1" / "data" / f.name), 0)
        z = filter_depth(depth_lr(rect, local.sgbm, left, right, band, p.lr_max_diff), occ)
        z[~valid] = np.nan
        rgb = colour_images(rect, left, export_dir / "mav0" / "cam0" / "color" / f"{f.stem}.jpg")
        return z, rgb

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(4)  # OpenCV releases the GIL: stereo on 4 cores
    seen, through = np.zeros(len(BANDS) - 1), np.zeros(len(BANDS) - 1)
    n_obs, n_removed_fine = 0, 0
    snaps: list[dict] = []
    dyn_t, dyn_pts, removed_t = [], [], []
    next_snap = p.snapshot_s
    aligner = ScanAligner(vol, p) if p.align_every else None
    path_used, k_frame = [], 0
    t0 = time.time()
    # observable and resumable (Axel, 2026-11-13): progress.json after every chunk, the whole state
    # in checkpoint.pkl every CHECKPOINT_MIN; a re-run with the same settings continues from it
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pkl"
    start, last_ckpt = 0, time.time()
    grid_arrays = ("cert", "wsdf", "w", "free_since", "free_total", "t_pass", "solid_cell")
    if ckpt.exists():
        import pickle

        st = pickle.loads(ckpt.read_bytes())
        if st["params"] == asdict(p) and st["files"] == len(files):
            vol.keys, vol.vals = st["vol"]
            for k_, v_ in st["grid"].items():
                setattr(grid, k_, v_)
            snaps, dyn_t, dyn_pts, removed_t = (
                st["snaps"],
                st["dyn_t"],
                st["dyn_pts"],
                st["removed_t"],
            )
            seen, through, n_obs, n_removed_fine = st["seen"], st["through"], st["n_obs"], st["nrm"]
            next_snap, k_frame, path_used = st["next_snap"], st["k_frame"], st["path_used"]
            if aligner is not None and st["align"] is not None:
                aligner.a.R, aligner.a.t, aligner.a.stats = st["align"]
            start, t0 = st["next_c0"], time.time() - st["elapsed"]
            print(f"resumed at keyframe {start}/{len(files)}", flush=True)
        else:
            print("checkpoint.pkl is from other settings: starting over", flush=True)
    for c0 in range(start, len(files), p.chunk):
        ks, vs, gone = [], [], []
        batch = files[c0 : c0 + p.chunk]
        for f, (z, rgb) in zip(batch, pool.map(one, batch), strict=True):
            M = poses[int(f.stem)]
            t_walk = (int(f.stem) - t_first) / 1e9
            zs = z[:: p.ray_every, :: p.ray_every]
            cs = rgb[:: p.ray_every, :: p.ray_every]
            v, u = np.nonzero(np.isfinite(zs))
            col = cs[v, u].astype(np.float32)
            zz = zs[v, u]
            u, v = u * p.ray_every, v * p.ray_every
            d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(zz)], 1)
            norm = np.linalg.norm(d, axis=1)
            r = zz * norm
            if aligner is not None:  # this frame's depth, placed with the correction so far
                Mc = aligner.correct(M)
                dc = (d / norm[:, None]) @ (Mc[:3, :3] @ rect["R1"].T).T
                aligner.step(k_frame, t_walk, Mc[:3, 3] + dc * r[:, None])
                M = aligner.correct(M)
            k_frame += 1
            path_used.append(M[:3, 3].copy())
            o = M[:3, 3]
            R_wr = M[:3, :3] @ rect["R1"].T  # world <- rectified cam0
            dw = (d / norm[:, None]) @ R_wr.T
            w = look_weight(zz, p)
            seen += np.histogram(zz, BANDS)[0]
            blocked, passed = grid.step(o, dw, r, zz, w, t_walk)
            through += np.histogram(zz[blocked], BANDS)[0]
            keep = ~blocked
            zz, dw, r, w, col = zz[keep], dw[keep], r[keep], w[keep], col[keep]
            # TSDF samples along each ray, sdf > 0 in front of the surface
            X = o + dw[:, None, :] * (r[:, None] - offs[None, :])[..., None]
            k, ok = vol.key(X.reshape(-1, 3))
            sdf = np.broadcast_to(offs / p.trunc_m, (len(r), len(offs))).reshape(-1)[ok]
            ww = np.repeat(w, len(offs))[ok]
            if p.dropoff:
                ww = ww * np.where(sdf < 0, np.clip(1 + sdf, 0, 1), 1)
            near = np.exp(-((sdf * p.trunc_m / p.voxel_m) ** 2))  # colour from the surface itself
            cw = np.repeat(col, len(offs), axis=0)[ok] * (ww * near)[:, None]
            vals = np.zeros((len(k), len(COLS)), np.float32)
            vals[:, 0], vals[:, 1], vals[:, 2:5], vals[:, 5] = ww * sdf, ww, cw, ww * near
            # certainty and viewing distance: once per keyframe per surface voxel
            Xh = o + dw * r[:, None]
            hk, hok = vol.key(Xh)
            uk, first_i = np.unique(hk, return_index=True)
            zh = zz[hok][first_i]
            hv = np.zeros((len(uk), len(COLS)), np.float32)
            hv[:, 6], hv[:, 7], hv[:, 8] = look_weight(zh, p) * kw, zh, 1
            moving = grid.hits(Xh, w, X.reshape(-1, 3)[ok], sdf, ww)
            gone.append(grid.removals(passed))
            if moving.any():
                dyn_t.append(np.full(int(moving.sum()), t_walk, np.float32))
                dyn_pts.append(Xh[moving].astype(np.float32))
            ks += [k, uk]
            vs += [vals, hv]
            n_obs += len(zz)
        vol.add(np.concatenate(ks), np.concatenate(vs))
        g = np.unique(np.concatenate(gone))
        if len(g):  # after the chunk's hits: a voxel removed mid-chunk loses its later hits too
            n_removed_fine += vol.reset(grid.fine_keys(g, dims))
            removed_t.append((t_walk, int(len(g))))
        done = min(c0 + p.chunk, len(files))
        if t_walk >= next_snap or done == len(files):
            snaps.append({"t": round(t_walk, 2), **extract(vol, p)})
            next_snap += p.snapshot_s
        el = time.time() - t0
        eta = el / max(done - start, 1) * (len(files) - done) if done > start else None
        print(
            f"fuse {done}/{len(files)}  t {t_walk:5.1f} s  {len(vol.keys) / 1e6:.1f} M voxels  "
            f"removed {n_removed_fine}  {el:.0f} s  eta {eta or 0:.0f} s",
            flush=True,
        )
        write_progress(out, "fuse", done, len(files), el, eta, voxels=int(len(vol.keys)))
        if time.time() - last_ckpt > CHECKPOINT_MIN * 60 and done < len(files):
            import pickle

            st = {
                "params": asdict(p),
                "files": len(files),
                "next_c0": c0 + p.chunk,
                "elapsed": el,
                "vol": (vol.keys, vol.vals),
                "grid": {a: getattr(grid, a) for a in grid_arrays},
                "snaps": snaps,
                "dyn_t": dyn_t,
                "dyn_pts": dyn_pts,
                "removed_t": removed_t,
                "seen": seen,
                "through": through,
                "n_obs": n_obs,
                "nrm": n_removed_fine,
                "next_snap": next_snap,
                "k_frame": k_frame,
                "path_used": path_used,
                "align": None if aligner is None else (aligner.a.R, aligner.a.t, aligner.a.stats),
            }
            tmp = out / "checkpoint.tmp"
            tmp.write_bytes(pickle.dumps(st, protocol=5))
            tmp.replace(ckpt)  # atomic: a kill mid-write leaves the previous checkpoint
            last_ckpt = time.time()
            print(f"checkpoint at keyframe {done}", flush=True)
    np.savez(
        out / "volume.npz",
        version=VOLUME_VERSION,
        keys=vol.keys,
        vals=vol.vals,
        lo=lo,
        dims=dims,
        voxel=p.voxel_m,
        path=np.array(path_used),  # the poses fused (after alignment, when on)
        path_t=(np.array([int(f.stem) for f in files]) - t_first) / 1e9,
        params=json.dumps(asdict(p)),
    )
    save_snapshots(out / "snapshots.npz", snaps)
    np.savez(
        out / "dynamic.npz",
        t=np.concatenate(dyn_t) if dyn_t else np.zeros(0, np.float32),
        xyz=np.concatenate(dyn_pts) if dyn_pts else np.zeros((0, 3), np.float32),
    )

    def pct(a):
        return {  # noqa: E731
            f"{x}-{min(y, 3):g} m": round(100 * float(t / max(n, 1)), 1)
            for x, y, t, n in zip(BANDS[:-1], BANDS[1:], a, seen, strict=True)
        }

    info = {
        "keyframes": len(files),
        "voxels": int(len(vol.keys)),
        "depth_points": int(n_obs),
        "seen_through_pct_by_depth": pct(through),
        "removed_mesh_voxels": n_removed_fine,
        "moving_points": int(sum(len(t) for t in dyn_t)),
        "align": None
        if aligner is None
        else {
            "tried": aligner.a.stats["tried"],
            "accepted": aligner.a.stats["accepted"],
            "final_offset_mm": round(float(np.linalg.norm(aligner.a.t)) * 1000, 1),
        },
        "snapshots": len(snaps),
        "seconds": round(time.time() - t0, 1),
        # compute per keyframe and per second of walk (Mac, 4 stereo threads): the cost axis of
        # scan_autotune; includes stereo depth and the snapshot meshes
        "ms_per_keyframe": round(1000 * (time.time() - t0) / len(files), 1),
        "walk_s": round(t_walk, 1),
        "compute_per_walk_s": round((time.time() - t0) / max(t_walk, 1e-3), 3),
        "session": str(session),
        "trajectory": traj,
        "params": asdict(p),
    }
    (out / "fuse.json").write_text(json.dumps(info, indent=2))
    write_progress(out, "done", len(files), len(files), time.time() - t0, 0)
    ckpt.unlink(missing_ok=True)
    return info


def write_progress(out: Path, stage: str, done: int, total: int, elapsed: float, eta, **extra):
    """<out>/progress.json: what a watcher (or Axel) reads to see a run is alive and how far."""
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
    d = {
        "stage": stage,
        "done": done,
        "total": total,
        "elapsed_s": round(elapsed),
        "eta_s": None if eta is None else round(eta),
        "peak_rss_mb": round(rss),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    tmp = out / "progress.tmp"
    tmp.write_text(json.dumps(d))
    tmp.replace(out / "progress.json")


def extract(vol: Volume, p: ScanParams, base_level: float = 0.05) -> dict:
    """Marching cubes over the whole volume; per-vertex certainty, so a drawing level is a face
    filter later (no re-fuse). Faces whose certainty is below base_level are dropped already."""
    from scipy.ndimage import maximum_filter
    from skimage.measure import marching_cubes

    empty = {
        "v": np.zeros((0, 3), np.float32),
        "f": np.zeros((0, 3), np.int32),
        "rgb": np.zeros((0, 3), np.uint8),
        "c": np.zeros(0, np.float16),
        "z": np.zeros(0, np.float16),
    }
    live = vol.vals[:, 1] > 1e-6
    if live.sum() < 8:
        return empty
    keys, vals = vol.keys[live], vol.vals[live]
    d = vol.dims
    ijk = np.stack([keys % d[0], (keys // d[0]) % d[1], keys // (d[0] * d[1])], 1)
    i0 = ijk.min(0)
    shape = tuple(ijk.max(0) - i0 + 2)
    idx = tuple((ijk - i0).T)
    tsdf = np.ones(shape, np.float32)  # unseen = free; masked out below
    tsdf[idx] = vals[:, 0] / vals[:, 1]
    seen = np.zeros(shape, bool)
    seen[idx] = True
    cert = np.zeros(shape, np.float32)
    cert[idx] = vals[:, 6]
    cert_near = maximum_filter(cert, size=3)  # the surface voxel next to a mesh vertex
    zview = np.zeros(
        shape, np.float32
    )  # mean distance the surface was seen from (scan_score bands)
    zview[idx] = np.where(vals[:, 8] > 0, vals[:, 7] / np.maximum(vals[:, 8], 1), 0)
    zview = maximum_filter(zview, size=3)
    # a cube is meshed only if all 8 corners were seen: no false surface where seen meets unseen
    m = seen.copy()
    for ax in range(3):
        a = tuple(slice(0, -1) if k == ax else slice(None) for k in range(3))
        b = tuple(slice(1, None) if k == ax else slice(None) for k in range(3))
        m[a] &= seen[b]
    try:
        v, f, _, _ = marching_cubes(tsdf, level=0.0, mask=m, allow_degenerate=False)
    except (ValueError, RuntimeError):
        return empty
    vi = np.clip(np.round(v).astype(np.int64), 0, np.array(shape) - 1)
    vc = cert_near[tuple(vi.T)]
    vz = zview[tuple(vi.T)]
    f = f[(vc[f] >= base_level).all(1)]
    key = (vi[:, 0] + i0[0]) + d[0] * ((vi[:, 1] + i0[1]) + d[1] * (vi[:, 2] + i0[2]))
    pos = np.clip(np.searchsorted(keys, key), 0, len(keys) - 1)
    has = (keys[pos] == key) & (vals[pos, 5] > 1e-9)
    rgb = np.where(has[:, None], vals[pos, 2:5] / np.maximum(vals[pos, 5:6], 1e-9), 128)
    used = np.unique(f)
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    xyz = vol.lo + (v[used] + i0) * vol.voxel + vol.voxel / 2
    return {
        "v": xyz.astype(np.float32),
        "f": remap[f].astype(np.int32),
        "rgb": rgb[used].clip(0, 255).astype(np.uint8),
        "c": vc[used].astype(np.float16),
        "z": vz[used].astype(np.float16),
    }


def save_snapshots(path: Path, snaps: list[dict]) -> None:
    arrs = {"t": np.array([s["t"] for s in snaps], np.float32)}
    for i, s in enumerate(snaps):
        for k in ("v", "f", "rgb", "c", "z"):
            arrs[f"{k}{i}"] = s[k]
    np.savez(path, **arrs)


def load_snapshots(path: Path) -> list[dict]:
    d = np.load(path)
    keys = ("v", "f", "rgb", "c", "z")
    return [
        {"t": float(t), **{k: d[f"{k}{i}"] for k in keys if f"{k}{i}" in d}}
        for i, t in enumerate(d["t"])
    ]


def drawn(s: dict, level: float, min_faces: int):
    """The trimesh of a snapshot at a drawing level, without loose pieces below min_faces."""
    import trimesh

    keep = (s["c"][s["f"]].astype(np.float32) >= level).all(1)
    m = trimesh.Trimesh(s["v"], s["f"][keep], vertex_colors=s["rgb"], process=False)
    if "z" in s:
        m.vertex_attributes["zview"] = s["z"].astype(np.float32)
    if len(m.faces):
        lab = trimesh.graph.connected_component_labels(m.face_adjacency, len(m.faces))
        m.update_faces(np.bincount(lab)[lab] >= min_faces)
        m.remove_unreferenced_vertices()
    return m


def load_volume(out: Path):
    d = np.load(out / "volume.npz")
    assert int(d["version"]) == VOLUME_VERSION, "old volume.npz: re-run fuse"
    vol = Volume(d["lo"], d["dims"], float(d["voxel"]))
    vol.keys, vol.vals = d["keys"], d["vals"]
    return vol, d, json.loads(str(d["params"]))


def mesh(out: Path, levels: list[float]) -> dict:
    vol, _, prm = load_volume(out)
    final = load_snapshots(out / "snapshots.npz")[-1]
    report = {"levels": {}}
    for lev in levels:
        m = drawn(final, lev, prm["min_faces"])
        m.export(out / f"mesh_c{lev:g}.ply")
        report["levels"][f"{lev:g}"] = {"faces": int(len(m.faces)), "area_m2": round(m.area, 2)}
    report["by_distance"] = distance_table(vol, levels)
    (out / "mesh.json").write_text(json.dumps(report, indent=2))
    return report


def distance_table(vol: Volume, levels: list[float]) -> list[dict]:
    """Surface voxels by mean viewing distance: how many, and the share that reaches each level."""
    n = vol.vals[:, 8]
    s = n > 0
    zm = vol.vals[s, 7] / n[s]
    c = vol.vals[s, 6]
    rows = []
    for a, b in zip(BANDS[:-1], BANDS[1:], strict=True):
        sel = (zm >= a) & (zm < b)
        row = {"band_m": f"{a}-{min(b, 3):g}", "voxels": int(sel.sum())}
        for lev in levels:
            row[f">={lev:g}"] = round(100 * float((c[sel] >= lev).mean()), 1) if sel.any() else None
        rows.append(row)
    return rows


def log_scan(
    rr, root: str, out: Path, level: float | None = None, min_faces: int | None = None
) -> None:
    """Log a fused scan under `root` on the 'walk' timeline: mesh snapshots, moving points (red,
    each shown for 2 s), camera path so far."""
    _, d, prm = load_volume(out)
    level = prm["draw_level"] if level is None else level
    min_faces = prm["min_faces"] if min_faces is None else min_faces
    rr.log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for s in load_snapshots(out / "snapshots.npz"):
        m = drawn(s, level, min_faces)
        rr.set_time("walk", duration=s["t"])
        rr.log(
            f"{root}/mesh",
            rr.Mesh3D(
                vertex_positions=m.vertices,
                triangle_indices=m.faces,
                vertex_normals=m.vertex_normals,
                vertex_colors=m.visual.vertex_colors[:, :3],
            ),
        )
    path, pt = d["path"], d["path_t"]
    for t in np.arange(0, pt[-1] + 1, 1.0):
        rr.set_time("walk", duration=float(t))
        rr.log(f"{root}/camera_path", rr.LineStrips3D([path[pt <= t]], colors=[(255, 200, 0)]))
    dyn = np.load(out / "dynamic.npz")
    for t in np.arange(0, pt[-1] + 1, 0.5):
        sel = (dyn["t"] > t - 2) & (dyn["t"] <= t)
        rr.set_time("walk", duration=float(t))
        rr.log(f"{root}/moving", rr.Points3D(dyn["xyz"][sel], colors=[(255, 40, 40)], radii=0.01))


def view(out: Path, level: float | None, spawn: bool) -> Path:
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init("tars_scan_mesh", recording_id=out.name)
    rrd = out / "scan.rrd"
    rr.save(str(rrd), default_blueprint=rrb.Blueprint(rrb.Spatial3DView(origin="/scan")))
    log_scan(rr, "/scan", out, level)
    if spawn:
        import subprocess
        import sys

        subprocess.Popen([str(Path(sys.executable).with_name("rerun")), str(rrd)])  # noqa: S603
    return rrd


VOX_PREVIEW = 0.02


def preview(out: Path, level: float, size=(1200, 900)) -> Path:
    """A shaded PNG from above at 50 deg (no GPU), to check a mesh without a viewer."""
    import cv2

    _, _, prm = load_volume(out)
    m = drawn(load_snapshots(out / "snapshots.npz")[-1], level, prm["min_faces"])
    c = m.vertices.mean(0)
    el, az = np.radians(50), np.radians(30)
    fwd = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, -up, fwd])
    eye = c - fwd * np.ptp(m.vertices, 0).max() * 1.1
    P = (m.vertices - eye) @ R.T
    fpx = size[0] * 0.9
    uv = P[:, :2] / P[:, 2:] * fpx + np.array(size) / 2
    shade = np.clip(np.abs(m.vertex_normals @ -fwd) * 0.7 + 0.3, 0, 1)
    col = (m.visual.vertex_colors[:, :3] * shade[:, None]).astype(np.uint8)[:, ::-1]
    img = np.full((size[1], size[0], 3), 30, np.uint8)
    ui = np.round(uv).astype(int)
    z = P[:, 2]
    rad = np.clip(np.ceil(0.6 * VOX_PREVIEW * fpx / np.maximum(z, 1e-3)), 1, 6).astype(int)
    xs, ys, zs, cs = [], [], [], []
    for r in range(1, 7):
        sel = np.nonzero((z > 0.05) & (rad == r))[0]
        for du in range(-(r // 2), r - r // 2):
            for dv in range(-(r // 2), r - r // 2):
                xs.append(ui[sel, 0] + du)
                ys.append(ui[sel, 1] + dv)
                zs.append(z[sel])
                cs.append(sel)
    x, y, zz, ci = (np.concatenate(t) for t in (xs, ys, zs, cs))
    ok = (x >= 0) & (x < size[0]) & (y >= 0) & (y < size[1])
    o = np.argsort(-zz[ok])  # far first: nearer splats overwrite
    img[y[ok][o], x[ok][o]] = col[ci[ok][o]]
    png = out / f"preview_c{level:g}.png"
    cv2.imwrite(str(png), img)
    return png


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["fuse", "mesh", "view", "preview", "all"])
    ap.add_argument("session", nargs="?")
    ap.add_argument("export", nargs="?")
    ap.add_argument("trajectory", nargs="?")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="0.5", help="certainty levels to draw")
    ap.add_argument("--set", action="append", default=[], help="ScanParams override, e.g. power=2")
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--spawn", action="store_true", help="open the Rerun viewer")
    a = ap.parse_args(argv)
    out = Path(a.out)
    levels = [float(x) for x in a.levels.split(",")]
    if a.stage in ("fuse", "all"):
        kv = {k: str(v) for k, v in PRESETS.get(a.preset, {}).items()}
        p = parse_params({**kv, **dict(x.split("=", 1) for x in a.set)})
        info = fuse(Path(a.session), Path(a.export), a.trajectory, p, out)
        print(json.dumps({k: v for k, v in info.items() if k != "params"}))
    if a.stage in ("mesh", "all"):
        print(json.dumps(mesh(out, levels), indent=1))
    if a.stage in ("view", "all"):
        print(view(out, levels[0], a.spawn))
    if a.stage == "preview":
        for lev in levels:
            print(preview(out, lev))


if __name__ == "__main__":
    main()
