"""Temporal occupancy map: stereo depth fused with free-space carving and confidence-scaled decay.

  python -m tars_scan.occupancy_map <session> <export dir> <trajectory.txt> [--step 6] [--test]

What it is for (Axel, 2026-11-10): clean walls without junk in the air; an obstacle counts only
once it has been there for several frames (a person walking through does not stay); a door that
opens stops being a wall; the less certain something is, the faster it fades; and when a place is
seen again, what is seen now overrides what was remembered.

Each fused frame:
1. Depth from stereo (small rectification + band-pass) with an own left-right check (depth_lr):
   one-sided matches are the main source of floating junk. Pixels on depth edges are dropped.
2. One ray per `ray_every`-th pixel from camera 0. The end voxel gets a hit (and its running mean
   hit point: walls at sub-voxel position). Every `carve_stride`-th ray carves: voxels it crosses
   before the measured surface, less a margin from the depth's uncertainty, get a miss. A hit wins
   over a miss in the same frame.
3. Log-odds, clamped to [l_min, l_max]. A miss on a voxel the map holds occupied is evidence the
   scene changed (an opened door, a person who left) and weighs l_miss_occ instead of l_miss.
4. Between updates every voxel decays toward unknown (0) with time constant
   tau = tau_min_s * (1 + hits / hits_ref), capped at tau_max_s: the less often something was
   seen, the faster it fades.
Occupied: decayed log-odds above l_occ, with at least min_neighbors occupied neighbours.
Defaults are what the 2026-11-10/11 night sweeps on two walks chose (runs/night/night_log.md).

--test runs the self-checks (no ground truth needed), with tolerances that no tuned parameter
changes:
- predict: before each 5th frame is fused, its rays are cast through the map built so far:
  agree / phantom (map obstacle in front of the measured surface) / missing (map free there).
- change test: a fake person and a fake door are planted in the real rays, only where the camera
  saw through that space; reported while there, 3 s after the place is seen again, and at the end.
- walls (thickness, floaters), held-out agreement with the final map, and on a walk with board
  stops the board in the map at each stop.
Writes <export>/occupancy_<name>.json, .npz (surface points) and _2d.png (obstacle grid above the
detected floor).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Params:
    # defaults from the walk1 predictive sweep (2026-11-10): 5 cm beat 3 cm; the L/R check won
    # on every metric
    voxel_m: float = 0.05
    ray_every: int = 2  # rectified pixels between rays (410x308 small rectification)
    max_depth_m: float = 2.0  # beyond this stereo depth is too coarse (~5 cm per pixel)
    edge_rel: float = 0.04  # drop pixels whose 3x3 depth range exceeds this share of their depth
    disp_sigma_px: float = 0.5  # disparity noise behind the carving margin
    l_hit: float = 0.85
    l_miss: float = -0.2
    l_min: float = -2.0
    l_max: float = 3.5
    l_occ: float = 1.5
    tau_min_s: float = 60.0
    hits_ref: float = 5.0
    tau_max_s: float = 3600.0
    hit_by_sigma: bool = False  # a hit counts min(1, voxel / (2 sigma_r)): far depths are less sure
    lr_max_diff: float = 1.0  # left-right check in px (depth_lr; -1 = off)
    # a ray passing clearly through a voxel the map holds occupied is evidence the scene changed (an
    # opened door): it gets this miss instead of l_miss (0 = same as l_miss)
    l_miss_occ: float = -1.0  # walk2 2026-11-10: phantoms 0.32 -> 0.15, door cleared 67% -> 82%
    # with subvoxel: the strong miss only when the ray passes within this share of a voxel of the
    # voxel's mean hit point (0 = any pass). A grazing ray that clips a wall voxel's corner misses
    # the surface inside it and gets l_miss; a ray through an opened door crosses its surface.
    miss_occ_through: float = 0.0
    # the strong miss only once the voxel has had no hit for this long (s; 0 = always): an opened
    # door stops being hit, a partly filled voxel (table edge) keeps being hit by some rays
    # while others pass through its empty part
    miss_occ_after_s: float = 0.0
    # rectified image size relative to the export (0.5 = small, focal ~133 px: 1 px of disparity is
    # ~21 cm at 2 m; 1.0 = focal ~266 px, half the depth noise, 4x the matching work)
    rect_out: float = 0.5
    sgbm_block: int = 5  # SGBM block size in rectified px (odd)
    # scan-to-map alignment (map_align): every k-th fused frame nudges the running pose correction
    # so the frame's depth sits on the map's surfaces (0 = off); gain = share applied per step
    align_every: int = 1  # walk2 raw Basalt: leg drift 185 -> 106 mm, returns 295 -> 90 mm median
    align_gain: float = 0.3
    align_memory: bool = False  # map_align.AlignParams.memory
    align_max_dist_m: float = 0.05  # ICP capture range: 3 Basalt runs, map checks better than 0.10
    align_dof: str = "t"  # map_align.AlignParams.dof: position only (ICP yaw drifted 5-9 deg)
    # a valid depth beyond max_depth_m carves free space up to max_depth_m without a hit (+inf in
    # filter_depth): a ray through an opened door to a far wall clears the door (2026-11-11: 3 s
    # after a revisit only 12 % of the planted person and 36 % of the door were gone without it)
    carve_far: bool = False
    # every k-th ray carves free space, all rays place hits (walk2: 61 -> 33 ms, same PREDICT)
    carve_stride: int = 4
    # an occupied voxel needs at least this many occupied 26-neighbours to count (0 = off): lone
    # voxels are stereo junk or wall fragments the change miss left (walk1/2: floaters 3 -> 5 %)
    min_neighbors: int = 2  # walk2: floaters 4.3 -> 1.9 %, PREDICT unchanged (0.663 -> 0.660)
    reset_hits_when_free: bool = False  # a voxel carved free forgets its hits (and so its long tau)
    # each voxel keeps a running mean of where its hits landed: walls at sub-voxel position, not at
    # voxel centres (+-voxel/2). Mean over the first 1/mu_alpha_min hits, then an exponential mean
    # so a moved surface is followed; reset when the voxel is carved free.
    subvoxel: bool = True
    mu_alpha_min: float = 0.05


MAX_POSE_DT_NS = 2e6  # a board frame and a SLAM pose further apart in time are not paired


class Grid:
    """Dense voxel grid of log-odds, last update time and hit count (flat arrays)."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray, p: Params):
        self.p, self.lo = p, np.asarray(lo, float)
        self.shape = np.ceil((np.asarray(hi) - self.lo) / p.voxel_m).astype(int)
        n = int(np.prod(self.shape))
        self.L = np.zeros(n, np.float32)
        self.t = np.zeros(n, np.float32)
        self.hits = np.zeros(n, np.uint16)
        self.t_hit = np.zeros(n, np.float32) if p.miss_occ_after_s > 0 else None
        self.mu = np.zeros((n, 3), np.float32) if p.subvoxel else None
        self.mn = np.zeros(n, np.uint16) if p.subvoxel else None  # hits in the current mean

    def index(self, pts: np.ndarray) -> np.ndarray:
        """Flat voxel index per point, -1 outside the grid."""
        ijk = np.floor((pts - self.lo) / self.p.voxel_m).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < self.shape), axis=-1)
        sy, sz = self.shape[1], self.shape[2]
        return np.where(ok, (ijk[..., 0] * sy + ijk[..., 1]) * sz + ijk[..., 2], -1)

    def centers(self, idx: np.ndarray) -> np.ndarray:
        sy, sz = self.shape[1], self.shape[2]
        ijk = np.stack([idx // (sy * sz), (idx // sz) % sy, idx % sz], -1)
        return self.lo + (ijk + 0.5) * self.p.voxel_m

    def surface(self, idx: np.ndarray) -> np.ndarray:
        """Where the surface in each voxel is: the mean hit point (subvoxel) or the centre."""
        return self.mu[idx].astype(float) if self.mu is not None else self.centers(idx)

    def _tau(self, idx):
        p = self.p
        return np.minimum(p.tau_min_s * (1 + self.hits[idx] / p.hits_ref), p.tau_max_s)

    def logodds(self, now: float, idx: np.ndarray | None = None) -> np.ndarray:
        """Decayed log-odds at time `now` (all voxels, or idx)."""
        idx = np.arange(len(self.L)) if idx is None else idx
        return self.L[idx] * np.exp(-(now - self.t[idx]) / self._tau(idx))

    def update(self, hit, miss, w, pts, through, now: float) -> None:
        """w: per-hit weight in (0, 1] (a voxel hit by several rays keeps the largest); pts: the
        hit points (None when not tracked); through: per miss sample, may it get l_miss_occ (None =
        all may)."""
        p = self.p
        keep = hit >= 0
        hit, inv = np.unique(hit[keep], return_inverse=True)
        inv = inv.ravel()
        wh = np.zeros(len(hit))
        np.maximum.at(wh, inv, w[keep])
        if self.mu is not None and pts is not None and len(hit):
            m = np.zeros((len(hit), 3))
            np.add.at(m, inv, pts[keep])
            m /= np.bincount(inv, minlength=len(hit))[:, None]
            a = np.maximum(1.0 / (self.mn[hit].astype(float) + 1), p.mu_alpha_min)[:, None]
            self.mu[hit] = (1 - a) * self.mu[hit] + a * m
            self.mn[hit] = np.minimum(self.mn[hit].astype(np.int32) + 1, 65535)
        strong = None if through is None else np.unique(miss[through])
        miss = np.setdiff1d(np.unique(miss[miss >= 0]), hit, assume_unique=True)
        L = self.logodds(now, hit) + p.l_hit * wh
        self.L[hit], self.t[hit] = np.minimum(L, p.l_max), now
        L0 = self.logodds(now, miss)
        occ = L0 > p.l_occ
        if strong is not None:
            occ &= np.isin(miss, strong, assume_unique=True)
        if self.t_hit is not None:
            occ &= now - self.t_hit[miss] > p.miss_occ_after_s
        step = np.where(occ, p.l_miss_occ, p.l_miss) if p.l_miss_occ else p.l_miss
        L = np.maximum(L0 + step, p.l_min)
        self.L[miss], self.t[miss] = L, now
        if p.reset_hits_when_free:
            self.hits[miss[L < 0]] = 0
        if self.mu is not None:  # carved free: the mean restarts when a surface comes back
            self.mn[miss[L < 0]] = 0
        self.hits[hit] = np.minimum(self.hits[hit].astype(np.int32) + 1, 65535)
        if self.t_hit is not None:
            self.t_hit[hit] = now

    def occupied(self, now: float) -> np.ndarray:
        """Flat indices of occupied voxels at time `now`."""
        live = np.flatnonzero(self.L > self.p.l_occ)  # decay only lowers |L|
        occ = live[self.logodds(now, live) > self.p.l_occ]
        if self.p.min_neighbors <= 0 or not len(occ):
            return occ
        mask = np.zeros(len(self.L), bool)
        mask[occ] = True
        sy, sz = self.shape[1], self.shape[2]
        count = np.zeros(len(occ), np.int16)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx or dy or dz:  # grid-edge wrap-around is harmless: the edges are padding
                        nb = np.clip(occ + (dx * sy + dy) * sz + dz, 0, len(self.L) - 1)
                        count += mask[nb]
        return occ[count >= self.p.min_neighbors]


def filter_depth(z: np.ndarray, p: Params) -> np.ndarray:
    """NaN out depth-edge pixels (mixed near/far depth) and anything beyond max_depth_m (+inf
    instead when p.carve_far: free up to max range, no surface)."""
    import cv2

    ok = np.isfinite(z)
    k = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(np.where(ok, z, 0).astype(np.float32), k)
    lo = cv2.erode(np.where(ok, z, 1e6).astype(np.float32), k)
    edge = ~ok | (hi - lo > p.edge_rel * np.where(ok, z, 1))
    far = ~edge & (z > p.max_depth_m)
    out = np.where(edge | far, np.nan, z)
    return np.where(far, np.inf, out) if p.carve_far else out


def camera_rays(rect: dict, shape: tuple[int, int], every: int):
    """Unit ray directions in camera 0 [n, 3], their pixel grid, and |(x, y, 1)| per ray."""
    K = rect["P1"][:, :3]
    v, u = np.mgrid[0 : shape[0] : every, 0 : shape[1] : every]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(u.shape)], -1)
    d = d.reshape(-1, 3)
    norm = np.linalg.norm(d, axis=1)
    return (d / norm[:, None]) @ rect["R1"], (v.ravel(), u.ravel()), norm


def ray_updates(grid: Grid, o, dirs, r, sigma_r, nohit=None):
    """(hit, miss, hit weight, hit points, through) for rays from o with measured ranges r (NaN =
    no surface); through: per miss sample, did the ray cross its voxel's surface (None: unchecked).
    """
    p = grid.p
    ok = np.isfinite(r)
    dirs, r, sigma_r = dirs[ok], r[ok], sigma_r[ok]
    hit = grid.index(o + dirs * r[:, None])
    if nohit is not None:  # far rays: carve up to r, no surface there
        hit[nohit[ok]] = -1
    stop = r - np.maximum(2 * p.voxel_m, 2 * sigma_r)
    # only every carve_stride-th ray carves free space (all rays place hits): carving is ~60 % of
    # the fusion time (walk1 profile 2026-11-11), and neighbouring rays carve mostly the same voxels
    cdirs, stop = dirs[:: p.carve_stride], stop[:: p.carve_stride]
    # a third of a voxel between samples: at a full voxel a ray can cross the corner of the voxel it
    # hit earlier without a sample landing in it, and a person who left would stay in the map
    steps = (np.arange(int(3 * p.max_depth_m / p.voxel_m)) + 0.5) * (p.voxel_m / 3)
    t = np.broadcast_to(steps, (len(stop), len(steps)))
    keep = t < stop[:, None]
    pts = o + cdirs[:, None, :] * t[..., None]
    miss = grid.index(pts[keep])
    w = np.minimum(1.0, p.voxel_m / (2 * sigma_r)) if p.hit_by_sigma else np.ones(len(r))
    through = None
    if p.miss_occ_through > 0 and grid.mu is not None:
        ray = np.nonzero(keep)[0]
        safe = np.maximum(miss, 0)
        cand = np.flatnonzero((miss >= 0) & (grid.L[safe] > p.l_occ) & (grid.mn[safe] > 0))
        v = grid.mu[miss[cand]].astype(float) - o
        dist = np.linalg.norm(np.cross(v, cdirs[ray[cand]]), axis=1)
        through = np.zeros(len(miss), bool)
        through[cand] = dist < p.miss_occ_through * p.voxel_m
    return hit, miss, w, o + dirs * r[:, None], through


def inject(o, dirs, r, boxes):
    """Shorten ranges where a ray meets one of `boxes` (lo, hi corners) first: planted objects."""
    r = r.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi in boxes:
            a, b = (lo - o) / dirs, (hi - o) / dirs
            t0 = np.nanmax(np.minimum(a, b), axis=1)
            t1 = np.nanmin(np.maximum(a, b), axis=1)
            hit = (t0 <= t1) & (t0 > 0.05) & ((t0 < r) | ~np.isfinite(r))
            r[hit] = t0[hit]
    return r


@dataclass
class Frames:
    """The fused frames of a walk: rectified depth source, poses and times."""

    files: list
    stamps: np.ndarray
    poses: dict
    rect: dict
    sgbm: object
    export_dir: Path
    lr_max_diff: int = -1

    def images(self, f):
        """(left, right) of the exported pair whose cam0 file is f."""
        import cv2

        left = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(self.export_dir / "mav0" / "cam1" / "data" / f.name), 0)
        return left, right

    def depth(self, f):
        return stereo_depth_of(self.rect, self.sgbm, *self.images(f), self.lr_max_diff)


def stereo_depth_of(rect, sgbm, left, right, lr_max_diff: float) -> np.ndarray:
    """Depth along rectified cam0 (NaN = no match); left-right check when lr_max_diff >= 0."""
    from .stereo_depth import depth
    from .voxel_map import BAND

    if lr_max_diff >= 0:
        return depth_lr(rect, sgbm, left, right, BAND, lr_max_diff)
    return depth(rect, sgbm, left, right, BAND)


def depth_lr(rect, sgbm, left, right, band, max_diff: float) -> np.ndarray:
    """stereo_depth.depth with a left-right check: the match is also made from the right image, and
    a pixel keeps its depth only if both directions agree within max_diff px. One-sided matches
    (occlusion edges, repeated texture, noise) are where floating junk comes from. OpenCV's own
    disp12MaxDiff does nothing in SGBM_3WAY mode (measured 2026-11-10), hence this."""
    import cv2

    from .track_eval import bandpass

    l_r = cv2.remap(left, *rect["maps"][0], cv2.INTER_LINEAR)
    r_r = cv2.remap(right, *rect["maps"][1], cv2.INTER_LINEAR)
    if band:
        l_r, r_r = bandpass(l_r, *band), bandpass(r_r, *band)
    flip = rect["baseline"] < 0
    if flip:  # cam0 is the right-hand camera: mirror so the matcher sees left/right
        l_r, r_r = l_r[:, ::-1].copy(), r_r[:, ::-1].copy()
    dl = sgbm.compute(l_r, r_r).astype(np.float32) / 16
    dr = sgbm.compute(r_r[:, ::-1].copy(), l_r[:, ::-1].copy())[:, ::-1].astype(np.float32) / 16
    h, w = dl.shape
    xr = np.round(np.arange(w)[None, :] - dl).astype(int)
    inside = (dl > 0) & (xr >= 0) & (xr < w)
    back = np.take_along_axis(dr, np.clip(xr, 0, w - 1), axis=1)
    d = np.where(inside & (np.abs(back - dl) <= max_diff), dl, 0)
    if flip:
        d = d[:, ::-1]
    with np.errstate(divide="ignore"):
        return np.where(d > 0, rect["f"] * abs(rect["baseline"]) / d, np.nan)


def rect_and_matcher(session: Path, scale: float, size: tuple[int, int], p: Params | None):
    """Rectification (from <session>/calib) and SGBM matcher for images of `size` (w, h) taken at
    export `scale`: the same for recorded walks and live maps."""
    from .stereo_depth import NUM_DISP, matcher, rectifier

    out = p.rect_out if p is not None else 0.5
    rect = rectifier(session, scale, tuple(size), out)
    block = p.sgbm_block if p is not None else 5
    return rect, matcher(max(16, int(round(NUM_DISP * out / 16)) * 16), block)  # same metric range


def load_frames(session: Path, export_dir: Path, traj: str, p: Params | None = None) -> Frames:
    import cv2

    from .voxel_map import poses_world_cam0

    exp = json.loads((export_dir / "export.json").read_text())
    files = sorted((export_dir / "mav0" / "cam0" / "data").glob("*.png"))
    stamps = np.array([int(f.stem) for f in files])
    first = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    rect, sgbm = rect_and_matcher(session, exp.get("scale", 0.5), first.shape[::-1], p)
    return Frames(
        files,
        stamps,
        poses_world_cam0(session, traj, stamps),
        rect,
        sgbm,
        export_dir,
        p.lr_max_diff if p is not None else -1,
    )


def grid_bounds(fr: Frames, p: Params, pad: float = 3.0):
    c = np.array([M[:3, 3] for M in fr.poses.values()])
    return c.min(0) - pad, c.max(0) + pad


@dataclass(frozen=True)
class FusedFrame:
    """What Mapper.step used for one fused frame (contract v2, read-only; viewers such as
    map_live_view). z_raw: depth before filter_depth (NaN = no stereo match), z: after; pix: the ray
    pixels (v, u); o, dirs, r, far: the rays as fused (world frame, after alignment and planting);
    boxes: planted objects active now; left/right: the raw (unrectified) pair; file: the exported
    cam0 file on a recorded walk (None live)."""

    k: int
    now: float
    file: Path | None
    M: np.ndarray
    z_raw: np.ndarray
    z: np.ndarray
    pix: tuple
    o: np.ndarray
    dirs: np.ndarray
    r: np.ndarray
    far: np.ndarray
    boxes: list
    left: np.ndarray | None = None
    right: np.ndarray | None = None


class Mapper:
    """One temporal map fed one stereo frame at a time: by fuse() on a recorded walk and by
    live_map on the rig, so both run the same code. Not thread-safe: one caller."""

    def __init__(self, p: Params, lo, hi, rect: dict, sgbm, align_log=None, predict=None):
        self.p, self.rect, self.sgbm = p, rect, sgbm
        self.grid = Grid(lo, hi, p)
        self.align_log, self.predict = align_log, predict
        self.k = 0  # frames fused so far
        self.rays = None
        self.aligner = None
        if p.align_every:
            from .map_align import AlignParams, MapAligner

            self.aligner = MapAligner(
                AlignParams(
                    every=p.align_every,
                    gain=p.align_gain,
                    dof=p.align_dof,
                    memory=p.align_memory,
                    max_dist_m=p.align_max_dist_m,
                )
            )

    def step(
        self, now: float, M, left=None, right=None, z_raw=None, planted=(), file=None, stamp=0
    ) -> FusedFrame:
        """Fuse one frame at time now (s) with SLAM camera-0 pose M (world <- cam0). Depth comes
        from z_raw when given (recorded walks), else from the raw pair (left, right)."""
        p, grid = self.p, self.grid
        if z_raw is None:
            z_raw = stereo_depth_of(self.rect, self.sgbm, left, right, p.lr_max_diff)
        z = filter_depth(z_raw, p)
        if self.rays is None:
            self.rays = camera_rays(self.rect, z.shape, p.ray_every)
        dirs_c, (v, u), norm = self.rays
        if self.aligner is not None:
            aligner = self.aligner
            Mc = aligner.correct(M)
            zv = z[v, u]
            good = np.isfinite(zv)
            pts = Mc[:3, 3] + (dirs_c[good] * (zv[good] * norm[good])[:, None]) @ Mc[:3, :3].T
            aligner.step(grid, now, self.k, pts)
            M = aligner.correct(M)
            if self.align_log is not None:
                self.align_log.append((int(stamp), aligner.R.copy(), aligner.t.copy()))
        o, dirs = M[:3, 3], dirs_c @ M[:3, :3].T
        zs = z[v, u]
        far = np.isposinf(zs)
        zs = np.where(far, p.max_depth_m, zs)
        r = zs * norm
        boxes = [(b0, b1) for b0, b1, ta, tb in planted if ta <= now <= tb]
        if boxes:
            r_in = inject(o, dirs, r, boxes)
            far &= r_in == r  # a planted object in front makes a far ray a hit
            r = r_in
            zs = r / norm
        f_px, base = self.rect["f"], abs(self.rect["baseline"])
        sigma_r = zs**2 / (f_px * base) * p.disp_sigma_px * norm
        if self.predict is not None and self.k % PREDICT_EVERY == PREDICT_EVERY - 1:
            predict_rays(
                grid,
                o,
                dirs[::PREDICT_STRIDE],
                np.where(far, np.nan, r)[::PREDICT_STRIDE],
                self.predict,
                now,
            )
        grid.update(*ray_updates(grid, o, dirs, r, sigma_r, far), now)
        ff = FusedFrame(self.k, now, file, M, z_raw, z, (v, u), o, dirs, r, far, boxes, left, right)
        self.k += 1
        return ff


def fuse(
    fr: Frames,
    p: Params,
    step: int,
    planted=(),
    snapshots=(),
    predict: dict | None = None,
    align_log: list | None = None,
    on_frame=None,
):
    """Fuse every step-th frame. planted: [(lo, hi, t_from_s, t_to_s)]; snapshots: times (s) at
    which to record the occupied set; predict: a dict that collects predict_rays counts;
    on_frame(grid, FusedFrame) is called after each frame is fused (viewers; must not change grid).
    Returns (grid, end time s, {time: occupied idx}, ms/frame)."""
    lo, hi = grid_bounds(fr, p)
    mapper = Mapper(p, lo, hi, fr.rect, fr.sgbm, align_log, predict)
    grid = mapper.grid
    t0 = fr.stamps[0]
    snaps, todo, ms = {}, sorted(snapshots), []
    now = 0.0
    for f, s in list(zip(fr.files, fr.stamps, strict=True))[::step]:
        if int(s) not in fr.poses:
            continue
        now = (s - t0) / 1e9
        while todo and todo[0] <= now:
            snaps[todo.pop(0)] = grid.occupied(now)
        c = time.perf_counter()
        left, right = fr.images(f)
        z_raw = stereo_depth_of(fr.rect, fr.sgbm, left, right, fr.lr_max_diff)  # Frames' own L/R
        ff = mapper.step(now, fr.poses[int(s)], left, right, z_raw, planted, f, int(s))
        ms.append((time.perf_counter() - c) * 1000)
        if on_frame is not None:
            on_frame(grid, ff)
    for ts in todo:
        snaps[ts] = grid.occupied(now)
    return grid, now, snaps, float(np.median(ms)) if ms else None


PREDICT_EVERY = 5  # fused frames between predictive checks
PREDICT_STRIDE = 7  # every 7th ray of the frame


# The predictive check's tolerance must not depend on any tuned parameter, or tuning loosens the
# test (2026-11-10: tol was 3 sigma_r from disp_sigma_px, so a wider carving margin scored better by
# itself). Fixed in metres: 3 sigma of 0.5 px disparity noise on the small rectification
# (f 133 px, b 0.142 m), at least 10 cm.
PREDICT_TOL_MIN_M = 0.10
PREDICT_TOL_PER_M2 = 3 * 0.5 / (133.0 * 0.142)


def predict_rays(grid: Grid, o, dirs, r, acc: dict, now: float) -> None:
    """Before a frame is fused: where does the map built so far put the first obstacle along each
    ray, against the frame's own measured range? agree, phantom (map obstacle in front of the
    surface: junk or something that left), missing (map says free where the surface is), new (the
    surface's voxel was never observed). Drift between distant moments barely enters: the map
    there was built from the preceding frames."""
    p = grid.p
    ok = np.isfinite(r)
    dirs, r = dirs[ok], r[ok]
    if not len(r):
        return
    steps = np.arange(0.1, p.max_depth_m + 0.3, p.voxel_m / 2)
    idx = grid.index(o + dirs[:, None, :] * steps[None, :, None])
    mask = np.zeros(len(grid.L), bool)
    mask[grid.occupied(now)] = True  # the same occupied set the map hands out (filters included)
    occ = np.where(idx >= 0, mask[np.maximum(idx, 0)], False)
    first = np.where(occ.any(1), steps[np.argmax(occ, 1)], np.inf)
    tol = np.maximum(PREDICT_TOL_MIN_M, PREDICT_TOL_PER_M2 * r**2)
    end = grid.index(o + dirs * r[:, None])
    seen = np.where(end >= 0, grid.L[np.maximum(end, 0)] != 0, False)
    agree = np.abs(first - r) <= tol
    phantom = first < r - tol
    missing = ~agree & ~phantom & seen
    for k, v in (("agree", agree), ("phantom", phantom), ("missing", missing), ("rays", ok[ok])):
        acc[k] = acc.get(k, 0) + int(v.sum())


# --- self-checks ---------------------------------------------------------------------------------


def wall_metrics(grid: Grid, occ: np.ndarray, sample: int = 20000) -> dict:
    """Floaters (occupied voxels with <= 2 occupied neighbours) and local surface thickness."""
    from scipy.ndimage import convolve
    from scipy.spatial import cKDTree

    dense = np.zeros(len(grid.L), np.uint8)
    dense[occ] = 1
    dense = dense.reshape(grid.shape)
    nb = convolve(dense, np.ones((3, 3, 3), np.uint8), mode="constant").ravel()[occ] - 1
    pts = grid.surface(occ)
    tree = cKDTree(pts)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(pts), min(sample, len(pts)), replace=False)
    thick = []
    for nbr in tree.query_ball_point(pts[pick], 0.10):
        if len(nbr) >= 8:
            q = pts[nbr] - pts[nbr].mean(0)
            thick.append(np.sqrt(max(np.linalg.eigvalsh(q.T @ q / len(q))[0], 0.0)))
    return {
        "occupied": len(occ),
        "floater_share": round(float(np.mean(nb <= 2)), 4),
        "thickness_mm_median": round(float(np.median(thick)) * 1000, 1) if thick else None,
        "thickness_mm_p90": round(float(np.percentile(thick, 90)) * 1000, 1) if thick else None,
    }


def heldout_agreement(fr: Frames, grid: Grid, now: float, step: int, every: int = 8) -> dict:
    """Ray-cast the map from frames it was not built from and compare with their measured depth.
    phantom: the map has an obstacle clearly in front of the measured surface (junk, or something
    that is gone); missing: the measured surface is not in the map."""
    p = grid.p
    occ = np.zeros(len(grid.L), bool)
    occ[grid.occupied(now)] = True
    agree = phantom = missing = n = 0
    rays = None
    for f, s in list(zip(fr.files, fr.stamps, strict=True))[step // 2 :: step * 4]:
        if int(s) not in fr.poses:
            continue
        z_raw = fr.depth(f)
        z = filter_depth(z_raw, p)
        if rays is None:
            rays = camera_rays(fr.rect, z.shape, every)
        dirs_c, (v, u), norm = rays
        M = fr.poses[int(s)]
        o, dirs = M[:3, 3], dirs_c @ M[:3, :3].T
        r = z[v, u] * norm
        ok = np.isfinite(r)
        if not ok.any():
            continue
        dirs, r = dirs[ok], r[ok]
        steps = np.arange(0.1, p.max_depth_m + 0.3, p.voxel_m / 2)
        idx = grid.index(o + dirs[:, None, :] * steps[None, :, None])
        filled = np.where(idx >= 0, occ[np.maximum(idx, 0)], False)
        first = np.where(filled.any(1), steps[np.argmax(filled, 1)], np.inf)
        tol = np.maximum(PREDICT_TOL_MIN_M, PREDICT_TOL_PER_M2 * r**2)  # parameter-free
        agree += int(np.sum(np.abs(first - r) <= tol))
        phantom += int(np.sum(first < r - tol))
        missing += int(np.sum(first > r + tol))
        n += len(r)
    return {
        "rays": n,
        "agree": round(agree / max(n, 1), 4),
        "phantom": round(phantom / max(n, 1), 4),
        "missing": round(missing / max(n, 1), 4),
    }


# planted objects go at the farthest of these that fits in front of what the camera saw
PLANT_DISTS_M = (1.2, 1.0, 0.8, 0.6, 0.5, 0.4)


def _free_in_front(fr: Frames, key: int, dist: float, half_lat: float, half_depth: float) -> bool:
    """Did the camera see through where the object would go? Over the object's image patch, most
    pixels have depth and nearly all of it lies 20 cm or more behind the object (2026-11-11:
    objects planted into real structure got hit again at the revisit and could never clear)."""
    f = fr.files[int(np.searchsorted(fr.stamps, key))]
    z = fr.depth(f)
    P1 = fr.rect["P1"]
    cx, cy, rad = int(P1[0, 2]), int(P1[1, 2]), int(P1[0, 0] * half_lat / dist)
    patch = z[max(cy - rad, 0) : cy + rad, max(cx - rad, 0) : cx + rad]
    ok = np.isfinite(patch)
    return bool(ok.mean() > 0.5 and np.percentile(patch[ok], 10) > dist + half_depth + 0.2)


def plan_change_test(fr: Frames, step: int):
    """Place a fake person and a fake door in front of the camera at moments the place is seen
    again later. Returns [(name, lo, hi, t_from_s, t_to_s, revisit_s)]."""
    t0 = fr.stamps[0]
    keys = [int(s) for s in fr.stamps[::step] if int(s) in fr.poses]
    T = np.array([(k - t0) / 1e9 for k in keys])
    C = np.array([fr.poses[k][:3, 3] for k in keys])
    F = np.array([fr.poses[k][:3, 2] for k in keys])  # optical axis in the world
    out = []
    for name, frac, dist, size, dur in (
        # a cube, not a tall box: on anchored poses the world is the board frame and its z need not
        # be vertical
        ("person", 0.25, 1.2, (0.5, 0.5, 0.5), 3.0),
        ("door", 0.45, 1.5, (0.9, 0.9, 1.2), 10.0),
    ):
        best = None
        for a in range(int(frac * len(T)), len(T)):
            half_depth = 0.03 if name == "door" else max(size) / 2  # the door is a thin slab
            for d in PLANT_DISTS_M:
                if d > dist:
                    continue
                target = C[a] + F[a] * d
                later = T > T[a] + dur + 10
                to = target - C
                dist_all = np.linalg.norm(to, axis=1)
                seen = later & (dist_all < 2.2) & (np.sum(to * F, 1) > 0.85 * dist_all)
                lat = max(size) / 2
                if seen.sum() >= 5 and _free_in_front(fr, keys[a], d, lat, half_depth):
                    best = (a, target, T[np.flatnonzero(seen)[0]])
                    break
            if best is not None:
                break
        if best is None:
            continue
        a, target, revisit = best
        half = np.array(size) / 2
        if name == "door":  # thin slab facing the camera
            ax = np.argmax(np.abs(F[a]))
            half = half.copy()
            half[ax] = 0.03
        out.append((name, target - half, target + half, T[a], T[a] + dur, revisit))
    return out


REVISIT_LAG_S = 3.0


def change_test(fr: Frames, p: Params, step: int, alog: list | None = None) -> dict:
    """Plant the objects, fuse, and compare with the same fuse without them: the occupied voxels
    the plant added inside its box (grown by one voxel, so the faces count) while it stood there,
    3 s after the place was seen again, and at the end. With scan-to-map alignment, alog (the main
    run's corrections) turns the poses into fixed aligned ones first: planning, planted and base run
    then all use the same poses, so the plant is the only difference (2026-11-11: aligning each run
    separately put plants and revisits in different frames)."""
    if alog:
        fr = replace(fr, poses=apply_alignment(fr.poses, alog))
    p = replace(p, align_every=0)
    plan = plan_change_test(fr, step)
    planted = [(lo, hi, ta, tb) for _, lo, hi, ta, tb, _ in plan]
    # snapshots: while it stood there, and 3 s after the place was seen again (cleared by the
    # revisit, not by decay; the end state mixes both)
    snaps = sorted({tb for *_, tb, _ in plan} | {rv + REVISIT_LAG_S for *_, rv in plan})
    grid, now, with_at, _ = fuse(fr, p, step, planted, snaps)
    base_grid, _, base_at, _ = fuse(fr, p, step, (), snaps + [now])
    end_with, end_base = set(grid.occupied(now).tolist()), set(base_at[now].tolist())
    res = {}
    for name, lo, hi, ta, tb, rv in plan:
        axes = [
            np.arange(a - p.voxel_m, b + p.voxel_m, p.voxel_m / 2)
            for a, b in zip(lo, hi, strict=True)
        ]
        box = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        idx = set(np.unique(grid.index(box)).tolist()) - {-1}

        there = len((set(with_at[tb].tolist()) & idx) - set(base_at[tb].tolist()))
        end_set = (end_with & idx) - end_base
        end = len(end_set)
        # left over although rays crossed the place after the object left (unseen space can only
        # fade): the base run's last update of the voxel came after the plant ended
        seen = [v for v in end_set if base_grid.t[v] > tb]
        there_seen = [
            v
            for v in (set(with_at[tb].tolist()) & idx) - set(base_at[tb].tolist())
            if base_grid.t[v] > tb
        ]
        ra = rv + REVISIT_LAG_S
        revisit = len((set(with_at[ra].tolist()) & idx) - set(base_at[ra].tolist()))
        res[name] = {
            "planted_s": [round(ta, 1), round(tb, 1)],
            "seen_again_s": round(float(rv), 1),
            "extra_voxels_while_there": there,
            "extra_voxels_after_revisit": revisit,
            "cleared_by_revisit": round(1 - revisit / there, 3) if there else None,
            "extra_voxels_at_end": end,
            "cleared": round(1 - end / there, 3) if there else None,
            "seen_again_voxels": len(there_seen),
            "seen_again_left_at_end": len(seen),
            "cleared_where_seen_again": round(1 - len(seen) / len(there_seen), 3)
            if there_seen
            else None,
        }
    return res


FLOOR_CLEAR_M = 0.10  # obstacles start this far above the floor (thresholds, cables, floor noise)


def floor_height(z: np.ndarray, ref_z: float) -> float | None:
    """The floor: the lowest 5 cm layer of occupied voxels more than 0.5 m below the camera that
    holds at least half as many as the fullest such layer (tables are higher; walk2: 1.0 m below
    the camera). None when too little lies down there."""
    below = z[z < ref_z - 0.5]
    if len(below) < 0.03 * len(z):
        return None
    h, e = np.histogram(below, bins=np.arange(below.min(), below.max() + 0.1, 0.05))
    k = int(np.flatnonzero(h >= 0.5 * h.max())[0])
    return float((e[k] + e[k + 1]) / 2)


def obstacle_grid(grid: Grid, occ: np.ndarray, ref_z: float, band=(-1.2, 0.3), cell_m=0.05):
    """2D obstacle map for driving: a cell is an obstacle when an occupied voxel sits between
    FLOOR_CLEAR_M above the floor and band[1] above ref_z (the camera's height); free when rays
    crossed it in that band or the floor was seen there; unknown otherwise. Without a floor the band
    is band[0]..band[1] around ref_z. Returns (uint8 image: 0 obstacle, 255 free, 128 unknown;
    origin xy of pixel (0, 0); cell_m; floor z or None)."""
    c = grid.centers(occ)
    floor = floor_height(c[:, 2], ref_z)
    z_lo = floor + FLOOR_CLEAR_M if floor is not None else ref_z + band[0]
    z_hi = ref_z + band[1]
    lo = grid.lo[:2]
    n = np.ceil((grid.lo[:2] + grid.shape[:2] * grid.p.voxel_m - lo) / cell_m).astype(int)
    img = np.full((n[1], n[0]), 128, np.uint8)

    def paint(pts, val):
        ij = ((pts[:, :2] - lo) / cell_m).astype(int)
        img[ij[:, 1], ij[:, 0]] = val

    free = grid.centers(np.flatnonzero(grid.L < 0))
    paint(free[(free[:, 2] > z_lo) & (free[:, 2] < z_hi)], 255)
    if floor is not None:
        paint(c[np.abs(c[:, 2] - floor) < FLOOR_CLEAR_M], 255)
    paint(c[(c[:, 2] > z_lo) & (c[:, 2] < z_hi)], 0)
    return img, lo, cell_m, floor


def board_stops_world(session: Path, fr: Frames) -> list[tuple[float, np.ndarray]]:
    """Per board stop: (end time s, T_world_board). The board's pose in the SLAM world comes from
    the board seen by the camera (PnP) placed with the SLAM pose of the stop's middle frame."""
    from .slam_eval import ground_truth
    from .slam_legs import stops

    det = session / "calib" / "detections_all.npz"
    if not det.exists() or len(np.load(det)["keys0"]) < 10:
        return []
    t_gt, R_bc, p_bc = ground_truth(session)  # camera 0 in the board frame
    keys = np.array(sorted(fr.poses))
    out = []
    for run in stops(t_gt):
        g = run[len(run) // 2]
        k = keys[np.argmin(np.abs(keys - t_gt[g]))]
        if abs(k - t_gt[g]) > MAX_POSE_DT_NS:
            continue
        T_bc = np.eye(4)
        T_bc[:3, :3], T_bc[:3, 3] = R_bc[g], p_bc[g]
        out.append(((t_gt[run[-1]] - fr.stamps[0]) / 1e9, fr.poses[int(k)] @ np.linalg.inv(T_bc)))
    return out


def board_area(session: Path) -> tuple[np.ndarray, np.ndarray]:
    """Inner corner rectangle shrunk by one square (edges mix in the wall), board frame."""
    from .calib import board_points

    cal = json.loads((session / "calib" / "calib.json").read_text())
    sq = cal["square_m"]
    obj = board_points(tuple(cal["board"]), sq)
    return obj[:, :2].min(0) + sq, obj[:, :2].max(0) - sq


def board_in_map(grid: Grid, occ: np.ndarray, T_wb: np.ndarray, area) -> dict:
    """The board in the map against its true plane: thickness, offset, ghosts, coverage."""
    lo, hi = area
    b = (grid.surface(occ) - T_wb[:3, 3]) @ T_wb[:3, :3]
    over = np.all((b[:, :2] >= lo) & (b[:, :2] <= hi), axis=1)
    z = b[over & (np.abs(b[:, 2]) < 0.10), 2]
    on = over & (np.abs(b[:, 2]) < grid.p.voxel_m)
    cells = {tuple(c) for c in np.floor(b[on, :2] / grid.p.voxel_m).astype(int)}
    n_cells = int(np.prod(np.ceil((hi - lo) / grid.p.voxel_m)))
    if not len(z):
        return {"voxels": 0, "coverage": 0.0}
    return {
        "voxels": len(z),
        "thickness_mm": round(float(np.sqrt(np.mean(z**2))) * 1000, 1),
        "offset_mm": round(float(np.mean(z)) * 1000, 1),
        "ghost_share": round(float(np.mean(np.abs(z) >= 2 * grid.p.voxel_m)), 3),
        "coverage": round(min(len(cells) / n_cells, 1.0), 3),
    }


def board_test(session: Path, fr: Frames, grid: Grid, now: float, snaps: dict, plan) -> dict:
    """At the end of each board stop, the map as it was then against that stop's board plane
    ("now") and the first stop's ("first", i.e. drift between visits); the final map against
    every stop's plane."""
    area = board_area(session)
    end = grid.occupied(now)
    rows = []
    for k, (te, T_wb) in enumerate(plan):
        row = {"stop": k, "end_s": round(te, 1)}
        row["now"] = board_in_map(grid, snaps[te], T_wb, area)
        row["first"] = board_in_map(grid, snaps[te], plan[0][1], area)
        row["final_map"] = board_in_map(grid, end, T_wb, area)
        rows.append(row)
    shift = [np.linalg.norm(T[:3, 3] - plan[0][1][:3, 3]) * 1000 for _, T in plan]
    return {"stops": rows, "board_shift_vs_first_mm": [round(float(s), 1) for s in shift]}


def apply_alignment(poses: dict, alog: list) -> dict:
    """Camera poses {stamp: 4x4} with the running scan-to-map correction of their time (causal)."""
    ts = np.array([x[0] for x in alog])
    out = {}
    for s, M in poses.items():
        k = int(np.searchsorted(ts, s, side="right")) - 1
        if k < 0:
            out[s] = M
            continue
        R, t = alog[k][1], alog[k][2]
        N = M.copy()
        N[:3, :3], N[:3, 3] = R @ M[:3, :3], R @ M[:3, 3] + t
        out[s] = N
    return out


def write_aligned(traj: Path, alog: list, out: Path) -> dict:
    """Apply the running scan-to-map correction (world frame) to the IMU trajectory and write it as
    TUM. Poses before the first correction keep the identity."""
    from .slam_eval import read_tum

    t, R_wi, p_wi = read_tum(traj)
    ts = np.array([x[0] for x in alog])
    k = np.searchsorted(ts, t, side="right") - 1
    Rc = np.stack([np.eye(3)] + [x[1] for x in alog])[k + 1]
    tc = np.stack([np.zeros(3)] + [x[2] for x in alog])[k + 1]
    R_new = np.einsum("nij,njk->nik", Rc, R_wi)
    p_new = np.einsum("nij,nj->ni", Rc, p_wi) + tc
    raw = np.loadtxt(traj, comments="#")
    q = Rotation.from_matrix(R_new).as_quat()
    np.savetxt(out, np.column_stack([raw[:, 0], p_new, q]), fmt="%.9f")
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(alog[-1][1]).as_rotvec()))
    return {
        "frames": len(alog),
        "final_offset_mm": round(float(np.linalg.norm(alog[-1][2])) * 1000, 1),
        "final_offset_deg": round(float(ang), 2),
        "trajectory": str(out),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tars_scan.occupancy_map", description=__doc__.split("\n")[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("export", type=Path)
    ap.add_argument("trajectory")
    ap.add_argument("--step", type=int, default=6, help="fuse every n-th frame (6 = 5 Hz)")
    ap.add_argument("--test", action="store_true", help="run the self-checks too")
    ap.add_argument("--set", action="append", default=[], help="Params override, e.g. l_occ=2.5")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    p = Params()
    for kv in a.set:
        k, v = kv.split("=", 1)
        cur = getattr(p, k)  # bool("0") is True, so flags parse their text
        p = replace(p, **{k: v.lower() in ("1", "true") if isinstance(cur, bool) else type(cur)(v)})
    fr = load_frames(a.session, a.export, a.trajectory, p)
    plan = board_stops_world(a.session, fr) if a.test else []
    t = time.perf_counter()
    pred: dict = {}
    alog: list = []
    grid, now, snaps, ms = fuse(
        fr, p, a.step, snapshots=[te for te, _ in plan], predict=pred, align_log=alog
    )
    occ = grid.occupied(now)
    r = {
        "params": asdict(p),
        "step": a.step,
        "fuse_s": round(time.perf_counter() - t, 1),
        "ms_per_frame": ms,
    }
    n = max(pred.get("rays", 0), 1)
    r["predict"] = {k: round(pred.get(k, 0) / n, 4) for k in ("agree", "phantom", "missing")}
    r["predict"]["new"] = round(1 - sum(r["predict"].values()), 4)
    r["predict"]["rays"] = pred.get("rays", 0)
    fr_raw = fr  # the change test fuses again (and aligns again): it needs the uncorrected poses
    if alog:  # the other checks read the poses the map was built with
        fr = replace(fr, poses=apply_alignment(fr.poses, alog))
        plan = board_stops_world(a.session, fr) if plan else plan
    if plan:
        r["board"] = board_test(a.session, fr, grid, now, snaps, plan)
    if a.test:
        r["walls"] = wall_metrics(grid, occ)
        r["heldout"] = heldout_agreement(fr, grid, now, a.step)
        r["change"] = change_test(fr_raw, p, a.step, alog)
    name = a.name or Path(a.trajectory).stem
    if alog:  # the SLAM trajectory with the scan-to-map correction (causal: latest one so far)
        r["align"] = write_aligned(
            Path(a.trajectory), alog, a.export / f"occupancy_{name}_aligned.txt"
        )
    np.savez_compressed(
        a.export / f"occupancy_{name}.npz",
        centers=grid.surface(occ).astype(np.float32),  # mean hit points when subvoxel
        logodds=grid.logodds(now, occ).astype(np.float32),
    )
    import cv2

    ref_z = float(np.median([M[2, 3] for M in fr.poses.values()]))
    img, lo, cell, floor = obstacle_grid(grid, occ, ref_z)
    cv2.imwrite(str(a.export / f"occupancy_{name}_2d.png"), img[::-1])  # north up
    r["obstacle_grid"] = {
        "origin_xy": lo.round(3).tolist(),
        "cell_m": cell,
        "shape": img.shape,
        "floor_below_camera_m": None if floor is None else round(ref_z - floor, 3),
        "obstacle_cells": int((img == 0).sum()),
        "free_cells": int((img == 255).sum()),
    }
    (a.export / f"occupancy_{name}.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
