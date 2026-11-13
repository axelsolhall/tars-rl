"""Live 3D view of the temporal map as it is built, one view per algorithm (Rerun viewer).

  python -m tars_scan.map_live_view [--variants start,noalign,final,anchored] [--no-plant]
                                    [--realtime] [--jobs 2]

Replays a recorded walk (default walk2, 2026-11-10) through occupancy_map.fuse and streams every
fused frame into one Rerun window: a 3D view per variant you can fly around in, plus each variant's
camera image and floor plan, all on one shared timeline (play, pause, scrub). Each variant is also
saved to <out>/<variant>.rrd with the window layout in <out>/layout.rbl, so it replays without
recomputing: `rerun <out>/*.rrd <out>/layout.rbl`.

What each 3D view shows (each line its own entity, toggle it in the left panel):
- camera: the rectified left image in its frustum at the pose the map used, and the path so far
- frame/kept (grey): this frame's depth points that were fused
- frame/lr_rejected (red): points one-sided stereo produces that the left-right check drops, the
  main source of junk in the air (the start-of-night variant has no check, so none are dropped)
- frame/edge_rejected (orange): depth-edge pixels (mixed near/far) dropped before fusion
- map/obstacles: occupied voxels, coloured by height
- map/floaters (red): occupied but isolated (fewer than min_neighbors occupied neighbours), so not
  an obstacle: junk the neighbour filter rejects
- map/not_yet (cyan): surfaces seen, but not often enough yet to count as an obstacle
- map/cleared (magenta): obstacles a ray just saw through (a door opened, a person left)
- planted (yellow boxes): the change test's fake person and door while they are planted into the
  real rays, so you can watch them become obstacles and then vanish when the place is seen again
The camera image carries the same rejection colours per pixel. The board, the SLAM and the map code
are untouched: this module only reads fuse()'s FusedFrame (occupancy_map, contract v1).

Eval: python -m pytest rig/tests/test_map_live_view.py (pixel and voxel classification).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np

from .occupancy_map import (
    FusedFrame,
    Grid,
    Params,
    camera_rays,
    fuse,
    load_frames,
    obstacle_grid,
    plan_change_test,
)

APP = "tars_map_live"
# the start-of-night map: what the night's changes are measured against (night_log.md Reproduce)
OLD = {
    "l_miss_occ": 0.0,
    "subvoxel": False,
    "min_neighbors": 0,
    "carve_stride": 1,
    "lr_max_diff": -1.0,
    "align_every": 0,
}
VARIANTS = {  # name: (title, pose source, Params overrides)
    "start": ("Start of night (raw Basalt)", "raw", OLD),
    "noalign": ("Final, no alignment (raw Basalt)", "raw", {"align_every": 0}),
    "final": ("Final + alignment (live pipeline)", "raw", {}),
    "anchored": ("Final, board-anchored poses (offline)", "anchored", {"align_every": 0}),
}
WALK2 = {
    "session": "runs/rig/20261110-184222-capture",
    "export": "runs/slam/walk2-half-rs",
    "raw": "runs/slam/out/walk2-b-hyp-Ryday.txt",
    "anchored": "runs/slam/out/walk2-anch-up.txt",
}
# pixel classes (camera image overlay)
KEPT, LR_REJECTED, EDGE_REJECTED, FAR = 1, 2, 3, 4
CLASS_INFO = [
    (KEPT, "kept", (150, 150, 150)),
    (LR_REJECTED, "left-right check", (230, 40, 40)),
    (EDGE_REJECTED, "depth edge", (255, 150, 0)),
    (FAR, "beyond max depth", (60, 60, 170)),
]
TILE_M = 1.0  # obstacle map is logged in columns this wide (only changed ones are re-sent)
NOT_YET_MIN_L = 0.5  # log-odds above this but below l_occ: seen, not yet an obstacle
NOT_YET_RECENT_S = 2.0  # ...and updated this recently (not the ones fading out)


def pixel_classes(z_one: np.ndarray | None, z_raw: np.ndarray, z: np.ndarray, max_depth: float):
    """Per pixel: KEPT (fused), LR_REJECTED (one-sided stereo had depth, the left-right check
    dropped it), EDGE_REJECTED (dropped by filter_depth), FAR (beyond max_depth), 0 = no depth.
    z_one: depth without the left-right check (None when the variant has no check)."""
    out = np.zeros(z.shape, np.uint8)
    raw = np.isfinite(z_raw)
    far = raw & (z_raw > max_depth)
    if z_one is not None:
        out[np.isfinite(z_one) & ~raw & (z_one <= max_depth)] = LR_REJECTED
    out[raw & ~np.isfinite(z) & ~far] = EDGE_REJECTED
    out[far] = FAR
    out[np.isfinite(z) & (z <= max_depth)] = KEPT
    return out


def map_layers(grid: Grid, now: float, prev_obstacles: np.ndarray) -> dict:
    """Flat voxel indices per layer: obstacles (what the map reports), floaters (occupied but
    rejected by the neighbour filter), not_yet (updated in the last NOT_YET_RECENT_S, still below
    l_occ), cleared (obstacles at the previous call that rays have since carved free)."""
    p = grid.p
    live = np.flatnonzero(grid.L > NOT_YET_MIN_L)  # decay only lowers |L|
    L = grid.logodds(now, live)
    occ = live[L > p.l_occ]
    obstacles = grid.occupied(now)
    return {
        "obstacles": obstacles,
        "floaters": np.setdiff1d(occ, obstacles, assume_unique=True),
        "not_yet": live[
            (L > NOT_YET_MIN_L) & (L <= p.l_occ) & (grid.t[live] > now - NOT_YET_RECENT_S)
        ],
        "cleared": prev_obstacles[grid.L[prev_obstacles] < 0],
    }


def upright_K(K: np.ndarray, w: int, h: int, flip: bool) -> np.ndarray:
    """Intrinsics of the image turned 180 deg (camera turned by diag(-1, -1, 1))."""
    if not flip:
        return K
    out = K.copy()
    out[0, 2], out[1, 2] = w - 1 - K[0, 2], h - 1 - K[1, 2]
    return out


# blue (floor) -> cyan -> green -> yellow -> red (above the camera); no matplotlib on the Pi
_STOPS = np.array([[48, 18, 160], [30, 160, 230], [60, 220, 90], [250, 200, 30], [220, 40, 30]])


def _height_colors(z: np.ndarray, z_cam: float) -> np.ndarray:
    t = np.clip((z - z_cam + 1.3) / 1.9, 0, 1) * (len(_STOPS) - 1)  # floor ~1 m below .. +0.6 m
    x = np.arange(len(_STOPS))
    return np.stack([np.interp(t, x, _STOPS[:, c]) for c in range(3)], -1).astype(np.uint8)


class LiveLogger:
    """Logs one map's frames and map to the active Rerun recording: fuse()'s on_frame hook on a
    recorded walk, called by live_map on the rig. rect/sgbm: the Mapper's."""

    def __init__(self, v: str, rect: dict, sgbm, p: Params, plan: list, map_every: int, realtime):
        import rerun as rr

        self.rr, self.v, self.rect, self.sgbm, self.p, self.plan = rr, v, rect, sgbm, p, plan
        self.map_every, self.realtime = map_every, realtime
        self.prev = np.zeros(0, np.int64)
        self.tile_hash: dict = {}
        self.planted_on = None
        self.path: list = []
        self.wall0 = time.perf_counter()
        self.z_cam = None  # camera height at the first frame: colours are relative to it
        self.norm = None
        self.R_rect = rect["R1"].T  # rectified cam0 -> cam0 (camera_rays: dirs_cam0 = R1^T d)
        self.flip, self.F = False, np.eye(3)
        rr.log(v, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(f"{v}/camera", rr.AnnotationContext(CLASS_INFO), static=True)
        rr.log(f"floorplan/{v}", rr.AnnotationContext(CLASS_INFO), static=True)

    def _image(self, ff: FusedFrame):
        import cv2

        l_r = cv2.remap(ff.left, *self.rect["maps"][0], cv2.INTER_LINEAR)
        z_one = None
        if self.p.lr_max_diff >= 0:  # what one-sided stereo would have given
            from .stereo_depth import depth
            from .voxel_map import BAND

            z_one = depth(self.rect, self.sgbm, ff.left, ff.right, BAND)
        return l_r, z_one

    def __call__(self, grid: Grid, ff: FusedFrame) -> None:
        import cv2

        rr, v, p = self.rr, self.v, self.p
        rr.set_time("walk", duration=ff.now)
        l_r, z_one = self._image(ff)
        if self.norm is None:
            self.norm = camera_rays(self.rect, ff.z.shape, p.ray_every)[2]
            self.z_cam = float(ff.M[2, 3])
            # the rig's cameras are mounted upside down (cam0 ends up on the right): show the
            # image turned 180 deg, with the logged camera turned the same way, so the room is
            # upright (decided from the first pose: image down axis pointing world-up)
            self.flip = bool((ff.M[:3, :3] @ self.R_rect)[2, 1] > 0)
            self.F = np.diag([-1.0, -1.0, 1.0]) if self.flip else np.eye(3)
            h, w = ff.z.shape
            rr.log(
                f"{v}/camera/image",
                rr.Pinhole(
                    image_from_camera=upright_K(self.rect["P1"][:, :3], w, h, self.flip),
                    resolution=[w, h],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.4,
                ),
                static=True,
            )
        rr.log(
            f"{v}/camera",
            rr.Transform3D(translation=ff.o, mat3x3=ff.M[:3, :3] @ self.R_rect @ self.F),
        )
        shown = l_r[::-1, ::-1] if self.flip else l_r
        ok, jpg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 80])
        rr.log(
            f"{v}/camera/image", rr.EncodedImage(contents=jpg.tobytes(), media_type="image/jpeg")
        )
        cls = pixel_classes(z_one, ff.z_raw, ff.z, p.max_depth_m)
        shown = cls[::-1, ::-1] if self.flip else cls
        rr.log(f"{v}/camera/image/rejected", rr.SegmentationImage(shown, opacity=0.45))

        vv, uu = ff.pix
        c = cls[vv, uu]

        def world(mask, z):
            mask = mask.copy()
            mask[1::2] = False  # every 2nd ray: the junk stays visible at half the recording size
            return ff.o + ff.dirs[mask] * (z[vv, uu][mask] * self.norm[mask])[:, None]

        kept = np.isfinite(ff.r) & ~ff.far
        kept[np.arange(len(kept)) % 3 > 0] = False  # every 3rd fused point shows the frame
        rr.log(
            f"{v}/frame/kept",
            rr.Points3D(ff.o + ff.dirs[kept] * ff.r[kept, None], radii=0.008, colors=[170] * 3),
        )
        if z_one is not None:
            pts = world(c == LR_REJECTED, z_one)
            rr.log(f"{v}/frame/lr_rejected", rr.Points3D(pts, radii=0.012, colors=[230, 40, 40]))
        pts = world(c == EDGE_REJECTED, ff.z_raw)
        rr.log(f"{v}/frame/edge_rejected", rr.Points3D(pts, radii=0.012, colors=[255, 150, 0]))
        n_depth = max(int((c > 0).sum()), 1)
        rr.log(
            f"stats/rejected_pct/{v}", rr.Scalars(100 * float((c == LR_REJECTED).sum()) / n_depth)
        )
        self._planted(ff.now)
        if ff.k % self.map_every == 0:
            self._map(grid, ff)
        if self.realtime > 0:  # pace to the walk's own clock (no polling: one sleep per frame)
            ahead = ff.now / self.realtime - (time.perf_counter() - self.wall0)
            if ahead > 0:
                time.sleep(ahead)

    def _planted(self, now: float) -> None:
        rr = self.rr
        on = [(n, lo, hi) for n, lo, hi, ta, tb, _ in self.plan if ta <= now <= tb]
        if [n for n, _, _ in on] == self.planted_on:
            return
        self.planted_on = [n for n, _, _ in on]
        if on:
            rr.log(
                f"{self.v}/planted",
                rr.Boxes3D(
                    mins=[lo for _, lo, _ in on],
                    sizes=[hi - lo for _, lo, hi in on],
                    colors=[255, 220, 0],
                    labels=[f"planted {n}" for n, _, _ in on],
                ),
            )
        else:
            rr.log(f"{self.v}/planted", rr.Clear(recursive=False))

    def _tiles(self, grid: Grid, obstacles: np.ndarray) -> None:
        """Log the obstacles as TILE_M x TILE_M columns, re-sending only the columns whose voxel
        set changed: walls rarely change, so the recording stays small (whole map every snapshot
        was 631 MB for walk2's final variant, 2026-11-11)."""
        c = grid.centers(obstacles)
        key = np.floor(c[:, :2] / TILE_M).astype(np.int64)
        tid = key[:, 0] * 100003 + key[:, 1]
        order = np.argsort(tid, kind="stable")
        tid_s, idx_s = tid[order], obstacles[order]
        cut = np.flatnonzero(np.diff(tid_s)) + 1
        now_tiles = {}
        for part_t, part_i in zip(np.split(tid_s, cut), np.split(idx_s, cut), strict=True):
            if len(part_t):
                now_tiles[int(part_t[0])] = part_i
        for t, idx in now_tiles.items():
            h = hash(idx.tobytes())
            if self.tile_hash.get(t) != h:
                pts = grid.surface(idx)
                self.rr.log(
                    f"{self.v}/map/obstacles/{t}",
                    self.rr.Points3D(
                        pts, radii=0.022, colors=_height_colors(pts[:, 2], self.z_cam)
                    ),
                )
                self.tile_hash[t] = h
        for t in [t for t in self.tile_hash if t not in now_tiles]:
            self.rr.log(f"{self.v}/map/obstacles/{t}", self.rr.Clear(recursive=False))
            del self.tile_hash[t]

    def _map(self, grid: Grid, ff: FusedFrame) -> None:
        rr, v = self.rr, self.v
        lay = map_layers(grid, ff.now, self.prev)
        self.prev = lay["obstacles"]
        self._tiles(grid, lay["obstacles"])
        for name, col, rad in (
            ("floaters", [230, 30, 30], 0.03),
            ("not_yet", [80, 220, 255], 0.012),
            ("cleared", [255, 0, 220], 0.03),
        ):
            rr.log(f"{v}/map/{name}", rr.Points3D(grid.centers(lay[name]), radii=rad, colors=col))
        self.path.append(ff.o)
        if len(self.path) % 10 == 2:  # the whole path is re-sent: not every snapshot
            rr.log(f"{v}/path", rr.LineStrips3D([self.path], radii=0.006, colors=[255, 255, 255]))
        rr.log(f"stats/obstacles/{v}", rr.Scalars(len(lay["obstacles"])))
        rr.log(f"stats/floaters/{v}", rr.Scalars(len(lay["floaters"])))
        if ff.k % (self.map_every * 5) == 0 and len(lay["obstacles"]):
            img, *_ = obstacle_grid(grid, lay["obstacles"], self.z_cam)
            rr.log(f"floorplan/{v}", rr.Image(img[::-1]))


def blueprint(variants: list[str], titles: dict | None = None):
    import rerun.blueprint as rrb

    titles = titles or {v: VARIANTS[v][0] for v in variants}
    short = {v: titles[v].split(" (")[0] for v in variants}
    views3d = [rrb.Spatial3DView(origin=f"/{v}", name=titles[v]) for v in variants]
    cams = [
        rrb.Tabs(
            rrb.Spatial2DView(origin=f"/{v}/camera/image", name=f"{short[v]}: camera"),
            rrb.Spatial2DView(origin=f"/floorplan/{v}", name=f"{short[v]}: floor plan"),
        )
        for v in variants
    ]
    plots = rrb.Tabs(
        rrb.TimeSeriesView(origin="/stats/obstacles", name="obstacle voxels"),
        rrb.TimeSeriesView(origin="/stats/floaters", name="floaters rejected"),
        rrb.TimeSeriesView(origin="/stats/rejected_pct", name="% depth dropped by L/R check"),
    )
    return rrb.Blueprint(
        rrb.Vertical(rrb.Grid(*views3d), rrb.Horizontal(*cams, plots), row_shares=[3, 1]),
        collapse_panels=False,
    )


def run_variant(v: str, a, recording_id: str) -> None:
    """One variant: fuse the walk and stream it (worker process)."""
    import rerun as rr

    _, src, over = VARIANTS[v]
    p = replace(Params(), **over)
    fr = load_frames(Path(a.session), Path(a.export), a.raw if src == "raw" else a.anchored, p)
    plan = plan_change_test(fr, a.step) if a.plant else []
    rr.init(APP, recording_id=recording_id)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sinks = [rr.FileSink(str(out / f"{v}.rrd"))]
    if not a.no_viewer:
        sinks.insert(0, rr.GrpcSink())
    rr.set_sinks(*sinks)
    log = LiveLogger(v, fr.rect, fr.sgbm, p, plan, a.map_every, a.realtime)
    t = time.perf_counter()
    fuse(fr, p, a.step, planted=[(lo, hi, ta, tb) for _, lo, hi, ta, tb, _ in plan], on_frame=log)
    print(f"{v}: done in {time.perf_counter() - t:.0f} s -> {out / f'{v}.rrd'}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tars_scan.map_live_view", description=__doc__.split("\n")[0])
    for k, val in WALK2.items():
        ap.add_argument(f"--{k}", default=val)
    ap.add_argument("--variants", default="start,noalign,final,anchored")
    ap.add_argument("--step", type=int, default=6, help="fuse every n-th frame (6 = 5 Hz)")
    ap.add_argument("--map-every", type=int, default=3, help="log the map every n fused frames")
    ap.add_argument("--no-plant", dest="plant", action="store_false", help="no fake person/door")
    ap.add_argument(
        "--realtime", type=float, default=0.0, help="pace to walk time x this (1 = live)"
    )
    ap.add_argument("--jobs", type=int, default=2, help="variants computed at once")
    ap.add_argument("--out", default="runs/live_view/walk2")
    ap.add_argument("--memory", default="6GB", help="viewer memory limit")
    ap.add_argument("--no-viewer", action="store_true", help="only write the .rrd files")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--recording-id", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        run_variant(a.worker, a, a.recording_id)
        return
    import rerun as rr

    variants = a.variants.split(",")
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
    rid = str(uuid.uuid4())
    bp = blueprint(variants)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    bp.save(APP, str(Path(a.out) / "layout.rbl"))
    if not a.no_viewer:
        rr.init(APP, recording_id=rid)
        exe = Path(sys.executable).with_name("rerun")  # the venv's viewer; PATH may not have it
        rr.spawn(
            memory_limit=a.memory,
            default_blueprint=bp,
            executable_path=str(exe) if exe.exists() else None,
        )
    passthrough = [x for x in (argv if argv is not None else sys.argv[1:])]
    todo, running = list(variants), []
    while todo or running:  # waits on the exact worker processes, never polls
        while todo and len(running) < a.jobs:
            v = todo.pop(0)
            cmd = [sys.executable, "-m", "tars_scan.map_live_view", *passthrough]
            running.append(subprocess.Popen([*cmd, "--worker", v, "--recording-id", rid]))
        running.pop(0).wait()
    print(f"replay: rerun {a.out}/*.rrd {a.out}/layout.rbl")


if __name__ == "__main__":
    main()
