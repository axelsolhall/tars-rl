"""Rough ground for training: one big heightfield, a different patch of it under every env.

Roadmap stage 3. Every model trained on the flat floor fell on bumpy ground, steps and slopes
(2026-11-03 level sweep, runs/chain-dr/*/levels), so training needs uneven floors. Built the way
MJX can afford with 8192 parallel envs:

    one terrain    a `size_m` square heightfield, generated once from `seed` (numpy) and shared
                   by all envs, so GPU memory holds one grid, not one per env
    rougher along x: difficulty 0 at the first pad column (flat) to `difficulty_max` at the last
    pads           flat round spots (`pad_flat_m`) on a grid; each env is placed on one pad
                   with its own yaw by moving and turning the heightfield's body per env
                   (`randomize.for_env`), so TARS still starts at the origin facing +x
    features       scaled by the local difficulty, a bit beyond the test levels at difficulty 1
                   so the levels (boxes there, a heightfield here) stay a fair held-out test:
                     tiles      0.3 m squares, 0-`tile_m` high (bumpy)
                     bands      long narrow strips up to `band_m` high (door thresholds)
                     rugs       big thin plates, `rug_m` high
                     mounds     round hills up to `mound_m` high, 1-2.5 m radius (slopes)

Contact: on terrain each slab touches the ground through small spheres on its long edges, each
meeting the one heightfield triangle under it (`point_hfield`; why: see "Contact" below).

With terrain on, the heightfield replaces the plane as the floor's collision (the plane stays
for looks only). The heightfield is large enough that TARS never walks off it: pads keep
`margin_m` from the edge, and a 20 s episode took TARS at most 8.2 m from its start
(2026-11-03, 128 episodes of runs/chain-dr/default). The nominal model (no randomization) puts
pad 0 (difficulty 0: perfectly flat) under the origin, so nominal training-curve evals stay on
flat ground.

CPU MuJoCo is not a reference here: its own sphere-heightfield collision let resting spheres sink
15-20 mm into rough ground and found 8 of 12 corner contacts on flat ground (2026-11-03), where
`point_hfield` rests within 0.1 mm and matches the plane exactly. The engines are compared on the
box-built levels (`tars_rl.env.levels`) instead.

Eval: `python -m tars_rl.env.terrain` prints the terrain's heights, steps and slopes by
difficulty, lets TARS stand on 64 random pads with zero action (none should fall) and times MJX
physics with and without terrain; `--png FILE` draws the terrain from above.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

TERRAIN_GEOM = "terrain"
TERRAIN_VERSION = 1


@dataclass(frozen=True)
class TerrainConfig:
    seed: int = 0
    size_m: float = 80.0  # side of the square heightfield
    cell_m: float = 0.05  # grid spacing: a 3 cm threshold rises over one cell
    margin_m: float = 12.0  # pads keep this far from the edge (> the 8.2 m TARS walks)
    pad_spacing_m: float = 4.0
    pad_flat_m: float = 0.5  # flat radius around each pad, then a blend of the same width
    difficulty_max: float = 1.0
    tile_m: float = 0.04
    band_m: float = 0.035
    rug_m: float = 0.012
    mound_m: float = 0.18
    version: int = TERRAIN_VERSION

    def check(self) -> TerrainConfig:
        if not 0 < self.cell_m < self.pad_flat_m:
            raise ValueError(f"cell_m {self.cell_m} must be > 0 and smaller than pad_flat_m")
        if self.size_m <= 2 * self.margin_m + self.pad_spacing_m:
            raise ValueError(f"size_m {self.size_m} leaves no pads inside margin_m "
                             f"{self.margin_m}")
        if not 0 <= self.difficulty_max <= 1:
            raise ValueError(f"difficulty_max {self.difficulty_max} must be in [0, 1]")
        return self

    @property
    def n(self) -> int:
        """Grid points per side."""
        return int(round(self.size_m / self.cell_m)) + 1

    def pads(self) -> np.ndarray:
        """(k, 2) pad centres in the terrain's own frame, row by row along x."""
        half = self.size_m / 2 - self.margin_m
        ticks = np.arange(-half, half + 1e-9, self.pad_spacing_m)
        return np.stack(np.meshgrid(ticks, ticks, indexing="xy"), -1).reshape(-1, 2)

    def difficulty(self, x: np.ndarray) -> np.ndarray:
        """Difficulty at terrain x: 0 at the first pad column, difficulty_max at the last."""
        half = self.size_m / 2 - self.margin_m
        return self.difficulty_max * np.clip((x + half) / (2 * half), 0.0, 1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TerrainConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names}).check()


def _cosine_bump(r: np.ndarray, radius: float) -> np.ndarray:
    return np.where(r < radius, 0.5 * (1 + np.cos(np.pi * np.minimum(r / radius, 1))), 0.0)


def heights(cfg: TerrainConfig) -> np.ndarray:
    """(n, n) heights in metres, [row = y, col = x], grid point (r, c) at
    (-size/2 + c*cell, -size/2 + r*cell). Deterministic in cfg."""
    cfg.check()
    n, cell, half = cfg.n, cfg.cell_m, cfg.size_m / 2
    rng = np.random.default_rng(cfg.seed)
    ax = -half + cell * np.arange(n)
    x, y = np.meshgrid(ax, ax, indexing="xy")
    area = cfg.size_m ** 2

    # tiles: 0.3 m squares of random height
    k = int(math.ceil(cfg.size_m / 0.3))
    tiles = rng.uniform(0, 1, (k, k))[np.minimum(((y + half) / 0.3).astype(int), k - 1),
                                      np.minimum(((x + half) / 0.3).astype(int), k - 1)]
    tiles *= cfg.tile_m

    plates = np.zeros_like(x)  # bands and rugs: raised plates, the highest one wins

    def plate(cx, cy, yaw, length, width, h):
        reach = 0.5 * math.hypot(length, width) + cell
        c0, c1 = np.clip(((np.array([cx - reach, cx + reach]) + half) / cell).astype(int), 0, n)
        r0, r1 = np.clip(((np.array([cy - reach, cy + reach]) + half) / cell).astype(int), 0, n)
        dx, dy = x[r0:r1, c0:c1] - cx, y[r0:r1, c0:c1] - cy
        u = dx * math.cos(yaw) + dy * math.sin(yaw)
        v = -dx * math.sin(yaw) + dy * math.cos(yaw)
        inside = (np.abs(u) <= length / 2) & (np.abs(v) <= width / 2)
        win = plates[r0:r1, c0:c1]
        win[...] = np.where(inside, np.maximum(win, h), win)

    for _ in range(int(area / 4)):  # thresholds: about one per 4 m^2
        plate(*rng.uniform(-half, half, 2), rng.uniform(0, math.pi), rng.uniform(0.6, 2.5),
              rng.uniform(0.06, 0.25), cfg.band_m * rng.uniform(0.4, 1.0))
    for _ in range(int(area / 12)):  # rugs: about one per 12 m^2
        plate(*rng.uniform(-half, half, 2), rng.uniform(0, math.pi), rng.uniform(1.0, 2.2),
              rng.uniform(0.6, 1.5), cfg.rug_m * rng.uniform(0.6, 1.0))

    mounds = np.zeros_like(x)
    for _ in range(int(area / 10)):  # about one per 10 m^2, they overlap into rolling ground
        cx, cy = rng.uniform(-half, half, 2)
        radius, h = rng.uniform(1.0, 2.5), cfg.mound_m * rng.uniform(0.3, 1.0)
        c0, c1 = np.clip(((np.array([cx - radius, cx + radius]) + half) / cell).astype(int), 0, n)
        r0, r1 = np.clip(((np.array([cy - radius, cy + radius]) + half) / cell).astype(int), 0, n)
        r = np.hypot(x[r0:r1, c0:c1] - cx, y[r0:r1, c0:c1] - cy)
        mounds[r0:r1, c0:c1] = np.maximum(mounds[r0:r1, c0:c1], h * _cosine_bump(r, radius))

    z = cfg.difficulty(x) * (tiles + plates + mounds)

    # flat pads: zero inside pad_flat_m, blending back to the terrain over the same width
    pad = np.zeros_like(x)
    reach = 2 * cfg.pad_flat_m + cell
    for px, py in cfg.pads():
        c0, c1 = np.clip(((np.array([px - reach, px + reach]) + half) / cell).astype(int), 0, n)
        r0, r1 = np.clip(((np.array([py - reach, py + reach]) + half) / cell).astype(int), 0, n)
        r = np.hypot(x[r0:r1, c0:c1] - px, y[r0:r1, c0:c1] - py)
        flat = 1 - np.clip((r - cfg.pad_flat_m) / cfg.pad_flat_m, 0, 1)
        pad[r0:r1, c0:c1] = np.maximum(pad[r0:r1, c0:c1], flat)
    return (z * (1 - pad)).astype(np.float32)


def rise_cm(z: np.ndarray, cells: int) -> np.ndarray:
    """Height change in cm between grid points `cells` apart along x or y (the larger)."""
    gx = np.abs(z[:, cells:] - z[:, :-cells])[:-cells or None, :]
    gy = np.abs(z[cells:, :] - z[:-cells, :])[:, :-cells or None]
    return 100 * np.maximum(gx, gy)


# ---- MuJoCo ------------------------------------------------------------------------------------
#
# Contact. MJX collides a box with a heightfield by testing it against every grid triangle its
# bounding sphere covers: 160 box-vs-prism tests per slab per substep at 5 cm cells, 57x the
# flat floor's cost (2026-11-03, Mac CPU). So on terrain each slab touches the ground through
# small spheres along its four long edges (corners included) instead of its box, and each
# sphere meets only the one triangle under its centre (`point_hfield`, one contact). On flat
# ground the corner spheres touch where the box's corners would, so a standing TARS gets the
# same 4 contacts per slab as on the plane. `max_contact_points` keeps the solver at the
# deepest few contacts however many spheres there are. The boxes still collide with each
# other as before; only box-vs-ground goes through the spheres. Cost: 1.6x the flat floor
# (same Mac CPU measurement); the grid spacing no longer matters.

CONTACT_RADIUS_M = 0.002
POINTS_PER_EDGE = 7  # along each slab's 36 cm long edges: 6 cm apart, the narrowest band's width
MAX_CONTACTS = 20  # a standing TARS needs 12 (4 corners x 3 slabs)
SLAB_GEOMS = ("central_geom", "left_arm_geom", "right_arm_geom")
_GROUND_BIT = 2  # contype/conaffinity bit shared only by the terrain and the contact spheres


def contact_points(half: np.ndarray, per_edge: int = POINTS_PER_EDGE) -> np.ndarray:
    """(4 * per_edge, 3) sphere centres in a slab's frame: along the 4 edges parallel to z,
    inset by the radius so each sphere's surface reaches the box's faces (corners exact)."""
    hx, hy, hz = np.asarray(half) - CONTACT_RADIUS_M
    zs = np.linspace(-hz, hz, per_edge)
    return np.array([(sx * hx, sy * hy, z) for sx in (-1, 1) for sy in (-1, 1) for z in zs])


def _add_contact_points(xml: str) -> str:
    import re

    for name in SLAB_GEOMS:
        found = re.findall(rf'<geom name="{name}" type="box" size="([^"]+)"[^>]*/>', xml, re.S)
        if len(found) != 1:
            raise ValueError(f"body MJCF layout changed: expected one box geom {name!r}")
        tag = re.search(rf'<geom name="{name}" type="box"[^>]*/>', xml, re.S).group(0)
        if "pos=" in tag or "quat=" in tag:
            raise ValueError(f"{name} is offset in its body; contact points assume it is not")
        pts = contact_points(np.array([float(v) for v in found[0].split()]))
        spheres = "".join(
            f'\n        <geom name="{name}_ground{i}" type="sphere" size="{CONTACT_RADIUS_M}" '
            f'pos="{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}" contype="{_GROUND_BIT}" conaffinity="0" '
            f'mass="0" group="4"/>' for i, p in enumerate(pts))
        xml = xml.replace(tag, tag + spheres)
    return xml


def add_to_mjcf(xml: str, cfg: TerrainConfig) -> str:
    """The arena with the heightfield as its floor (the plane keeps drawing, collides no more)
    and contact spheres on the slabs. Fill the heights with `fill_model` after compiling."""
    cfg.check()
    plane = '<geom name="floor" type="plane" size="0 0 0.05" material="grid"/>'
    if xml.count(plane) != 1 or xml.count("  </asset>") != 1 or xml.count("</mujoco>") != 1:
        raise ValueError("arena MJCF layout changed: expected one floor plane, one </asset> "
                         "and one </mujoco>")
    half = cfg.size_m / 2
    # size: x and y half-extents, elevation scale 1 (data in metres), base depth below z = 0
    hfield = (f'    <hfield name="{TERRAIN_GEOM}" nrow="{cfg.n}" ncol="{cfg.n}" '
              f'size="{half:.6g} {half:.6g} 1 0.1"/>\n')
    floor = ('<geom name="floor" type="plane" size="0 0 0.05" material="grid" '
             'contype="0" conaffinity="0" pos="0 0 -0.001"/>')
    # A body without joints, last in the world so no body ids shift. Not a world geom: CPU
    # MuJoCo never moves world geoms after compiling, so per-env placement would differ.
    body = (f'    <body name="{TERRAIN_GEOM}" pos="{_nominal_pos(cfg)}">\n'
            f'      <geom name="{TERRAIN_GEOM}" type="hfield" hfield="{TERRAIN_GEOM}" '
            f'contype="{_GROUND_BIT}" conaffinity="{_GROUND_BIT}" mass="0" material="grid"/>\n'
            f'    </body>\n  </worldbody>')
    custom = (f'  <custom>\n    <numeric name="max_contact_points" data="{MAX_CONTACTS}"/>\n'
              f'  </custom>\n</mujoco>')
    if xml.count("  </worldbody>") != 1:
        raise ValueError("arena MJCF layout changed: expected one </worldbody>")
    xml = xml.replace("  </asset>", hfield + "  </asset>").replace(plane, floor)
    xml = xml.replace("  </worldbody>", body)
    return _add_contact_points(xml).replace("</mujoco>", custom)


def point_hfield(h, s, subgrid_size=None):
    """MJX contact of a small sphere with the heightfield triangle under its centre.

    MuJoCo splits each grid cell (c, r) into the triangles (c,r)-(c+1,r+1)-(c,r+1) (upper left)
    and (c,r)-(c+1,r)-(c+1,r+1); the distance is the centre's height above that triangle's
    plane along its normal, minus the radius. Off the heightfield: no contact."""
    import jax.numpy as jnp
    from mujoco.mjx._src import math as mjx_math

    p = h.mat.T @ (s.pos - h.pos)
    dx = 2 * h.size[0] / (h.ncol - 1)
    dy = 2 * h.size[1] / (h.nrow - 1)
    fx, fy = (p[0] + h.size[0]) / dx, (p[1] + h.size[1]) / dy
    c = jnp.clip(jnp.floor(fx), 0, h.ncol - 2).astype(int)
    r = jnp.clip(jnp.floor(fy), 0, h.nrow - 2).astype(int)
    u, v = fx - c, fy - r
    z00, z10 = h.data[c, r] * h.size[2], h.data[c + 1, r] * h.size[2]
    z01, z11 = h.data[c, r + 1] * h.size[2], h.data[c + 1, r + 1] * h.size[2]
    upper = v >= u
    a = jnp.where(upper, z11 - z01, z10 - z00)  # rise per cell along x
    b = jnp.where(upper, z01 - z00, z11 - z10)  # rise per cell along y
    n = jnp.array([-a / dx, -b / dy, 1.0])
    n = n / jnp.linalg.norm(n)
    radius = s.size[0]
    dist = (p[2] - (z00 + a * u + b * v)) * n[2] - radius
    on = (fx >= 0) & (fx <= h.ncol - 1) & (fy >= 0) & (fy <= h.nrow - 1)
    dist = jnp.where(on, dist, 1.0)
    pos = p - n * (radius + dist / 2)
    return dist, h.mat @ pos + h.pos, mjx_math.make_frame(h.mat @ n)


def install_point_contact() -> None:
    """Make MJX collide spheres with heightfields through `point_hfield` (idempotent). Only the
    terrain's contact spheres meet a heightfield in this project."""
    from mujoco.mjx._src import collision_convex, collision_driver
    from mujoco.mjx._src.types import GeomType

    key = (GeomType.HFIELD, GeomType.SPHERE)
    if getattr(collision_driver._COLLISION_FUNC[key], "tars_point", False):
        return
    fn = collision_convex.collider(ncon=1)(point_hfield)
    fn.tars_point = True
    collision_driver._COLLISION_FUNC[key] = fn


def _nominal_pos(cfg: TerrainConfig) -> str:
    px, py = cfg.pads()[0]  # difficulty 0: flat
    return f"{-px:.6g} {-py:.6g} 0"


def fill_model(m, cfg: TerrainConfig) -> None:
    """Write the terrain's heights into a compiled MjModel (MuJoCo's row-major hfield data)."""
    import mujoco

    hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, TERRAIN_GEOM)
    if hid < 0:
        raise ValueError("model has no terrain heightfield: build it with terrain.add_to_mjcf")
    z = heights(cfg)
    adr, n = m.hfield_adr[hid], m.hfield_nrow[hid] * m.hfield_ncol[hid]
    m.hfield_data[adr:adr + n] = z.ravel()


def placements(cfg: TerrainConfig, keys) -> tuple:
    """Per env (pad index, yaw, terrain body pos (3,), quat (4,) wxyz), as jax arrays: the
    heightfield moved and turned so the env's pad sits under the origin."""
    import jax
    import jax.numpy as jnp

    pads = jnp.asarray(cfg.pads())

    def one(key):
        k_pad, k_yaw = jax.random.split(jax.random.fold_in(key, 7))
        i = jax.random.randint(k_pad, (), 0, pads.shape[0])
        yaw = jax.random.uniform(k_yaw, (), minval=-jnp.pi, maxval=jnp.pi)
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        p = pads[i]
        pos = -jnp.array([c * p[0] - s * p[1], s * p[0] + c * p[1], 0.0])
        quat = jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
        return i, yaw, pos, quat

    return jax.vmap(one)(keys)


def pad_difficulty(cfg: TerrainConfig, pad_index) -> np.ndarray:
    return cfg.difficulty(cfg.pads()[np.asarray(pad_index), 0])


# ---- eval --------------------------------------------------------------------------------------

def draw_png(cfg: TerrainConfig, z: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = cfg.size_m / 2
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(100 * z, origin="lower", extent=(-half, half, -half, half), cmap="terrain")
    pads = cfg.pads()
    ax.plot(pads[:, 0], pads[:, 1], "r.", ms=3, label="pads (TARS starts on one)")
    ax.set(title=f"terrain v{cfg.version}: rougher along x", xlabel="x (m)", ylabel="y (m)")
    fig.colorbar(im, label="height (cm)")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import jax
    import jax.numpy as jnp

    from tars_rl.env.config import EnvConfig
    from tars_rl.env.flag_env import FlagEnv
    from tars_rl.env.randomize import for_env

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--envs", type=int, default=64, help="pads to stand on")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--png", type=Path)
    a = p.parse_args(argv)
    cfg = TerrainConfig()
    t0 = time.perf_counter()
    z = heights(cfg)
    print(f"Terrain v{cfg.version}: {cfg.size_m:.0f} m square, {cfg.n}x{cfg.n} points every "
          f"{100 * cfg.cell_m:.0f} cm, {len(cfg.pads())} pads ({time.perf_counter() - t0:.1f} s)\n")
    ax = -cfg.size_m / 2 + cfg.cell_m * np.arange(cfg.n)
    d = cfg.difficulty(ax)
    metre = int(round(1.0 / cfg.cell_m))
    step, over_1m = rise_cm(z, 1), rise_cm(z, metre)
    print("Steps are the height change between neighbouring grid points (5 cm apart), slopes the\n"
          "change over 1 m. The test levels: tiles 0-3 cm, thresholds 1.5-3 cm, ramps 4-8 deg.\n")
    print(f"{'difficulty':<12}{'height p50':>11}{'p99':>6}{'max':>6}  {'step p99':>9}{'max':>6}"
          f"  {'slope over 1 m p99':>19}{'max':>6}")
    for lo in np.arange(0, 1.0, 0.25):
        cols = (d >= lo * cfg.difficulty_max) & (d <= (lo + 0.25) * cfg.difficulty_max)
        zz = 100 * z[:, cols]
        st = step[:, cols[:step.shape[1]]]
        sl = np.degrees(np.arctan(over_1m[:, cols[:over_1m.shape[1]]] / 100))
        print(f"{lo:.2f}-{lo + 0.25:.2f}   {np.median(zz):>8.1f} cm{np.percentile(zz, 99):>5.1f}"
              f"{zz.max():>6.1f}  {np.percentile(st, 99):>6.1f} cm{st.max():>6.1f}"
              f"  {np.percentile(sl, 99):>15.1f} deg{sl.max():>6.1f}")
    if a.png:
        draw_png(cfg, z, a.png)
        print(f"\npicture: {a.png}")

    flat_env = FlagEnv(config=EnvConfig(episode_seconds=a.seconds))
    env = FlagEnv(config=EnvConfig(episode_seconds=a.seconds, terrain=cfg))
    keys = jax.random.split(jax.random.PRNGKey(1), a.envs)
    model_v, axes = for_env(env, None)(env.mjx_model, keys)

    def stand(e):
        def run(model, key):
            old = e.mjx_model
            e.mjx_model = model
            try:
                zero = jnp.zeros(e.action_size)
                s = jax.lax.fori_loop(0, e.config.max_steps, lambda _, s: e.step(s, zero),
                                      e.reset(key))
                return s.fell, e._cos_tilt(s.data)
            finally:
                e.mjx_model = old
        return run

    run = jax.jit(jax.vmap(stand(env), in_axes=(axes, 0)))
    fell, cos_tilt = run(model_v, keys)
    tilt = np.degrees(np.arccos(np.clip(np.asarray(cos_tilt), -1, 1)))
    print(f"\nZero-action stand on {a.envs} random pads, {a.seconds:.0f} s: fell "
          f"{int(np.asarray(fell).sum())}, max tilt {tilt.max():.1f} deg")

    # physics cost: the same zero-action stand, flat floor vs terrain
    flat_run = jax.jit(jax.vmap(stand(flat_env), in_axes=(None, 0)))
    timings = {}
    for name, f, args in (("flat", flat_run, (flat_env.mjx_model, keys)),
                          ("terrain", run, (model_v, keys))):
        jax.block_until_ready(f(*args))
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        timings[name] = time.perf_counter() - t0
    steps = a.envs * env.config.max_steps
    print(f"MJX on {jax.default_backend()}: flat {steps / timings['flat']:,.0f} env-steps/s, "
          f"terrain {steps / timings['terrain']:,.0f} "
          f"({timings['terrain'] / timings['flat']:.1f}x the time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
