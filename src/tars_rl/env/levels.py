"""Levels: the world TARS walks in, as plain boxes both engines build identically.

A `Level` is a list of boxes on top of the flat floor (z = 0), in MuJoCo's frame (m, z up,
x forward, y left), plus where TARS starts and where flags may go. MuJoCo adds the boxes as
static geoms (`add_to_mjcf`); Unreal builds the same boxes from the bridge's `level` message
(`tars_rl.unreal.bridge`). Boxes only, so neither engine needs meshes or heightfields, and a
level is fully described by a few hundred numbers.

    flat        nothing: the training arena, unchanged (MuJoCo and Unreal must agree here first)
    bumpy       uneven floor: tiles 0-3 cm high (the "+-1.5 cm" of an old floor)
    steps       rugs (1 cm) and door thresholds (1.5-3 cm) across the floor
    slopes      ramps of 4-8 degrees up onto 5-10 cm plateaus and down again
    obstacles   flat floor with boxes in the way (furniture feet, toys, bins)
    apartment   two rooms and a door: walls, threshold, rug, sofa, table and chair legs, bed

The motion policy is blind (docs/ROADMAP.md): obstacles are the decider's job. So a flag is only
valid when the straight walk to it is clear of solid boxes (walls, furniture) by a margin; flags
across rugs, steps and ramps are fine. On `flat` the first draw is always valid, so the flags are
exactly the ones `FlagEnv` draws.

Eval: `python -m tars_rl.env.levels` prints every level's box count, height range and how often a
flag draw is valid; `--png DIR` also draws each level from above.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

LEVEL_VERSION = 1
SOLID_KINDS = ("wall", "furniture", "obstacle")  # the blind policy cannot cross these
WALK_KINDS = ("tile", "rug", "threshold", "ramp", "plateau")
FLAG_CLEARANCE_M = 0.2  # straight path to a flag keeps this far from solid boxes
TALL_ENOUGH_M = 0.5  # a box whose bottom is above this (a table top) is passable underneath
COLORS = {
    "tile": (0.42, 0.45, 0.5), "rug": (0.55, 0.25, 0.22), "threshold": (0.6, 0.48, 0.32),
    "ramp": (0.38, 0.5, 0.42), "plateau": (0.38, 0.5, 0.42), "wall": (0.85, 0.84, 0.8),
    "furniture": (0.35, 0.3, 0.27), "obstacle": (0.75, 0.55, 0.15),
}


@dataclass(frozen=True)
class Box:
    centre: tuple[float, float, float]  # m, world
    size: tuple[float, float, float]  # full extents along the box's own x, y, z, m
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w x y z
    kind: str = "obstacle"
    friction: float = 0.8

    @property
    def solid(self) -> bool:
        return self.kind in SOLID_KINDS and self.bottom_z < TALL_ENOUGH_M

    @property
    def bottom_z(self) -> float:
        return float(np.min(self.corners()[:, 2]))

    def rot(self) -> np.ndarray:
        return quat_to_mat(np.asarray(self.quat))

    def corners(self) -> np.ndarray:
        half = np.asarray(self.size) / 2
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return np.asarray(self.centre) + (signs * half) @ self.rot().T

    def top_z_at(self, xy: np.ndarray) -> float | None:
        """Height of the top face above floor point `xy`, or None when `xy` is off the box."""
        r = self.rot()
        local = r.T @ (np.array([xy[0], xy[1], 0.0]) - np.asarray(self.centre))
        n = r[:, 2]  # top-face normal
        if abs(n[2]) < 1e-6:
            return None
        # walk along world z from the floor point onto the top-face plane
        top = np.asarray(self.centre) + n * self.size[2] / 2
        z = top[2] - (n[0] * (xy[0] - top[0]) + n[1] * (xy[1] - top[1])) / n[2]
        local = r.T @ (np.array([xy[0], xy[1], z]) - np.asarray(self.centre))
        if abs(local[0]) > self.size[0] / 2 or abs(local[1]) > self.size[1] / 2:
            return None
        return float(z)


@dataclass(frozen=True)
class Level:
    name: str
    seed: int
    boxes: tuple[Box, ...] = ()
    spawn_xy: tuple[float, float] = (0.0, 0.0)
    bounds: tuple[float, float, float, float] | None = None  # flags inside x0, x1, y0, y1
    description: str = ""
    version: int = LEVEL_VERSION
    stats: dict = field(default_factory=dict, compare=False)

    def ground_z(self, xy) -> float:
        """Floor height at `xy`: the highest walkable top there, 0 on the bare floor."""
        tops = [z for b in self.boxes if b.kind in WALK_KINDS
                if (z := b.top_z_at(np.asarray(xy, float))) is not None]
        return max([0.0, *tops])

    def path_clear(self, start_xy, end_xy, clearance: float = FLAG_CLEARANCE_M) -> bool:
        """The straight walk start -> end stays `clearance` away from every solid box (xy)."""
        a, b = np.asarray(start_xy, float), np.asarray(end_xy, float)
        n = max(2, int(np.linalg.norm(b - a) / 0.05) + 1)
        pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
        for box in self.boxes:
            if box.solid and _min_dist_xy(box, pts) < clearance:
                return False
        return True

    def flag_ok(self, from_xy, flag_xy) -> bool:
        if self.bounds is not None:
            x0, x1, y0, y1 = self.bounds
            if not (x0 <= flag_xy[0] <= x1 and y0 <= flag_xy[1] <= y1):
                return False
        return self.path_clear(from_xy, flag_xy)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("stats")
        return d


# ---- geometry helpers -------------------------------------------------------------------------
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_pitch_quat(yaw: float, pitch: float = 0.0) -> tuple[float, float, float, float]:
    """Turn by `yaw` about world z, then tilt by `pitch` about the turned y axis (w x y z)."""
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def _min_dist_xy(box: Box, pts: np.ndarray) -> float:
    """Smallest xy distance from points to the box's footprint (an upright box: its rectangle)."""
    r = box.rot()
    yaw = math.atan2(r[1, 0], r[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    d = pts - np.asarray(box.centre[:2])
    local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=-1)
    out = np.maximum(np.abs(local) - np.asarray(box.size[:2]) / 2, 0.0)
    return float(np.min(np.linalg.norm(out, axis=-1)))


def _slab(x: float, y: float, sx: float, sy: float, h: float, kind: str, yaw: float = 0.0,
          z0: float = 0.0) -> Box:
    """A box standing on height z0 with its top at z0 + h. Thin floor pieces reach 5 cm below
    the floor so no engine ever sees a sliver."""
    bottom = z0 - (0.05 if z0 == 0.0 and kind in WALK_KINDS else 0.0)
    height = z0 + h - bottom
    return Box((x, y, bottom + height / 2), (sx, sy, height), yaw_pitch_quat(yaw), kind)


def _ramp(x: float, y: float, yaw: float, length: float, width: float, rise: float) -> Box:
    """A ramp whose top face climbs `rise` over horizontal `length`, starting at (x, y) on the
    floor and heading along `yaw`."""
    angle = math.atan2(rise, length)
    t = 0.05  # thickness
    slant = math.hypot(length, rise)
    d = np.array([math.cos(yaw), math.sin(yaw)])
    top_mid = np.array([*(np.array([x, y]) + d * length / 2), rise / 2])
    normal = np.array([-math.sin(angle) * d[0], -math.sin(angle) * d[1], math.cos(angle)])
    c = top_mid - normal * t / 2
    return Box(tuple(c), (slant, width, t), yaw_pitch_quat(yaw, -angle), "ramp")


def _far_from(xy, spawn, radius: float) -> bool:
    return math.dist(xy, spawn) > radius


# ---- generators -------------------------------------------------------------------------------
def flat(seed: int = 0) -> Level:
    return Level("flat", seed, description="the training arena: an endless flat floor")


def bumpy(seed: int = 0, amplitude: float = 0.015, tile: float = 0.3, half: float = 6.0) -> Level:
    rng = np.random.default_rng(seed)
    boxes = []
    n = int(2 * half / tile)
    for i in range(n):
        for j in range(n):
            x, y = -half + (i + 0.5) * tile, -half + (j + 0.5) * tile
            if max(abs(x), abs(y)) < 0.45:  # start on the bare floor
                continue
            boxes.append(_slab(x, y, tile, tile, float(rng.uniform(0, 2 * amplitude)), "tile"))
    m = half - 0.5
    return Level("bumpy", seed, tuple(boxes), bounds=(-m, m, -m, m),
                 description=f"{tile * 100:.0f} cm tiles 0-{2 * amplitude * 100:.0f} cm high")


def steps(seed: int = 0, half: float = 6.0, rugs: int = 6, thresholds: int = 10) -> Level:
    rng = np.random.default_rng(seed)
    boxes = []
    while len(boxes) < rugs:
        xy = rng.uniform(-half + 1, half - 1, 2)
        if _far_from(xy, (0, 0), 1.2):
            boxes.append(_slab(*xy, rng.uniform(1.0, 2.2), rng.uniform(0.6, 1.5), 0.01, "rug",
                               rng.uniform(0, math.pi)))
    while len(boxes) < rugs + thresholds:
        xy = rng.uniform(-half + 1, half - 1, 2)
        if _far_from(xy, (0, 0), 0.8):
            boxes.append(_slab(*xy, 0.08, rng.uniform(0.8, 2.0), rng.uniform(0.015, 0.03),
                               "threshold", rng.uniform(0, math.pi)))
    m = half - 0.5
    return Level("steps", seed, tuple(boxes), bounds=(-m, m, -m, m),
                 description="1 cm rugs and 1.5-3 cm thresholds")


def slopes(seed: int = 0, half: float = 6.0, mounds: int = 6) -> Level:
    """Each mound: ramp up, a plateau, ramp down, in a random direction."""
    rng = np.random.default_rng(seed)
    boxes: list[Box] = []
    centres: list[np.ndarray] = []
    tries = 0
    while len(centres) < mounds and tries < 1000:
        tries += 1
        xy = rng.uniform(-half + 1.5, half - 1.5, 2)
        if not _far_from(xy, (0, 0), 1.8) or any(math.dist(xy, c) < 2.6 for c in centres):
            continue
        centres.append(xy)
        yaw = rng.uniform(0, 2 * math.pi)
        angle = math.radians(rng.uniform(4, 8))
        rise = rng.uniform(0.05, 0.10)
        run = rise / math.tan(angle)
        top, width = rng.uniform(0.6, 1.0), rng.uniform(1.0, 1.6)
        d = np.array([math.cos(yaw), math.sin(yaw)])
        start = xy - d * (run + top / 2)
        boxes.append(_ramp(*start, yaw, run, width, rise))
        boxes.append(_slab(*xy, top, width, rise, "plateau", yaw))
        end = xy + d * (run + top / 2)
        boxes.append(_ramp(*end, yaw + math.pi, run, width, rise))
    m = half - 0.5
    return Level("slopes", seed, tuple(boxes), bounds=(-m, m, -m, m),
                 description="4-8 degree ramps onto 5-10 cm plateaus")


def obstacles(seed: int = 0, half: float = 5.0, count: int = 18) -> Level:
    rng = np.random.default_rng(seed)
    boxes = []
    while len(boxes) < count:
        xy = rng.uniform(-half + 0.5, half - 0.5, 2)
        if _far_from(xy, (0, 0), 1.0):
            boxes.append(_slab(*xy, rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5),
                               rng.uniform(0.1, 0.6), "obstacle", rng.uniform(0, math.pi)))
    m = half - 0.3
    return Level("obstacles", seed, tuple(boxes), bounds=(-m, m, -m, m),
                 description="boxes 10-50 cm wide, 10-60 cm tall; flags only on clear lines")


def apartment(seed: int = 0) -> Level:
    """Living room (x < 0) and bedroom (x > 0), 8 x 6 m, joined by a door with a threshold.
    Furniture is fixed; the seed only nudges it so levels differ between seeds."""
    rng = np.random.default_rng(seed)
    j = lambda: float(rng.uniform(-0.15, 0.15))  # noqa: E731 - a small nudge
    wall, h, t = "wall", 2.4, 0.1
    b: list[Box] = [
        _slab(0, 3.05, 8.2, t, h, wall), _slab(0, -3.05, 8.2, t, h, wall),
        _slab(-4.05, 0, t, 6.0, h, wall), _slab(4.05, 0, t, 6.0, h, wall),
        # interior wall at x = 0 with a 0.9 m door at y in [0.4, 1.3]
        _slab(0, -1.3, t, 3.4, h, wall), _slab(0, 2.175, t, 1.65, h, wall),
        _slab(0, 0.85, 0.12, 0.9, 0.018, "threshold"),
        # living room: rug, sofa, coffee table on legs, a chair on legs
        _slab(-2.0 + j(), 0.0 + j(), 2.0, 1.4, 0.012, "rug"),
        _slab(-2.0 + j(), -2.45, 2.0, 0.85, 0.45, "furniture"),
    ]

    def legged(cx: float, cy: float, sx: float, sy: float, top_z: float) -> list[Box]:
        legs = [_slab(cx + ex * (sx / 2 - 0.04), cy + ey * (sy / 2 - 0.04), 0.05, 0.05, top_z,
                      "furniture") for ex in (-1, 1) for ey in (-1, 1)]
        return legs + [_slab(cx, cy, sx, sy, 0.03, "furniture", z0=top_z)]

    b += legged(-2.0 + j(), 0.2 + j(), 1.1, 0.6, 0.42)  # coffee table: too low to walk under
    b += legged(-3.3 + j(), 2.2 + j(), 0.45, 0.45, 0.45)  # chair
    b += legged(-1.0 + j(), 2.3 + j(), 1.2, 0.8, 0.72)  # dining table: TARS walks under it
    # bedroom: bed, wardrobe, a box on the floor
    b += [
        _slab(2.6 + j(), 1.9, 1.6, 2.0, 0.5, "furniture"),
        _slab(3.7, -1.8 + j(), 0.6, 1.2, 2.0, "furniture"),
        _slab(1.6 + j(), -1.6 + j(), 0.4, 0.3, 0.25, "obstacle"),
    ]
    return Level("apartment", seed, tuple(b), spawn_xy=(-2.6, -0.8),
                 bounds=(-3.8, 3.8, -2.8, 2.8),
                 description="two rooms, a door with an 1.8 cm threshold, rug, furniture")


LEVELS = {f.__name__: f for f in (flat, bumpy, steps, slopes, obstacles, apartment)}


def make_level(name: str, seed: int = 0) -> Level:
    if name not in LEVELS:
        raise ValueError(f"unknown level {name!r}; pick one of {sorted(LEVELS)}")
    return LEVELS[name](seed)


def add_to_mjcf(xml: str, level: Level) -> str:
    """The arena MJCF with the level's boxes as static world geoms (same friction rule as the
    body: MuJoCo takes the larger of the two frictions in a contact)."""
    if not level.boxes:
        return xml
    anchor = "  </worldbody>"
    if xml.count(anchor) != 1:
        raise ValueError("arena MJCF layout changed: expected one '</worldbody>'")
    geoms = []
    for i, b in enumerate(level.boxes):
        r, g, bl = COLORS.get(b.kind, (0.5, 0.5, 0.5))
        geoms.append(
            f'    <geom name="level_{i}_{b.kind}" type="box" '
            f'pos="{" ".join(f"{v:.6g}" for v in b.centre)}" '
            f'size="{" ".join(f"{v / 2:.6g}" for v in b.size)}" '
            f'quat="{" ".join(f"{v:.8g}" for v in b.quat)}" '
            f'friction="{b.friction:.6g} 0.005 0.0001" rgba="{r} {g} {bl} 1"/>\n')
    return xml.replace(anchor, "".join(geoms) + anchor)


# ---- eval -------------------------------------------------------------------------------------
def flag_validity(level: Level, draws: int = 400, seed: int = 0, min_r: float = 1.08,
                  max_r: float = 3.6) -> float:
    """Share of FlagEnv-style draws (3-10 body heights around the spawn) that are valid."""
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(draws):
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(min_r, max_r)
        f = np.asarray(level.spawn_xy) + r * np.array([math.cos(a), math.sin(a)])
        ok += level.flag_ok(level.spawn_xy, f)
    return ok / draws


def draw_png(level: Level, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(6, 6))
    for b in sorted(level.boxes, key=lambda b: b.centre[2]):
        top = b.corners()[1::2, :2]  # the four corners with local z = +half: the top face
        hull = top[np.argsort(np.arctan2(*(top - top.mean(0)).T[::-1]))]
        ax.add_patch(Polygon(hull, color=COLORS.get(b.kind, (0.5, 0.5, 0.5)), alpha=0.85))
    ax.plot(*level.spawn_xy, "c^", ms=10, label="TARS start")
    if level.bounds:
        x0, x1, y0, y1 = level.bounds
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", lw=0.8, label="flag area")
    lim = 6.5
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", title=f"{level.name}: "
           f"{level.description}", xlabel="x (m)", ylabel="y (m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", type=Path, help="write a top-down picture of each level here")
    a = p.parse_args(argv)
    print(f"{'level':<11}{'boxes':>7}{'solid':>7}{'top (cm)':>10}{'flags ok':>10}  description")
    for name in LEVELS:
        lv = make_level(name, a.seed)
        tops = [max(c[2] for c in b.corners()) for b in lv.boxes if b.kind in WALK_KINDS]
        top = f"{100 * max(tops):.1f}" if tops else "0"
        ok = flag_validity(lv, seed=a.seed)
        print(f"{name:<11}{len(lv.boxes):>7}{sum(b.solid for b in lv.boxes):>7}{top:>10}"
              f"{ok:>10.0%}  {lv.description}")
        if a.png:
            a.png.mkdir(parents=True, exist_ok=True)
            draw_png(lv, a.png / f"{name}.png")
    if a.png:
        print(f"\npictures in {a.png}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
