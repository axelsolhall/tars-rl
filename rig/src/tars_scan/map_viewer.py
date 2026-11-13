"""A map as one self-contained HTML page you can turn around (no CDN, no library).

  python -m tars_scan.map_viewer <points.npz|.npy> <trajectory.txt> <out.html> [--title T]

Points come from occupancy_map (`centers` in its .npz) or voxel_map (.npy). They are stored as
int16 millimetres around the middle of the camera path and drawn on a 2D canvas with a per-pixel
depth buffer, so the page opens offline in any browser (a three.js page from a CDN rendered blank
on Axel's machine, 2026-11-10). Colour is height relative to the camera path.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).with_name("viewer") / "map3d.html"


def page(points: np.ndarray, path: np.ndarray, title: str, voxel_cm: float) -> str:
    cam = np.median(path[:, 2])
    keep = (points[:, 2] - cam > -1.6) & (points[:, 2] - cam < 1.0)
    points = points[keep]
    c0 = np.r_[np.median(path[:, :2], 0), cam]

    def pack(a):
        return base64.b64encode(np.round((a - c0) * 1000).astype("<i2").tobytes()).decode()

    return (
        TEMPLATE.read_text()
        .replace("__TITLE__", title)
        .replace("__VOX__", f"{voxel_cm:g} cm")
        .replace("__N__", f"{len(points):,}")
        .replace("__PTS__", pack(points))
        .replace("__TRJ__", pack(path))
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tars_scan.map_viewer", description=__doc__.split("\n")[0])
    ap.add_argument("points", type=Path)
    ap.add_argument("trajectory", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--title", default="Walk map")
    ap.add_argument("--voxel-cm", type=float, default=3.0)
    a = ap.parse_args(argv)
    pts = np.load(a.points)
    pts = pts["centers"] if a.points.suffix == ".npz" else pts
    path = np.loadtxt(a.trajectory)[:, 1:4]
    a.out.write_text(page(np.asarray(pts, float), path, a.title, a.voxel_cm))
    print(f"file://{a.out.resolve()}")


if __name__ == "__main__":
    main()
