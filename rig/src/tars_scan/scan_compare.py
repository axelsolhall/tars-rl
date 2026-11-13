"""Side-by-side scan meshes for picking hyperparameters by eye (Axel, 2026-11-13: "a few meshes,
each with different HPs; I say which ones agree best with reality and we iterate").

  python -m tars_scan.scan_compare --out runs/scan/r1 A=runs/scan/walk2_p2 B=runs/scan/walk2 ...
                                   [--level 0.5] [--no-spawn]

Each NAME=DIR[@LEVEL] is a scan_mesh output dir (fuse + mesh already run) and the certainty level
to draw (default --level). One Rerun window, one 3D view per variant in a grid, each with the
camera path. Views are named by letter only, so the choice is made on the picture; the
hyperparameters that differ are printed and saved to <out>/variants.json.
Reads only scan_mesh's mesh_c<level>.ply and fuse.json (ScanVolume v1 outputs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_variant(spec: str, default_level: float) -> dict:
    name, rest = spec.split("=", 1)
    d, _, lev = rest.partition("@")
    level = float(lev) if lev else default_level
    fuse = json.loads((Path(d) / "fuse.json").read_text())
    params = dict(fuse["params"])
    params.setdefault("block_level", 0.0)  # fused before the sight check existed: it was off
    params["draw_level"] = level
    return {"name": name, "dir": d, "level": level, "params": params}


def differing(variants: list[dict]) -> list[str]:
    keys = sorted({k for v in variants for k in v["params"]})
    return [k for k in keys if len({json.dumps(v["params"].get(k)) for v in variants}) > 1]


def table(variants: list[dict], keys: list[str]) -> str:
    rows = [["variant", *keys]] + [
        [v["name"], *(f"{v['params'].get(k)}" for k in keys)] for v in variants
    ]
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def line(left, mid, right):
        return left + mid.join("─" * (x + 2) for x in w) + right

    out = [line("┌", "┬", "┐")]
    for j, r in enumerate(rows):
        out.append("│" + "│".join(f" {c:<{w[i]}} " for i, c in enumerate(r)) + "│")
        if j == 0:
            out.append(line("├", "┼", "┤"))
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    import rerun as rr
    import rerun.blueprint as rrb
    import trimesh

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("variants", nargs="+", help="NAME=DIR[@LEVEL]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=float, default=0.5)
    ap.add_argument("--no-spawn", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vs = [load_variant(s, a.level) for s in a.variants]
    keys = differing(vs)
    print(table(vs, keys))
    (out / "variants.json").write_text(json.dumps(vs, indent=2))

    bp = rrb.Blueprint(
        rrb.Grid(*[rrb.Spatial3DView(origin=f"/{v['name']}", name=v["name"]) for v in vs]),
        collapse_panels=True,
    )
    rr.init("tars_scan_compare", recording_id=out.name)
    rrd = out / "compare.rrd"
    rr.save(str(rrd), default_blueprint=bp)
    for v in vs:
        m = trimesh.load(Path(v["dir"]) / f"mesh_c{v['level']:g}.ply", process=False)
        path = np.load(Path(v["dir"]) / "volume.npz")["path"]
        n = v["name"]
        rr.log(f"/{n}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            f"/{n}/mesh",
            rr.Mesh3D(
                vertex_positions=m.vertices,
                triangle_indices=m.faces,
                vertex_normals=m.vertex_normals,
                vertex_colors=m.visual.vertex_colors[:, :3],
            ),
            static=True,
        )
        rr.log(f"/{n}/camera_path", rr.LineStrips3D([path], colors=[(255, 200, 0)]), static=True)
    print(rrd)
    if not a.no_spawn:
        import subprocess
        import sys

        exe = Path(sys.executable).with_name("rerun")
        subprocess.Popen([str(exe), str(rrd)])  # noqa: S603  the viewer outlives this script


if __name__ == "__main__":
    main()
