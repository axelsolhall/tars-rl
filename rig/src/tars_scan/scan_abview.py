"""A/B viewer for two scans: one camera drives both views, so moving in A moves B the same way
(Axel, 2026-11-13; Rerun cannot link two views' cameras). Self-contained local page, no CDN:
viewer/ab.html (plain WebGL2) plus one data/<side>_<snapshot>.js per mesh snapshot, loaded when
first shown (script tags work from file://).

  python -m tars_scan.scan_abview <out dir> A=<scan dir>[@level] B=<scan dir>[@level] [--open]

Keys in the page: F flips A/B in place at full width (blinking shows differences best), space
toggles split/flip, arrows step the snapshots, R resets the view. Red: points judged moving since
the previous snapshot; yellow: the camera path so far.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

import numpy as np

from .scan_mesh import drawn, load_snapshots, load_volume

Q = 0.005  # vertex quantisation (m): int16 covers +-163 m
TEMPLATE = Path(__file__).with_name("viewer") / "ab.html"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _q(P: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.round((P - center) / Q).astype(np.int16)


def build(out: Path, sides: dict[str, tuple[Path, float | None, int | None]], title: str) -> Path:
    """sides: name -> (scan dir, draw level or None = its own, min_faces or None = its own)."""
    (out / "data").mkdir(parents=True, exist_ok=True)
    loaded = {}
    for name, (d, level, min_faces) in sides.items():
        _, vol, prm = load_volume(d)
        loaded[name] = (
            load_snapshots(d / "snapshots.npz"),
            level if level is not None else prm["draw_level"],
            min_faces if min_faces is not None else prm["min_faces"],
            vol["path"],
            vol["path_t"],
            np.load(d / "dynamic.npz"),
        )
    n_snap = min(len(v[0]) for v in loaded.values())
    first = next(iter(loaded.values()))
    final = drawn(first[0][n_snap - 1], first[1], first[2])
    center = np.asarray(final.vertices).mean(0) if len(final.vertices) else np.zeros(3)
    extent = float(np.ptp(final.vertices, 0).max()) if len(final.vertices) else 5.0
    times = [float(s["t"]) for s in first[0][:n_snap]]
    for name, (snaps, level, mf, path, path_t, dyn) in loaded.items():
        for i in range(n_snap):
            m = drawn(snaps[i], level, mf)
            t, t_prev = times[i], times[i - 1] if i else -1.0
            sel = (dyn["t"] > t_prev) & (dyn["t"] <= t)
            d = {
                "v": _b64(_q(np.asarray(m.vertices), center)),
                "f": _b64(np.asarray(m.faces, np.uint32)),
                "c": _b64(np.asarray(m.visual.vertex_colors[:, :3], np.uint8)),
                "m": _b64(_q(dyn["xyz"][sel], center)),
                "pa": _b64(_q(path[path_t <= t], center)),
            }
            (out / "data" / f"{name}_{i}.js").write_text(
                f"__scan({json.dumps(f'{name}_{i}')},{json.dumps(d)})"
            )
    meta = {"names": list(sides), "times": times, "q": Q, "extent": extent}
    html = TEMPLATE.read_text().replace("__META__", json.dumps(meta)).replace("__TITLE__", title)
    page = out / "index.html"
    page.write_text(html)
    return page


def open_page(page: Path) -> None:
    subprocess.Popen(["open", str(page)])  # noqa: S603, S607  macOS: the default browser


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("sides", nargs=2, help="NAME=DIR[@LEVEL]")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    sides = {}
    for s in a.sides:
        name, rest = s.split("=", 1)
        d, _, lev = rest.partition("@")
        sides[name] = (Path(d), float(lev) if lev else None, None)
    page = build(Path(a.out), sides, f"A/B: {Path(a.out).name}")
    print(page)
    if a.open:
        open_page(page)


if __name__ == "__main__":
    main()
