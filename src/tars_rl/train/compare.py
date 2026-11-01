"""Race trained policies: one eval table side by side and their learning curves on one time axis.

Run: python -m tars_rl.train.compare runs/full-seed0 runs/full-fasttd3 runs/full-fastsac \
         --episodes 64 --seed 0 --svg runs/race.svg

Each folder is scored by `train.eval` on the same flags (same seed), and its "curve" lines from
progress.jsonl are drawn against training wall-clock seconds, so PPO and the off-policy runs are
compared at equal time. The plot is a plain SVG (success rate and distance left vs wall time).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tars_rl.train.eval import evaluate

ROWS = (
    ("success rate", "success_rate", "{:.1%}"),
    ("dist left (m)", "final_dist_m", "{:.3f}"),
    ("dist closed (frac)", "closed_frac", "{:+.1%}"),
    ("fall rate", "fall_rate", "{:.1%}"),
    ("timeout rate", "timeout_rate", "{:.1%}"),
    ("episode length (s)", "length_s", "{:.2f}"),
)
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")


def table(results: list[dict], meta: list[dict]) -> str:
    names = [f"{r['algo']}:{Path(r['policy']).name}" for r in results]
    w = max(14, *(len(n) + 2 for n in names))
    lines = [f"{'':<22}" + "".join(f"{n:>{w}}" for n in names)]
    for label, key, fmt in ROWS:
        cells = (fmt.format(r["policies"]["trained"][key]) for r in results)
        lines.append(f"{label:<22}" + "".join(f"{c:>{w}}" for c in cells))
    lines.append(f"{'train wall (s)':<22}" + "".join(f"{m['train_wall_s']:>{w}.0f}" for m in meta))
    lines.append(f"{'env steps':<22}" + "".join(f"{m['num_timesteps']:>{w},}" for m in meta))
    lines.append(f"{'beats random+still':<22}" + "".join(
        f"{'YES' if r['beats_baselines'] else 'NO':>{w}}" for r in results))
    return "\n".join(lines)


def svg(curves: dict[str, list[dict]], path: str | Path) -> Path:
    """Two panels sharing the wall-clock x axis: success rate, distance left."""
    W, H, P = 720, 260, 50
    t_max = max((r["wall_s"] for c in curves.values() for r in c), default=1.0) or 1.0
    d_max = max((r["final_dist_m"] for c in curves.values() for r in c), default=1.0) or 1.0
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{2 * H + 40}" '
           'font-family="sans-serif" font-size="12">']
    for i, (key, label, y_max, fmt) in enumerate((
            ("success_rate", "success rate", 1.0, "{:.0%}"),
            ("final_dist_m", "distance left (m)", d_max, "{:.2f}"))):
        y0 = i * H + 10
        x = lambda t: P + (W - 2 * P) * t / t_max  # noqa: E731
        y = lambda v, y0=y0, y_max=y_max: y0 + H - P - (H - 2 * P + 10) * v / y_max  # noqa: E731
        out.append(f'<text x="{P}" y="{y0 + 12}">{label}</text>')
        out.append(f'<line x1="{P}" y1="{y(0)}" x2="{W - P}" y2="{y(0)}" stroke="black"/>')
        out.append(f'<line x1="{P}" y1="{y(0)}" x2="{P}" y2="{y(y_max)}" stroke="black"/>')
        out.append(f'<text x="{P - 4}" y="{y(y_max) + 4}" text-anchor="end">'
                   f'{fmt.format(y_max)}</text>')
        out.append(f'<text x="{W - P}" y="{y(0) + 16}" text-anchor="end">'
                   f'{t_max:.0f} s wall clock</text>')
        for j, (name, c) in enumerate(curves.items()):
            col = COLORS[j % len(COLORS)]
            pts = " ".join(f"{x(r['wall_s']):.1f},{y(r[key]):.1f}" for r in c)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
            if i == 0:
                out.append(f'<text x="{W - P + 4 - 150}" y="{y0 + 14 + 14 * j}" fill="{col}">'
                           f'{name}</text>')
    out.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("policies", nargs="+", help="policy folders (PPO or off-policy)")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--svg", default="runs/race.svg", help="learning-curve plot ('' to skip)")
    p.add_argument("--json", help="also write the results to this JSON file")
    a = p.parse_args(argv)
    results = [evaluate(d, a.episodes, a.seed) for d in a.policies]
    meta = [json.loads((Path(d) / "meta.json").read_text()) for d in a.policies]
    print(f"Trained policies, {a.episodes} episodes, seed {a.seed} (same flags for all)\n")
    print(table(results, meta))
    if a.svg:
        curves = {f"{r['algo']}:{Path(r['policy']).name}": r["curve"] for r in results}
        print(f"\nLearning curves on one wall-clock axis: {svg(curves, a.svg)}")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
