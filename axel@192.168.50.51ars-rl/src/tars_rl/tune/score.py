"""A trial's score (A84, A106): success rate, distance closed and learning speed, blended.

  success  share of held-out flags reached (scored after training, flags no trial trained on)
  closed   mean share of the start distance closed on those flags, clipped to [-1, 1]
  speed    area under the learning curve, 0..1: the curve's success and closed (same 2:1 mix)
           averaged over training progress, so a run that gets good early scores higher
  total    0.6 * success + 0.3 * closed + 0.1 * speed: success most, speed least (A84)

A trial whose training fails or produces NaN gets FAILED, below any trial that ran.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

WEIGHTS = {"success": 0.6, "closed": 0.3, "speed": 0.1}
FAILED = -1.0  # lowest real total is 0.3 * -1 = -0.3


def _clip(x: float) -> float:
    x = float(x)
    return x if math.isnan(x) else max(-1.0, min(1.0, x))  # min/max would turn NaN into 1


def _quality(success: float, closed: float) -> float:
    ws, wc = WEIGHTS["success"], WEIGHTS["closed"]
    return (ws * success + wc * _clip(closed)) / (ws + wc)


def curve_speed(curve: list[dict]) -> float:
    """Normalised area under the curve's quality over steps (trapezoid, first to last point)."""
    pts = sorted((c["step"], _quality(c["success_rate"], c["closed_frac"])) for c in curve)
    if not pts:
        return 0.0
    if len(pts) == 1 or pts[-1][0] == pts[0][0]:
        return pts[-1][1]
    area = sum((s1 - s0) * (q0 + q1) / 2 for (s0, q0), (s1, q1) in zip(pts, pts[1:], strict=False))
    return area / (pts[-1][0] - pts[0][0])


@dataclass(frozen=True)
class Score:
    success: float
    closed: float
    speed: float
    total: float

    def to_dict(self) -> dict:
        return asdict(self) | {"weights": WEIGHTS}


def blend(success: float, closed: float, curve: list[dict]) -> Score:
    parts = {"success": float(success), "closed": _clip(closed), "speed": curve_speed(curve)}
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    if not math.isfinite(total):
        return Score(**{k: (v if math.isfinite(v) else float("nan")) for k, v in parts.items()},
                     total=FAILED)
    return Score(**parts, total=total)
