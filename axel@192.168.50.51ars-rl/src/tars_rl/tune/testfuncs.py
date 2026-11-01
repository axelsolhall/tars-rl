"""Standard optimisation test functions with known best answers, as search spaces + objectives.

All are minimised. Domains of Ackley, Levy and Styblinski-Tang are the textbook ones shifted or
kept so the optimum is not at the middle of the space (the starting box is centred there).
`mixed` has a pick-one and an integer setting (A76).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from tars_rl.tune.space import Choice, Float, Int, Space


@dataclass(frozen=True)
class TestFunction:
    name: str
    space: Space
    f: Callable[[dict[str, Any]], float]
    f_opt: float
    x_opt: tuple[dict[str, Any], ...]  # every global optimum, as settings

    def __call__(self, settings: dict[str, Any]) -> float:
        return float(self.f(settings))

    def gap(self, value: float) -> float:
        return max(value - self.f_opt, 0.0)

    def dist_to_opt(self, u: np.ndarray) -> float:
        """Distance in unit coordinates from u to the nearest optimum."""
        return min(float(np.linalg.norm(u - self.space.encode(x))) for x in self.x_opt)


def _floats(prefix: str, d: int, low: float, high: float) -> tuple[Float, ...]:
    return tuple(Float(f"{prefix}{i}", low, high) for i in range(d))


def _vec(s: dict[str, Any], prefix: str = "x") -> np.ndarray:
    return np.array([v for k, v in s.items() if k.startswith(prefix)], dtype=float)


def _branin(s):
    x1, x2 = s["x0"], s["x1"]
    b, c, t = 5.1 / (4 * math.pi**2), 5 / math.pi, 1 / (8 * math.pi)
    return (x2 - b * x1**2 + c * x1 - 6) ** 2 + 10 * (1 - t) * math.cos(x1) + 10


_H3_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
_H3_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547],
                         [381, 5743, 8828]])
_H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
_H_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])


def _hartmann(a, p):
    return lambda s: -float(_H_ALPHA @ np.exp(-(a * (_vec(s) - p) ** 2).sum(1)))


def _rosenbrock(s):
    x = _vec(s)
    return float((100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum())


def _ackley(s):
    x = _vec(s)
    d = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt((x**2).sum() / d))
                 - np.exp(np.cos(2 * np.pi * x).sum() / d) + 20 + math.e)


def _levy(s):
    w = 1 + (_vec(s) - 1) / 4
    mid = ((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)).sum()
    end = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return float(np.sin(np.pi * w[0]) ** 2 + mid + end)


def _styblinski(s):
    x = _vec(s)
    return float(0.5 * (x**4 - 16 * x**2 + 5 * x).sum())


_MIX_SHIFT = np.array([1.5, -2.0, 0.5])
_MIX_KIND = {"a": 3.0, "b": 0.0, "c": 1.0, "d": 5.0}


def _mixed(s):
    return float(((_vec(s) - _MIX_SHIFT) ** 2).sum() + _MIX_KIND[s["kind"]]
                 + 0.3 * (s["n"] - 7) ** 2)


def _pt(*xs, **extra) -> dict[str, Any]:
    return {f"x{i}": v for i, v in enumerate(xs)} | extra


_ST = -2.903534


def suite() -> list[TestFunction]:
    return [
        TestFunction("branin", Space((Float("x0", -5, 10), Float("x1", 0, 15))), _branin,
                     0.397887, (_pt(-math.pi, 12.275), _pt(math.pi, 2.275), _pt(9.42478, 2.475))),
        TestFunction("hartmann3", Space(_floats("x", 3, 0, 1)), _hartmann(_H3_A, _H3_P),
                     -3.86278, (_pt(0.114614, 0.555649, 0.852547),)),
        TestFunction("hartmann6", Space(_floats("x", 6, 0, 1)), _hartmann(_H6_A, _H6_P),
                     -3.32237,
                     (_pt(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),)),
        TestFunction("rosenbrock4", Space(_floats("x", 4, -5, 10)), _rosenbrock, 0.0,
                     (_pt(1, 1, 1, 1),)),
        TestFunction("ackley5", Space(_floats("x", 5, -10, 30)), _ackley, 0.0,
                     (_pt(0, 0, 0, 0, 0),)),
        TestFunction("levy4", Space(_floats("x", 4, -5, 15)), _levy, 0.0, (_pt(1, 1, 1, 1),)),
        TestFunction("styblinski4", Space(_floats("x", 4, -5, 5)), _styblinski,
                     4 * _styblinski(_pt(_ST)), (_pt(_ST, _ST, _ST, _ST),)),
        TestFunction("mixed5",
                     Space(_floats("x", 3, -5, 5) + (Choice("kind", ("a", "b", "c", "d")),
                                                     Int("n", 1, 10))),
                     _mixed, 0.0, (_pt(*_MIX_SHIFT, kind="b", n=7),)),
    ]


def get(name: str) -> TestFunction:
    for f in suite():
        if f.name == name:
            return f
    raise KeyError(name)
