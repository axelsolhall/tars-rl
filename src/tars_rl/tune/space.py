"""Search space: named settings encoded into a unit cube, and the moving box inside it.

Every setting maps to one coordinate u in [0, 1]:
  Float  linear or log scale between low and high
  Int    like Float, rounded to an integer
  Choice u split into equal bins, one per option (A76: same box as the numbers)
The space's [0, 1]^d is the hard limit; the search box is a smaller region inside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Float:
    name: str
    low: float
    high: float
    log: bool = False

    def decode(self, u: float) -> float:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(lo + u * (hi - lo)))
        return float(self.low + u * (self.high - self.low))

    def encode(self, v: Any) -> float:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return (math.log(float(v)) - lo) / (hi - lo)
        return (float(v) - self.low) / (self.high - self.low)


@dataclass(frozen=True)
class Int:
    name: str
    low: int
    high: int
    log: bool = False

    def decode(self, u: float) -> int:
        return int(round(Float(self.name, self.low, self.high, self.log).decode(u)))

    def encode(self, v: Any) -> float:
        return Float(self.name, self.low, self.high, self.log).encode(v)


@dataclass(frozen=True)
class Choice:
    name: str
    options: tuple[Any, ...]

    def decode(self, u: float) -> Any:
        k = len(self.options)
        return self.options[min(int(u * k), k - 1)]

    def encode(self, v: Any) -> float:
        return (self.options.index(v) + 0.5) / len(self.options)  # centre of its bin


Param = Float | Int | Choice


@dataclass(frozen=True)
class Space:
    params: tuple[Param, ...]

    def __post_init__(self):
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate setting names: {names}")

    @property
    def dim(self) -> int:
        return len(self.params)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    def decode(self, u: np.ndarray) -> dict[str, Any]:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        return {p.name: p.decode(float(x)) for p, x in zip(self.params, u, strict=True)}

    def encode(self, settings: dict[str, Any]) -> np.ndarray:
        return np.clip([p.encode(settings[p.name]) for p in self.params], 0.0, 1.0)


@dataclass
class Box:
    """Axis-aligned box in unit coordinates: centre and side length (same in every dim)."""

    centre: np.ndarray
    length: float
    lows: np.ndarray = field(init=False)
    highs: np.ndarray = field(init=False)

    def __post_init__(self):
        self.centre = np.clip(np.asarray(self.centre, dtype=float), 0.0, 1.0)
        half = self.length / 2
        # Clipped to the hard limits: near an edge the box is cut, not shifted.
        self.lows = np.clip(self.centre - half, 0.0, 1.0)
        self.highs = np.clip(self.centre + half, 0.0, 1.0)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(self.lows, self.highs, size=(n, self.centre.size))

    def to_dict(self) -> dict:
        return {"centre": self.centre.tolist(), "length": self.length}
