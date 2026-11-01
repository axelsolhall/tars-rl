"""Trajectory: a persisted rollout, the contract between sim/training and every viewer (A10).

One .npz per rollout: per-frame qpos, the simulation timestep between frames and the BodySpec
version the rollout was recorded with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRAJECTORY_VERSION = 1


@dataclass(frozen=True)
class Trajectory:
    qpos: np.ndarray  # (frames, nq)
    dt: float  # seconds between consecutive frames
    body_spec_version: int
    version: int = TRAJECTORY_VERSION

    def __post_init__(self) -> None:
        if self.qpos.ndim != 2 or len(self.qpos) == 0:
            raise ValueError(f"qpos must be (frames, nq) with frames > 0, got {self.qpos.shape}")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")

    @property
    def n_frames(self) -> int:
        return len(self.qpos)

    @property
    def duration(self) -> float:
        """Simulated seconds covered: n_frames steps of dt."""
        return self.n_frames * self.dt

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:  # file handle: np.savez would otherwise append ".npz"
            np.savez(
                f,
                version=self.version,
                qpos=self.qpos,
                dt=self.dt,
                body_spec_version=self.body_spec_version,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        with np.load(path) as d:
            version = int(d["version"])
            if version != TRAJECTORY_VERSION:
                raise ValueError(f"Trajectory version {version} != {TRAJECTORY_VERSION}")
            return cls(
                qpos=d["qpos"],
                dt=float(d["dt"]),
                body_spec_version=int(d["body_spec_version"]),
                version=version,
            )
