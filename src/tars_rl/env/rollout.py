"""Flag-arena rollout file: the Trajectory .npz (A10) plus where the flag stood (A51).

Same keys and version as `tars_rl.viz.Trajectory` (T9): version, qpos (frames, nq), dt,
body_spec_version. One extra key, flag_pos (3,), world frame on the floor; a Trajectory
loader ignores it, a flag-aware player sets the arena's flag mocap body to it. Optional key
flag_track (frames, 3): the flag in every frame, for chain mode (T26) where a tagged flag moves;
flag_pos is then the first flag, so a player that ignores flag_track still shows that one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRAJECTORY_VERSION = 1  # must match tars_rl.viz.trajectory.TRAJECTORY_VERSION (T9)


@dataclass(frozen=True)
class FlagRollout:
    qpos: np.ndarray  # (frames, nq)
    dt: float
    body_spec_version: int
    flag_pos: np.ndarray  # (3,)
    flag_track: np.ndarray | None = None  # (frames, 3), T26

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:  # file handle: np.savez would otherwise append ".npz"
            extra = ({} if self.flag_track is None
                     else {"flag_track": np.asarray(self.flag_track, dtype=np.float64)})
            np.savez(
                f,
                version=TRAJECTORY_VERSION,
                qpos=np.asarray(self.qpos, dtype=np.float64),
                dt=self.dt,
                body_spec_version=self.body_spec_version,
                flag_pos=np.asarray(self.flag_pos, dtype=np.float64),
                **extra,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FlagRollout:
        with np.load(path) as d:
            if int(d["version"]) != TRAJECTORY_VERSION:
                raise ValueError(f"rollout version {int(d['version'])} != {TRAJECTORY_VERSION}")
            return cls(
                qpos=d["qpos"], dt=float(d["dt"]), body_spec_version=int(d["body_spec_version"]),
                flag_pos=d["flag_pos"],
                flag_track=d["flag_track"] if "flag_track" in d.files else None,
            )
