"""Flag-arena settings: episode rules and reward weights in one place (A18, A22-A24)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    """Per-step reward = progress + upright - fall - effort (A18). Weights only need to run."""

    progress: float = 1.0  # per metre of flag distance closed this step
    upright: float = 0.02  # times cos(tilt) of the central body, every step
    fall: float = 1.0  # once, on the step the episode ends as a fall
    effort: float = 0.005  # times mean(action**2), action in [-1, 1]


@dataclass(frozen=True)
class EnvConfig:
    """Lengths are in body heights (the central slab's height) so a resized body keeps the rules."""

    ctrl_dt: float = 0.02  # seconds per policy step; physics substeps = ctrl_dt / timestep
    episode_seconds: float = 20.0
    flag_min_heights: float = 3.0  # A22
    flag_max_heights: float = 10.0  # A22
    success_heights: float = 1.0  # A23: centre within this many heights of the flag (xy)
    fall_tilt_rad: float = math.radians(60.0)  # A24
    reward: RewardConfig = field(default_factory=RewardConfig)
    # Senses (T24): which observation set, see `sensors.OBS_SETS`.
    obs: str = "default"
    touch_threshold_n: float = 0.1  # an edge reads 1 when its normal contact force exceeds this
    # Sim-to-real knobs, off by default: channel name -> Gaussian std in the channel's units
    # (flip probability for the binary touch channel), and 0 or 1 policy step of sensor delay.
    sensor_noise: dict[str, float] = field(default_factory=dict)
    sensor_delay_steps: int = 0

    @property
    def max_steps(self) -> int:
        return round(self.episode_seconds / self.ctrl_dt)

    def to_dict(self) -> dict:
        return asdict(self)
