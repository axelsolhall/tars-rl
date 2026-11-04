"""Flag-arena settings: episode rules and reward weights in one place (A18, A22-A24)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from tars_rl.env.terrain import TerrainConfig

FLAG_MODES = ("chain", "single")


@dataclass(frozen=True)
class RewardConfig:
    """Per-step reward = progress + upright - fall - effort (A18). Weights only need to run."""

    progress: float = 1.0  # per metre of flag distance closed this step
    upright: float = 0.02  # times cos(tilt) of the central body, every step
    fall: float = 1.0  # once, on the step the episode ends as a fall
    effort: float = 0.005  # times mean(action**2), action in [-1, 1]
    tag: float = 0.0  # once per flag tagged; 0: the next flag's progress is the incentive (T26)


@dataclass(frozen=True)
class EnvConfig:
    """Lengths are in body heights (the central slab's height) so a resized body keeps the rules."""

    ctrl_dt: float = 0.02  # seconds per policy step; physics substeps = ctrl_dt / timestep
    episode_seconds: float = 20.0
    flag_min_heights: float = 3.0  # A22
    flag_max_heights: float = 10.0  # A22
    success_heights: float = 1.0  # A23: centre within this many heights of the flag (xy)
    # T26: "chain" = a tagged flag respawns 3-10 heights around TARS and the episode goes on
    # (ends on a fall or timeout); "single" = the benchmark's one flag, tagging ends the episode.
    flags: str = "chain"
    fall_tilt_rad: float = math.radians(60.0)  # A24
    reward: RewardConfig = field(default_factory=RewardConfig)
    # Senses (T24): which observation set, see `sensors.OBS_SETS`.
    obs: str = "default"
    touch_threshold_n: float = 0.1  # an edge reads 1 when its normal contact force exceeds this
    # Sim-to-real knobs, off by default: channel name -> Gaussian std in the channel's units
    # (flip probability for the binary touch channel), and 0 or 1 policy step of sensor delay.
    sensor_noise: dict[str, float] = field(default_factory=dict)
    sensor_delay_steps: int = 0
    # Memory: the observation is the last this-many frames (0 = the current frame only). Set from
    # the network description's `memory=<kind>-<steps>` by training, so env and network agree.
    history_steps: int = 0
    # Rough ground (stage 3, `tars_rl.env.terrain`): None = the flat floor.
    terrain: TerrainConfig | None = None
    # MuJoCo's friction cone. "elliptic" lets `DomainRandConfig.impratio` harden friction per env
    # (2026-11-04: pyramidal friction creeps 0.17 cm/s under a 10 N push that Bullet holds still).
    friction_cone: str = "pyramidal"

    def __post_init__(self) -> None:
        if self.flags not in FLAG_MODES:
            raise ValueError(f"flags must be one of {FLAG_MODES}, got {self.flags!r}")
        if self.friction_cone not in ("pyramidal", "elliptic"):
            raise ValueError(
                f"friction_cone must be pyramidal or elliptic, got {self.friction_cone!r}")

    @property
    def max_steps(self) -> int:
        return round(self.episode_seconds / self.ctrl_dt)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, **overrides) -> EnvConfig:
        """Inverse of `to_dict` (a saved policy's meta["env"]); runs saved before a field
        existed get its default."""
        d = {**d, **overrides}
        d["reward"] = RewardConfig(**d["reward"]) if isinstance(d.get("reward"), dict) else \
            d.get("reward", RewardConfig())
        if isinstance(d.get("terrain"), dict):
            d["terrain"] = TerrainConfig.from_dict(d["terrain"])
        return cls(**d)
