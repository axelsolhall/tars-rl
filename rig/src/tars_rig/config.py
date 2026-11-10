"""Typed settings, read from rig/config.toml (baked into the image at /app/config.toml)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(
    os.environ.get("RIG_CONFIG", Path(__file__).resolve().parents[2] / "config.toml")
)
MODES = ("calibrate", "imu", "capture")


@dataclass(frozen=True)
class Pins:
    button: int
    led_green: int
    led_red: int
    led_common_anode: bool = False  # common leg to 3.3 V: a pin drives its colour by going low


@dataclass(frozen=True)
class ImuConfig:
    bus: int
    address: int
    rate_hz: int
    accel_range_g: int
    gyro_range_dps: int
    dlpf: int
    poll_ms: int


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fps: int
    jpeg_quality: int
    sync: bool
    meter_seconds: float
    max_exposure_us: int
    max_gain: float
    queue_frames: int = 45  # frames the encoder may fall behind before one is dropped


@dataclass(frozen=True)
class SessionConfig:
    cameras: bool
    imu: bool
    max_minutes: float
    # Overrides [cameras] max_exposure_us for this session (calibrate: its pauses tolerate more).
    max_exposure_us: int | None = None
    # Optional script: [{name, seconds}, ...]. The red LED flashes at each change, and the session
    # stops by itself after the last one (Axel's calibration film, 2026-11-09).
    phases: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    pins: Pins
    imu: ImuConfig
    cameras: CameraConfig
    sessions: dict[str, SessionConfig]
    root: Path
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def sessions_dir(self) -> Path:
        return self.root / "sessions"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"


def load(path: Path | str = DEFAULT_PATH) -> Config:
    raw = tomllib.loads(Path(path).read_text())
    sessions = {name: SessionConfig(**raw["sessions"][name]) for name in MODES}
    root = Path(os.environ.get("RIG_ROOT", raw["storage"]["root"]))
    return Config(
        Pins(**raw["pins"]),
        ImuConfig(**raw["imu"]),
        CameraConfig(**raw["cameras"]),
        sessions,
        root,
        raw,
    )
