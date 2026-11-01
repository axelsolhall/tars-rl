"""BodySpec: the versioned, JSON-serialisable description of the TARS body (A8).

Frame: z up, x forward, y to the robot's left. The central body and both arms are boxes
standing side by side along y. Both arm hinges share one axis along y through the centre of
the central body (A39, A5); each arm also slides along its own length (A40).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
SIDES = ("left", "right")  # left = +y, right = -y


@dataclass(frozen=True)
class Box:
    """A rigid slab: depth along x, width along y, height along z (metres), mass in kg."""

    depth: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class Servo:
    """Position actuator (A2, A41): kp in N/m or Nm/rad, kv damping, force limit, and top speed
    (rad/s or m/s; None = unlimited). MuJoCo has no speed cap, so the env slews the target."""

    kp: float
    kv: float
    force_limit: float
    max_speed: float | None = None


@dataclass(frozen=True)
class BodySpec:
    version: int
    name: str
    central: Box
    arm: Box
    arm_gap: float  # y clearance between the central body and each arm
    hinge_range: tuple[float, float]  # radians
    slide_range: tuple[float, float]  # metres along the arm, 0 = arm centred on the hinge
    hinge_servo: Servo
    slide_servo: Servo
    friction: float = 0.8
    timestep: float = 0.002
    neutral: dict[str, float] = field(default_factory=dict)  # joint name -> target

    # Contract: joint and actuator names, in actuator order.
    @property
    def joint_names(self) -> list[str]:
        return [f"{s}_{k}" for k in ("hinge", "slide") for s in SIDES]

    @property
    def actuator_names(self) -> list[str]:
        return [f"{j}_act" for j in self.joint_names]

    def joint_range(self, joint: str) -> tuple[float, float]:
        return self.hinge_range if joint.endswith("hinge") else self.slide_range

    def neutral_ctrl(self) -> list[float]:
        return [self.neutral.get(j, 0.0) for j in self.joint_names]

    @property
    def total_mass(self) -> float:
        return self.central.mass + 2 * self.arm.mass

    def to_json(self) -> str:
        d = asdict(self)
        d["schema"] = "tars_rl.body.BodySpec"
        d["joint_names"] = self.joint_names
        d["actuator_names"] = self.actuator_names
        return json.dumps(d, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> BodySpec:
        d = json.loads(text)
        if d.get("version") != SCHEMA_VERSION:
            raise ValueError(f"BodySpec version {d.get('version')} != {SCHEMA_VERSION}")
        for derived in ("schema", "joint_names", "actuator_names"):
            d.pop(derived, None)
        d["central"] = Box(**d["central"])
        d["arm"] = Box(**d["arm"])
        d["hinge_servo"] = Servo(**d["hinge_servo"])
        d["slide_servo"] = Servo(**d["slide_servo"])
        d["hinge_range"] = tuple(d["hinge_range"])
        d["slide_range"] = tuple(d["slide_range"])
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> BodySpec:
        return cls.from_json(Path(path).read_text())


def default_spec() -> BodySpec:
    """Printable scale, about 36 cm tall (A7). Slide range is +-25 % of arm length (A42)."""
    height = 0.36
    slide = 0.25 * height
    return BodySpec(
        version=SCHEMA_VERSION,
        name="tars_v1",
        # Built weight incl. battery and electronics, 3 kg (Axel 2026-11-01: 1.5 kg was too light).
        central=Box(depth=0.09, width=0.09, height=height, mass=2.0),
        arm=Box(depth=0.09, width=0.045, height=height, mass=0.5),
        arm_gap=0.004,
        hinge_range=(-math.radians(150), math.radians(150)),
        slide_range=(-slide, slide),
        # MG996R-class hobby servo (Axel 2026-11-01): 1.5 Nm, 6 rad/s (~0.17 s per 60 deg); with
        # 3 Nm and no speed cap the arms whirled like spokes at 1.3 m/s. The screw slide is stiff
        # because it is not backdrivable (A41), and heavily damped because a screw is slow: with
        # kv=40 / 80 N a full-stroke step threw the arm at 3.5 m/s and the recoil tipped the
        # body (A55).
        hinge_servo=Servo(kp=8.0, kv=0.3, force_limit=1.5, max_speed=6.0),
        slide_servo=Servo(kp=3000.0, kv=300.0, force_limit=10.0),
    )
