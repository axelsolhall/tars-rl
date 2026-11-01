"""TARS's senses: the flag sensor (A19) and the observation sets built from named channels.

Flag sensor: any sensor takes (body position, body rotation matrix, flag position), all world
frame, and returns FLAG_OBS_DIM floats. A camera-based estimate later replaces
`ground_truth_flag` with the same signature and output layout, so the policy's input does not
change.

Observation sets (`EnvConfig.obs`): an ordered tuple of channel names from CHANNELS.
  default  the 17 numbers the benchmark was trained on (A44), unchanged
  mark2    flag, joint encoders, three IMUs (accelerometer + gyro), ten edge touch bits and the
           last action (T24). The IMUs replace default's exact gravity / angular velocity.

Mark 2 hardware lives in the arena model only (`add_mark2_sensors`), never in the body asset,
so the default model stays byte-identical to the one the benchmark ran on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco

from tars_rl.body.spec import SIDES, BodySpec

FLAG_OBS_DIM = 3

FlagSensor = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def ground_truth_flag(
    body_pos: jax.Array, body_rot: jax.Array, flag_pos: jax.Array
) -> jax.Array:
    """Exact flag position relative to the central body, in the central body's own frame (m)."""
    return body_rot.T @ (flag_pos - body_pos)


def check_sensor(sensor: FlagSensor) -> FlagSensor:
    out = jax.eval_shape(sensor, jnp.zeros(3), jnp.eye(3), jnp.zeros(3))
    if out.shape != (FLAG_OBS_DIM,):
        raise ValueError(f"flag sensor must return shape ({FLAG_OBS_DIM},), got {out.shape}")
    return sensor


# ---- Mark 2 hardware ----------------------------------------------------------------------
IMU_BODIES = ("central", "left_arm", "right_arm")
EDGE_SITE_GROUP = 4  # hidden in the default viewer / renderer


def imu_site(body: str) -> str:
    return "imu" if body == "central" else f"{body}_imu"  # "imu" is the body asset's own site


def touch_edges() -> list[tuple[str, str, float, float]]:
    """(edge name, body, x sign, z sign) for the ten touch sensors, in observation order.

    Edges run along y (the hinge axis). Front = +x, bottom = -z in the slab's own frame.
    The central body has its two bottom edges, each arm all four.
    """
    edges = [(f"central_bottom_{f}", "central", sx, -1.0)
             for f, sx in (("front", 1.0), ("back", -1.0))]
    for side in SIDES:
        for v, sz in (("bottom", -1.0), ("top", 1.0)):
            for f, sx in (("front", 1.0), ("back", -1.0)):
                edges.append((f"{side}_{v}_{f}", f"{side}_arm", sx, sz))
    return edges


def edge_radius(spec: BodySpec) -> float:
    """Half thickness of each edge's touch box: 10 % of the thinnest slab dimension (A99)."""
    return 0.1 * min(spec.central.depth, spec.central.height, spec.arm.depth, spec.arm.height)


def add_mark2_sensors(xml: str, spec: BodySpec) -> mujoco.MjModel:
    """Arena MJCF + IMU sites at each arm's joint, ten edge touch boxes, and their sensors.

    The central IMU uses the body asset's `imu` site at the central body's centre, which is on
    the hinge axis (A39). Each arm's IMU sits at the arm's origin: on the hinge axis while the
    slide is centred, and fixed to the arm as it slides (as a board screwed to the arm would be).
    Sites have no mass and do not collide, so physics is unchanged.
    """
    ms = mujoco.MjSpec.from_string(xml)
    r = edge_radius(spec)
    for side in SIDES:
        ms.body(f"{side}_arm").add_site(name=imu_site(f"{side}_arm"), pos=[0, 0, 0])
    for name, body, sx, sz in touch_edges():
        box = spec.central if body == "central" else spec.arm
        ms.body(body).add_site(
            name=f"touch_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, group=EDGE_SITE_GROUP,
            pos=[sx * box.depth / 2, 0, sz * box.height / 2],
            size=[r, box.width / 2 + r, r], rgba=[1, 0.3, 0.1, 0.5],
        )
    site = mujoco.mjtObj.mjOBJ_SITE
    for body in IMU_BODIES:
        s = imu_site(body)
        ms.add_sensor(name=f"{body}_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                      objtype=site, objname=s)
        ms.add_sensor(name=f"{body}_gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                      objtype=site, objname=s)
    for name, *_ in touch_edges():
        ms.add_sensor(name=f"touch_{name}", type=mujoco.mjtSensor.mjSENS_TOUCH,
                      objtype=site, objname=f"touch_{name}")
    return ms.compile()


# ---- channels -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Channel:
    """One named block of the observation. `lo`/`hi` are the nominal range, not a clip."""

    name: str
    units: str
    labels: tuple[str, ...]  # one per float
    lo: float
    hi: float
    binary: bool = False  # noise on a binary channel is a flip probability
    needs_mark2: bool = False  # reads sensors only the mark2 model has

    @property
    def size(self) -> int:
        return len(self.labels)


def _xyz(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{a}" for a in "xyz")


def channels(spec: BodySpec) -> dict[str, Channel]:
    """Every channel the env can emit, keyed by name. Ranges follow the BodySpec."""
    j = spec.joint_names
    lim = max(abs(x) for x in (*spec.hinge_range, *spec.slide_range))
    g = 9.81
    out = [
        Channel("flag", "m, central-body frame", _xyz("flag"), -10 * spec.central.height,
                10 * spec.central.height),
        Channel("joint_pos", "rad (hinges), m (slides)", tuple(f"{n}_pos" for n in j), -lim, lim),
        Channel("joint_vel", "rad/s (hinges), m/s (slides)", tuple(f"{n}_vel" for n in j),
                -20.0, 20.0),
        Channel("gravity", "unit vector, central-body frame", _xyz("gravity"), -1.0, 1.0),
        Channel("body_angvel", "rad/s, central-body frame", _xyz("angvel"), -20.0, 20.0),
    ]
    for b in IMU_BODIES:
        out.append(Channel(f"{b}_acc", "m/s^2, own frame (+g up at rest)", _xyz(f"{b}_acc"),
                           -4 * g, 4 * g, needs_mark2=True))
        out.append(Channel(f"{b}_gyro", "rad/s, own frame", _xyz(f"{b}_gyro"), -20.0, 20.0,
                           needs_mark2=True))
    out += [
        Channel("touch", "0/1 (edge pressed)", tuple(n for n, *_ in touch_edges()), 0.0, 1.0,
                binary=True, needs_mark2=True),
        Channel("last_action", "[-1, 1], actuator order", tuple(f"{n}_last" for n in j),
                -1.0, 1.0),
    ]
    return {c.name: c for c in out}


OBS_SETS: dict[str, tuple[str, ...]] = {
    # Order is the contract: default is A44's layout, bit for bit.
    "default": ("flag", "joint_pos", "joint_vel", "gravity", "body_angvel"),
    "mark2": (
        "flag", "joint_pos", "joint_vel",
        "central_acc", "central_gyro", "left_arm_acc", "left_arm_gyro",
        "right_arm_acc", "right_arm_gyro", "touch", "last_action",
    ),
}


def obs_set(name: str, spec: BodySpec) -> list[Channel]:
    if name not in OBS_SETS:
        raise ValueError(f"unknown observation set {name!r}; pick one of {sorted(OBS_SETS)}")
    ch = channels(spec)
    return [ch[n] for n in OBS_SETS[name]]
