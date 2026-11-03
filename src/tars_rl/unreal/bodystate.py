"""What either engine reports each control step, and the policy's observation built from it.

`BodyState` is the contract between an engine and the verification loop: per slab (central, left
arm, right arm) the world position of the body frame, its orientation (w x y z) and its linear
(at the body frame's origin) and angular velocity, all in MuJoCo's frame and SI units. MuJoCo
fills it from `MjData`; Unreal sends it over the bridge in its own frame and
`tars_rl.unreal.bridge` converts.

`observe_default` turns a BodyState and the flag into the 17-number "default" frame FlagEnv
gives the policy (flag in the body frame, joint positions and velocities, gravity in the body
frame, body angular velocity in the body frame). Joints are read back from the slab poses, so an
engine never has to report them; tests check the result against `FlagEnv.observe`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from tars_rl.body.spec import SIDES, BodySpec
from tars_rl.unreal.poses import BODIES, Poses, joints_from_poses, qconj, qrotate

OBS_SETS_SUPPORTED = ("default",)  # mark2's IMUs and touch bits need more from the engine


@dataclass(frozen=True)
class BodyState:
    t: float  # s since the episode started
    pos: np.ndarray  # (3 bodies, 3) m
    quat: np.ndarray  # (3, 4) w x y z
    linvel: np.ndarray  # (3, 3) m/s, world frame, at the body frame origin
    angvel: np.ndarray  # (3, 3) rad/s, world frame

    def __post_init__(self) -> None:
        for name, shape in (("pos", (3, 3)), ("quat", (3, 4)), ("linvel", (3, 3)),
                            ("angvel", (3, 3))):
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"BodyState.{name} must be {shape}, got "
                                 f"{np.shape(getattr(self, name))}")

    @property
    def central_pos(self) -> np.ndarray:
        return self.pos[0]

    def central_rot(self) -> np.ndarray:
        """World-from-body rotation matrix of the central slab."""
        e = np.eye(3)
        return np.stack([qrotate(self.quat[0], e[i]) for i in range(3)], axis=-1)

    def cos_tilt(self) -> float:
        return float(self.central_rot()[2, 2])


def joint_state(spec: BodySpec, s: BodyState) -> tuple[np.ndarray, np.ndarray]:
    """Joint positions and velocities in BodySpec joint order, from the slab states alone.

    Hinge rate: the arm's angular velocity relative to the central body, about the hinge axis.
    Slide rate: the arm origin's velocity relative to the hinge point, along the arm's length
    (the arm origin is hinge point + slide * arm z, and d(arm z)/dt is perpendicular to it).
    """
    q = joints_from_poses(spec, Poses(np.zeros(1), s.pos[None], s.quat[None]))[0]
    arm_y = spec.central.width / 2 + spec.arm_gap + spec.arm.width / 2
    rot = s.central_rot()
    hinge_axis = rot[:, 1]
    rates = {}
    for i, (side, sign) in enumerate(zip(SIDES, (1, -1), strict=True), start=1):
        rates[f"{side}_hinge"] = float((s.angvel[i] - s.angvel[0]) @ hinge_axis)
        hinge_pt = s.pos[0] + rot @ np.array([0.0, sign * arm_y, 0.0])
        v_hinge = s.linvel[0] + np.cross(s.angvel[0], hinge_pt - s.pos[0])
        arm_z = qrotate(s.quat[i], np.array([0.0, 0.0, 1.0]))
        rates[f"{side}_slide"] = float((s.linvel[i] - v_hinge) @ arm_z)
    return q, np.array([rates[j] for j in spec.joint_names])


def observe_default(spec: BodySpec, s: BodyState, flag_pos: np.ndarray) -> np.ndarray:
    """FlagEnv's "default" observation frame (A44 layout), from a BodyState."""
    rot = s.central_rot()
    q, qd = joint_state(spec, s)
    flag = rot.T @ (np.asarray(flag_pos, float) - s.pos[0])
    gravity = -rot[2, :]
    angvel_local = qrotate(qconj(s.quat[0]), s.angvel[0])
    return np.concatenate([flag, q, qd, gravity, angvel_local]).astype(np.float32)


def from_mujoco(m: mujoco.MjModel, d: mujoco.MjData, body_ids: list[int]) -> BodyState:
    """BodyState of CPU MuJoCo data (positions after mj_forward / mj_step)."""
    mujoco.mj_kinematics(m, d)  # mj_step leaves xpos one step behind qpos
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    lin, ang = np.zeros((3, 3)), np.zeros((3, 3))
    res = np.zeros(6)
    for k, b in enumerate(body_ids):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)  # world frame
        ang[k], lin[k] = res[:3], res[3:]
    return BodyState(float(d.time), d.xpos[body_ids].copy(), d.xquat[body_ids].copy(), lin, ang)


def body_ids(m: mujoco.MjModel) -> list[int]:
    return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BODIES]
