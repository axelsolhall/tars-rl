"""Body-pose recordings from either engine, in one frame and unit system (T10).

`Poses` is the contract the comparison reads: per sample time, the world position (m) and
orientation (quaternion w, x, y, z) of the three slabs, in MuJoCo's frame (z up, x forward,
y left, right-handed). MuJoCo fills it directly; Unreal writes a CSV in its own frame
(cm, y right, left-handed, FQuat x, y, z, w) and `from_unreal_csv` converts (A73).
Joint values are read back from the poses the same way for both engines.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from tars_rl.body.mjcf import load_model
from tars_rl.body.spec import SIDES, BodySpec
from tars_rl.unreal.sequence import Sequence

BODIES = ("central", "left_arm", "right_arm")
SAMPLE_DT = 0.02  # s between recorded samples, both engines
CSV_FIELDS = ("px", "py", "pz", "qx", "qy", "qz", "qw")
CSV_COLUMNS = ["t"] + [f"{b}_{k}" for b in BODIES for k in CSV_FIELDS]


@dataclass
class Poses:
    t: np.ndarray  # (T,)
    pos: np.ndarray  # (T, 3 bodies, 3) m
    quat: np.ndarray  # (T, 3 bodies, 4) w x y z

    def save(self, path: Path) -> None:
        np.savez(path, t=self.t, pos=self.pos, quat=self.quat, bodies=np.array(BODIES))

    @classmethod
    def load(cls, path: Path) -> Poses:
        z = np.load(path)
        return cls(z["t"], z["pos"], z["quat"])

    def resample(self, t: np.ndarray) -> Poses:
        """Linear interpolation onto other sample times (quaternions renormalised)."""
        pos = np.empty((len(t), *self.pos.shape[1:]))
        quat = np.empty((len(t), *self.quat.shape[1:]))
        q = _continuous(self.quat)
        for b in range(pos.shape[1]):
            for k in range(3):
                pos[:, b, k] = np.interp(t, self.t, self.pos[:, b, k])
            for k in range(4):
                quat[:, b, k] = np.interp(t, self.t, q[:, b, k])
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        return Poses(np.asarray(t, float), pos, quat)


def _continuous(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive samples lie in the same hemisphere."""
    q = q.copy()
    for i in range(1, len(q)):
        flip = np.sum(q[i] * q[i - 1], axis=-1) < 0
        q[i][flip] *= -1
    return q


# --- quaternion helpers (w, x, y, z) ---------------------------------------------------------


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    vq = np.concatenate([np.zeros((*v.shape[:-1], 1)), v], axis=-1)
    return qmul(qmul(q, vq), qconj(q))[..., 1:]


def qangle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle (rad) between orientations a and b."""
    d = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(d, 0.0, 1.0))


def tilt_rad(quat_central: np.ndarray) -> np.ndarray:
    up = qrotate(quat_central, np.broadcast_to([0.0, 0.0, 1.0], (*quat_central.shape[:-1], 3)))
    return np.arccos(np.clip(up[..., 2], -1.0, 1.0))


# --- frame conversion (A73) ------------------------------------------------------------------


def ue_to_mj(pos_cm: np.ndarray, quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unreal (cm, y right, FQuat x y z w) -> MuJoCo (m, y left, w x y z).
    Mirroring y maps rotation matrix R to M R M with M = diag(1, -1, 1), i.e. the quaternion
    (w, x, y, z) -> (w, -x, y, -z)."""
    pos = pos_cm / 100.0 * np.array([1.0, -1.0, 1.0])
    x, y, z, w = np.moveaxis(quat_xyzw, -1, 0)
    return pos, np.stack([w, -x, y, -z], axis=-1)


def mj_to_ue(pos_m: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = pos_m * 100.0 * np.array([1.0, -1.0, 1.0])
    w, x, y, z = np.moveaxis(quat_wxyz, -1, 0)
    return pos, np.stack([-x, y, -z, w], axis=-1)


def from_unreal_csv(path: str | Path) -> Poses:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or list(rows[0].keys()) != CSV_COLUMNS:
        raise ValueError(f"{path}: expected columns {CSV_COLUMNS}")
    a = np.array([[float(r[c]) for c in CSV_COLUMNS] for r in rows])
    raw = a[:, 1:].reshape(len(a), len(BODIES), 7)
    pos, quat = ue_to_mj(raw[..., :3], raw[..., 3:])
    return Poses(a[:, 0], pos, quat)


def to_unreal_csv(p: Poses, path: str | Path) -> None:
    """The format Unreal's recorder writes; used by tests and to sanity-check the parser."""
    pos, quat = mj_to_ue(p.pos, p.quat)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for i, t in enumerate(p.t):
            w.writerow([f"{t:.6f}"] + [
                f"{v:.6f}" for b in range(len(BODIES)) for v in (*pos[i, b], *quat[i, b])
            ])


# --- joints from poses -----------------------------------------------------------------------


def joints_from_poses(spec: BodySpec, p: Poses) -> np.ndarray:
    """(T, 4) joint values in actuator order, read from slab poses alone.
    Hinge: arm rotation about the body y axis relative to the central body. Slide: arm
    offset from its neutral attachment, along the arm's own length (as in tars_rl.body.mjcf)."""
    arm_y = spec.central.width / 2 + spec.arm_gap + spec.arm.width / 2
    qc, pc = p.quat[:, 0], p.pos[:, 0]
    hinge, slide = {}, {}
    for i, (side, sign) in enumerate(zip(SIDES, (1, -1), strict=True), start=1):
        rel = qmul(qconj(qc), p.quat[:, i])
        hinge[side] = 2.0 * np.arctan2(rel[:, 2], rel[:, 0])
        hinge[side] = (hinge[side] + np.pi) % (2 * np.pi) - np.pi
        off = qrotate(qconj(qc), p.pos[:, i] - pc) - np.array([0.0, sign * arm_y, 0.0])
        axis = qrotate(rel, np.broadcast_to([0.0, 0.0, 1.0], off.shape))
        slide[side] = np.sum(off * axis, axis=-1)
    cols = {f"{s}_hinge": hinge[s] for s in SIDES} | {f"{s}_slide": slide[s] for s in SIDES}
    return np.stack([cols[j] for j in spec.joint_names], axis=-1)


# --- MuJoCo reference ------------------------------------------------------------------------


def mujoco_reference(spec: BodySpec, seq: Sequence) -> tuple[Poses, np.ndarray]:
    """Play `seq` in CPU MuJoCo. Returns the poses and the qpos joint values (T, 4)."""
    seq.check(spec)
    m = load_model(spec)
    d = mujoco.MjData(m)
    bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BODIES]
    qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
            for j in spec.joint_names]
    d.ctrl[:] = seq.at(0.0)
    mujoco.mj_forward(m, d)
    every = round(SAMPLE_DT / m.opt.timestep)
    n = int(round(seq.duration / m.opt.timestep))
    ts, pos, quat, q = [], [], [], []
    for i in range(n + 1):
        if i % every == 0:
            mujoco.mj_kinematics(m, d)  # mj_step leaves xpos one step behind qpos
            ts.append(d.time)
            pos.append(d.xpos[bids].copy())
            quat.append(d.xquat[bids].copy())
            q.append(d.qpos[qadr].copy())
        if i == n:
            break
        d.ctrl[:] = seq.at(d.time)
        mujoco.mj_step(m, d)
    return Poses(np.array(ts), np.array(pos), np.array(quat)), np.array(q)
