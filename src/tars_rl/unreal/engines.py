"""The physics engines a policy is verified in, behind one small interface.

    load(level)                      build the level's boxes
    reset(spawn_xy, flag) -> state   TARS standing at neutral on the floor at spawn_xy, facing +x
    set_flag(pos)                    move the flag marker (chain mode respawns)
    step(ctrl, hud) -> state         apply servo targets (SI, actuator order), advance one
                                     control step (ctrl_dt), return the new BodyState

Four engines, from the training physics outwards:
    mjx      MJX on the model FlagEnv trains on (plus the level's boxes): the training physics
    mujoco   CPU MuJoCo on the same model. Not identical to MJX: box-floor contacts differ (up to
             0.5 rad/s per 2 ms substep from the same state, 2026-11-03), so this is already a
             small sim-to-sim step
    bullet   Bullet (PyBullet), hard contacts (`tars_rl.unreal.bullet`)
    unreal   Unreal's own physics (Chaos) through the bridge
For probes (`tars_rl.unreal.probe`) the CPU engines also take `set_push(force, at)`: a constant
force (N, world) at a point of the central slab (m, its own frame) until the next reset.
For one-step comparisons (`tars_rl.unreal.onestep`) mujoco, bullet and unreal take
`set_state(state) -> state`: TARS put exactly in that BodyState (poses and velocities).
The verification loop (`tars_rl.unreal.verify`) only sees this interface, so the gaps it reports
are the engines' gaps, not different episode loops.
"""

from __future__ import annotations

from typing import Protocol

import jax
import mujoco
import numpy as np
from mujoco import mjx

from tars_rl.body.spec import BodySpec
from tars_rl.env.arena import FLAG_BODY, build_arena_mjcf
from tars_rl.env.flag_env import ROOT_JOINT
from tars_rl.env.levels import Level, add_to_mjcf
from tars_rl.unreal.bodystate import BodyState, body_ids, from_mujoco, joint_state
from tars_rl.unreal.bridge import (
    DEFAULT_PORT,
    BridgeClient,
    BridgeError,
    level_to_ue,
    point_to_ue,
    state_from_ue,
    state_to_ue,
)
from tars_rl.unreal.poses import qconj, qrotate


class Engine(Protocol):
    name: str

    def load(self, level: Level) -> None: ...
    def reset(self, spawn_xy: np.ndarray, flag: np.ndarray) -> BodyState: ...
    def set_flag(self, pos: np.ndarray) -> None: ...
    def step(self, ctrl: np.ndarray, hud: str | None = None) -> BodyState: ...
    def close(self) -> None: ...


def check_neutral_is_zero(spec: BodySpec) -> None:
    # Unreal builds TARS with every joint at 0 and both engines start the servo targets there.
    if np.any(np.asarray(spec.neutral_ctrl()) != 0):
        raise ValueError("BodySpec neutral pose is not all zeros; Unreal's TarsActor builds TARS "
                         "at zero joints, so teach it the neutral pose first")


class MujocoEngine:
    name = "mujoco"

    def __init__(self, spec: BodySpec, ctrl_dt: float) -> None:
        check_neutral_is_zero(spec)
        self.spec, self.ctrl_dt = spec, ctrl_dt
        self.level: Level | None = None
        self.push: tuple[np.ndarray, np.ndarray] | None = None

    def load(self, level: Level) -> None:
        self.level = level
        self.m = mujoco.MjModel.from_xml_string(add_to_mjcf(build_arena_mjcf(self.spec), level))
        self.d = mujoco.MjData(self.m)
        self.n_sub = round(self.ctrl_dt / self.m.opt.timestep)
        self.ids = body_ids(self.m)
        root = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT)
        self.root_adr = int(self.m.jnt_qposadr[root])
        self.mocap = int(self.m.body_mocapid[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, FLAG_BODY)])

    def reset(self, spawn_xy: np.ndarray, flag: np.ndarray) -> BodyState:
        assert self.level is not None, "load a level first"
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)  # joints at 0 = neutral, root at its MJCF pose (on the floor)
        d.qpos[self.root_adr:self.root_adr + 2] = spawn_xy
        d.qpos[self.root_adr + 2] += self.level.ground_z(spawn_xy)
        d.ctrl[:] = 0.0
        d.mocap_pos[self.mocap] = flag
        self.push = None
        mujoco.mj_forward(m, d)
        return from_mujoco(m, d, self.ids)

    def set_flag(self, pos: np.ndarray) -> None:
        self.d.mocap_pos[self.mocap] = pos

    def set_push(self, force: np.ndarray, at: np.ndarray) -> None:
        self.push = (np.asarray(force, float), np.asarray(at, float))

    def set_state(self, s: BodyState) -> BodyState:
        """One-step probes: TARS put in this state (the root from the central slab, the joints
        from the slabs' relative poses); actuator state is MuJoCo's only memory, and it has none."""
        m, d = self.m, self.d
        q, qd = joint_state(self.spec, s)
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT)
        a, v = self.root_adr, int(m.jnt_dofadr[root])
        d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = s.pos[0], s.quat[0]
        d.qvel[v:v + 3] = s.linvel[0]
        d.qvel[v + 3:v + 6] = qrotate(qconj(s.quat[0]), s.angvel[0])  # a free joint's spin is local
        for name, qi, qdi in zip(self.spec.joint_names, q, qd, strict=True):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]] = qi, qdi
        d.time = s.t
        mujoco.mj_forward(m, d)
        return from_mujoco(m, d, self.ids)

    def step(self, ctrl: np.ndarray, hud: str | None = None) -> BodyState:
        self.d.ctrl[:] = ctrl
        if self.push is None:
            mujoco.mj_step(self.m, self.d, nstep=self.n_sub)
        else:  # the push's lever turns with the slab: recompute it every physics step
            c, (force, at) = self.ids[0], self.push
            for _ in range(self.n_sub):
                self.d.xfrc_applied[c, :3] = force
                self.d.xfrc_applied[c, 3:] = np.cross(self.d.xmat[c].reshape(3, 3) @ at, force)
                mujoco.mj_step(self.m, self.d)
            self.d.xfrc_applied[c] = 0.0
        return from_mujoco(self.m, self.d, self.ids)

    def close(self) -> None:
        pass


class MjxEngine(MujocoEngine):
    """MJX physics (what training runs on), read back through CPU MjData for the BodyState."""

    name = "mjx"

    def load(self, level: Level) -> None:
        super().load(level)
        model, n = mjx.put_model(self.m), self.n_sub
        self._advance = jax.jit(
            lambda d: jax.lax.fori_loop(0, n, lambda _, x: mjx.step(model, x), d))

    def reset(self, spawn_xy: np.ndarray, flag: np.ndarray) -> BodyState:
        s = super().reset(spawn_xy, flag)
        self.dx = mjx.put_data(self.m, self.d)
        return s

    def step(self, ctrl: np.ndarray, hud: str | None = None) -> BodyState:
        self.dx = self._advance(self.dx.replace(ctrl=np.asarray(ctrl, self.dx.ctrl.dtype)))
        mjx.get_data_into(self.d, self.m, self.dx)
        return from_mujoco(self.m, self.d, self.ids)


class UnrealEngine:
    name = "unreal"

    def __init__(self, spec: BodySpec, ctrl_dt: float, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT) -> None:
        check_neutral_is_zero(spec)
        self.client = BridgeClient(host, port)
        if abs(self.client.dt - ctrl_dt) > 1e-9:
            raise ValueError(f"Unreal steps {self.client.dt} s per control step, the policy was "
                             f"trained at {ctrl_dt} s: start Unreal with -TarsDt={ctrl_dt}")
        sub = spec.timestep * self.client.substeps
        if abs(sub - ctrl_dt) > 1e-9:
            raise ValueError(f"Unreal runs {self.client.substeps} physics substeps per step; "
                             f"MuJoCo runs {round(ctrl_dt / spec.timestep)}")
        self.ctrl_dt = ctrl_dt
        self.level: Level | None = None

    def load(self, level: Level) -> None:
        self.level = level
        self.client.request(level_to_ue(level))

    def reset(self, spawn_xy: np.ndarray, flag: np.ndarray) -> BodyState:
        assert self.level is not None, "load a level first"
        at = point_to_ue([spawn_xy[0], spawn_xy[1], self.level.ground_z(spawn_xy)])
        return state_from_ue(self.client.request(
            {"op": "reset", "at": at, "yaw": 0.0, "flag": point_to_ue(flag)}))

    def set_flag(self, pos: np.ndarray) -> None:
        self.client.request({"op": "flag", "pos": point_to_ue(pos)})

    def set_push(self, force: np.ndarray, at: np.ndarray) -> None:
        # The slab's own frame mirrors like the world's (A73): flip y, metres -> cm.
        self.client.request({"op": "push", "force": point_to_ue(force), "at": point_to_ue(at)})

    def set_state(self, s: BodyState) -> BodyState:
        return state_from_ue(self.client.request({"op": "set", **state_to_ue(s)}))

    def step(self, ctrl: np.ndarray, hud: str | None = None) -> BodyState:
        msg: dict = {"op": "step", "ctrl": [float(c) for c in ctrl]}
        if hud:
            msg["hud"] = hud
        reply = self.client.request(msg)
        if abs(float(reply["dt"]) - self.ctrl_dt) > 1e-6:
            raise BridgeError(f"Unreal advanced {reply['dt']} s for one step, not {self.ctrl_dt}: "
                              "its fixed time step is off (bridge mode turns it on)")
        return state_from_ue(reply)

    def close(self) -> None:
        self.client.close()
