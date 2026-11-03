"""The two physics engines a policy is verified in, behind one small interface.

    load(level)                      build the level's boxes
    reset(spawn_xy, flag) -> state   TARS standing at neutral on the floor at spawn_xy, facing +x
    set_flag(pos)                    move the flag marker (chain mode respawns)
    step(ctrl, hud) -> state         apply servo targets (SI, actuator order), advance one
                                     control step (ctrl_dt), return the new BodyState

`MujocoEngine` is CPU MuJoCo on the exact model FlagEnv trains on, plus the level's boxes: on the
flat level it is the training environment. `UnrealEngine` is Unreal's own physics (Chaos) through
the bridge. The verification loop (`tars_rl.unreal.verify`) only sees this interface, so the gap
it reports is the engines' gap, not two different episode loops.
"""

from __future__ import annotations

from typing import Protocol

import mujoco
import numpy as np

from tars_rl.body.spec import BodySpec
from tars_rl.env.arena import FLAG_BODY, build_arena_mjcf
from tars_rl.env.flag_env import ROOT_JOINT
from tars_rl.env.levels import Level, add_to_mjcf
from tars_rl.unreal.bodystate import BodyState, body_ids, from_mujoco
from tars_rl.unreal.bridge import (
    DEFAULT_PORT,
    BridgeClient,
    level_to_ue,
    point_to_ue,
    state_from_ue,
)


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
        mujoco.mj_forward(m, d)
        return from_mujoco(m, d, self.ids)

    def set_flag(self, pos: np.ndarray) -> None:
        self.d.mocap_pos[self.mocap] = pos

    def step(self, ctrl: np.ndarray, hud: str | None = None) -> BodyState:
        self.d.ctrl[:] = ctrl
        mujoco.mj_step(self.m, self.d, nstep=self.n_sub)
        return from_mujoco(self.m, self.d, self.ids)

    def close(self) -> None:
        pass


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

    def step(self, ctrl: np.ndarray, hud: str | None = None) -> BodyState:
        msg: dict = {"op": "step", "ctrl": [float(c) for c in ctrl]}
        if hud:
            msg["hud"] = hud
        return state_from_ue(self.client.request(msg))

    def close(self) -> None:
        self.client.close()
