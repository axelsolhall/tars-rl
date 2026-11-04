"""FlagEnv: TARS on a flat arena walking to a randomly placed flag, in MJX (A1).

With `config.terrain` the floor is rough ground instead (`tars_rl.env.terrain`, stage 3).

Functional and jit/vmap-friendly: `reset(rng) -> State`, `step(state, action) -> State`.

Action (A43): one float in [-1, 1] per BodySpec actuator, in actuator order (both hinges, then
both slides); it maps linearly onto that actuator's ctrlrange, so 0 is the centred target.

Observation: the channels of the set `config.obs` names (`sensors.OBS_SETS`), concatenated in
order; `channel_slices()` gives each one's span. The default set (A19, A44), 9 + 2n floats:
  [0:3]          flag vector from the flag sensor (default: flag in the central body's frame)
  [3:3+n]        joint positions, BodySpec actuator order
  [3+n:3+2n]     joint velocities, same order
  [3+2n:6+2n]    gravity direction in the central body's frame (unit vector)
  [6+2n:9+2n]    central body angular velocity in its own frame (rad/s)
The mark2 set (T24) swaps the last two for three IMUs and adds ten touch bits and the last action.

Flags (T26, `config.flags`): in "chain" mode (default) tagging a flag (centre within the success
radius, A23) moves it to a new spot drawn by the same rules around TARS's current position and the
episode goes on; it ends only on a fall (A24) or timeout (config.episode_seconds). The tag step's
progress is measured against the old flag; from the next step on, against the new one. In "single"
mode (the benchmark) tagging ends the episode as a success.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from tars_rl.body.spec import BodySpec
from tars_rl.env import terrain
from tars_rl.env.arena import FLAG_BODY, build_arena_mjcf, load_body_spec
from tars_rl.env.config import EnvConfig
from tars_rl.env.sensors import (
    IMU_BODIES,
    Channel,
    FlagSensor,
    add_mark2_sensors,
    check_sensor,
    ground_truth_flag,
    obs_set,
    touch_edges,
)

CENTRAL_BODY = "central"
METRIC_KEYS = ("progress", "upright", "fall", "effort", "tag")
ROOT_JOINT = "root"


class State(NamedTuple):
    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array  # single: success | fell | truncated; chain: fell | truncated
    success: jax.Array  # a flag was tagged this step
    fell: jax.Array
    truncated: jax.Array  # timeout without fall (single: and without success)
    flag_pos: jax.Array  # (3,) world frame, on the floor; the current flag
    dist: jax.Array  # xy distance from the central body's centre to the current flag (m)
    start_dist: jax.Array  # dist when the current flag appeared
    step: jax.Array  # policy steps taken
    metrics: dict[str, jax.Array]  # the reward's terms this step, already weighted
    tags: jax.Array  # flags tagged this episode (int32)
    first_tag: jax.Array  # step of the first tag, -1 before it (int32)
    rng: jax.Array | None = None  # sensor noise key; None (e.g. the training adapter) = no noise
    flag_rng: jax.Array | None = None  # draws the next flag in chain mode


class FlagEnv:
    def __init__(
        self,
        spec: BodySpec | None = None,
        config: EnvConfig | None = None,
        flag_sensor: FlagSensor = ground_truth_flag,
    ) -> None:
        self.spec = spec or load_body_spec()
        self.config = config or EnvConfig()
        self.flag_sensor = check_sensor(flag_sensor)
        self.channels: list[Channel] = obs_set(self.config.obs, self.spec)
        self._mark2 = any(c.needs_mark2 for c in self.channels)
        unknown = set(self.config.sensor_noise) - {c.name for c in self.channels}
        if unknown:
            raise ValueError(f"sensor_noise names channels not in {self.config.obs!r}: {unknown}")
        if self.config.sensor_delay_steps not in (0, 1):
            raise ValueError("sensor_delay_steps must be 0 or 1")
        xml = build_arena_mjcf(self.spec)
        if self.config.terrain is not None:
            terrain.install_point_contact()
            xml = terrain.add_to_mjcf(xml, self.config.terrain)
        # The default model is built exactly as the benchmark's was; mark2 adds sites + sensors.
        self.mj_model = (add_mark2_sensors(xml, self.spec) if self._mark2
                         else mujoco.MjModel.from_xml_string(xml))
        if self.config.terrain is not None:
            terrain.fill_model(self.mj_model, self.config.terrain)
        self.mjx_model = mjx.put_model(self.mj_model)
        m = self.mj_model

        ts = float(m.opt.timestep)
        self.n_substeps = round(self.config.ctrl_dt / ts)
        if self.n_substeps < 1 or abs(self.n_substeps * ts - self.config.ctrl_dt) > 1e-9:
            raise ValueError(f"ctrl_dt {self.config.ctrl_dt} is not a multiple of timestep {ts}")
        self.height = float(self.spec.central.height)

        def jid(name: str) -> int:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise ValueError(f"joint {name!r} not in the body MJCF")
            return i

        joints = [jid(j) for j in self.spec.joint_names]
        self._qadr = jnp.asarray(m.jnt_qposadr[joints])
        self._dadr = jnp.asarray(m.jnt_dofadr[joints])
        root_dof = int(m.jnt_dofadr[jid(ROOT_JOINT)])
        self._angvel = jnp.arange(root_dof + 3, root_dof + 6)  # free joint: local-frame omega
        self._central = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CENTRAL_BODY)
        flag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, FLAG_BODY)
        self._mocap = int(m.body_mocapid[flag])

        # Actuator order is BodySpec order (A8); check rather than trust.
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        if names != self.spec.actuator_names:
            raise ValueError(f"actuator order {names} != BodySpec {self.spec.actuator_names}")
        lo, hi = m.actuator_ctrlrange.T
        self._ctrl_mid = jnp.asarray((lo + hi) / 2)
        self._ctrl_half = jnp.asarray((hi - lo) / 2)
        # Servo top speed as the largest target move per control step; inf = no cap.
        speeds = [self.spec.hinge_servo.max_speed if j.endswith("hinge") else
                  self.spec.slide_servo.max_speed for j in self.spec.joint_names]
        self._ctrl_max_step = jnp.asarray(
            [np.inf if s is None else s * self.config.ctrl_dt for s in speeds])

        d = mujoco.MjData(m)
        neutral = self.spec.neutral_ctrl()
        for a, v in zip(m.jnt_qposadr[joints], neutral, strict=True):
            d.qpos[a] = v
        d.ctrl[:] = neutral
        mujoco.mj_forward(m, d)
        self._data0 = mjx.put_data(m, d)
        self.start_xy = np.asarray(d.xpos[self._central][:2]).copy()

        if self._mark2:
            def sadr(name: str) -> np.ndarray:
                i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
                return np.arange(m.sensor_adr[i], m.sensor_adr[i] + m.sensor_dim[i])

            self._sensor_adr = {f"{b}_{k}": jnp.asarray(sadr(f"{b}_{k}"))
                                for b in IMU_BODIES for k in ("acc", "gyro")}
            self._sensor_adr["touch"] = jnp.asarray(
                np.concatenate([sadr(f"touch_{n}") for n, *_ in touch_edges()]))

    # ---- sizes ---------------------------------------------------------------------------
    @property
    def action_size(self) -> int:
        return int(self.mj_model.nu)

    @property
    def frame_size(self) -> int:
        """One step's senses; `channel_slices` index into this (the newest frame)."""
        return sum(c.size for c in self.channels)

    @property
    def observation_size(self) -> int:
        return self.frame_size * max(1, self.config.history_steps)

    def _window(self, prev_obs: jax.Array | None, frame: jax.Array) -> jax.Array:
        """Memory (history_steps > 0): the obs is the last K frames, oldest first, so it carries
        its own history and needs no extra state; a reset fills the window with the first frame."""
        k = self.config.history_steps
        if k == 0:
            return frame
        if prev_obs is None:
            return jnp.tile(frame, k)
        return jnp.concatenate([prev_obs[self.frame_size:], frame])

    def channel_slices(self) -> dict[str, slice]:
        out, i = {}, 0
        for c in self.channels:
            out[c.name] = slice(i, i + c.size)
            i += c.size
        return out

    @property
    def dt(self) -> float:
        return self.config.ctrl_dt

    # ---- pieces --------------------------------------------------------------------------
    def action_to_ctrl(self, action: jax.Array, prev_ctrl: jax.Array | None = None) -> jax.Array:
        """Action -> servo targets; with `prev_ctrl`, each target moves at most the servo's top
        speed per step (a real servo turns at its rated speed, not instantly)."""
        ctrl = self._ctrl_mid + self._ctrl_half * jnp.clip(action, -1.0, 1.0)
        if prev_ctrl is None:
            return ctrl
        return prev_ctrl + jnp.clip(ctrl - prev_ctrl, -self._ctrl_max_step, self._ctrl_max_step)

    def sample_flag(self, rng: jax.Array, centre_xy: jax.Array | None = None) -> jax.Array:
        """Uniform direction, uniform distance in [min, max] body heights (A22) from `centre_xy`,
        default the start position."""
        c = self.config
        k_ang, k_dist = jax.random.split(rng)
        ang = jax.random.uniform(k_ang, (), minval=0.0, maxval=2 * jnp.pi)
        r = jax.random.uniform(
            k_dist, (), minval=c.flag_min_heights * self.height,
            maxval=c.flag_max_heights * self.height,
        )
        centre = jnp.asarray(self.start_xy if centre_xy is None else centre_xy)
        xy = centre + r * jnp.array([jnp.cos(ang), jnp.sin(ang)])
        return jnp.array([xy[0], xy[1], 0.0])

    def _dist(self, data: mjx.Data, flag_pos: jax.Array) -> jax.Array:
        return jnp.linalg.norm(data.xpos[self._central][:2] - flag_pos[:2])

    def _cos_tilt(self, data: mjx.Data) -> jax.Array:
        return data.xmat[self._central][2, 2]  # world z of the body's up axis

    def _channel(self, name: str, data: mjx.Data, flag_pos: jax.Array) -> jax.Array:
        rot = data.xmat[self._central]
        if name == "flag":
            return self.flag_sensor(data.xpos[self._central], rot, flag_pos)
        if name == "joint_pos":
            return data.qpos[self._qadr]
        if name == "joint_vel":
            return data.qvel[self._dadr]
        if name == "gravity":
            return -rot[2, :]  # R^T @ (0, 0, -1)
        if name == "body_angvel":
            return data.qvel[self._angvel]
        if name == "touch":
            return (data.sensordata[self._sensor_adr["touch"]]
                    > self.config.touch_threshold_n).astype(jnp.float32)
        if name == "last_action":
            return (data.ctrl - self._ctrl_mid) / self._ctrl_half  # servo target sent, speed-capped
        return data.sensordata[self._sensor_adr[name]]

    def observe(self, data: mjx.Data, flag_pos: jax.Array, rng: jax.Array | None = None,
                sensed: mjx.Data | None = None) -> jax.Array:
        """Concatenate the set's channels. `sensed` = the data the senses read (the previous
        step's under a sensor delay); the last action always comes from `data`. Noise needs rng."""
        sensed = data if sensed is None else sensed
        noise = self.config.sensor_noise
        if any(noise.values()) and rng is None:
            raise ValueError("sensor_noise needs State.rng; this caller does not carry one")
        keys = jax.random.split(rng, len(self.channels)) if rng is not None else None
        parts = []
        for i, c in enumerate(self.channels):
            x = self._channel(c.name, data if c.name == "last_action" else sensed, flag_pos)
            std = noise.get(c.name, 0.0)
            if std and c.binary:
                x = jnp.where(jax.random.uniform(keys[i], x.shape) < std, 1.0 - x, x)
            elif std:
                x = x + std * jax.random.normal(keys[i], x.shape)
            parts.append(x)
        return jnp.concatenate(parts)

    # ---- API -----------------------------------------------------------------------------
    def reset(self, rng: jax.Array, flag_pos: jax.Array | None = None) -> State:
        if flag_pos is None:
            flag_pos = self.sample_flag(rng)
        data = self._data0.replace(mocap_pos=self._data0.mocap_pos.at[self._mocap].set(flag_pos))
        data = mjx.forward(self.mjx_model, data)
        dist = self._dist(data, flag_pos)
        zero = jnp.zeros(())
        no = jnp.zeros((), dtype=bool)
        # fold_in, not split: the flag drawn from rng stays the benchmark's.
        k_noise, k_obs = jax.random.split(jax.random.fold_in(rng, 1))
        k_flag = jax.random.fold_in(rng, 2)
        none = jnp.zeros((), dtype=jnp.int32)
        return State(
            data=data, obs=self._window(None, self.observe(data, flag_pos, k_obs)),
            reward=zero, done=no,
            success=no, fell=no, truncated=no, flag_pos=flag_pos, dist=dist, start_dist=dist,
            step=jnp.zeros((), dtype=jnp.int32),
            metrics={k: zero for k in METRIC_KEYS}, tags=none, first_tag=none - 1,
            rng=k_noise, flag_rng=k_flag,
        )

    def step(self, state: State, action: jax.Array) -> State:
        c, w = self.config, self.config.reward
        data = state.data.replace(ctrl=self.action_to_ctrl(action, state.data.ctrl))
        def physics(_, d: mjx.Data) -> mjx.Data:
            return mjx.step(self.mjx_model, d)

        data = jax.lax.fori_loop(0, self.n_substeps, physics, data)

        dist = self._dist(data, state.flag_pos)
        cos_tilt = self._cos_tilt(data)
        fell = cos_tilt < jnp.cos(c.fall_tilt_rad)
        success = (dist < c.success_heights * self.height) & ~fell  # tagged the current flag
        step = state.step + 1
        flag_pos, flag_rng, start_dist, next_dist = (
            state.flag_pos, state.flag_rng, state.start_dist, dist)
        if c.flags == "chain":
            if flag_rng is None:
                raise ValueError("chain mode needs State.flag_rng to draw the next flag")
            flag_rng, k_flag = jax.random.split(flag_rng)
            new_flag = self.sample_flag(k_flag, data.xpos[self._central][:2])
            flag_pos = jnp.where(success, new_flag, state.flag_pos)
            data = data.replace(mocap_pos=data.mocap_pos.at[self._mocap].set(flag_pos))
            next_dist = jnp.where(success, self._dist(data, flag_pos), dist)
            start_dist = jnp.where(success, next_dist, state.start_dist)
            truncated = (step >= c.max_steps) & ~fell
            done = fell | truncated
        else:
            truncated = (step >= c.max_steps) & ~fell & ~success
            done = fell | success | truncated
        metrics = {
            # against the flag TARS was walking to, so a respawn never jumps the reward
            "progress": w.progress * (state.dist - dist),
            "upright": w.upright * cos_tilt,
            "fall": -w.fall * fell.astype(jnp.float32),
            "effort": -w.effort * jnp.mean(jnp.square(jnp.clip(action, -1.0, 1.0))),
            "tag": w.tag * success.astype(jnp.float32),
        }
        rng, k_obs = jax.random.split(state.rng) if state.rng is not None else (None, None)
        sensed = state.data if c.sensor_delay_steps else data
        return State(
            data=data, obs=self._window(state.obs, self.observe(data, flag_pos, k_obs, sensed)),
            reward=sum(metrics.values()), done=done, success=success,
            fell=fell, truncated=truncated, flag_pos=flag_pos, dist=next_dist,
            start_dist=start_dist, step=step, metrics=metrics,
            tags=state.tags + success.astype(jnp.int32),
            first_tag=jnp.where(success & (state.first_tag < 0), step, state.first_tag),
            rng=rng, flag_rng=flag_rng,
        )
