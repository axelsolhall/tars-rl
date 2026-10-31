"""FlagEnv (T13) seen through MuJoCo Playground's env interface, so its PPO can train on it (A58).

Nothing but `data` is carried: the flag lives in the arena's mocap body (T13 puts it there) and
the previous flag distance is recomputed from `data` every step. Training wraps this with
Playground's auto-reset in `full_reset` mode (see `run.train`), so each finished episode calls
`FlagEnv.reset` and draws a new flag (A22); the default mode would replay each env's first flag.

Episode length is enforced by Playground's EpisodeWrapper (PPOConfig.episode_length), not by
FlagEnv's own timeout, which never fires here because the step counter is not carried.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground import State

from tars_rl.env.flag_env import FlagEnv
from tars_rl.env.flag_env import State as EnvState

METRICS = ("progress", "upright", "fall", "effort", "success", "fell")


class PlaygroundFlagEnv:
    def __init__(self, env: FlagEnv) -> None:
        self.env = env

    # Attributes Playground's wrappers and brax PPO read.
    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def mj_model(self):
        return self.env.mj_model

    @property
    def mjx_model(self):
        return self.env.mjx_model

    @property
    def dt(self) -> float:
        return self.env.dt

    @property
    def unwrapped(self) -> PlaygroundFlagEnv:
        return self

    def flag_pos(self, data: mjx.Data) -> jax.Array:
        return data.mocap_pos[self.env._mocap]

    def _env_state(self, data: mjx.Data, obs: jax.Array) -> EnvState:
        flag = self.flag_pos(data)
        dist = self.env._dist(data, flag)
        zero = jnp.zeros(())
        no = jnp.zeros((), dtype=bool)
        return EnvState(
            data=data, obs=obs, reward=zero, done=no, success=no, fell=no, truncated=no,
            flag_pos=flag, dist=dist, start_dist=dist, step=jnp.zeros((), dtype=jnp.int32),
            metrics={},
        )

    @staticmethod
    def _metrics(s: EnvState) -> dict[str, jax.Array]:
        m = {k: jnp.asarray(s.metrics.get(k, 0.0), jnp.float32) for k in METRICS[:4]}
        m["success"] = s.success.astype(jnp.float32)
        m["fell"] = s.fell.astype(jnp.float32)
        return m

    def reset(self, rng: jax.Array) -> State:
        s = self.env.reset(rng)
        return State(
            data=s.data, obs=s.obs, reward=jnp.zeros(()), done=jnp.zeros(()),
            metrics=self._metrics(s), info={},
        )

    def step(self, state: State, action: jax.Array) -> State:
        s = self.env.step(self._env_state(state.data, state.obs), action)
        return state.replace(
            data=s.data, obs=s.obs, reward=s.reward.astype(jnp.float32),
            # merge: wrappers add their own keys (e.g. "reward") at reset; the carry must keep them
            done=s.done.astype(jnp.float32), metrics={**state.metrics, **self._metrics(s)},
        )
