"""Training profiles: PPO settings for a quick Mac test and the real 5080 run (A58, A59).

`full` copies MuJoCo Playground's PPO settings for its own walking robots
(locomotion_params.brax_ppo_config, Go1JoystickFlatTerrain), changed only if TARS fails (A59).
`smoke` only has to prove the pipeline runs end to end on a CPU in a few minutes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from tars_rl.train.networks import NetSpec, NetworkSpec


@dataclass(frozen=True)
class PPOConfig:
    num_timesteps: int
    num_evals: int  # learning-curve points (brax counts the one before training too)
    episode_length: int  # policy steps; matches EnvConfig.max_steps for `full`
    num_envs: int
    num_eval_envs: int
    batch_size: int
    num_minibatches: int
    unroll_length: int
    num_updates_per_batch: int
    discounting: float = 0.97
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    reward_scaling: float = 1.0
    max_grad_norm: float = 1.0
    normalize_observations: bool = True
    action_repeat: int = 1
    network: NetworkSpec = NetworkSpec()  # policy mlp-4x128-swish, value mlp-5x256-swish (T18)
    curve_episodes: int = 128  # own flag eval at every learning-curve point (A61)

    def check(self) -> PPOConfig:
        """Brax PPO's batch rules, checked here so a bad profile fails before compiling."""
        if (self.batch_size * self.num_minibatches) % self.num_envs:
            raise ValueError("batch_size * num_minibatches must be a multiple of num_envs")
        per_step = self.batch_size * self.unroll_length * self.num_minibatches * self.action_repeat
        self.network.check()
        if self.num_timesteps < per_step:
            raise ValueError(f"num_timesteps {self.num_timesteps} < one training step {per_step}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


PROFILES: dict[str, PPOConfig] = {
    "full": PPOConfig(
        num_timesteps=200_000_000,
        num_evals=10,
        episode_length=1000,
        num_envs=8192,
        num_eval_envs=128,
        batch_size=256,
        num_minibatches=32,
        unroll_length=20,
        num_updates_per_batch=4,
    ),
    "smoke": PPOConfig(
        num_timesteps=2_560,
        num_evals=2,
        episode_length=100,
        num_envs=32,
        num_eval_envs=16,
        batch_size=32,
        num_minibatches=4,
        unroll_length=10,
        num_updates_per_batch=2,
        network=NetworkSpec(NetSpec(depth=2, width=32), NetSpec(depth=2, width=32)),
        curve_episodes=16,
    ),
}


def get_profile(name: str, **overrides) -> PPOConfig:
    if isinstance(overrides.get("network"), str):
        overrides["network"] = NetworkSpec.parse(overrides["network"])
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(PROFILES)}")
    return replace(PROFILES[name], **overrides).check()
