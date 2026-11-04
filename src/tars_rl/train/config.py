"""Training profiles: PPO (A58, A59) and FastTD3 / FastSAC (T21) settings for a quick Mac test
and the real 5080 run.

`full` copies MuJoCo Playground's PPO settings for its own walking robots
(locomotion_params.brax_ppo_config, Go1JoystickFlatTerrain), changed only if TARS fails (A59).
`smoke` only has to prove the pipeline runs end to end on a CPU in a few minutes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from tars_rl.train.memory import MemorySpec
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
        # Transformer memory over 25 frames is the standard since 2026-11-03: with domain
        # randomization it tied gru for the most flags on randomized bodies and led the
        # apartment level (runs/chain-dr), at twice gru's training time.
        network=NetworkSpec(memory=MemorySpec("transformer", 25)),
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


ALGOS = ("fasttd3", "fastsac")


@dataclass(frozen=True)
class OffPolicyConfig:
    """FastTD3 / FastSAC settings (T21). `full` starts from the papers' published settings:
    FastTD3 (arXiv 2505.22642, its MuJoCo Playground defaults) and, for FastSAC and the
    Q-average, the humanoid recipe of arXiv 2512.01996 (holosoma). T19 searches them, not this."""

    algo: str
    num_envs: int
    episode_length: int  # policy steps before a truncation (bootstrapped, not terminal)
    buffer_per_env: int  # replay rows per env; capacity = num_envs * buffer_per_env
    batch_size: int
    total_env_steps: int  # stop after this many env steps ...
    max_wall_s: float | None = None  # ... or this many seconds, whichever comes first
    num_evals: int = 10  # learning-curve points after the one before training, evenly in time
    learning_starts: int = 10  # iterations of uniform random actions before any update
    num_updates: int = 2  # gradient steps per iteration (one step of every env)
    n_step: int = 3
    gamma: float = 0.99
    tau: float = 0.1  # target critic polyak rate
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 0.1  # AdamW
    policy_frequency: int = 2  # actor update every this many critic updates
    q_agg: str = "mean"  # "mean" (2512.01996) or "min" (clipped double-Q)
    num_atoms: int = 101  # distributional (C51) critic
    v_min: float = -10.0
    v_max: float = 10.0
    actor_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (1024, 512, 256)
    normalize_observations: bool = True
    # FastTD3: per-env exploration noise std drawn in [std_min, std_max] at each episode start
    std_min: float = 0.001
    std_max: float = 0.4
    target_noise: float = 0.2  # target policy smoothing
    target_noise_clip: float = 0.5
    # FastSAC: tanh-squashed Gaussian, learned temperature
    alpha_init: float = 0.001
    target_entropy_per_dim: float = 0.0  # target entropy = -this * action_size
    log_std_min: float = -5.0
    log_std_max: float = 0.0
    chunk: int = 50  # iterations per compiled call (between wall-clock checks)
    curve_episodes: int = 128

    def check(self) -> OffPolicyConfig:
        if self.algo not in ALGOS:
            raise ValueError(f"unknown algo {self.algo!r}; known: {ALGOS}")
        if self.q_agg not in ("mean", "min"):
            raise ValueError("q_agg must be 'mean' or 'min'")
        if self.buffer_per_env <= self.n_step:
            raise ValueError("buffer_per_env must exceed n_step")
        if self.learning_starts < self.n_step:
            raise ValueError("learning_starts must be >= n_step so the first batch exists")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


_OFFPOLICY_BASE: dict[str, OffPolicyConfig] = {
    "full": OffPolicyConfig(
        algo="fasttd3", num_envs=1024, episode_length=1000, buffer_per_env=10_240,
        batch_size=32_768, total_env_steps=2_000_000_000, max_wall_s=None,
    ),
    "smoke": OffPolicyConfig(
        algo="fasttd3", num_envs=16, episode_length=100, buffer_per_env=64, batch_size=64,
        total_env_steps=16 * 40, num_evals=2, learning_starts=8, chunk=10,
        actor_hidden=(32, 32), critic_hidden=(32, 32), num_atoms=21, curve_episodes=16,
    ),
}
# FastSAC differs from FastTD3 only in the actor and the per-update actor schedule.
_SAC = {"policy_frequency": 1}


def get_offpolicy_profile(name: str, algo: str, **overrides) -> OffPolicyConfig:
    if name not in _OFFPOLICY_BASE:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(_OFFPOLICY_BASE)}")
    extra = _SAC if algo == "fastsac" else {}
    return replace(_OFFPOLICY_BASE[name], algo=algo, **{**extra, **overrides}).check()
