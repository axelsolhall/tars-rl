"""FastTD3 / FastSAC networks: plain ReLU MLPs, no LayerNorm or residuals (FastTD3 found they
slowed training without gains), a C51 distributional twin critic, and the policy-folder inference.

The saved params are `(normalizer, actor)` like PPO's, so `policy.as_policy` and the eval, curve
and rollout code run on them unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics


class MLP(nn.Module):
    hidden: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(self.out)(x)


def actor_module(algo: str, act_size: int, hidden: Sequence[int]) -> MLP:
    """TD3: tanh(mean). SAC: mean and log-std, so twice the outputs."""
    return MLP(tuple(hidden), act_size * (2 if algo == "fastsac" else 1))


def critic_module(num_atoms: int, hidden: Sequence[int]) -> MLP:
    return MLP(tuple(hidden), num_atoms)


def sac_split(out: jax.Array, log_std_min: float, log_std_max: float):
    mean, log_std = jnp.split(out, 2, axis=-1)
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (jnp.tanh(log_std) + 1)
    return mean, log_std


def sac_sample(out: jax.Array, rng: jax.Array, log_std_min: float, log_std_max: float):
    """Tanh-squashed Gaussian sample and its log-probability (summed over action dims)."""
    mean, log_std = sac_split(out, log_std_min, log_std_max)
    std = jnp.exp(log_std)
    u = mean + std * jax.random.normal(rng, mean.shape)
    a = jnp.tanh(u)
    logp = -0.5 * (((u - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    logp = logp - jnp.log(1 - a**2 + 1e-6)
    return a, logp.sum(-1)


def normalize(norm, obs: jax.Array, enabled: bool) -> jax.Array:
    return running_statistics.normalize(obs, norm) if enabled else obs


def make_inference(algo: str, act_size: int, actor_hidden, normalize_observations: bool = True,
                   log_std: tuple[float, float] = (-5.0, 0.0)):
    """`params -> (obs, rng) -> (action, extras)`, the brax shape `policy.as_policy` expects.

    Deterministic: TD3's tanh output, SAC's tanh(mean). Stochastic: SAC samples; TD3 adds no noise
    (exploration noise is per-env training state, not part of the policy)."""
    actor = actor_module(algo, act_size, actor_hidden)

    def make(params, deterministic: bool = True):
        norm, actor_params = params

        def act(obs, rng):
            out = actor.apply(actor_params, normalize(norm, obs, normalize_observations))
            if algo == "fasttd3":
                return jnp.tanh(out), {}
            if deterministic:
                return jnp.tanh(sac_split(out, *log_std)[0]), {}
            return sac_sample(out, rng, *log_std)[0], {}

        return act

    return make
