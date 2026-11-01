"""TARS's policy and value networks, built from a short saved description (A77).

A `NetSpec` describes one tower (policy or value) in five fields, and has a short string form:

    mlp-4x128-swish            plain layers, as T15 trained (the default; same params, same init)
    mlp-3x256-swish-layer      plain layers with a norm after each Dense (Dense -> norm -> act)
    simba-2x128-relu-layer-e4  SimBa residual blocks (Lee et al. 2024): x + MLP(norm(x)),
                               inverted bottleneck of width*expansion, final norm before the head
    bro-2x128-relu-layer       BroNet residual blocks (Nauman et al. 2024): Dense-norm-act stem,
                               x + norm(Dense(act(norm(Dense(x))))) per block

`depth` counts hidden layers for `mlp` and residual blocks for `simba`/`bro`. A `NetworkSpec`
holds one `NetSpec` per tower, so the tuner (T19) can shape policy and value separately. Every
tower ends in a plain Dense head (action-distribution params or one value) and uses brax's
lecun_uniform init, so `mlp` without a norm is exactly brax's MLP and T15's policies.

`make_ppo_networks` returns brax's `PPONetworks`, so brax PPO trains them unchanged (A58).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen

BLOCKS = ("mlp", "simba", "bro")
ACTIVATIONS: dict[str, Callable] = {
    "swish": linen.swish, "relu": linen.relu, "gelu": linen.gelu, "elu": linen.elu,
    "tanh": linen.tanh, "mish": lambda x: x * jnp.tanh(jax.nn.softplus(x)),
}
NORMS: dict[str, Callable[[], linen.Module] | None] = {
    "none": None, "layer": linen.LayerNorm, "rms": linen.RMSNorm,
}
KERNEL_INIT = jax.nn.initializers.lecun_uniform  # brax's default for both towers


@dataclass(frozen=True)
class NetSpec:
    block: str = "mlp"
    depth: int = 4
    width: int = 128
    activation: str = "swish"
    norm: str = "none"
    expansion: int = 4  # simba only: inner width = width * expansion

    def check(self) -> NetSpec:
        if self.block not in BLOCKS:
            raise ValueError(f"block {self.block!r} not in {BLOCKS}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation {self.activation!r} not in {sorted(ACTIVATIONS)}")
        if self.norm not in NORMS:
            raise ValueError(f"norm {self.norm!r} not in {sorted(NORMS)}")
        if self.block != "mlp" and self.norm == "none":
            raise ValueError(f"{self.block} blocks need a norm (layer or rms)")
        if self.depth < (0 if self.block == "mlp" else 1) or self.width < 1 or self.expansion < 1:
            raise ValueError(f"bad size in {self}")
        return self

    def __str__(self) -> str:
        s = f"{self.block}-{self.depth}x{self.width}-{self.activation}"
        if self.norm != "none":
            s += f"-{self.norm}"
        if self.block == "simba":
            s += f"-e{self.expansion}"
        return s

    @classmethod
    def parse(cls, text: str) -> NetSpec:
        """Inverse of `str`: 'block-DEPTHxWIDTH-activation[-norm][-eEXPANSION]'."""
        block, size, activation, *rest = text.split("-")
        depth, width = (int(v) for v in size.split("x"))
        kw: dict = {}
        for r in rest:
            if r.startswith("e") and r[1:].isdigit():
                kw["expansion"] = int(r[1:])
            else:
                kw["norm"] = r
        return cls(block=block, depth=depth, width=width, activation=activation, **kw).check()

    @classmethod
    def from_hidden(cls, hidden, activation: str = "swish") -> NetSpec:
        """A T15-style hidden-size tuple as a plain `mlp` (all layers must share one width)."""
        hidden = tuple(hidden)
        if len(set(hidden)) > 1:
            raise ValueError(f"hidden sizes {hidden} are not one width")
        return cls(depth=len(hidden), width=hidden[0] if hidden else 1, activation=activation)


@dataclass(frozen=True)
class NetworkSpec:
    policy: NetSpec = NetSpec(depth=4, width=128)
    value: NetSpec = NetSpec(depth=5, width=256)

    def check(self) -> NetworkSpec:
        self.policy.check()
        self.value.check()
        return self

    def __str__(self) -> str:
        return f"policy={self.policy} value={self.value}"

    @classmethod
    def parse(cls, text: str) -> NetworkSpec:
        """'policy=<NetSpec> value=<NetSpec>', or one NetSpec for both towers."""
        parts = dict(p.split("=", 1) for p in text.split()) if "=" in text else None
        if parts is None:
            return cls(NetSpec.parse(text), NetSpec.parse(text))
        return cls(NetSpec.parse(parts["policy"]), NetSpec.parse(parts["value"]))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> NetworkSpec:
        return cls(NetSpec(**d["policy"]), NetSpec(**d["value"])).check()


class Tower(linen.Module):
    """One tower: hidden layers or residual blocks from a NetSpec, then a plain Dense head."""

    spec: NetSpec
    out_size: int

    @linen.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        s = self.spec
        act = ACTIVATIONS[s.activation]
        norm = NORMS[s.norm]
        dense = lambda n, name: linen.Dense(n, kernel_init=KERNEL_INIT(), name=name)  # noqa: E731
        if s.block == "mlp":
            for i in range(s.depth):  # names as brax's MLP, so its params load unchanged
                x = dense(s.width, f"hidden_{i}")(x)
                if norm is not None:
                    x = norm(name=f"norm_{i}")(x)
                x = act(x)
            return dense(self.out_size, f"hidden_{s.depth}")(x)
        if s.block == "simba":
            x = dense(s.width, "embed")(x)
            for i in range(s.depth):
                h = norm(name=f"block_{i}_norm")(x)
                h = act(dense(s.width * s.expansion, f"block_{i}_up")(h))
                x = x + dense(s.width, f"block_{i}_down")(h)
            x = norm(name="final_norm")(x)
        else:  # bro
            x = act(norm(name="embed_norm")(dense(s.width, "embed")(x)))
            for i in range(s.depth):
                h = act(norm(name=f"block_{i}_norm_a")(dense(s.width, f"block_{i}_a")(x)))
                x = x + norm(name=f"block_{i}_norm_b")(dense(s.width, f"block_{i}_b")(h))
        return dense(self.out_size, "head")(x)


def _feed_forward(module: linen.Module, obs_size: int, preprocess, squeeze: bool):
    """A brax FeedForwardNetwork around `module`: normalise obs, then apply (as brax does)."""

    def apply(processor_params, params, obs):
        out = module.apply(params, preprocess(obs, processor_params))
        return jnp.squeeze(out, axis=-1) if squeeze else out

    dummy = jnp.zeros((1, networks._get_obs_state_size(obs_size, "state")))  # int or (n,)
    return networks.FeedForwardNetwork(init=lambda key: module.init(key, dummy), apply=apply)


DEFAULT_NETWORK = NetworkSpec()
IDENTITY = types.identity_observation_preprocessor


def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = IDENTITY,
    spec: NetworkSpec = DEFAULT_NETWORK,
) -> ppo_networks.PPONetworks:
    """brax's PPONetworks (tanh-normal actions, as brax's default) with towers from `spec`."""
    spec.check()
    dist = distribution.NormalTanhDistribution(event_size=action_size)
    return ppo_networks.PPONetworks(
        policy_network=_feed_forward(Tower(spec.policy, dist.param_size), observation_size,
                                     preprocess_observations_fn, squeeze=False),
        value_network=_feed_forward(Tower(spec.value, 1), observation_size,
                                    preprocess_observations_fn, squeeze=True),
        parametric_action_distribution=dist,
    )


def network_factory(spec: NetworkSpec):
    """What brax PPO's `network_factory` wants: `(obs_size, act_size, preprocess_fn) -> nets`."""

    def factory(observation_size, action_size, preprocess_observations_fn=None, **_):
        pre = preprocess_observations_fn or IDENTITY
        return make_ppo_networks(observation_size, action_size, pre, spec)

    return factory


def param_count(params) -> int:
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))


def tower_sizes(spec: NetworkSpec, obs_size: int, act_size: int) -> dict[str, int]:
    """Parameter counts of each tower, from shapes only (nothing is run)."""
    nets = make_ppo_networks(obs_size, act_size, spec=spec)
    key = jax.random.PRNGKey(0)
    p = jax.eval_shape(nets.policy_network.init, key)
    v = jax.eval_shape(nets.value_network.init, key)
    return {"policy": param_count(p), "value": param_count(v)}
