"""TARS's tuning space: which PPO settings and network shapes the campaign searches (A109).

Training settings are the PPOConfig fields brax PPO exposes and that keep the batch rules valid
for any value (batch size, minibatches and env count stay as the profile has them). Each tower
(policy `p_*`, value `v_*`) gets its own block, depth, width, activation and norm (A92).

`START` is the centre of the starting box: MuJoCo Playground's PPO settings for its walking
robots (Go1 joystick, = `train.config.PROFILES["full"]`, A59) and T15's default networks, so the
campaign begins at the published point and the box moves away only if trials say so.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tars_rl.train.config import PROFILES, PPOConfig
from tars_rl.train.networks import NetSpec, NetworkSpec
from tars_rl.tune.space import Choice, Float, Int, Space

BLOCKS = ("mlp", "simba", "bro")
WIDTHS = (32, 64, 128, 256, 512)
ACTIVATIONS = ("swish", "relu", "gelu", "elu")
NORMS = ("none", "layer", "rms")


def _tower(prefix: str) -> tuple:
    return (
        Choice(f"{prefix}_block", BLOCKS),
        Int(f"{prefix}_depth", 1, 6),
        Choice(f"{prefix}_width", WIDTHS),
        Choice(f"{prefix}_activation", ACTIVATIONS),
        Choice(f"{prefix}_norm", NORMS),
    )


SPACE = Space((
    Float("learning_rate", 3e-5, 3e-3, log=True),
    Float("entropy_cost", 1e-4, 1e-1, log=True),
    Float("discounting", 0.95, 0.995),
    Int("unroll_length", 5, 40, log=True),
    Int("num_updates_per_batch", 1, 8),
    Float("max_grad_norm", 0.1, 10.0, log=True),
    Float("reward_scaling", 0.1, 10.0, log=True),
    *_tower("p"),
    *_tower("v"),
))

PPO_KEYS = ("learning_rate", "entropy_cost", "discounting", "unroll_length",
            "num_updates_per_batch", "max_grad_norm", "reward_scaling")


def _tower_settings(prefix: str, s: NetSpec) -> dict[str, Any]:
    return {f"{prefix}_block": s.block, f"{prefix}_depth": s.depth, f"{prefix}_width": s.width,
            f"{prefix}_activation": s.activation, f"{prefix}_norm": s.norm}


def published_start() -> dict[str, Any]:
    """The published PPO settings (Go1 joystick) and T15's networks, as search settings."""
    full = PROFILES["full"]
    return ({k: getattr(full, k) for k in PPO_KEYS}
            | _tower_settings("p", full.network.policy) | _tower_settings("v", full.network.value))


START = published_start()


def network_of(settings: dict[str, Any]) -> NetworkSpec:
    """The network description a trial trains. Residual blocks need a norm, so a residual block
    drawn with norm `none` gets a layer norm (the trial record keeps both)."""

    def tower(prefix: str) -> NetSpec:
        block, norm = settings[f"{prefix}_block"], settings[f"{prefix}_norm"]
        if block != "mlp" and norm == "none":
            norm = "layer"
        return NetSpec(block=block, depth=int(settings[f"{prefix}_depth"]),
                       width=int(settings[f"{prefix}_width"]),
                       activation=settings[f"{prefix}_activation"], norm=norm)

    return NetworkSpec(tower("p"), tower("v")).check()


def to_config(settings: dict[str, Any], base: PPOConfig) -> PPOConfig:
    """Search settings applied on top of a profile (which keeps env counts, batches, length)."""
    ppo = {k: settings[k] for k in PPO_KEYS}
    ppo["unroll_length"] = int(ppo["unroll_length"])
    ppo["num_updates_per_batch"] = int(ppo["num_updates_per_batch"])
    return replace(base, **ppo, network=network_of(settings)).check()
