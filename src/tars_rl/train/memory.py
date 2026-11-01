"""Memory for TARS's networks: encoders that read the last K frames of senses (a 0.5 s window).

The env puts the window in the observation itself (`EnvConfig.history_steps`, oldest frame
first), so brax PPO trains every kind unchanged: each is a feed-forward function of the window,
recomputed every step, with no hidden state to carry between steps. A `MemorySpec` has a short
string form, `<kind>-<steps>`:

    none            the current frame only (no memory; the default)
    stack-25        frame stacking: the whole window, flattened, straight into the tower
    tcn-25          1D temporal convolutions over the window, as RMA's adaptation module
                    (Kumar et al. 2021)
    gru-25          a GRU run over the window, its last hidden state as the summary
    transformer-25  a small causal transformer over the window, its last token as the summary,
                    as in Radosavovic et al. 2024 (humanoid locomotion)

Every kind but `stack` gives the tower [current frame, summary], so the tower always sees the
present exactly and the encoder only adds what happened before.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen

MEMORY_KINDS = ("none", "stack", "tcn", "gru", "transformer")


@dataclass(frozen=True)
class MemorySpec:
    kind: str = "none"
    steps: int = 0  # frames in the window, the current one included; 0 for `none`
    size: int = 64  # summary width (tcn, gru, transformer)

    def check(self) -> MemorySpec:
        if self.kind not in MEMORY_KINDS:
            raise ValueError(f"memory kind {self.kind!r} not in {MEMORY_KINDS}")
        if (self.kind == "none") != (self.steps == 0):
            raise ValueError(f"memory {self.kind!r} with steps={self.steps}: 'none' takes 0 "
                             "steps, every other kind at least 2, e.g. 'gru-25'")
        if self.kind != "none" and (self.steps < 2 or self.size < 1):
            raise ValueError(f"bad size in {self}")
        if self.kind == "tcn" and self.steps < 8:
            raise ValueError(f"tcn needs at least 8 steps, got {self.steps}")
        return self

    def __str__(self) -> str:
        return "none" if self.kind == "none" else f"{self.kind}-{self.steps}"

    @classmethod
    def parse(cls, text: str) -> MemorySpec:
        """Inverse of `str`: 'none' or '<kind>-<steps>'."""
        if text == "none":
            return cls()
        kind, steps = text.rsplit("-", 1)
        return cls(kind=kind, steps=int(steps)).check()


class HistoryEncoder(linen.Module):
    """(..., steps * frame) window -> what the tower reads (see the module docstring)."""

    spec: MemorySpec
    frame: int

    @linen.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        s = self.spec
        if s.kind == "stack":
            return x
        lead = x.shape[:-1]
        h = x.reshape((-1, s.steps, self.frame))
        current = h[:, -1, :]
        if s.kind == "tcn":
            h = linen.swish(linen.Dense(s.size, name="embed")(h))
            for i, (k, st) in enumerate(((5, 2), (3, 1))):
                conv = linen.Conv(s.size, (k,), strides=(st,), padding="VALID", name=f"conv_{i}")
                h = linen.swish(conv(h))
            summary = linen.Dense(s.size, name="out")(h.reshape((h.shape[0], -1)))
        elif s.kind == "gru":
            summary = linen.RNN(linen.GRUCell(s.size), name="gru")(h)[:, -1, :]
        else:  # transformer: 2 pre-norm blocks, 4 heads, learned positions, causal mask
            t = linen.Dense(s.size, name="embed")(h)
            pos = self.param("pos", linen.initializers.normal(0.02), (s.steps, s.size))
            t = t + pos
            mask = linen.make_causal_mask(jnp.ones((t.shape[0], s.steps)))
            for i in range(2):
                a = linen.LayerNorm(name=f"block_{i}_norm_a")(t)
                t = t + linen.MultiHeadDotProductAttention(
                    num_heads=4, qkv_features=s.size, name=f"block_{i}_attn")(a, a, mask=mask)
                m = linen.LayerNorm(name=f"block_{i}_norm_m")(t)
                m = linen.Dense(s.size, name=f"block_{i}_mlp_out")(
                    linen.gelu(linen.Dense(4 * s.size, name=f"block_{i}_mlp_in")(m)))
                t = t + m
            summary = linen.LayerNorm(name="final_norm")(t[:, -1, :])
        out = jnp.concatenate([current, summary], axis=-1)
        return out.reshape(lead + (out.shape[-1],))
