"""Flag sensor: the one swap point for how TARS sees the flag (A19).

Any sensor takes (body position, body rotation matrix, flag position), all world frame, and
returns FLAG_OBS_DIM floats. A camera-based estimate later replaces `ground_truth_flag` with
the same signature and output layout, so the policy's input does not change.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

FLAG_OBS_DIM = 3

FlagSensor = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def ground_truth_flag(
    body_pos: jax.Array, body_rot: jax.Array, flag_pos: jax.Array
) -> jax.Array:
    """Exact flag position relative to the central body, in the central body's own frame (m)."""
    return body_rot.T @ (flag_pos - body_pos)


def check_sensor(sensor: FlagSensor) -> FlagSensor:
    out = jax.eval_shape(sensor, jnp.zeros(3), jnp.eye(3), jnp.zeros(3))
    if out.shape != (FLAG_OBS_DIM,):
        raise ValueError(f"flag sensor must return shape ({FLAG_OBS_DIM},), got {out.shape}")
    return sensor
