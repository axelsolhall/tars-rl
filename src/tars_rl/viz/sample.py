"""Scripted sample rollout: sine targets on both arm hinges and both arm slides, no policy."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from tars_rl.viz.trajectory import Trajectory


def scripted_sine(
    model: mujoco.MjModel,
    body_spec_version: int,
    seconds: float = 4.0,
    period: float = 2.0,
    hinge_amp: float = 0.5,  # rad
    slide_frac: float = 0.5,  # fraction of the slide's half range
    record_every: int = 5,
) -> Trajectory:
    """Hinges swing in phase; slides run a quarter period behind, so both motions are visible."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(model.nu)
    ]
    lo, hi = model.actuator_ctrlrange.T
    mid = (lo + hi) / 2
    amp = np.zeros(model.nu)
    phase = np.zeros(model.nu)
    for i, name in enumerate(names):
        if name.endswith("hinge_act"):
            amp[i] = hinge_amp
        elif name.endswith("slide_act"):
            amp[i] = slide_frac * (hi[i] - lo[i]) / 2
            phase[i] = -math.pi / 2
    if not amp.any():
        raise ValueError(f"No *_hinge_act / *_slide_act actuators in model: {names}")

    steps = round(seconds / model.opt.timestep)
    frames = []
    for step in range(steps):
        if step % record_every == 0:
            frames.append(data.qpos.copy())
        t = data.time
        data.ctrl[:] = np.clip(mid + amp * np.sin(2 * math.pi * t / period + phase), lo, hi)
        mujoco.mj_step(model, data)
    return Trajectory(
        qpos=np.array(frames),
        dt=record_every * model.opt.timestep,
        body_spec_version=body_spec_version,
    )
