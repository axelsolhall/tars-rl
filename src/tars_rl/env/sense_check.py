"""Mark 2 senses check (T24), run by `python -m tars_rl.env.eval --obs mark2`.

A short scripted run (stand, wiggle every joint, lie on the front, lie on the back) gives every
channel's name, units, nominal range and the range actually seen. Then two tipped checks: TARS
laid flat on its front and on its back, after settling, must light exactly the touch edges of
that face, and all three accelerometers must read gravity's reaction (+g, pointing away from the
floor) along the expected axis of their own frame.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from tars_rl.env.flag_env import FlagEnv, State
from tars_rl.env.sensors import IMU_BODIES, touch_edges

G = 9.81
# Pitch about y: +90 deg puts the +x (front) face down; "up" is then -x in every slab's frame
# (hinges at 0, so the arms are parallel to the central body).
TIPS = {"front": (math.pi / 2, (-1.0, 0.0, 0.0)), "back": (-math.pi / 2, (1.0, 0.0, 0.0))}


def tipped(env: FlagEnv, state: State, face: str) -> State:
    """`state` with TARS laid flat on its front or back face, just above the floor, at rest,
    joints at their neutral pose (same xy as `state`)."""
    m = env.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(env._data0.qpos)
    d.qpos[:2] = np.asarray(state.data.qpos[:2])
    d.ctrl[:] = np.asarray(env._data0.ctrl)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0.0, 1.0, 0.0]), TIPS[face][0])
    d.qpos[3:7] = q
    d.qpos[2] = env.spec.central.depth / 2 + 0.005
    mujoco.mj_forward(m, d)
    return state._replace(data=mjx.put_data(m, d))


def expected_touch(face: str) -> np.ndarray:
    """Lying on a face, the edges of that face touch the floor (central: its bottom one only)."""
    return np.array([float(n.endswith(face)) for n, *_ in touch_edges()])


def sense_check(env: FlagEnv, seed: int = 0, settle_s: float = 1.0) -> dict:
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    n = round(settle_s / env.dt)
    s = reset(jax.random.PRNGKey(seed))
    frames = [s.obs]
    t = np.arange(n) * env.dt
    rates = 0.7 + 0.3 * np.arange(env.action_size)  # Hz, one per actuator
    script = [np.zeros(env.action_size)] * n + [0.5 * np.sin(2 * np.pi * rates * x) for x in t]
    for a in script:
        s = step(s, jnp.asarray(a, jnp.float32))
        frames.append(s.obs)
    end = {}
    for face in TIPS:
        st = tipped(env, s, face)
        for _ in range(n):
            st = step(st, jnp.zeros(env.action_size))
            frames.append(st.obs)
        end[face] = np.asarray(st.obs)
    obs = np.stack([np.asarray(f) for f in frames])

    sl = env.channel_slices()
    table = [{"name": c.name, "units": c.units, "size": c.size, "nominal": [c.lo, c.hi],
              "seen": [float(obs[:, sl[c.name]].min()), float(obs[:, sl[c.name]].max())]}
             for c in env.channels]
    names = [e for e, *_ in touch_edges()]
    tipped_res = {}
    for face, o in end.items():
        lit, want = o[sl["touch"]], expected_touch(face)
        up = np.asarray(TIPS[face][1])
        acc = {}
        for b in IMU_BODIES:
            a = o[sl[f"{b}_acc"]]
            cos = float(a @ up / max(float(np.linalg.norm(a)), 1e-9))
            acc[b] = {"reading": a.tolist(), "expected": (G * up).tolist(), "cos": cos,
                      "ok": bool(cos > 0.98 and abs(np.linalg.norm(a) - G) < 0.5)}
        tipped_res[face] = {
            "lit": [e for e, v in zip(names, lit, strict=True) if v > 0.5],
            "expected_lit": [e for e, v in zip(names, want, strict=True) if v],
            "touch_ok": bool(np.array_equal(lit, want)), "acc": acc,
        }
    ok = all(r["touch_ok"] and all(a["ok"] for a in r["acc"].values())
             for r in tipped_res.values())
    return {"obs_set": env.config.obs, "observation_size": env.observation_size,
            "script_steps": len(frames), "channels": table, "tipped": tipped_res, "ok": ok}


def report_senses(r: dict) -> None:
    print(f"\nSenses '{r['obs_set']}': {r['observation_size']} floats. Seen = range over "
          f"{r['script_steps']} scripted steps (stand, wiggle, lie on front, lie on back)\n")
    print(f"{'channel':<16}{'n':>3}  {'units':<34}{'nominal':>18}{'seen':>20}")
    for c in r["channels"]:
        nom = "[{:.3g}, {:.3g}]".format(*c["nominal"])
        seen = "[{:.3g}, {:.3g}]".format(*c["seen"])
        print(f"{c['name']:<16}{c['size']:>3}  {c['units']:<34}{nom:>18}{seen:>20}")
    for face, c in r["tipped"].items():
        verdict = "OK" if c["touch_ok"] else "FAIL"
        print(f"\nLying on its {face}:")
        print(f"  touch lit       {', '.join(c['lit']) or '-'}")
        print(f"  touch expected  {', '.join(c['expected_lit'])}  {verdict}")
        for b, a in c["acc"].items():
            rd = " ".join(f"{v:+6.2f}" for v in a["reading"])
            ex = " ".join(f"{v:+6.2f}" for v in a["expected"])
            verdict = "OK" if a["ok"] else "FAIL"
            print(f"  {b + '_acc':<15} ({rd}) m/s^2, expected ({ex}), "
                  f"cos {a['cos']:+.3f}  {verdict}")
    print(f"\nTipped checks: {'all OK' if r['ok'] else 'FAILED'}")
