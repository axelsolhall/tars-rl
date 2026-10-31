"""Stand + tracking eval for the TARS body. Run: python -m tars_rl.body.eval --seconds 10

Prints, against an unpowered baseline:
  1. stand: max tilt, COM drop and joint drift at the neutral pose (powered vs unpowered),
     also after a forward push;
  2. per-joint step tracking: final error, 90 % rise time, overshoot, for all four actuators;
  3. COM shift with one arm slid fully off centre;
  4. MJX vs CPU MuJoCo agreement on the same scripted rollout, and MJX speed.
Exits non-zero if a pass criterion fails, so the numbers point at the broken part.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from tars_rl.body.mjcf import load_model, load_spec
from tars_rl.body.spec import BodySpec

FALL_TILT_DEG = 45.0
PUSH_RAD_S = 1.5  # initial pitch rate for the push test
TOL = {"hinge": 0.05, "slide": 0.003}  # final step-tracking error, rad / m
MJX_TOL = {"pos": 0.005, "hinge": 0.02, "slide": 0.002}  # max |MJX - CPU| over the rollout


@dataclass
class Body:
    spec: BodySpec
    model: mujoco.MjModel

    def jid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def qadr(self, name: str) -> int:
        return int(self.model.jnt_qposadr[self.jid(name)])

    def neutral_data(self) -> mujoco.MjData:
        d = mujoco.MjData(self.model)
        for j, v in zip(self.spec.joint_names, self.spec.neutral_ctrl(), strict=True):
            d.qpos[self.qadr(j)] = v
        d.ctrl[:] = self.spec.neutral_ctrl()
        mujoco.mj_forward(self.model, d)
        return d


def unpowered(spec: BodySpec) -> mujoco.MjModel:
    """The same body with every actuator's gain and bias zeroed: joints move freely."""
    m = load_model(spec)
    m.actuator_gainprm[:] = 0
    m.actuator_biasprm[:] = 0
    return m


def tilt_deg(d: mujoco.MjData, body: int = 1) -> float:
    return math.degrees(math.acos(float(np.clip(d.xmat[body][8], -1.0, 1.0))))


def stand(b: Body, model: mujoco.MjModel, seconds: float, push: float = 0.0) -> dict:
    d = mujoco.MjData(model)
    d.qpos[:] = b.neutral_data().qpos
    d.ctrl[:] = b.spec.neutral_ctrl()
    d.qvel[4] = push  # free-joint angular velocity about the body y axis (pitch)
    mujoco.mj_forward(model, d)
    com0 = float(d.subtree_com[0][2])
    neutral = np.array([d.qpos[b.qadr(j)] for j in b.spec.joint_names])
    max_tilt, max_drift = 0.0, np.zeros(4)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, d)
        max_tilt = max(max_tilt, tilt_deg(d))
        q = np.array([d.qpos[b.qadr(j)] for j in b.spec.joint_names])
        max_drift = np.maximum(max_drift, np.abs(q - neutral))
    return {
        "max_tilt_deg": max_tilt,
        "final_tilt_deg": tilt_deg(d),
        "com_drop_mm": 1000 * (com0 - float(d.subtree_com[0][2])),
        "max_joint_drift": dict(zip(b.spec.joint_names, max_drift.tolist(), strict=True)),
        "fell": max_tilt > FALL_TILT_DEG,
    }


def step_response(b: Body, joint: str, target: float, pre: dict[str, float],
                  seconds: float = 1.5) -> dict:
    """Settle 1 s with `pre` targets, then step `joint` to `target` and measure tracking."""
    m = b.model
    d = b.neutral_data()
    names = b.spec.actuator_names
    for j, v in pre.items():
        d.ctrl[names.index(f"{j}_act")] = v
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    a = b.qadr(joint)
    q0 = float(d.qpos[a])
    d.ctrl[names.index(f"{joint}_act")] = target
    span = target - q0
    t90, peak, ts = None, 0.0, m.opt.timestep
    for i in range(int(seconds / ts)):
        mujoco.mj_step(m, d)
        frac = (float(d.qpos[a]) - q0) / span
        peak = max(peak, frac)
        if t90 is None and frac >= 0.9:
            t90 = (i + 1) * ts
    err = abs(float(d.qpos[a]) - target)
    kind = "hinge" if joint.endswith("hinge") else "slide"
    return {
        "joint": joint,
        "from": q0,
        "target": target,
        "final_error": err,
        "rise_90_s": t90,
        "overshoot_pct": 100 * max(0.0, peak - 1.0),
        "tilt_deg_after": tilt_deg(d),
        "ok": err < TOL[kind],
    }


def com_shift(b: Body, seconds: float = 3.0) -> dict:
    """Left arm slid fully up (slide max): static COM shift, then whether it still stands."""
    m = b.model
    d = b.neutral_data()
    com0 = d.subtree_com[0].copy()
    smax = b.spec.slide_range[1]
    d.qpos[b.qadr("left_slide")] = smax
    mujoco.mj_forward(m, d)
    delta = 1000 * (d.subtree_com[0] - com0)
    d = b.neutral_data()
    d.ctrl[b.spec.actuator_names.index("left_slide_act")] = smax
    max_tilt = 0.0
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        max_tilt = max(max_tilt, tilt_deg(d))
    return {
        "slide_m": smax,
        "com_shift_mm": {"x": delta[0], "y": delta[1], "z": delta[2]},
        "max_tilt_deg_holding": max_tilt,
    }


def scripted_ctrl(spec: BodySpec, steps: int, dt: float) -> np.ndarray:
    """Small sinusoids on every actuator, the same for CPU and MJX.

    Both arms are held lifted on their slides so they swing clear of the floor and the body
    stays upright: a rollout that falls over is chaotic and CPU/MJX diverge for that reason
    alone, which would say nothing about MJX. The lift ramps in over 0.3 s: a step saturates
    the slide force, and while saturated MJX lags CPU by up to ~4 mm for ~0.1 s.
    """
    t = np.arange(steps)[:, None] * dt
    ramp = np.clip(t / 0.3, 0.0, 1.0)
    ramp = ramp * ramp * (3 - 2 * ramp)
    offset = np.array([0.0, 0.0, 1.0, 1.0]) * 0.45 * spec.slide_range[1] * ramp
    # The swing starts once the arms are lifted, so the feet never drag on the floor.
    amp = np.array([0.3, 0.3, 0.01, 0.01]) * (t >= 0.3)
    phase = np.array([0.0, math.pi, math.pi / 2, -math.pi / 2])
    return (np.asarray(spec.neutral_ctrl()) + offset
            + amp * np.sin(2 * math.pi * 1.0 * (t - 0.3) + phase))


def mjx_vs_cpu(b: Body, seconds: float = 1.0, batch: int = 64) -> dict:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    m = b.model
    steps = int(seconds / m.opt.timestep)
    ctrl = scripted_ctrl(b.spec, steps, m.opt.timestep)

    d = b.neutral_data()
    cpu_q = np.zeros((steps, m.nq))
    cpu_tilt = 0.0
    for i in range(steps):
        d.ctrl[:] = ctrl[i]
        mujoco.mj_step(m, d)
        cpu_q[i] = d.qpos
        cpu_tilt = max(cpu_tilt, tilt_deg(d))

    mx = mjx.put_model(m)
    dx0 = mjx.put_data(m, b.neutral_data())

    def body(dx, u):
        dx = mjx.step(mx, dx.replace(ctrl=u))
        return dx, dx.qpos

    rollout = jax.jit(lambda dx, us: jax.lax.scan(body, dx, us)[1])
    t0 = time.perf_counter()
    mjx_q = np.asarray(rollout(dx0, jnp.asarray(ctrl, dtype=jnp.float32)))
    compile_s = time.perf_counter() - t0

    diff = np.abs(mjx_q - cpu_q)
    hinge = [b.qadr(f"{s}_hinge") for s in ("left", "right")]
    slide = [b.qadr(f"{s}_slide") for s in ("left", "right")]
    res = {
        "steps": steps,
        "max_pos_diff_m": float(diff[:, 0:3].max()),
        "max_hinge_diff_rad": float(diff[:, hinge].max()),
        "max_slide_diff_m": float(diff[:, slide].max()),
        "cpu_max_tilt_deg": cpu_tilt,
        "compile_plus_first_run_s": compile_s,
    }
    res["ok"] = (res["max_pos_diff_m"] < MJX_TOL["pos"]
                 and res["max_hinge_diff_rad"] < MJX_TOL["hinge"]
                 and res["max_slide_diff_m"] < MJX_TOL["slide"])

    # Speed: a batch of identical bodies, steps/s on whatever device JAX has.
    us = jnp.broadcast_to(jnp.asarray(ctrl, dtype=jnp.float32)[:, None], (steps, batch, m.nu))
    dxb = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (batch, *x.shape)), dx0)

    def bbody(dx, u):
        return jax.vmap(lambda d_, u_: mjx.step(mx, d_.replace(ctrl=u_)))(dx, u), None

    brollout = jax.jit(lambda dx, us: jax.lax.scan(bbody, dx, us)[0].qpos)
    brollout(dxb, us).block_until_ready()
    t0 = time.perf_counter()
    brollout(dxb, us).block_until_ready()
    res["batch"] = batch
    res["device"] = str(jax.devices()[0])
    res["steps_per_s"] = batch * steps / (time.perf_counter() - t0)
    return res


def run(seconds: float, spec_path: str | None = None, mjx_check: bool = True) -> dict:
    spec = load_spec(spec_path)
    model = load_model(spec)
    b = Body(spec, model)
    off = unpowered(spec)
    r: dict = {"spec": spec.name, "version": spec.version, "seconds": seconds}
    r["stand"] = {
        "powered": stand(b, model, seconds),
        "unpowered": stand(b, off, seconds),
        "powered_push": stand(b, model, seconds, push=PUSH_RAD_S),
        "unpowered_push": stand(b, off, seconds, push=PUSH_RAD_S),
    }
    lift = 0.6 * spec.slide_range[1]
    r["tracking"] = [
        # Hinge steps with that arm lifted on its slide so it swings clear of the floor.
        step_response(b, "left_hinge", 0.4, {"left_slide": lift}),
        step_response(b, "right_hinge", -0.4, {"right_slide": lift}),
        step_response(b, "left_slide", 0.04, {}),
        step_response(b, "right_slide", 0.04, {}),
    ]
    r["com_shift"] = com_shift(b)
    if mjx_check:
        r["mjx"] = mjx_vs_cpu(b)
    r["pass"] = {
        "powered_stands": not r["stand"]["powered"]["fell"],
        "tracking": all(t["ok"] for t in r["tracking"]),
        "mjx_agrees": r["mjx"]["ok"] if mjx_check else None,
    }
    return r


def report(r: dict) -> None:
    s = r["stand"]
    print(f"TARS body eval: {r['spec']} v{r['version']}, {r['seconds']:.1f} s stand\n")
    print(f"1. Stand at neutral pose (fall = tilt > {FALL_TILT_DEG:.0f} deg; "
          f"push = {PUSH_RAD_S} rad/s pitch)")
    print(f"   {'case':<16}{'max tilt':>10}{'final tilt':>12}{'COM drop':>11}"
          f"{'max drift h/s':>18}  fell")
    for k, v in s.items():
        dr = v["max_joint_drift"]
        h = max(dr["left_hinge"], dr["right_hinge"])
        sl = max(dr["left_slide"], dr["right_slide"])
        print(f"   {k:<16}{v['max_tilt_deg']:>8.2f} d{v['final_tilt_deg']:>10.2f} d"
              f"{v['com_drop_mm']:>8.1f} mm{h:>9.3f}r/{1000 * sl:.1f}mm  {v['fell']}")
    print("\n2. Step tracking (hinges stepped with that arm lifted on its slide)")
    print(f"   {'joint':<13}{'from':>8}{'target':>8}{'final err':>11}{'rise90':>9}"
          f"{'overshoot':>11}{'tilt':>8}  ok")
    for t in r["tracking"]:
        rise = f"{t['rise_90_s']:.3f}s" if t["rise_90_s"] is not None else "never"
        print(f"   {t['joint']:<13}{t['from']:>8.3f}{t['target']:>8.3f}{t['final_error']:>11.4f}"
              f"{rise:>9}{t['overshoot_pct']:>10.1f}%{t['tilt_deg_after']:>7.1f}d  {t['ok']}")
    print("   (errors in rad for hinges, m for slides; "
          f"tolerance {TOL['hinge']} rad / {TOL['slide']} m)")
    c = r["com_shift"]
    cs = c["com_shift_mm"]
    print(f"\n3. Left arm slid fully off centre ({c['slide_m'] * 1000:.0f} mm): COM shift "
          f"x {cs['x']:+.1f}, y {cs['y']:+.1f}, z {cs['z']:+.1f} mm; "
          f"max tilt while holding {c['max_tilt_deg_holding']:.2f} deg")
    if "mjx" in r:
        x = r["mjx"]
        print(f"\n4. MJX vs CPU MuJoCo, same scripted ctrl, {x['steps']} steps: "
              f"max |diff| body pos {1000 * x['max_pos_diff_m']:.2f} mm, "
              f"hinge {x['max_hinge_diff_rad']:.4f} rad, "
              f"slide {1000 * x['max_slide_diff_m']:.2f} mm  ok={x['ok']}")
        print(f"   (rollout max tilt {x['cpu_max_tilt_deg']:.1f} deg; arms lifted and swinging)")
        print(f"   MJX speed: {x['steps_per_s']:,.0f} steps/s (batch {x['batch']}, {x['device']}), "
              f"compile {x['compile_plus_first_run_s']:.1f} s")
    print(f"\nPASS: {r['pass']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--spec", help="BodySpec JSON (default: committed tars_v1.json)")
    p.add_argument("--no-mjx", action="store_true", help="skip the MJX comparison")
    p.add_argument("--json", help="also write the results to this JSON file")
    a = p.parse_args(argv)
    r = run(a.seconds, a.spec, mjx_check=not a.no_mjx)
    report(r)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=2, default=float)
    return 0 if all(v is not False for v in r["pass"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
