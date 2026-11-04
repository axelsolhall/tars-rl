"""Domain randomization: every parallel training env gets its own TARS and floor, so the policy
cannot rely on one exact simulator and has a chance on the real robot.

Randomized per env (drawn once when training starts, fixed for the run, as MuJoCo Playground does):

    friction        sliding friction of every surface, absolute (spec nominal 0.8)
    mass            each body's mass and inertia, scaled (body 2.0 kg, arms 0.5 kg)
    motor strength  each actuator's gain kp and force limit, scaled together (weaker/stronger servo)
    centre of mass  the central body's centre of mass, shifted (battery, electronics placement)
    friction hardness  MuJoCo's impratio, log-uniform, when set (needs EnvConfig.friction_cone
                    "elliptic"): 1 is MuJoCo's soft default, which lets TARS creep under pushes
                    static friction should hold; 10 is near Bullet's hard friction (2026-11-04)

Ranges are `DomainRandConfig`; the defaults sit inside what legged-robot sim-to-real work commonly
uses (motor strength 0.8-1.2, friction ~0.4-1.25). Not here yet, by design: pushes, action
latency and sensor noise in training (the training adapter carries no per-step noise key yet),
and ranges that grow over training (a curriculum).

With rough ground (`EnvConfig.terrain`), each env also starts on its own patch of the terrain
(`terrain_fields_fn`, `tars_rl.env.terrain`). `for_env` returns what brax PPO's
`randomization_fn` wants: `(mjx_model, rng[num_envs]) -> (batched model, in_axes)`.
Eval: `python -m tars_rl.env.randomize` samples models, prints the drawn ranges and lets every
sampled TARS stand with zero action (none should fall or blow up).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from tars_rl.env import terrain
from tars_rl.env.terrain import TerrainConfig

ROBOT_BODIES = ("central", "left_arm", "right_arm")


@dataclass(frozen=True)
class DomainRandConfig:
    friction: tuple[float, float] = (0.4, 1.25)
    mass_scale: tuple[float, float] = (0.8, 1.2)
    motor_strength: tuple[float, float] = (0.8, 1.2)
    com_offset_m: float = 0.01  # each axis, +-
    impratio: tuple[float, float] | None = None  # None: the model's own (1)

    def check(self) -> DomainRandConfig:
        names = ("friction", "mass_scale", "motor_strength")
        names += ("impratio",) if self.impratio else ()
        for name in names:
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(f"{name} range {lo, hi} must satisfy 0 < low <= high")
        if self.com_offset_m < 0:
            raise ValueError(f"com_offset_m {self.com_offset_m} must be >= 0")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def _uniform(key: jax.Array, shape: tuple[int, ...], bounds: tuple[float, float]) -> jax.Array:
    return jax.random.uniform(key, shape, minval=bounds[0], maxval=bounds[1])


def body_fields_fn(cfg: DomainRandConfig, body_ids: tuple[int, ...], central_id: int):
    """`(model, rng[n]) -> {field: batched value}` for the body and floor randomization.
    `body_ids`: the robot's bodies (masses scaled); `central_id`: whose centre of mass moves."""
    cfg.check()
    bodies = jnp.asarray(body_ids)

    def draw_all(model: mjx.Model, rng: jax.Array) -> dict:
        @jax.vmap
        def draw(key):
            k_f, k_m, k_s, k_c = jax.random.split(key, 4)
            friction = model.geom_friction.at[:, 0].set(_uniform(k_f, (), cfg.friction))
            scale = _uniform(k_m, (bodies.size,), cfg.mass_scale)
            mass = model.body_mass.at[bodies].multiply(scale)
            inertia = model.body_inertia.at[bodies].multiply(scale[:, None])
            s = _uniform(k_s, (model.nu,), cfg.motor_strength)
            gain = model.actuator_gainprm.at[:, 0].multiply(s)
            # position actuator: force = kp*ctrl - kp*q - kv*qdot, so kp sits in gain and bias
            bias = model.actuator_biasprm.at[:, 1].multiply(s)
            force = model.actuator_forcerange * s[:, None]
            shift = jax.random.uniform(k_c, (3,), minval=-cfg.com_offset_m,
                                       maxval=cfg.com_offset_m)
            ipos = model.body_ipos.at[central_id].add(shift)
            return friction, mass, inertia, gain, bias, force, ipos

        friction, mass, inertia, gain, bias, force, ipos = draw(rng)
        out = {"geom_friction": friction, "body_mass": mass, "body_inertia": inertia,
               "actuator_gainprm": gain, "actuator_biasprm": bias,
               "actuator_forcerange": force, "body_ipos": ipos}
        if cfg.impratio is not None:
            if int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
                raise ValueError("impratio randomization needs EnvConfig(friction_cone="
                                 "'elliptic'): on the pyramidal cone it barely changes friction")
            # A key of its own (folded), so the draws above stay what they were without it.
            lo, hi = np.log(cfg.impratio)
            out["opt.impratio"] = jnp.exp(jax.vmap(
                lambda k: _uniform(jax.random.fold_in(k, 11), (), (lo, hi)))(rng))
        return out

    return draw_all


def terrain_fields_fn(cfg: TerrainConfig, body_id: int):
    """`(model, rng[n]) -> {field: batched value}`: each env stands on its own pad of the shared
    terrain, turned its own way (the heightfield's body moves; TARS still starts at the origin)."""

    def place(model: mjx.Model, rng: jax.Array) -> dict:
        _, _, pos, quat = terrain.placements(cfg, rng)
        return {"body_pos": jax.vmap(lambda p: model.body_pos.at[body_id].set(p))(pos),
                "body_quat": jax.vmap(lambda q: model.body_quat.at[body_id].set(q))(quat)}

    return place


def as_randomization_fn(*field_fns):
    """brax PPO's `randomization_fn`: `(model, rng[n]) -> (batched model, in_axes)`, from
    functions that each draw their own (disjoint) model fields."""

    def randomize(model: mjx.Model, rng: jax.Array):
        fields: dict = {}
        for fn in field_fns:
            drawn = fn(model, rng)
            if fields.keys() & drawn.keys():
                raise ValueError(f"two randomizations set {fields.keys() & drawn.keys()}")
            fields |= drawn
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        return model.tree_replace(fields), in_axes.tree_replace({k: 0 for k in fields})

    return randomize


def make_randomization_fn(cfg: DomainRandConfig, body_ids: tuple[int, ...], central_id: int):
    """Body and floor randomization alone, as brax's `randomization_fn`."""
    return as_randomization_fn(body_fields_fn(cfg, body_ids, central_id))


def for_env(env, cfg: DomainRandConfig | None):
    """brax's randomization function for a FlagEnv: the body randomization `cfg` (None: off)
    plus, when the env has terrain, a pad of it per env. None when there is nothing to vary.
    Each part draws from the env's key on its own, so adding terrain leaves the bodies as they
    were; terrain folds the key first (`terrain.placements`) so pads and bodies are independent."""
    m = env.mj_model
    fns = []
    if cfg is not None:
        ids = tuple(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ROBOT_BODIES)
        fns.append(body_fields_fn(cfg, ids, ids[0]))
    if env.config.terrain is not None:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, terrain.TERRAIN_GEOM)
        fns.append(terrain_fields_fn(env.config.terrain, bid))
    return as_randomization_fn(*fns) if fns else None


def main(argv: list[str] | None = None) -> int:
    from tars_rl.env.config import EnvConfig
    from tars_rl.env.flag_env import FlagEnv

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--seconds", type=float, default=3.0, help="zero-action stand per model")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    cfg = DomainRandConfig()
    env = FlagEnv(config=EnvConfig(episode_seconds=a.seconds))
    keys = jax.random.split(jax.random.PRNGKey(a.seed), a.envs)
    model_v, in_axes = for_env(env, cfg)(env.mjx_model, keys)
    m = env.mj_model
    force_v, force = model_v.actuator_forcerange, m.actuator_forcerange
    print(f"Domain randomization eval: {a.envs} sampled TARS, config {cfg.to_dict()}\n")
    rows = [
        ("friction", model_v.geom_friction[:, 0, 0], m.geom_friction[0, 0]),
        ("mass central (kg)", model_v.body_mass[:, 1], m.body_mass[1]),
        ("mass left arm (kg)", model_v.body_mass[:, 2], m.body_mass[2]),
        ("hinge force limit (Nm)", force_v[:, 0, 1], force[0, 1]),
        ("slide force limit (N)", force_v[:, 2, 1], force[2, 1]),
        ("COM shift x (mm)", 1000 * (model_v.body_ipos[:, 1, 0] - m.body_ipos[1, 0]), 0.0),
    ]
    print(f"{'parameter':<24}{'min':>9}{'max':>9}{'nominal':>9}")
    for name, v, nominal in rows:
        v = np.asarray(v)
        print(f"{name:<24}{v.min():>9.3f}{v.max():>9.3f}{nominal:>9.3f}")

    def stand(model, key):
        old = env.mjx_model
        env.mjx_model = model
        try:
            zero = jnp.zeros(env.action_size)
            s = jax.lax.fori_loop(0, env.config.max_steps, lambda _, s: env.step(s, zero),
                                  env.reset(key))
            return s.fell, env._cos_tilt(s.data), jnp.isfinite(s.data.qpos).all()
        finally:
            env.mjx_model = old

    keys = jax.random.split(jax.random.PRNGKey(a.seed + 1), a.envs)
    fell, cos_tilt, finite = jax.jit(jax.vmap(stand, in_axes=(in_axes, 0)))(model_v, keys)
    tilt = np.degrees(np.arccos(np.clip(np.asarray(cos_tilt), -1, 1)))
    print(f"\nZero-action stand, {a.seconds:.0f} s: fell {int(np.asarray(fell).sum())}/{a.envs}, "
          f"max tilt {tilt.max():.1f} deg, non-finite states {int((~np.asarray(finite)).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
