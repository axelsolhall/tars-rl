"""Rough-ground training terrain: the grid, its placement per env, and the point contact."""

import functools

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from tars_rl.env import terrain
from tars_rl.env.config import EnvConfig
from tars_rl.env.flag_env import FlagEnv
from tars_rl.env.randomize import DomainRandConfig, for_env
from tars_rl.env.terrain import TERRAIN_GEOM, TerrainConfig

CFG = TerrainConfig()


@pytest.fixture(scope="module")
def z():
    return terrain.heights(CFG)


@pytest.fixture(scope="module")
def env():
    return FlagEnv(config=EnvConfig(episode_seconds=1.0, terrain=CFG))


def _surface(z, xy):
    """The heightfield's surface from the grid, independent of `point_hfield`: MuJoCo splits
    each cell along its (c, r)-(c+1, r+1) diagonal."""
    fx, fy = (xy[:, 0] + CFG.size_m / 2) / CFG.cell_m, (xy[:, 1] + CFG.size_m / 2) / CFG.cell_m
    c, r = np.floor(fx).astype(int), np.floor(fy).astype(int)
    u, v = fx - c, fy - r
    z00, z10, z01, z11 = z[r, c], z[r, c + 1], z[r + 1, c], z[r + 1, c + 1]
    return np.where(v >= u, z00 + (z11 - z01) * u + (z01 - z00) * v,
                    z00 + (z10 - z00) * u + (z11 - z10) * v)


def test_terrain_is_reproducible_flat_on_pads_and_rougher_along_x(z):
    assert np.array_equal(z, terrain.heights(CFG))
    assert z.min() >= 0.0
    ax = -CFG.size_m / 2 + CFG.cell_m * np.arange(CFG.n)
    x, y = np.meshgrid(ax, ax, indexing="xy")
    for px, py in CFG.pads():
        near = np.hypot(x - px, y - py) <= CFG.pad_flat_m
        assert np.all(z[near] == 0.0), f"pad ({px}, {py}) is not flat"
    d = CFG.difficulty(ax)
    assert np.all(z[:, d == 0] == 0.0), "difficulty 0 is the flat floor"
    rough = [z[:, (d > lo) & (d <= lo + 0.25)].std() for lo in (0.0, 0.25, 0.5, 0.75)]
    assert rough == sorted(rough), f"not rougher along x: {rough}"


def test_config_survives_the_saved_meta():
    cfg = EnvConfig(terrain=TerrainConfig(difficulty_max=0.5))
    import json

    back = EnvConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert back == cfg
    old = {k: v for k, v in EnvConfig().to_dict().items() if k != "terrain"}
    assert EnvConfig.from_dict(old).terrain is None, "runs saved before terrain stay flat"


def test_cpu_mujoco_moves_the_terrain_with_its_body(env):
    # World geoms never move in CPU MuJoCo after compiling; the terrain hangs on its own body so
    # a placement means the same in both engines.
    m = env.mj_model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TERRAIN_GEOM)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, TERRAIN_GEOM)
    m2 = mujoco.MjModel.from_xml_string(terrain.add_to_mjcf(
        __import__("tars_rl.env.arena", fromlist=["x"]).build_arena_mjcf(env.spec), CFG))
    m2.body_quat[bid] = [np.cos(0.35), 0, 0, np.sin(0.35)]
    d = mujoco.MjData(m2)
    mujoco.mj_kinematics(m2, d)
    assert np.allclose(d.geom_xmat[gid][:2], [np.cos(0.7), -np.sin(0.7)], atol=1e-6)


def _settle(env, model, d0, seconds=1.0):
    step = jax.jit(lambda d: jax.lax.fori_loop(0, round(seconds / 0.002),
                                               lambda _, x: mjx.step(model, x), d))
    return step(d0)


def test_nominal_terrain_rests_tars_exactly_like_the_plane(env):
    flat = FlagEnv(config=EnvConfig(episode_seconds=1.0))
    rest = {}
    for name, e in (("plane", flat), ("terrain", env)):
        d = mujoco.MjData(e.mj_model)
        d.qpos[2] += 0.05
        rest[name] = np.asarray(_settle(e, e.mjx_model, mjx.put_data(e.mj_model, d)).qpos[:7])
    assert np.abs(rest["terrain"] - rest["plane"]).max() < 1e-4


def test_point_contact_holds_tars_on_rough_ground(env, z):
    # Drop TARS onto rough patches; at rest no contact sphere may be inside the ground (checked
    # against the grid itself) and several must touch it.
    m = env.mj_model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TERRAIN_GEOM)
    spheres = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE]
    rng = np.random.default_rng(1)
    for _ in range(3):
        yaw = rng.uniform(-np.pi, np.pi)
        m.body_pos[bid] = [-rng.uniform(14, 26), rng.uniform(-20, 20), 0]  # difficulty 0.7-1
        m.body_quat[bid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d = mujoco.MjData(m)
        d.qpos[2] += 0.12
        rest = _settle(env, mjx.put_model(m), mjx.put_data(m, d))
        e = mujoco.MjData(m)
        e.qpos[:] = np.asarray(rest.qpos)
        mujoco.mj_kinematics(m, e)
        local = (e.geom_xpos[spheres] - e.xpos[bid]) @ e.xmat[bid].reshape(3, 3)
        gap = local[:, 2] - terrain.CONTACT_RADIUS_M - _surface(z, local[:, :2])
        assert gap.min() > -1e-3, f"a sphere rests {-gap.min() * 1000:.1f} mm inside the ground"
        assert (gap < 1e-3).sum() >= 3, "TARS hangs in the air"
    m.body_pos[bid] = env.mj_model.body_pos[bid]  # module fixture: leave it nominal


def test_point_contact_is_the_surface_under_the_sphere(z):
    from mujoco.mjx._src.collision_types import GeomInfo, HFieldInfo

    h = HFieldInfo(jnp.zeros(3), jnp.eye(3), np.array([CFG.size_m / 2, CFG.size_m / 2, 1, 0.1]),
                   CFG.n, CFG.n, jnp.asarray(z.T))  # MJX indexes data[col, row]
    rng = np.random.default_rng(0)
    xy = rng.uniform(5, 25, (64, 2))
    top = _surface(z, xy)
    fn = jax.vmap(lambda p: terrain.point_hfield(h, GeomInfo(p, jnp.eye(3), jnp.array(
        [terrain.CONTACT_RADIUS_M, 0, 0])))[0])
    dist = np.asarray(fn(jnp.asarray(np.c_[xy, top + 0.01])))
    # 1 cm above the surface along the vertical: the normal distance is shorter on a slope
    assert np.all(dist <= 0.01 - terrain.CONTACT_RADIUS_M + 1e-6)
    assert np.all(dist > 0.006 - terrain.CONTACT_RADIUS_M)


def test_each_env_stands_on_its_own_patch_and_bodies_are_drawn_as_before(env):
    keys = jax.random.split(jax.random.PRNGKey(2), 8)
    model_v, axes = for_env(env, DomainRandConfig())(env.mjx_model, keys)
    assert axes.body_pos == 0 and axes.body_quat == 0
    bid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, TERRAIN_GEOM)
    pos = np.asarray(model_v.body_pos[:, bid])
    assert len({tuple(p.round(6)) for p in pos}) == 8
    flat = FlagEnv(config=EnvConfig(episode_seconds=1.0))
    bodies_only, _ = for_env(flat, DomainRandConfig())(flat.mjx_model, keys)
    robot = slice(0, flat.mj_model.nbody)  # the terrain body comes after the robot's
    assert np.array_equal(np.asarray(model_v.body_mass)[:, robot],
                          np.asarray(bodies_only.body_mass)), \
        "adding terrain changed the body draws"
    # the placement puts the env's pad under the origin
    pad, yaw, p, _ = terrain.placements(CFG, keys)
    c, s = np.cos(np.asarray(yaw)), np.sin(np.asarray(yaw))
    pads = CFG.pads()[np.asarray(pad)]
    world = np.stack([c * pads[:, 0] - s * pads[:, 1], s * pads[:, 0] + c * pads[:, 1]], 1)
    assert np.allclose(world + np.asarray(p)[:, :2], 0.0, atol=1e-5)


def test_tars_stands_on_every_kind_of_pad(env):
    keys = jax.random.split(jax.random.PRNGKey(4), 16)
    model_v, axes = for_env(env, None)(env.mjx_model, keys)

    def stand(model, key):
        old = env.mjx_model
        env.mjx_model = model
        try:
            zero = jnp.zeros(env.action_size)
            s = jax.lax.fori_loop(0, env.config.max_steps, lambda _, s: env.step(s, zero),
                                  env.reset(key))
            return s.fell, env._cos_tilt(s.data)
        finally:
            env.mjx_model = old

    fell, cos_tilt = jax.jit(jax.vmap(stand, in_axes=(axes, 0)))(model_v, keys)
    assert not np.asarray(fell).any() and np.asarray(cos_tilt).min() > 0.99


def test_training_wrapper_gives_each_env_its_terrain(env):
    from mujoco_playground import wrapper

    from tars_rl.train.adapter import PlaygroundFlagEnv

    n = 4
    keys = jax.random.split(jax.random.PRNGKey(5), n)
    fn = functools.partial(for_env(env, None), rng=keys)
    w = wrapper.wrap_for_brax_training(PlaygroundFlagEnv(env), episode_length=20,
                                       randomization_fn=fn, full_reset=True)
    s = jax.jit(w.reset)(jnp.stack([jax.random.PRNGKey(7)] * n))
    bid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, TERRAIN_GEOM)
    xpos = np.asarray(s.data.xpos[:, bid])
    assert len({tuple(p.round(5)) for p in xpos}) == n, "every env sees the same terrain"
