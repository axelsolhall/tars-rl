import math

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from tars_rl.env import eval as env_eval
from tars_rl.env.arena import FLAG_BODY, build_arena_mjcf
from tars_rl.env.config import EnvConfig
from tars_rl.env.flag_env import FlagEnv
from tars_rl.env.rollout import FlagRollout
from tars_rl.env.sensors import check_sensor, ground_truth_flag


@pytest.fixture(scope="module")
def env():
    return FlagEnv(config=EnvConfig(episode_seconds=0.2))


@pytest.fixture(scope="module")
def fns(env):
    return jax.jit(env.reset), jax.jit(env.step)


def test_arena_has_flag_that_never_collides(env):
    m = mujoco.MjModel.from_xml_string(build_arena_mjcf(env.spec))
    flag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, FLAG_BODY)
    assert m.body_mocapid[flag] == 0
    geoms = np.flatnonzero(m.geom_bodyid == flag)
    assert len(geoms) == 2
    assert np.all(m.geom_contype[geoms] == 0) and np.all(m.geom_conaffinity[geoms] == 0)
    assert m.nu == len(env.spec.actuator_names)


def test_sizes(env):
    n = len(env.spec.actuator_names)
    assert env.action_size == n == 4
    assert env.observation_size == 3 + 2 * n + 6


def test_action_maps_onto_ctrlrange(env):
    lo, hi = env.mj_model.actuator_ctrlrange.T
    assert np.allclose(env.action_to_ctrl(-jnp.ones(4)), lo, atol=1e-6)
    assert np.allclose(env.action_to_ctrl(jnp.ones(4)), hi, atol=1e-6)
    assert np.allclose(env.action_to_ctrl(jnp.zeros(4)), (lo + hi) / 2, atol=1e-6)
    assert np.allclose(env.action_to_ctrl(5 * jnp.ones(4)), hi, atol=1e-6)  # clipped


def test_flag_placement_follows_a22(env):
    flags = jax.vmap(env.sample_flag)(jax.random.split(jax.random.PRNGKey(0), 2000))
    r = np.linalg.norm(np.asarray(flags[:, :2]) - env.start_xy, axis=1) / env.height
    assert r.min() >= 3.0 - 1e-4 and r.max() <= 10.0 + 1e-4
    assert np.allclose(np.asarray(flags[:, 2]), 0.0)
    ang = np.arctan2(*(np.asarray(flags[:, :2]) - env.start_xy).T[::-1])
    counts, _ = np.histogram(ang, bins=4, range=(-math.pi, math.pi))
    assert counts.min() > 400  # every quadrant used


def test_reset_observation(env, fns):
    reset, _ = fns
    s = reset(jax.random.PRNGKey(1))
    obs = np.asarray(s.obs)
    assert obs.shape == (env.observation_size,)
    assert np.all(np.isfinite(obs))
    assert np.linalg.norm(obs[:3]) == pytest.approx(
        np.linalg.norm(np.asarray(s.flag_pos) - np.asarray(s.data.xpos[env._central])), rel=1e-4)
    assert np.allclose(obs[3 + 8 : 3 + 11], [0, 0, -1], atol=1e-4)  # upright: gravity down
    assert not bool(s.done)
    assert float(s.dist) == pytest.approx(float(s.start_dist))


def test_sensor_is_swappable():
    with pytest.raises(ValueError):
        check_sensor(lambda p, r, f: f[:2])
    rot = jnp.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])  # yawed 90 deg
    v = ground_truth_flag(jnp.zeros(3), rot, jnp.array([0.0, 2.0, 0.0]))
    assert np.allclose(v, [2.0, 0.0, 0.0], atol=1e-6)  # flag dead ahead in the body frame
    env = FlagEnv(config=EnvConfig(episode_seconds=0.2), flag_sensor=lambda p, r, f: jnp.ones(3))
    s = env.reset(jax.random.PRNGKey(0))
    assert np.allclose(s.obs[:3], 1.0)


def test_zero_action_stands_and_times_out(env, fns):
    reset, step = fns
    s = reset(jax.random.PRNGKey(2))
    for _ in range(env.config.max_steps):
        assert not bool(s.done)
        s = step(s, jnp.zeros(4))
        assert float(s.metrics["upright"]) > 0.99 * env.config.reward.upright
        assert float(s.metrics["fall"]) == 0.0
    assert bool(s.truncated) and bool(s.done) and not bool(s.fell) and not bool(s.success)
    assert int(s.step) == env.config.max_steps


def test_success_when_flag_within_one_height(env, fns):
    reset, step = fns
    flag = jnp.array([env.start_xy[0] + 0.5 * env.height, env.start_xy[1], 0.0])
    s = step(reset(jax.random.PRNGKey(0), flag), jnp.zeros(4))
    assert bool(s.success) and bool(s.done) and not bool(s.truncated)


def test_fall_when_tilted_past_60_degrees(env, fns):
    reset, step = fns
    s = reset(jax.random.PRNGKey(0))
    q = np.asarray(s.data.qpos).copy()
    a = math.radians(70) / 2  # pitch 70 deg about y, lifted so it is not inside the floor
    q[3:7] = [math.cos(a), 0, math.sin(a), 0]
    q[2] += 0.3
    s = s._replace(data=s.data.replace(qpos=jnp.asarray(q, dtype=s.data.qpos.dtype)))
    s = step(s, jnp.zeros(4))
    assert bool(s.fell) and bool(s.done)
    assert float(s.metrics["fall"]) == -env.config.reward.fall


def test_reward_terms_sum(env, fns):
    reset, step = fns
    s = step(reset(jax.random.PRNGKey(3)), jnp.full(4, 0.5))
    assert float(s.reward) == pytest.approx(sum(float(v) for v in s.metrics.values()), abs=1e-6)
    assert float(s.metrics["effort"]) == pytest.approx(-env.config.reward.effort * 0.25)
    assert float(s.metrics["progress"]) == pytest.approx(
        env.config.reward.progress * (float(s.start_dist) - float(s.dist)), abs=1e-6)


def test_rollout_roundtrip_keeps_trajectory_keys(tmp_path):
    q = np.random.default_rng(0).normal(size=(5, 13))
    p = FlagRollout(q, 0.02, 1, np.array([1.0, 2.0, 0.0])).save(tmp_path / "r.npz")
    r = FlagRollout.load(p)
    assert np.allclose(r.qpos, q) and r.dt == 0.02 and r.body_spec_version == 1
    assert np.allclose(r.flag_pos, [1, 2, 0])
    with np.load(p) as d:
        assert {"version", "qpos", "dt", "body_spec_version", "flag_pos"} <= set(d.files)


def test_eval_runs_small(tmp_path):
    res = env_eval.run(4, 0, EnvConfig(episode_seconds=0.2), tmp_path / "ep0.npz")
    for name in ("random", "zero"):
        s = res["policies"][name]
        assert s["episodes"] == 4
        assert 0.0 <= s["success_rate"] <= 1.0
        assert s["length_steps"] <= res["max_steps"]
    # same flags for both policies
    assert res["policies"]["random"]["start_dist_m"] == pytest.approx(
        res["policies"]["zero"]["start_dist_m"])
    r = FlagRollout.load(tmp_path / "ep0.npz")
    assert r.qpos.shape[1] == 7 + 4 and len(r.qpos) >= 2
