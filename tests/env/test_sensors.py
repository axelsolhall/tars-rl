"""Mark 2 senses (T24) and the guarantee that the default observation is the benchmark's."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from tars_rl.env.arena import build_arena_mjcf
from tars_rl.env.config import EnvConfig
from tars_rl.env.flag_env import FlagEnv
from tars_rl.env.sense_check import expected_touch, sense_check, tipped
from tars_rl.env.sensors import OBS_SETS, touch_edges

SHORT = 0.2


@pytest.fixture(scope="module")
def env2():
    return FlagEnv(config=EnvConfig(obs="mark2", episode_seconds=SHORT))


def _benchmark_obs(env, data, flag_pos):
    """The 17-float observation exactly as runs/full-seed0 was trained on (T13's observe)."""
    rot = data.xmat[env._central]
    return jnp.concatenate([
        rot.T @ (flag_pos - data.xpos[env._central]),
        data.qpos[env._qadr],
        data.qvel[env._dadr],
        -rot[2, :],
        data.qvel[env._angvel],
    ])


def test_default_obs_is_benchmark_bit_for_bit():
    env = FlagEnv(config=EnvConfig(episode_seconds=SHORT))
    assert env.observation_size == 17
    assert [c.name for c in env.channels] == list(OBS_SETS["default"])
    # same model as the benchmark's: no extra sites or sensors
    m0 = mujoco.MjModel.from_xml_string(build_arena_mjcf(env.spec))
    assert env.mj_model.nsensor == m0.nsensor == 0 and env.mj_model.nsite == m0.nsite
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    ref = jax.jit(lambda d, f: _benchmark_obs(env, d, f))
    s = reset(jax.random.PRNGKey(7))
    acts = jax.random.uniform(jax.random.PRNGKey(8), (8, 4), minval=-1, maxval=1)
    for a in acts:
        assert np.array_equal(np.asarray(s.obs), np.asarray(ref(s.data, s.flag_pos)))
        s = step(s, a)


def test_mark2_layout(env2):
    sl = env2.channel_slices()
    assert env2.observation_size == 3 + 4 + 4 + 3 * 6 + 10 + 4 == 43
    assert list(sl) == list(OBS_SETS["mark2"])
    assert sl["touch"].stop - sl["touch"].start == len(touch_edges()) == 10
    for c in env2.channels:
        assert c.units and c.lo < c.hi and c.size == len(c.labels)


def test_mark2_standing_reads_gravity_and_bottom_edges(env2):
    reset, step = jax.jit(env2.reset), jax.jit(env2.step)
    s = reset(jax.random.PRNGKey(0))
    for _ in range(10):
        s = step(s, jnp.zeros(4))
    sl, o = env2.channel_slices(), np.asarray(s.obs)
    for b in ("central", "left_arm", "right_arm"):
        assert np.allclose(o[sl[f"{b}_acc"]], [0, 0, 9.81], atol=0.05)
        assert np.allclose(o[sl[f"{b}_gyro"]], 0, atol=0.05)
    lit = {n for (n, *_), v in zip(touch_edges(), o[sl["touch"]], strict=True) if v}
    assert lit == {"central_bottom_front", "central_bottom_back", "left_bottom_front",
                   "left_bottom_back", "right_bottom_front", "right_bottom_back"}


def test_mark2_last_action_is_the_clipped_action(env2):
    reset, step = jax.jit(env2.reset), jax.jit(env2.step)
    s = reset(jax.random.PRNGKey(0))
    assert np.allclose(s.obs[env2.channel_slices()["last_action"]], 0.0, atol=1e-6)
    a = jnp.array([0.3, -0.7, 2.0, -0.1])
    s = step(s, a)
    assert np.allclose(s.obs[env2.channel_slices()["last_action"]], jnp.clip(a, -1, 1),
                       atol=1e-5)


def test_arm_accelerometer_gives_arm_tilt(env2):
    reset, step = jax.jit(env2.reset), jax.jit(env2.step)
    s = reset(jax.random.PRNGKey(0))
    act = jnp.array([0.2, 0.0, 0.0, 0.0])  # left hinge only
    for _ in range(40):
        s = step(s, act)
    sl, o = env2.channel_slices(), np.asarray(s.obs)
    hinge = float(o[sl["joint_pos"]][0])
    tilt_c = np.arctan2(*o[sl["central_acc"]][[0, 2]])
    tilt_l = np.arctan2(*o[sl["left_arm_acc"]][[0, 2]])
    assert abs(hinge) > 0.3
    assert abs(abs(tilt_l - tilt_c) - abs(hinge)) < 0.05  # arm tilt minus body tilt = hinge


@pytest.mark.parametrize("face", ["front", "back"])
def test_tipped_lights_that_face(env2, face):
    reset, step = jax.jit(env2.reset), jax.jit(env2.step)
    s = tipped(env2, reset(jax.random.PRNGKey(0)), face)
    for _ in range(30):
        s = step(s, jnp.zeros(4))
    assert np.array_equal(np.asarray(s.obs[env2.channel_slices()["touch"]]),
                          expected_touch(face))


def test_noise_and_delay_are_off_by_default_and_work_when_set():
    base = FlagEnv(config=EnvConfig(obs="mark2", episode_seconds=SHORT))
    noisy = FlagEnv(config=EnvConfig(obs="mark2", episode_seconds=SHORT,
                                     sensor_noise={"central_acc": 0.5, "touch": 1.0}))
    late = FlagEnv(config=EnvConfig(obs="mark2", episode_seconds=SHORT, sensor_delay_steps=1))
    k, a = jax.random.PRNGKey(0), jnp.full(4, 0.4)
    s0, s1, s2 = base.reset(k), noisy.reset(k), late.reset(k)
    s0b, s1b, s2b = base.step(s0, a), noisy.step(s1, a), late.step(s2, a)
    sl = base.channel_slices()
    o0, o1, o2 = (np.asarray(x.obs) for x in (s0b, s1b, s2b))
    assert not np.allclose(o0[sl["central_acc"]], o1[sl["central_acc"]])
    assert np.array_equal(o1[sl["touch"]], 1.0 - o0[sl["touch"]])  # flip probability 1
    assert np.array_equal(o0[sl["joint_vel"]], o1[sl["joint_vel"]])  # no noise asked
    # delayed: senses are the previous step's, the last action is this step's
    assert np.array_equal(o2[sl["joint_pos"]], np.asarray(s0.obs)[sl["joint_pos"]])
    assert np.array_equal(o2[sl["last_action"]], o0[sl["last_action"]])
    with pytest.raises(ValueError):
        FlagEnv(config=EnvConfig(sensor_noise={"touch": 0.1}))  # not in the default set
    with pytest.raises(ValueError):
        FlagEnv(config=EnvConfig(obs="mark3"))


def test_sense_check_passes(env2):
    r = sense_check(env2, seed=0, settle_s=0.4)
    assert r["ok"], r["tipped"]
    assert [c["name"] for c in r["channels"]] == list(OBS_SETS["mark2"])
