"""Train module tests: profiles, the Playground adapter and the policy folder. CPU, no training."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs

from tars_rl.env.flag_env import FlagEnv
from tars_rl.train.adapter import METRICS, PlaygroundFlagEnv
from tars_rl.train.config import PROFILES, get_profile
from tars_rl.train.networks import network_factory
from tars_rl.train.policy import (
    PROGRESS_FILE,
    load_policy,
    make_inference,
    read_progress,
    save_policy,
)


@pytest.fixture(scope="module")
def env() -> FlagEnv:
    return FlagEnv()


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_profiles_pass_brax_batch_rules(name):
    get_profile(name)


def test_full_profile_episode_matches_env_timeout(env):
    assert get_profile("full").episode_length == env.config.max_steps


def test_bad_profile_rejected():
    with pytest.raises(ValueError):
        get_profile("smoke", num_envs=7)
    with pytest.raises(KeyError):
        get_profile("nope")


def test_adapter_reset_and_step(env):
    penv = PlaygroundFlagEnv(env)
    s = penv.reset(jax.random.PRNGKey(0))
    assert s.obs.shape == (env.observation_size,)
    assert set(s.metrics) == set(METRICS)
    s2 = penv.step(s, jnp.zeros(env.action_size))
    assert set(s2.metrics) == set(s.metrics)
    assert s2.reward.dtype == jnp.float32 and s2.done.dtype == jnp.float32
    # The flag rides in data, so it survives a step unchanged.
    np.testing.assert_allclose(penv.flag_pos(s2.data), penv.flag_pos(s.data))


def test_training_wrapper_draws_new_flag_each_episode(env):
    """A22: an auto-reset during training must draw a new flag, not replay the first one."""
    from tars_rl.train.run import WRAP_FOR_TRAINING

    wenv = WRAP_FOR_TRAINING(PlaygroundFlagEnv(env), episode_length=1)
    s = wenv.reset(jax.random.split(jax.random.PRNGKey(2), 4))
    first = np.asarray(s.data.mocap_pos)
    for _ in range(2):  # every step ends an episode (length 1), so each one resets
        s = wenv.step(s, jnp.zeros((4, env.action_size)))
        assert not np.allclose(np.asarray(s.data.mocap_pos), first)
        first = np.asarray(s.data.mocap_pos)


def test_adapter_progress_uses_distance_from_data(env):
    """After an auto-reset only data is swapped back; progress must come from that data."""
    penv = PlaygroundFlagEnv(env)
    s = penv.reset(jax.random.PRNGKey(1))
    direct = env.step(env.reset(jax.random.PRNGKey(1)), jnp.zeros(env.action_size))
    via = penv.step(s, jnp.zeros(env.action_size))
    np.testing.assert_allclose(via.metrics["progress"], direct.metrics["progress"], atol=1e-6)


def test_policy_folder_roundtrip(env, tmp_path):
    cfg = get_profile("smoke")
    nets = network_factory(cfg.network)(env.observation_size, env.action_size)
    norm = running_statistics.init_state(specs.Array((env.observation_size,), jnp.float32))
    params = (norm, nets.policy_network.init(jax.random.PRNGKey(0)))
    meta = {
        "profile": "smoke", "ppo": cfg.to_dict(), "env": env.config.to_dict(),
        "body_spec_version": env.spec.version, "observation_size": env.observation_size,
        "action_size": env.action_size,
    }
    save_policy(tmp_path, params, meta)
    saved = load_policy(tmp_path)
    obs = jax.random.normal(jax.random.PRNGKey(1), (3, env.observation_size))
    a = saved.policy()(jax.random.PRNGKey(0), obs)
    assert a.shape == (3, env.action_size)
    inf = make_inference(env.observation_size, env.action_size, cfg.network)
    want = inf(params, deterministic=True)(obs, jax.random.PRNGKey(0))[0]
    np.testing.assert_allclose(a, want, atol=1e-6)


def test_read_progress(tmp_path):
    assert read_progress(tmp_path) == []
    lines = [{"kind": "curve", "step": 1}, {"kind": "ppo", "step": 1}]
    (tmp_path / PROGRESS_FILE).write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert read_progress(tmp_path) == lines
