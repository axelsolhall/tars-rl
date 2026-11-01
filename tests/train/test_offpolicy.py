"""FastTD3 / FastSAC pieces (T21): profiles, replay n-step returns, C51 projection, policy folder.
CPU, no training (the smoke runs cover that)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs

from tars_rl.env.flag_env import FlagEnv
from tars_rl.train import offpolicy_nets as nets
from tars_rl.train.config import ALGOS, get_offpolicy_profile
from tars_rl.train.offpolicy import project, replay_add, replay_init, replay_sample
from tars_rl.train.policy import load_policy, save_policy


@pytest.fixture(scope="module")
def env() -> FlagEnv:
    return FlagEnv()


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("name", ["smoke", "full"])
def test_profiles_valid(algo, name):
    assert get_offpolicy_profile(name, algo).algo == algo


def test_full_profile_follows_papers(env):
    c = get_offpolicy_profile("full", "fasttd3")
    assert c.episode_length == env.config.max_steps
    assert c.batch_size == 32_768
    assert c.critic_hidden == (1024, 512, 256) and c.actor_hidden == (512, 256, 128)
    assert c.q_agg == "mean"


def test_bad_offpolicy_profile_rejected():
    with pytest.raises(ValueError):
        get_offpolicy_profile("smoke", "ppo")
    with pytest.raises(ValueError):
        get_offpolicy_profile("smoke", "fasttd3", q_agg="max")
    with pytest.raises(KeyError):
        get_offpolicy_profile("nope", "fasttd3")


def _filled(rewards, terms, dones):
    """One env, rows written in order; obs = row index so the bootstrap obs is identifiable."""
    buf = replay_init(8, 1, 1, 1)
    for t, (r, te, d) in enumerate(zip(rewards, terms, dones, strict=True)):
        buf = replay_add(buf, jnp.full((1, 1), t), jnp.zeros((1, 1)), jnp.full((1,), r),
                         jnp.full((1, 1), t + 1), jnp.array([te]), jnp.array([d]))
    return buf


def _sample_from(buf, t0):
    """Sample until row t0 comes up (sampling is random over valid rows)."""
    for k in range(200):
        b = replay_sample(buf, jax.random.PRNGKey(k), 1, 3, 0.5)
        if int(b.obs[0, 0]) == t0:
            return b
    raise AssertionError(f"row {t0} never sampled")


def test_nstep_return_runs_full_n_without_done():
    b = _sample_from(_filled([1, 2, 4, 8], [0] * 4, [0] * 4), 0)
    assert float(b.ret[0]) == pytest.approx(1 + 0.5 * 2 + 0.25 * 4)
    assert float(b.next_obs[0, 0]) == 3  # next obs of row 2
    assert float(b.discount[0]) == pytest.approx(0.125)


def test_nstep_return_stops_at_terminal_without_bootstrap():
    b = _sample_from(_filled([1, 2, 4, 8], [0, 1, 0, 0], [0, 1, 0, 0]), 0)
    assert float(b.ret[0]) == pytest.approx(1 + 0.5 * 2)
    assert float(b.discount[0]) == 0.0


def test_nstep_return_bootstraps_at_truncation():
    b = _sample_from(_filled([1, 2, 4, 8], [0] * 4, [0, 1, 0, 0]), 0)
    assert float(b.ret[0]) == pytest.approx(1 + 0.5 * 2)
    assert float(b.next_obs[0, 0]) == 2
    assert float(b.discount[0]) == pytest.approx(0.25)


def test_sample_never_reads_unwritten_rows():
    buf = _filled([1, 1, 1, 1], [0] * 4, [0] * 4)
    b = replay_sample(buf, jax.random.PRNGKey(0), 512, 3, 0.5)
    assert set(np.asarray(b.obs[:, 0]).astype(int)) <= {0, 1}


def test_c51_projection_keeps_mass_and_places_points():
    probs = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (4, 11)))
    z = jnp.linspace(-1.0, 1.0, 11)
    tz = jnp.stack([z, z + 0.05, z * 3, z * 0 - 0.3])
    out = project(probs, tz, -1.0, 1.0)
    np.testing.assert_allclose(out.sum(-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(out[0], probs[0], atol=1e-6)  # identity shift
    np.testing.assert_allclose((out[3] * z).sum(), -0.3, atol=1e-5)  # point mass between atoms


@pytest.mark.parametrize("algo", ALGOS)
def test_offpolicy_policy_folder_roundtrip(env, tmp_path, algo):
    cfg = get_offpolicy_profile("smoke", algo)
    actor = nets.actor_module(algo, env.action_size, cfg.actor_hidden)
    norm = running_statistics.init_state(specs.Array((env.observation_size,), jnp.float32))
    params = (norm, actor.init(jax.random.PRNGKey(0), jnp.zeros((1, env.observation_size))))
    meta = {
        "profile": "smoke", "algo": algo, "offpolicy": cfg.to_dict(), "env": env.config.to_dict(),
        "body_spec_version": env.spec.version, "observation_size": env.observation_size,
        "action_size": env.action_size,
    }
    save_policy(tmp_path, params, meta)
    saved = load_policy(tmp_path)
    assert saved.algo == algo
    obs = jax.random.normal(jax.random.PRNGKey(1), (3, env.observation_size))
    a = saved.policy()(jax.random.PRNGKey(0), obs)
    assert a.shape == (3, env.action_size)
    assert float(jnp.abs(a).max()) <= 1.0
    inf = nets.make_inference(algo, env.action_size, cfg.actor_hidden)
    np.testing.assert_allclose(a, inf(params)(obs, jax.random.PRNGKey(0))[0], atol=1e-6)
