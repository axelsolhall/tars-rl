import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tars_rl.env.config import EnvConfig
from tars_rl.env.flag_env import FlagEnv
from tars_rl.train.memory import MEMORY_KINDS, MemorySpec
from tars_rl.train.networks import NetworkSpec, make_ppo_networks, tower_sizes

STEPS, FRAME, ACT = 10, 17, 4
KINDS = [k for k in MEMORY_KINDS if k != "none"]


@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_memory_spec_roundtrips(kind):
    m = MemorySpec() if kind == "none" else MemorySpec(kind, STEPS)
    assert MemorySpec.parse(str(m)) == m.check()
    n = NetworkSpec(memory=m)
    assert NetworkSpec.parse(str(n)) == n
    assert NetworkSpec.from_dict(n.to_dict()) == n


def test_memory_spec_rejects_mismatched_steps():
    with pytest.raises(ValueError, match="'none' takes 0"):
        MemorySpec("gru", 0).check()
    with pytest.raises(ValueError, match="not in"):
        MemorySpec.parse("lstm-25")


def test_no_memory_keeps_old_parameter_names():
    # Policies saved before memory existed must still load: plain towers are unchanged.
    nets = make_ppo_networks(FRAME, ACT, spec=NetworkSpec())
    params = nets.policy_network.init(jax.random.PRNGKey(0))
    assert "hidden_0" in params["params"]


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_reads_the_past(kind):
    spec = NetworkSpec(memory=MemorySpec(kind, STEPS))
    nets = make_ppo_networks(STEPS * FRAME, ACT, spec=spec)
    params = nets.policy_network.init(jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 3, STEPS * FRAME))  # brax's (T, B, obs)
    out = nets.policy_network.apply(None, params, obs)
    assert out.shape == (2, 3, 2 * ACT)
    # the oldest frame changes the action: the encoder really looks back
    grad = jax.grad(lambda o: nets.policy_network.apply(None, params, o).sum())(obs)
    assert float(jnp.abs(grad[..., :FRAME]).sum()) > 0
    v = make_ppo_networks(STEPS * FRAME, ACT, spec=spec).value_network
    assert v.apply(None, v.init(jax.random.PRNGKey(0)), obs).shape == (2, 3)
    assert tower_sizes(spec, STEPS * FRAME, ACT)["policy"] > 0


def test_network_refuses_obs_that_is_not_whole_frames():
    with pytest.raises(ValueError, match="history_steps"):
        make_ppo_networks(STEPS * FRAME + 1, ACT, spec=NetworkSpec(memory=MemorySpec("gru", STEPS)))


def test_env_window_slides_one_frame_per_step():
    env = FlagEnv(config=EnvConfig(episode_seconds=0.2, history_steps=3))
    d = env.frame_size
    assert env.observation_size == 3 * d
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s0 = reset(jax.random.PRNGKey(0))
    assert np.allclose(s0.obs, np.tile(s0.obs[-d:], 3))  # filled with the first frame
    s1 = step(s0, jnp.ones(4))
    s2 = step(s1, -jnp.ones(4))
    assert np.allclose(s2.obs[: 2 * d], s1.obs[d:])  # older frames shift left
    assert np.allclose(s2.obs[d : 2 * d], s1.obs[-d:])
    assert not np.allclose(s2.obs[-d:], s1.obs[-d:])  # newest frame is new
