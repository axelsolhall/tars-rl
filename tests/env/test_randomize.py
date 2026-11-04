import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco_playground import wrapper

from tars_rl.env.config import EnvConfig
from tars_rl.env.flag_env import FlagEnv
from tars_rl.env.randomize import DomainRandConfig, for_env
from tars_rl.train.adapter import PlaygroundFlagEnv

N = 8


@pytest.fixture(scope="module")
def env():
    return FlagEnv(config=EnvConfig(episode_seconds=0.2))


@pytest.fixture(scope="module")
def sampled(env):
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    return for_env(env, DomainRandConfig())(env.mjx_model, keys)


def test_draws_stay_in_their_ranges(env, sampled):
    model_v, _ = sampled
    cfg, m = DomainRandConfig(), env.mj_model
    friction = np.asarray(model_v.geom_friction[:, :, 0])
    assert friction.min() >= cfg.friction[0] and friction.max() <= cfg.friction[1]
    assert np.allclose(friction, friction[:, :1]), "one floor/body friction per env"
    mass = np.asarray(model_v.body_mass[:, 1:4]) / m.body_mass[1:4]
    assert mass.min() >= cfg.mass_scale[0] and mass.max() <= cfg.mass_scale[1]
    force = np.asarray(model_v.actuator_forcerange[:, :, 1]) / m.actuator_forcerange[:, 1]
    kp = np.asarray(model_v.actuator_gainprm[:, :, 0]) / m.actuator_gainprm[:, 0]
    assert np.allclose(force, kp), "kp and force limit scale together (one servo strength)"
    assert np.allclose(np.asarray(model_v.actuator_biasprm[:, :, 1]) / m.actuator_biasprm[:, 1], kp)
    shift = np.asarray(model_v.body_ipos[:, 1]) - m.body_ipos[1]
    assert np.abs(shift).max() <= cfg.com_offset_m + 1e-7
    assert len({tuple(r) for r in np.round(force, 6)}) == N, "every env draws its own"


def test_only_randomized_fields_are_batched(env, sampled):
    model_v, in_axes = sampled
    # None leaves vanish when flattened: what is left is exactly what vmap batches.
    leaves, _ = jax.tree_util.tree_flatten_with_path(in_axes)
    batched = {jax.tree_util.keystr(path).strip(".") for path, axis in leaves if axis == 0}
    assert batched == {"geom_friction", "body_mass", "body_inertia", "actuator_gainprm",
                       "actuator_biasprm", "actuator_forcerange", "body_ipos"}
    assert np.asarray(model_v.body_mass)[:, 0].tolist() == [0.0] * N, "world untouched"


def test_bad_ranges_are_refused():
    with pytest.raises(ValueError, match="0 < low <= high"):
        DomainRandConfig(motor_strength=(1.2, 0.8)).check()


def test_playground_training_wrapper_steps_each_env_with_its_own_model(env):
    # The adapter must pass the wrapper's per-env model swap through to FlagEnv; otherwise every
    # env silently trains on the nominal model.
    keys = jax.random.split(jax.random.PRNGKey(1), N)
    fn = functools.partial(for_env(env, DomainRandConfig()), rng=keys)
    same = jnp.stack([jax.random.PRNGKey(7)] * N)  # identical start and flag everywhere

    def run(randomization_fn):
        w = wrapper.wrap_for_brax_training(PlaygroundFlagEnv(env), episode_length=50,
                                           randomization_fn=randomization_fn, full_reset=True)
        s = jax.jit(w.reset)(same)
        step = jax.jit(w.step)
        for _ in range(10):
            s = step(s, jnp.ones((N, env.action_size)))
        return np.asarray(s.data.qpos)

    nominal = run(None)
    assert np.allclose(nominal, nominal[0]), "same start, same model, same actions"
    randomized = run(fn)
    assert np.abs(randomized - randomized[0]).max() > 1e-4, "each env moved by its own physics"


def test_scoring_on_randomized_bodies_gives_each_episode_its_own_physics(env):
    # The eval's randomized mode must actually step each episode with its own model: same flags
    # and the same constant action, so any spread between episodes comes from the bodies.
    from tars_rl.env.eval import run_episodes

    push = lambda rng, obs: jnp.ones((obs.shape[0], env.action_size))  # noqa: E731
    keys = jax.random.split(jax.random.PRNGKey(3), N)
    models = for_env(env, DomainRandConfig())(env.mjx_model, keys)
    rng = jax.random.PRNGKey(0)
    nominal = run_episodes(env, push, rng, N)
    randomized = run_episodes(env, push, rng, N, models)
    assert np.allclose(nominal["final_dist"], run_episodes(env, push, rng, N)["final_dist"])
    gap = np.abs(randomized["final_dist"] - nominal["final_dist"])
    assert gap.max() > 1e-4, "randomized episodes moved exactly like nominal ones"
