"""One-step engine comparison: putting an engine in a given state, and the comparison itself.

The Bullet test needs PyBullet (the `sim2sim` extra, the box):
`uv run --extra sim2sim pytest tests/unreal/test_onestep.py`.
"""

import numpy as np
import pytest

from tars_rl.body import load_spec
from tars_rl.env.levels import make_level
from tars_rl.unreal import onestep
from tars_rl.unreal.engines import MujocoEngine

SPEC = load_spec()
LEVEL = make_level("flat", 0)


def walked(steps: int = 60):
    """A MuJoCo run under random servo targets: states (steps + 1) and controls."""
    e = MujocoEngine(SPEC, 0.02)
    e.load(LEVEL)
    rng = np.random.default_rng(0)
    states = [e.reset(np.zeros(2), np.array([5.0, 0, 0]))]
    ctrls = [rng.uniform(-0.5, 0.5, 4) * np.array([1, 1, 0.08, 0.08]) for _ in range(steps)]
    states += [e.step(c) for c in ctrls]
    return e, states, ctrls


def assert_same_state(a, b, tol=1e-9):
    np.testing.assert_allclose(a.pos, b.pos, atol=tol)
    np.testing.assert_allclose(np.abs(np.sum(a.quat * b.quat, axis=1)), 1, atol=tol)  # q ~ -q
    np.testing.assert_allclose(a.linvel, b.linvel, atol=tol)
    np.testing.assert_allclose(a.angvel, b.angvel, atol=tol)


def test_mujoco_set_state_restores_a_moving_state_exactly():
    e, states, _ = walked()
    fresh = MujocoEngine(SPEC, 0.02)
    fresh.load(LEVEL)
    fresh.reset(np.zeros(2), np.array([5.0, 0, 0]))
    for s in states[10::10]:
        assert_same_state(fresh.set_state(s), s)


def test_mujoco_against_itself_has_no_one_step_error():
    e, states, ctrls = walked()
    err = onestep.one_step(e, SPEC, LEVEL, states, ctrls, lift=False)
    assert err.shape == (len(ctrls), len(onestep.MEASURES))
    assert np.abs(err).max() < 1e-6


def test_lift_clears_mujocos_soft_contact():
    e = MujocoEngine(SPEC, 0.02)
    e.load(LEVEL)
    e.reset(np.zeros(2), np.array([5.0, 0, 0]))
    for _ in range(25):  # settled: MuJoCo's soft contact holds TARS a little into the floor
        s = e.step(np.zeros(4))
    dz = onestep.penetration(SPEC, LEVEL, s)
    assert 0 < dz < 0.002
    assert onestep.penetration(SPEC, LEVEL, onestep.shifted(s, dz)) < 1e-12


def test_bullet_set_state_restores_mujocos_state():
    pytest.importorskip("pybullet")
    from tars_rl.unreal.bullet import BulletEngine

    _, states, _ = walked()
    b = BulletEngine(SPEC, 0.02)
    b.load(LEVEL)
    b.reset(np.zeros(2), np.array([5.0, 0, 0]))
    for s in states[10::10]:
        assert_same_state(b.set_state(s), s, tol=1e-6)
    b.close()
