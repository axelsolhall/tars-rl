"""Body and contact probes, and the Bullet engine behind them.

The Bullet tests need PyBullet (the `sim2sim` extra; it builds on the Linux box, not on
macOS 26): `uv run --extra sim2sim pytest tests/unreal/test_probe.py`.
"""

import numpy as np
import pytest

from tars_rl.body import load_spec
from tars_rl.env.levels import make_level
from tars_rl.unreal import probe
from tars_rl.unreal.bodystate import joint_state
from tars_rl.unreal.engines import MujocoEngine

SPEC = load_spec()


@pytest.fixture(scope="module")
def mujoco_probes():
    e = probe.make_engine("mujoco", SPEC)
    return probe.run_probes(e, SPEC)


def test_mujoco_meets_the_rigid_body_answers(mujoco_probes):
    rigid = probe.rigid_answers(SPEC)
    tips = mujoco_probes["push centre"]["tips from N"]
    assert abs(tips - rigid["push centre"]["tips from N"]) <= 0.6
    # Soft contacts let TARS creep under a push below mu * weight, but not by half.
    assert 0.6 * rigid["push feet"]["moves from N"] <= mujoco_probes["push feet"]["moves from N"] \
        <= rigid["push feet"]["moves from N"]
    assert mujoco_probes["sequence"]["fell at s"] is None, "TARS falls in its own scripted sequence"


def test_a_push_moves_tars_only_while_it_is_set():
    level = make_level("flat", 0)
    e = MujocoEngine(SPEC, 0.02)
    e.load(level)
    s0 = e.reset(np.zeros(2), np.array([5.0, 0, 0]))
    e.set_push(np.array([40.0, 0, 0]), np.array([0, 0, -0.17]))
    s = e.step(np.zeros(4))
    assert s.pos[0, 0] > s0.pos[0, 0] + 1e-4, "the push did nothing"
    s = e.reset(np.zeros(2), np.array([5.0, 0, 0]))
    for _ in range(10):
        s = e.step(np.zeros(4))
    assert np.allclose(s.pos[0, :2], s0.pos[0, :2], atol=1e-5), "a reset kept the push"


@pytest.fixture(scope="module")
def bullet():
    pytest.importorskip("pybullet")
    from tars_rl.unreal.bullet import BulletEngine

    e = BulletEngine(SPEC, 0.02)
    yield e
    e.close()


def test_bullet_moves_tars_in_the_air_exactly_like_mujoco(bullet):
    # No contact: the servo law, masses, inertias and joint axes alone (0.233 rad at 0.04 s).
    out = {}
    for e in (MujocoEngine(SPEC, 0.02), bullet):
        e.load(make_level("flat", 0))
        probe.reset_lifted(e, np.zeros(2), np.array([5.0, 0, 0]), 1.0)
        out[e.name] = [e.step(np.array([0.5, -0.3, 0.02, -0.02])) for _ in range(5)]
    for a, b in zip(out["mujoco"], out["bullet"], strict=True):
        assert np.allclose(joint_state(SPEC, a)[0], joint_state(SPEC, b)[0], atol=2e-3)
        assert np.allclose(a.pos - a.pos[0], b.pos - b.pos[0], atol=1e-4)


def test_bullet_tars_stands_and_meets_the_rigid_body_answers(bullet):
    res = probe.run_probes(bullet, SPEC)
    rigid = probe.rigid_answers(SPEC)
    assert max(abs(res["rest"][k]) for k in ("central mm", "left mm", "right mm")) < 0.1
    assert abs(res["push centre"]["tips from N"] - rigid["push centre"]["tips from N"]) <= 0.6
    assert res["push feet"]["moves from N"] >= 0.8 * rigid["push feet"]["moves from N"]
    assert res["sequence"]["fell at s"] is None
