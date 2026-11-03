"""The Python half of the Unreal bridge, checked against a stand-in that answers like Unreal.

The stand-in speaks the bridge protocol in Unreal's units and frame, with CPU MuJoCo behind it.
Driving it through `UnrealEngine` must therefore give exactly what `MujocoEngine` gives: any
difference is a bug in the protocol client or the frame conversions, not physics.
"""

import json
import socket
import threading

import jax
import mujoco
import numpy as np
import pytest

from tars_rl.env.levels import make_level
from tars_rl.unreal import bridge
from tars_rl.unreal.bodystate import BodyState, body_ids, from_mujoco, observe_default
from tars_rl.unreal.engines import MujocoEngine, UnrealEngine
from tars_rl.unreal.poses import mj_to_ue, qmul, ue_to_mj
from tars_rl.unreal.verify import Rules, run_episode


class StandIn(threading.Thread):
    """One-client Unreal stand-in: MuJoCo physics behind the bridge protocol."""

    def __init__(self, spec, level):
        super().__init__(daemon=True)
        self.engine = MujocoEngine(spec, 0.02)
        self.engine.load(level)
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.boxes = None

    def run(self):
        conn, _ = self.sock.accept()
        f = conn.makefile("rwb")
        for line in f:
            cmd = json.loads(line)
            op = cmd["op"]
            if op == "hello":
                out = {"ok": True, "protocol": bridge.PROTOCOL, "dt": 0.02, "substeps": 10}
            elif op == "level":
                self.boxes = cmd["boxes"]
                out = {"ok": True, "boxes": len(cmd["boxes"])}
            elif op == "reset":
                at = np.asarray(cmd["at"]) / 100 * bridge.MIRROR
                flag = np.asarray(cmd["flag"]) / 100 * bridge.MIRROR
                out = {**bridge.state_to_ue(self.engine.reset(at[:2], flag)), "dt": 0.0}
            elif op == "flag":
                self.engine.set_flag(np.asarray(cmd["pos"]) / 100 * bridge.MIRROR)
                out = {"ok": True}
            elif op == "step":
                out = {**bridge.state_to_ue(self.engine.step(np.asarray(cmd["ctrl"]))), "dt": 0.02}
            else:
                f.write(b'{"ok": true}\n')
                f.flush()
                break
            f.write(json.dumps(out).encode() + b"\n")
            f.flush()
        conn.close()


@pytest.fixture(scope="module")
def env_and_policy():
    from tars_rl.env.config import EnvConfig
    from tars_rl.env.flag_env import FlagEnv

    env = FlagEnv(config=EnvConfig(episode_seconds=1.0, history_steps=3))

    def act(obs):  # a fixed, non-trivial function of the observation
        return np.tanh(obs[: env.action_size] * 3.0 + obs[-env.action_size:])

    return env, act


@pytest.mark.parametrize("level", ["flat", "steps"])
def test_unreal_path_reproduces_mujoco_exactly(env_and_policy, level):
    env, act = env_and_policy
    lv = make_level(level)
    rules = Rules(env, lv)
    key = jax.random.PRNGKey(5)
    ref = run_episode(rules, _loaded(MujocoEngine(env.spec, env.dt), lv), act, key, "", False)
    stand_in = StandIn(env.spec, lv)
    stand_in.start()
    ue = UnrealEngine(env.spec, env.dt, port=stand_in.port)
    ue.load(lv)
    got = run_episode(rules, ue, act, key, "", False)
    ue.close()
    assert len(stand_in.boxes) == len(lv.boxes)
    for k in ("tags", "fell", "length", "first_tag"):
        assert getattr(got, k) == getattr(ref, k), k
    assert got.final_dist == pytest.approx(ref.final_dist, abs=1e-9)
    assert got.ret == pytest.approx(ref.ret, abs=1e-9)


def _loaded(engine, level):
    engine.load(level)
    return engine


def test_state_round_trips_through_unreal_units(env_and_policy):
    env, _ = env_and_policy
    s = _moving_state(env)
    back = bridge.state_from_ue(json.loads(json.dumps(bridge.state_to_ue(s))))
    for k in ("pos", "quat", "linvel", "angvel"):
        assert np.allclose(getattr(back, k), getattr(s, k), atol=1e-12), k
    flag = np.array([1.0, -0.5, 0.0])
    assert np.allclose(observe_default(env.spec, back, flag), observe_default(env.spec, s, flag))


def test_angular_velocity_maps_like_the_quaternions():
    # Unreal integrates q' = exp(w dt / 2) q in its own numbers. Rotating in MuJoCo's frame and
    # converting must land where Unreal's own integration of the converted w lands.
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w = rng.normal(size=3)
    dt = 1e-3

    def advance(q, w):
        a = np.linalg.norm(w) * dt
        dq = np.concatenate([[np.cos(a / 2)], np.sin(a / 2) * w / np.linalg.norm(w)])
        return qmul(dq, q)

    _, q_ue = mj_to_ue(np.zeros(3), q)
    _, q1_ue = mj_to_ue(np.zeros(3), advance(q, w))
    # Unreal's quaternions are x y z w; integrate in w x y z order on Unreal's numbers.
    as_wxyz = lambda x: np.array([x[3], x[0], x[1], x[2]])  # noqa: E731
    mine = advance(as_wxyz(q_ue), bridge.angvel_mj_to_ue(w))
    want = as_wxyz(q1_ue)
    assert min(np.abs(mine - want).max(), np.abs(mine + want).max()) < 1e-9
    assert np.allclose(ue_to_mj(np.zeros(3), q_ue)[1], q)


def test_a_missing_unreal_says_how_to_start_it():
    with pytest.raises(bridge.BridgeError, match="scripts/unreal.sh"):
        bridge.BridgeClient(port=1, timeout_s=0.5)


def _moving_state(env) -> BodyState:
    m = env.mj_model
    d = mujoco.MjData(m)
    rng = np.random.default_rng(1)
    d.qpos[3:7] = [0.9, 0.1, -0.2, 0.3]
    d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
    d.qvel[:] = rng.normal(size=m.nv)
    mujoco.mj_forward(m, d)
    return from_mujoco(m, d, body_ids(m))


def test_observation_from_slab_states_is_flagenvs():
    import jax.numpy as jnp
    from mujoco import mjx

    from tars_rl.env.flag_env import FlagEnv

    env = FlagEnv()
    m = env.mj_model
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    for _ in range(3):
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = [0.3, -0.2, 0.25]
        q = rng.normal(size=4)
        d.qpos[3:7] = q / np.linalg.norm(q)
        d.qpos[7:] = rng.uniform(-0.05, 0.05, m.nq - 7)
        d.qvel[:] = rng.normal(size=m.nv)
        mujoco.mj_forward(m, d)
        flag = np.array([1.0, 2.0, 0.0])
        ref = np.asarray(env.observe(mjx.put_data(m, d), jnp.asarray(flag)))
        got = observe_default(env.spec, from_mujoco(m, d, body_ids(m)), flag)
        assert np.abs(ref - got).max() < 1e-5
