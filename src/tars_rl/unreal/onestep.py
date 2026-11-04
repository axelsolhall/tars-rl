"""One-step comparison of physics engines on a policy's own states.

A trained policy walks in CPU MuJoCo; every engine is then put in each of MuJoCo's states in turn
and steps once (20 ms) with the policy's action. The engine's next state against MuJoCo's says how
differently it moves TARS from exactly the same state, step by step. Whole episodes cannot say
this: two engines' episodes drift apart within a second, and every later difference inherits the
first. Steps are grouped by the slabs on the floor (in MuJoCo, C central, L and R arms; "-" in
the air; "changing" when a slab lands or lifts within the step), so a bad group points to the
contact the engines disagree on.

Each engine also runs a self-check: the same actions open-loop from its own reset, then one step
from each of its own states. That should be zero; what it is not is the error the teleport itself
adds (Chaos keeps contacts and friction anchors between steps; a teleport drops them), a floor
under the comparison.

MuJoCo's soft contacts hold TARS ~1 mm into the floor at rest. Handed such a state, a hard-contact
engine pushes the slab out within the step, a jump no episode of its own has. So every other
engine gets the state lifted clear of the floor by MuJoCo's deepest corner, and positions are
compared with the lift taken off again.

    errors (mean over steps): central velocity (cm/s), central spin (deg/s), hinge rates (deg/s),
    slide rates (cm/s), and central position (mm) after the 20 ms

Run (bullet needs the sim2sim extra, Linux only; unreal needs scripts/unreal.sh fast 1):
    uv run --extra sim2sim python -m tars_rl.unreal.onestep --policy runs/chain-dr/transformer \
        --engines bullet unreal
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import jax
import numpy as np

from tars_rl.env.levels import make_level
from tars_rl.unreal.bodystate import BodyState, joint_state
from tars_rl.unreal.bridge import DEFAULT_PORT
from tars_rl.unreal.engines import MujocoEngine
from tars_rl.unreal.poses import qrotate
from tars_rl.unreal.probe import make_engine
from tars_rl.unreal.verify import Rules, episode_keys, load

MEASURES = ("v cm/s", "w deg/s", "hinge deg/s", "slide cm/s", "pos mm")
SLABS = "CLR"  # central, left arm, right arm


def rollout(policy: str, level_name: str, episodes: int, steps: int, seed: int = 0):
    """MuJoCo episodes of the policy: per episode its states (steps + 1) and controls (steps).
    Flags move on when tagged, as in training; an episode ends early if TARS falls."""
    env, act = load(policy)
    level = make_level(level_name, seed)
    rules = Rules(env, level)
    mj = MujocoEngine(env.spec, env.dt)
    mj.load(level)
    spawn = np.asarray(level.spawn_xy, float)
    out = []
    for key in episode_keys(seed, episodes):
        flag, _ = rules.flag(key, spawn)
        flag_rng = jax.random.fold_in(key, 2)
        s = mj.reset(spawn, flag)
        obs = rules.window(None, rules.frame(s, flag))
        ctrl = np.zeros(env.action_size)
        states, ctrls = [s], []
        for _ in range(steps):
            ctrl = rules.ctrl(np.asarray(act(obs), float), ctrl)
            s = mj.step(ctrl)
            states.append(s)
            ctrls.append(ctrl.copy())
            flag_rng, k_flag = rules._split(flag_rng)
            if s.cos_tilt() < rules.cos_fall:
                break
            if np.linalg.norm(s.central_pos[:2] - flag[:2]) < rules.tag_m:
                flag, _ = rules.flag(k_flag, s.central_pos[:2])
                mj.set_flag(flag)
            obs = rules.window(obs, rules.frame(s, flag))
        out.append((states, ctrls))
    return env, level, mj, out


def floor_contacts(mj: MujocoEngine, s: BodyState) -> str:
    """Which slabs touch the floor or a level box in MuJoCo's state `s`, e.g. "CLR" or "-"."""
    mj.set_state(s)
    touching = set()
    for c in mj.d.contact[:mj.d.ncon]:
        if c.dist > 1e-3:
            continue
        bodies = {int(mj.m.geom_bodyid[c.geom1]), int(mj.m.geom_bodyid[c.geom2])}
        for i, b in enumerate(mj.ids):
            if b in bodies and len(bodies - set(mj.ids)) > 0:
                touching.add(SLABS[i])
    return "".join(x for x in SLABS if x in touching) or "-"


def errors(spec, pred: BodyState, true: BodyState) -> np.ndarray:
    """|pred - true| per measure (MEASURES order)."""
    qp, qdp = joint_state(spec, pred)
    qt, qdt = joint_state(spec, true)
    return np.array([
        100 * np.linalg.norm(pred.linvel[0] - true.linvel[0]),
        np.degrees(np.linalg.norm(pred.angvel[0] - true.angvel[0])),
        np.degrees(np.abs(qdp[:2] - qdt[:2]).mean()),
        100 * np.abs(qdp[2:] - qdt[2:]).mean(),
        1000 * np.linalg.norm(pred.pos[0] - true.pos[0]),
    ])


def penetration(spec, level, s: BodyState) -> float:
    """How deep (m) the deepest slab corner is below the ground under it; 0 when clear."""
    deepest = 0.0
    for i, box in enumerate((spec.central, spec.arm, spec.arm)):
        half = np.array([box.depth, box.width, box.height]) / 2
        for sign in itertools.product((-1, 1), repeat=3):
            c = s.pos[i] + qrotate(s.quat[i], half * np.array(sign))
            deepest = max(deepest, level.ground_z(c[:2]) - c[2])
    return deepest


def shifted(s: BodyState, dz: float) -> BodyState:
    return BodyState(s.t, s.pos + np.array([0.0, 0.0, dz]), s.quat, s.linvel, s.angvel)


def one_step(engine, spec, level, states: list[BodyState], ctrls: list[np.ndarray],
             lift: bool = True) -> np.ndarray:
    """Errors of `engine` stepping once from each of `states` against the next state; `lift`:
    each state raised out of the floor first (see the module docstring)."""
    spawn = np.asarray(level.spawn_xy, float)
    engine.reset(spawn, np.array([spawn[0] + 5.0, spawn[1], 0.0]))
    out = []
    for s, c, nxt in zip(states[:-1], ctrls, states[1:], strict=True):
        dz = penetration(spec, level, s) if lift else 0.0
        engine.set_state(shifted(s, dz))
        out.append(errors(spec, shifted(engine.step(c), -dz), nxt))
    return np.array(out)


def own_states(engine, level, ctrls: list[np.ndarray]) -> list[BodyState]:
    spawn = np.asarray(level.spawn_xy, float)
    states = [engine.reset(spawn, np.array([spawn[0] + 5.0, spawn[1], 0.0]))]
    states += [engine.step(c) for c in ctrls]
    return states


def run(policy: str, level_name: str, engines: list[str], episodes: int, steps: int,
        port: int) -> dict:
    env, level, mj, eps = rollout(policy, level_name, episodes, steps)
    spec = env.spec
    # Slabs down at the start of the step; "changing" when the set differs at its end (a landing
    # or lift-off inside the 20 ms, so "-" is a whole step in the air).
    groups = []
    for states, _ in eps:
        down = [floor_contacts(mj, s) for s in states]
        groups += [a if a == b else "changing" for a, b in zip(down[:-1], down[1:], strict=True)]
    res: dict = {"policy": policy, "level": level_name, "steps": len(groups),
                 "measures": list(MEASURES), "groups": {}, "engines": {}, "self_check": {}}
    for g in sorted(set(groups), key=groups.count, reverse=True):
        res["groups"][g] = groups.count(g)
    for name in ["mujoco", *engines]:
        e = mj if name == "mujoco" else make_engine(name, spec, port)
        try:
            e.load(level)
            lift = name != "mujoco"
            err = np.concatenate([one_step(e, spec, level, st, c, lift) for st, c in eps])
            own = np.concatenate([one_step(e, spec, level, own_states(e, level, c), c, False)
                                  for _, c in eps])
        finally:
            if e is not mj:
                e.close()
        g = np.array(groups)
        res["engines"][name] = {"all": err.mean(0).round(2).tolist(),
                                **{k: err[g == k].mean(0).round(2).tolist() for k in res["groups"]}}
        res["self_check"][name] = own.mean(0).round(2).tolist()
    return res


def report(res: dict) -> None:
    print(f"Policy {res['policy']} on {res['level']}: {res['steps']} MuJoCo steps; one step "
          f"(20 ms) from each, error against MuJoCo's next state (mean)")
    head = "".join(f"{m:>13}" for m in res["measures"])
    print(f"\n{'engine':<10}{'slabs down':<12}{'steps':>6}" + head)
    for name, by in res["engines"].items():
        for g, row in by.items():
            n = res["steps"] if g == "all" else res["groups"][g]
            print(f"{name:<10}{g:<12}{n:>6}" + "".join(f"{v:>13.2f}" for v in row))
    print("\nSelf-check: the same from the engine's own states (the teleport's own error)")
    for name, row in res["self_check"].items():
        print(f"{name:<28}" + "".join(f"{v:>13.2f}" for v in row))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", required=True)
    p.add_argument("--level", default="flat")
    p.add_argument("--engines", nargs="*", default=["bullet"], choices=["bullet", "unreal"])
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    res = run(a.policy, a.level, a.engines, a.episodes, a.steps, a.port)
    report(res)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
