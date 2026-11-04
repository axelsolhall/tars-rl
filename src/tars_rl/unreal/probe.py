"""Body and contact probes: the same small experiments in each physics engine, side by side, with
rigid-body mechanics' answer where it has one. They check an engine's TARS before any policy
runs in it, and say which part differs (servos and inertia, rest, contact) when one fails.

    free fall     TARS dropped from 1 m, left hinge target 0.5 rad: hinge angle and tilt after
                  0.04 and 0.10 s. No contact at all: servos, masses and inertias only.
    rest          2 s at neutral: each slab's lowest corner (mm) and how far TARS drifts.
    hinges        both hinge targets 0.5 rad on the floor for 2 s: hinge angle and tilt at
                  0.18 s, the largest tilt, the tilt at 2 s and how far TARS moved forward. The
                  arms' feet must grip: Unreal's slid back with Chaos's default friction
                  iterations, and TARS fell over backwards (DefaultEngine.ini, 2026-11-04).
    slide         left slide target +4 cm on the floor: slide position after 0.1 and 0.5 s.
    push feet     a forward force 1 cm above the floor for 1 s, swept up: the smallest that moves
                  TARS 1 cm. Rigid TARS: mu * weight = 23.5 N.
    push centre   the same at the central slab's centre: the smallest that tips TARS past 30 deg.
                  Rigid TARS: weight * (depth / 2) / 18 cm = 7.4 N.
    drag          30 N at the feet for 0.3 s: how far TARS slides (kinetic friction) and how far
                  it tilts. Rigid TARS: a t^2 / 2, a = (F - mu * weight) / mass. (Let go at that
                  speed, any engine's TARS tips over forwards: its 4.5 cm deep feet can't stop it.)
    sequence      the T10 scripted sequence, targets held per 20 ms step: stand time, max tilt.

Run (bullet needs the sim2sim extra, Linux only; unreal needs scripts/unreal.sh fast):
    uv run --extra sim2sim python -m tars_rl.unreal.probe --engines mujoco bullet unreal
2026-11-04 (box): the three engines agree on every probe (VERIFY.md, Engine probes and Bullet).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from tars_rl.body import load_spec
from tars_rl.body.spec import BodySpec
from tars_rl.env.levels import make_level
from tars_rl.unreal.bodystate import BodyState, joint_state
from tars_rl.unreal.bridge import DEFAULT_PORT, point_to_ue, state_from_ue
from tars_rl.unreal.engines import MujocoEngine, UnrealEngine
from tars_rl.unreal.poses import qrotate
from tars_rl.unreal.sequence import default_sequence

CTRL_DT = 0.02
DRAG_N, DRAG_S = 30.0, 0.3
FEET_FORCES = (5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0)
CENTRE_FORCES = (4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0)


def make_engine(name: str, spec: BodySpec, port: int = DEFAULT_PORT):
    if name == "mujoco":
        return MujocoEngine(spec, CTRL_DT)
    if name == "bullet":
        from tars_rl.unreal.bullet import BulletEngine

        return BulletEngine(spec, CTRL_DT)
    if name == "unreal":
        return UnrealEngine(spec, CTRL_DT, port=port)
    raise ValueError(f"no engine {name!r} (mujoco, bullet, unreal)")


def tilt_deg(s: BodyState) -> float:
    return float(np.degrees(np.arccos(np.clip(s.cos_tilt(), -1.0, 1.0))))


def reset_lifted(e, spawn: np.ndarray, flag: np.ndarray, dz: float) -> None:
    """Reset with TARS dz above the floor (each engine has its own way to place it)."""
    if e.name == "unreal":
        at = point_to_ue([spawn[0], spawn[1], dz])
        state_from_ue(e.client.request({"op": "reset", "at": at, "yaw": 0.0,
                                        "flag": point_to_ue(flag)}))
        return
    e.reset(spawn, flag)
    if e.name == "mujoco":
        e.d.qpos[e.root_adr + 2] += dz
        mujoco.mj_forward(e.m, e.d)
    else:  # bullet
        p = e.p
        pos, q = p.getBasePositionAndOrientation(e.tars, physicsClientId=e.client)
        p.resetBasePositionAndOrientation(e.tars, [pos[0], pos[1], pos[2] + dz], q,
                                          physicsClientId=e.client)


def run_probes(e, spec: BodySpec) -> dict:
    level = make_level("flat", 0)
    e.load(level)
    spawn = np.asarray(level.spawn_xy, float)
    flag = np.array([spawn[0] + 5.0, spawn[1], 0.0])
    out: dict = {}

    reset_lifted(e, spawn, flag, 1.0)
    ss = [e.step(np.array([0.5, 0, 0, 0])) for _ in range(5)]
    out["free fall"] = {f"hinge {t}s": round(float(joint_state(spec, ss[i])[0][0]), 3)
                        for t, i in ((0.04, 1), (0.10, 4))}
    out["free fall"]["tilt 0.10s"] = round(tilt_deg(ss[4]), 2)

    s0 = e.reset(spawn, flag)
    for _ in range(100):
        s = e.step(np.zeros(4))
    halves = [np.array([spec.central.depth, spec.central.width, spec.central.height]) / 2] + \
        [np.array([spec.arm.depth, spec.arm.width, spec.arm.height]) / 2] * 2
    def lowest_mm(i: int) -> float:
        corners = [(s.pos[i] + qrotate(s.quat[i], halves[i] * np.array(sg)))[2]
                   for sg in itertools.product((-1, 1), repeat=3)]
        return round(1000 * min(corners), 3)

    out["rest"] = {name: lowest_mm(i)
                   for i, name in enumerate(("central mm", "left mm", "right mm"))}
    out["rest"]["drift mm"] = round(1000 * float(np.linalg.norm(s.pos[0, :2] - s0.pos[0, :2])), 3)

    s0 = e.reset(spawn, flag)
    ss = [e.step(np.array([0.5, 0.5, 0, 0])) for _ in range(100)]
    out["hinges"] = {"hinge 0.18s": round(float(joint_state(spec, ss[8])[0][0]), 3),
                     "tilt 0.18s": round(tilt_deg(ss[8]), 1),
                     "max tilt": round(max(tilt_deg(x) for x in ss), 1),
                     "tilt 2s": round(tilt_deg(ss[-1]), 1),
                     "moved mm": round(1000 * float(ss[-1].pos[0, 0] - s0.pos[0, 0]))}

    e.reset(spawn, flag)
    ss = [e.step(np.array([0, 0, 0.04, 0])) for _ in range(25)]
    out["slide"] = {f"slide {t}s mm": round(1000 * float(joint_state(spec, ss[i])[0][2]), 1)
                    for t, i in ((0.1, 4), (0.5, 24))}

    def pushed(force: float, at_z: float) -> tuple[float, float]:
        s0 = e.reset(spawn, flag)
        e.set_push(np.array([force, 0.0, 0.0]), np.array([0.0, 0.0, at_z]))
        worst = 0.0
        for _ in range(50):
            s = e.step(np.zeros(4))
            worst = max(worst, tilt_deg(s))
        return float(np.linalg.norm(s.pos[0, :2] - s0.pos[0, :2])), worst

    feet_z = 0.01 - spec.central.height / 2
    moves = next((f for f in FEET_FORCES if pushed(f, feet_z)[0] > 0.01), None)
    tips = next((f for f in CENTRE_FORCES if pushed(f, 0.0)[1] > 30), None)
    out["push feet"] = {"moves from N": moves}
    out["push centre"] = {"tips from N": tips}

    s0 = e.reset(spawn, flag)
    e.set_push(np.array([DRAG_N, 0.0, 0.0]), np.array([0.0, 0.0, feet_z]))
    ss = [e.step(np.zeros(4)) for _ in range(round(DRAG_S / CTRL_DT))]
    out["drag"] = {"slid mm": round(1000 * float(ss[-1].pos[0, 0] - s0.pos[0, 0]), 1),
                   "max tilt": round(max(tilt_deg(x) for x in ss), 1)}

    seq = default_sequence(spec)
    e.reset(spawn, flag)
    fell, worst = None, 0.0
    for k in range(round(seq.duration / CTRL_DT)):
        s = e.step(np.asarray(seq.at(k * CTRL_DT), float))
        worst = max(worst, tilt_deg(s))
        if fell is None and worst > 45:
            fell = round((k + 1) * CTRL_DT, 2)
    out["sequence"] = {"fell at s": fell, "max tilt": round(worst, 1)}
    return out


def rigid_answers(spec: BodySpec) -> dict:
    mass = spec.central.mass + 2 * spec.arm.mass
    weight = 9.81 * mass
    a = (DRAG_N - spec.friction * weight) / mass
    return {"drag": {"slid mm": round(1000 * a * DRAG_S**2 / 2, 1)},
            "push feet": {"moves from N": round(spec.friction * weight, 1)},
            "push centre": {"tips from N": round(
                weight * (spec.central.depth / 2) / (spec.central.height / 2), 1)}}


def report(res: dict) -> None:
    names = list(res["engines"])
    print("probe / measure".ljust(30) + "".join(f"{n:>12}" for n in names) + f"{'rigid':>12}")
    first = res["engines"][names[0]]
    for probe, measures in first.items():
        for m in measures:
            row = f"{probe} / {m}".ljust(30)
            for n in names:
                v = res["engines"][n].get(probe, {}).get(m)
                row += f"{'-' if v is None else v:>12}"
            v = res["rigid"].get(probe, {}).get(m)
            print(row + f"{'' if v is None else v:>12}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engines", nargs="+", default=["mujoco", "bullet"],
                   choices=["mujoco", "bullet", "unreal"])
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Unreal bridge port")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    spec = load_spec()
    res = {"engines": {}, "rigid": rigid_answers(spec)}
    for name in a.engines:
        e = make_engine(name, spec, a.port)
        try:
            res["engines"][name] = run_probes(e, spec)
        finally:
            e.close()
    report(res)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
