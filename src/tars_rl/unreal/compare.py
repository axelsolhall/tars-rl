"""Metrics on one engine's run, and the gap between two engines' runs (T10)."""

from __future__ import annotations

import math

import numpy as np

from tars_rl.body.spec import BodySpec
from tars_rl.unreal.poses import BODIES, Poses, joints_from_poses, qangle, tilt_rad
from tars_rl.unreal.sequence import STAND_SECONDS, Sequence

FALL_TILT_DEG = 45.0  # same fall line as tars_rl.body.eval


def run_metrics(spec: BodySpec, seq: Sequence, p: Poses) -> dict:
    """Stand time and per-joint tracking error (|joint - target|) for one engine."""
    tilt = np.degrees(tilt_rad(p.quat[:, 0]))
    fallen = np.nonzero(tilt > FALL_TILT_DEG)[0]
    stand_s = float(p.t[fallen[0]]) if len(fallen) else float(p.t[-1])
    q = joints_from_poses(spec, p)
    err = np.abs(q - seq.at(p.t))
    moving = p.t >= STAND_SECONDS
    tracking = {}
    for i, j in enumerate(spec.joint_names):
        e = err[moving, i]
        # Sign check: a joint driven the wrong way moves against its target (A74).
        tgt = seq.at(p.t)[moving, i]
        corr = float(np.corrcoef(q[moving, i], tgt)[0, 1]) if np.std(tgt) > 0 else math.nan
        tracking[j] = {"mean": float(e.mean()), "max": float(e.max()), "corr": corr}
    return {
        "stand_s": stand_s,
        "stood_full_10s": stand_s >= STAND_SECONDS,
        "fell": bool(len(fallen)),
        "max_tilt_deg": float(tilt.max()),
        "max_tilt_stand_deg": float(tilt[p.t <= STAND_SECONDS].max()),
        "tracking": tracking,
    }


def gap(ref: Poses, other: Poses) -> dict:
    """Per-body position (m) and rotation (deg) gap of `other` versus `ref`, on ref's times
    clipped to the span both recordings cover."""
    t = ref.t[(ref.t >= other.t[0]) & (ref.t <= other.t[-1])]
    a, b = ref.resample(t), other.resample(t)
    out = {"seconds_compared": float(t[-1] - t[0]) if len(t) else 0.0}
    for i, name in enumerate(BODIES):
        dp = np.linalg.norm(a.pos[:, i] - b.pos[:, i], axis=-1)
        dr = np.degrees(qangle(a.quat[:, i], b.quat[:, i]))
        out[name] = {
            "pos_max_m": float(dp.max()), "pos_mean_m": float(dp.mean()),
            "rot_max_deg": float(dr.max()), "rot_mean_deg": float(dr.mean()),
        }
    return out


def format_run(name: str, r: dict) -> str:
    lines = [
        f"[{name}] stand {r['stand_s']:.2f} s ({'fell' if r['fell'] else 'never fell'}), "
        f"max tilt {r['max_tilt_deg']:.1f} deg (first 10 s: {r['max_tilt_stand_deg']:.2f})",
        f"  {'joint':<12} {'mean err':>9} {'max err':>9} {'corr':>6}   (rad / m, after 10 s)",
    ]
    for j, e in r["tracking"].items():
        lines.append(f"  {j:<12} {e['mean']:9.4f} {e['max']:9.4f} {e['corr']:6.2f}")
    return "\n".join(lines)


def format_gap(g: dict) -> str:
    lines = [
        f"[Unreal vs MuJoCo] over {g['seconds_compared']:.1f} s",
        f"  {'body':<10} {'pos max':>8} {'pos mean':>9} {'rot max':>8} {'rot mean':>9}"
        "   (m, deg)",
    ]
    for b in BODIES:
        e = g[b]
        lines.append(f"  {b:<10} {e['pos_max_m']:8.4f} {e['pos_mean_m']:9.4f} "
                     f"{e['rot_max_deg']:8.2f} {e['rot_mean_deg']:9.2f}")
    return "\n".join(lines)
