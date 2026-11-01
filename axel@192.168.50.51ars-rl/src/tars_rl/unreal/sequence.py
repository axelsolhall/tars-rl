"""The scripted actuator sequence both engines play (T10), and its JSON contract.

A sequence is a list of keyframes (time in s, one target per BodySpec actuator, in actuator
order). Targets are linearly interpolated between keyframes and held after the last one;
`unreal/TarsRl/Source/TarsRl/TarsActor.cpp` implements the same rule (A70).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tars_rl.body.spec import BodySpec

SEQUENCE_VERSION = 1
STAND_SECONDS = 10.0  # neutral hold before anything moves (task: "stands on its own for 10 s")


@dataclass(frozen=True)
class Sequence:
    version: int
    actuator_names: tuple[str, ...]
    times: tuple[float, ...]
    targets: tuple[tuple[float, ...], ...]  # one row per keyframe, actuator order

    @property
    def duration(self) -> float:
        return self.times[-1]

    def at(self, t: float | np.ndarray) -> np.ndarray:
        """Targets at time(s) t, shape (4,) or (len(t), 4)."""
        tg = np.asarray(self.targets)
        cols = [np.interp(t, self.times, tg[:, i]) for i in range(tg.shape[1])]
        return np.stack(cols, axis=-1)

    def check(self, spec: BodySpec) -> None:
        if list(self.actuator_names) != spec.actuator_names:
            raise ValueError(f"sequence actuators {self.actuator_names} != {spec.actuator_names}")
        if any(b <= a for a, b in zip(self.times, self.times[1:], strict=False)):
            raise ValueError("keyframe times must increase")
        for row in self.targets:
            for j, v in zip(spec.joint_names, row, strict=True):
                lo, hi = spec.joint_range(j)
                if not lo <= v <= hi:
                    raise ValueError(f"{j} target {v} outside {lo}..{hi}")

    def to_json(self) -> str:
        d = {
            "schema": "tars_rl.unreal.Sequence",
            "version": self.version,
            "actuator_names": list(self.actuator_names),
            "keyframes": [
                {"t": t, "targets": list(row)}
                for t, row in zip(self.times, self.targets, strict=True)
            ],
        }
        return json.dumps(d, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Sequence:
        d = json.loads(text)
        if d.get("version") != SEQUENCE_VERSION:
            raise ValueError(f"Sequence version {d.get('version')} != {SEQUENCE_VERSION}")
        kf = d["keyframes"]
        return cls(
            version=d["version"],
            actuator_names=tuple(d["actuator_names"]),
            times=tuple(float(k["t"]) for k in kf),
            targets=tuple(tuple(float(v) for v in k["targets"]) for k in kf),
        )

    @classmethod
    def load(cls, path: str | Path) -> Sequence:
        return cls.from_json(Path(path).read_text())


def default_sequence(spec: BodySpec) -> Sequence:
    """10 s neutral stand, then hinge swings, arm slides and a combined move, each ramped over
    0.5 s and held 1.5 s so both servo types are exercised without throwing an arm (A55, A70).
    MuJoCo stays upright through all of it; both hinges forward together, or opposite hinges
    with opposite slides, tip it over, so those are left out (a fall would end the comparison).
    Order: left hinge, right hinge, left slide, right slide."""
    h, s = 0.5, 0.04  # rad, m
    poses = [
        (h, 0, 0, 0),
        (0, 0, 0, 0),
        (0, -h, 0, 0),
        (0, 0, 0, 0),
        (0, 0, s, s),
        (0, 0, -s, -s),
        (0, 0, 0, 0),
        (h, -h, 0, 0),
        (0, 0, 0, 0),
        (h, 0, s, 0),
        (0, 0, 0, 0),
    ]
    times, rows, t = [0.0, STAND_SECONDS], [(0.0,) * 4, (0.0,) * 4], STAND_SECONDS
    for p in poses:
        t += 0.5
        times.append(t)
        rows.append(tuple(float(v) for v in p))
        t += 1.5
        times.append(t)
        rows.append(tuple(float(v) for v in p))
    seq = Sequence(SEQUENCE_VERSION, tuple(spec.actuator_names), tuple(times), tuple(rows))
    seq.check(spec)
    return seq
