"""The Python side of the Unreal bridge: a lockstep TCP protocol, one JSON object per line.

Unreal (`ATarsBridge`, unreal/TarsRl/Source/TarsRl/TarsBridge.cpp) listens on 127.0.0.1:PORT.
Python sends a request and waits for its reply; Unreal advances physics only on `step`, so the
simulation runs exactly as fast as Python asks: paced to the wall clock to watch it, or flat out
to evaluate. Unreal itself runs fixed 20 ms frames and never waits for the wall clock.

Requests (Unreal units: cm, cm/s, degrees for yaw, quaternions x y z w, Unreal's frame):
    {"op": "hello"}                                  -> {"ok", "protocol", "dt", "substeps"}
    {"op": "level", "name", "boxes": [{"p", "q", "s", "mu", "rgb"}]}  -> {"ok", "boxes"}
    {"op": "reset", "at", "yaw", "flag"}             -> state (TARS rebuilt at neutral, no step)
    {"op": "flag", "pos"}                            -> {"ok"}
    {"op": "step", "ctrl": [4 servo targets, SI], "hud"?: str}  -> state after one control step
    {"op": "bye"}                                    -> {"ok"} and Unreal waits for a new client
A state is {"t": s, "bodies": [{"p", "q", "v", "w"} x 3]} with v in cm/s and w in rad/s, for
central, left arm, right arm, positions relative to the level origin (the floor under (0, 0)).

Frames (A73): Unreal is MuJoCo mirrored in y. Positions and velocities flip y; quaternions map
(w, x, y, z) -> (w, -x, y, -z); angular velocity is a pseudovector, so it flips the two axes
the mirror does not: (wx, wy, wz) -> (-wx, wy, -wz). Tests check the last against the
quaternion map by integrating a rotation in both frames.
"""

from __future__ import annotations

import json
import socket

import numpy as np

from tars_rl.env.levels import COLORS, Level
from tars_rl.unreal.bodystate import BodyState
from tars_rl.unreal.poses import mj_to_ue, ue_to_mj

PROTOCOL = 1
DEFAULT_PORT = 5577
MIRROR = np.array([1.0, -1.0, 1.0])


# ---- frame conversion of velocities ----------------------------------------------------------
def linvel_ue_to_mj(v_cms: np.ndarray) -> np.ndarray:
    return np.asarray(v_cms, float) / 100.0 * MIRROR


def angvel_ue_to_mj(w: np.ndarray) -> np.ndarray:
    return np.asarray(w, float) * -MIRROR


def linvel_mj_to_ue(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, float) * 100.0 * MIRROR


def angvel_mj_to_ue(w: np.ndarray) -> np.ndarray:
    return np.asarray(w, float) * -MIRROR


def state_from_ue(msg: dict) -> BodyState:
    bodies = msg["bodies"]
    if len(bodies) != 3:
        raise ValueError(f"Unreal sent {len(bodies)} bodies, expected central + 2 arms")
    p = np.array([b["p"] for b in bodies], float)
    q = np.array([b["q"] for b in bodies], float)
    pos, quat = ue_to_mj(p, q)
    lin = linvel_ue_to_mj(np.array([b["v"] for b in bodies]))
    ang = angvel_ue_to_mj(np.array([b["w"] for b in bodies]))
    return BodyState(float(msg["t"]), pos, quat, lin, ang)


def state_to_ue(s: BodyState) -> dict:
    """The message Unreal would send for this state (tests and the protocol stand-in)."""
    p, q = mj_to_ue(s.pos, s.quat)
    v, w = linvel_mj_to_ue(s.linvel), angvel_mj_to_ue(s.angvel)
    return {"t": s.t, "bodies": [{"p": p[i].tolist(), "q": q[i].tolist(), "v": v[i].tolist(),
                                  "w": w[i].tolist()} for i in range(3)]}


def level_to_ue(level: Level) -> dict:
    boxes = []
    for b in level.boxes:
        p, q = mj_to_ue(np.asarray(b.centre), np.asarray(b.quat))
        boxes.append({"p": p.tolist(), "q": q.tolist(), "s": [100.0 * x for x in b.size],
                      "mu": b.friction, "rgb": list(COLORS.get(b.kind, (0.5, 0.5, 0.5)))})
    return {"op": "level", "name": level.name, "boxes": boxes}


def point_to_ue(xyz) -> list[float]:
    return (np.asarray(xyz, float) * 100.0 * MIRROR).tolist()


# ---- client ----------------------------------------------------------------------------------
class BridgeError(RuntimeError):
    pass


class BridgeClient:
    """One connection to one Unreal instance."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 timeout_s: float = 30.0) -> None:
        try:
            self.sock = socket.create_connection((host, port), timeout=timeout_s)
        except OSError as e:
            raise BridgeError(
                f"no Unreal bridge on {host}:{port} ({e}). Start it with "
                f"scripts/unreal.sh view (or fast); see VERIFY.md 'Unreal verification'") from e
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = b""
        hello = self.request({"op": "hello"})
        if hello.get("protocol") != PROTOCOL:
            raise BridgeError(f"Unreal speaks bridge protocol {hello.get('protocol')}, "
                              f"Python {PROTOCOL}: rebuild the Unreal project")
        self.dt, self.substeps = float(hello["dt"]), int(hello["substeps"])

    def request(self, msg: dict) -> dict:
        self.sock.sendall(json.dumps(msg, separators=(",", ":")).encode() + b"\n")
        while b"\n" not in self._buf:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                raise BridgeError("Unreal closed the connection (did the game exit?)")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        reply = json.loads(line)
        if "error" in reply:
            raise BridgeError(f"Unreal: {reply['error']}")
        return reply

    def close(self) -> None:
        try:
            self.request({"op": "bye"})
        except (OSError, BridgeError):
            pass
        self.sock.close()
