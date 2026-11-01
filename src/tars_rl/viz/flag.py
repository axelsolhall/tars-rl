"""Flag-arena rollout -> MP4: the arena model with the flag where it stood (A51) and a camera
that frames TARS and the flag for the whole clip. A chain-mode rollout (T26) carries a per-frame
flag track: the flag moves after each tag and the camera frames every flag of the chain."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from tars_rl.env.arena import FLAG_BODY, build_arena_mjcf, load_body_spec
from tars_rl.env.rollout import FlagRollout
from tars_rl.viz.render import RenderStats, render_mp4
from tars_rl.viz.trajectory import Trajectory

CENTRAL_BODY = "central"


def arena_model(body_spec_path: str | Path | None = None) -> tuple[mujoco.MjModel, int]:
    """The flag arena's MuJoCo model and its BodySpec version."""
    spec = load_body_spec(body_spec_path)
    return mujoco.MjModel.from_xml_string(build_arena_mjcf(spec)), spec.version


def as_trajectory(r: FlagRollout) -> Trajectory:
    return Trajectory(qpos=r.qpos, dt=r.dt, body_spec_version=r.body_spec_version)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise ValueError(f"model has no body '{name}'")
    return i


def central_xy(model: mujoco.MjModel, qpos: np.ndarray, body: str = CENTRAL_BODY) -> np.ndarray:
    """(frames, 2) world xy of the central body's centre, as the env measures flag distance."""
    bid = _body_id(model, body)
    data = mujoco.MjData(model)
    out = np.empty((len(qpos), 2))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        out[i] = data.xpos[bid][:2]
    return out


def flag_distances(model: mujoco.MjModel, qpos: np.ndarray, flag_pos: np.ndarray) -> np.ndarray:
    """Per-frame xy distance from the central body to the flag (same as FlagEnv's `dist`).
    `flag_pos` is one flag (3,) or a per-frame track (frames, 3)."""
    return np.linalg.norm(central_xy(model, qpos) - np.asarray(flag_pos)[..., :2], axis=1)


def framing_camera(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    flag_pos: np.ndarray,
    width: int,
    height: int,
    pad: float = 0.4,  # metres around TARS's path and the flag (body is ~0.36 m tall, A7)
    elevation: float = -30.0,
) -> mujoco.MjvCamera:
    """Fixed camera that sees every central-body position of the run and the flag.

    Looks at the centre of their xy bounding box, far enough that the box's half diagonal + pad
    fits the narrower field of view. The heading is the one that best keeps TARS and the flag
    apart across the image (view across the start->flag line) while seeing the checkered cloth,
    which hangs in the arena's y-z plane, face-on (view along x): max of the worse of the two.
    """
    xy = central_xy(model, qpos)
    flags_xy = np.atleast_2d(np.asarray(flag_pos))[:, :2]  # one flag, or every one of a chain
    flag_xy = flags_xy[0]
    pts = np.vstack([xy, flags_xy])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) / 2 + pad

    fovy = math.radians(model.vis.global_.fovy)
    fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
    half_fov = min(fovy, fovx) / 2

    line = flag_xy - xy[0]
    line_az = math.atan2(line[1], line[0])
    cands = np.radians(np.arange(0.0, 180.0, 5.0))
    score = np.minimum(np.abs(np.sin(cands - line_az)), np.abs(np.cos(cands)))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (centre[0], centre[1], pad / 2)
    cam.distance = radius / math.tan(half_fov)
    cam.azimuth = math.degrees(cands[int(np.argmax(score))])
    cam.elevation = elevation
    return cam


def render_flag_mp4(
    model: mujoco.MjModel,
    rollout: FlagRollout,
    out: str | Path,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> RenderStats:
    traj = as_trajectory(rollout)
    _body_id(model, FLAG_BODY)
    if model.nmocap != 1:
        raise ValueError(f"expected one mocap body (the flag), model has {model.nmocap}")
    if rollout.flag_track is None:  # rollouts saved before T26: one flag for the whole clip
        cam = framing_camera(model, rollout.qpos, rollout.flag_pos, width, height)
        return render_mp4(model, traj, out, fps=fps, width=width, height=height,
                          camera=cam, mocap_pos=np.asarray(rollout.flag_pos)[None, :])
    track = np.asarray(rollout.flag_track)  # chain mode: the flag moves after each tag
    cam = framing_camera(model, rollout.qpos, np.unique(track, axis=0), width, height)
    return render_mp4(model, traj, out, fps=fps, width=width, height=height,
                      camera=cam, mocap_track=track[:, None, :])
