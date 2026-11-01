"""Trajectory -> MP4 with MuJoCo's offscreen renderer and imageio-ffmpeg (A46)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from tars_rl.viz.trajectory import Trajectory


@dataclass(frozen=True)
class RenderStats:
    frames: int  # frames decoded back from the written MP4 (catches encoder drops/dups)
    expected_frames: int  # round(trajectory duration * fps)
    fps: int
    video_seconds: float
    sim_seconds: float
    render_seconds: float  # wall clock spent rendering and encoding

    @property
    def render_s_per_sim_s(self) -> float:
        return self.render_seconds / self.sim_seconds


def frame_indices(traj: Trajectory, fps: int) -> np.ndarray:
    """Trajectory frame shown at each video frame: video frame k shows sim time k / fps."""
    n_video = max(1, round(traj.duration * fps))
    t = np.arange(n_video) / fps
    return np.minimum(np.round(t / traj.dt).astype(int), traj.n_frames - 1)


def tracking_camera(model: mujoco.MjModel, body: str = "central") -> mujoco.MjvCamera:
    """Camera that follows the body so TARS stays in view when it walks, rolls or falls."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    cam.trackbodyid = body_id if body_id >= 0 else 1
    cam.distance = 4.0 * model.stat.extent
    cam.azimuth = 135.0
    cam.elevation = -20.0
    return cam


def count_frames(path: str | Path) -> int:
    """Frames actually in the video file, by decoding it (container metadata can be off)."""
    with imageio.get_reader(path) as reader:
        return sum(1 for _ in reader)


def render_mp4(
    model: mujoco.MjModel,
    traj: Trajectory,
    out: str | Path,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> RenderStats:
    if traj.qpos.shape[1] != model.nq:
        raise ValueError(f"Trajectory nq {traj.qpos.shape[1]} != model nq {model.nq}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    cam = tracking_camera(model)
    idx = frame_indices(traj, fps)

    start = time.perf_counter()
    with (
        mujoco.Renderer(model, height=height, width=width) as renderer,
        imageio.get_writer(out, fps=fps, codec="libx264", macro_block_size=8) as writer,
    ):
        for i in idx:
            data.qpos[:] = traj.qpos[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
    elapsed = time.perf_counter() - start
    frames = count_frames(out)

    return RenderStats(
        frames=frames,
        expected_frames=round(traj.duration * fps),
        fps=fps,
        video_seconds=frames / fps,
        sim_seconds=traj.duration,
        render_seconds=elapsed,
    )
