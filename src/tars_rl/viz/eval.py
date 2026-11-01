"""Eval: render the fixed scripted sample to MP4 and print frame count, duration and speed.

    uv run python -m tars_rl.viz.eval --out /tmp/tars_sample.mp4
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

from tars_rl.body.mjcf import load_model, load_spec
from tars_rl.viz.render import render_mp4
from tars_rl.viz.sample import scripted_sine
from tars_rl.viz.trajectory import Trajectory


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("/tmp/tars_sample.mp4"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    a = p.parse_args()

    spec = load_spec()
    model = load_model(spec)
    # Round-trip through the saved file so the eval exercises the Trajectory contract (A10).
    traj_path = a.out.with_suffix(".npz")
    scripted_sine(model, spec.version, seconds=a.seconds).save(traj_path)
    traj = Trajectory.load(traj_path)
    if traj.body_spec_version != spec.version:
        raise SystemExit(f"Trajectory BodySpec v{traj.body_spec_version} != v{spec.version}")

    s = render_mp4(model, traj, a.out, fps=a.fps, width=a.width, height=a.height)
    ok = s.frames == s.expected_frames
    print(f"host            {platform.system()} {platform.machine()}  "
          f"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')}")
    print(f"trajectory      {traj_path}  {traj.n_frames} frames x {traj.dt:g} s = "
          f"{traj.duration:.2f} s sim (BodySpec v{traj.body_spec_version})")
    print(f"video           {a.out}  {a.width}x{a.height} @ {s.fps} fps")
    print(f"frames          {s.frames} (expected {s.expected_frames}) {'OK' if ok else 'MISMATCH'}")
    print(f"video seconds   {s.video_seconds:.2f} (sim {s.sim_seconds:.2f})")
    print(f"render speed    {s.render_seconds:.2f} s wall = "
          f"{s.render_s_per_sim_s:.3f} s render per sim s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
