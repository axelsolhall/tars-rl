"""Eval: render a sample to MP4 and print frame count, duration and speed.

    uv run python -m tars_rl.viz.eval --out /tmp/tars_sample.mp4             # scripted sine (T9)
    uv run python -m tars_rl.viz.eval --flag-sample --out /tmp/tars_flag.mp4  # flag arena (T16)
    uv run python -m tars_rl.viz.eval --rollout runs/x/rollout.npz --out /tmp/x.mp4

Flag runs (the fixed random-policy episode, or any saved flag rollout) also print the start and
end flag distance, measured from the rollout's own poses.
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
    p.add_argument("--flag-sample", action="store_true",
                   help="render a fixed random-policy episode in the flag arena")
    p.add_argument("--rollout", type=Path, help="render this saved flag rollout (.npz, A51)")
    p.add_argument("--seed", type=int, default=0, help="flag sample: episode seed")
    a = p.parse_args()
    if a.flag_sample or a.rollout:
        return flag_main(a)

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


def flag_sample(seed: int, path: Path) -> Path:
    """One random-policy flag-arena episode (same env and policy as the env baseline), saved."""
    import jax

    from tars_rl.env.eval import random_policy, run_episodes
    from tars_rl.env.flag_env import FlagEnv
    from tars_rl.env.rollout import FlagRollout

    env = FlagEnv()
    r = run_episodes(env, random_policy(env.action_size), jax.random.PRNGKey(seed), 1)
    return FlagRollout(r["qpos_ep0"], env.dt, env.spec.version, r["flag_pos"][0]).save(path)


def flag_main(a: argparse.Namespace) -> int:
    from tars_rl.env.rollout import FlagRollout
    from tars_rl.viz.flag import arena_model, flag_distances, render_flag_mp4

    model, spec_version = arena_model()
    path = a.rollout or flag_sample(a.seed, a.out.with_suffix(".npz"))
    rollout = FlagRollout.load(path)  # always render from the saved file (A51)
    if rollout.body_spec_version != spec_version:
        raise SystemExit(f"Rollout BodySpec v{rollout.body_spec_version} != v{spec_version}")

    s = render_flag_mp4(model, rollout, a.out, fps=a.fps, width=a.width, height=a.height)
    d = flag_distances(model, rollout.qpos, rollout.flag_pos)
    ok = s.frames == s.expected_frames
    fx, fy = rollout.flag_pos[:2]
    src = "given" if a.rollout else f"random policy, seed {a.seed}"
    trend = "closer" if d[-1] < d[0] else "farther"
    print(f"host            {platform.system()} {platform.machine()}  "
          f"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')}")
    print(f"rollout         {path}  ({src})  {len(rollout.qpos)} frames x {rollout.dt:g} s = "
          f"{s.sim_seconds:.2f} s sim (BodySpec v{rollout.body_spec_version})")
    print(f"flag            x {fx:+.3f} m, y {fy:+.3f} m")
    print(f"flag distance   start {d[0]:.3f} m -> end {d[-1]:.3f} m "
          f"({trend} by {abs(d[0] - d[-1]):.3f} m)")
    print(f"video           {a.out}  {a.width}x{a.height} @ {s.fps} fps")
    print(f"frames          {s.frames} (expected {s.expected_frames}) {'OK' if ok else 'MISMATCH'}")
    print(f"video seconds   {s.video_seconds:.2f} (sim {s.sim_seconds:.2f})")
    print(f"render speed    {s.render_seconds:.2f} s wall = "
          f"{s.render_s_per_sim_s:.3f} s render per sim s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
