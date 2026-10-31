"""Flag-arena baseline eval. Run: python -m tars_rl.env.eval --episodes 64 --seed 0

Runs a batch of episodes in MJX for two reference policies on the same flags:
  random  uniform actions in [-1, 1] every step
  zero    every actuator held at its centred target
and prints success, fall and timeout rates, start and final flag distance, distance closed,
episode length and return. These are the floor a trained policy (T14) must beat. Also saves
episode 0 of the random policy as a flag rollout (.npz, A51) for the MP4 / Unreal players.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tars_rl.env.config import EnvConfig
from tars_rl.env.flag_env import FlagEnv, State
from tars_rl.env.rollout import FlagRollout

Policy = Callable[[jax.Array, jax.Array], jax.Array]  # (rng, obs batch) -> action batch


def random_policy(n_act: int) -> Policy:
    return lambda rng, obs: jax.random.uniform(rng, (obs.shape[0], n_act), minval=-1, maxval=1)


def zero_policy(n_act: int) -> Policy:
    return lambda rng, obs: jnp.zeros((obs.shape[0], n_act))


def _freeze(done: jax.Array, old, new):
    """Keep finished episodes exactly as they ended."""

    def pick(o, n):
        return jnp.where(done.reshape(done.shape + (1,) * (n.ndim - done.ndim)), o, n)

    return jax.tree_util.tree_map(pick, old, new)


def run_episodes(env: FlagEnv, policy: Policy, rng: jax.Array, n: int) -> dict[str, np.ndarray]:
    """n episodes in parallel to their end; per-episode results plus episode 0's qpos frames."""
    k_reset, k_pol = jax.random.split(rng)

    @jax.jit
    def go(k_reset, k_pol):
        s0 = jax.vmap(env.reset)(jax.random.split(k_reset, n))

        def body(carry, k):
            s, ret = carry
            new = jax.vmap(env.step)(s, policy(k, s.obs))
            ret = ret + jnp.where(s.done, 0.0, new.reward)
            new = _freeze(s.done, s, new)
            return (new, ret), new.data.qpos[0]

        keys = jax.random.split(k_pol, env.config.max_steps)
        (s, ret), q0 = jax.lax.scan(body, (s0, jnp.zeros(n)), keys)
        return s, ret, jnp.concatenate([s0.data.qpos[:1], q0])

    s: State
    s, ret, q0 = go(k_reset, k_pol)
    out = {
        "success": s.success, "fell": s.fell, "truncated": s.truncated, "done": s.done,
        "start_dist": s.start_dist, "final_dist": s.dist, "length": s.step, "return": ret,
        "flag_pos": s.flag_pos,
    }
    out = {k: np.asarray(v) for k, v in out.items()}
    out["qpos_ep0"] = np.asarray(q0)[: int(out["length"][0]) + 1]
    return out


def summarize(r: dict[str, np.ndarray], env: FlagEnv) -> dict[str, float]:
    h = env.height
    return {
        "episodes": int(len(r["success"])),
        "success_rate": float(r["success"].mean()),
        "fall_rate": float(r["fell"].mean()),
        "timeout_rate": float(r["truncated"].mean()),
        "start_dist_m": float(r["start_dist"].mean()),
        "final_dist_m": float(r["final_dist"].mean()),
        "final_dist_heights": float(r["final_dist"].mean() / h),
        "closed_m": float((r["start_dist"] - r["final_dist"]).mean()),
        "closed_frac": float(((r["start_dist"] - r["final_dist"]) / r["start_dist"]).mean()),
        "length_steps": float(r["length"].mean()),
        "length_s": float(r["length"].mean() * env.dt),
        "return": float(r["return"].mean()),
    }


def run(episodes: int, seed: int, config: EnvConfig | None = None,
        rollout_path: str | Path | None = None) -> dict:
    env = FlagEnv(config=config)
    rng = jax.random.PRNGKey(seed)  # same key for both policies: same flags
    res: dict = {
        "spec": env.spec.name, "body_spec_version": env.spec.version, "seed": seed,
        "body_height_m": env.height, "max_steps": env.config.max_steps, "ctrl_dt": env.dt,
        "config": env.config.to_dict(), "policies": {},
    }
    raw = {}
    for name, make in (("random", random_policy), ("zero", zero_policy)):
        t0 = time.perf_counter()
        raw[name] = run_episodes(env, make(env.action_size), rng, episodes)
        s = summarize(raw[name], env)
        s["wall_s"] = time.perf_counter() - t0
        res["policies"][name] = s
    if rollout_path:
        r = raw["random"]
        FlagRollout(r["qpos_ep0"], env.dt, env.spec.version, r["flag_pos"][0]).save(rollout_path)
        res["rollout"] = str(rollout_path)
    return res


def report(res: dict) -> None:
    print(f"Flag arena baseline: {res['spec']} v{res['body_spec_version']}, "
          f"body height {res['body_height_m']:.2f} m, seed {res['seed']}, "
          f"{res['max_steps']} steps x {res['ctrl_dt']} s max\n")
    rows = [
        ("episodes", "episodes", "{:.0f}"),
        ("success rate", "success_rate", "{:.1%}"),
        ("fall rate", "fall_rate", "{:.1%}"),
        ("timeout rate", "timeout_rate", "{:.1%}"),
        ("start dist (m)", "start_dist_m", "{:.3f}"),
        ("final dist (m)", "final_dist_m", "{:.3f}"),
        ("final dist (heights)", "final_dist_heights", "{:.2f}"),
        ("dist closed (m)", "closed_m", "{:+.3f}"),
        ("dist closed (frac)", "closed_frac", "{:+.1%}"),
        ("episode length (steps)", "length_steps", "{:.1f}"),
        ("episode length (s)", "length_s", "{:.2f}"),
        ("return", "return", "{:+.3f}"),
        ("wall time (s)", "wall_s", "{:.1f}"),
    ]
    pols = list(res["policies"])
    print(f"{'':<24}" + "".join(f"{p:>12}" for p in pols))
    for label, key, fmt in rows:
        cells = (fmt.format(res["policies"][p][key]) for p in pols)
        print(f"{label:<24}" + "".join(f"{c:>12}" for c in cells))
    if "rollout" in res:
        print(f"\nSample rollout (random policy, episode 0): {res['rollout']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, help="episode length (default: EnvConfig)")
    p.add_argument("--rollout", default="runs/env/random_ep0.npz",
                   help="where to save the sample rollout ('' to skip)")
    p.add_argument("--json", help="also write the results to this JSON file")
    a = p.parse_args(argv)
    cfg = EnvConfig(episode_seconds=a.seconds) if a.seconds else EnvConfig()
    res = run(a.episodes, a.seed, cfg, a.rollout or None)
    report(res)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
