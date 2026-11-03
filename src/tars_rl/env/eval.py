"""Flag-arena baseline eval. Run: python -m tars_rl.env.eval --episodes 64 --seed 0

Runs a batch of episodes in MJX for two reference policies on the same flags:
  random  uniform actions in [-1, 1] every step
  zero    every actuator held at its centred target
and prints success, fall and timeout rates, flags tagged per episode and time to first tag, start
and final flag distance, distance closed, episode length and return. `--flags single|chain`
picks the flag mode (T26): in chain mode success = at least one flag tagged, and the distances
are to the flag standing at the end (start = when it appeared). These are the floor a trained
policy (T14) must beat. Also saves episode 0 of the random policy as a flag rollout (.npz, A51,
with the per-frame flag track) for the MP4 / Unreal players.

`--obs mark2` runs the same baselines with the Mark 2 senses and then the senses check
(`env.sense_check`): every channel's name, units and range, and the tipped-over touch and
accelerometer checks. The exit code is non-zero if a tipped check fails.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tars_rl.env.config import FLAG_MODES, EnvConfig
from tars_rl.env.flag_env import FlagEnv, State
from tars_rl.env.rollout import FlagRollout
from tars_rl.env.sense_check import report_senses, sense_check
from tars_rl.env.sensors import OBS_SETS

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


def _with_model(env: FlagEnv, fn):
    """`fn` run against one sampled model: vmapping this over a batched model (domain
    randomization) gives every episode its own TARS and floor."""

    def call(model, *args):
        old = env.mjx_model
        env.mjx_model = model
        try:
            return fn(*args)
        finally:
            env.mjx_model = old

    return call


def run_episodes(env: FlagEnv, policy: Policy, rng: jax.Array, n: int,
                 models: tuple | None = None) -> dict[str, np.ndarray]:
    """n episodes in parallel to their end; per-episode results plus episode 0's qpos frames.
    `models`: (batched model, in_axes) from `tars_rl.env.randomize`, one per episode; None runs
    every episode on the nominal model."""
    k_reset, k_pol = jax.random.split(rng)
    if models is None:
        reset, step = jax.vmap(env.reset), jax.vmap(env.step)
    else:
        model_v, axes = models
        reset = functools.partial(jax.vmap(_with_model(env, env.reset), in_axes=(axes, 0)),
                                  model_v)
        step = functools.partial(jax.vmap(_with_model(env, env.step), in_axes=(axes, 0, 0)),
                                 model_v)

    @jax.jit
    def go(k_reset, k_pol):
        s0 = reset(jax.random.split(k_reset, n))

        def body(carry, k):
            s, ret = carry
            new = step(s, policy(k, s.obs))
            ret = ret + jnp.where(s.done, 0.0, new.reward)
            new = _freeze(s.done, s, new)
            return (new, ret), (new.data.qpos[0], new.flag_pos[0])

        keys = jax.random.split(k_pol, env.config.max_steps)
        (s, ret), (q0, f0) = jax.lax.scan(body, (s0, jnp.zeros(n)), keys)
        return (s, ret, jnp.concatenate([s0.data.qpos[:1], q0]),
                jnp.concatenate([s0.flag_pos[:1], f0]), s0.flag_pos)

    s: State
    s, ret, q0, f0, first_flag = go(k_reset, k_pol)
    out = {
        "success": s.tags > 0, "fell": s.fell, "truncated": s.truncated, "done": s.done,
        "start_dist": s.start_dist, "final_dist": s.dist, "length": s.step, "return": ret,
        "flag_pos": first_flag, "tags": s.tags, "first_tag": s.first_tag,
    }
    out = {k: np.asarray(v) for k, v in out.items()}
    out["qpos_ep0"] = np.asarray(q0)[: int(out["length"][0]) + 1]
    out["flag_track_ep0"] = np.asarray(f0)[: int(out["length"][0]) + 1]
    return out


def summarize(r: dict[str, np.ndarray], env: FlagEnv) -> dict[str, float]:
    h = env.height
    tagged = r["first_tag"] >= 0
    return {
        "episodes": int(len(r["success"])),
        "success_rate": float(r["success"].mean()),
        "fall_rate": float(r["fell"].mean()),
        "timeout_rate": float(r["truncated"].mean()),
        "tags_per_ep": float(r["tags"].mean()),
        # mean over the episodes that tagged a flag; nan when none did
        "first_tag_s": float(r["first_tag"][tagged].mean() * env.dt) if tagged.any()
        else float("nan"),
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
        "config": env.config.to_dict(), "obs_set": env.config.obs, "flags": env.config.flags,
        "observation_size": env.observation_size, "policies": {},
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
        FlagRollout(r["qpos_ep0"], env.dt, env.spec.version, r["flag_pos"][0],
                    r["flag_track_ep0"]).save(rollout_path)
        res["rollout"] = str(rollout_path)
    if any(c.needs_mark2 for c in env.channels):
        res["senses"] = sense_check(env, seed)
    return res


def report(res: dict) -> None:
    print(f"Flag arena baseline: {res['spec']} v{res['body_spec_version']}, "
          f"body height {res['body_height_m']:.2f} m, seed {res['seed']}, "
          f"{res['max_steps']} steps x {res['ctrl_dt']} s max, "
          f"observation '{res.get('obs_set', 'default')}' "
          f"({res.get('observation_size', '?')} floats), flags '{res.get('flags', 'single')}'\n")
    rows = [
        ("episodes", "episodes", "{:.0f}"),
        ("success rate", "success_rate", "{:.1%}"),
        ("fall rate", "fall_rate", "{:.1%}"),
        ("timeout rate", "timeout_rate", "{:.1%}"),
        ("flags tagged / episode", "tags_per_ep", "{:.2f}"),
        ("first tag (s)", "first_tag_s", "{:.2f}"),
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
    if "senses" in res:
        report_senses(res["senses"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, help="episode length (default: EnvConfig)")
    p.add_argument("--obs", default="default", choices=sorted(OBS_SETS),
                   help="observation set (mark2 also runs the senses check)")
    p.add_argument("--flags", default="chain", choices=FLAG_MODES,
                   help="chain: a tagged flag respawns; single: tagging ends the episode")
    p.add_argument("--rollout", default="runs/env/random_ep0.npz",
                   help="where to save the sample rollout ('' to skip)")
    p.add_argument("--json", help="also write the results to this JSON file")
    a = p.parse_args(argv)
    secs = {"episode_seconds": a.seconds} if a.seconds else {}
    cfg = EnvConfig(obs=a.obs, flags=a.flags, **secs)
    res = run(a.episodes, a.seed, cfg, a.rollout or None)
    report(res)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    return 0 if res.get("senses", {}).get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
