"""Train TARS to reach the flag.

Run: python -m tars_rl.train.run --profile smoke --seed 0 --out runs/smoke

MuJoCo Playground's PPO (brax) on the T13 flag arena (A58), settings from a profile (A59), the
policy and value networks built from the profile's network description or `--network` (T18).
Writes one policy folder (A60, layout in `tars_rl.train.policy`):
  progress.jsonl  learning curve (A61): a "ppo" line per brax eval (reward, episode metrics) and
                  a "curve" line with our own flag eval on fixed held-out flags (success rate,
                  flags tagged per episode, time to first tag, fall rate, distance left), so
                  the curve says whether TARS gets to the flag
  policy.pkl, meta.json, rollout.npz (episode 0 of the trained policy, for T9's MP4)
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import jax
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from tars_rl.env.config import EnvConfig
from tars_rl.env.eval import _freeze, run_episodes  # T13's batch runner; freezes done episodes
from tars_rl.env.flag_env import FlagEnv
from tars_rl.env.rollout import FlagRollout
from tars_rl.train.adapter import PlaygroundFlagEnv
from tars_rl.train.config import PPOConfig, get_profile
from tars_rl.train.networks import network_factory
from tars_rl.train.policy import PROGRESS_FILE, ROLLOUT_FILE, as_policy, make_inference, save_policy

CURVE_SEED_OFFSET = 1_000  # curve flags come from seed + this, never the training seed's draws

# full_reset: every finished episode calls FlagEnv.reset, so it draws a new flag (A22). The default
# would restore each env's cached first state, i.e. the same flag for the whole run.
WRAP_FOR_TRAINING = functools.partial(wrapper.wrap_for_brax_training, full_reset=True)


def make_curve_eval(env: FlagEnv, make_inference_fn, n: int, rng: jax.Array):
    """`params -> metrics` on the same n flags every call; compiled once for the whole run."""
    k_reset, k_pol = jax.random.split(rng)

    @jax.jit
    def go(params):
        policy = as_policy(make_inference_fn, params)
        s0 = jax.vmap(env.reset)(jax.random.split(k_reset, n))

        def body(s, k):
            new = jax.vmap(env.step)(s, policy(k, s.obs))
            return _freeze(s.done, s, new), None

        s, _ = jax.lax.scan(body, s0, jax.random.split(k_pol, env.config.max_steps))
        tagged = s.first_tag >= 0
        return {
            "success_rate": (s.tags > 0).mean(),
            "tags_per_ep": s.tags.mean(),
            # mean over the episodes that tagged a flag; nan when none did
            "first_tag_s": (s.first_tag * tagged).sum() / tagged.sum() * env.dt,
            "fall_rate": s.fell.mean(),
            "final_dist_m": s.dist.mean(),
            "closed_frac": ((s.start_dist - s.dist) / s.start_dist).mean(),
            "length_steps": s.step.mean(),
        }

    return lambda params: {k: float(v) for k, v in go(tuple(params[:2])).items()}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def train(profile: str, seed: int, out: str | Path, cfg: PPOConfig | None = None,
          env_config: EnvConfig | None = None, log=print) -> dict:
    cfg = cfg or get_profile(profile)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / PROGRESS_FILE
    progress.write_text("")

    env = FlagEnv(config=env_config)
    penv = PlaygroundFlagEnv(env)
    inference = make_inference(env.observation_size, env.action_size, cfg.network,
                               cfg.normalize_observations)
    log(f"network {cfg.network}")
    curve_eval = make_curve_eval(env, inference, cfg.curve_episodes,
                                 jax.random.PRNGKey(seed + CURVE_SEED_OFFSET))
    t0 = time.perf_counter()

    def write(line: dict) -> None:
        with progress.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def progress_fn(step: int, metrics: dict) -> None:
        m = {k: float(v) for k, v in metrics.items() if jax.numpy.ndim(v) == 0}
        write({"kind": "ppo", "step": int(step), "wall_s": time.perf_counter() - t0, **m})
        log(f"step {int(step):>12,}  reward {m.get('eval/episode_reward', float('nan')):+.3f}")

    def policy_params_fn(step: int, _make_policy, params) -> None:
        c = curve_eval(params)
        write({"kind": "curve", "step": int(step), "wall_s": time.perf_counter() - t0, **c})
        log(f"step {int(step):>12,}  success {c['success_rate']:.1%}  "
            f"tags/ep {c['tags_per_ep']:.2f}  "
            f"dist left {c['final_dist_m']:.2f} m  falls {c['fall_rate']:.1%}")

    _, params, _ = ppo.train(
        environment=penv,
        wrap_env_fn=WRAP_FOR_TRAINING,
        num_timesteps=cfg.num_timesteps,
        num_evals=cfg.num_evals,
        episode_length=cfg.episode_length,
        num_envs=cfg.num_envs,
        num_eval_envs=cfg.num_eval_envs,
        batch_size=cfg.batch_size,
        num_minibatches=cfg.num_minibatches,
        unroll_length=cfg.unroll_length,
        num_updates_per_batch=cfg.num_updates_per_batch,
        discounting=cfg.discounting,
        learning_rate=cfg.learning_rate,
        entropy_cost=cfg.entropy_cost,
        reward_scaling=cfg.reward_scaling,
        max_grad_norm=cfg.max_grad_norm,
        normalize_observations=cfg.normalize_observations,
        action_repeat=cfg.action_repeat,
        network_factory=network_factory(cfg.network),
        seed=seed,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )
    wall = time.perf_counter() - t0

    meta = {
        "profile": profile, "seed": seed, "ppo": cfg.to_dict(), "env": env.config.to_dict(),
        "body_spec": env.spec.name, "body_spec_version": env.spec.version,
        "observation_size": env.observation_size, "action_size": env.action_size,
        "num_timesteps": cfg.num_timesteps, "train_wall_s": wall, "git_commit": _git_commit(),
        "versions": {p: _version(p) for p in ("jax", "mujoco", "mujoco-mjx", "brax", "playground")},
    }
    save_policy(out, params, meta)

    r = run_episodes(env, as_policy(inference, params), jax.random.PRNGKey(seed), 1)
    FlagRollout(r["qpos_ep0"], env.dt, env.spec.version, r["flag_pos"][0],
                r["flag_track_ep0"]).save(out / ROLLOUT_FILE)
    log(f"saved policy, curve and rollout to {out} ({wall:.0f} s training)")
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--timesteps", type=int, help="override the profile's num_timesteps")
    p.add_argument("--network", help="network description (tars_rl.train.networks), e.g. "
                   "'policy=simba-2x128-relu-layer-e4 value=bro-2x256-relu-layer'; "
                   "default: the profile's")
    a = p.parse_args(argv)
    over = {"num_timesteps": a.timesteps} if a.timesteps else {}
    if a.network:
        over["network"] = a.network
    train(a.profile, a.seed, a.out, get_profile(a.profile, **over))
    return 0


if __name__ == "__main__":
    sys.exit(main())
