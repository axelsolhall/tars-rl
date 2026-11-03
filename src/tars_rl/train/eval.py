"""Score a trained policy against random moves and standing still.

Run: python -m tars_rl.train.eval --policy runs/smoke --episodes 64 --seed 0

Runs the trained policy, random moves and standing still (all actuators at their centred target)
on the same flags with T13's batch runner, and prints success, falls, distance left, distance
closed and episode length side by side, then the run's learning curve from progress.jsonl.
Passes when trained beats both baselines on success rate and final distance; prints the verdict
but exits 0 either way (the smoke policy is not expected to beat them).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

from tars_rl.env import randomize
from tars_rl.env.config import FLAG_MODES, EnvConfig, RewardConfig
from tars_rl.env.eval import random_policy, report, run_episodes, summarize, zero_policy
from tars_rl.env.flag_env import FlagEnv
from tars_rl.env.randomize import DomainRandConfig
from tars_rl.train.policy import load_policy, read_progress

EVAL_SEED_OFFSET = 2_000  # eval flags differ from training and learning-curve flags
EVAL_MODELS_OFFSET = 3_000  # held-out randomized bodies (training draws from its own seed)


def evaluate(policy_dir: str | Path, episodes: int, seed: int, baselines: bool = True,
             flags: str | None = None, domain_rand: DomainRandConfig | None = None) -> dict:
    """`flags` overrides the flag mode (T26); runs saved before T26 have none and get chain.
    `domain_rand`: every episode gets its own randomized TARS and floor, drawn from a key no
    training run uses (held-out bodies), the same draws for every policy."""
    saved = load_policy(policy_dir)
    m = saved.meta
    over = {"flags": flags} if flags else {}
    env = FlagEnv(config=EnvConfig(**{**m["env"], "reward": RewardConfig(**m["env"]["reward"]),
                                      **over}))
    if env.spec.version != m["body_spec_version"]:
        raise ValueError(f"policy trained on BodySpec v{m['body_spec_version']}, "
                         f"env has v{env.spec.version}")
    rng = jax.random.PRNGKey(seed + EVAL_SEED_OFFSET)  # same key per policy: same flags
    res: dict = {
        "spec": env.spec.name, "body_spec_version": env.spec.version, "seed": seed,
        "body_height_m": env.height, "max_steps": env.config.max_steps, "ctrl_dt": env.dt,
        "policy": str(policy_dir), "profile": m["profile"], "algo": saved.algo,
        "obs_set": env.config.obs, "observation_size": env.observation_size,
        "flags": env.config.flags, "policies": {},
        "domain_rand": domain_rand.to_dict() if domain_rand is not None else None,
    }
    models = None
    if domain_rand is not None:
        keys = jax.random.split(jax.random.PRNGKey(seed + EVAL_MODELS_OFFSET), episodes)
        models = randomize.for_env(env, domain_rand)(env.mjx_model, keys)
    runs = [("trained", saved.policy())]
    if baselines:
        n = env.action_size
        runs += [("random", random_policy(n)), ("still", zero_policy(n))]
    for name, pol in runs:
        t0 = time.perf_counter()
        s = summarize(run_episodes(env, pol, rng, episodes, models), env)
        s["wall_s"] = time.perf_counter() - t0
        res["policies"][name] = s
    p = res["policies"]
    res["beats_baselines"] = baselines and all(
        p["trained"]["success_rate"] > p[b]["success_rate"]
        and p["trained"]["final_dist_m"] < p[b]["final_dist_m"]
        for b in ("random", "still")
    )
    res["curve"] = [r for r in read_progress(policy_dir) if r.get("kind") == "curve"]
    return res


def report_all(res: dict) -> None:
    print(f"Policy {res['policy']} ({res.get('algo', 'ppo')}, profile {res['profile']})")
    if res.get("domain_rand"):
        print(f"Scored on randomized TARS: {res['domain_rand']}")
    report(res)
    print(f"\nTrained beats random and still (success up, distance down): "
          f"{'YES' if res['beats_baselines'] else 'NO'}")
    if res["curve"]:
        print(f"\nLearning curve ({len(res['curve'])} points, fixed held-out flags)")
        print(f"{'step':>14}{'success':>10}{'tags/ep':>9}{'1st tag (s)':>12}"
              f"{'dist left (m)':>15}{'falls':>8}{'wall (s)':>10}")
        nan = float("nan")  # curves written before T26 have no tag columns
        for r in res["curve"]:
            print(f"{r['step']:>14,}{r['success_rate']:>10.1%}{r.get('tags_per_ep', nan):>9.2f}"
                  f"{r.get('first_tag_s', nan):>12.2f}{r['final_dist_m']:>15.3f}"
                  f"{r['fall_rate']:>8.1%}{r['wall_s']:>10.0f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", required=True, help="policy folder written by tars_rl.train.run")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flags", choices=FLAG_MODES,
                   help="flag mode (default: the run's own; chain for runs saved before T26)")
    p.add_argument("--domain-rand", action="store_true",
                   help="score on randomized TARS bodies and floors (default ranges)")
    p.add_argument("--json", help="also write the results to this JSON file")
    a = p.parse_args(argv)
    res = evaluate(a.policy, a.episodes, a.seed, flags=a.flags,
                   domain_rand=DomainRandConfig() if a.domain_rand else None)
    report_all(res)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
