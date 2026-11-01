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

from tars_rl.env.config import EnvConfig, RewardConfig
from tars_rl.env.eval import random_policy, report, run_episodes, summarize, zero_policy
from tars_rl.env.flag_env import FlagEnv
from tars_rl.train.policy import load_policy, read_progress

EVAL_SEED_OFFSET = 2_000  # eval flags differ from training and learning-curve flags


def evaluate(policy_dir: str | Path, episodes: int, seed: int, baselines: bool = True) -> dict:
    saved = load_policy(policy_dir)
    m = saved.meta
    env = FlagEnv(config=EnvConfig(**{**m["env"], "reward": RewardConfig(**m["env"]["reward"])}))
    if env.spec.version != m["body_spec_version"]:
        raise ValueError(f"policy trained on BodySpec v{m['body_spec_version']}, "
                         f"env has v{env.spec.version}")
    rng = jax.random.PRNGKey(seed + EVAL_SEED_OFFSET)  # same key per policy: same flags
    res: dict = {
        "spec": env.spec.name, "body_spec_version": env.spec.version, "seed": seed,
        "body_height_m": env.height, "max_steps": env.config.max_steps, "ctrl_dt": env.dt,
        "policy": str(policy_dir), "profile": m["profile"], "policies": {},
    }
    runs = [("trained", saved.policy())]
    if baselines:
        n = env.action_size
        runs += [("random", random_policy(n)), ("still", zero_policy(n))]
    for name, pol in runs:
        t0 = time.perf_counter()
        s = summarize(run_episodes(env, pol, rng, episodes), env)
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
    print(f"Policy {res['policy']} (profile {res['profile']})")
    report(res)
    print(f"\nTrained beats random and still (success up, distance down): "
          f"{'YES' if res['beats_baselines'] else 'NO'}")
    if res["curve"]:
        print(f"\nLearning curve ({len(res['curve'])} points, fixed held-out flags)")
        print(f"{'step':>14}{'success':>10}{'dist left (m)':>15}{'falls':>8}{'wall (s)':>10}")
        for r in res["curve"]:
            print(f"{r['step']:>14,}{r['success_rate']:>10.1%}{r['final_dist_m']:>15.3f}"
                  f"{r['fall_rate']:>8.1%}{r['wall_s']:>10.0f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", required=True, help="policy folder written by tars_rl.train.run")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write the results to this JSON file")
    a = p.parse_args(argv)
    res = evaluate(a.policy, a.episodes, a.seed)
    report_all(res)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
