"""The G3 comparison: retrain points along a finished campaign from scratch on fresh seeds.

Run: python -m tars_rl.tune.compare --campaign runs/tune_smoke --profile smoke --seeds 2
     --points 3 --out runs/compare_smoke

Points (A86): the published start (trial 0), the final best, and the best-so-far settings at
evenly spaced places in between (after 1, ..., N trials of the campaign). A trial that is the
best at several places is trained once and shared.

Every point trains from scratch on the same seeds (A87), fresh ones the campaign never used
(campaign seed + 1, + 2, ...), until its validation quality stops improving: training runs up to
`max_timesteps`, our flag eval runs `num_evals` times along the way on the run's validation
flags, and the run stops once `patience` evals in a row fail to beat the best by `min_delta`
(A110). The policy kept is the one at the best eval. It is then scored on held-out flags (fresh
too: campaign score seed + 1, T15's eval offset) that neither the campaign nor stopping saw.

Writes, in --out:
  compare.json                      what the comparison runs with: campaign, profile, seeds,
                                    held-out score seed, and the points (place, trial, settings)
  trials/NNNN/seedS/                the retrained policy folder (A60) plus result.json: held-out
                                    results, validation curve, stop reason and step, wall time
  video/rollout.npz, video.json     one held-out episode of the final best reaching the flag (or
                                    the closest it got), for `eval_compare` to render

Stopped halfway, the same command resumes: finished (trial, seed) runs are not retrained.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tars_rl.train.config import PPOConfig, get_profile
from tars_rl.tune.campaign import CAMPAIGN_FILE, read_records
from tars_rl.tune.score import quality
from tars_rl.tune.tars_space import to_config

SCHEMA_VERSION = 1
COMPARE_FILE = "compare.json"
RESULT_FILE = "result.json"
VIDEO_DIR = "video"
VIDEO_ROLLOUT = "rollout.npz"
VIDEO_META = "video.json"


@dataclass(frozen=True)
class CompareProfile:
    base: str  # train profile giving env counts, batches and episode length
    max_timesteps: int  # longest a point may train if it never stops improving
    num_evals: int  # validation evals over max_timesteps (brax counts the one before training)
    patience: int  # evals in a row without improvement before the run stops (A110)
    min_delta: float  # improvement in validation quality that counts
    heldout_episodes: int  # held-out flags each retrained policy is scored on
    video_episodes: int  # held-out flags searched for the video's successful episode


PROFILES: dict[str, CompareProfile] = {
    # Proves the pipeline on the Mac. 20,480 steps = 4 training epochs of one step each at the
    # longest unroll (32 x 40 x 4), so a patience of 2 can trigger before the budget ends.
    "smoke": CompareProfile("smoke", 20_480, 5, 2, 0.01, 16, 16),
    # The 5080 comparison (A110): up to twice T15's 200M-step run, an eval every 10M steps,
    # stop after 50M steps without improvement.
    "full": CompareProfile("full", 400_000_000, 41, 5, 0.01, 256, 64),
}

# (cfg, out_dir, seed, profile, score_seed, log) -> result dict (see `retrain_point`)
Retrain = Callable[..., dict]
# (policy_dir, out_dir, episodes, score_seed, log) -> video meta dict
Record = Callable[..., dict]


class _Plateau(Exception):
    """Raised from brax's eval callback to end a run whose validation stopped improving."""


def best_so_far(recs: list[dict]) -> list[dict]:
    """For each campaign trial in order, the best scored record up to and including it."""
    out, best = [], None
    for r in sorted(recs, key=lambda r: r["id"]):
        if best is None or r["score"] > best["score"]:
            best = r
        out.append(best)
    return out


def pick_points(recs: list[dict], n: int) -> list[dict]:
    """A86: the published start, n - 2 evenly spaced best-so-far points, the final best."""
    recs = [r for r in recs if r.get("score") is not None]
    if not recs or min(r["id"] for r in recs) != 0:
        raise ValueError("the campaign needs its trial 0 (the published settings) scored")
    if n < 2:
        raise ValueError("--points must be at least 2 (start and final best)")
    bsf = best_so_far(recs)
    last = len(bsf) - 1
    places = sorted({round(i * last / (n - 1)) for i in range(n)})
    points = []
    for j, k in enumerate(places):
        r = bsf[k]
        label = ("published start" if j == 0 else "final best" if k == last
                 else f"best after {k + 1} trials")
        points.append({"index": j, "after_trials": k + 1, "trial": r["id"], "label": label,
                       "campaign_score": r["score"], "settings": r["settings"],
                       "network_str": r.get("network_str")})
    return points


def _manifest(campaign: Path, profile: str, seeds: list[int], score_seed: int,
              points: list[dict]) -> dict:
    return {"schema_version": SCHEMA_VERSION, "campaign": str(campaign), "profile": profile,
            "compare_profile": asdict(PROFILES[profile]), "seeds": seeds,
            "score_seed": score_seed, "points": points}


def _check_manifest(out: Path, want: dict) -> None:
    path = out / COMPARE_FILE
    if not path.exists():
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(want, indent=1) + "\n")
        return
    have = json.loads(path.read_text())
    diff = [k for k in want if have.get(k) != json.loads(json.dumps(want[k]))]
    if diff:
        raise ValueError(f"{out} holds a comparison with different {diff}; use another --out")


def run_dir(out: Path, trial: int, seed: int) -> Path:
    return out / "trials" / f"{trial:04d}" / f"seed{seed}"


def read_result(out: str | Path, trial: int, seed: int) -> dict | None:
    p = run_dir(Path(out), trial, seed) / RESULT_FILE
    return json.loads(p.read_text()) if p.exists() else None


def train_until_plateau(cfg: PPOConfig, out: Path, seed: int, cp: CompareProfile,
                        profile: str, log=print) -> dict:
    """T15's training (`train.run.train`, same env, wrapper, curve eval and policy folder) with
    an early stop: keeps the params of the best validation eval and ends the run once `patience`
    evals in a row miss it by `min_delta`. Returns the stop record."""
    import jax
    from brax.training.agents.ppo import train as ppo

    from tars_rl.env.eval import run_episodes
    from tars_rl.env.flag_env import FlagEnv
    from tars_rl.env.rollout import FlagRollout
    from tars_rl.train.adapter import PlaygroundFlagEnv
    from tars_rl.train.networks import network_factory
    from tars_rl.train.policy import (
        PROGRESS_FILE,
        ROLLOUT_FILE,
        as_policy,
        make_inference,
        save_policy,
    )
    from tars_rl.train.run import (
        CURVE_SEED_OFFSET,
        WRAP_FOR_TRAINING,
        _git_commit,
        _version,
        make_curve_eval,
    )

    out.mkdir(parents=True, exist_ok=True)
    progress = out / PROGRESS_FILE
    progress.write_text("")
    env = FlagEnv()
    inference = make_inference(env.observation_size, env.action_size, cfg.network,
                               cfg.normalize_observations)
    curve_eval = make_curve_eval(env, inference, cfg.curve_episodes,
                                 jax.random.PRNGKey(seed + CURVE_SEED_OFFSET))
    st = {"best_quality": float("-inf"), "best_step": 0, "since": 0, "params": None,
          "last_step": 0}
    t0 = time.perf_counter()

    def write(line: dict) -> None:
        with progress.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def progress_fn(step: int, metrics: dict) -> None:
        m = {k: float(v) for k, v in metrics.items() if jax.numpy.ndim(v) == 0}
        write({"kind": "ppo", "step": int(step), "wall_s": time.perf_counter() - t0, **m})

    def policy_params_fn(step: int, _make_policy, params) -> None:
        c = curve_eval(params)
        q = quality(c["success_rate"], c["closed_frac"])
        write({"kind": "curve", "step": int(step), "wall_s": time.perf_counter() - t0,
               "quality": q, **c})
        st["last_step"] = int(step)
        if q > st["best_quality"] + cp.min_delta or st["params"] is None:
            st.update(best_quality=max(q, st["best_quality"]), best_step=int(step), since=0,
                      params=jax.device_get(tuple(params[:2])))
        else:
            st["since"] += 1
        log(f"  step {int(step):>12,}  val success {c['success_rate']:.1%}  quality {q:+.3f}  "
            f"best {st['best_quality']:+.3f} @ {st['best_step']:,}  ({st['since']}/{cp.patience})")
        if st["since"] >= cp.patience:
            raise _Plateau

    reason = "budget"  # ran to max_timesteps: still improving, or improving too slowly to stop
    try:
        ppo.train(
            environment=PlaygroundFlagEnv(env), wrap_env_fn=WRAP_FOR_TRAINING,
            num_timesteps=cfg.num_timesteps, num_evals=cfg.num_evals,
            episode_length=cfg.episode_length, num_envs=cfg.num_envs,
            num_eval_envs=cfg.num_eval_envs, batch_size=cfg.batch_size,
            num_minibatches=cfg.num_minibatches, unroll_length=cfg.unroll_length,
            num_updates_per_batch=cfg.num_updates_per_batch, discounting=cfg.discounting,
            learning_rate=cfg.learning_rate, entropy_cost=cfg.entropy_cost,
            reward_scaling=cfg.reward_scaling, max_grad_norm=cfg.max_grad_norm,
            normalize_observations=cfg.normalize_observations, action_repeat=cfg.action_repeat,
            network_factory=network_factory(cfg.network), seed=seed,
            progress_fn=progress_fn, policy_params_fn=policy_params_fn,
        )
    except _Plateau:
        reason = "plateau"
    wall = time.perf_counter() - t0
    stop = {"reason": reason, "best_step": st["best_step"], "steps_trained": st["last_step"],
            "best_quality": st["best_quality"], "patience": cp.patience,
            "min_delta": cp.min_delta}
    meta = {
        "profile": profile, "seed": seed, "ppo": cfg.to_dict(), "env": env.config.to_dict(),
        "body_spec": env.spec.name, "body_spec_version": env.spec.version,
        "observation_size": env.observation_size, "action_size": env.action_size,
        "num_timesteps": st["best_step"], "train_wall_s": wall, "git_commit": _git_commit(),
        "versions": {p: _version(p) for p in ("jax", "mujoco", "mujoco-mjx", "brax", "playground")},
        "stop": stop,
    }
    save_policy(out, st["params"], meta)
    r = run_episodes(env, as_policy(inference, st["params"]), jax.random.PRNGKey(seed), 1)
    FlagRollout(r["qpos_ep0"], env.dt, env.spec.version, r["flag_pos"][0]).save(out / ROLLOUT_FILE)
    log(f"  stopped ({reason}) after {st['last_step']:,} steps; kept step {st['best_step']:,} "
        f"({wall:.0f} s)")
    return stop


def retrain_point(cfg: PPOConfig, out: Path, seed: int, profile: str, score_seed: int,
                  log=print) -> dict:
    """Train until plateau, then score the kept policy on the held-out flags."""
    from tars_rl.train.eval import evaluate

    cp = PROFILES[profile]
    stop = train_until_plateau(cfg, out, seed, cp, f"compare-{profile}", log)
    res = evaluate(out, cp.heldout_episodes, score_seed, baselines=False)
    return {"heldout": res["policies"]["trained"], "curve": res["curve"], "stop": stop}


def record_video(policy_dir: Path, out: Path, episodes: int, score_seed: int, log=print) -> dict:
    """One held-out episode of a policy for the video: the first that reaches the flag, else the
    one that ends closest. Replays that episode's reset alone and saves it as a FlagRollout."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from tars_rl.env.config import EnvConfig, RewardConfig
    from tars_rl.env.eval import _freeze, run_episodes
    from tars_rl.env.flag_env import FlagEnv
    from tars_rl.env.rollout import FlagRollout
    from tars_rl.train.eval import EVAL_SEED_OFFSET
    from tars_rl.train.policy import load_policy

    saved = load_policy(policy_dir)
    m = saved.meta
    env = FlagEnv(config=EnvConfig(**{**m["env"], "reward": RewardConfig(**m["env"]["reward"])}))
    policy = saved.policy()
    rng = jax.random.PRNGKey(score_seed + EVAL_SEED_OFFSET)  # the held-out flags (train.eval)
    r = run_episodes(env, policy, rng, episodes)
    hits = np.flatnonzero(r["success"])
    i = int(hits[0]) if len(hits) else int(np.argmin(r["final_dist"]))
    k_reset, k_pol = jax.random.split(rng)
    key = jax.random.split(k_reset, episodes)[i]

    @jax.jit
    def go(key, k_pol):
        s0 = jax.vmap(env.reset)(key[None])

        def body(s, k):
            new = _freeze(s.done, s, jax.vmap(env.step)(s, policy(k, s.obs)))
            return new, new.data.qpos[0]

        s, q = jax.lax.scan(body, s0, jax.random.split(k_pol, env.config.max_steps))
        return s, jnp.concatenate([s0.data.qpos[:1], q])

    s, q = go(key, k_pol)
    n = int(s.step[0])
    out.mkdir(parents=True, exist_ok=True)
    FlagRollout(np.asarray(q)[: n + 1], env.dt, env.spec.version,
                np.asarray(s.flag_pos[0])).save(out / VIDEO_ROLLOUT)
    meta = {"policy": str(policy_dir), "episode": i, "of_episodes": episodes,
            "reached_flag": bool(s.success[0]), "fell": bool(s.fell[0]), "steps": n,
            "start_dist_m": float(s.start_dist[0]), "final_dist_m": float(s.dist[0]),
            "heldout_successes": int(len(hits))}
    (out / VIDEO_META).write_text(json.dumps(meta, indent=1) + "\n")
    log(f"video episode {i}: reached flag {meta['reached_flag']}, "
        f"{meta['start_dist_m']:.2f} m -> {meta['final_dist_m']:.2f} m")
    return meta


def compare(campaign: str | Path, profile: str, seeds: int, points: int, out: str | Path,
            seed_base: int | None = None, score_seed: int | None = None,
            runner: Retrain | None = None, recorder: Record | None = None, log=print) -> dict:
    """Retrain every point on every seed (or read the finished ones), then record the video."""
    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
    campaign, out, cp = Path(campaign), Path(out), PROFILES[profile]
    cm = json.loads((campaign / CAMPAIGN_FILE).read_text())
    seed_base = cm["seed"] + 1 if seed_base is None else seed_base  # fresh seeds (A87, A111)
    score_seed = cm["score_seed"] + 1 if score_seed is None else score_seed  # fresh flags (A111)
    seed_list = list(range(seed_base, seed_base + seeds))
    pts = pick_points(read_records(campaign), points)
    _check_manifest(out, _manifest(campaign, profile, seed_list, score_seed, pts))
    runner = runner or retrain_point
    recorder = recorder or record_video
    base = get_profile(cp.base, num_timesteps=cp.max_timesteps, num_evals=cp.num_evals)
    trials = {p["trial"]: p for p in pts}  # one training per trial, shared by its points
    for tid, p in trials.items():
        cfg = replace(to_config(p["settings"], base), num_timesteps=cp.max_timesteps)
        for s in seed_list:
            if read_result(out, tid, s) is not None:
                log(f"trial {tid:>3} seed {s}: already done, not retrained")
                continue
            d = run_dir(out, tid, s)
            log(f"trial {tid:>3} ({p['label']}) seed {s}: {cfg.network}  "
                f"lr {cfg.learning_rate:.2e}")
            t0 = time.perf_counter()
            res = runner(cfg, d, s, profile, score_seed, log)
            rec = {"schema_version": SCHEMA_VERSION, "trial": tid, "seed": s,
                   "score_seed": score_seed, "settings": p["settings"],
                   "network_str": str(cfg.network), "ppo": cfg.to_dict(), **res,
                   "wall_s": time.perf_counter() - t0}
            (d / RESULT_FILE).write_text(json.dumps(rec, indent=1) + "\n")
            h = res["heldout"]
            log(f"trial {tid:>3} seed {s}: held-out success {h['success_rate']:.1%}, "
                f"dist left {h['final_dist_m']:.2f} m")
    final = pts[-1]["trial"]
    best_seed = max(seed_list, key=lambda s: read_result(out, final, s)["heldout"]["success_rate"])
    video = recorder(run_dir(out, final, best_seed), out / VIDEO_DIR, cp.video_episodes,
                     score_seed, log)
    return {"points": pts, "seeds": seed_list, "video": video}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--campaign", required=True, help="folder written by tars_rl.tune.campaign")
    p.add_argument("--profile", default="smoke", choices=sorted(PROFILES))
    p.add_argument("--seeds", type=int, default=3, help="training seeds per point")
    p.add_argument("--points", type=int, default=5, help="points incl. start and final best")
    p.add_argument("--seed-base", type=int, help="first seed (default: campaign seed + 1)")
    p.add_argument("--score-seed", type=int, help="held-out flags (default: campaign's + 1)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    compare(a.campaign, a.profile, a.seeds, a.points, a.out, a.seed_base, a.score_seed)
    print(f"comparison done; inspect: python -m tars_rl.tune.eval_compare --compare {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
