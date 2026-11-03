"""Verify a trained policy in Unreal: the same flag episodes in MuJoCo and Unreal, side by side.

Run (Unreal started with scripts/unreal.sh, see VERIFY.md):
    python -m tars_rl.unreal.verify --policy runs/chain-realistic/gru --level flat
    python -m tars_rl.unreal.verify --policy ... --level apartment --engines unreal --realtime
    python -m tars_rl.unreal.verify --policy ... --ports 5577 5578 5579 5580   # 4 Unreals

One episode loop drives either engine (`tars_rl.unreal.engines`) with FlagEnv's rules: the
policy's observation window, the servo speed cap, chain flags drawn by FlagEnv's own sampler
from the same keys as `tars_rl.train.eval`, tags, falls and the reward. On the flat level the
mjx column is the training physics, so it should match `train.eval`'s numbers; mujoco (CPU)
and unreal next to it are the sim-to-sim gaps (roadmap stage 4). Other levels add the boxes
of `tars_rl.env.levels` to both engines; flags there are redrawn until the straight walk to
them is clear of furniture (the blind policy leaves routing to the decider).

--realtime paces every step to the wall clock so you can watch; without it Unreal runs as fast
as it can step (lockstep, no frame waits; add -nullrhi for no rendering at all).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tars_rl.env.config import EnvConfig, RewardConfig
from tars_rl.env.eval import summarize
from tars_rl.env.flag_env import FlagEnv
from tars_rl.env.levels import LEVELS, Level, make_level
from tars_rl.train.eval import EVAL_SEED_OFFSET
from tars_rl.train.policy import load_policy
from tars_rl.unreal.bodystate import OBS_SETS_SUPPORTED, BodyState, observe_default
from tars_rl.unreal.bridge import DEFAULT_PORT
from tars_rl.unreal.engines import Engine, MjxEngine, MujocoEngine, UnrealEngine

FLAG_TRIES = 64  # redraws before a level gives up on finding a clear flag
METRICS = ("success_rate", "fall_rate", "tags_per_ep", "first_tag_s", "final_dist_m",
           "closed_frac", "length_s", "return")


@dataclass
class Episode:
    tags: int
    fell: bool
    truncated: bool
    length: int
    first_tag: int
    start_dist: float
    final_dist: float
    ret: float
    forced_flags: int  # chain flags kept although no clear one was found
    wall_s: float


class Rules:
    """FlagEnv's episode rules, in numpy, for one policy and level (shared by every worker)."""

    def __init__(self, env: FlagEnv, level: Level) -> None:
        self.env, self.level = env, level
        c = env.config
        if c.obs not in OBS_SETS_SUPPORTED:
            raise ValueError(f"observation set {c.obs!r}: the Unreal bridge provides "
                             f"{OBS_SETS_SUPPORTED} only")
        self.mid, self.half = np.asarray(env._ctrl_mid), np.asarray(env._ctrl_half)
        self.max_step = np.asarray(env._ctrl_max_step)
        self.cos_fall = float(np.cos(c.fall_tilt_rad))
        self.tag_m = c.success_heights * env.height
        self._sample = jax.jit(env.sample_flag)
        self._split = jax.jit(jax.random.split)

    def ctrl(self, action: np.ndarray, prev: np.ndarray) -> np.ndarray:
        target = self.mid + self.half * np.clip(action, -1.0, 1.0)
        return prev + np.clip(target - prev, -self.max_step, self.max_step)

    def flag(self, key: jax.Array, centre_xy) -> tuple[np.ndarray, bool]:
        """FlagEnv's draw; on a level, redrawn (from keys folded off the original) until the
        straight walk to it is clear. The first draw is FlagEnv's own, so flat matches it."""
        f = None
        for i in range(FLAG_TRIES):
            k = key if i == 0 else jax.random.fold_in(key, i)
            f = np.array(self._sample(k, jnp.asarray(centre_xy, jnp.float32)), float)  # a copy
            if self.level.flag_ok(centre_xy, f[:2]):
                f[2] = self.level.ground_z(f[:2])
                return f, True
        f[2] = self.level.ground_z(f[:2])
        return f, False

    def frame(self, s: BodyState, flag: np.ndarray) -> np.ndarray:
        return observe_default(self.env.spec, s, flag)

    def window(self, prev: np.ndarray | None, frame: np.ndarray) -> np.ndarray:
        k = self.env.config.history_steps
        if k == 0:
            return frame
        if prev is None:
            return np.tile(frame, k)
        return np.concatenate([prev[frame.size:], frame])


def run_episode(rules: Rules, engine: Engine, act, key: jax.Array, label: str,
                realtime: bool) -> Episode:
    env, c, w = rules.env, rules.env.config, rules.env.config.reward
    spawn = np.asarray(rules.level.spawn_xy, float)
    # Same keys as FlagEnv.reset: the flag from `key`, the chain's next flags from fold_in(key, 2).
    flag, _ = rules.flag(key, spawn)
    flag_rng = jax.random.fold_in(key, 2)
    s = engine.reset(spawn, flag)
    obs = rules.window(None, rules.frame(s, flag))
    ctrl = np.zeros(env.action_size)
    dist = float(np.linalg.norm(s.central_pos[:2] - flag[:2]))
    start_dist, ret, tags, first_tag, forced = dist, 0.0, 0, -1, 0
    fell = False
    t0 = time.perf_counter()
    step = 0
    while step < c.max_steps:
        action = np.asarray(act(obs), float)
        ctrl = rules.ctrl(action, ctrl)
        hud = f"{label}  t {step * env.dt:4.1f} s  flags {tags}  dist {dist:.2f} m"
        s = engine.step(ctrl, hud)
        step += 1
        new_dist = float(np.linalg.norm(s.central_pos[:2] - flag[:2]))
        cos_tilt = s.cos_tilt()
        fell = cos_tilt < rules.cos_fall
        success = new_dist < rules.tag_m and not fell
        ret += (w.progress * (dist - new_dist) + w.upright * cos_tilt - w.fall * fell
                - w.effort * float(np.mean(np.square(np.clip(action, -1, 1))))
                + w.tag * success)
        flag_rng, k_flag = rules._split(flag_rng)  # FlagEnv splits every step
        dist = new_dist
        if success:
            tags += 1
            first_tag = step if first_tag < 0 else first_tag
            if c.flags == "single":
                break
            flag, ok = rules.flag(k_flag, s.central_pos[:2])
            forced += not ok
            engine.set_flag(flag)
            dist = start_dist = float(np.linalg.norm(s.central_pos[:2] - flag[:2]))
        if fell:
            break
        obs = rules.window(obs, rules.frame(s, flag))
        if realtime:
            lag = t0 + step * env.dt - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
    return Episode(tags, fell, (step >= c.max_steps) and not fell, step, first_tag, start_dist,
                   dist, ret, forced, time.perf_counter() - t0)


def load(policy_dir: str | Path, flags: str = "chain"):
    saved = load_policy(policy_dir)
    m = saved.meta
    env = FlagEnv(config=EnvConfig(**{**m["env"], "reward": RewardConfig(**m["env"]["reward"]),
                                      "flags": flags}))
    if env.spec.version != m["body_spec_version"]:
        raise ValueError(f"policy trained on BodySpec v{m['body_spec_version']}, "
                         f"env has v{env.spec.version}")
    policy = jax.jit(saved.policy())
    key = jax.random.PRNGKey(0)  # deterministic policy: the key is unused
    return env, (lambda obs: policy(key, jnp.asarray(obs)[None])[0])


def episode_keys(seed: int, n: int) -> list[jax.Array]:
    # train.eval: k_reset, k_pol = split(PRNGKey(seed + offset)); reset keys = split(k_reset, n)
    k_reset, _ = jax.random.split(jax.random.PRNGKey(seed + EVAL_SEED_OFFSET))
    return list(jax.random.split(k_reset, n))


def run_engine(rules: Rules, make_engine, workers: int, act, keys, realtime: bool,
               name: str) -> list[Episode]:
    """Episodes spread over `workers` engines (threads); results in episode order."""
    out: list[Episode | None] = [None] * len(keys)
    lock = threading.Lock()
    todo = list(range(len(keys)))

    def worker(w: int) -> None:
        engine = make_engine(w)
        engine.load(rules.level)
        try:
            while True:
                with lock:
                    if not todo:
                        return
                    i = todo.pop(0)
                label = f"{name} {rules.level.name}  ep {i + 1}/{len(keys)}"
                out[i] = run_episode(rules, engine, act, keys[i], label, realtime)
                e = out[i]
                print(f"  {name:<7} ep {i + 1:>3}: flags {e.tags}  "
                      f"{'FELL' if e.fell else 'ok  '}  {e.length * rules.env.dt:5.1f} s  "
                      f"({e.wall_s:.1f} s wall)", flush=True)
        finally:
            engine.close()

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(worker, range(workers)))
    return [e for e in out if e is not None]


def summarize_episodes(eps: list[Episode], env: FlagEnv) -> dict:
    r = {
        "success": np.array([e.tags > 0 for e in eps]), "fell": np.array([e.fell for e in eps]),
        "truncated": np.array([e.truncated for e in eps]),
        "start_dist": np.array([e.start_dist for e in eps]),
        "final_dist": np.array([e.final_dist for e in eps]),
        "length": np.array([e.length for e in eps]), "return": np.array([e.ret for e in eps]),
        "tags": np.array([e.tags for e in eps]), "first_tag": np.array([e.first_tag for e in eps]),
    }
    s = summarize(r, env)
    s["forced_flags"] = int(sum(e.forced_flags for e in eps))
    s["wall_s"] = float(sum(e.wall_s for e in eps))
    s["steps_per_s"] = float(r["length"].sum() / max(s["wall_s"], 1e-9))
    return s


def report(res: dict) -> None:
    names = list(res["engines"])
    print(f"\nPolicy {res['policy']}  level {res['level']}  {res['episodes']} episodes, "
          f"seed {res['seed']}, flags {res['flags']}")
    gaps = [f"{n}-{names[0]}" for n in names[1:]]
    print(f"{'':<16}" + "".join(f"{n:>12}" for n in names) + "".join(f"{g:>16}" for g in gaps))
    for k in (*METRICS, "forced_flags", "steps_per_s"):
        vals = [res["engines"][n][k] for n in names]
        pct = k.endswith("_rate") or k == "closed_frac"
        fmt = "{:>%d.1%%}" if pct else "{:>%d.2f}"
        line = f"{k:<16}" + "".join((fmt % 12).format(v) for v in vals)
        if k not in ("forced_flags", "steps_per_s"):
            line += "".join((fmt % 16).format(v - vals[0]) for v in vals[1:])
        print(line)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", required=True)
    p.add_argument("--level", default="flat", choices=sorted(LEVELS))
    p.add_argument("--level-seed", type=int, default=0)
    p.add_argument("--engines", nargs="+", default=["mjx", "mujoco", "unreal"],
                   choices=["mjx", "mujoco", "unreal"],
                   help="mjx = the training physics; gaps print against the first engine")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flags", default="chain", choices=["chain", "single"])
    p.add_argument("--realtime", action="store_true", help="pace steps to the wall clock")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--ports", type=int, nargs="+", default=[DEFAULT_PORT],
                   help="one Unreal instance per port; episodes are spread over them")
    p.add_argument("--json", type=Path, help="also write the results here")
    a = p.parse_args(argv)

    env, act = load(a.policy, a.flags)
    level = make_level(a.level, a.level_seed)
    rules = Rules(env, level)
    keys = episode_keys(a.seed, a.episodes)
    res: dict = {"policy": str(a.policy), "level": level.name, "level_seed": a.level_seed,
                 "episodes": a.episodes, "seed": a.seed, "flags": a.flags,
                 "realtime": a.realtime, "engines": {}}
    for name in a.engines:
        if name in ("mjx", "mujoco"):
            cls = MjxEngine if name == "mjx" else MujocoEngine
            make, workers = (lambda w, cls=cls: cls(env.spec, env.dt)), 1
        else:
            make, workers = (lambda w: UnrealEngine(env.spec, env.dt, a.host, a.ports[w])), len(
                a.ports)
        print(f"{name}: {a.episodes} episodes on {level.name} ({workers} worker(s))")
        eps = run_engine(rules, make, workers, act, keys, a.realtime and name == "unreal", name)
        res["engines"][name] = summarize_episodes(eps, env)
    report(res)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
