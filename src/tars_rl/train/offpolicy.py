"""Train TARS to reach the flag with FastTD3 or FastSAC (T21), raced against PPO (`train.run`).

Run: python -m tars_rl.train.offpolicy --algo fasttd3 --profile smoke --seed 0 --out runs/smoke-td3

The 2025 off-policy recipe for thousands of GPU envs, all in JAX so physics, replay and updates
stay on the GPU: many parallel MJX envs feed a large on-device replay buffer, big batches, n-step
returns, a C51 distributional twin critic, observation normalization and the average of the two
Q-values (`OffPolicyConfig.q_agg`). Same arena and reward as PPO, read straight from FlagEnv:
success and falls end an episode (no bootstrap), timeouts truncate it (bootstrap from the last
obs), and each finished env resets with a new flag (A22).

Writes the same policy folder as PPO (A60): policy.pkl holds `(normalizer, actor)`, meta.json adds
"algo" and the "offpolicy" settings, progress.jsonl has "offpolicy" lines (losses) and "curve"
lines (our flag eval on fixed held-out flags, A61), rollout.npz is one run for T9's MP4. A run
stops at `total_env_steps` or `max_wall_s`; `--match-wall runs/full-seed0` gives it PPO's
wall-clock budget, and the curve points are spread evenly over that budget.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from brax.training.acme import running_statistics, specs

from tars_rl.env.config import EnvConfig
from tars_rl.env.eval import run_episodes
from tars_rl.env.flag_env import FlagEnv
from tars_rl.env.rollout import FlagRollout
from tars_rl.train import offpolicy_nets as nets
from tars_rl.train.config import ALGOS, OffPolicyConfig, get_offpolicy_profile
from tars_rl.train.policy import META_FILE, PROGRESS_FILE, ROLLOUT_FILE, as_policy, save_policy
from tars_rl.train.run import CURVE_SEED_OFFSET, _git_commit, _version, make_curve_eval


# ---- replay: a (time, env) ring on the device ----------------------------------------------------
class Replay(NamedTuple):
    obs: jax.Array  # (T, N, obs)
    action: jax.Array  # (T, N, act)
    reward: jax.Array  # (T, N)
    next_obs: jax.Array  # (T, N, obs): the true last obs, also when the env then reset
    term: jax.Array  # (T, N) success or fall: no bootstrap
    done: jax.Array  # (T, N) term or truncated: the n-step return stops here
    ptr: jax.Array  # next row to write
    size: jax.Array  # rows written, up to T


class Batch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    ret: jax.Array  # n-step discounted reward sum
    next_obs: jax.Array  # obs to bootstrap from
    discount: jax.Array  # gamma**k, 0 if the episode ended in success or a fall


def replay_init(rows: int, n_envs: int, obs_size: int, act_size: int) -> Replay:
    z = jnp.zeros
    return Replay(z((rows, n_envs, obs_size)), z((rows, n_envs, act_size)), z((rows, n_envs)),
                  z((rows, n_envs, obs_size)), z((rows, n_envs)), z((rows, n_envs)),
                  jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32))


def replay_add(buf: Replay, obs, action, reward, next_obs, term, done) -> Replay:
    i, rows = buf.ptr, buf.reward.shape[0]
    new = (obs, action, reward, next_obs, term.astype(jnp.float32), done.astype(jnp.float32))
    fields = [x.at[i].set(v) for x, v in zip(buf[:6], new, strict=True)]
    return Replay(*fields, (i + 1) % rows, jnp.minimum(buf.size + 1, rows))


def replay_sample(buf: Replay, rng: jax.Array, batch: int, n_step: int, gamma: float) -> Batch:
    """Uniform over (row, env) whose n rows are all written; the return stops at the first done."""
    rows, n_envs = buf.reward.shape
    k_t, k_e = jax.random.split(rng)
    off = jax.random.randint(k_t, (batch,), 0, jnp.maximum(buf.size - n_step + 1, 1))
    t0 = (buf.ptr - buf.size + off) % rows
    e = jax.random.randint(k_e, (batch,), 0, n_envs)
    ts = (t0[:, None] + jnp.arange(n_step)) % rows  # (B, n)
    r, d = buf.reward[ts, e[:, None]], buf.done[ts, e[:, None]]
    alive = jnp.cumprod(1.0 - d, axis=1)
    before = jnp.concatenate([jnp.ones((batch, 1)), alive[:, :-1]], axis=1)  # step k is used
    ret = (before * gamma ** jnp.arange(n_step) * r).sum(1)
    last = (before.sum(1) - 1).astype(jnp.int32)  # index of the last step used
    tl = ts[jnp.arange(batch), last]
    return Batch(buf.obs[t0, e], buf.action[t0, e], ret, buf.next_obs[tl, e],
                 gamma ** (last + 1.0) * (1.0 - buf.term[tl, e]))


# ---- C51 -----------------------------------------------------------------------------------------
def project(probs: jax.Array, target_z: jax.Array, v_min: float, v_max: float) -> jax.Array:
    """Project `probs` sitting on atoms `target_z` (both (B, atoms)) back onto the fixed support."""
    batch, atoms = probs.shape
    dz = (v_max - v_min) / (atoms - 1)
    b = (jnp.clip(target_z, v_min, v_max) - v_min) / dz
    lo, hi = jnp.floor(b), jnp.ceil(b)
    lo = jnp.where((hi > 0) & (lo == hi), lo - 1, lo)  # b on an atom: all mass to `hi`
    hi = jnp.where((lo < atoms - 1) & (lo == hi), hi + 1, hi)  # b == 0: all mass to `lo`
    rows = jnp.arange(batch)[:, None]
    out = jnp.zeros_like(probs).at[rows, lo.astype(jnp.int32)].add(probs * (hi - b))
    return out.at[rows, hi.astype(jnp.int32)].add(probs * (b - lo))


# ---- learner -------------------------------------------------------------------------------------
class Learner(NamedTuple):
    norm: Any
    actor: Any
    critic: Any  # two critics stacked on axis 0
    critic_target: Any
    actor_opt: Any
    critic_opt: Any
    log_alpha: jax.Array  # FastSAC temperature (unused by FastTD3)
    alpha_opt: Any
    updates: jax.Array


class Carry(NamedTuple):
    env: Any  # batched FlagEnv State
    noise_std: jax.Array  # (N,) FastTD3 per-env exploration std
    learner: Learner
    buf: Replay
    rng: jax.Array
    it: jax.Array


def make_trainer(env: FlagEnv, cfg: OffPolicyConfig):
    """`init(rng) -> Carry` and `chunk(Carry) -> (Carry, metrics)` (cfg.chunk iterations)."""
    c, n_env = cfg, cfg.num_envs
    obs_size, act_size = env.observation_size, env.action_size
    sac = c.algo == "fastsac"
    actor = nets.actor_module(c.algo, act_size, c.actor_hidden)
    critic = nets.critic_module(c.num_atoms, c.critic_hidden)
    support = jnp.linspace(c.v_min, c.v_max, c.num_atoms)
    actor_tx = optax.adamw(c.actor_lr, weight_decay=c.weight_decay)
    critic_tx = optax.adamw(c.critic_lr, weight_decay=c.weight_decay)
    alpha_tx = optax.adam(c.critic_lr)
    target_entropy = -c.target_entropy_per_dim * act_size
    log_std = (c.log_std_min, c.log_std_max)

    def norm_obs(norm, obs):
        return nets.normalize(norm, obs, c.normalize_observations)

    def critics(params, o, a):  # -> (2, B, atoms) logits
        return jax.vmap(critic.apply, in_axes=(0, None))(params, jnp.concatenate([o, a], -1))

    def q_of(logits):  # -> (B,) aggregated Q
        q = (jax.nn.softmax(logits) * support).sum(-1)
        return q.mean(0) if c.q_agg == "mean" else q.min(0)

    def sample_std(rng, shape):
        return jax.random.uniform(rng, shape, minval=c.std_min, maxval=c.std_max)

    def init(rng: jax.Array) -> Carry:
        k_env, k_std, k_a, k_c, rng = jax.random.split(rng, 5)
        o, a = jnp.zeros((1, obs_size)), jnp.zeros((1, act_size))
        actor_p = actor.init(k_a, o)
        critic_p = jax.vmap(critic.init, in_axes=(0, None))(
            jax.random.split(k_c, 2), jnp.concatenate([o, a], -1))
        log_alpha = jnp.log(jnp.asarray(c.alpha_init, jnp.float32))
        learner = Learner(
            norm=running_statistics.init_state(specs.Array((obs_size,), jnp.float32)),
            actor=actor_p, critic=critic_p, critic_target=critic_p,
            actor_opt=actor_tx.init(actor_p), critic_opt=critic_tx.init(critic_p),
            log_alpha=log_alpha, alpha_opt=alpha_tx.init(log_alpha),
            updates=jnp.zeros((), jnp.int32),
        )
        carry = Carry(
            env=jax.vmap(env.reset)(jax.random.split(k_env, n_env)),
            noise_std=sample_std(k_std, (n_env,)), learner=learner,
            buf=replay_init(c.buffer_per_env, n_env, obs_size, act_size), rng=rng,
            it=jnp.zeros((), jnp.int32),
        )
        return jax.tree_util.tree_map(jnp.copy, carry)  # no shared buffers: `chunk` donates them

    def act(L: Learner, obs, std, rng):
        out = actor.apply(L.actor, norm_obs(L.norm, obs))
        if sac:
            return nets.sac_sample(out, rng, *log_std)[0]
        noise = jax.random.normal(rng, out.shape) * std[:, None]
        return jnp.clip(jnp.tanh(out) + noise, -1.0, 1.0)

    def update(L: Learner, buf: Replay, rng) -> tuple[Learner, dict]:
        k_s, k_n, k_a = jax.random.split(rng, 3)
        b = replay_sample(buf, k_s, c.batch_size, c.n_step, c.gamma)
        o, no = norm_obs(L.norm, b.obs), norm_obs(L.norm, b.next_obs)
        alpha = jnp.exp(L.log_alpha)

        # critic target: next action from the current actor (FastTD3 keeps no actor target)
        nout = actor.apply(L.actor, no)
        if sac:
            na, nlogp = nets.sac_sample(nout, k_n, *log_std)
            ent = alpha * nlogp
        else:
            eps = jnp.clip(jax.random.normal(k_n, nout.shape) * c.target_noise,
                           -c.target_noise_clip, c.target_noise_clip)
            na, ent = jnp.clip(jnp.tanh(nout) + eps, -1.0, 1.0), jnp.zeros(c.batch_size)
        tprobs = jax.nn.softmax(critics(L.critic_target, no, na))  # (2, B, atoms)
        tz = b.ret[:, None] + b.discount[:, None] * (support[None] - ent[:, None])
        target = jax.vmap(project, in_axes=(0, None, None, None))(tprobs, tz, c.v_min, c.v_max)
        if c.q_agg == "min":  # clipped double-Q: both critics learn the lower one's target
            tq = (tprobs * support).sum(-1)
            target = jnp.broadcast_to(jnp.where((tq[0] < tq[1])[:, None], target[0], target[1]),
                                      target.shape)
        target = jax.lax.stop_gradient(target)

        def critic_loss(p):
            logp = jax.nn.log_softmax(critics(p, o, b.action))
            return -(target * logp).sum(-1).mean(-1).sum()

        closs, g = jax.value_and_grad(critic_loss)(L.critic)
        upd, copt = critic_tx.update(g, L.critic_opt, L.critic)
        cp = optax.apply_updates(L.critic, upd)
        ct = optax.incremental_update(cp, L.critic_target, c.tau)
        L = L._replace(critic=cp, critic_target=ct, critic_opt=copt, updates=L.updates + 1)

        def actor_step(L: Learner):
            def loss(p, log_alpha):
                out = actor.apply(p, o)
                if sac:
                    a, logp = nets.sac_sample(out, k_a, *log_std)
                else:
                    a, logp = jnp.tanh(out), jnp.zeros(c.batch_size)
                q = q_of(critics(L.critic, o, a))
                aloss = (jax.lax.stop_gradient(jnp.exp(log_alpha)) * logp - q).mean()
                tloss = -(log_alpha * jax.lax.stop_gradient(logp + target_entropy)).mean()
                return aloss + (tloss if sac else 0.0), (aloss, q.mean())

            (_, (aloss, q)), (ga, gl) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(
                L.actor, L.log_alpha)
            upd, aopt = actor_tx.update(ga, L.actor_opt, L.actor)
            L = L._replace(actor=optax.apply_updates(L.actor, upd), actor_opt=aopt)
            if sac:
                upd, lopt = alpha_tx.update(gl, L.alpha_opt)
                L = L._replace(log_alpha=optax.apply_updates(L.log_alpha, upd), alpha_opt=lopt)
            return L, aloss, q

        L, aloss, q = jax.lax.cond(
            (L.updates - 1) % c.policy_frequency == 0, actor_step,
            lambda L: (L, jnp.nan, jnp.nan), L)
        return L, {"critic_loss": closs, "actor_loss": aloss, "q": q,
                   "alpha": jnp.exp(L.log_alpha)}

    def updates(L: Learner, buf: Replay, rng) -> tuple[Learner, dict]:
        L, m = jax.lax.scan(lambda L, k: update(L, buf, k), L, jax.random.split(rng, c.num_updates))
        return L, jax.tree_util.tree_map(jnp.nanmean, m)

    no_update = {k: jnp.nan for k in ("critic_loss", "actor_loss", "q", "alpha")}

    def iteration(s: Carry, _) -> tuple[Carry, dict]:
        rng, k_act, k_rand, k_reset, k_std, k_up = jax.random.split(s.rng, 6)
        L, obs = s.learner, s.env.obs
        if c.normalize_observations:
            L = L._replace(norm=running_statistics.update(L.norm, obs))
        learning = s.it >= c.learning_starts
        a = jnp.where(learning, act(L, obs, s.noise_std, k_act),
                      jax.random.uniform(k_rand, (n_env, act_size), minval=-1, maxval=1))
        ns = jax.vmap(env.step)(s.env, a)
        term = ns.fell | ns.success
        done = term | ns.truncated | (ns.step >= c.episode_length)
        buf = replay_add(s.buf, obs, a, ns.reward, ns.obs, term, done)
        fresh = jax.vmap(env.reset)(jax.random.split(k_reset, n_env))

        def pick(new, old):
            return jnp.where(done.reshape(done.shape + (1,) * (new.ndim - 1)), new, old)

        es = jax.tree_util.tree_map(pick, fresh, ns)
        std = jnp.where(done, sample_std(k_std, (n_env,)), s.noise_std)
        L, m = jax.lax.cond(learning, lambda L: updates(L, buf, k_up),
                            lambda L: (L, no_update), L)
        m = {**m, "reward": ns.reward.mean(), "episodes": done.sum(), "success": ns.success.sum(),
             "fell": ns.fell.sum()}
        return Carry(es, std, L, buf, rng, s.it + 1), m

    def chunk(s: Carry):
        return jax.lax.scan(iteration, s, None, length=c.chunk)

    return init, jax.jit(chunk, donate_argnums=0)  # donated: the replay is updated in place


def train(algo: str, profile: str, seed: int, out: str | Path, cfg: OffPolicyConfig | None = None,
          env_config: EnvConfig | None = None, log=print) -> dict:
    cfg = cfg or get_offpolicy_profile(profile, algo)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / PROGRESS_FILE
    progress.write_text("")

    env = FlagEnv(config=env_config)
    inference = nets.make_inference(algo, env.action_size, cfg.actor_hidden,
                                    cfg.normalize_observations, (cfg.log_std_min, cfg.log_std_max))
    curve_eval = make_curve_eval(env, inference, cfg.curve_episodes,
                                 jax.random.PRNGKey(seed + CURVE_SEED_OFFSET))
    init, chunk = make_trainer(env, cfg)

    def write(line: dict) -> None:
        with progress.open("a") as f:
            f.write(json.dumps(line) + "\n")

    t0 = time.perf_counter()
    s = init(jax.random.PRNGKey(seed))

    def curve(step: int) -> None:
        r = curve_eval((s.learner.norm, s.learner.actor))
        write({"kind": "curve", "step": step, "wall_s": time.perf_counter() - t0, **r})
        log(f"step {step:>12,}  success {r['success_rate']:.1%}  "
            f"dist left {r['final_dist_m']:.2f} m  falls {r['fall_rate']:.1%}")

    curve(0)
    steps, next_eval, totals = 0, 1, {}
    while True:
        s, m = chunk(s)
        steps += cfg.chunk * cfg.num_envs
        wall = time.perf_counter() - t0
        for k in ("episodes", "success", "fell"):
            totals[k] = totals.get(k, 0) + int(m[k].sum())
        frac = steps / cfg.total_env_steps
        if cfg.max_wall_s:
            frac = max(frac, wall / cfg.max_wall_s)
        if frac >= next_eval / cfg.num_evals or frac >= 1:
            mm = {k: float(jnp.nanmean(v)) for k, v in m.items() if k not in totals}
            clean = {k: None if math.isnan(v) else v for k, v in mm.items()}  # NaN: no update yet
            write({"kind": "offpolicy", "step": steps, "wall_s": wall, **clean, **totals})
            log(f"step {steps:>12,}  critic {mm['critic_loss']:.3f}  q {mm['q']:+.3f}  "
                f"episodes {totals['episodes']:,}")
            curve(steps)
            next_eval = math.floor(frac * cfg.num_evals) + 1
        if frac >= 1:
            break
    wall = time.perf_counter() - t0

    meta = {
        "profile": profile, "seed": seed, "algo": algo, "offpolicy": cfg.to_dict(),
        "env": env.config.to_dict(), "body_spec": env.spec.name,
        "body_spec_version": env.spec.version, "observation_size": env.observation_size,
        "action_size": env.action_size, "num_timesteps": steps, "train_wall_s": wall,
        "git_commit": _git_commit(),
        "versions": {p: _version(p) for p in ("jax", "mujoco", "mujoco-mjx", "flax", "optax")},
    }
    params = (s.learner.norm, s.learner.actor)
    save_policy(out, params, meta)

    r = run_episodes(env, as_policy(inference, params), jax.random.PRNGKey(seed), 1)
    FlagRollout(r["qpos_ep0"], env.dt, env.spec.version, r["flag_pos"][0]).save(out / ROLLOUT_FILE)
    log(f"saved policy, curve and rollout to {out} ({wall:.0f} s training, {steps:,} env steps)")
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--algo", required=True, choices=ALGOS)
    p.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--timesteps", type=int, help="override the profile's total_env_steps")
    p.add_argument("--wall-s", type=float, help="stop after this many seconds of training")
    p.add_argument("--match-wall", help="policy folder whose train_wall_s is this run's budget")
    a = p.parse_args(argv)
    over: dict = {"total_env_steps": a.timesteps} if a.timesteps else {}
    if a.match_wall:
        ppo_meta = json.loads((Path(a.match_wall) / META_FILE).read_text())
        over["max_wall_s"] = ppo_meta["train_wall_s"]
    if a.wall_s:
        over["max_wall_s"] = a.wall_s
    train(a.algo, a.profile, a.seed, a.out, get_offpolicy_profile(a.profile, a.algo, **over))
    return 0


if __name__ == "__main__":
    sys.exit(main())
