"""A trained policy as one folder that also records what it was trained with (A60).

    <out>/policy.pkl     brax params (normalizer, policy) as numpy arrays
    <out>/meta.json      version, profile + PPO settings, EnvConfig, BodySpec name/version,
                         obs/action sizes, seed, steps trained, wall time
    <out>/progress.jsonl learning curve, one line per eval point (A61)
    <out>/rollout.npz    one recorded run of the trained policy (FlagRollout, A51)
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

POLICY_VERSION = 1
PARAMS_FILE = "policy.pkl"
META_FILE = "meta.json"
PROGRESS_FILE = "progress.jsonl"
ROLLOUT_FILE = "rollout.npz"

Policy = Callable[[jax.Array, jax.Array], jax.Array]  # (rng, obs batch) -> action batch, as T13


def network_factory(policy_hidden: tuple[int, ...], value_hidden: tuple[int, ...]):
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(policy_hidden),
        value_hidden_layer_sizes=tuple(value_hidden),
    )


def make_inference(obs_size: int, act_size: int, policy_hidden, value_hidden, normalize=True):
    """The same networks PPO trained, as `params -> (obs, rng) -> (action, extras)`."""
    pre = running_statistics.normalize if normalize else (lambda x, y: x)
    nets = network_factory(policy_hidden, value_hidden)(
        obs_size, act_size, preprocess_observations_fn=pre
    )
    return ppo_networks.make_inference_fn(nets)


def as_policy(make_inference_fn, params, deterministic: bool = True) -> Policy:
    act = make_inference_fn(tuple(params[:2]), deterministic=deterministic)
    return lambda rng, obs: act(obs, rng)[0]


@dataclass(frozen=True)
class SavedPolicy:
    path: Path
    meta: dict
    params: tuple

    def policy(self, deterministic: bool = True) -> Policy:
        m = self.meta
        inf = make_inference(
            m["observation_size"], m["action_size"], m["ppo"]["policy_hidden"],
            m["ppo"]["value_hidden"], m["ppo"]["normalize_observations"],
        )
        return as_policy(inf, self.params, deterministic)


def save_policy(out: str | Path, params, meta: dict) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    host = jax.tree_util.tree_map(np.asarray, jax.device_get(tuple(params[:2])))
    brax_model.save_params(str(out / PARAMS_FILE), host)
    (out / META_FILE).write_text(json.dumps({"version": POLICY_VERSION, **meta}, indent=2) + "\n")
    return out


def load_policy(path: str | Path) -> SavedPolicy:
    path = Path(path)
    meta = json.loads((path / META_FILE).read_text())
    if meta.get("version") != POLICY_VERSION:
        raise ValueError(f"policy folder version {meta.get('version')} != {POLICY_VERSION}")
    params = tuple(brax_model.load_params(str(path / PARAMS_FILE)))
    return SavedPolicy(path=path, meta=meta, params=params)


def read_progress(path: str | Path) -> list[dict]:
    f = Path(path) / PROGRESS_FILE
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
