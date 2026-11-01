"""A trained policy as one folder that also records what it was trained with (A60).

    <out>/policy.pkl     brax params (normalizer, policy) as numpy arrays
    <out>/meta.json      version, profile + PPO settings, the network description it was built
                         from ("network" as fields, "network_str" in short form, T18), EnvConfig,
                         BodySpec name/version, obs/action sizes, seed, steps trained, wall time
    <out>/progress.jsonl learning curve, one line per eval point (A61)
    <out>/rollout.npz    one recorded run of the trained policy (FlagRollout, A51)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from tars_rl.train.networks import NetSpec, NetworkSpec, network_factory

POLICY_VERSION = 2  # 2: network description in meta (T18); 1: T15's hidden-size tuples, still read
PARAMS_FILE = "policy.pkl"
META_FILE = "meta.json"
PROGRESS_FILE = "progress.jsonl"
ROLLOUT_FILE = "rollout.npz"

Policy = Callable[[jax.Array, jax.Array], jax.Array]  # (rng, obs batch) -> action batch, as T13


def make_inference(obs_size: int, act_size: int, spec: NetworkSpec, normalize=True):
    """The same networks PPO trained, as `params -> (obs, rng) -> (action, extras)`."""
    pre = running_statistics.normalize if normalize else (lambda x, y: x)
    nets = network_factory(spec)(obs_size, act_size, preprocess_observations_fn=pre)
    return ppo_networks.make_inference_fn(nets)


def as_policy(make_inference_fn, params, deterministic: bool = True) -> Policy:
    act = make_inference_fn(tuple(params[:2]), deterministic=deterministic)
    return lambda rng, obs: act(obs, rng)[0]


def network_from_meta(meta: dict) -> NetworkSpec:
    """The network description a policy folder was built from (v1 folders: from hidden sizes)."""
    if "network" in meta:
        return NetworkSpec.from_dict(meta["network"])
    ppo = meta["ppo"]
    if "network" in ppo:
        return NetworkSpec.from_dict(ppo["network"])
    return NetworkSpec(NetSpec.from_hidden(ppo["policy_hidden"]),
                       NetSpec.from_hidden(ppo["value_hidden"]))


@dataclass(frozen=True)
class SavedPolicy:
    path: Path
    meta: dict
    params: tuple

    @property
    def network(self) -> NetworkSpec:
        return network_from_meta(self.meta)

    def policy(self, deterministic: bool = True) -> Policy:
        m = self.meta
        inf = make_inference(m["observation_size"], m["action_size"], self.network,
                             m["ppo"]["normalize_observations"])
        return as_policy(inf, self.params, deterministic)


def save_policy(out: str | Path, params, meta: dict) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    host = jax.tree_util.tree_map(np.asarray, jax.device_get(tuple(params[:2])))
    brax_model.save_params(str(out / PARAMS_FILE), host)
    spec = network_from_meta(meta)  # always recorded at the top level, in both forms
    meta = {**meta, "network": spec.to_dict(), "network_str": str(spec)}
    (out / META_FILE).write_text(json.dumps({"version": POLICY_VERSION, **meta}, indent=2) + "\n")
    return out


def load_policy(path: str | Path) -> SavedPolicy:
    path = Path(path)
    meta = json.loads((path / META_FILE).read_text())
    if meta.get("version") not in (1, POLICY_VERSION):
        raise ValueError(f"policy folder version {meta.get('version')} not in 1..{POLICY_VERSION}")
    params = tuple(brax_model.load_params(str(path / PARAMS_FILE)))
    return SavedPolicy(path=path, meta=meta, params=params)


def read_progress(path: str | Path) -> list[dict]:
    f = Path(path) / PROGRESS_FILE
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
