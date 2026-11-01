"""Train a few fixed network descriptions and print size, training speed and score side by side.

Run: python -m tars_rl.train.eval_networks --profile smoke --seed 0

Each description trains through the normal PPO command (`tars_rl.train.run.train`) with the same
profile and seed, then is scored by `tars_rl.train.eval` on the same held-out flags. The first row
is the profile's default network (T15's), the baseline the others are read against; random moves
and standing still are printed below as the floor. Smoke runs only prove each shape builds and
trains; their scores say nothing about which shape is better (that needs the tuner, T19).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from tars_rl.train.config import get_profile
from tars_rl.train.eval import evaluate
from tars_rl.train.networks import NetSpec, NetworkSpec, tower_sizes
from tars_rl.train.policy import META_FILE
from tars_rl.train.run import train


def candidates(default: NetworkSpec) -> dict[str, NetworkSpec]:
    """Default plus one of each block kind, at the default's widths."""
    pw, vw = default.policy.width, default.value.width

    def both(**kw) -> NetworkSpec:
        return NetworkSpec(NetSpec(width=pw, **kw), NetSpec(width=vw, **kw))

    return {
        "default": default,
        "mlp+layernorm": NetworkSpec(replace(default.policy, norm="layer"),
                                     replace(default.value, norm="layer")),
        "simba": both(block="simba", depth=2, activation="relu", norm="layer", expansion=4),
        "bro": both(block="bro", depth=2, activation="relu", norm="layer"),
        "simba-rms-gelu": both(block="simba", depth=1, activation="gelu", norm="rms", expansion=2),
    }


def run(profile: str, seed: int, out: Path, episodes: int, log=print) -> dict:
    base = get_profile(profile)
    res: dict = {"profile": profile, "seed": seed, "episodes": episodes, "networks": {}}
    for name, spec in candidates(base.network).items():
        log(f"--- {name}: {spec}")
        cfg = replace(base, network=spec).check()
        folder = out / name.replace("+", "_")
        meta = train(profile, seed, folder, cfg, log=lambda *_: None)
        ev = evaluate(folder, episodes, seed, baselines="baselines" not in res)  # same flags
        recorded = json.loads((folder / META_FILE).read_text())["network_str"]
        t = ev["policies"]["trained"]
        res["networks"][name] = {
            "network": str(spec), "recorded": recorded,
            "params": tower_sizes(spec, meta["observation_size"], meta["action_size"]),
            "train_wall_s": meta["train_wall_s"],
            "steps_per_s": cfg.num_timesteps / meta["train_wall_s"],
            "success_rate": t["success_rate"], "fall_rate": t["fall_rate"],
            "final_dist_m": t["final_dist_m"], "closed_frac": t["closed_frac"],
        }
        if "baselines" not in res:
            res["baselines"] = {b: ev["policies"][b] for b in ("random", "still")}
    return res


def report(res: dict) -> None:
    print(f"\nNetworks on profile {res['profile']}, seed {res['seed']}, "
          f"{res['episodes']} held-out flags (wall time includes JIT compile)")
    hdr = (f"{'name':<16}{'policy params':>14}{'value params':>13}{'train s':>9}{'steps/s':>9}"
           f"{'success':>9}{'falls':>7}{'dist (m)':>10}{'closed':>8}")
    print(hdr)
    for name, r in res["networks"].items():
        print(f"{name:<16}{r['params']['policy']:>14,}{r['params']['value']:>13,}"
              f"{r['train_wall_s']:>9.0f}{r['steps_per_s']:>9.0f}{r['success_rate']:>9.1%}"
              f"{r['fall_rate']:>7.1%}{r['final_dist_m']:>10.3f}{r['closed_frac']:>8.1%}")
    for name, b in res["baselines"].items():
        print(f"{name:<16}{'':>14}{'':>13}{'':>9}{'':>9}{b['success_rate']:>9.1%}"
              f"{b['fall_rate']:>7.1%}{b['final_dist_m']:>10.3f}{b['closed_frac']:>8.1%}")
    print("\nDescriptions (as recorded in each policy folder's meta.json):")
    for name, r in res["networks"].items():
        print(f"  {name:<16}{r['recorded']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--out", help="folder for the policy folders (default runs/networks-<profile>)")
    a = p.parse_args(argv)
    out = Path(a.out or f"runs/networks-{a.profile}")
    res = run(a.profile, a.seed, out, a.episodes)
    report(res)
    out.mkdir(parents=True, exist_ok=True)
    (out / "networks.json").write_text(json.dumps(res, indent=2) + "\n")
    bad = [n for n, r in res["networks"].items() if r["recorded"] != r["network"]]
    if bad:
        print(f"FAIL: saved description differs from the one trained for {bad}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
