"""Tuning campaign: the self-guiding search (T17) drives real TARS training runs (T15, T18).

Run: python -m tars_rl.tune.campaign --profile smoke --trials 6 --seed 0 --out runs/tune_smoke

The box starts centred on the published PPO settings for walking robots (`tars_space.START`).
Each round the search proposes a batch of settings and network descriptions; each trial trains
TARS for a medium-length run (A83, `TuneProfile.num_timesteps`) with the campaign seed, is
scored on held-out flags by the blended score (`tune.score`), and is saved as one record (A80).
Every trial shares the training seed and the scoring flags; a run that errors scores -1 (A108).
Trial 0 is the box centre, i.e. the published settings: the baseline every later trial must beat.

Writes, in --out:
  campaign.json            what the campaign runs with (profile, seeds, space, start, weights)
  search.json              the search's full state (T17); saved after every ask and tell
  trials/NNNN/trial.json   the trial record: settings, network description, PPO config, score and
                           its parts, held-out results, learning curve, search round and box
  trials/NNNN/             the trial's policy folder (A60: policy.pkl, meta.json, progress.jsonl)

Stopped halfway, the same command resumes: finished trials are read back from their records,
never retrained, and the search continues from search.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from tars_rl.train.config import PPOConfig, get_profile
from tars_rl.tune.score import WEIGHTS, blend
from tars_rl.tune.search import BoxSearch, SearchConfig, Trial
from tars_rl.tune.tars_space import SPACE, START, to_config

SCHEMA_VERSION = 1
SEARCH_FILE = "search.json"
CAMPAIGN_FILE = "campaign.json"
TRIAL_FILE = "trial.json"


@dataclass(frozen=True)
class TuneProfile:
    base: str  # train profile giving env counts, batches and episode length
    num_timesteps: int  # one trial's training length (A83)
    num_evals: int  # learning-curve points per trial
    score_episodes: int  # held-out flags each trial is scored on
    search: SearchConfig


PROFILES: dict[str, TuneProfile] = {
    # Proves the loop end to end on the Mac: two rounds of three tiny runs. 5,120 steps covers
    # one training step at the longest unroll (32 x 40 x 4).
    "smoke": TuneProfile("smoke", 5_120, 2, 16, SearchConfig(batch=3, init_batch=3)),
    # The 5080 campaign (A107): a quarter of T15's 200M-step full run per trial.
    "full": TuneProfile("full", 50_000_000, 10, 256, SearchConfig(batch=4, init_batch=8)),
}

# (cfg, out_dir, seed, episodes, score_seed, log) -> (held-out summary, learning curve)
RunTrial = Callable[..., tuple[dict, list[dict]]]


def train_and_score(cfg: PPOConfig, out_dir: Path, seed: int, episodes: int, score_seed: int,
                    log=print, profile: str = "tune") -> tuple[dict, list[dict]]:
    """Train with T15's command, then score on held-out flags with T15's eval (same flags for
    every trial: they depend only on score_seed)."""
    from tars_rl.train.eval import evaluate
    from tars_rl.train.run import train

    train(profile, seed, out_dir, cfg, log=log)
    res = evaluate(out_dir, episodes, score_seed, baselines=False)
    return res["policies"]["trained"], res["curve"]


def trial_dir(out: Path, trial_id: int) -> Path:
    return out / "trials" / f"{trial_id:04d}"


def read_record(out: Path, trial_id: int) -> dict | None:
    p = trial_dir(out, trial_id) / TRIAL_FILE
    return json.loads(p.read_text()) if p.exists() else None


def read_records(out: str | Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(out).glob(f"trials/*/{TRIAL_FILE}"))]


def _manifest(profile: str, seed: int, score_seed: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "profile": profile, "seed": seed,
        "score_seed": score_seed, "tune_profile": asdict(PROFILES[profile]),
        "space": SPACE.names, "start": START, "weights": WEIGHTS,
    }


def _check_manifest(out: Path, want: dict) -> None:
    path = out / CAMPAIGN_FILE
    if not path.exists():
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(want, indent=1) + "\n")
        return
    have = json.loads(path.read_text())
    diff = [k for k in want if have.get(k) != json.loads(json.dumps(want[k]))]
    if diff:
        raise ValueError(f"{out} holds a campaign with different {diff}; use another --out")


def run_trial(t: Trial, search: BoxSearch, out: Path, profile: str, seed: int, score_seed: int,
              runner: RunTrial, log=print) -> float:
    """Train and score one trial and write its record, or read the record if it exists."""
    rec = read_record(out, t.id)
    if rec is not None and rec.get("score") is not None:
        log(f"trial {t.id:>3}: already done (score {rec['score']:+.3f}), not retrained")
        return rec["score"]
    tp = PROFILES[profile]
    base = get_profile(tp.base, num_timesteps=tp.num_timesteps, num_evals=tp.num_evals)
    d = trial_dir(out, t.id)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"schema_version": SCHEMA_VERSION, "id": t.id, "round": t.round, "u": t.u,
           "settings": t.settings, "box": asdict(search.boxes[t.round]), "seed": seed,
           "score_seed": score_seed, "profile": profile, "error": None}
    t0 = time.perf_counter()
    try:
        cfg = to_config(t.settings, base)
        rec |= {"network": cfg.network.to_dict(), "network_str": str(cfg.network),
                "ppo": cfg.to_dict()}
        log(f"trial {t.id:>3} (round {t.round}): {cfg.network}  lr {cfg.learning_rate:.2e}  "
            f"entropy {cfg.entropy_cost:.2e}  gamma {cfg.discounting:.3f}")
        heldout, curve = runner(cfg, d, seed, tp.score_episodes, score_seed, log)
        score = blend(heldout["success_rate"], heldout["closed_frac"], curve)
        rec |= {"heldout": heldout, "curve": curve}
    except Exception as e:  # a diverged or broken run is a bad trial, not a dead campaign
        score = blend(float("nan"), float("nan"), [])
        rec["error"] = f"{type(e).__name__}: {e}"
        log(f"trial {t.id:>3}: FAILED ({rec['error']})")
    rec |= {"score": score.total, "parts": score.to_dict(), "wall_s": time.perf_counter() - t0}
    (d / TRIAL_FILE).write_text(json.dumps(rec, indent=1) + "\n")
    w = WEIGHTS
    log(f"trial {t.id:>3}: score {score.total:+.3f} = {w['success']} x success "
        f"{score.success:.1%} + {w['closed']} x closed {score.closed:+.1%} + {w['speed']} x "
        f"speed {score.speed:.3f}  ({rec['wall_s']:.0f} s)")
    return score.total


def campaign(profile: str, trials: int, seed: int, out: str | Path, score_seed: int = 0,
             runner: RunTrial | None = None, log=print) -> BoxSearch:
    """Run (or resume) a campaign until `trials` trials are scored."""
    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
    out = Path(out)
    _check_manifest(out, _manifest(profile, seed, score_seed))
    if runner is None:
        def runner(*a, **kw):
            return train_and_score(*a, **kw, profile=f"tune-{profile}")
    sp = out / SEARCH_FILE
    if sp.exists():
        search = BoxSearch.load(sp, SPACE)
        log(f"resuming {out}: {len(search.scored())} trials scored, round {search.round}")
    else:
        search = BoxSearch(SPACE, PROFILES[profile].search, START, goal="max", seed=seed)
    while len(search.scored()) < trials:
        batch = [t for t in search.trials if t.score is None]  # asked before a stop
        if not batch:
            batch = search.ask()
            extra = len(search.trials) - trials
            if extra > 0:  # last batch cut to the budget
                del search.trials[-extra:]
                batch = batch[:-extra]
            search.save(sp)
        scores = [run_trial(t, search, out, profile, seed, score_seed, runner, log)
                  for t in batch]
        box = search.tell(batch, scores)
        search.save(sp)
        best = search.best()
        log(f"round {box.round - 1} done: best trial {best.id} score {best.score:+.3f}; "
            f"box {box.event}, side {box.length:.3f}")
    return search


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", default="smoke", choices=sorted(PROFILES))
    p.add_argument("--trials", type=int, required=True, help="total trials (resumes count too)")
    p.add_argument("--seed", type=int, default=0, help="training and search seed")
    p.add_argument("--score-seed", type=int, default=0, help="held-out scoring flags")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    s = campaign(a.profile, a.trials, a.seed, a.out, a.score_seed)
    best = s.best()
    print(f"campaign done: {len(s.scored())} trials, best trial {best.id} "
          f"score {best.score:+.3f}; inspect: python -m tars_rl.tune.eval_campaign "
          f"--campaign {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
