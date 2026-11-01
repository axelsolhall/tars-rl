"""Tuning campaign eval: is the search finding better settings than the published ones?

Run: python -m tars_rl.tune.eval_campaign --campaign runs/tune_smoke

Reads a campaign's trial records and search state and prints:
  1. every trial: round, network, the three score parts (raw), each part's weighted share and the
     total, the best total so far, and whether it beats trial 0 (the published settings)
  2. the box's path: per round its event, side length, best score, and the centre's settings
  3. the best trial's settings next to the published start
and saves campaign.png in the campaign folder: best score per trial against the published-
settings score, every trial's score, and the box side length per round. The campaign passes
(A75's signpost on real runs) when the best score rises above trial 0's; the eval prints the
verdict but exits 0 either way (the smoke campaign is not expected to improve).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tars_rl.tune.campaign import CAMPAIGN_FILE, SEARCH_FILE, read_records
from tars_rl.tune.score import WEIGHTS
from tars_rl.tune.tars_space import PPO_KEYS, SPACE, network_of


def summarize(campaign: str | Path) -> dict:
    campaign = Path(campaign)
    manifest = json.loads((campaign / CAMPAIGN_FILE).read_text())
    search = json.loads((campaign / SEARCH_FILE).read_text())
    recs = [r for r in read_records(campaign) if r.get("score") is not None]
    if not recs:
        raise SystemExit(f"{campaign} has no finished trials yet")
    best, rows = None, []
    base = next((r for r in recs if r["id"] == 0), None)
    for r in recs:
        if best is None or r["score"] > best["score"]:
            best = r
        rows.append({"id": r["id"], "round": r["round"], "network": r.get("network_str", "-"),
                     "parts": r["parts"], "score": r["score"], "best_so_far": best["score"],
                     "best_id": best["id"], "error": r.get("error")})
    return {"campaign": str(campaign), "manifest": manifest, "rows": rows, "boxes": search["boxes"],
            "baseline": base, "best": best,
            "improved": base is not None and best["score"] > base["score"]}


def _short(settings: dict) -> str:
    s = settings
    return (f"lr {s['learning_rate']:.1e} ent {s['entropy_cost']:.1e} "
            f"gamma {s['discounting']:.3f} unroll {s['unroll_length']} "
            f"upd {s['num_updates_per_batch']} clip {s['max_grad_norm']:.2f} "
            f"rscale {s['reward_scaling']:.2f} | {network_of(s)}")


def report(res: dict) -> None:
    m = res["manifest"]
    print(f"Campaign {res['campaign']}: profile {m['profile']}, seed {m['seed']}, "
          f"{len(res['rows'])} trials, {m['tune_profile']['num_timesteps']:,} steps each, "
          f"{m['tune_profile']['score_episodes']} held-out flags")
    w = WEIGHTS
    print(f"\n1. Trials (score = {w['success']} x success + {w['closed']} x closed "
          f"+ {w['speed']} x speed; trial 0 = published settings)")
    print(f"{'trial':>5}{'rnd':>4}{'success':>9}{'closed':>9}{'speed':>7}   "
          f"{'=succ':>6}{'+clos':>7}{'+spd':>7}{'total':>8}{'best':>8}  network")
    for r in res["rows"]:
        p = r["parts"]
        if r["error"]:
            print(f"{r['id']:>5}{r['round']:>4}  FAILED: {r['error'][:60]}  "
                  f"{r['score']:+.3f}{r['best_so_far']:>+8.3f}")
            continue
        print(f"{r['id']:>5}{r['round']:>4}{p['success']:>9.1%}{p['closed']:>+9.1%}"
              f"{p['speed']:>7.3f}   {w['success'] * p['success']:>6.3f}"
              f"{w['closed'] * p['closed']:>+7.3f}{w['speed'] * p['speed']:>+7.3f}"
              f"{r['score']:>+8.3f}{r['best_so_far']:>+8.3f}  {r['network']}")

    print("\n2. Box path (centre = best trial so far; side = share of each setting's range)")
    print(f"{'round':>5}  {'event':<8}{'side':>7}{'best':>8}  centre")
    for b in res["boxes"]:
        best = "-" if b["best"] is None else f"{b['best']:+.3f}"
        print(f"{b['round']:>5}  {b['event']:<8}{b['length']:>7.3f}{best:>8}  "
              f"{_short(SPACE.decode(b['centre']))}")

    base, best = res["baseline"], res["best"]
    print("\n3. Best trial vs published start")
    if base is not None:
        print(f"  published (trial 0) {base['score']:+.3f}  {_short(base['settings'])}")
    print(f"  best (trial {best['id']:>3}) {best['score']:+.3f}  {_short(best['settings'])}")
    changed = [k for k in PPO_KEYS if base and best["settings"][k] != base["settings"][k]]
    if changed:
        print(f"  training settings moved: {', '.join(changed)}")
    print(f"\nBest score above the published settings' score: "
          f"{'YES' if res['improved'] else 'NO'}")


def plot(res: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rows = res["rows"]
    ids = [r["id"] for r in rows]
    fig, (a1, a3, a2) = plt.subplots(1, 3, figsize=(16, 4.5))
    a1.scatter(ids, [r["score"] for r in rows], s=14, c="grey", label="trial score")
    a1.step(ids, [r["best_so_far"] for r in rows], where="post", c="C0", label="best so far")
    if res["baseline"] is not None:
        a1.axhline(res["baseline"]["score"], ls="--", c="C3", label="published settings")
    a1.set(xlabel="trial", ylabel="blended score", title="Best score per trial")
    a1.legend()
    boxes = res["boxes"]
    rounds = [b["round"] for b in boxes]
    for k in PPO_KEYS:  # the centre's path through the training settings, unit coordinates
        i = SPACE.names.index(k)
        a3.plot(rounds, [b["centre"][i] for b in boxes], marker=".", label=k)
    a3.set(xlabel="round", ylabel="box centre (0..1 of range)", title="Box path", ylim=(0, 1))
    a3.legend(fontsize=7)
    a2.step([b["round"] for b in boxes], [b["length"] for b in boxes], where="post", c="C2")
    for b in boxes:
        if b["event"] in ("grow", "shrink", "restart"):
            a2.annotate(b["event"], (b["round"], b["length"]), fontsize=7)
    a2.set(xlabel="round", ylabel="box side (share of range)", title="Search box size",
           yscale="log")
    for ax in (a1, a2, a3):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--campaign", required=True, help="folder written by tars_rl.tune.campaign")
    p.add_argument("--json", help="also write the summary to this JSON file")
    a = p.parse_args(argv)
    res = summarize(a.campaign)
    report(res)
    print(f"plot: {plot(res, Path(a.campaign) / 'campaign.png')}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
