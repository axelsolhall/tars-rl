"""Self-guiding search eval.

Run: python -m tars_rl.tune.eval_search --budget 200 --repeats 5 --seed 0

On every test function (known best value f*), runs the moving-box search and plain random
search over the whole space with the same number of tries and the same seeds, and prints:
  1. gap = best found - f*, median over repeats, after 25 / 50 / 100 / budget tries, for box
     and random, and which one is closer at the full budget (lower gap wins).
  2. the box history of repeat 0 per function: tries so far, best gap, distance from the box
     centre to the nearest optimum (unit coordinates, 0..sqrt(d)), side length, and event.
The starting box is centred in the middle of each space (length 0.4); for ackley5, levy4,
styblinski4 and mixed5 the optimum lies outside it, so the box has to travel.
Random search covers the whole space (A89). Exit code 1 if the box wins on fewer than half
the functions (A75's signpost).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from tars_rl.tune.search import BoxSearch, SearchConfig, random_search, run
from tars_rl.tune.testfuncs import TestFunction, suite


def checkpoints(budget: int) -> list[int]:
    return sorted({c for c in (25, 50, 100) if c < budget} | {budget})


def best_curve(values: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(values)


def eval_function(fn: TestFunction, budget: int, repeats: int, seed: int, cfg: SearchConfig):
    box_curves, rand_curves, history = [], [], None
    for r in range(repeats):
        s = run(BoxSearch(fn.space, cfg, seed=seed + r), fn, budget)
        box_curves.append(best_curve(np.array([t.score for t in s.trials])))
        u = random_search(fn.space, budget, seed + r)
        rand_curves.append(best_curve(np.array([fn(fn.space.decode(x)) for x in u])))
        if r == 0:
            history = box_history(fn, s)
    gaps = lambda c: np.maximum(np.array(c) - fn.f_opt, 0.0)  # noqa: E731
    return gaps(box_curves), gaps(rand_curves), history


def box_history(fn: TestFunction, s: BoxSearch) -> list[dict]:
    rows, tried = [], 0
    for b in s.boxes:
        tried += sum(t.round == b.round - 1 for t in s.trials) if b.round else 0
        rows.append({
            "round": b.round, "tries": tried,
            "gap": None if b.best is None else fn.gap(b.best),
            "dist": fn.dist_to_opt(np.array(b.centre)), "length": b.length, "event": b.event,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="run only these test functions")
    ap.add_argument("--json", type=Path, help="save all numbers here")
    args = ap.parse_args(argv)

    cfg = SearchConfig()
    fns = [f for f in suite() if not args.only or f.name in args.only]
    cps = checkpoints(args.budget)
    out, wins, t0 = {}, 0, time.time()

    print(f"1. Median gap to the known best (lower is better), {args.repeats} repeats")
    head = "".join(f"{f'box@{c}':>10}{f'rand@{c}':>10}" for c in cps)
    print(f"{'function':<12}{'d':>3}{head}  closer")
    histories = {}
    for fn in fns:
        bg, rg, hist = eval_function(fn, args.budget, args.repeats, args.seed, cfg)
        mb = {c: float(np.median(bg[:, c - 1])) for c in cps}
        mr = {c: float(np.median(rg[:, c - 1])) for c in cps}
        win = mb[args.budget] < mr[args.budget]
        wins += win
        cells = "".join(f"{mb[c]:>10.3g}{mr[c]:>10.3g}" for c in cps)
        print(f"{fn.name:<12}{fn.space.dim:>3}{cells}  {'box' if win else 'random'}")
        histories[fn.name] = hist
        out[fn.name] = {"box_median": mb, "random_median": mr, "box_wins": bool(win),
                        "box_gaps_final": bg[:, -1].tolist(),
                        "random_gaps_final": rg[:, -1].tolist(),
                        "history": hist}
    print(f"box closer on {wins}/{len(fns)} functions  ({time.time() - t0:.1f} s)")

    print("\n2. Box history, repeat 0 (dist = box centre to nearest optimum, unit coords)")
    for name, hist in histories.items():
        print(f"\n{name}")
        print(f"{'round':>5}{'tries':>6}{'gap':>11}{'dist':>7}{'length':>8}  event")
        for h in hist:
            gap = "-" if h["gap"] is None else f"{h['gap']:.3g}"
            print(f"{h['round']:>5}{h['tries']:>6}{gap:>11}{h['dist']:>7.3f}"
                  f"{h['length']:>8.4f}  {h['event']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"args": vars(args) | {"json": str(args.json)},
                                         "results": out}, indent=1))
    return 0 if 2 * wins >= len(fns) else 1


if __name__ == "__main__":
    sys.exit(main())
