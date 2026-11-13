"""Automatic scan-mesh tuning against a reference scan (Axel, 2026-11-13: a very thorough pass with
board resets is the reference, all about quality; everyday quick passes are scored against it,
"automagically", for performance: quality per compute and per data).

  scripts/scan_autotune <study> --ref <reference scan dir> --scan <session>,<export>,<traj> ...
                        [--trials 30] [--jobs 2] [--to-tune <scan_tune name>]

Each trial: Optuna's multi-objective TPE proposes the settings (scan_tune.SPACE plus the cost
settings below), every quick scan is fused with them (cached by settings), and each is scored with
scan_score at several drawing levels (a face filter, no re-fuse). Two objectives: the best mean
F-score over the scans (the level that gave it is kept with the trial), and compute per second of
walk (fuse.json, Mac wall time; jobs=1 keeps it fair). Fewer keyframes (every), fewer rays
(ray_every), coarser voxels and smaller depth images cost less and use less of the data: the result
is a Pareto front, the best F at each cost, and Axel picks the trade-off (e.g. what runs live on
the Pi). The study lives in <study>/study.db and resumes where it stopped; pareto.json lists the
front. --to-tune makes a front point the champion of a scan_tune run for an eye check (the best F
within --budget, else the best F). Nothing here sets a pass bar: it ranks settings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .scan_mesh import MESH_ONLY, ScanParams
from .scan_score import score
from .scan_tune import SPACE, fuse_key

ROOT = Path("runs/scan/autotune")
LEVELS = (0.1, 0.25, 0.5, 1.0, 2.0)


# the cost settings, searched here but not stepped by the eye tuner
COST = {
    "every": ("int_log", 2, 32),  # keyframe every n-th pair: data per second of walk
    "ray_every": ("int", 1, 6),  # rectified pixels between rays
    "voxel_m": ("cat", (0.015, 0.02, 0.03, 0.04)),
    "rect_out": ("cat", (0.5, 0.75, 1.0)),  # depth image size vs the export
    "align_every": ("int", 0, 3),  # scan-to-map drift alignment every k-th keyframe (0 = off)
}


def suggest(trial) -> dict:
    p = asdict(ScanParams())
    for k, (kind, lo, hi, _) in SPACE.items():
        if k in MESH_ONLY or k == "trunc_m":
            continue
        if kind == "bool":
            p[k] = trial.suggest_categorical(k, [True, False])
        else:
            p[k] = trial.suggest_float(k, lo, hi, log=kind == "log")
    for k, spec in COST.items():
        if spec[0] == "cat":
            p[k] = trial.suggest_categorical(k, list(spec[1]))
        else:
            p[k] = trial.suggest_int(k, spec[1], spec[2], log=spec[0] == "int_log")
    # truncation in voxels (Voxblox: 4), so it stays meaningful when the voxel size changes
    p["trunc_m"] = round(trial.suggest_float("trunc_vox", 2.0, 6.0) * p["voxel_m"], 4)
    return p


def fuse_all(study_dir: Path, scans: list[list[str]], p: dict, jobs: int) -> list[Path]:
    outs, queue = [], []
    for i, (session, export, traj) in enumerate(scans):
        out = study_dir / "cache" / f"s{i}_{fuse_key(p)}"
        outs.append(out)
        if not (out / "fuse.json").exists():
            sets = [f"--set={k}={v}" for k, v in p.items() if k not in MESH_ONLY]
            cmd = [sys.executable, "-m", "tars_scan.scan_mesh", "fuse", session, export, traj]
            queue.append((out, [*cmd, "--out", str(out), *sets]))
    running: list = []
    while queue or running:
        while queue and len(running) < jobs:
            out, cmd = queue.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = open(out / "fuse.log", "w")  # noqa: SIM115
            running.append((out, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        out, pr = running.pop(0)
        if pr.wait() != 0:
            raise RuntimeError(f"fuse failed: {out / 'fuse.log'}")
    return outs


def objective(study_dir: Path, ref: Path, scans: list[list[str]], jobs: int):
    def run(trial) -> tuple[float, float]:
        p = suggest(trial)
        t = time.time()
        outs = fuse_all(study_dir, scans, p, jobs)
        per_level = {}
        for lev in LEVELS:
            per_level[lev] = [score(ref, o, level=lev) for o in outs]
        best = max(LEVELS, key=lambda lv: np.mean([s["f"] for s in per_level[lv]]))
        s = per_level[best]
        trial.set_user_attr("draw_level", best)
        trial.set_user_attr("precision", float(np.mean([x["precision"] for x in s])))
        trial.set_user_attr("recall", float(np.mean([x["recall"] for x in s])))
        trial.set_user_attr("registered", all(x["registered"] for x in s))
        trial.set_user_attr("seconds", round(time.time() - t))
        cost = float(
            np.mean([json.loads((o / "fuse.json").read_text())["compute_per_walk_s"] for o in outs])
        )
        trial.set_user_attr("params", {k: v for k, v in p.items()})
        f = float(np.mean([x["f"] for x in s]))
        print(
            f"trial {trial.number}: F {f:.3f} (P {trial.user_attrs['precision']:.3f} "
            f"R {trial.user_attrs['recall']:.3f}) level {best}  cost {cost:.2f} s/s  "
            f"{time.time() - t:.0f} s",
            flush=True,
        )
        return f, cost

    return run


def front(study) -> list[dict]:
    """The Pareto front, cheapest first: each point's F, cost and full settings."""
    pts = []
    for t in study.best_trials:
        p = {**t.user_attrs["params"], "draw_level": t.user_attrs["draw_level"]}
        p = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}
        pts.append(
            {
                "trial": t.number,
                "f": round(t.values[0], 4),
                "cost": round(t.values[1], 3),
                "precision": t.user_attrs["precision"],
                "recall": t.user_attrs["recall"],
                "params": p,
            }
        )
    return sorted(pts, key=lambda x: x["cost"])


def main(argv: list[str] | None = None) -> None:
    import optuna

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("study")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--scan", action="append", required=True, help="session,export,trajectory")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=2, help="fuses at once (~2.5 GB each)")
    ap.add_argument(
        "--to-tune", default=None, help="make a front point the champion of this scan_tune"
    )
    ap.add_argument("--budget", type=float, default=None, help="compute s per walk s for --to-tune")
    a = ap.parse_args(argv)
    d = ROOT / a.study
    d.mkdir(parents=True, exist_ok=True)
    scans = [s.split(",") for s in a.scan]
    study = optuna.create_study(
        study_name=a.study,
        storage=f"sqlite:///{d / 'study.db'}",
        directions=["maximize", "minimize"],
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8),
    )
    if len(study.trials) == 0:  # start from the hand-set defaults
        dflt = asdict(ScanParams())
        start = {k: dflt[k] for k in SPACE if k not in MESH_ONLY and k != "trunc_m"}
        start.update({k: dflt[k] for k in COST})
        start["trunc_vox"] = dflt["trunc_m"] / dflt["voxel_m"]
        study.enqueue_trial(start)
    study.optimize(objective(d, Path(a.ref), scans, a.jobs), n_trials=a.trials)
    pts = front(study)
    (d / "pareto.json").write_text(json.dumps(pts, indent=2))
    print("Pareto front (cheapest first):")
    for x in pts:
        print(
            f"  trial {x['trial']:3d}  F {x['f']:.3f}  cost {x['cost']:.2f} s/s  "
            f"every {x['params']['every']} ray_every {x['params']['ray_every']} "
            f"voxel {x['params']['voxel_m']}"
        )
    if a.to_tune and pts:
        from .scan_tune import load, save

        ok = [x for x in pts if a.budget is None or x["cost"] <= a.budget] or pts[:1]
        pick = max(ok, key=lambda x: x["f"])
        st = load(a.to_tune)
        st["champion"] = pick["params"]
        save(a.to_tune, st)
        print(f"champion of scan_tune {a.to_tune} set to trial {pick['trial']}")


if __name__ == "__main__":
    main()
