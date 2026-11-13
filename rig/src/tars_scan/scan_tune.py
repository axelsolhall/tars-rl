"""Pairwise tuning of the scan mesh by eye (Axel, 2026-11-13): two meshes of the same scan, he says
which agrees better with reality, the next pair is built from his pick.

  scripts/scan_tune init <name> <session> <export dir> <trajectory> [--from <name>] [--set k=v]
  scripts/scan_tune next <name> [--hp power] [--set k=v]   # build a pair and open it in Rerun
  scripts/scan_tune pick <name> A|B|same [note]            # record the verdict, build the next pair
  scripts/scan_tune status <name>

Method: champion vs challenger, one hyperparameter at a time (a (1+1) evolution strategy driven by
preferences). The challenger is the champion with one setting stepped up or down. A win makes it
the champion and the next step goes the same way, 1.5x larger; a loss tries the other direction,
and when both directions have lost the step halves and the next setting is tried. "same" moves on.
Sides are random and the views are named A and B only: the verdict is on the picture. Every duel
is logged (history in state.json), so a preferential Bayesian optimiser (Gonzalez et al. 2017) can
be fitted to the same data later. Fuses are cached by their fuse-relevant parameters; draw_level
and min_faces only filter faces, so changing them costs no fuse.
State: runs/scan/tune/<name>/ (state.json, cache/<hash>/ = scan_mesh outputs, pairs/<n>/index.html:
the A/B page, one camera for both views).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .scan_mesh import MESH_ONLY, ScanParams, parse_params

ROOT = Path("runs/scan/tune")
# name: (kind, lo, hi, first step). lin: +-step; log: x/ step; bool: flip. Ranges from
# Voxblox (truncation 3-6 voxels), stereo noise (power 2 = quadratic, 4 = inverse variance) and
# what the walk2 runs showed (levels in seconds of close looking).
SPACE: dict[str, tuple[str, float, float, float]] = {
    "power": ("lin", 1.0, 8.0, 1.0),
    "draw_level": ("log", 0.05, 8.0, 2.0),
    "remove_level": ("log", 0.05, 8.0, 2.0),
    "remove_tau_s": ("log", 1.0, 120.0, 2.0),
    "sight_margin_m": ("lin", 0.04, 0.5, 0.06),
    "block_level": ("log", 0.1, 8.0, 2.0),
    "ever_free_level": ("log", 0.25, 32.0, 2.0),
    "free_clear_m": ("lin", 0.05, 0.6, 0.1),
    "trunc_m": ("lin", 0.04, 0.16, 0.02),
    "max_depth_m": ("lin", 1.5, 4.0, 0.5),
    "z_ref_m": ("lin", 0.5, 2.0, 0.25),
    "dropoff": ("bool", 0, 1, 1),
    "min_faces": ("log", 10, 10000, 3.0),
}


def tune_dir(name: str) -> Path:
    return ROOT / name


def load(name: str) -> dict:
    return json.loads((tune_dir(name) / "state.json").read_text())


def save(name: str, st: dict) -> None:
    (tune_dir(name) / "state.json").write_text(json.dumps(st, indent=2))


def fuse_key(params: dict) -> str:
    core = {k: v for k, v in sorted(params.items()) if k not in MESH_ONLY}
    return hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:10]


def stepped(params: dict, hp: str, direction: int, step: float) -> dict | None:
    kind, lo, hi, _ = SPACE[hp]
    x = params[hp]
    if kind == "bool":
        y = not bool(x)
    elif kind == "lin":
        y = round(min(hi, max(lo, x + direction * step)), 4)
    else:
        y = min(hi, max(lo, x * step**direction))
        y = round(y, 4) if isinstance(x, float) else int(round(y))
    return None if y == x else {**params, hp: y}


def challenger(st: dict, hp: str | None, overrides: dict) -> tuple[str, dict]:
    champ = st["champion"]
    if overrides:  # Axel's own proposal
        p = asdict(parse_params({k: str(v) for k, v in overrides.items()}, ScanParams(**champ)))
        return ",".join(overrides), p
    order = st["order"]
    names = [hp] if hp else order[st["cursor"] :] + order[: st["cursor"]]
    for h in names:
        s = st["steps"][h]
        for d in (s["dir"], -s["dir"]):
            p = stepped(champ, h, d, s["step"])
            tried = any(x["champion"] == champ and x["challenger"] == p for x in st["history"])
            if p is not None and not tried:
                s["dir"] = d
                st["cursor"] = order.index(h)
                return h, p
    raise SystemExit("every setting has been tried from this champion: widen SPACE or --set")


def ensure_fused(st: dict, params: list[dict]) -> list[Path]:
    """Fuse what the cache lacks, in parallel (one process each); returns the output dirs."""
    sc, outs, procs = st["scan"], [], []
    for p in params:
        out = tune_dir(st["name"]) / "cache" / fuse_key(p)
        outs.append(out)
        if (out / "fuse.json").exists() or any(out == o for o, _ in procs):
            continue
        sets = [f"--set={k}={v}" for k, v in p.items() if k not in MESH_ONLY]
        cmd = [sys.executable, "-m", "tars_scan.scan_mesh", "fuse", sc["session"], sc["export"]]
        cmd += [sc["trajectory"], "--out", str(out), *sets]
        out.mkdir(parents=True, exist_ok=True)
        log = open(out / "fuse.log", "w")  # noqa: SIM115
        procs.append((out, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
    for out, _ in procs:
        print(f"fusing {out.name} ...", flush=True)
    for out, pr in procs:
        if pr.wait() != 0:
            raise SystemExit(f"fuse failed: see {out / 'fuse.log'}")
    return outs


def build_pair(st: dict) -> Path:
    """Fuse what is missing and write the linked-camera A/B page (scan_abview)."""
    from .scan_abview import build

    pend = st["pending"]
    sides = {"A": pend[pend["A"]], "B": pend[pend["B"]]}
    t = time.time()
    dirs = ensure_fused(st, [sides["A"], sides["B"]])
    print(f"fused in {time.time() - t:.0f} s", flush=True)
    n = len(st["history"]) + 1
    spec = {
        side: (d, sides[side]["draw_level"], sides[side]["min_faces"])
        for side, d in zip(("A", "B"), dirs, strict=True)
    }
    return build(tune_dir(st["name"]) / "pairs" / f"{n:03d}", spec, f"{st['name']} pair {n}")


def open_viewer(page: Path) -> None:
    from .scan_abview import open_page

    open_page(page)


def cmd_next(name: str, hp: str | None, overrides: dict, spawn: bool = True) -> None:
    st = load(name)
    h, p = challenger(st, hp, overrides)
    champ_side = random.choice("AB")
    other = "B" if champ_side == "A" else "A"
    st["pending"] = {
        "hp": h,
        "champion": st["champion"],
        "challenger": p,
        champ_side: "champion",
        other: "challenger",
    }
    save(name, st)
    rrd = build_pair(st)
    st["pending"]["page"] = str(rrd)
    save(name, st)
    print(
        f"pair {len(st['history']) + 1}: {rrd}\nWhich looks more like the real place? A, B or same"
    )
    if spawn:
        open_viewer(rrd)


def cmd_pick(name: str, verdict: str, note: str) -> None:
    st = load(name)
    pend = st.get("pending")
    if not pend:
        raise SystemExit("no pair is waiting: run next")
    verdict = verdict.upper() if verdict.lower() != "same" else "same"
    winner = pend[verdict] if verdict in ("A", "B") else "same"
    h = pend["hp"]
    entry = {
        "n": len(st["history"]) + 1,
        "hp": h,
        "champion": pend["champion"],
        "challenger": pend["challenger"],
        "sides": {"A": pend["A"], "B": pend["B"]},
        "verdict": verdict,
        "winner": winner,
        "note": note,
        "time": time.strftime("%Y-%m-%d %H:%M"),
    }
    st["history"].append(entry)
    s = st["steps"].get(h)
    if winner == "challenger":
        st["champion"] = pend["challenger"]
        if s:
            kind = SPACE[h][0]
            s["step"] = s["step"] ** 1.5 if kind == "log" else s["step"] * 1.5
            s["lost"] = 0
    elif s:
        s["lost"] = s.get("lost", 0) + 1
        s["dir"] = -s["dir"]
        if s["lost"] >= 2 or winner == "same":  # both ways lost (or no difference): smaller, next
            kind = SPACE[h][0]
            s["step"] = math.sqrt(s["step"]) if kind == "log" else s["step"] / 2
            s["lost"] = 0
            st["cursor"] = (st["order"].index(h) + 1) % len(st["order"])
    st["pending"] = None
    save(name, st)
    print(f"{verdict}: {winner} wins ({h})")
    cmd_status(name)


def cmd_status(name: str) -> None:
    st = load(name)
    base = asdict(ScanParams())
    diff = {k: v for k, v in st["champion"].items() if base.get(k) != v}
    print(
        f"{name}: {len(st['history'])} duels; champion differs from defaults in {diff or 'nothing'}"
    )
    for e in st["history"][-8:]:
        c, ch = e["champion"].get(e["hp"], ""), e["challenger"].get(e["hp"], "")
        print(f"  {e['n']:3d} {e['hp']:<16} {c} vs {ch}: {e['winner']} {e['note']}")


def cmd_init(name: str, session: str, export: str, traj: str, frm: str | None, sets: dict) -> None:
    d = tune_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    champ = load(frm)["champion"] if frm else asdict(ScanParams())
    champ = asdict(parse_params({k: str(v) for k, v in sets.items()}, ScanParams(**champ)))
    st = {
        "name": name,
        "scan": {"session": session, "export": export, "trajectory": traj},
        "champion": champ,
        "order": list(SPACE),
        "cursor": 0,
        "steps": {k: {"step": v[3], "dir": 1, "lost": 0} for k, v in SPACE.items()},
        "history": [],
        "pending": None,
    }
    save(name, st)
    print(f"{d / 'state.json'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("name")
    i.add_argument("session")
    i.add_argument("export")
    i.add_argument("trajectory")
    i.add_argument("--from", dest="frm")
    i.add_argument("--set", action="append", default=[])
    n = sub.add_parser("next")
    n.add_argument("name")
    n.add_argument("--hp", choices=list(SPACE))
    n.add_argument("--set", action="append", default=[])
    n.add_argument("--no-spawn", action="store_true")
    p = sub.add_parser("pick")
    p.add_argument("name")
    p.add_argument("verdict", choices=["A", "B", "a", "b", "same"])
    p.add_argument("note", nargs="*")
    p.add_argument("--no-next", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("name")
    a = ap.parse_args(argv)
    kv = lambda xs: dict(x.split("=", 1) for x in xs)  # noqa: E731
    if a.cmd == "init":
        cmd_init(a.name, a.session, a.export, a.trajectory, a.frm, kv(a.set))
    elif a.cmd == "next":
        cmd_next(a.name, a.hp, kv(a.set), not a.no_spawn)
    elif a.cmd == "pick":
        cmd_pick(a.name, a.verdict, " ".join(a.note))
        if not a.no_next:
            cmd_next(a.name, None, {})
    else:
        cmd_status(a.name)


if __name__ == "__main__":
    main()
