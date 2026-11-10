"""Run a SLAM system (in its Docker image) on an exported session and score it.

  python -m tars_scan.slam_run {basalt,openvins,okvis2} <session> <export dir> <name> [--set k=v]

Writes runs/slam/out/<name>.txt (trajectory), <name>.json (score + settings + runtime) and appends
one line to runs/slam/results.jsonl, so every variant tried stays comparable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from . import slam_calib, slam_eval

ROOT = Path(__file__).resolve().parents[3] / "runs" / "slam"
SLAM_SRC = Path.home() / "dev/slam"
BASALT_BASE_CONFIG = SLAM_SRC / "basalt/data/hilti26/hilti_config.json"
OPENVINS_BASE_CONFIG = SLAM_SRC / "open_vins/config/euroc_mav/estimator_config.yaml"
OKVIS2_BASE_CONFIG = SLAM_SRC / "okvis2x/config/euroc/okvis2.yaml"
# Out-of-the-box settings changed for every run: no windows in Docker, all 30 fps frames used.
OPENVINS_DEFAULTS = {"verbosity": '"WARNING"', "track_frequency": "31.0"}
OKVIS2_DEFAULTS = {
    "display_topview": "false",
    "display_matches": "false",
    "enable_submapping": "false",
}


def _value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def run_basalt(
    session: Path, dataset: Path, name: str, overrides: dict, scale: float, imu: bool
) -> dict:
    cfg_dir = ROOT / "cfg" / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(BASALT_BASE_CONFIG.read_text())
    for k, v in overrides.items():
        if not k.startswith(("calib.", "env.")):
            cfg["value0"][f"config.{k}"] = v
    (cfg_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    calib = slam_calib.basalt_calib(slam_calib.rig_model(session), (1640, 1232), scale)
    for k, v in overrides.items():
        if k.startswith("calib."):
            calib["value0"][k[6:]] = v
    (cfg_dir / "calib.json").write_text(json.dumps(calib, indent=2))
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    # env.NAME=value goes to the process environment (e.g. env.TARS_STEREO_DEPTHS, our patch).
    envs = [x for k, v in overrides.items() if k.startswith("env.") for x in ("-e", f"{k[4:]}={v}")]
    t0 = time.time()
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            *envs,
            "-v",
            "slam-basalt-build:/build",
            "-v",
            f"{ROOT}:/work",
            "-v",
            f"{dataset.resolve()}:/data",
            "-w",
            f"/work/cfg/{name}",
            "slam-basalt-deps",
            "/build/basalt_vio",
            "--dataset-path",
            "/data",
            "--dataset-type",
            "euroc",
            "--cam-calib",
            f"/work/cfg/{name}/calib.json",
            "--config-path",
            f"/work/cfg/{name}/config.json",
            "--show-gui",
            "0",
            "--save-trajectory",
            "tum",
            "--save-trajectory-fn",
            f"/work/out/{name}.txt",
            "--num-threads",
            "4",
            "--use-imu",
            "1" if imu else "0",
        ],
        capture_output=True,
        text=True,
    )
    runtime = next((ln for ln in proc.stdout.splitlines() if "Total runtime" in ln), "")
    return _finish(session, dataset, name, "basalt", imu, overrides, proc, t0, runtime.strip())


def _finish(session, dataset, name, system, imu, overrides, proc, t0, runtime="") -> dict:
    """Score the trajectory at out/<name>.txt and log the run to results.jsonl."""
    out = ROOT / "out"
    result = {
        "name": name,
        "system": system,
        "dataset": dataset.name,
        "imu": imu,
        "overrides": overrides,
        "wall_s": round(time.time() - t0, 1),
        "runtime": runtime,
    }
    traj = out / f"{name}.txt"
    if proc.returncode != 0 or not traj.exists() or traj.stat().st_size < 100:
        result["error"] = (proc.stderr or proc.stdout)[-800:]
    else:
        result["score"] = slam_eval.score(session, traj)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))
    with open(ROOT / "results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    return result


def _set_yaml_keys(text: str, values: dict) -> str:
    """Replace `key: value` lines (any indent) of an OpenCV-style yaml; unknown keys are an error,
    so a typo in --set never silently does nothing."""
    import re

    for k, v in values.items():
        text, n = re.subn(rf"^(\s*){re.escape(k)}:[^\n#]*", rf"\g<1>{k}: {v} ", text, flags=re.M)
        if n == 0:
            raise KeyError(f"{k} is not a key of the base config")
    return text


def _docker(
    image: str, volumes: list[str], args: list[str], env: dict | None = None
) -> subprocess.CompletedProcess:
    vols = [x for v in volumes for x in ("-v", v)]
    envs = [x for k, v in (env or {}).items() for x in ("-e", f"{k}={v}")]
    cmd = ["docker", "run", "--rm", *vols, *envs, image, *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def run_openvins(session, dataset, name, overrides, scale, imu) -> dict:
    cfg_dir = ROOT / "cfg" / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    model = slam_calib.rig_model(session)
    for fname, text in slam_calib.openvins_files(model, (1640, 1232), scale).items():
        (cfg_dir / fname).write_text(text)
    est = _set_yaml_keys(OPENVINS_BASE_CONFIG.read_text(), {**OPENVINS_DEFAULTS, **overrides})
    (cfg_dir / "estimator_config.yaml").write_text(est)
    (ROOT / "out").mkdir(exist_ok=True)
    t0 = time.time()
    proc = _docker(
        "slam-vio2-deps",
        ["slam-ov-build:/build", f"{ROOT}:/work", f"{dataset.resolve()}:/data"],
        [
            "/build/ov_euroc",
            f"/work/cfg/{name}/estimator_config.yaml",
            "/data/mav0",
            f"/work/out/{name}.txt",
        ],
    )
    return _finish(session, dataset, name, "openvins", imu, overrides, proc, t0)


def run_okvis2(session, dataset, name, overrides, scale, imu) -> dict:
    """OKVIS2 with loop closure. Scores the causal (real-time) trajectory under <name>; the
    after-the-fact final trajectory is scored as <name>-final."""
    cfg_dir = ROOT / "cfg" / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    patch = slam_calib.okvis2_patch(slam_calib.rig_model(session), (1640, 1232), scale)
    base = OKVIS2_BASE_CONFIG.read_text()
    rest = base[base.index("# additional camera parameters") :]
    cams = "cameras:\n" + "".join(
        "     - {" + ",\n        ".join(f"{k}: {json.dumps(v)}" for k, v in c.items()) + "}\n"
        for c in patch["cameras"]
    )
    imu_keys = {k: json.dumps(v) for k, v in patch["imu_parameters"].items()}
    imu_keys["use"] = "true" if imu else "false"
    text = _set_yaml_keys("%YAML:1.0\n" + cams + "\n" + rest, {**imu_keys, **OKVIS2_DEFAULTS})
    # env.NAME=value goes to the process environment (e.g. env.TARS_PREFILTER, our patch).
    env = {k[4:]: v for k, v in overrides.items() if k.startswith("env.")}
    keys = {k: v for k, v in overrides.items() if not k.startswith("env.")}
    (cfg_dir / "okvis2.yaml").write_text(_set_yaml_keys(text, keys))
    run_dir = ROOT / "out" / f"{name}-okvis"
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = _docker(
        "slam-vio2-deps",
        ["slam-okvis-build:/build", f"{ROOT}:/work", f"{dataset.resolve()}:/data"],
        [
            "/build/okvis_app_synchronous",
            f"/work/cfg/{name}/okvis2.yaml",
            "/data/mav0",
            f"/work/out/{name}-okvis",
        ],
        env,
    )
    files = {p.name: p for p in run_dir.glob("*trajectory.csv")}
    live = next(
        (p for n, p in files.items() if n.endswith("_trajectory.csv") and "final" not in n), None
    )
    final = next((p for n, p in files.items() if "final-ba" in n and "global" not in n), None)
    final = final or next((p for n, p in files.items() if "final_trajectory" in n), None)
    for src, nm in ((live, name), (final, f"{name}-final")):
        if src is not None:  # CSV "t_ns, p xyz, q xyzw, v, biases" -> TUM (t_ns kept)
            rows = [ln.split(",")[:8] for ln in src.read_text().splitlines()[1:] if ln.strip()]
            tum = "\n".join(" ".join(c.strip() for c in r) for r in rows)
            (ROOT / "out" / f"{nm}.txt").write_text(tum + "\n")
    r = _finish(session, dataset, name, "okvis2", imu, overrides, proc, t0)
    if final is not None:
        _finish(session, dataset, f"{name}-final", "okvis2-final", imu, overrides, proc, t0)
    return r


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tars_scan.slam_run", description=__doc__.split("\n")[0])
    p.add_argument("system", choices=["basalt", "openvins", "okvis2"])
    p.add_argument("session", type=Path)
    p.add_argument("dataset", type=Path)
    p.add_argument("name")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--no-imu", action="store_true")
    p.add_argument(
        "--set", action="append", default=[], help="config key=value (calib.key for calibration)"
    )
    a = p.parse_args(argv)
    overrides = {k: _value(v) for k, v in (s.split("=", 1) for s in a.set)}
    run = {"basalt": run_basalt, "openvins": run_openvins, "okvis2": run_okvis2}[a.system]
    r = run(a.session, a.dataset, a.name, overrides, a.scale, not a.no_imu)
    print(json.dumps({k: r.get(k) for k in ("name", "wall_s", "score", "error")}))


if __name__ == "__main__":
    main()
