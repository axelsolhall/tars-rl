"""One recording: make the session directory, start the camera and IMU processes, stop them, write
done.json and check.json. Stops on stop_event, on the session's time cap, or when a recorder process
dies.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from . import check
from .config import Config
from .contracts import CameraMeta, ImuMeta, SessionMeta, write_json
from .mpu6050 import accel_lsb_per_g, chip_name, gyro_lsb_per_dps

log = logging.getLogger("rig.session")


def _log_to(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def new_session_dir(cfg: Config, mode: str) -> Path:
    d = cfg.sessions_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{mode}"
    d.mkdir(parents=True)
    return d


def _proc_main(target, args, log_path):
    # Child processes (spawn) start with no logging setup; send theirs to the session log too.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _log_to(log_path)
    target(*args)


def start_live_map(cfg: Config, session: Path) -> subprocess.Popen:
    """The live map for a session being recorded: copy the rig's calibration (<root>/calib,
    `scripts/rig push-calib`) into the session, then run tars_scan.live_map on it, following the
    files. It talks to tars_live_vio on this host and streams to the viewer over the SSH tunnel."""
    live_cfg = cfg.raw.get("live", {})
    src = cfg.root / "calib"
    if not (src / "calib.json").exists():
        raise FileNotFoundError(f"{src}/calib.json missing: run scripts/rig push-calib first")
    shutil.copytree(src, session / "calib")
    (session / "live").mkdir()
    log_f = open(session / "live" / "log.txt", "w")  # noqa: SIM115 — the child owns it
    cmd = [
        sys.executable,
        "-m",
        "tars_scan.live_map",
        str(session),
        "--follow",
        "--connect",
        live_cfg.get("viewer_url", "none"),
        "--basalt-port",
        str(live_cfg.get("basalt_port", 5077)),
    ]
    if not live_cfg.get("rs_correct", False):
        cmd.append("--no-rs")
    log.info("live map: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)


def phase_plan(phases: list[dict]) -> list[tuple[str, float]]:
    """(name, end second) per phase, counted from when the recorders are running."""
    plan, end = [], 0.0
    for ph in phases:
        end += float(ph["seconds"])
        plan.append((str(ph["name"]), end))
    return plan


def run_session(
    cfg: Config, mode: str, stop_event, session_dir: Path | None = None, on_phase=None
) -> dict:
    from .cameras import LENS, meter_exposure, run_camera, sync_mode
    from .imu import run_imu

    scfg = cfg.sessions[mode]
    session = session_dir or new_session_dir(cfg, mode)
    handler = _log_to(session / "log.txt")
    ctx = mp.get_context("spawn")
    child_stop = ctx.Event()
    procs: list[tuple[str, mp.Process]] = []
    meta = SessionMeta(
        mode=mode,
        started_utc=datetime.now(UTC).isoformat(),
        host=socket.gethostname(),
        max_minutes=scfg.max_minutes,
        settings=cfg.raw,
    )
    errors: list[str] = []
    phases: list[dict] = []  # phases.json: name, start_ns, end_ns (CLOCK_MONOTONIC, as the frames)
    stop_reason = "stopped"
    t0 = time.monotonic()
    live = None
    try:
        if scfg.live:
            live = start_live_map(cfg, session)
        if scfg.imu:
            rate, who = ctx.Value("i", 0), ctx.Value("i", 0)
            p = ctx.Process(
                target=_proc_main,
                name="imu",
                args=(
                    run_imu,
                    (cfg.imu, session / "imu", child_stop, rate, who, scfg.live),
                    session / "log.txt",
                ),
            )
            p.start()
            procs.append(("imu", p))
        if scfg.cameras:
            exposure, gain = meter_exposure(cfg.cameras, scfg.max_exposure_us)
            roles = ("server", "client") if cfg.cameras.sync else ("none", "none")
            started = []
            for index in (1, 0):  # the sync client must be running before the server starts
                ev = ctx.Event()
                p = ctx.Process(
                    target=_proc_main,
                    name=f"cam{index}",
                    args=(
                        run_camera,
                        (
                            index,
                            roles[index],
                            cfg.cameras,
                            exposure,
                            gain,
                            session / f"cam{index}",
                            child_stop,
                            ev,
                            1 if scfg.live else 30,
                        ),
                        session / "log.txt",
                    ),
                )
                p.start()
                procs.append((f"cam{index}", p))
                started.append(ev)
                ev.wait(10)
            for index in (0, 1):
                timing_path = session / f"cam{index}" / "sensor.json"
                timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
                if "line_time_ns" not in timing:
                    log.warning("cam%d: no sensor line timing (%s)", index, timing.get("error"))
                meta.cameras.append(
                    CameraMeta(
                        index,
                        cfg.cameras.width,
                        cfg.cameras.height,
                        cfg.cameras.fps,
                        exposure,
                        round(gain, 3),
                        roles[index],
                        sync_mode(roles[index]) is not None,
                        cfg.cameras.jpeg_quality,
                        timing.get("line_time_ns"),
                        timing.get("readout_us"),
                        index in cfg.cameras.color_cams,
                    )
                )
            meta.settings = {**cfg.raw, "lens": LENS}
        if scfg.imu:
            meta.imu = ImuMeta(
                f"{chip_name(who.value)} (GY-521)" if who.value else "unknown (GY-521)",
                rate.value or cfg.imu.rate_hz,
                accel_lsb_per_g(cfg.imu.accel_range_g),
                gyro_lsb_per_dps(cfg.imu.gyro_range_dps),
                cfg.imu.dlpf,
            )
        write_json(session / "meta.json", meta)
        cap_s = scfg.max_minutes * 60
        plan, p0 = phase_plan(scfg.phases), time.monotonic()
        if plan:
            phases.append({"name": plan[0][0], "start_ns": time.monotonic_ns(), "end_ns": None})
            write_json(session / "phases.json", phases)
        while True:
            wait = 0.5
            if plan:
                wait = min(wait, max(0.01, p0 + plan[len(phases) - 1][1] - time.monotonic()))
            if stop_event.wait(wait):
                break
            if plan and time.monotonic() - p0 >= plan[len(phases) - 1][1]:
                phases[-1]["end_ns"] = time.monotonic_ns()
                if len(phases) == len(plan):
                    stop_reason = "phases done"
                    break
                name = plan[len(phases)][0]
                phases.append({"name": name, "start_ns": phases[-1]["end_ns"], "end_ns": None})
                write_json(session / "phases.json", phases)
                log.info("phase %d/%d: %s", len(phases), len(plan), name)
                if on_phase is not None:
                    on_phase(name)
            if time.monotonic() - t0 >= cap_s:
                stop_reason = f"time cap ({scfg.max_minutes} min)"
                break
            if live is not None and live.poll() is not None:
                log.warning("live map exited (code %s); recording goes on", live.returncode)
                errors.append(f"live map exited with code {live.returncode}")
                live = None
            dead = [name for name, p in procs if not p.is_alive()]
            if dead:
                stop_reason = f"recorder died: {', '.join(dead)}"
                errors.append(stop_reason)
                break
    except Exception as e:  # noqa: BLE001 — any failure must still stop the recorders and leave a done.json
        log.exception("session failed")
        errors.append(repr(e))
        stop_reason = "error"
    finally:
        if phases:
            if phases[-1]["end_ns"] is None:
                phases[-1]["end_ns"] = time.monotonic_ns()
            write_json(session / "phases.json", phases)
        child_stop.set()
        for name, p in procs:
            p.join(15)
            if p.is_alive():
                p.terminate()
                errors.append(f"{name} did not stop in 15 s; terminated")
            elif p.exitcode not in (0, None):
                errors.append(f"{name} exit code {p.exitcode}")
        done = {
            "stop_reason": stop_reason,
            "ended_utc": datetime.now(UTC).isoformat(),
            "seconds": round(time.monotonic() - t0, 1),
            "errors": errors,
        }
        write_json(session / "done.json", done)
        if live is not None:  # it maps what is left, sees done.json and ends
            try:
                live.wait(30)
            except subprocess.TimeoutExpired:
                live.terminate()
                errors.append("live map did not end in 30 s; terminated")
        try:
            report = check.check_session(session)
            write_json(session / "check.json", report)
            done["check"] = check.summary_line(report)
        except Exception as e:  # noqa: BLE001
            errors.append(f"check failed: {e!r}")
            write_json(session / "done.json", done)
        logging.getLogger().removeHandler(handler)
    log.info("session %s: %s %s", session.name, stop_reason, errors or "")
    return {"session": str(session), **done}
