"""One recording: make the session directory, start the camera and IMU processes, stop them, write
done.json and check.json. Stops on stop_event, on the session's time cap, or when a recorder process
dies.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import socket
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


def run_session(cfg: Config, mode: str, stop_event, session_dir: Path | None = None) -> dict:
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
    stop_reason = "stopped"
    t0 = time.monotonic()
    try:
        if scfg.imu:
            rate, who = ctx.Value("i", 0), ctx.Value("i", 0)
            p = ctx.Process(
                target=_proc_main,
                name="imu",
                args=(
                    run_imu,
                    (cfg.imu, session / "imu", child_stop, rate, who),
                    session / "log.txt",
                ),
            )
            p.start()
            procs.append(("imu", p))
        if scfg.cameras:
            exposure, gain = meter_exposure(cfg.cameras)
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
        while True:
            if stop_event.wait(0.5):
                break
            if time.monotonic() - t0 >= cap_s:
                stop_reason = f"time cap ({scfg.max_minutes} min)"
                break
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
