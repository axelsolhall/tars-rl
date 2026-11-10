"""The rig's main loop: the button (or `rig start` / `rig stop`) starts and stops the armed session;
the LED shows state.

Files in <root>/state, shared with the `rig` CLI (a separate process, via `docker exec`):
  armed.json    {"mode": "calibrate" | "imu" | "capture"}; written by `rig arm`, kept
  request.json  {"action": "start" | "stop" | "press"}; written by the CLI, consumed here
  status.json   what the daemon is doing; written here, read by `rig status`
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import MODES, Config
from .contracts import write_json

log = logging.getLogger("rig.daemon")


def read_armed(state_dir: Path) -> str | None:
    try:
        mode = json.loads((state_dir / "armed.json").read_text()).get("mode")
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return mode if mode in MODES else None


def write_armed(state_dir: Path, mode: str | None) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    if mode is None:
        (state_dir / "armed.json").unlink(missing_ok=True)
    else:
        write_json(
            state_dir / "armed.json", {"mode": mode, "armed_utc": datetime.now(UTC).isoformat()}
        )


def write_request(state_dir: Path, action: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    write_json(state_dir / "request.json", {"action": action})


class Rig:
    def __init__(self, cfg: Config, leds, runner):
        self.cfg, self.leds, self.runner = cfg, leds, runner
        self.events: queue.Queue[str] = queue.Queue()
        self.session_stop: threading.Event | None = None
        self.thread: threading.Thread | None = None
        self.mode: str | None = None
        self.started: float | None = None
        self.last: dict | None = None
        self.error = False
        self.shown_armed: str | None = (
            None  # the armed mode the LED last showed, to notice `rig arm` changes
        )
        cfg.state_dir.mkdir(parents=True, exist_ok=True)

    # --- inputs -----------------------------------------------------------------------------------
    def press(self) -> None:
        """Called from the button's callback thread; the main loop does the work."""
        self.events.put("press")

    @property
    def recording(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def handle(self, action: str) -> None:
        if action == "phase":
            if self.recording:
                self.leds.show("phase")
                self.leds.show("recording")
            return
        if action == "stop" or (action == "press" and self.recording):
            if self.recording:
                log.info("stop requested")
                self.session_stop.set()
            return
        if self.recording:
            return
        mode = read_armed(self.cfg.state_dir)
        if mode is None:
            log.info("pressed with nothing armed")
            self.leds.show("refused")
            self.show_idle()
            return
        self.start(mode)

    def start(self, mode: str) -> None:
        self.error, self.mode, self.started = False, mode, time.time()
        self.session_stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run, args=(mode, self.session_stop), daemon=True
        )
        self.leds.show("recording")
        log.info("starting %s", mode)
        self.thread.start()
        self.write_status()

    def _run(self, mode: str, stop: threading.Event) -> None:
        try:
            self.last = self.runner(
                self.cfg, mode, stop, on_phase=lambda name: self.events.put("phase")
            )
        except Exception as e:  # noqa: BLE001
            log.exception("session crashed")
            self.last = {"stop_reason": "crashed", "errors": [repr(e)]}
        self.error = bool(self.last.get("errors"))
        self.events.put("finished")

    # --- outputs ----------------------------------------------------------------------------------
    def show_idle(self) -> None:
        self.shown_armed = read_armed(self.cfg.state_dir)
        if self.error:
            self.leds.show("error")
        else:
            self.leds.show("armed" if read_armed(self.cfg.state_dir) else "off")

    def write_status(self) -> None:
        status = {
            "state": "recording" if self.recording else "idle",
            "armed": read_armed(self.cfg.state_dir),
            "mode": self.mode if self.recording else None,
            "elapsed_s": round(time.time() - self.started, 1) if self.recording else None,
            "last": self.last,
            "updated_utc": datetime.now(UTC).isoformat(),
        }
        write_json(self.cfg.state_dir / "status.json", status)

    # --- loop -------------------------------------------------------------------------------------
    def poll_requests(self) -> None:
        path = self.cfg.state_dir / "request.json"
        if path.exists():
            try:
                action = json.loads(path.read_text()).get("action", "")
            except json.JSONDecodeError:
                action = ""
            path.unlink(missing_ok=True)
            if action in ("start", "stop", "press"):
                self.events.put(action)

    def step(self, timeout: float = 0.5) -> None:
        """One loop turn: wait briefly for a button, CLI request or session end, and act on it."""
        self.poll_requests()
        try:
            event = self.events.get(timeout=timeout)
        except queue.Empty:
            event = None
        if event == "finished":
            log.info("session finished: %s", self.last)
            self.show_idle()
        elif event is not None:
            self.handle(event)
        elif not self.recording and read_armed(self.cfg.state_dir) != self.shown_armed:
            self.error = False  # re-arming clears a shown error
            self.show_idle()
        self.write_status()

    def run_forever(self, stop: threading.Event) -> None:
        self.show_idle()
        while not stop.is_set():
            self.step()
        if self.recording:
            self.session_stop.set()
            self.thread.join(30)
        self.leds.show("off")
