"""Camera recorder: one process per IMX219, grayscale JPEGs into frames.mjpeg, plus frames.csv.

Both cameras run the same sensor mode, frame rate, exposure and gain. With sync on, camera 0 is
libcamera's sync server and camera 1 its client (Raspberry Pi software camera sync); the client must
start first. Whether sync really lines the frames up is measured afterwards by tars_rig.check.

Capture and encoding run in separate threads joined by a bounded queue: a slow JPEG encode or an SD
card write stall then costs at most a counted, logged drop (offset -1 in frames.csv) instead of
silently stalling libcamera's buffers, and the camera keeps its frame timing.
"""

from __future__ import annotations

import csv
import json
import logging
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from .config import CameraConfig
from .contracts import FRAMES_COLUMNS, FRAMES_FILE

log = logging.getLogger("rig.cameras")
LENS = "Waveshare IMX219, 120 deg FOV (fisheye-type model needed)"


def _video_config(picam2, cfg: CameraConfig, controls: dict):
    return picam2.create_video_configuration(
        main={"size": (cfg.width, cfg.height), "format": "YUV420"},
        sensor={"output_size": (cfg.width, cfg.height)},
        controls=controls,
        buffer_count=8,
    )


def meter_exposure(cfg: CameraConfig, max_exposure_us: int | None = None) -> tuple[int, float]:
    """Run auto-exposure on camera 0 for meter_seconds, then cap exposure with gain.

    The cap limits motion blur and rolling-shutter smear."""
    from picamera2 import Picamera2

    period_us = round(1e6 / cfg.fps)
    with Picamera2(0) as picam2:
        picam2.configure(
            _video_config(picam2, cfg, {"FrameDurationLimits": (period_us, period_us)})
        )
        picam2.start()
        time.sleep(cfg.meter_seconds)
        md = picam2.capture_metadata()
        picam2.stop()
    exposure, gain = int(md["ExposureTime"]), float(md["AnalogueGain"])
    cap = max_exposure_us or cfg.max_exposure_us
    if exposure > cap:
        gain, exposure = gain * exposure / cap, cap
    gain = min(gain, cfg.max_gain)
    log.info("metered exposure %d us, gain %.2f", exposure, gain)
    return exposure, gain


def sync_mode(role: str):
    """libcamera's SyncMode enum for this role, or None if this libcamera has no software sync."""
    if role == "none":
        return None
    try:
        from libcamera import controls

        enum = controls.rpi.SyncModeEnum
    except (ImportError, AttributeError):
        return None
    return enum.Server if role == "server" else enum.Client


def parse_v4l2_ctrls(text: str) -> dict[str, int]:
    """`v4l2-ctl --get-ctrl a,b` output ("a: 123" per line) as {name: value}."""
    return {m[1]: int(m[2]) for m in re.finditer(r"^\s*(\w+):\s*(-?\d+)\s*$", text, re.M)}


def line_timing(width: int, height: int, ctrls: dict[str, int]) -> dict:
    """Rolling-shutter timing from the sensor's V4L2 controls: one line takes (active width +
    horizontal blanking) pixel clocks; the frame's rows are read one line time apart (A155)."""
    rate, hblank = ctrls.get("pixel_rate"), ctrls.get("horizontal_blanking")
    out: dict = {"ctrls": ctrls, "width": width, "height": height}
    if rate and hblank is not None:
        line_ns = (width + hblank) / rate * 1e9
        out |= {"line_time_ns": round(line_ns, 2), "readout_us": round(line_ns * height / 1e3, 1)}
    return out


def read_sensor_timing(width: int, height: int) -> dict:
    """The IMX219 subdevice's line timing; both cameras run the same mode, so either sensor
    answers for both. Must be called while streaming (blanking depends on the configured mode)."""
    for d in sorted(Path("/sys/class/video4linux").glob("v4l-subdev*")):
        try:
            if not (d / "name").read_text().startswith("imx219"):
                continue
            out = subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    f"/dev/{d.name}",
                    "--get-ctrl",
                    "pixel_rate,horizontal_blanking,vertical_blanking",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.SubprocessError) as e:
            return {"error": repr(e)}
        return line_timing(width, height, parse_v4l2_ctrls(out))
    return {"error": "no imx219 v4l-subdev found"}


class FrameWriter:
    """Encodes queued luma planes to JPEG and appends them to frames.mjpeg; owns frames.csv."""

    def __init__(self, out_dir: Path, quality: int, max_queue: int):
        self.f_jpg = open(out_dir / FRAMES_FILE, "wb")
        self.f_csv = open(out_dir / "frames.csv", "w", newline="")
        self.csv = csv.writer(self.f_csv)
        self.csv.writerow(FRAMES_COLUMNS)
        self.quality, self.lock = quality, threading.Lock()
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, name="jpeg", daemon=True)
        self.thread.start()

    def submit(self, row: list, luma) -> None:
        """row = the FRAMES_COLUMNS values up to sync_ready; the writer adds offset, size, time."""
        try:
            self.q.put_nowait((row, luma))
        except queue.Full:
            self.dropped += 1
            with self.lock:
                self.csv.writerow([*row, -1, 0, 0])

    def _run(self) -> None:
        import simplejpeg

        n = 0
        while (item := self.q.get()) is not None:
            row, luma = item
            t = time.perf_counter_ns()
            jpg = simplejpeg.encode_jpeg(luma[..., None], quality=self.quality, colorspace="GRAY")
            enc_us = (time.perf_counter_ns() - t) // 1000
            offset = self.f_jpg.tell()
            self.f_jpg.write(jpg)
            with self.lock:
                self.csv.writerow([*row, offset, len(jpg), enc_us])
            n += 1
            if n % 30 == 0:
                self.f_jpg.flush()
                self.f_csv.flush()

    def close(self) -> None:
        self.q.put(None)
        self.thread.join()
        self.f_jpg.close()
        self.f_csv.close()


def run_camera(
    index: int,
    role: str,
    cfg: CameraConfig,
    exposure_us: int,
    gain: float,
    out_dir: Path,
    stop_event,
    started_event=None,
) -> None:
    from picamera2 import Picamera2

    out_dir.mkdir(parents=True, exist_ok=True)
    period_us = round(1e6 / cfg.fps)
    controls = {
        "FrameDurationLimits": (period_us, period_us),
        "AeEnable": False,
        "ExposureTime": exposure_us,
        "AnalogueGain": gain,
    }
    mode = sync_mode(role)
    if mode is not None:
        controls["SyncMode"] = mode
    elif role != "none":
        log.warning("cam%d: libcamera has no SyncMode; recording unsynced", index)
    h, w = cfg.height, cfg.width
    writer = FrameWriter(out_dir, cfg.jpeg_quality, cfg.queue_frames)
    i = 0
    try:
        with Picamera2(index) as picam2:
            picam2.configure(_video_config(picam2, cfg, controls))
            picam2.start()
            (out_dir / "sensor.json").write_text(json.dumps(read_sensor_timing(w, h), indent=2))
            if started_event is not None:
                started_event.set()
            while not stop_event.is_set():
                request = picam2.capture_request()
                try:
                    md = request.get_metadata()
                    host_ns = time.monotonic_ns()
                    # The Y plane; copied so the buffer goes back to libcamera at once.
                    luma = request.make_array("main")[:h, :w].copy()
                finally:
                    request.release()
                writer.submit(
                    [
                        i,
                        md.get("SensorTimestamp", ""),
                        host_ns,
                        md.get("ExposureTime", ""),
                        round(md.get("AnalogueGain", 0.0), 3),
                        md.get("FrameDuration", ""),
                        int(bool(md.get("SyncReady", False))),
                    ],
                    luma,
                )
                i += 1
            picam2.stop()
    finally:
        writer.close()
    log.info("cam%d stopped after %d frames (%d dropped by the encoder)", index, i, writer.dropped)
