"""What a session directory holds (schema version 2). Every downstream step (export, calibration,
SLAM) reads only this.

<session>/
  meta.json            SessionMeta: mode, settings, per-camera exposure, sync role and sensor
                       timing (rolling-shutter line time), IMU scale factors
  done.json            how it ended: stop reason, counts, errors
  check.json           the eval of this session (tars_rig.check)
  phases.json          scripted sessions only: [{name, start_ns, end_ns}], the phases as recorded
  log.txt
  cam0/ cam1/
    frames.mjpeg       every frame as a JPEG (grayscale, or YCbCr 4:2:0 when meta "color"), back
                       to back (one file, not one per frame: far fewer SD-card metadata writes,
                       and one file to copy); ffplay plays it
    frames.csv         one row per frame: timestamps, exposure, and where its JPEG sits in
                       frames.mjpeg (offset -1: dropped because the encoder fell behind)
    sensor.json        the sensor's line timing, read from its V4L2 controls while streaming
  imu/imu.csv          one row per IMU sample, raw int16 counts
  imu/reads.csv        one row per FIFO read: host clock and running sample count
                       (to refit the IMU clock)
  imu/temp.csv         die temperature about once a second

Clocks: every *_ns column is CLOCK_MONOTONIC on the Pi (libcamera's SensorTimestamp uses the same
clock), so cameras and IMU share one time base. sensor_ns is libcamera's SensorTimestamp, the start
of the frame; tars_scan.export moves it to mid-exposure of the middle row for SLAM (A154).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 2

FRAMES_COLUMNS = [
    "index",
    "sensor_ns",
    "host_ns",
    "exposure_us",
    "gain",
    "frame_duration_us",
    "sync_ready",
    "offset",
    "size",
    "encode_us",
]
IMU_COLUMNS = ["k", "t_ns", "ax", "ay", "az", "gx", "gy", "gz"]
READS_COLUMNS = ["host_ns", "samples_total", "fifo_bytes", "overflow"]
TEMP_COLUMNS = ["host_ns", "temp_c"]
FRAMES_FILE = "frames.mjpeg"


@dataclass
class CameraMeta:
    index: int
    width: int
    height: int
    fps: float
    exposure_us: int
    gain: float
    sync_role: str  # "server", "client" or "none"
    sync_supported: bool
    jpeg_quality: int
    # Rolling shutter: row r starts exposing line_time_ns * r after row 0. None if unreadable.
    line_time_ns: float | None = None
    readout_us: float | None = None
    color: bool = False  # frames are colour JPEGs (decode with colorspace="GRAY" for luma)


@dataclass
class ImuMeta:
    chip: str
    rate_hz: int
    accel_lsb_per_g: float
    gyro_lsb_per_dps: float
    dlpf: int
    timestamps: str = (
        "t_ns = nominal rate anchored at the first read; tars_rig.export refits from reads.csv"
    )


@dataclass
class SessionMeta:
    mode: str
    started_utc: str
    host: str
    max_minutes: float
    cameras: list[CameraMeta] = field(default_factory=list)
    imu: ImuMeta | None = None
    settings: dict = field(default_factory=dict)
    schema: int = SCHEMA_VERSION


def write_json(path: Path, obj) -> None:
    data = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def iter_jpegs(cam_dir: Path, rows: Iterable[tuple[int, int, int]]) -> Iterator[tuple[int, bytes]]:
    """(index, jpeg bytes) for each (index, offset, size) row that was written (offset >= 0)."""
    with open(cam_dir / FRAMES_FILE, "rb") as f:
        for index, offset, size in rows:
            if offset < 0:
                continue
            f.seek(offset)
            yield index, f.read(size)
