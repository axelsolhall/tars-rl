"""IMU recorder: drains the MPU6050 FIFO every poll_ms and writes imu.csv, reads.csv and temp.csv.

Runs as its own process; stops when stop_event is set. Timestamps: each FIFO read logs the host
clock and the running sample count (reads.csv), so the chip's clock drift can be fitted afterwards
(tars_rig.check.fit_imu_clock).
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from .config import ImuConfig
from .contracts import IMU_COLUMNS, READS_COLUMNS, TEMP_COLUMNS
from .mpu6050 import FIFO_SIZE, SAMPLE_BYTES, Mpu6050, chip_name

log = logging.getLogger("rig.imu")
MAX_READ = (FIFO_SIZE // SAMPLE_BYTES) * SAMPLE_BYTES


def run_imu(
    cfg: ImuConfig, out_dir: Path, stop_event, real_rate_out=None, who_out=None, flush=False
) -> None:
    """flush: write every read to disk at once (a live session reads the files as they grow)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chip = Mpu6050(cfg.bus, cfg.address)
    log.info("IMU chip %s (WHO_AM_I 0x%02x)", chip_name(chip.who), chip.who)
    if who_out is not None:
        who_out.value = chip.who
    rate = chip.setup(cfg.rate_hz, cfg.accel_range_g, cfg.gyro_range_dps, cfg.dlpf)
    if real_rate_out is not None:
        real_rate_out.value = rate
    period_ns = 1_000_000_000 // rate
    log.info("IMU at %d Hz (asked %d)", rate, cfg.rate_hz)

    with (
        open(out_dir / "imu.csv", "w", newline="") as f_imu,
        open(out_dir / "reads.csv", "w", newline="") as f_reads,
        open(out_dir / "temp.csv", "w", newline="") as f_temp,
    ):
        w_imu, w_reads, w_temp = csv.writer(f_imu), csv.writer(f_reads), csv.writer(f_temp)
        w_imu.writerow(IMU_COLUMNS)
        w_reads.writerow(READS_COLUMNS)
        w_temp.writerow(TEMP_COLUMNS)
        k, anchor_ns, anchor_k, next_temp = 0, None, 0, 0
        while not stop_event.is_set():
            time.sleep(cfg.poll_ms / 1000)
            overflow = chip.overflowed()
            if overflow:
                # Samples were lost and the count restarts: re-anchor at the next read.
                log.warning("IMU FIFO overflow at sample %d; resetting", k)
                chip.reset_fifo()
                w_reads.writerow([time.monotonic_ns(), k, 0, 1])
                anchor_ns = None
                continue
            n = min(chip.fifo_count(), MAX_READ) // SAMPLE_BYTES * SAMPLE_BYTES
            if n == 0:
                continue
            data = chip.read_fifo(n)
            now = time.monotonic_ns()
            samples = [data[i : i + SAMPLE_BYTES] for i in range(0, n, SAMPLE_BYTES)]
            if anchor_ns is None:  # the newest sample in this read happened just before `now`
                anchor_ns, anchor_k = now, k + len(samples) - 1
            for s in samples:
                vals = [
                    int.from_bytes(s[j : j + 2], "big", signed=True)
                    for j in range(0, SAMPLE_BYTES, 2)
                ]
                w_imu.writerow([k, anchor_ns + (k - anchor_k) * period_ns, *vals])
                k += 1
            w_reads.writerow([now, k, n, 0])
            if flush:
                f_imu.flush()
                f_reads.flush()
            if now >= next_temp:
                w_temp.writerow([now, round(chip.temperature(), 2)])
                next_temp = now + 1_000_000_000
                f_imu.flush()
    log.info("IMU stopped after %d samples", k)
