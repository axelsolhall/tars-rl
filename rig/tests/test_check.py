"""The session check on a synthetic recording with known faults: it must find exactly what was put
in."""

import csv
import json

import numpy as np
import pytest
from tars_rig import check
from tars_rig.contracts import FRAMES_COLUMNS, IMU_COLUMNS, READS_COLUMNS, TEMP_COLUMNS

FPS, RATE = 30, 200


def write_frames(path, t_ns, encoder_drops=(), sync_lock_at=0):
    path.mkdir(parents=True)
    with open(path / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAMES_COLUMNS)
        for i, t in enumerate(t_ns):
            stored = [-1, 0, 0] if i in encoder_drops else [i * 200_000, 200_000, 5000]
            ready = int(i >= sync_lock_at)
            w.writerow([i, int(t), int(t) + 8_000_000, 6000, 4.0, 33333, ready, *stored])


@pytest.fixture
def session(tmp_path):
    rng = np.random.default_rng(0)
    s = tmp_path / "20261109-120000-capture"
    period = 1e9 / FPS
    t0 = 1_000_000_000 + np.arange(300) * period
    t0 = np.delete(t0, [100, 200, 201])  # 3 dropped frames on the left camera
    t1 = (
        1_000_000_000 + np.arange(300) * period + 150_000 + rng.normal(0, 20_000, 300)
    )  # right 150 us late
    write_frames(s / "cam0", t0)
    write_frames(s / "cam1", t1, encoder_drops={50, 51})

    imu = s / "imu"
    imu.mkdir()
    real_period = 1e9 / (RATE * (1 + 500e-6))  # chip runs 500 ppm fast
    acc_lsb, gyro_lsb = 8192.0, 65.5
    with open(imu / "imu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(IMU_COLUMNS)
        for k in range(RATE * 12):
            w.writerow(
                [k, int(0.9e9 + k * 1e9 / RATE), 0, 0, int(acc_lsb), int(0.5 * gyro_lsb), 0, 0]
            )
    with open(imu / "reads.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(READS_COLUMNS)
        for r in range(1, 12 * 50):  # a read every 20 ms, 0.2-2 ms late
            total = r * 4
            w.writerow([int(0.9e9 + total * real_period + rng.uniform(0.2e6, 2e6)), total, 48, 0])
    with open(imu / "temp.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TEMP_COLUMNS)
        w.writerow([1e9, 31.5])
        w.writerow([2e9, 31.9])
    (s / "meta.json").write_text(
        json.dumps(
            {
                "mode": "capture",
                "cameras": [{"fps": FPS}, {"fps": FPS}],
                "imu": {"rate_hz": RATE, "accel_lsb_per_g": acc_lsb, "gyro_lsb_per_dps": gyro_lsb},
            }
        )
    )
    (s / "done.json").write_text(json.dumps({"stop_reason": "stopped", "errors": []}))
    return s


def test_cameras(session):
    r = check.check_session(session)
    assert r["cam0"]["dropped_frames"] == 3
    assert r["cam1"]["dropped_frames"] == 0
    assert r["cam1"]["fps_measured"] == pytest.approx(FPS, rel=1e-3)
    assert r["cam0"]["host_latency_ms_median"] == pytest.approx(8.0)
    # Encoder drops still have sensor timestamps, so they are not sensor drops.
    assert r["cam1"]["encoder_dropped"] == 2 and r["cam0"]["encoder_dropped"] == 0
    assert r["cam0"]["encode_ms_p50"] == 5.0
    assert r["cam0"]["write_mb_per_s"] == pytest.approx(297 * 0.2 / 9.97, rel=0.01)


def test_imu_sample_times_survive_an_overflow():
    period = 5_000_000.0
    # 50 reads of 4 samples, then an overflow loses 10 samples' worth of time, then 50 more reads.
    host, total, over = [], [], []
    for r in range(1, 51):
        host.append(r * 4 * period + 300_000), total.append(r * 4), over.append(0)
    host.append(host[-1] + 1e6), total.append(200), over.append(1)
    for r in range(1, 51):
        (
            host.append((200 + 10 + r * 4) * period + 300_000),
            total.append(200 + r * 4),
            over.append(0),
        )
    reads = {
        k: np.array(v, float)
        for k, v in (("host_ns", host), ("samples_total", total), ("overflow", over))
    }
    segs = check.imu_clock_segments(reads)
    assert [(s["k0"], s["k1"]) for s in segs] == [(0, 200), (200, 400)]
    t = check.imu_sample_times(np.array([0, 199, 200, 399]), segs)
    # A constant read latency (0.3 ms here) cannot be told apart from the clock; camera-IMU
    # calibration estimates the remaining time offset.
    expected = np.array([1, 200, 211, 410]) * period + 300_000
    assert t == pytest.approx(expected, abs=1)


def test_stereo_offset(session):
    s = check.check_session(session)["stereo"]
    assert s["pairs"] == 297
    assert s["offset_us_median"] == pytest.approx(150, abs=10)
    assert s["abs_offset_us_p95"] < 200


def test_stereo_offset_ignores_frames_before_sync_lock(tmp_path):
    # As on the first rig: free-running 14.5 ms apart until the sync locks at frame 97.
    t0 = 1e9 + np.arange(300) * 1e9 / FPS
    t1 = t0 + np.where(np.arange(300) < 97, 14_500_000, 30_000)
    write_frames(tmp_path / "cam0", t0, sync_lock_at=97)
    write_frames(tmp_path / "cam1", t1, sync_lock_at=97)
    s = check.check_session(tmp_path)["stereo"]
    assert s["after_sync_lock"] and s["left_frames_before_lock"] == 97
    assert s["pairs"] == 203 and s["abs_offset_us_max"] == pytest.approx(30, abs=1)


def test_imu_clock_and_still_numbers(session):
    m = check.check_session(session)["imu"]
    assert m["drift_ppm_vs_nominal"] == pytest.approx(500, abs=20)
    assert m["accel_norm_g_mean"] == pytest.approx(1.0, abs=1e-3)
    assert m["gyro_dps_mean"][0] == pytest.approx(int(0.5 * 65.5) / 65.5, abs=1e-3)
    assert m["overflows"] == 0
    assert m["temp_c_range"] == [31.5, 31.9]


def test_summary_line(session):
    line = check.summary_line(check.check_session(session))
    assert "3 dropped" in line and "L-R offset" in line and "imu" in line
