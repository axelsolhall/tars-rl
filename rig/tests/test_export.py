"""The camera writer and the EuRoC export on a small real session: frames written by FrameWriter
(the code the Pi runs), then exported; timestamps, pairing, units and pixels must come out right."""

import csv
import json

import numpy as np
import pytest
from tars_rig.cameras import FrameWriter, line_timing, parse_v4l2_ctrls
from tars_rig.contracts import IMU_COLUMNS, READS_COLUMNS, iter_jpegs
from tars_scan import export

pytest.importorskip("cv2")
pytest.importorskip("simplejpeg")
FPS, N, RATE = 30, 20, 200
PERIOD = 1e9 / FPS


def frame(i):
    y, x = np.mgrid[0:96, 0:128]
    return ((x * 2 + y + i * 7) % 256).astype(np.uint8)


@pytest.fixture
def session(tmp_path):
    s = tmp_path / "s"
    for cam, lag_ns in ((0, 0), (1, 40_000)):
        d = s / f"cam{cam}"
        d.mkdir(parents=True)
        w = FrameWriter(d, quality=95, max_queue=100)
        for i in range(N):
            t = 1_000_000_000 + i * PERIOD + lag_ns
            w.submit([i, int(t), int(t) + 5_000_000, 4000, 2.0, 33333, 1], frame(i))
        w.close()
    imu = s / "imu"
    imu.mkdir()
    with open(imu / "imu.csv", "w", newline="") as f:
        c = csv.writer(f)
        c.writerow(IMU_COLUMNS)
        for k in range(RATE):
            c.writerow([k, 0, 0, 0, 8192, 655, 0, 0])  # 1 g on z, 10 dps on x
    with open(imu / "reads.csv", "w", newline="") as f:
        c = csv.writer(f)
        c.writerow(READS_COLUMNS)
        for r in range(1, RATE // 4 + 1):
            c.writerow([900_000_000 + r * 4 * 5_000_000, r * 4, 48, 0])
    meta = {
        "cameras": [{"fps": FPS, "readout_us": 23_000.0}, {"fps": FPS, "readout_us": 23_000.0}],
        "imu": {"rate_hz": RATE, "accel_lsb_per_g": 8192.0, "gyro_lsb_per_dps": 65.5},
    }
    (s / "meta.json").write_text(json.dumps(meta))
    return s


def test_writer_round_trip(session):
    from tars_rig.check import read_csv

    f = read_csv(session / "cam0" / "frames.csv")
    rows = zip(f["index"].astype(int), f["offset"].astype(int), f["size"].astype(int), strict=True)
    import simplejpeg

    got = {
        i: simplejpeg.decode_jpeg(j, colorspace="GRAY")[..., 0]
        for i, j in iter_jpegs(session / "cam0", rows)
    }
    assert sorted(got) == list(range(N))
    assert np.abs(got[7].astype(int) - frame(7)).mean() < 2  # JPEG at q95


def test_export(session, tmp_path):
    out = tmp_path / "euroc"
    r = export.export_session(session, out)
    assert r["stereo"]["pairs"] == N
    assert r["stereo"]["offset_us_median"] == pytest.approx(40.0)
    lines = (out / "mav0" / "cam0" / "data.csv").read_text().splitlines()[1:]
    t0 = int(lines[0].split(",")[0])
    # mid-exposure of the middle row: start - exposure/2 + readout/2
    assert t0 == 1_000_000_000 - 2_000_000 + 11_500_000
    assert (out / "mav0" / "cam1" / "data.csv").read_text() == (
        out / "mav0" / "cam0" / "data.csv"
    ).read_text()
    assert len(list((out / "mav0" / "cam0" / "data").glob("*.png"))) == N
    imu = (out / "mav0" / "imu0" / "data.csv").read_text().splitlines()
    first = [float(v) for v in imu[1].split(",")]
    assert first[0] == 905_000_000  # refitted clock: sample 0 is ready one period after the anchor
    assert first[1] == pytest.approx(np.deg2rad(10.0), rel=1e-3)
    assert first[6] == pytest.approx(9.80665, rel=1e-4)


def test_export_half_scale(session, tmp_path):
    import cv2

    export.export_session(session, tmp_path / "half", scale=0.5)
    png = next((tmp_path / "half" / "mav0" / "cam0" / "data").glob("*.png"))
    assert cv2.imread(str(png), cv2.IMREAD_GRAYSCALE).shape == (48, 64)


def test_line_timing():
    ctrls = parse_v4l2_ctrls(
        "pixel_rate: 182400000\nhorizontal_blanking: 1808\nvertical_blanking: 531\n"
    )
    t = line_timing(1640, 1232, ctrls)
    assert t["line_time_ns"] == pytest.approx((1640 + 1808) / 182.4e6 * 1e9, rel=1e-6)
    assert t["readout_us"] == pytest.approx(t["line_time_ns"] * 1232 / 1e3, rel=1e-3)
