import numpy as np
from tars_scan.live_map import CsvTail, ImuClock


def test_csv_tail_returns_complete_rows_only_and_in_chunks(tmp_path):
    p = tmp_path / "x.csv"
    t = CsvTail(p, max_rows=2)
    assert t.rows() == []  # not created yet
    p.write_text("a,b\n1,2\n3,4\n5,")
    assert t.rows() == [["1", "2"], ["3", "4"]]
    assert t.rows() == []  # "5," is still being written
    with open(p, "a") as f:
        f.write("6\n7,8\n")
    assert t.rows() == [["5", "6"], ["7", "8"]]


def test_imu_clock_finds_the_real_period_and_the_lower_envelope():
    rng = np.random.default_rng(0)
    period, t0 = 5_012_000.0, 1e12  # the chip runs 0.24 % slow against its nominal 200 Hz
    c = ImuClock()
    for i in range(200):
        total = 4 * (i + 1)  # four samples per read
        newest = t0 + period * total  # time of sample k = total - 1 is period * (k + 1) + t0
        c.add(newest + rng.uniform(0, 2e6), total, False)  # reads land up to 2 ms late
    k = np.array([0, 400, 799])
    assert np.abs(c.time(k) - (t0 + period * (k + 1))).max() < 0.3e6  # under 0.3 ms
    c.add(0, 0, True)  # FIFO overflow: the count restarts, so the fit does too
    assert c.time(k) is None
