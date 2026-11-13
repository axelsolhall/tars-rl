import numpy as np
from tars_scan.rs_correct import FisheyeRemap, LinearRemap


def test_linear_remap_matches_the_exact_rolling_shutter_remap():
    K = np.array([[135.0, 0, 205], [0, 135.0, 154], [0, 0, 1]])
    exact = FisheyeRemap(K, np.array([0.02, -0.01, 0.003, -0.001]), (410, 308))
    line = 12e-3 / 308
    lin = LinearRemap(exact, line)
    for w in ([1.5, -0.8, 0.4], [0.0, 2.5, 0.0]):  # fast turns, rad/s
        ex, ey = exact.maps(np.array(w), line)
        lx, ly = lin.maps(np.array(w))
        # past the lens model's valid field the exact projection itself is garbage (shifts of
        # hundreds of px); a real rolling-shutter shift here is a few px
        valid = np.hypot(ex - lin.m0[0], ey - lin.m0[1]) < 10
        assert valid.mean() > 0.9
        err = np.hypot(ex - lx, ey - ly)[valid]
        assert np.percentile(err, 99) < 0.05 and np.median(err) < 0.005  # pixels
