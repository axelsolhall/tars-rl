import numpy as np
from scipy.spatial.transform import Rotation
from tars_scan.slam_legs import legs, stops


def _walk():
    """Three board stops 10 s apart (1 s of frames each) on a random camera path."""
    rng = np.random.default_rng(0)
    t = np.concatenate([np.arange(30) * 33e6 + s * 10e9 for s in range(3)])
    R = Rotation.from_rotvec(rng.normal(0, 0.3, (len(t), 3))).as_matrix()
    p = rng.normal(0, 1, (len(t), 3))
    return t, R, p


def test_stops_split_at_gaps():
    t, _, _ = _walk()
    assert [len(r) for r in stops(t)] == [30, 30, 30]


def test_rigidly_moved_slam_has_no_leg_error():
    t, R, p = _walk()
    Rw = Rotation.from_rotvec([0.2, -0.4, 1.0]).as_matrix()  # SLAM world = board frame moved
    r = legs(t, R, p, t, Rw @ R, p @ Rw.T + [1.0, 2.0, 3.0])
    assert r["stops"] == 3 and len(r["legs"]) == 2 and len(r["returns"]) == 2
    assert r["leg_err_mm_median"] < 1e-6 and r["leg_err_deg_median"] < 1e-6


def test_scale_error_shows_in_the_legs():
    t, R, p = _walk()
    r = legs(t, R, p, t, R, p * 1.1)
    assert r["leg_err_mm_median"] > 10
