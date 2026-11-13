import numpy as np
from scipy.spatial.transform import Rotation
from tars_scan import anchor_poses as A

DRIFT_PER_S = np.array([0.01, -0.02, 0.0])


def _walk(drift_per_s=DRIFT_PER_S, yaw_drift_dps=0.2):
    """A camera walking in a circle at constant speed, back at the board every 20 s; the SLAM drifts
    linearly in position and yaw, so interpolating the anchors removes it exactly."""
    t = np.arange(0, 100, 0.05)
    ang = 2 * np.pi * t / 20
    p_true = np.stack([np.cos(ang) - 1, np.sin(ang), 0 * t], axis=1)
    R_true = Rotation.from_euler("z", ang[:, None]).as_matrix()
    drift = Rotation.from_euler("z", np.radians(yaw_drift_dps * t)[:, None]).as_matrix()
    p_es = np.einsum("nij,nj->ni", drift, p_true) + t[:, None] * drift_per_s
    R_es = np.einsum("nij,njk->nik", drift, R_true)
    board = (t % 20 < 1.5) | (t % 20 > 18.5)  # 1.5 s on each side of every return
    t_ns = (t * 1e9).astype(np.int64)
    return t_ns[board], R_true[board], p_true[board], t_ns, R_es, p_es


def test_anchors_remove_linear_drift_at_held_out_stops():
    t_gt, R_gt, p_gt, t_es, R_es, p_es = _walk(yaw_drift_dps=0.0)
    r = A.leave_one_out(t_gt, R_gt, p_gt, t_es, R_es, p_es, by="time")
    inner = [x["err_mm"] for x in r["rows"][:-1]]  # the last stop has no later anchor
    assert r["stops"] >= 5 and max(inner) < 5


def test_causal_anchoring_is_worse_than_interpolated_under_drift():
    t_gt, R_gt, p_gt, t_es, R_es, p_es = _walk()
    a = A.leave_one_out(t_gt, R_gt, p_gt, t_es, R_es, p_es, by="time")
    c = A.leave_one_out(t_gt, R_gt, p_gt, t_es, R_es, p_es, by="time", causal=True)
    assert a["err_mm_median"] < c["err_mm_median"]


def test_rotation_onto_z():
    rng = np.random.default_rng(1)
    for _ in range(5):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        assert np.allclose(A._rotation_onto_z(u) @ u, [0, 0, 1], atol=1e-9)
