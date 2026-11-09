"""tars_scan.calib end to end on rendered images: a checkerboard drawn through two known fisheye
lenses 60 mm apart, written by the Pi's FrameWriter, then searched, selected and calibrated. The
lens and the baseline must come back."""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("simplejpeg")
from scipy.spatial.transform import Rotation  # noqa: E402
from tars_rig.cameras import FrameWriter  # noqa: E402
from tars_scan import calib  # noqa: E402

W, H, SQ, BOARD = 820, 616, 0.03, (9, 6)
K0 = np.array([[391.0, 0, 412.0], [0, 392.0, 306.0], [0, 0, 1]])
K1 = np.array([[389.5, 0, 408.0], [0, 390.0, 309.5], [0, 0, 1]])
LENSES = [
    (K0, np.array([0.03, -0.01, 0.004, -0.001])),
    (K1, np.array([0.025, -0.008, 0.003, -0.001])),
]
BASELINE = np.array([0.06, 0.0, 0.0])


def render(K, D, R, t, scale=2):
    """Board-to-camera pose (R, t); 2x supersampled, then averaged down."""
    u, v = np.meshgrid(
        (np.arange(W * scale) + 0.5) / scale - 0.5, (np.arange(H * scale) + 0.5) / scale - 0.5
    )
    rays = cv2.fisheye.undistortPoints(
        np.c_[u.ravel(), v.ravel()].reshape(-1, 1, 2), K, D.reshape(4, 1)
    )
    d = np.c_[rays.reshape(-1, 2), np.ones(u.size)] @ R  # = (R^T d^T)^T: rays in the board frame
    c = -R.T @ t
    lam = -c[2] / d[:, 2]
    p = c + lam[:, None] * d
    ix, iy = np.floor(p[:, 0] / SQ).astype(int) + 1, np.floor(p[:, 1] / SQ).astype(int) + 1
    on = (lam > 0) & (ix >= 0) & (ix < BOARD[0] + 1) & (iy >= 0) & (iy < BOARD[1] + 1)
    margin = (
        (lam > 0)
        & (p[:, 0] > -2 * SQ)
        & (p[:, 0] < (BOARD[0] + 1) * SQ)
        & (p[:, 1] > -2 * SQ)
        & (p[:, 1] < (BOARD[1] + 1) * SQ)
    )
    img = np.where(margin, 235, 110).astype(float)
    img[on & ((ix + iy) % 2 == 0)] = 25
    img = img.reshape(H * scale, W * scale)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.uint8)


def poses(n, rng):
    obj = calib.board_points(BOARD, SQ).reshape(1, -1, 3)
    out = []
    while len(out) < n:
        R = Rotation.from_euler("xyz", rng.uniform([-0.6, -0.6, -np.pi], [0.6, 0.6, np.pi]))
        t = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.12, 0.02), rng.uniform(0.25, 0.45)])
        Rm = R.as_matrix()
        ok = True
        for (K, D), shift in zip(LENSES, (np.zeros(3), -BASELINE), strict=True):
            img, _ = cv2.fisheye.projectPoints(obj, R.as_rotvec(), t + shift, K, D.reshape(4, 1))
            img = img.reshape(-1, 2)
            ok &= bool(
                (img.min(0) > 30).all() and img[:, 0].max() < W - 30 and img[:, 1].max() < H - 30
            )
        if ok:
            out.append((Rm, t))
    return out


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    s = tmp_path_factory.mktemp("calib") / "session"
    rng = np.random.default_rng(3)
    ps = poses(24, rng)
    for cam, ((K, D), shift) in enumerate(zip(LENSES, (np.zeros(3), -BASELINE), strict=True)):
        d = s / f"cam{cam}"
        d.mkdir(parents=True)
        w = FrameWriter(d, quality=95, max_queue=100)
        for i, (R, t) in enumerate(ps):
            ts = 1_000_000_000 + i * 33_333_333
            w.submit([i, ts, ts, 3000, 1.0, 33333, 1], render(K, D, R, t + shift))
        w.close()
    cams = [{"width": W, "height": H, "fps": 30, "readout_us": 0.0}] * 2
    (s / "meta.json").write_text(json.dumps({"cameras": cams}))
    return s


def test_calibration_recovers_lens_and_baseline(session):
    r = calib.calibrate_session(session, BOARD, SQ, budget=20, max_px=np.inf, stride=1)
    for cam, (K, _) in enumerate(LENSES):
        c = r["cameras"][cam]
        assert c["detected"] >= 20
        assert c["rms_px"] < 0.3
        assert np.array(c["K"])[[0, 1, 0, 1], [0, 1, 2, 2]] == pytest.approx(
            K[[0, 1, 0, 1], [0, 1, 2, 2]], abs=1.0
        )
    assert r["stereo"]["baseline_m"] == pytest.approx(0.06, abs=0.001)
    assert all(c["heldout_rms_px"] < 0.3 for c in r["cameras"])
    assert (session / "calib" / "camchain.yaml").read_text().count("T_cn_cnm1") == 1
