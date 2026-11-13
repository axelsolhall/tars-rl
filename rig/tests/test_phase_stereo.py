import numpy as np
from tars_scan.phase_stereo import PhaseParams, _shift, phase_disparity, phase_step


def _texture(seed=0, shape=(120, 200)):
    import cv2

    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.uniform(0, 255, shape).astype(np.float32), (0, 0), 1.2)


def test_phase_step_reads_a_subpixel_shift():
    left = _texture()
    right = _shift(left, np.full(left.shape, -0.4, np.float32))  # R(x) = L(x + 0.4): d = 0.4
    d, ok = phase_step(left, right, PhaseParams())
    inner = (slice(20, -20), slice(20, -20))
    assert ok[inner].mean() > 0.6
    assert abs(np.median(d[inner][ok[inner]]) - 0.4) < 0.05


def test_coarse_to_fine_reaches_large_disparities():
    left = _texture(1)
    right = _shift(left, np.full(left.shape, -13.3, np.float32))  # d = 13.3 px
    d, ok = phase_disparity(left, right, PhaseParams())
    inner = (slice(25, -25), slice(40, -40))
    assert ok[inner].mean() > 0.5
    assert abs(np.median(d[inner][ok[inner]]) - 13.3) < 0.2
