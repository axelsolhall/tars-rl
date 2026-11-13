"""Colour recording: YUV420 plane extraction from picamera2's padded layout and the JPEG round trip
(colour JPEG decodes to the same luma the grayscale path would give)."""

import numpy as np
import pytest

simplejpeg = pytest.importorskip("simplejpeg")

from tars_rig.cameras import encode, yuv420_planes  # noqa: E402


def _packed(y, u, v, stride):
    """picamera2's YUV420 array: Y rows of stride bytes, then U and V at stride/2, two per row."""
    h, w = y.shape
    buf = np.zeros(h * stride * 3 // 2, np.uint8)
    buf[: h * stride].reshape(h, stride)[:, :w] = y
    n_c = (h // 2) * (stride // 2)
    buf[h * stride : h * stride + n_c].reshape(h // 2, stride // 2)[:, : w // 2] = u
    buf[h * stride + n_c :].reshape(h // 2, stride // 2)[:, : w // 2] = v
    return buf.reshape(-1, stride)


def test_planes_from_padded_layout():
    rng = np.random.default_rng(0)
    h, w, stride = 1232, 1640, 1664
    y = rng.integers(0, 255, (h, w), np.uint8)
    u = rng.integers(0, 255, (h // 2, w // 2), np.uint8)
    v = rng.integers(0, 255, (h // 2, w // 2), np.uint8)
    got = yuv420_planes(_packed(y, u, v, stride), w, h)
    for a, b in zip(got, (y, u, v), strict=True):
        assert np.array_equal(a, b)


def test_colour_jpeg_luma_matches_gray_path():
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    y = (64 + 128 * (xx / w)).astype(np.uint8)
    u = np.full((h // 2, w // 2), 90, np.uint8)  # bluish-red tint, far from neutral 128
    v = np.full((h // 2, w // 2), 170, np.uint8)
    col = encode((y, u, v), 90)
    gray = simplejpeg.decode_jpeg(col, colorspace="GRAY")[..., 0]
    assert np.abs(gray.astype(int) - y).mean() < 1.5
    rgb = simplejpeg.decode_jpeg(col, colorspace="RGB")
    r, g, b = rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()
    assert r > g + 20 and r > b  # V 170 = red tint survives
