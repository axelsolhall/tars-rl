"""Compare patch trackers on real frames: follow each board corner from frame k to k+1 and score
the result against the corner the board detector finds in k+1.

  python -m tars_scan.track_eval <session> <export dir> [--step 8]

Trackers (same window, same 3-level coarse-to-fine pyramid, so only the matching differs):
- klt: pyramidal Lucas-Kanade (OpenCV), the intensity-gradient tracker Basalt/OpenVINS use.
- phase: phase correlation (FFT of both patches, keep only the phase, peak = shift). Ignoring the
  amplitude should make it insensitive to brightness and contrast changes.

The window is kept below one board square: a checkerboard repeats, and a window spanning several
squares has several equally good shifts for any matcher. Corner detections are themselves good to
~0.1 px, so errors below that are not resolved. Results are split by turn rate (gyro) and patch
brightness, the two conditions our toy cameras are bad at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOARD = (10, 7)
LEVELS = 3
FAIL_PX = 2.0


def detect(img: np.ndarray) -> np.ndarray | None:
    import cv2

    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    ok, c = cv2.findChessboardCornersSB(norm, BOARD, flags=cv2.CALIB_CB_ACCURACY)
    return c.reshape(-1, 2).astype(np.float64) if ok else None


def klt(i0: np.ndarray, i1: np.ndarray, pts: np.ndarray, win: int) -> np.ndarray:
    import cv2

    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        i0,
        i1,
        pts.astype(np.float32).reshape(-1, 1, 2),
        None,
        winSize=(win, win),
        maxLevel=LEVELS - 1,
        criteria=crit,
    )
    out = p1.reshape(-1, 2).astype(np.float64)
    out[st.ravel() == 0] = np.nan
    return out


def phase(i0: np.ndarray, i1: np.ndarray, pts: np.ndarray, win: int) -> np.ndarray:
    """Coarse-to-fine phase correlation; each level re-centres the second patch on the estimate."""
    import cv2

    pyr0, pyr1 = [i0.astype(np.float32)], [i1.astype(np.float32)]
    for _ in range(LEVELS - 1):
        pyr0.append(cv2.pyrDown(pyr0[-1]))
        pyr1.append(cv2.pyrDown(pyr1[-1]))
    hann = cv2.createHanningWindow((win, win), cv2.CV_32F)
    out = np.empty_like(pts)
    for n, p in enumerate(pts):
        d = np.zeros(2)
        for lvl in range(LEVELS - 1, -1, -1):
            s = 2.0**lvl
            for _ in range(2 if lvl == 0 else 1):  # one extra pass at full resolution
                a = cv2.getRectSubPix(pyr0[lvl], (win, win), tuple(p / s))
                b = cv2.getRectSubPix(pyr1[lvl], (win, win), tuple((p + d) / s))
                (dx, dy), _ = cv2.phaseCorrelate(a, b, hann)
                d += s * np.array([dx, dy])
        out[n] = p + d
    return out


def bandpass(img: np.ndarray, lo_sigma: float, hi_sigma: float) -> np.ndarray:
    """Difference of Gaussians: keeps detail between ~hi_sigma and ~lo_sigma pixels. Dropping the
    low end removes slow brightness changes (vignetting, exposure); dropping the top removes pixel
    noise. Rescaled to 8 bit around 128 so both trackers take it like an image."""
    import cv2

    f = img.astype(np.float32)
    d = cv2.GaussianBlur(f, (0, 0), hi_sigma) - cv2.GaussianBlur(f, (0, 0), lo_sigma)
    return np.clip(128 + d * (60 / (d.std() + 1e-6)), 0, 255).astype(np.uint8)


def _banded(fn, lo: float, hi: float):
    return lambda i0, i1, pts, win: fn(bandpass(i0, lo, hi), bandpass(i1, lo, hi), pts, win)


def pc_shift(a: np.ndarray, b: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """Shift d with b(x) = a(x - d). Cross-power spectrum divided by |.|^alpha (1 = classic
    phase correlation, <1 keeps some amplitude so noisy weak frequencies count less), peak refined
    with a 3-point Gaussian fit per axis: Ren et al. 2010 measured it far less biased than a
    centroid, which OpenCV's phaseCorrelate uses."""
    fa, fb = np.fft.fft2(a * w), np.fft.fft2(b * w)
    r = np.conj(fa) * fb
    r /= np.abs(r) ** alpha + 1e-9
    c = np.fft.fftshift(np.real(np.fft.ifft2(r)))
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    h = np.array(c.shape) // 2
    sub = []
    for v0, vm, vp in (
        (c[iy, ix], c[iy, ix - 1], c[iy, (ix + 1) % c.shape[1]]),
        (c[iy, ix], c[iy - 1, ix], c[(iy + 1) % c.shape[0], ix]),
    ):
        lm, l0, lp = (np.log(max(x, 1e-12)) for x in (vm, v0, vp))
        den = lm - 2 * l0 + lp
        sub.append(0.5 * (lm - lp) / den if den < 0 else 0.0)
    return np.array([ix - h[1] + sub[0], iy - h[0] + sub[1]])


def phase_fit(alpha: float):
    def track(i0, i1, pts, win):
        import cv2

        pyr0, pyr1 = [i0.astype(np.float32)], [i1.astype(np.float32)]
        for _ in range(LEVELS - 1):
            pyr0.append(cv2.pyrDown(pyr0[-1]))
            pyr1.append(cv2.pyrDown(pyr1[-1]))
        w = cv2.createHanningWindow((win, win), cv2.CV_32F)
        out = np.empty_like(pts)
        for n, p in enumerate(pts):
            d = np.zeros(2)
            for lvl in range(LEVELS - 1, -1, -1):
                s = 2.0**lvl
                for _ in range(3 if lvl == 0 else 1):  # re-centre: iterative PC at full res
                    a = cv2.getRectSubPix(pyr0[lvl], (win, win), tuple(p / s))
                    b = cv2.getRectSubPix(pyr1[lvl], (win, win), tuple((p + d) / s))
                    d += s * pc_shift(a - a.mean(), b - b.mean(), w, alpha)
            out[n] = p + d
        return out

    return track


TRACKERS = {"klt": klt, "phase": phase}
for _a in (1.0, 0.7):
    TRACKERS[f"phase_gauss_a{_a:g}"] = phase_fit(_a)
for _lo, _hi in ((4.0, 0.7), (8.0, 1.0)):
    TRACKERS[f"klt_bp{_hi:g}-{_lo:g}"] = _banded(klt, _lo, _hi)
    TRACKERS[f"phase_bp{_hi:g}-{_lo:g}"] = _banded(phase, _lo, _hi)
    TRACKERS[f"phase_gauss_a0.7_bp{_hi:g}-{_lo:g}"] = _banded(phase_fit(0.7), _lo, _hi)

# Degradations applied to both frames of a real pair (true corner positions stay valid: dimming,
# noise and a centred blur kernel move no corner). "dark" mimics a 4x underexposed frame lifted by
# 4x gain: 1/4 the signal, then sensor noise amplified by the gain.
CONDITIONS = {  # stand-ins for a dim room and motion blur (loopdesk A162)
    "clean": dict(dim=1.0, noise=0.0, blur=0),
    "dark": dict(dim=0.25, noise=2.0, blur=0),
    "blur": dict(dim=1.0, noise=0.0, blur=7),
    "dark+blur": dict(dim=0.25, noise=2.0, blur=7),
}


def degrade(img: np.ndarray, dim: float, noise: float, blur: int, rng) -> np.ndarray:
    """Horizontal motion blur of `blur` px, signal * dim, Gaussian noise (sigma in the dimmed
    image's 8-bit units), then gain back up by 1/dim, as the camera's analogue gain would."""
    import cv2

    f = img.astype(np.float32)
    if blur:
        f = cv2.filter2D(f, -1, np.full((1, blur), 1.0 / blur, np.float32))
    f = f * dim + rng.normal(0, noise, f.shape).astype(np.float32)
    return np.clip(np.round(f) / dim, 0, 255).astype(np.uint8)


def frame_list(export: Path) -> tuple[np.ndarray, list[Path]]:
    rows = [ln.split(",") for ln in (export / "mav0/cam0/data.csv").read_text().splitlines()[1:]]
    return np.array([int(r[0]) for r in rows]), [export / "mav0/cam0/data" / r[1] for r in rows]


def evaluate(session: Path, export: Path, step: int = 8) -> dict:
    import cv2

    from .rs_correct import gyro_cam0
    from .slam_calib import rig_model

    t_gyro, w_gyro = gyro_cam0(session, rig_model(session))
    t, paths = frame_list(export)
    rng = np.random.default_rng(0)
    recs = []  # (tracker, err_px, rate_deg_s, brightness, condition)
    for k in range(0, len(paths) - 1, step):
        i0 = cv2.imread(str(paths[k]), cv2.IMREAD_GRAYSCALE)
        i1 = cv2.imread(str(paths[k + 1]), cv2.IMREAD_GRAYSCALE)
        c0, c1 = detect(i0), detect(i1)
        if c0 is None or c1 is None:
            continue
        # The detector may number the corners from the opposite end in the next frame.
        if np.linalg.norm(c0 - c1, axis=1).mean() > np.linalg.norm(c0 - c1[::-1], axis=1).mean():
            c1 = c1[::-1]
        square = np.median(
            np.linalg.norm(np.diff(c0.reshape(BOARD[1], BOARD[0], 2), axis=1), axis=-1)
        )
        win = int(max(9, min(31, 0.9 * square))) | 1
        m = (t_gyro >= t[k]) & (t_gyro <= t[k + 1])
        rate = float(np.degrees(np.linalg.norm(w_gyro[m].mean(0)))) if m.any() else np.nan
        bright = [float(cv2.getRectSubPix(i1, (win, win), tuple(p)).mean()) for p in c1]
        for cond, kw in CONDITIONS.items():
            j0, j1 = degrade(i0, rng=rng, **kw), degrade(i1, rng=rng, **kw)
            for name, fn in TRACKERS.items():
                err = np.linalg.norm(fn(j0, j1, c0, win) - c1, axis=1)
                recs += [(name, e, rate, b, cond) for e, b in zip(err, bright, strict=True)]
    return summarise(recs)


def _stats(e: np.ndarray) -> dict:
    ok = np.isfinite(e)
    e = np.where(ok, e, np.inf)
    return {
        "n": int(len(e)),
        "median_px": round(float(np.median(e)), 3),
        "p90_px": round(float(np.percentile(e, 90)), 3),
        f"fail_over_{FAIL_PX:g}px": round(float(np.mean(e > FAIL_PX)), 4),
    }


def summarise(recs: list) -> dict:
    name = np.array([r[0] for r in recs])
    err, rate, bright = (np.array([r[i] for r in recs], dtype=float) for i in (1, 2, 3))
    cond = np.array([r[4] for r in recs])
    out: dict = {"by_condition": {}, "by_turn_rate_deg_s": {}, "by_brightness": {}}
    for c in CONDITIONS:
        out["by_condition"][c] = {tr: _stats(err[(cond == c) & (name == tr)]) for tr in TRACKERS}
    clean = cond == "clean"  # rate and brightness splits on the real, undegraded frames
    name = np.where(clean, name, "")
    for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 1e9)):
        key = f"{lo}-{hi if hi < 1e9 else 'up'}"
        sel = (rate >= lo) & (rate < hi)
        if sel.any():
            out["by_turn_rate_deg_s"][key] = {
                tr: _stats(err[sel & (name == tr)]) for tr in TRACKERS
            }
    for lo, hi in ((0, 40), (40, 80), (80, 140), (140, 256)):
        sel = (bright >= lo) & (bright < hi)
        if sel.any():
            out["by_brightness"][f"{lo}-{hi}"] = {
                tr: _stats(err[sel & (name == tr)]) for tr in TRACKERS
            }
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tars_scan.track_eval", description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path)
    p.add_argument("export", type=Path)
    p.add_argument("--step", type=int, default=8)
    a = p.parse_args(argv)
    res = evaluate(a.session, a.export, a.step)
    (a.export / "track_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
