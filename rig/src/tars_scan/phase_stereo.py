"""Phase-based stereo: disparity from the local phase of band-pass (Gabor) responses.

  python -m tars_scan.phase_stereo <session> <export dir> [--method phase|sgbm_phase] [--step 10]

Axel's idea (2026-11-10): match left and right patches by the phase of their high-frequency
content. Phase stereo (Sanger 1988; Fleet, Jepson & Jenkin 1991) reads disparity straight from
the phase difference of complex Gabor responses, d = dphi / omega, with sub-pixel resolution, and
is reported to tolerate contrast and brightness differences between the cameras. Its known
weaknesses are phase singularities (low amplitude, odd local frequency) and a range of about
two-thirds of one filter period, hence coarse-to-fine.

Methods (on the small rectification, mirrored like stereo_depth so disparity is >= 0):
- phase: pure coarse-to-fine phase. The coarsest level estimates, each finer level warps the
  right image by the current disparity and adds the phase-measured residual.
- sgbm_phase: SGBM's disparity, refined by one phase step at full resolution. Pixels where the
  phase is unreliable (low amplitude, local frequency off by more than half, or a residual over
  `max_step_px`) keep no depth.
Scored like stereo_depth: against the board plane from PnP, plus ms per frame on this machine.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PhaseParams:
    wavelength_px: float = 6.0  # Gabor period at each pyramid level: high frequencies only
    sigma_px: float = 3.0  # Gaussian envelope (about half a period on each side)
    levels: int = 4  # coarse-to-fine levels for `phase` (range ~ 2/3 * wavelength * 2^(levels-1))
    iters: int = 2  # warp-and-refine iterations per level
    min_amp_rel: float = 0.15  # amplitude below this share of the local mean = unreliable phase
    max_step_px: float = 1.0  # sgbm_phase: larger corrections are rejected, not applied
    fallback: bool = False  # sgbm_phase: unreliable pixels keep SGBM's disparity instead of none


def gabor(img: np.ndarray, p: PhaseParams) -> np.ndarray:
    """Complex horizontal Gabor response (DC removed), float32 complex."""
    import cv2

    r = int(np.ceil(3 * p.sigma_px))
    x = np.arange(-r, r + 1, dtype=np.float32)
    env = np.exp(-(x**2) / (2 * p.sigma_px**2))
    w = 2 * np.pi / p.wavelength_px
    kc, ks = env * np.cos(w * x), env * np.sin(w * x)
    kc -= env * (kc.sum() / env.sum())  # zero DC: brightness offsets do not leak into the phase
    ky = env[:, None] / env.sum()
    f = img.astype(np.float32)
    re = cv2.sepFilter2D(f, cv2.CV_32F, kc, ky)
    # OpenCV filters by correlation: minus makes the phase grow with x, as w*x does
    im = -cv2.sepFilter2D(f, cv2.CV_32F, ks, ky)
    return re + 1j * im


def _shift(img: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Right image sampled at x - d (so a correct d lines it up with the left image)."""
    import cv2

    h, w = img.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs - d.astype(np.float32), ys, cv2.INTER_LINEAR, borderValue=0)


def phase_step(left, right_warped, p: PhaseParams):
    """(residual disparity, reliable mask) of right_warped against left."""
    qa, qb = gabor(left, p), gabor(right_warped, p)
    dphi = np.angle(qa * np.conj(qb))

    # local frequency: phase derivative along x of both responses (more accurate than nominal w)
    def dx_phase(q):
        return np.angle(q[:, 2:] * np.conj(q[:, :-2])) / 2

    w_loc = np.zeros(left.shape, np.float32)
    w_loc[:, 1:-1] = (dx_phase(qa) + dx_phase(qb)) / 2
    w0 = 2 * np.pi / p.wavelength_px
    amp = np.sqrt(np.abs(qa) * np.abs(qb))
    import cv2

    local_mean = cv2.blur(amp.astype(np.float32), (15, 15)) + 1e-6
    ok = (amp > p.min_amp_rel * local_mean) & (w_loc > 0.5 * w0) & (w_loc < 1.5 * w0)
    d = np.where(ok, -dphi / np.where(ok, w_loc, 1), 0)  # R(x) = L(x + d): R's phase leads by w*d
    return d.astype(np.float32), ok


def phase_disparity(left, right, p: PhaseParams):
    """Pure coarse-to-fine phase disparity and its reliable mask."""
    import cv2

    pyr_l, pyr_r = [left.astype(np.float32)], [right.astype(np.float32)]
    for _ in range(p.levels - 1):
        pyr_l.append(cv2.pyrDown(pyr_l[-1]))
        pyr_r.append(cv2.pyrDown(pyr_r[-1]))
    d = np.zeros(pyr_l[-1].shape, np.float32)
    ok = np.ones(d.shape, bool)
    for lv in range(p.levels - 1, -1, -1):
        if d.shape != pyr_l[lv].shape:
            d = cv2.resize(d, pyr_l[lv].shape[::-1], interpolation=cv2.INTER_LINEAR) * 2
            ok = cv2.resize(ok.astype(np.uint8), pyr_l[lv].shape[::-1], interpolation=0) > 0
        for _ in range(p.iters):
            step, good = phase_step(pyr_l[lv], _shift(pyr_r[lv], d), p)
            d = d + step
            ok = good
    return d, ok & (d > 0.5)


DEFAULT = PhaseParams()


def depth_phase(rect, sgbm, left, right, band=None, method="phase", p: PhaseParams = DEFAULT):
    """Depth (m) along rectified cam0's z, NaN where unreliable (stereo_depth.depth's contract)."""
    import cv2

    l_r = cv2.remap(left, *rect["maps"][0], cv2.INTER_LINEAR)
    r_r = cv2.remap(right, *rect["maps"][1], cv2.INTER_LINEAR)
    flip = rect["baseline"] < 0
    if flip:  # cam0 is the right-hand camera: mirror so disparity is >= 0, as stereo_depth does
        l_r, r_r = l_r[:, ::-1].copy(), r_r[:, ::-1].copy()
    if method == "phase":
        d, ok = phase_disparity(l_r, r_r, p)
    else:
        if band:
            from .track_eval import bandpass

            bl, br = bandpass(l_r, *band), bandpass(r_r, *band)
        else:
            bl, br = l_r, r_r
        d0 = sgbm.compute(bl, br).astype(np.float32) / 16
        has = d0 > 0
        step, good = phase_step(l_r.astype(np.float32), _shift(r_r.astype(np.float32), d0), p)
        fix = has & good & (np.abs(step) <= p.max_step_px)
        d = d0 + np.where(fix, step, 0)
        ok = has if p.fallback else fix
    if flip:
        d, ok = d[:, ::-1], ok[:, ::-1]
    with np.errstate(divide="ignore"):
        return np.where(ok & (d > 0), rect["f"] * abs(rect["baseline"]) / d, np.nan)


def main(argv: list[str] | None = None) -> None:
    from . import stereo_depth

    ap = argparse.ArgumentParser(prog="tars_scan.phase_stereo", description=__doc__.split("\n")[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("export", type=Path)
    ap.add_argument("--method", choices=("phase", "sgbm_phase", "sgbm"), default="sgbm_phase")
    ap.add_argument("--degrade", help="gain,noise_sd for the right image: unequal cameras")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--set", action="append", default=[], help="PhaseParams override, k=v")
    a = ap.parse_args(argv)
    p = PhaseParams()
    for kv in a.set:
        k, v = kv.split("=", 1)
        p = replace(p, **{k: type(getattr(p, k))(v)})

    gain, noise = (float(x) for x in a.degrade.split(",")) if a.degrade else (1.0, 0.0)
    rng = np.random.default_rng(0)

    def fn(rect, sgbm, left, right, band=None):
        if a.degrade:  # the right camera darker and noisier than the left
            r = right.astype(np.float32) * gain + rng.normal(0, noise, right.shape)
            right = np.clip(r, 0, 255).astype(np.uint8)
        if a.method == "sgbm":
            return stereo_depth.depth(rect, sgbm, left, right, band)
        return depth_phase(rect, sgbm, left, right, band, a.method, p)

    r = stereo_depth.evaluate(a.session, a.export, a.step, (4.0, 0.5), True, depth_fn=fn)
    r = {"method": a.method, "phase_params": asdict(p), **r}
    tag = "_".join(kv.replace("=", "") for kv in a.set)
    tag += f"_deg{a.degrade.replace(',', '-')}" if a.degrade else ""
    out = a.export / f"phase_stereo_{a.method}{'_' + tag if tag else ''}.json"
    out.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
