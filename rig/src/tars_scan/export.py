"""Session -> EuRoC/ASL directory (mav0/cam0, cam1, imu0), the input format Basalt, OpenVINS,
ORB-SLAM3 and Kalibr's bag tools all read. Writes export.json with what was done and how well the
stereo frames paired.

Timestamps:
  cameras  one time per frame at mid-exposure of the middle row (a rolling-shutter frame has no
           single time; this is the one a global-shutter model fits best). From libcamera's
           SensorTimestamp, taken as the frame start of readout (A154): t = sensor_ns -
           exposure/2 + readout/2. Calibration estimates whatever constant error remains.
  imu      refitted from reads.csv (tars_rig.check.imu_clock_segments), not the recorder's nominal
           rate, so the chip's clock drift is gone.

Stereo frames are paired by nearest timestamp (within half a frame) and both get the left time, as
the EuRoC readers expect; the real offsets go to export.json. `scale` < 1 decodes at reduced size
in the JPEG's DCT domain (1/2, 1/4, 1/8: nearly free) for cheaper SLAM runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tars_rig.check import imu_clock_segments, imu_sample_times, read_csv
from tars_rig.contracts import iter_jpegs, read_json

G = 9.80665
SOF_EXPOSURE_FRACTION = -0.5  # A154: sensor_ns marks the end of row 0's exposure


def frame_times_ns(frames: dict[str, np.ndarray], readout_us: float | None) -> np.ndarray:
    exposure_ns = frames["exposure_us"] * 1e3
    readout_ns = (readout_us or 0.0) * 1e3
    return frames["sensor_ns"] + SOF_EXPOSURE_FRACTION * exposure_ns + readout_ns / 2


def pair_frames(t0: np.ndarray, t1: np.ndarray, max_gap_ns: float) -> list[tuple[int, int]]:
    """(i0, i1) positions with t1[i1] the nearest right frame to t0[i0], within max_gap_ns, each
    right frame used once."""
    order = np.argsort(t1)
    s1 = t1[order]
    pairs, used = [], set()
    for i0, t in enumerate(t0):
        j = int(np.searchsorted(s1, t))
        best = min(
            (c for c in (j - 1, j) if 0 <= c < len(s1)), key=lambda c: abs(s1[c] - t), default=None
        )
        if best is not None and abs(s1[best] - t) <= max_gap_ns and best not in used:
            used.add(best)
            pairs.append((i0, int(order[best])))
    return pairs


def _decode(jpeg: bytes, scale: float) -> np.ndarray:
    import simplejpeg

    if scale >= 1:
        return simplejpeg.decode_jpeg(jpeg, colorspace="GRAY")[..., 0]
    h, w = simplejpeg.decode_jpeg_header(jpeg)[:2]
    return simplejpeg.decode_jpeg(
        jpeg, colorspace="GRAY", min_height=math.ceil(h * scale), min_width=math.ceil(w * scale)
    )[..., 0]


def export_imu(session: Path, meta: dict, out: Path) -> dict:
    imu, reads = read_csv(session / "imu" / "imu.csv"), read_csv(session / "imu" / "reads.csv")
    segs = imu_clock_segments(reads)
    t = imu_sample_times(imu["k"], segs)
    ok = np.isfinite(t)
    acc_scale = G / meta["imu"]["accel_lsb_per_g"]
    gyro_scale = math.pi / 180 / meta["imu"]["gyro_lsb_per_dps"]
    (out / "imu0").mkdir(parents=True, exist_ok=True)
    with open(out / "imu0" / "data.csv", "w") as f:
        f.write(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        )
        for i in np.flatnonzero(ok):
            g = [imu[c][i] * gyro_scale for c in ("gx", "gy", "gz")]
            a = [imu[c][i] * acc_scale for c in ("ax", "ay", "az")]
            f.write(
                f"{round(t[i])},{g[0]:.6f},{g[1]:.6f},{g[2]:.6f},{a[0]:.5f},{a[1]:.5f},{a[2]:.5f}\n"
            )
    return {
        "samples": int(ok.sum()),
        "dropped_outside_fit": int((~ok).sum()),
        "segments": len(segs),
        "rate_hz": [round(s["rate_hz"], 3) for s in segs],
    }


def export_session(session: Path, out: Path, scale: float = 1.0, step: int = 1) -> dict:
    """Write the EuRoC layout for one session to out/mav0; returns (and writes) export.json."""
    import cv2

    meta = read_json(session / "meta.json")
    mav = out / "mav0"
    report: dict = {"session": session.name, "scale": scale, "step": step}
    cams = meta.get("cameras") or []
    if cams:
        fps = cams[0]["fps"]
        frames = [read_csv(session / f"cam{i}" / "frames.csv") for i in (0, 1)]
        stored = [f["offset"] >= 0 for f in frames]
        # Frames before the software sync locked are up to half a frame apart: not a stereo pair.
        ready = [f["sync_ready"] > 0 for f in frames]
        if all(r.any() for r in ready):
            report["skipped_before_sync_lock"] = [
                int((s & ~r).sum()) for s, r in zip(stored, ready, strict=True)
            ]
            stored = [s & r for s, r in zip(stored, ready, strict=True)]
        times = [frame_times_ns(f, c.get("readout_us")) for f, c in zip(frames, cams, strict=True)]
        idx = [np.flatnonzero(s) for s in stored]
        pairs = pair_frames(times[0][idx[0]], times[1][idx[1]], 0.5e9 / fps)[::step]
        offsets_us = np.array([(times[1][idx[1][b]] - times[0][idx[0][a]]) / 1e3 for a, b in pairs])
        report["stereo"] = {
            "pairs": len(pairs),
            "left_stored": int(len(idx[0])),
            "right_stored": int(len(idx[1])),
            "offset_us_median": round(float(np.median(offsets_us)), 1) if len(pairs) else None,
            "abs_offset_us_max": round(float(np.abs(offsets_us).max()), 1) if len(pairs) else None,
        }
        for cam in (0, 1):
            d = mav / f"cam{cam}" / "data"
            d.mkdir(parents=True, exist_ok=True)
            rows = [idx[cam][p[cam]] for p in pairs]
            stamps = {int(r): int(times[0][idx[0][p[0]]]) for r, p in zip(rows, pairs, strict=True)}
            f = frames[cam]
            jpeg_rows = ((int(f["index"][r]), int(f["offset"][r]), int(f["size"][r])) for r in rows)
            by_index = {int(f["index"][r]): int(r) for r in rows}
            with open(mav / f"cam{cam}" / "data.csv", "w") as csv_out:
                csv_out.write("#timestamp [ns],filename\n")
                for index, jpeg in iter_jpegs(session / f"cam{cam}", jpeg_rows):
                    ts = stamps[by_index[index]]
                    cv2.imwrite(str(d / f"{ts}.png"), _decode(jpeg, scale))
                    csv_out.write(f"{ts},{ts}.png\n")
    if meta.get("imu"):
        report["imu"] = export_imu(session, meta, mav)
    (out / "export.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tars_scan.export", description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--scale", type=float, default=1.0, help="1, 0.5, 0.25 or 0.125")
    p.add_argument("--step", type=int, default=1, help="keep every n-th stereo pair")
    args = p.parse_args(argv)
    print(json.dumps(export_session(args.session, args.out, args.scale, args.step), indent=2))


if __name__ == "__main__":
    main()
