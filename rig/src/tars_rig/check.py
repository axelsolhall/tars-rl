"""Eval of one recorded session: numbers that say whether the recording can feed calibration and
SLAM.

Reports, per camera: real frame rate, frame-interval spread, dropped frames, exposure; for the pair:
how far apart in time the left and right frames are (the software-sync result); for the IMU: real
rate after fitting its clock, drift against the nominal rate, overflows, read jitter, and the still-
rig numbers (gravity, gyro bias). No pass bars: it reports, the reader judges. Runs on the Pi after
each session and on the Mac (`rig check <session>`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    cols = {}
    for i, name in enumerate(header):
        vals = [r[i] for r in body]
        try:
            cols[name] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            cols[name] = np.array(vals)
    return cols


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def camera_stats(frames: dict[str, np.ndarray], fps: float) -> dict:
    t = frames["sensor_ns"]
    t = t[np.isfinite(t)]
    out = {"frames": int(len(frames["index"])), "frames_with_sensor_time": int(len(t))}
    if len(t) < 2:
        return out
    dt = np.diff(t) / 1e6  # ms
    nominal = 1000 / fps
    out.update(
        {
            "seconds": round((t[-1] - t[0]) / 1e9, 2),
            "fps_measured": round((len(t) - 1) / ((t[-1] - t[0]) / 1e9), 3),
            "interval_ms_median": round(float(np.median(dt)), 3),
            "interval_ms_p99": round(_pct(dt, 99), 3),
            "interval_ms_max": round(float(dt.max()), 3),
            "dropped_frames": int(np.sum(np.maximum(np.round(dt / nominal) - 1, 0))),
            "backwards_steps": int(np.sum(dt <= 0)),
            "host_latency_ms_median": round(
                _pct((frames["host_ns"] - frames["sensor_ns"]) / 1e6, 50), 2
            ),
            "exposure_us_values": sorted({int(v) for v in frames["exposure_us"] if np.isfinite(v)})[
                :5
            ],
            "sync_ready_fraction": round(float(np.mean(frames["sync_ready"])), 3),
        }
    )
    if "offset" in frames:  # what the Pi spent: encoder drops, encode time, disk bandwidth
        written = frames["offset"] >= 0
        secs = max((t[-1] - t[0]) / 1e9, 1e-9)
        enc = frames["encode_us"][written] / 1e3
        out.update(
            {
                "encoder_dropped": int((~written).sum()),
                "encode_ms_p50": round(_pct(enc, 50), 2),
                "encode_ms_p99": round(_pct(enc, 99), 2),
                "jpeg_kb_mean": round(float(frames["size"][written].mean()) / 1024, 1)
                if written.any()
                else None,
                "write_mb_per_s": round(float(frames["size"][written].sum()) / secs / 1e6, 2),
            }
        )
    return out


def stereo_offsets(t0: np.ndarray, t1: np.ndarray, fps: float) -> dict:
    """For each left frame, the time to the nearest right frame.

    Pairs further apart than half a frame are unpaired."""
    t0, t1 = t0[np.isfinite(t0)], np.sort(t1[np.isfinite(t1)])
    if len(t0) == 0 or len(t1) == 0:
        return {"pairs": 0}
    j = np.clip(np.searchsorted(t1, t0), 1, len(t1) - 1)
    nearest = np.where(np.abs(t1[j] - t0) < np.abs(t1[j - 1] - t0), t1[j], t1[j - 1])
    off_us = (nearest - t0) / 1e3
    half = 0.5e6 / fps
    paired = np.abs(off_us) <= half
    o = off_us[paired]
    return {
        "pairs": int(paired.sum()),
        "unpaired_left": int((~paired).sum()),
        "offset_us_median": round(float(np.median(o)), 1) if len(o) else None,
        "abs_offset_us_p95": round(_pct(np.abs(o), 95), 1) if len(o) else None,
        "abs_offset_us_max": round(float(np.abs(o).max()), 1) if len(o) else None,
    }


def fit_imu_clock(host_ns: np.ndarray, samples_total: np.ndarray) -> dict:
    """Fit host time against sample count: the slope is the chip's real sample period.

    Each read's host time comes a little after its newest sample, so the line is moved down to the
    earliest reads (the lower envelope); what is left above it is read latency."""
    if len(host_ns) < 3:
        return {}
    slope, intercept = np.polyfit(samples_total, host_ns, 1)
    resid = host_ns - (slope * samples_total + intercept)
    resid_min = resid.min()
    resid -= resid_min
    return {
        "period_ns": float(slope),
        "intercept_ns": float(intercept + resid_min),
        "rate_hz": 1e9 / slope,
        "read_latency_ms_p50": round(_pct(resid / 1e6, 50), 3),
        "read_latency_ms_p99": round(_pct(resid / 1e6, 99), 3),
    }


def imu_clock_segments(reads: dict[str, np.ndarray]) -> list[dict]:
    """One clock fit per stretch between FIFO overflows (an overflow loses samples, so the count
    jumps). Each: the sample range [k0, k1) it covers and t(count) = slope * count + intercept."""
    over = reads["overflow"] != 0
    seg_id = np.cumsum(over)
    segs = []
    for s in np.unique(seg_id[~over]):
        sel = (seg_id == s) & ~over
        fit = fit_imu_clock(reads["host_ns"][sel], reads["samples_total"][sel])
        if not fit:
            continue
        starts = reads["samples_total"][(seg_id == s) & over]  # the overflow row opening it
        k0 = int(starts[0]) if len(starts) else 0
        segs.append({"k0": k0, "k1": int(reads["samples_total"][sel][-1]), **fit})
    return segs


def imu_sample_times(k: np.ndarray, segs: list[dict]) -> np.ndarray:
    """Host-clock time of each IMU sample from the refitted clock: sample k is the newest in the
    FIFO when the count reaches k + 1. NaN for samples outside every fitted stretch."""
    t = np.full(len(k), np.nan)
    for s in segs:
        sel = (k >= s["k0"]) & (k < s["k1"])
        t[sel] = s["period_ns"] * (k[sel] + 1) + s["intercept_ns"]
    return t


def imu_stats(imu_dir: Path, meta_imu: dict | None) -> dict:
    imu = read_csv(imu_dir / "imu.csv")
    reads = read_csv(imu_dir / "reads.csv")
    out: dict = {"samples": int(len(imu["k"])), "overflows": int(np.nansum(reads["overflow"]))}
    # Fit the longest stretch without an overflow (an overflow loses samples, so the count jumps).
    ok = reads["overflow"] == 0
    seg_id = np.cumsum(~ok)
    best = max(set(seg_id[ok]), key=lambda s: np.sum((seg_id == s) & ok), default=None)
    if best is not None:
        sel = (seg_id == best) & ok
        fit = fit_imu_clock(reads["host_ns"][sel], reads["samples_total"][sel])
        if fit:
            nominal = (meta_imu or {}).get("rate_hz")
            out.update(
                {
                    "rate_hz_fitted": round(fit["rate_hz"], 4),
                    "drift_ppm_vs_nominal": round((fit["rate_hz"] / nominal - 1) * 1e6, 1)
                    if nominal
                    else None,
                    "read_latency_ms_p50": fit["read_latency_ms_p50"],
                    "read_latency_ms_p99": fit["read_latency_ms_p99"],
                    "seconds": round(
                        (reads["host_ns"][sel][-1] - reads["host_ns"][sel][0]) / 1e9, 1
                    ),
                }
            )
    if meta_imu and len(imu["k"]):
        acc = np.stack([imu["ax"], imu["ay"], imu["az"]], 1) / meta_imu["accel_lsb_per_g"]
        gyr = np.stack([imu["gx"], imu["gy"], imu["gz"]], 1) / meta_imu["gyro_lsb_per_dps"]
        out.update(
            {
                "accel_norm_g_mean": round(float(np.linalg.norm(acc, axis=1).mean()), 4),
                "gyro_dps_mean": [round(float(v), 4) for v in gyr.mean(0)],
                "gyro_dps_std": [round(float(v), 4) for v in gyr.std(0)],
            }
        )
    temp_path = imu_dir / "temp.csv"
    if temp_path.exists():
        temp = read_csv(temp_path)["temp_c"]
        if len(temp):
            out["temp_c_range"] = [
                round(float(np.nanmin(temp)), 2),
                round(float(np.nanmax(temp)), 2),
            ]
    return out


def check_session(session: Path) -> dict:
    meta = (
        json.loads((session / "meta.json").read_text()) if (session / "meta.json").exists() else {}
    )
    report: dict = {"session": session.name, "mode": meta.get("mode")}
    done = session / "done.json"
    if done.exists():
        d = json.loads(done.read_text())
        report["stop_reason"], report["errors"] = d.get("stop_reason"), d.get("errors", [])
    fps = meta.get("cameras", [{}])[0].get("fps", 30) if meta.get("cameras") else 30
    cams = {}
    for i in (0, 1):
        path = session / f"cam{i}" / "frames.csv"
        if path.exists():
            cams[i] = read_csv(path)
            report[f"cam{i}"] = camera_stats(cams[i], fps)
    if len(cams) == 2:
        # Before the software sync locks (about 100 frames, 3.3 s) the cameras free-run up to half
        # a frame apart; those frames are counted, and the offsets are measured after the lock.
        ready = [cams[i]["sync_ready"] > 0 for i in (0, 1)]
        locked = all(r.any() for r in ready)
        t = [cams[i]["sensor_ns"][ready[i]] if locked else cams[i]["sensor_ns"] for i in (0, 1)]
        report["stereo"] = {
            "after_sync_lock": locked,
            "left_frames_before_lock": int((~ready[0]).sum()) if locked else None,
            **stereo_offsets(t[0], t[1], fps),
        }
    if (session / "imu" / "imu.csv").exists():
        report["imu"] = imu_stats(session / "imu", meta.get("imu"))
        if cams:
            imu_t = read_csv(session / "imu" / "imu.csv")["t_ns"]
            cam_t = cams[0]["sensor_ns"]
            report["imu_covers_cameras"] = bool(
                len(imu_t) and imu_t.min() <= np.nanmin(cam_t) and imu_t.max() >= np.nanmax(cam_t)
            )
    return report


def summary_line(r: dict) -> str:
    parts = [f"{r.get('mode')}: {r.get('stop_reason', '?')}"]
    for i in (0, 1):
        c = r.get(f"cam{i}")
        if c and "fps_measured" in c:
            line = (
                f"cam{i} {c['frames']} fr @ {c['fps_measured']} fps, {c['dropped_frames']} dropped"
            )
            if "encoder_dropped" in c:
                line += f" + {c['encoder_dropped']} by encoder, {c['write_mb_per_s']} MB/s"
            parts.append(line)
    s = r.get("stereo")
    if s and s.get("pairs"):
        parts.append(
            f"L-R offset median {s['offset_us_median']} us, p95 {s['abs_offset_us_p95']} us"
        )
    m = r.get("imu")
    if m:
        parts.append(
            f"imu {m['samples']} samples @ {m.get('rate_hz_fitted', '?')} Hz, "
            f"{m['overflows']} overflows"
        )
    if r.get("errors"):
        parts.append(f"ERRORS: {r['errors']}")
    return "; ".join(parts)
