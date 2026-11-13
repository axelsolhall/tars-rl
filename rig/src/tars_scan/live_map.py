"""Live temporal map: a rig session (being recorded, or finished) -> Basalt -> map -> 3D viewer.

  python -m tars_scan.live_map <session dir> [--follow] [--realtime 1] [--connect URL]

Pipeline (each stage replaceable; contracts in brackets):
1. SessionSource reads the session's files as the recorders write them (--follow) or replays a
   finished session at --realtime speed: the IMU with an online clock fit (the lower envelope,
   as tars_rig.check does offline), stereo pairs matched by sensor time, decoded at half
   resolution, one gain, rolling shutter undone with the gyro: the same steps as tars_scan.export,
   so live Basalt sees what offline Basalt saw. [Imu / Pair]
2. BasaltClient streams it to tars_live_vio and gets T_world_imu per processed pair. [live_pose v1]
3. Mapper (occupancy_map, the same code as the offline maps) fuses every --map-every-th pair
   that has a pose, latest first; when mapping falls behind, older pairs are skipped. [FusedFrame]
4. LiveLogger (map_live_view) streams frames and map to a Rerun viewer and saves a .rrd.

Live on the rig: `scripts/rig arm live`, press the button; the Pi runs this with --follow and the
Mac's viewer is reached over the SSH tunnel (scripts/rig live). Here on the Mac, any recorded
session replays the same way: Basalt in Docker (scripts/live_vio.sh), this module natively.
Eval: the live trajectory is written to <out>/trajectory.txt (TUM, IMU poses); score it at the
board stops with `python -m tars_scan.slam_legs <session> <out>/trajectory.txt` and compare with
the offline Basalt run of the same session.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

G = 9.80665
CONFIG = Path(__file__).with_name("live_basalt_config.json")  # walk2-b-hyp-Ryday's VIO config


@dataclass(frozen=True)
class Imu:
    t_ns: int
    gyro: np.ndarray  # rad/s, IMU frame
    accel: np.ndarray  # m/s^2


@dataclass(frozen=True)
class Pair:
    t_ns: int  # IMU clock
    left: np.ndarray  # cam0, uint8, decoded at `scale`, gain and rolling-shutter fix applied
    right: np.ndarray


class CsvTail:
    """New complete rows of a CSV file that is still being written (header skipped)."""

    def __init__(self, path: Path, max_rows: int):
        self.path, self.f, self.header, self.max_rows = path, None, None, max_rows
        self.part = ""

    def rows(self) -> list[list[str]]:
        """At most max_rows rows per call, so a finished session streams instead of loading."""
        if self.f is None:
            if not self.path.exists():
                return []
            self.f = open(self.path, newline="")  # noqa: SIM115 — kept open while tailing
        out = []
        while len(out) < self.max_rows + (self.header is None):  # the header is not a row
            line = self.f.readline()
            if not line:
                break
            if not line.endswith("\n"):  # the writer is mid-line: keep it for the next call
                self.part += line
                break
            line, self.part = self.part + line, ""
            if line.strip():
                out.append(line.rstrip("\r\n").split(","))
        if self.header is None and out:
            self.header, out = out[0], out[1:]
        return out


class ImuClock:
    """Online version of tars_rig.check.fit_imu_clock: host time against FIFO sample count over
    the reads since the last overflow (at most `window` of them), moved down to the lower envelope.
    t(k) = period * (k + 1) + intercept, as imu_sample_times."""

    def __init__(self, window: int = 3000):
        self.h: collections.deque = collections.deque(maxlen=window)
        self.n: collections.deque = collections.deque(maxlen=window)
        self.fit = None

    def add(self, host_ns: float, total: float, overflow: bool) -> None:
        if overflow:
            self.h.clear()
            self.n.clear()
            self.fit = None
            return
        self.h.append(host_ns)
        self.n.append(total)
        if len(self.h) >= 10:
            n, h = np.array(self.n), np.array(self.h)
            slope, icpt = np.polyfit(n - n[0], h - h[0], 1)
            resid = (h - h[0]) - (slope * (n - n[0]) + icpt)
            self.fit = (slope, h[0] + icpt + resid.min() - slope * n[0])

    def time(self, k: np.ndarray) -> np.ndarray | None:
        if self.fit is None:
            return None
        return self.fit[0] * (k + 1) + self.fit[1]


class SessionSource:
    """Imu and Pair messages of one session, in time order (see the module docstring)."""

    def __init__(
        self,
        session: Path,
        scale: float = 0.5,
        follow: bool = False,
        realtime: float = 1.0,
        rs_correct: bool = True,
        gain: float | None = None,
    ):
        self.session, self.scale, self.follow, self.realtime = session, scale, follow, realtime
        self.rs, self.gain = rs_correct, gain
        self.stop = threading.Event()

    def _wait_meta(self) -> dict:
        p = self.session / "meta.json"
        while not p.exists():  # written once the cameras are metered and running
            if not self.follow or self.stop.wait(0.2):
                raise FileNotFoundError(p)
        return json.loads(p.read_text())

    def __iter__(self):
        from . import rs_correct as rsc
        from .export import SOF_EXPOSURE_FRACTION, _decode
        from .slam_calib import rig_model

        meta = self._wait_meta()
        model = rig_model(self.session)
        cams = meta["cameras"]
        fps, readout_s = cams[0]["fps"], (cams[0].get("readout_us") or 0) / 1e6
        acc_s = G / meta["imu"]["accel_lsb_per_g"]
        gyr_s = math.pi / 180 / meta["imu"]["gyro_lsb_per_dps"]
        R_c0_i = np.linalg.inv(model["cams"][0]["T_i_c"])[:3, :3]
        R_c1_c0 = (np.linalg.inv(model["cams"][1]["T_i_c"]) @ model["cams"][0]["T_i_c"])[:3, :3]
        offset = model["time_offset_ns"]
        size = (round(cams[0]["width"] * self.scale), round(cams[0]["height"] * self.scale))
        remaps = []
        if self.rs:
            for c in model["cams"]:
                K = c["K"].copy()
                K[:2] *= self.scale
                K[:2, 2] += 0.5 * self.scale - 0.5
                remaps.append(
                    rsc.LinearRemap(rsc.FisheyeRemap(K, c["D"], size), readout_s / size[1])
                )
        # chunks of about half a second each, so the streams advance together
        reads = CsvTail(self.session / "imu/reads.csv", 25)
        imu_tail = CsvTail(self.session / "imu/imu.csv", 100)
        cam_tails = [CsvTail(self.session / f"cam{i}/frames.csv", 15) for i in (0, 1)]
        jpg = [None, None]
        clock = ImuClock()
        raw_imu: collections.deque = collections.deque()  # (k, values) waiting for a clock
        gyro_hist: collections.deque = collections.deque(maxlen=400)  # (t, w_cam0) for RS
        frames: list[collections.deque] = [
            collections.deque(),
            collections.deque(),
        ]  # (t, off, size)
        imu_t = -math.inf  # time of the newest IMU sample sent
        lut = None if self.gain is None else _lut(self.gain)
        gain_samples: list[float] = []
        # one worker per camera: JPEG decode and cv2.remap release the GIL, and in one thread they
        # took a whole Pi 5 core at 30 fps (py-spy, 2026-11-11)
        pool = ThreadPoolExecutor(2, thread_name_prefix="cam")
        wall0 = t_first = None
        idle = 0
        while not self.stop.is_set():
            got = False
            for r in reads.rows():
                clock.add(float(r[0]), float(r[1]), r[3] == "1")
                got = True
            for r in imu_tail.rows():
                raw_imu.append((int(r[0]), np.array(r[2:8], float)))
                got = True
            out: list = []
            while raw_imu and clock.fit is not None:
                k, v = raw_imu.popleft()
                t = float(clock.time(np.array(k)))
                gyro, acc = v[3:6] * gyr_s, v[0:3] * acc_s
                gyro_hist.append((t, (gyro - model["gyro_bias"]) @ R_c0_i.T))
                out.append(Imu(int(round(t)), gyro, acc))
                imu_t = t
            for i, tail in enumerate(cam_tails):
                rows = tail.rows()
                got |= bool(rows)
                for r in rows:  # index, sensor_ns, host, exposure_us, gain, dur, sync, off, size
                    if r[6] != "1" or int(r[7]) < 0:
                        continue  # before the software sync locked, or dropped by the encoder
                    t = float(r[1]) + SOF_EXPOSURE_FRACTION * float(r[3]) * 1e3 + readout_s * 5e8
                    frames[i].append((t, int(r[7]), int(r[8])))
            # pair the oldest left frame once its right frame (or a later one) and the gyro past
            # its readout are in
            while frames[0] and frames[1]:
                t0 = frames[0][0][0]
                cand = min(frames[1], key=lambda f: abs(f[0] - t0))
                if abs(cand[0] - t0) > 0.5e9 / fps:
                    if frames[1][-1][0] < t0:
                        break  # the right frame may still come
                    frames[0].popleft()
                    continue
                ts = t0 + offset
                if imu_t < ts + readout_s * 5e8 + 5e6:
                    break  # gyro for the rolling-shutter fix not in yet
                f0 = frames[0].popleft()
                while frames[1] and frames[1][0][0] <= cand[0]:
                    frames[1].popleft()
                data = []
                for i, f in enumerate((f0, cand)):
                    if jpg[i] is None:
                        jpg[i] = open(self.session / f"cam{i}/frames.mjpeg", "rb")  # noqa: SIM115
                    jpg[i].seek(f[1])
                    data.append(jpg[i].read(f[2]))
                imgs = list(pool.map(_decode, data, (self.scale,) * 2))
                if lut is None:
                    gain_samples.append(float(np.percentile(imgs[0][::4, ::4], 99.5)))
                    if len(gain_samples) < 15:
                        continue  # the first half second only sets the gain (export: auto_gain)
                    self.gain = max(1.0, 230.0 / max(float(np.median(gain_samples)), 1.0))
                    lut = _lut(self.gain)
                if remaps:
                    g = np.array([w for t, w in gyro_hist if abs(t - ts) <= readout_s * 5e8])
                    w = g.mean(0) if len(g) else np.zeros(3)
                    omegas = (w, R_c1_c0 @ w)
                    imgs = list(pool.map(lambda r, im, om: r.correct(im, om), remaps, imgs, omegas))
                out.append(Pair(int(round(ts)), cv2.LUT(imgs[0], lut), cv2.LUT(imgs[1], lut)))
            out.sort(key=lambda m: m.t_ns)
            for m in out:
                if not self.follow and self.realtime > 0:  # replay at the walk's own pace
                    if wall0 is None:
                        wall0, t_first = time.perf_counter(), m.t_ns
                    ahead = (m.t_ns - t_first) / 1e9 / self.realtime - (time.perf_counter() - wall0)
                    if ahead > 0.002:
                        time.sleep(ahead)
                yield m
            if not got:
                idle += 1
                if not self.follow or (self.session / "done.json").exists() and idle > 50:
                    return
                self.stop.wait(0.02)  # tailing files a recorder is writing: 50 Hz checks
            else:
                idle = 0


def _lut(gain: float) -> np.ndarray:
    return np.clip(np.arange(256) * gain, 0, 255).astype(np.uint8)


STALL_S = 10.0  # a send blocked this long means Basalt stopped reading
IMU_LEAD_NS = 1_000_000_000  # < Basalt's 300-sample IMU queue (1.5 s at 200 Hz); see feed()
IMU_BEFORE_FIRST_NS = 100_000_000


class BasaltStalled(RuntimeError):
    pass


def run(a) -> None:
    import rerun as rr

    from .live_pose import BasaltClient, T_imu_cam0
    from .map_live_view import LiveLogger, blueprint
    from .occupancy_map import Mapper, Params, rect_and_matcher
    from .slam_calib import basalt_calib, rig_model

    session = Path(a.session)
    out = Path(a.out) if a.out else session / "live"
    out.mkdir(parents=True, exist_ok=True)
    p = replace(Params(), **dict(_kv(x) for x in a.set))
    src = SessionSource(session, a.scale, a.follow, a.realtime, rs_correct=not a.no_rs)
    meta = src._wait_meta()
    size = (
        round(meta["cameras"][0]["width"] * a.scale),
        round(meta["cameras"][0]["height"] * a.scale),
    )
    calib = basalt_calib(
        rig_model(session), (meta["cameras"][0]["width"], meta["cameras"][0]["height"]), a.scale
    )
    calib["value0"]["cam_time_offset_ns"] = 0  # the source already moved frames to the IMU clock
    T_ic0 = T_imu_cam0(calib)
    rect, sgbm = rect_and_matcher(session, a.scale, size, p)
    e = a.extent
    mapper = Mapper(p, np.array([-e, -e, -3.0]), np.array([e, e, 2.0]), rect, sgbm)

    rr.init("tars_live_map", recording_id=a.recording_id or session.name)
    sinks = [rr.FileSink(str(out / "live.rrd"))]
    if a.connect == "":  # a viewer on this machine (started unless one is already listening)
        import sys

        exe = Path(sys.executable).with_name("rerun")
        rr.spawn(connect=False, executable_path=str(exe) if exe.exists() else None)
        sinks.insert(0, rr.GrpcSink())
    elif a.connect != "none":
        sinks.insert(0, rr.GrpcSink(a.connect))
    rr.set_sinks(*sinks, default_blueprint=blueprint(["live"], {"live": f"Live: {session.name}"}))
    logger = LiveLogger("live", rect, sgbm, p, [], a.log_map_every, 0)

    basalt = BasaltClient(
        a.basalt_host, a.basalt_port, json.dumps(calib), Path(a.config).read_text()
    )
    todo: collections.deque = collections.deque(maxlen=2)  # latest pairs to map; older dropped
    cv = threading.Condition()
    done = threading.Event()
    stats = {"pairs": 0, "imu": 0}

    def feed():
        # Basalt's IMU queue holds 300 samples (1.5 s) and drains only as frames are processed, so
        # IMU far ahead of the frames blocks its socket reader for good. Walk2's IMU starts 6.3 s
        # before the first synced pair: IMU waits here, at most IMU_LEAD_NS past the newest pair.
        held: collections.deque = collections.deque()
        last_pair = None
        try:
            for m in src:
                if isinstance(m, Imu):
                    held.append(m)
                else:
                    if last_pair is None:  # Basalt discards IMU from before its first frame
                        while held and held[0].t_ns < m.t_ns - IMU_BEFORE_FIRST_NS:
                            held.popleft()
                    last_pair = m.t_ns
                    basalt.send_pair(m.t_ns, m.left, m.right)
                    stats["pairs"] += 1
                    if stats["pairs"] % a.map_every == 0:
                        with cv:
                            todo.append(m)
                            cv.notify()
                while held and last_pair is not None and held[0].t_ns <= last_pair + IMU_LEAD_NS:
                    i = held.popleft()
                    basalt.send_imu(i.t_ns, i.gyro, i.accel)
                    stats["imu"] += 1
        finally:
            basalt.end()
            done.set()
            with cv:
                cv.notify()

    feeder = threading.Thread(target=feed, name="feed", daemon=True)
    feeder.start()
    traj = open(out / "trajectory.txt", "w")  # noqa: SIM115
    t_first, mapped, skipped, ms = None, 0, 0, []
    print(f"live map: {session} -> {out} (Basalt {a.basalt_host}:{a.basalt_port})", flush=True)
    try:
        while True:
            with cv:
                while not cv.wait_for(lambda: todo or done.is_set(), timeout=2.0):
                    if basalt.stalled(STALL_S):
                        raise BasaltStalled(
                            f"a send to Basalt at {a.basalt_host}:{a.basalt_port} has blocked for "
                            f"over {STALL_S:.0f} s; it is not reading its socket (check "
                            "`docker logs tars-live-vio`)"
                        )
                if not todo:
                    break
                m = todo.popleft()
            T_wi = basalt.wait_pose(m.t_ns, timeout=3.0)
            if T_wi is None:
                skipped += 1
                continue
            t_first = t_first or m.t_ns
            c = time.perf_counter()
            ff = mapper.step((m.t_ns - t_first) / 1e9, T_wi @ T_ic0, m.left, m.right, stamp=m.t_ns)
            ms.append((time.perf_counter() - c) * 1000)
            logger(mapper.grid, ff)
            rr.log("stats/map_ms/live", rr.Scalars(ms[-1]))
            q = _quat(T_wi[:3, :3])
            traj.write(f"{m.t_ns / 1e9:.9f} {' '.join(f'{x:.6f}' for x in T_wi[:3, 3])} ")
            traj.write(" ".join(f"{x:.9f}" for x in q) + "\n")
            mapped += 1
            if mapped % 50 == 0:
                print(
                    f"t {(m.t_ns - t_first) / 1e9:6.1f} s  mapped {mapped}  skipped {skipped}  "
                    f"map {np.median(ms[-50:]):.0f} ms  pairs {stats['pairs']}",
                    flush=True,
                )
    except (KeyboardInterrupt, BasaltStalled):
        src.stop.set()
        raise
    finally:
        traj.close()
    summary = {
        "session": session.name,
        "pairs_sent": stats["pairs"],
        "imu_sent": stats["imu"],
        "mapped": mapped,
        "skipped_no_pose": skipped,
        "map_ms_median": round(float(np.median(ms)), 1) if ms else None,
        "gain": src.gain,
        "rs_correct": src.rs,
    }
    (out / "live.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def _quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()


def _kv(x: str):
    k, v = x.split("=", 1)
    return k, json.loads(v) if v not in ("True", "False") else v == "True"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tars_scan.live_map", description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--follow", action="store_true", help="the session is being recorded now")
    ap.add_argument("--realtime", type=float, default=1.0, help="replay speed (finished sessions)")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--no-rs", action="store_true", help="skip the rolling-shutter correction")
    ap.add_argument("--map-every", type=int, default=6, help="map every n-th pair (6 = 5 Hz)")
    ap.add_argument(
        "--log-map-every", type=int, default=3, help="send the map every n fused frames"
    )
    ap.add_argument("--extent", type=float, default=6.0, help="map half-width around the start (m)")
    ap.add_argument("--basalt-host", default="127.0.0.1")
    ap.add_argument("--basalt-port", type=int, default=5077)
    ap.add_argument("--config", default=str(CONFIG), help="Basalt VIO config json")
    ap.add_argument("--connect", default="", help="Rerun viewer URL ('' = local, none = file only)")
    ap.add_argument("--recording-id", default=None)
    ap.add_argument("--out", default=None, help="default <session>/live")
    ap.add_argument("--set", action="append", default=[], help="Params override, e.g. l_occ=2.5")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
