"""Client for tars_live_vio (Basalt on a live TCP stream; ~/dev/slam/basalt/src/tars_live_vio.cpp).

Sends IMU samples and stereo pairs as they arrive and receives Basalt's state for each processed
pair (T_world_imu). Protocol (little-endian), contract v1:
  hello: b"TARSLIVE1", u32 n + calib json, u32 n + VIO config json
  b"I" i64 t_ns, 6 f64 (gyro xyz rad/s, accel xyz m/s^2)
  b"F" i64 t_ns, u16 w, u16 h, w*h u8 cam0, w*h u8 cam1   (t_ns in the IMU clock)
  b"E" end
  back: b"P" i64 t_ns, 7 f64 (p xyz, q xyzw)
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

POSE = struct.Struct("<cq7d")


def pose_matrix(p: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    M[:3, 3] = p
    return M


def T_imu_cam0(calib: dict) -> np.ndarray:
    """IMU <- camera 0 from a Basalt calibration json."""
    c = calib["value0"]["T_imu_cam"][0]
    return pose_matrix(
        np.array([c["px"], c["py"], c["pz"]]), np.array([c["qx"], c["qy"], c["qz"], c["qw"]])
    )


class BasaltClient:
    """One live session with tars_live_vio. poses: {t_ns: T_world_imu}; new_pose wakes waiters."""

    def __init__(self, host: str, port: int, calib_json: str, config_json: str):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.lock = threading.Lock()
        self.new_pose = threading.Condition()
        self.poses: dict[int, np.ndarray] = {}
        self.last_t = -1
        self.closed = False
        self.sending_since: float | None = None  # monotonic start of the send in progress
        hello = b"TARSLIVE1"
        for text in (calib_json, config_json):
            data = text.encode()
            hello += struct.pack("<I", len(data)) + data
        self.sock.sendall(hello)
        self.reader = threading.Thread(target=self._read, name="basalt-poses", daemon=True)
        self.reader.start()

    def send_imu(self, t_ns: int, gyro, accel) -> None:
        self._send(struct.pack("<cq6d", b"I", int(t_ns), *map(float, gyro), *map(float, accel)))

    def send_pair(self, t_ns: int, left: np.ndarray, right: np.ndarray) -> None:
        h, w = left.shape
        head = struct.pack("<cqHH", b"F", int(t_ns), w, h)
        self._send(head + np.ascontiguousarray(left).tobytes() + right.tobytes())

    def _send(self, data: bytes) -> None:
        with self.lock:
            self.sending_since = time.monotonic()
            self.sock.sendall(data)
            self.sending_since = None

    def stalled(self, seconds: float) -> bool:
        """True when one send has blocked for longer than seconds: Basalt stopped reading (its
        socket reader blocks when its IMU queue is full; live_map's feed keeps IMU near frames)."""
        t = self.sending_since
        return t is not None and time.monotonic() - t > seconds

    def end(self) -> None:
        with self.lock:
            self.sock.sendall(b"E")
        self.reader.join(timeout=30)
        self.sock.close()

    def wait_pose(self, t_ns: int, timeout: float) -> np.ndarray | None:
        """T_world_imu for the pair at t_ns, or None when Basalt skipped it (a later pose came
        first) or nothing arrived within timeout seconds."""
        with self.new_pose:
            self.new_pose.wait_for(
                lambda: t_ns in self.poses or self.last_t > t_ns or self.closed, timeout
            )
            return self.poses.pop(t_ns, None)

    def _read(self) -> None:
        f = self.sock.makefile("rb")
        try:
            while (buf := f.read(POSE.size)) and len(buf) == POSE.size:
                _, t, *v = POSE.unpack(buf)
                M = pose_matrix(np.array(v[:3]), np.array(v[3:]))
                with self.new_pose:
                    self.poses[t] = M
                    self.last_t = max(self.last_t, t)
                    for old in [k for k in self.poses if k < t - 5_000_000_000]:
                        del self.poses[old]  # nobody waits that long
                    self.new_pose.notify_all()
        finally:
            with self.new_pose:
                self.closed = True
                self.new_pose.notify_all()
