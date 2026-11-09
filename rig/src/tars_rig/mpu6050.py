"""MPU6050 (GY-521 board) over I2C, sampled through its FIFO so no sample is lost between reads.

Register numbers are from the InvenSense MPU-6000/6050 register map (RM-MPU-6000A-00 rev 4.2). The
MPU6500 family (the first rig's GY-521 reports WHO_AM_I 0x70) shares every register used here,
except that its accelerometer has its own low-pass register and its temperature scale differs
(MPU-6500 register map RM-MPU-6500A-00).
"""

from __future__ import annotations

import struct

PWR_MGMT_1, SMPLRT_DIV, CONFIG, GYRO_CONFIG, ACCEL_CONFIG = 0x6B, 0x19, 0x1A, 0x1B, 0x1C
FIFO_EN, USER_CTRL, INT_STATUS, WHO_AM_I = 0x23, 0x6A, 0x3A, 0x75
FIFO_COUNT_H, FIFO_R_W, TEMP_OUT_H = 0x72, 0x74, 0x41
FIFO_EN_ACCEL_GYRO = 0x78  # XG, YG, ZG and ACCEL into the FIFO (no temperature)
USER_CTRL_FIFO_EN, USER_CTRL_FIFO_RESET = 0x40, 0x04
INT_FIFO_OFLOW = 0x10
ACCEL_CONFIG2 = 0x1D  # MPU6500 family only: A_DLPF_CFG; on the MPU6050 CONFIG filters both
MPU6500_FAMILY = {0x70: "MPU6500", 0x71: "MPU9250", 0x73: "MPU9255"}
SAMPLE_BYTES = 12  # ax ay az gx gy gz, big-endian int16, in that order
FIFO_SIZE = 1024  # the MPU6500 has 512; reads take min(fifo_count, ...) so either works
GYRO_OUTPUT_HZ = 1000  # with the DLPF on (CONFIG 1..6)

ACCEL_FS = {2: 0, 4: 1, 8: 2, 16: 3}
GYRO_FS = {250: 0, 500: 1, 1000: 2, 2000: 3}


def accel_lsb_per_g(range_g: int) -> float:
    return 16384.0 / (range_g / 2)


def gyro_lsb_per_dps(range_dps: int) -> float:
    return 131.0 / (range_dps / 250)


def parse_fifo(data: bytes) -> list[tuple[int, int, int, int, int, int]]:
    """Whole 12-byte samples only; a partial tail is a caller bug, not data."""
    if len(data) % SAMPLE_BYTES:
        raise ValueError(f"FIFO read of {len(data)} bytes is not a multiple of {SAMPLE_BYTES}")
    return list(struct.iter_unpack(">6h", data))


def chip_name(who: int) -> str:
    return MPU6500_FAMILY.get(who) or ("MPU6050" if who == 0x68 else f"MPU6050 clone 0x{who:02x}")


def temp_c(raw: int, who: int = 0x68) -> float:
    if who in MPU6500_FAMILY:
        return raw / 333.87 + 21.0
    return raw / 340.0 + 36.53


class Mpu6050:
    def __init__(self, bus: int, address: int):
        from smbus2 import SMBus  # imported here so the rest of the package runs on a Mac

        self.bus, self.address = SMBus(bus), address
        self.who = self.bus.read_byte_data(address, WHO_AM_I)

    def setup(self, rate_hz: int, accel_range_g: int, gyro_range_dps: int, dlpf: int) -> int:
        """Configure and start the FIFO. Returns the real sample rate (1 kHz / integer divider)."""
        div = max(0, round(GYRO_OUTPUT_HZ / rate_hz) - 1)
        w = self.bus.write_byte_data
        w(
            self.address, PWR_MGMT_1, 0x01
        )  # wake, clock from the X gyro PLL (steadier than 8 MHz RC)
        w(self.address, CONFIG, dlpf & 0x07)
        if self.who in MPU6500_FAMILY:  # else its accel runs unfiltered (~218 Hz) and aliases
            w(self.address, ACCEL_CONFIG2, dlpf & 0x07)
        w(self.address, SMPLRT_DIV, div)
        w(self.address, GYRO_CONFIG, GYRO_FS[gyro_range_dps] << 3)
        w(self.address, ACCEL_CONFIG, ACCEL_FS[accel_range_g] << 3)
        w(self.address, FIFO_EN, FIFO_EN_ACCEL_GYRO)
        self.reset_fifo()
        return GYRO_OUTPUT_HZ // (div + 1)

    def reset_fifo(self) -> None:
        self.bus.write_byte_data(self.address, USER_CTRL, USER_CTRL_FIFO_RESET)
        self.bus.write_byte_data(self.address, USER_CTRL, USER_CTRL_FIFO_EN)
        # Reading INT_STATUS clears it: an overflow flag left by an earlier run (e.g. imu-test,
        # which leaves the FIFO running) would otherwise read as an overflow at sample 0.
        self.bus.read_byte_data(self.address, INT_STATUS)

    def overflowed(self) -> bool:
        return bool(self.bus.read_byte_data(self.address, INT_STATUS) & INT_FIFO_OFLOW)

    def fifo_count(self) -> int:
        hi, lo = self.bus.read_i2c_block_data(self.address, FIFO_COUNT_H, 2)
        return (hi << 8) | lo

    def read_fifo(self, nbytes: int) -> bytes:
        """Burst-read nbytes from FIFO_R_W (the chip keeps the address fixed on the FIFO)."""
        from smbus2 import i2c_msg

        write, read = i2c_msg.write(self.address, [FIFO_R_W]), i2c_msg.read(self.address, nbytes)
        self.bus.i2c_rdwr(write, read)
        return bytes(read)

    def temperature(self) -> float:
        hi, lo = self.bus.read_i2c_block_data(self.address, TEMP_OUT_H, 2)
        return temp_c(struct.unpack(">h", bytes([hi, lo]))[0], self.who)
