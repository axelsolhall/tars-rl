import struct

import pytest
from tars_rig import mpu6050


def test_parse_fifo_big_endian_signed():
    data = struct.pack(">6h", 1, -2, 16384, -32768, 32767, 0) + struct.pack(">6h", *range(6))
    assert mpu6050.parse_fifo(data) == [(1, -2, 16384, -32768, 32767, 0), (0, 1, 2, 3, 4, 5)]


def test_parse_fifo_rejects_partial_sample():
    with pytest.raises(ValueError):
        mpu6050.parse_fifo(b"\x00" * 13)


def test_scale_factors_match_datasheet():
    assert mpu6050.accel_lsb_per_g(2) == 16384 and mpu6050.accel_lsb_per_g(4) == 8192
    assert mpu6050.gyro_lsb_per_dps(250) == 131 and mpu6050.gyro_lsb_per_dps(500) == 65.5
    assert mpu6050.temp_c(0) == pytest.approx(36.53)


def test_mpu6500_temperature_and_name():
    assert mpu6050.temp_c(0, 0x70) == pytest.approx(21.0)
    assert mpu6050.temp_c(3339, 0x70) == pytest.approx(31.0, abs=0.01)
    assert mpu6050.chip_name(0x70) == "MPU6500" and mpu6050.chip_name(0x68) == "MPU6050"
