"""tars-rig: the stereo + IMU capture rig on the Raspberry Pi 5 (two IMX219, one MPU6050, a button
and an LED).

Modules, each one job: config (settings), contracts (what a session directory holds), mpu6050 (the
IMU chip), imu / cameras (one recorder each, run as separate processes), sessions (one recording),
daemon (button and LED), check (the eval of a recorded session), cli (the `rig` command).
"""
