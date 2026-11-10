#!/usr/bin/env bash
# One-time setup of a fresh Raspberry Pi OS Lite (64-bit) for the rig. Run from the Mac:
#   ssh axel@tars-rig.local 'bash -s' < rig/pi_setup.sh
# then reboot the Pi (sudo reboot) so the I2C settings take effect.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y docker.io i2c-tools
sudo usermod -aG docker "$USER"

# I2C for the MPU6050, at 400 kHz: a 200 Hz FIFO drain of 12-byte samples is ~20 kbit/s of payload,
# but faster reads keep each burst short.
sudo raspi-config nonint do_i2c 0
CONFIG=/boot/firmware/config.txt
grep -q '^dtparam=i2c_arm_baudrate=' "$CONFIG" || echo 'dtparam=i2c_arm_baudrate=400000' | sudo tee -a "$CONFIG"

# Name both sensors explicitly: on the first rig (2026-11-09) camera_auto_detect found neither
# Waveshare IMX219, while these overlays bound both.
sudo sed -i 's/^camera_auto_detect=1/camera_auto_detect=0/' "$CONFIG"
grep -q '^dtoverlay=imx219,cam0' "$CONFIG" || printf 'dtoverlay=imx219,cam0\ndtoverlay=imx219,cam1\n' | sudo tee -a "$CONFIG"

# Session data lives outside the container, so redeploying never touches recordings.
sudo mkdir -p /srv/rig
sudo chown "$USER" /srv/rig

# Write recordings to the SD card continuously instead of in bursts. With the default ratios the
# kernel let ~450 MB pile up and flushed it every ~35 s; appends to half-written pages then waited
# for the whole flush, stalling the recorders 2-3 s (frames lost, IMU FIFO overflowed). With 16/64 MB
# a 150 s calibrate session at 2x8 MB/s lost nothing (2026-11-10).
printf 'vm.dirty_background_bytes = 16777216\nvm.dirty_bytes = 67108864\nvm.dirty_writeback_centisecs = 100\n' \
  | sudo tee /etc/sysctl.d/90-tars-rig.conf
sudo systemctl restart systemd-sysctl

echo "done; now: sudo reboot, then rig/deploy.sh on the Mac"
