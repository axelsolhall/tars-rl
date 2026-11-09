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

echo "done; now: sudo reboot, then rig/deploy.sh on the Mac"
