# tars-rig: stereo + IMU capture on the Raspberry Pi 5

Two Waveshare IMX219 cameras (120° FOV), one MPU6050 (GY-521), a button and a red/green LED. The
Pi runs one Docker container. The button starts and stops whichever session was armed from the
Mac. Sessions land in `/srv/rig/sessions/` on the Pi, and `scripts/rig pull` copies them to
`runs/rig/` on the Mac.

| Session | Records | Time cap | Used for |
|---|---|---|---|
| `calibrate` | both cameras + IMU | 5 min | the checkerboard film (move, pause, move) |
| `imu` | IMU only | 120 min | the still recording for IMU noise and bias (Allan variance) |
| `capture` | both cameras + IMU | 20 min | room sweeps for SLAM |

The caps are in `config.toml`. A press during a session stops it, and so does reaching the cap.

## Wiring

Header pin numbers are physical pins. Hold the Pi with the SD card pointing up: the left row is pins
1, 3, 5, ... from the top, and the right row is pins 2, 4, 6, ...

| Part | Leg | Pi header | Notes |
|---|---|---|---|
| Button | one side | pin 11 (GPIO17) | internal pull-up, no resistor |
| Button | other side | pin 9 (GND) | |
| LED | green (anode, long leg) | 330 Ω → pin 13 (GPIO27) | two separate LEDs or one bicolour; the pins are `led_green` / `led_red` in `config.toml` |
| LED | red (anode, long leg) | 330 Ω → pin 15 (GPIO22) | |
| LED | cathodes (short legs) | pin 14 (GND) | if it is common anode, use pin 17 (3.3 V) and set `led_common_anode = true` |
| MPU6050 | VCC | pin 1 (**3.3 V**) | never 5 V: the GY-521's I2C pull-ups go to VCC, and the Pi's pins take 3.3 V only |
| MPU6050 | GND | pin 6 | |
| MPU6050 | SDA | pin 3 (GPIO2) | |
| MPU6050 | SCL | pin 5 (GPIO3) | |
| MPU6050 | INT | pin 7 (GPIO4) | optional, not used yet |
| Cameras | ribbons | CAM0 and CAM1 | camera 0 is left (the sync server), camera 1 is right |

The LED shows the rig's state:

| LED | State |
|---|---|
| off | nothing armed |
| green | armed, waiting for a press |
| blinking green | recording |
| red | the last session had an error (arming again clears it) |
| three red flashes | pressed with nothing armed |

## First setup

1. Flash **Raspberry Pi OS Lite (64-bit)** with Raspberry Pi Imager. In the Imager settings:
   hostname `tars-rig`, user `axel`, your Wi-Fi, and SSH with your Mac's public key.
2. On the Mac, run `ssh axel@tars-rig.local 'bash -s' < rig/pi_setup.sh`. This installs Docker,
   enables I2C at 400 kHz and creates `/srv/rig`. Then reboot the Pi:
   `ssh axel@tars-rig.local sudo reboot`.
3. On the Mac, run `rig/deploy.sh`. It builds the image here (arm64, through colima), sends it over
   SSH and starts the container. The container restarts by itself after reboots.
4. Check the wiring:
   - `scripts/rig cam-test`: both cameras should be listed.
   - `scripts/rig imu-test`: about 200 Hz, |accel| near 1 g.
   - `scripts/rig find-button`: press the button and it prints GPIO17.

## Use

```
scripts/rig arm calibrate     # green LED: the next press records a calibration session
scripts/rig status            # what it is doing, and the last session's check
scripts/rig pull              # sessions to runs/rig/ on the Mac
scripts/rig check runs/rig/<session>
```

`scripts/rig start` and `scripts/rig stop` do the same as pressing the button. `scripts/rig logs`
follows the container log.

After code or config changes, run `rig/deploy.sh` again. It refuses while a session is recording,
unless you pass `--force`, which stops the session cleanly first.

## The session eval

`check.json` is written after every session, and `rig check <session>` prints it. It reports:

- **Per camera:** the measured fps, frame-interval spread and dropped frames, plus what the Pi
  spent: frames the encoder dropped, encode time, KB per frame and MB/s written to the card.
- **Stereo pair:** the left-right time offset (median, p95, max). This is the software-sync result,
  measured rather than assumed.
- **IMU:** the real sample rate fitted from the read log, drift against the nominal rate,
  overflows, read latency, and the still numbers (|accel|, gyro bias).

It has no pass marks: it reports and you judge. `rig/tests/` checks it against a synthetic session
with known faults. The tests run with the main suite (`pytest`).

## On the Mac: export and calibrate (`src/tars_scan/`)

```
scripts/rig export runs/rig/<session> runs/rig/<session>-euroc [--scale 0.5]
PYTHONPATH=rig/src .venv/bin/python -m tars_scan.calib runs/rig/<session>   # board: 11x8 squares of 25.03 mm
PYTHONPATH=rig/src .venv/bin/python -m tars_scan.select_eval     # the selection eval
```

- **`export`** writes the EuRoC layout that Basalt, OpenVINS, ORB-SLAM3 and Kalibr read.
- **`calib`** ranks frames by predicted smear from the gyro and searches the board most valuable
  first. It picks views by Max-logDet, then fits the fisheye model and the stereo pair. It writes
  `<session>/calib/calib.json` with held-out errors, plus a Kalibr-format `camchain.yaml`.

`docs/SCANNER.md` has the research behind this and the measured results.

## Session layout

See `src/tars_rig/contracts.py` (schema 2). Each camera writes one `frames.mjpeg`, which holds
back-to-back grayscale JPEGs, plus `frames.csv` with each frame's byte offset. Timestamps are
`CLOCK_MONOTONIC` nanoseconds on the Pi for both cameras and the IMU. Frames are 1640×1232, the
IMX219's full-field binned mode, at 30 fps. Each camera's rolling-shutter line time is in
`meta.json`.
