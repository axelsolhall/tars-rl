"""`rig` commands. On the Pi they run inside the container (`docker exec tars-rig rig ...`); the Mac
wrapper scripts/rig forwards them over SSH, except `check`, which runs wherever the session files
are.

  rig daemon                       the container's main process: button, LED, sessions
  rig arm calibrate|imu|capture    choose what the next button press records
  rig disarm
  rig start | rig stop             the same as pressing the button, from a terminal
  rig status                       what the rig is doing, and the last session's check
  rig list                         recorded sessions with their check summary
  rig check <session> [--json]     the eval of one session
  rig find-button [seconds]        press the button; prints which GPIO changed
  rig imu-test [seconds]           read the IMU and print rate, gravity and gyro numbers
  rig cam-test                     list the cameras libcamera sees
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from . import check
from .config import MODES, load
from .daemon import Rig, read_armed, write_armed, write_request


def header_pin_factory():
    """gpiozero pin factory on the 40-pin header's chip.

    That is RP1 on the Pi 5; its gpiochip number changed across kernels, so look it up by label."""
    import lgpio
    from gpiozero.pins.lgpio import LGPIOFactory

    for n in range(16):
        try:
            h = lgpio.gpiochip_open(n)
        except lgpio.error:
            continue
        try:
            # (status, lines, name, label), e.g. (0, 54, 'gpiochip0', 'pinctrl-rp1')
            _, _, name, label = lgpio.gpio_get_chip_info(h)
        finally:
            lgpio.gpiochip_close(h)
        if "rp1" in label.lower() or "rp1" in name.lower():
            return LGPIOFactory(chip=n)
    raise RuntimeError(
        "no gpiochip labelled rp1; is this a Raspberry Pi 5, "
        "and does the container see /dev/gpiochip*?"
    )


def cmd_daemon(cfg) -> None:
    from gpiozero import Button

    from .leds import ActLed, GpioLeds
    from .sessions import run_session

    factory = header_pin_factory()
    leds = (
        GpioLeds(cfg.pins.led_green, cfg.pins.led_red, cfg.pins.led_common_anode, factory)
        if cfg.pins.led_green >= 0
        else ActLed()
    )
    rig = Rig(cfg, leds, run_session)
    button = Button(cfg.pins.button, pull_up=True, bounce_time=0.05, pin_factory=factory)
    button.when_pressed = rig.press
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    logging.info(
        "rig daemon up: button GPIO%d, armed=%s", cfg.pins.button, read_armed(cfg.state_dir)
    )
    try:
        rig.run_forever(stop)
    except KeyboardInterrupt:
        pass


def cmd_status(cfg) -> None:
    path = cfg.state_dir / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {"state": "daemon has not run"}
    print(
        f"armed: {read_armed(cfg.state_dir) or '-'}   state: {status.get('state')}"
        + (f" ({status['mode']}, {status['elapsed_s']} s)" if status.get("mode") else "")
    )
    last = status.get("last")
    if last:
        print(
            f"last: {Path(last.get('session', '?')).name}  "
            f"{last.get('check') or last.get('stop_reason')}"
        )


def cmd_list(cfg) -> None:
    for d in sorted(cfg.sessions_dir.glob("*")) if cfg.sessions_dir.exists() else []:
        c = d / "check.json"
        line = check.summary_line(json.loads(c.read_text())) if c.exists() else "(no check.json)"
        print(f"{d.name}  {line}")


def cmd_check(path: str, as_json: bool) -> None:
    report = check.check_session(Path(path))
    print(json.dumps(report, indent=2) if as_json else check.summary_line(report))
    if not as_json:
        for key in ("cam0", "cam1", "stereo", "imu"):
            if key in report:
                print(f"  {key}: {report[key]}")


def cmd_find_button(cfg, seconds: float) -> None:
    from gpiozero import Button

    factory = header_pin_factory()
    skip = {2, 3, cfg.pins.led_green, cfg.pins.led_red}  # I2C and our outputs
    buttons = []
    for pin in range(2, 28):
        if pin in skip:
            continue
        try:
            b = Button(pin, pull_up=True, bounce_time=0.05, pin_factory=factory)
        except Exception:  # noqa: BLE001 — pins taken by something else are simply not watched
            continue
        b.when_pressed = lambda b=b: print(f"pressed: GPIO{b.pin.number}", flush=True)
        buttons.append(b)
    print(
        f"watching {len(buttons)} pins for {seconds:.0f} s (button wired to GND); press it now",
        flush=True,
    )
    time.sleep(seconds)


def cmd_imu_test(cfg, seconds: float) -> None:
    from .mpu6050 import (
        SAMPLE_BYTES,
        Mpu6050,
        accel_lsb_per_g,
        chip_name,
        gyro_lsb_per_dps,
        parse_fifo,
    )

    chip = Mpu6050(cfg.imu.bus, cfg.imu.address)
    print(f"WHO_AM_I 0x{chip.who:02x}: {chip_name(chip.who)}")
    rate = chip.setup(cfg.imu.rate_hz, cfg.imu.accel_range_g, cfg.imu.gyro_range_dps, cfg.imu.dlpf)
    samples, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        time.sleep(cfg.imu.poll_ms / 1000)
        n = min(chip.fifo_count(), 1020) // SAMPLE_BYTES * SAMPLE_BYTES
        if n:
            samples += parse_fifo(chip.read_fifo(n))
    a = [[v / accel_lsb_per_g(cfg.imu.accel_range_g) for v in s[:3]] for s in samples]
    g = [[v / gyro_lsb_per_dps(cfg.imu.gyro_range_dps) for v in s[3:]] for s in samples]
    n = max(len(samples), 1)
    gyro_mean = [round(sum(c) / n, 3) for c in zip(*g, strict=True)]
    norm = sum((x * x + y * y + z * z) ** 0.5 for x, y, z in a) / n
    print(
        f"{len(samples)} samples in {seconds:.1f} s = {len(samples) / seconds:.1f} Hz "
        f"(set {rate} Hz); "
        f"|accel| {norm:.3f} g; gyro mean {gyro_mean} dps; "
        f"temp {chip.temperature():.1f} C"
    )


def cmd_cam_test() -> None:
    from picamera2 import Picamera2

    for info in Picamera2.global_camera_info():
        print(info)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="rig", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daemon")
    sub.add_parser("arm").add_argument("mode", choices=MODES)
    for name in ("disarm", "start", "stop", "status", "list", "cam-test"):
        sub.add_parser(name)
    c = sub.add_parser("check")
    c.add_argument("session")
    c.add_argument("--json", action="store_true")
    sub.add_parser("find-button").add_argument("seconds", nargs="?", type=float, default=30)
    sub.add_parser("imu-test").add_argument("seconds", nargs="?", type=float, default=3)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.cmd == "check":
        return cmd_check(args.session, args.json)
    cfg = load()
    if args.cmd == "daemon":
        cmd_daemon(cfg)
    elif args.cmd == "arm":
        write_armed(cfg.state_dir, args.mode)
        print(
            f"armed: {args.mode} (press the button to start, again to stop; cap "
            f"{cfg.sessions[args.mode].max_minutes} min)"
        )
    elif args.cmd == "disarm":
        write_armed(cfg.state_dir, None)
        print("disarmed")
    elif args.cmd in ("start", "stop"):
        write_request(cfg.state_dir, args.cmd)
        print(f"{args.cmd} requested")
    elif args.cmd == "status":
        cmd_status(cfg)
    elif args.cmd == "list":
        cmd_list(cfg)
    elif args.cmd == "find-button":
        cmd_find_button(cfg, args.seconds)
    elif args.cmd == "imu-test":
        cmd_imu_test(cfg, args.seconds)
    elif args.cmd == "cam-test":
        cmd_cam_test()


if __name__ == "__main__":
    sys.exit(main())
