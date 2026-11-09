"""The button state machine with a fake recorder: press with nothing armed, arm, start, stop,
errors, CLI requests."""

from pathlib import Path

import pytest
from tars_rig.config import load
from tars_rig.daemon import Rig, read_armed, write_armed, write_request
from tars_rig.leds import NullLeds

CONFIG = Path(__file__).resolve().parents[1] / "config.toml"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("RIG_ROOT", str(tmp_path))
    cfg = load(CONFIG)
    runs = []

    def runner(cfg, mode, stop):
        runs.append(mode)
        stop.wait(5)
        return {
            "session": f"x-{mode}",
            "stop_reason": "stopped",
            "errors": [] if mode != "imu" else ["boom"],
        }

    r = Rig(cfg, NullLeds(), runner)
    r.runs = runs
    return r


def settle(rig, n=4):
    for _ in range(n):
        rig.step(timeout=0.05)


def test_press_with_nothing_armed_is_refused(rig):
    rig.press()
    settle(rig)
    assert rig.runs == [] and "refused" in rig.leds.history


def test_arm_press_press_records_once(rig):
    write_armed(rig.cfg.state_dir, "calibrate")
    settle(rig)
    assert rig.leds.history[-1] == "armed"
    rig.press()
    settle(rig)
    assert rig.recording and rig.leds.history[-1] == "recording"
    rig.press()
    rig.thread.join(2)
    settle(rig)
    assert rig.runs == ["calibrate"] and not rig.recording
    assert rig.leds.history[-1] == "armed"  # stays armed for the next take
    assert read_armed(rig.cfg.state_dir) == "calibrate"


def test_failed_session_shows_error_until_rearmed(rig):
    write_armed(rig.cfg.state_dir, "imu")
    rig.press()
    settle(rig)
    write_request(rig.cfg.state_dir, "stop")
    settle(rig)
    rig.thread.join(2)
    settle(rig)
    assert rig.leds.history[-1] == "error"
    write_armed(rig.cfg.state_dir, "capture")
    settle(rig)
    assert rig.leds.history[-1] == "armed"


def test_cli_start_does_not_toggle_off(rig):
    write_armed(rig.cfg.state_dir, "capture")
    write_request(rig.cfg.state_dir, "start")
    settle(rig)
    write_request(rig.cfg.state_dir, "start")
    settle(rig)
    assert rig.recording and rig.runs == ["capture"]
    rig.session_stop.set()
    rig.thread.join(2)


def test_status_file(rig):
    write_armed(rig.cfg.state_dir, "capture")
    rig.press()
    settle(rig)
    status = (rig.cfg.state_dir / "status.json").read_text()
    assert '"recording"' in status and '"capture"' in status
    rig.session_stop.set()
    rig.thread.join(2)


def test_config_loads():
    cfg = load(CONFIG)
    assert cfg.pins.button == 17 and cfg.imu.address == 0x68
    assert set(cfg.sessions) == {"calibrate", "imu", "capture"}
