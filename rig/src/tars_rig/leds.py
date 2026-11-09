"""Status light: green = armed, blinking green = recording, red = error, red flashes = pressed with
nothing armed.

GpioLeds drives the red/green LED on two header pins; ActLed falls back to the Pi's own green LED
(no red, so errors show as fast blinking); NullLeds is for tests and the Mac.
"""

from __future__ import annotations

from pathlib import Path

STATES = ("off", "armed", "recording", "error", "refused")


class NullLeds:
    def __init__(self):
        self.history: list[str] = []

    def show(self, state: str) -> None:
        assert state in STATES, state
        self.history.append(state)


class GpioLeds:
    def __init__(self, green_pin: int, red_pin: int, common_anode: bool = False, pin_factory=None):
        from gpiozero import LED

        high = not common_anode
        self.green, self.red = (
            LED(green_pin, active_high=high, pin_factory=pin_factory),
            LED(red_pin, active_high=high, pin_factory=pin_factory),
        )

    def show(self, state: str) -> None:
        g, r = self.green, self.red
        g.off()
        r.off()
        if state == "armed":
            g.on()
        elif state == "recording":
            g.blink(on_time=0.25, off_time=0.25)
        elif state == "error":
            r.on()
        elif state == "refused":
            r.blink(on_time=0.15, off_time=0.15, n=3)


class ActLed:
    """The Pi 5's green ACT LED through sysfs (the container runs privileged with /sys writable)."""

    def __init__(self, path: Path = Path("/sys/class/leds/ACT")):
        self.path = path

    def _write(self, name: str, value: str) -> None:
        (self.path / name).write_text(value)

    def show(self, state: str) -> None:
        if state in ("off", "armed"):
            self._write("trigger", "none")
            self._write("brightness", "1" if state == "armed" else "0")
        else:  # recording slow, error/refused fast
            self._write("trigger", "timer")
            on = "250" if state == "recording" else "60"
            self._write("delay_on", on)
            self._write("delay_off", on)
