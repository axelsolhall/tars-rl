"""Open the TARS body in the MuJoCo viewer. Run: python -m tars_rl.body.view [--demo]

Without --demo the actuators hold the neutral pose; drag the sliders in the viewer's
Control panel to move each hinge and slide. With --demo a scripted motion swings both arms
on the centre hinge and slides them along their length.
"""

from __future__ import annotations

import argparse
import math

import mujoco
import mujoco.viewer

from tars_rl.body.mjcf import load_model, load_spec


def demo_ctrl(t: float, neutral: list[float], slide_max: float) -> list[float]:
    swing = 0.5 * math.sin(2 * math.pi * 0.4 * t)
    lift = 0.8 * slide_max * max(0.0, math.sin(2 * math.pi * 0.4 * t + math.pi / 2))
    # left_hinge, right_hinge, left_slide, right_slide
    return [neutral[0] + swing, neutral[1] - swing, neutral[2] + lift, neutral[3] + lift]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--demo", action="store_true", help="play a scripted arm motion")
    p.add_argument("--spec", help="BodySpec JSON (default: committed tars_v1.json)")
    a = p.parse_args()
    spec = load_spec(a.spec)
    model = load_model(spec)
    data = mujoco.MjData(model)
    data.ctrl[:] = spec.neutral_ctrl()
    if a.demo:
        neutral, smax = spec.neutral_ctrl(), spec.slide_range[1]

        def cb(m: mujoco.MjModel, d: mujoco.MjData) -> None:
            d.ctrl[:] = demo_ctrl(d.time, neutral, smax)

        mujoco.set_mjcb_control(cb)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
