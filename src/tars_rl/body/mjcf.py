"""BodySpec -> MJCF. The generated XML plus the spec JSON are the body's contract (A8).

Regenerate the committed files with: python -m tars_rl.body.mjcf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tars_rl.body.spec import SIDES, BodySpec, Servo, default_spec

if TYPE_CHECKING:
    import mujoco

ASSETS = Path(__file__).parent / "assets"


def _f(*xs: float) -> str:
    return " ".join(f"{x:.10g}" for x in xs)


def _actuator(joint: str, servo: Servo, rng: tuple[float, float]) -> str:
    return (
        f'    <position name="{joint}_act" joint="{joint}" kp="{_f(servo.kp)}" '
        f'kv="{_f(servo.kv)}" ctrlrange="{_f(*rng)}" '
        f'forcerange="{_f(-servo.force_limit, servo.force_limit)}"/>'
    )


def build_mjcf(spec: BodySpec) -> str:
    c, a = spec.central, spec.arm
    # Neutral: every slab stands on the floor; the hinge axis sits at mid-height of the
    # slabs (A5) through the centre of the central body (A39).
    hinge_z = c.height / 2
    arm_y = c.width / 2 + spec.arm_gap + a.width / 2
    arms = []
    for side, sign in zip(SIDES, (1, -1), strict=True):
        arms.append(f"""
      <body name="{side}_arm" pos="0 {_f(sign * arm_y)} 0">
        <joint name="{side}_hinge" type="hinge" axis="0 1 0" range="{_f(*spec.hinge_range)}"/>
        <joint name="{side}_slide" type="slide" axis="0 0 1" range="{_f(*spec.slide_range)}"/>
        <geom name="{side}_arm_geom" type="box" size="{_f(a.depth / 2, a.width / 2, a.height / 2)}"
              mass="{_f(a.mass)}" material="arm"/>
        <site name="{side}_foot" pos="0 0 {_f(-a.height / 2)}"/>
      </body>""")
    acts = [_actuator(j, spec.hinge_servo if j.endswith("hinge") else spec.slide_servo,
                      spec.joint_range(j)) for j in spec.joint_names]
    return f"""<!-- Generated from BodySpec '{spec.name}' v{spec.version} by tars_rl.body.mjcf.
     Do not edit by hand. -->
<mujoco model="{spec.name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{_f(spec.timestep)}" integrator="implicitfast" solver="Newton"
          iterations="20" ls_iterations="10"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <default>
    <geom friction="{_f(spec.friction, 0.005, 0.0001)}" condim="3"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .2 .3"
             width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/>
    <material name="central" rgba=".55 .57 .6 1"/>
    <material name="arm" rgba=".7 .72 .75 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
    <body name="central" pos="0 0 {_f(hinge_z)}">
      <freejoint name="root"/>
      <geom name="central_geom" type="box" size="{_f(c.depth / 2, c.width / 2, c.height / 2)}"
            mass="{_f(c.mass)}" material="central"/>
      <site name="imu" pos="0 0 0"/>{"".join(arms)}
    </body>
  </worldbody>
  <contact>
    <exclude body1="central" body2="left_arm"/>
    <exclude body1="central" body2="right_arm"/>
  </contact>
  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
"""


def load_spec(path: str | Path | None = None) -> BodySpec:
    return BodySpec.load(path) if path else BodySpec.load(ASSETS / "tars_v1.json")


def load_model(spec: BodySpec | None = None) -> mujoco.MjModel:
    import mujoco  # lazy: generating the MJCF needs no simulator

    return mujoco.MjModel.from_xml_string(build_mjcf(spec or load_spec()))


def write_assets(spec: BodySpec, out: Path = ASSETS) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    j, x = out / f"{spec.name}.json", out / f"{spec.name}.xml"
    j.write_text(spec.to_json())
    x.write_text(build_mjcf(spec))
    return j, x


if __name__ == "__main__":
    spec = load_spec(sys.argv[1]) if len(sys.argv) > 1 else default_spec()
    for p in write_assets(spec):
        print(p)
