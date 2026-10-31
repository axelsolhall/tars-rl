"""Arena = the body's MJCF (T3, A8) plus a checkered flag on the same flat infinite floor (A50).

The flag is a mocap body: the env moves it per episode through `mocap_pos`, it never collides,
and any renderer that loads this XML draws it where the rollout file says it stood (A51).
"""

from __future__ import annotations

from pathlib import Path

from tars_rl.body.mjcf import ASSETS, build_mjcf
from tars_rl.body.spec import BodySpec, default_spec

FLAG_BODY = "flag"


def load_body_spec(path: str | Path | None = None) -> BodySpec:
    """The committed BodySpec JSON if present (A8), else the generator's default."""
    if path:
        return BodySpec.load(path)
    committed = ASSETS / "tars_v1.json"
    return BodySpec.load(committed) if committed.exists() else default_spec()


def _f(*xs: float) -> str:
    return " ".join(f"{x:.6g}" for x in xs)


def build_arena_mjcf(spec: BodySpec) -> str:
    xml = build_mjcf(spec)
    h = spec.central.height
    pole_h, pole_r = 1.2 * h, 0.006
    cloth = (0.002, 0.18 * h, 0.12 * h)  # half sizes: thin along x, out along y, down along z
    assets = """    <texture name="flag_checker" type="2d" builtin="checker" rgb1="1 1 1" rgb2="0 0 0"
             width="64" height="64"/>
    <material name="flag_checker" texture="flag_checker" texrepeat="3 2"/>
    <material name="flag_pole" rgba=".85 .85 .85 1"/>
"""
    flag = f"""    <body name="{FLAG_BODY}" mocap="true" pos="0 0 0">
      <geom name="flag_pole" type="box" size="{_f(pole_r, pole_r, pole_h / 2)}"
            pos="0 0 {_f(pole_h / 2)}" contype="0" conaffinity="0" mass="0"
            material="flag_pole"/>
      <geom name="flag_cloth" type="box" size="{_f(*cloth)}"
            pos="0 {_f(cloth[1] + pole_r)} {_f(pole_h - cloth[2])}" contype="0" conaffinity="0"
            mass="0" material="flag_checker"/>
    </body>
"""
    for anchor in ("  </asset>", "  </worldbody>"):
        if xml.count(anchor) != 1:
            raise ValueError(f"body MJCF layout changed: expected one '{anchor.strip()}'")
    xml = xml.replace("  </asset>", assets + "  </asset>")
    return xml.replace("  </worldbody>", flag + "  </worldbody>")
