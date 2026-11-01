"""TARS body: BodySpec (the versioned contract), its MJCF, a viewer and a stand eval."""

from tars_rl.body.mjcf import build_mjcf, load_model, load_spec
from tars_rl.body.spec import BodySpec, default_spec

__all__ = ["BodySpec", "build_mjcf", "default_spec", "load_model", "load_spec"]
