"""Visualisation: Trajectory contract (A10) and quick-check MP4 rendering (A46).

Import this package before `mujoco`: MuJoCo reads MUJOCO_GL once, at its own import, so the
headless default below has no effect if mujoco is already loaded.
"""

import os
import sys

# Headless Linux (the Ubuntu box over SSH) has no display: pick EGL before mujoco is imported.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
