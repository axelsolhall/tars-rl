"""Add the arm-mechanism parameters to the open "TARS_arm_mechanism" design (refuses any other document).

`fit` is left out on purpose: Axel adds that one by hand.
"""

import adsk.core
import adsk.fusion

MECH_PARAMS = [
    ("axle_d", "12 mm", "mm", "axle tube outer diameter"),
    ("brg_od", "21 mm", "mm", "6801 bearing outer diameter"),
    ("brg_w", "5 mm", "mm", "6801 bearing width"),
    ("slide_fit", "0.3 mm", "mm", "gap carrier <-> sleeve inside, per side"),
    ("belt_w", "10 mm", "mm", "hinge GT2 belt width"),
    ("pulley_t", "60", "", "GT2 pulley teeth (hinge, 1:1)"),
    ("arm_in_w", "59.6 mm", "mm", "arm sleeve inside width (arm_w - 2*wall_t at baby 1.05)"),
    ("in_d", "125.25 mm", "mm", "inside depth, arm and central (depth - 2*wall_t)"),
    ("in_h", "494 mm", "mm", "inside height (height - 2*wall_t)"),
    ("central_in_y", "39.8 mm", "mm", "arm centre to central body inside wall (arm_w/2 + arm_gap + wall_t)"),
]


def run(_context: str):
    app = adsk.core.Application.get()
    if app.activeDocument.name.split(" v")[0] != "TARS_arm_mechanism":
        raise RuntimeError(f"active document is {app.activeDocument.name!r}, not TARS_arm_mechanism")
    design = adsk.fusion.Design.cast(app.activeProduct)
    for name, expr, unit, comment in MECH_PARAMS:
        existing = design.userParameters.itemByName(name)
        if existing:
            existing.expression = expr
            existing.comment = comment
        else:
            design.userParameters.add(name, adsk.core.ValueInput.createByString(expr), unit, comment)
    print("parameters:", ", ".join(f"{p.name}={p.expression}" for p in design.userParameters))
