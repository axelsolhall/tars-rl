"""Reference boxes in TARS_arm_mechanism: the outside of the central body and of the right arm, from user parameters.

Frame: origin at the body centre on the hinge axis, X forward, Y along the axle (right arm at -arm_offset), Z up.
The boxes live in the `ref` component, are see-through, and are rebuilt when this runs again.
"""

import adsk.core
import adsk.fusion

REF_PARAMS = [
    ("height", "500 mm", "mm", "robot height (TARS 50)"),
    ("body_d", "131.25 mm", "mm", "depth of central body and arms, X (TARS 50 depth at baby 1.05)"),
    ("central_w", "131.25 mm", "mm", "central body width, Y (TARS 50 at baby 1.05)"),
    ("arm_w", "65.625 mm", "mm", "arm width, Y (TARS 50 at baby 1.05)"),
]


def val(expr):
    return adsk.core.ValueInput.createByString(expr)


def box(design, comp, name, cy_expr, w_expr):
    """Box centred at (0, cy, 0): body_d along X, w along Y, height along Z (symmetric about the hinge)."""
    ev = lambda e: design.unitsManager.evaluateExpression(e, "cm")  # noqa: E731
    sk = comp.sketches.add(comp.xYConstructionPlane)
    sk.name = name + " outline"
    hx, cy, hy = ev("body_d") / 2, ev(cy_expr), ev(w_expr) / 2
    lines = sk.sketchCurves.sketchLines.addTwoPointRectangle(
        adsk.core.Point3D.create(-hx, cy - hy, 0), adsk.core.Point3D.create(hx, cy + hy, 0))
    corners = {}
    for ln in lines:
        for p in (ln.startSketchPoint, ln.endSketchPoint):
            corners[(p.geometry.x > 0, p.geometry.y > cy)] = p
    dims, origin = sk.sketchDimensions, sk.originPoint
    hor = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    ver = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    # Distance dimensions are unsigned; the sketch starts at the true size, so abs() keeps each corner on its side.
    for key, sign in (((True, True), "+"), ((False, False), "-")):
        p = corners[key]
        g = p.geometry
        dims.addDistanceDimension(origin, p, hor, adsk.core.Point3D.create(g.x, g.y + 1, 0)).parameter.expression = "body_d / 2"
        dims.addDistanceDimension(origin, p, ver, adsk.core.Point3D.create(g.x + 1, g.y, 0)).parameter.expression = f"abs(({cy_expr}) {sign} ({w_expr}) / 2)"
    ext = comp.features.extrudeFeatures
    inp = ext.createInput(sk.profiles.item(0), adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    inp.setSymmetricExtent(val("height"), True)
    body = ext.add(inp).bodies.item(0)
    body.name = name
    body.opacity = 0.15
    return body


def run(_context: str):
    app = adsk.core.Application.get()
    if app.activeDocument.name.split(" v")[0] != "TARS_arm_mechanism":
        raise RuntimeError("wrong document: " + app.activeDocument.name)
    design = adsk.fusion.Design.cast(app.activeProduct)
    for name, expr, unit, comment in REF_PARAMS:
        if not design.userParameters.itemByName(name):
            design.userParameters.add(name, val(expr), unit, comment)
    ref = design.allComponents.itemByName("ref")
    if ref is None:
        ref = design.rootComponent.occurrences.addNewComponent(adsk.core.Matrix3D.create()).component
        ref.name = "ref"
    for b in list(ref.bRepBodies):
        b.deleteMe()
    for s in list(ref.sketches):
        s.deleteMe()
    for name, cy, w in (("central outside", "0 mm", "central_w"), ("right arm outside", "-arm_offset", "arm_w")):
        b = box(design, ref, name, cy, w)
        bb = b.boundingBox
        print(f"{name}: x {bb.minPoint.x*10:.2f}..{bb.maxPoint.x*10:.2f}  y {bb.minPoint.y*10:.2f}..{bb.maxPoint.y*10:.2f}  z {bb.minPoint.z*10:.1f}..{bb.maxPoint.z*10:.1f}")
