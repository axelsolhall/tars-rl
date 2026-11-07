"""Rough printed parts: hollow the central slab and the arm into closed shells of wall_t, and add
the arm's inner `carrier` box (hangs on the hinge, holds the slide servo; the arm sleeve slides
over it). Shapes are plain boxes for now: no rails, mounts, openings or fillets. Re-runnable."""
import json

import adsk.core
import adsk.fusion


def value(expr):
    return adsk.core.ValueInput.createByString(expr)


def sync_params(design):
    for p in json.loads(PARAMS_JSON)["parameters"]:  # noqa: F821 (injected by hardware/fusion.py)
        existing = design.userParameters.itemByName(p["name"])
        if existing:
            existing.expression = p["expr"]
        else:
            design.userParameters.add(p["name"], value(p["expr"]), p["unit"], p["comment"])


def shell(comp, body):
    """Hollow a body into a closed shell of wall_t (once; skips a body that is already hollow)."""
    if any(f.name == body.name + " shell" for f in comp.features.shellFeatures):
        return
    entities = adsk.core.ObjectCollection.create()
    entities.add(body)
    inp = comp.features.shellFeatures.createInput(entities, False)
    inp.insideThickness = value("wall_t")
    comp.features.shellFeatures.add(inp).name = body.name + " shell"


def centred_box(comp, dx, dy, dz, name):
    units = comp.parentDesign.unitsManager
    hx, hy = units.evaluateExpression(dx, "cm") / 2, units.evaluateExpression(dy, "cm") / 2
    sketch = comp.sketches.add(comp.xYConstructionPlane)
    sketch.name = name + " outline"
    lines = sketch.sketchCurves.sketchLines.addTwoPointRectangle(
        adsk.core.Point3D.create(-hx, -hy, 0), adsk.core.Point3D.create(hx, hy, 0))
    corners = {}
    for ln in lines:
        for pt in (ln.startSketchPoint, ln.endSketchPoint):
            corners[(pt.geometry.x > 0, pt.geometry.y > 0)] = pt
    dims, origin = sketch.sketchDimensions, sketch.originPoint
    H = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    V = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    for key, s in (((True, True), 1), ((False, False), -1)):
        pt = corners[key]
        dims.addDistanceDimension(origin, pt, H, adsk.core.Point3D.create(s * (hx + 1), 0, 0)).parameter.expression = f"({dx}) / 2"
        dims.addDistanceDimension(origin, pt, V, adsk.core.Point3D.create(0, s * (hy + 1), 0)).parameter.expression = f"({dy}) / 2"
    inp = comp.features.extrudeFeatures.createInput(sketch.profiles.item(0), adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    inp.setSymmetricExtent(value(dz), True)
    body = comp.features.extrudeFeatures.add(inp).bodies.item(0)
    body.name = name
    return body


def run(_context: str):
    design = adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct)
    sync_params(design)
    central = design.allComponents.itemByName("central")
    arm = design.allComponents.itemByName("arm")
    shell(central, central.bRepBodies.itemByName("central slab"))
    sleeve = arm.bRepBodies.itemByName("arm slab")
    sleeve.name = "arm sleeve"
    for f in arm.features.shellFeatures:
        if f.name == "arm slab shell":
            f.name = "arm sleeve shell"
    shell(arm, sleeve)

    carrier = design.allComponents.itemByName("carrier")
    if carrier is None:
        occ = arm.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        carrier = occ.component
        carrier.name = "carrier"
        body = centred_box(carrier, "carrier_d", "carrier_w", "carrier_h", "carrier box")
        shell(carrier, body)
    for name in ("central slab", "arm sleeve", "carrier box"):
        comp = {"central slab": central, "arm sleeve": arm, "carrier box": carrier}[name]
        b = comp.bRepBodies.itemByName(name)
        b.opacity = 0.25
        b.attributes.add("tars", "printed", "petg_density")
    print("printed:", ", ".join(f"{c.name}: {c.bRepBodies.item(0).name}" for c in (central, arm, carrier)))
