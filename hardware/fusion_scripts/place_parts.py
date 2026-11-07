"""Sync the user parameters and (re)place every bought part from hardware/tars_params.json as a block.

Each part becomes a sub-component of its parent (e.g. `central`), sized and positioned only by
parameter expressions, so it follows any parameter change. Its real mass is stored as an
attribute (group "tars", name "mass") for the export to the simulator; the block's own volume
does not stand for the part's mass. Running it again replaces the blocks.
"""

import json

import adsk.core
import adsk.fusion


def value(expr):
    return adsk.core.ValueInput.createByString(expr)


def sync_params(design, params):
    for p in params:
        existing = design.userParameters.itemByName(p["name"])
        if existing:
            existing.expression = p["expr"]
            existing.comment = p["comment"]
        else:
            design.userParameters.add(p["name"], value(p["expr"]), p["unit"], p["comment"])


def find_component(design, name):
    comp = design.allComponents.itemByName(name)
    if comp is None:
        raise RuntimeError(f"no component named {name!r}; build the blockout first (build_blockout.py)")
    return comp


def place(design, part):
    parent = find_component(design, part["parent"])
    for occ in list(parent.occurrences):
        if occ.component.name == part["name"]:
            occ.deleteMe()
    occ = parent.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    comp = occ.component
    comp.name = part["name"]

    units = design.unitsManager
    sx, sy, sz = part["size"]
    cx, cy = part["centre_x"], part["centre_y"]
    ev = lambda e: units.evaluateExpression(e, "cm")  # noqa: E731 (API lengths are cm)

    planes = comp.constructionPlanes
    plane_in = planes.createInput()
    plane_in.setByOffset(comp.xYConstructionPlane, value(part["bottom_z"]))
    plane = planes.add(plane_in)
    plane.name = part["name"] + " bottom"

    sketch = comp.sketches.add(plane)
    sketch.name = part["name"] + " outline"
    x0, x1 = ev(cx) - ev(sx) / 2, ev(cx) + ev(sx) / 2
    y0, y1 = ev(cy) - ev(sy) / 2, ev(cy) + ev(sy) / 2
    lines = sketch.sketchCurves.sketchLines.addTwoPointRectangle(
        adsk.core.Point3D.create(x0, y0, 0), adsk.core.Point3D.create(x1, y1, 0))
    corners = {}
    for ln in lines:
        for pt in (ln.startSketchPoint, ln.endSketchPoint):
            corners[(pt.geometry.x > ev(cx), pt.geometry.y > ev(cy))] = pt
    dims, origin = sketch.sketchDimensions, sketch.originPoint
    horizontal = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    vertical = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    # Distance dimensions are unsigned; the sketch starts at the true positions, so abs() keeps the side.
    for (hi_x, hi_y), sign in (((True, True), "+"), ((False, False), "-")):
        pt = corners[(hi_x, hi_y)]
        g = pt.geometry
        dims.addDistanceDimension(origin, pt, horizontal, adsk.core.Point3D.create(g.x, g.y + 1, 0)).parameter.expression = f"abs(({cx}) {sign} ({sx}) / 2)"
        dims.addDistanceDimension(origin, pt, vertical, adsk.core.Point3D.create(g.x + 1, g.y, 0)).parameter.expression = f"abs(({cy}) {sign} ({sy}) / 2)"

    extrudes = comp.features.extrudeFeatures
    ext_in = extrudes.createInput(sketch.profiles.item(0), adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ext_in.setDistanceExtent(False, value(sz))
    body = extrudes.add(ext_in).bodies.item(0)
    body.name = part["name"]
    comp.attributes.add("tars", "mass", part["mass"])
    return occ


def run(_context: str):
    design = adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct)
    spec = json.loads(PARAMS_JSON)  # noqa: F821 (injected by hardware/fusion.py)
    sync_params(design, spec["parameters"])
    for part in spec.get("parts", []):
        occ = place(design, part)
        bb = occ.bRepBodies.item(0).boundingBox
        span = [round(v * 10, 1) for v in (bb.minPoint.x, bb.maxPoint.x, bb.minPoint.y, bb.maxPoint.y, bb.minPoint.z, bb.maxPoint.z)]
        print(f"{part['name']} in {part['parent']}: x {span[0]}..{span[1]}  y {span[2]}..{span[3]}  z {span[4]}..{span[5]} mm, mass {part['mass']}")
