"""Blocked-out TARS, fully driven by the user parameters from hardware/tars_params.json.

Creates a new design: the parameters as live expressions; a `central` component (grounded, its
origin is the hinge centre, floor at z = -hinge_z); one `arm` component inserted twice. Every body is centred on its component's origin, and the arms sit on
construction points inside `central` at (0, +-arm_offset, 0), so changing any parameter in
Modify > Change Parameters moves and resizes everything. The hinges are revolute joints about y.
"""

import json

import adsk.core
import adsk.fusion


def value(expr):
    return adsk.core.ValueInput.createByString(expr)


def add_params(design):
    for p in json.loads(PARAMS_JSON)["parameters"]:  # noqa: F821 (injected by hardware/fusion.py)
        existing = design.userParameters.itemByName(p["name"])
        if existing:
            existing.expression = p["expr"]
        else:
            design.userParameters.add(p["name"], value(p["expr"]), p["unit"], p["comment"])


def centred_box(comp, depth_expr, width_expr, height_expr, name):
    """A box centred on the component origin: depth along x, width along y, height along z."""
    sketch = comp.sketches.add(comp.xYConstructionPlane)
    sketch.name = name + " outline"
    # Start at the real size (API lengths are cm) so the dimensions don't have to drag it there.
    units = comp.parentDesign.unitsManager
    hx = units.evaluateExpression(depth_expr, "cm") / 2
    hy = units.evaluateExpression(width_expr, "cm") / 2
    lines = sketch.sketchCurves.sketchLines.addTwoPointRectangle(
        adsk.core.Point3D.create(-hx, -hy, 0), adsk.core.Point3D.create(hx, hy, 0))
    corners = {}
    for ln in lines:
        for pt in (ln.startSketchPoint, ln.endSketchPoint):
            corners[(pt.geometry.x > 0, pt.geometry.y > 0)] = pt
    dims = sketch.sketchDimensions
    origin = sketch.originPoint
    horizontal = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    vertical = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    for key, (dx, dy) in {(True, True): (hx + 2, hy + 2), (False, False): (-hx - 2, -hy - 2)}.items():
        pt = corners[key]
        dims.addDistanceDimension(origin, pt, horizontal, adsk.core.Point3D.create(dx, 0, 0)).parameter.expression = f"({depth_expr}) / 2"
        dims.addDistanceDimension(origin, pt, vertical, adsk.core.Point3D.create(0, dy, 0)).parameter.expression = f"({width_expr}) / 2"
    ext_input = comp.features.extrudeFeatures.createInput(
        sketch.profiles.item(0), adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ext_input.setSymmetricExtent(value(height_expr), True)
    body = comp.features.extrudeFeatures.add(ext_input).bodies.item(0)
    body.name = name
    return body


def offset_plane(comp, base, expr):
    planes = comp.constructionPlanes
    inp = planes.createInput()
    inp.setByOffset(base, value(expr))
    return planes.add(inp)


def point_on_planes(comp, a, b, c, name):
    points = comp.constructionPoints
    inp = points.createInput()
    inp.setByThreePlanes(a, b, c)
    pt = points.add(inp)
    pt.name = name
    return pt


def run(_context: str):
    app = adsk.core.Application.get()
    doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    design = adsk.fusion.Design.cast(app.activeProduct)
    design.designType = adsk.fusion.DesignTypes.ParametricDesignType
    root = design.rootComponent
    add_params(design)

    identity = adsk.core.Matrix3D.create()
    central_occ = root.occurrences.addNewComponent(identity)
    central = central_occ.component
    central.name = "central"
    centred_box(central, "depth", "central_w", "height", "central slab")

    arm_occ_left = root.occurrences.addNewComponent(identity)
    arm = arm_occ_left.component
    arm.name = "arm"
    centred_box(arm, "depth", "arm_w", "height", "arm slab")
    arm_occ_right = root.occurrences.addExistingComponent(arm, identity)

    # Frame: the origin is the hinge centre (both hinge axes run along y through it), so the floor
    # is at z = -hinge_z. Central is grounded there; a joint to a floor point conflicted with the hinges.
    central_occ.isGrounded = True
    joints = root.joints

    # Hinge points inside central, one per side.
    for side, sign, occ in (("left", "", arm_occ_left), ("right", "-", arm_occ_right)):
        plane = offset_plane(central, central.xZConstructionPlane, f"{sign}arm_offset")
        pt = point_on_planes(central, central.yZConstructionPlane, plane, central.xYConstructionPlane, f"{side} hinge")
        j = joints.createInput(
            adsk.fusion.JointGeometry.createByPoint(arm.originConstructionPoint.createForAssemblyContext(occ)),
            adsk.fusion.JointGeometry.createByPoint(pt.createForAssemblyContext(central_occ)))
        j.setAsRevoluteJointMotion(adsk.fusion.JointDirections.YAxisJointDirection)
        joints.add(j).name = f"{side} hinge"
        occ.component  # same component for both arms: edit once, both change

    app.activeViewport.fit()
    print("built", doc.name, "| parameters:", design.userParameters.count,
          "| occurrences:", [o.name for o in root.occurrences])
