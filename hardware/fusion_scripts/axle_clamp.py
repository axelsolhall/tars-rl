"""Centre axle clamp for TARS_arm_mechanism: one half (body centre plane y = 0 toward the right arm, -Y), to be mirrored.

Split pillow block around the axle at the body centre: a thick bottom piece and a thinner top piece, each with a
half-circle cut for the tube, a gap between them so the screws squeeze the tube, and two M3 screws per half
(clearance through the top, pilot hole into the bottom). Everything is driven by user parameters. Running it again
replaces the clamp.
"""

import adsk.core
import adsk.fusion

CLAMP_PARAMS = [
    ("clamp_len", "60 mm", "mm", "full clamp length along Y (this half is clamp_len/2)"),
    ("clamp_w", "36 mm", "mm", "clamp size along X"),
    ("clamp_bot_h", "20 mm", "mm", "bottom piece: axle centre down to its underside"),
    ("clamp_top_h", "12 mm", "mm", "top piece: axle centre up to its top"),
    ("clamp_split", "1 mm", "mm", "gap between top and bottom piece so the screws squeeze the tube"),
    ("clamp_screw_x", "11 mm", "mm", "screw centre from the axle centre along X"),
    ("m3_clear", "3.4 mm", "mm", "M3 clearance hole"),
    ("m3_pilot", "2.8 mm", "mm", "M3 pilot hole for screwing into PETG"),
]


def dim(dims, *args, expr, where):
    try:
        dims.addDistanceDimension(*args).parameter.expression = expr
    except RuntimeError as e:
        print("DIM FAIL", where, expr, str(e)[:80])


def val(expr):
    return adsk.core.ValueInput.createByString(expr)


def ev(design, expr):
    return design.unitsManager.evaluateExpression(expr, "cm")


def add_params(design):
    for name, expr, unit, comment in CLAMP_PARAMS:
        if not design.userParameters.itemByName(name):
            design.userParameters.add(name, val(expr), unit, comment)


def away_from_normal(plane, want_negative_axis):
    """Extent direction along the plane normal that points to -axis (want_negative_axis) or +axis."""
    n = plane.geometry.normal
    comp = n.y if abs(n.y) > 0.5 else n.z
    positive_is_plus = comp > 0
    if want_negative_axis == positive_is_plus:
        return adsk.fusion.ExtentDirections.NegativeExtentDirection
    return adsk.fusion.ExtentDirections.PositiveExtentDirection


def xz_rect(design, comp, name, x_expr, z_lo_expr, z_hi_expr, z_lo_sign, z_hi_sign):
    """Rectangle on the XZ plane: X in +-x_expr, Z from z_lo to z_hi (signs give the side of the axle)."""
    sk = comp.sketches.add(comp.xZConstructionPlane)
    sk.name = name
    hx = ev(design, x_expr)
    z_lo, z_hi = z_lo_sign * ev(design, z_lo_expr), z_hi_sign * ev(design, z_hi_expr)
    p0 = sk.modelToSketchSpace(adsk.core.Point3D.create(-hx, 0, z_lo))
    p1 = sk.modelToSketchSpace(adsk.core.Point3D.create(hx, 0, z_hi))
    lines = sk.sketchCurves.sketchLines.addTwoPointRectangle(p0, p1)
    dims, origin = sk.sketchDimensions, sk.originPoint
    hor = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    ver = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    pts = {}
    for ln in lines:
        for sp in (ln.startSketchPoint, ln.endSketchPoint):
            pts[(round(sp.geometry.x, 4), round(sp.geometry.y, 4))] = sp
    for (want_x, want_y), x_e, y_e in (((p0.x, p0.y), x_expr, z_lo_expr), ((p1.x, p1.y), x_expr, z_hi_expr)):
        sp = pts[(round(want_x, 4), round(want_y, 4))]
        g = sp.geometry
        dim(dims, origin, sp, hor, adsk.core.Point3D.create(g.x, g.y + 1, 0), expr=x_e, where=name + " h")
        dim(dims, origin, sp, ver, adsk.core.Point3D.create(g.x + 1, g.y, 0), expr=y_e, where=name + " v")
    return sk


def extrude(comp, profile, op, dist_expr, direction, participants=None):
    ext = comp.features.extrudeFeatures
    inp = ext.createInput(profile, op)
    inp.setOneSideExtent(adsk.fusion.DistanceExtentDefinition.create(val(dist_expr)), direction)
    if participants:
        inp.participantBodies = participants
    return ext.add(inp)


def run(_context: str):
    app = adsk.core.Application.get()
    if app.activeDocument.name.split(" v")[0] != "TARS_arm_mechanism":
        raise RuntimeError("wrong document: " + app.activeDocument.name)
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    add_params(design)

    for occ in list(root.occurrences):
        if occ.component.name == "axle clamp":
            occ.deleteMe()
    clamp_occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    clamp = clamp_occ.component
    clamp.name = "axle clamp"

    new_body = adsk.fusion.FeatureOperations.NewBodyFeatureOperation
    cut = adsk.fusion.FeatureOperations.CutFeatureOperation
    to_minus_y = away_from_normal(clamp.xZConstructionPlane, want_negative_axis=True)

    pieces = {}
    for part, z_lo, z_hi, s_lo, s_hi in (("bottom", "clamp_bot_h", "clamp_split / 2", -1, -1),
                                         ("top", "clamp_split / 2", "clamp_top_h", 1, 1)):
        occ = clamp.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        comp = occ.component
        comp.name = f"axle clamp {part}"
        sk = xz_rect(design, comp, f"clamp {part} outline", "clamp_w / 2", z_lo, z_hi, s_lo, s_hi)
        body = extrude(comp, sk.profiles.item(0), new_body, "clamp_len / 2", to_minus_y).bodies.item(0)
        body.name = f"clamp {part}"
        # Half-circle seat for the tube: the axle's own diameter, the split gap does the squeezing.
        sk_c = comp.sketches.add(comp.xZConstructionPlane)
        sk_c.name = f"clamp {part} axle seat"
        circ = sk_c.sketchCurves.sketchCircles.addByCenterRadius(sk_c.originPoint.geometry, ev(design, "axle_d") / 2)
        circ.centerSketchPoint.merge(sk_c.originPoint)
        sk_c.sketchDimensions.addDiameterDimension(circ, adsk.core.Point3D.create(1, 1, 0)).parameter.expression = "axle_d"
        prof = min(sk_c.profiles, key=lambda p: p.areaProperties().area)
        extrude(comp, prof, cut, "clamp_len", to_minus_y, [body])
        # Screw holes: two per half, at y = -clamp_len/4, x = +-clamp_screw_x, through the piece.
        sk_h = comp.sketches.add(comp.xYConstructionPlane)
        sk_h.name = f"clamp {part} screw holes"
        hole_d = "m3_clear" if part == "top" else "m3_pilot"
        for sx in (-1, 1):
            c = sk_h.sketchCurves.sketchCircles.addByCenterRadius(
                adsk.core.Point3D.create(sx * ev(design, "clamp_screw_x"), -ev(design, "clamp_len") / 4, 0), ev(design, hole_d) / 2)
            d = sk_h.sketchDimensions
            d.addDiameterDimension(c, adsk.core.Point3D.create(c.centerSketchPoint.geometry.x + 0.5, c.centerSketchPoint.geometry.y + 0.5, 0)).parameter.expression = hole_d
            g = c.centerSketchPoint.geometry
            dim(d, sk_h.originPoint, c.centerSketchPoint, adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation,
                adsk.core.Point3D.create(g.x, g.y - 0.8, 0), expr="clamp_screw_x", where=f"{part} hole {sx} h")
            if sx > 0:
                continue  # Fusion already keeps the second centre level with the first; a second Y dimension over-constrains
            dim(d, sk_h.originPoint, c.centerSketchPoint, adsk.fusion.DimensionOrientations.VerticalDimensionOrientation,
                adsk.core.Point3D.create(g.x + 0.8, g.y, 0), expr="clamp_len / 4", where=f"{part} hole {sx} v")
        holes = adsk.core.ObjectCollection.create()
        for p in sk_h.profiles:
            holes.add(p)
        ext = comp.features.extrudeFeatures
        inp = ext.createInput(holes, cut)
        inp.setSymmetricExtent(val("clamp_bot_h + clamp_top_h"), True)
        inp.participantBodies = [body]
        ext.add(inp)
        pieces[part] = body

    for part, body in pieces.items():
        bb = body.boundingBox
        print(f"clamp {part}: x {bb.minPoint.x*10:.1f}..{bb.maxPoint.x*10:.1f}  y {bb.minPoint.y*10:.1f}..{bb.maxPoint.y*10:.1f}"
              f"  z {bb.minPoint.z*10:.1f}..{bb.maxPoint.z*10:.1f}  volume {body.volume*1000:.0f} mm3")
