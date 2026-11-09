"""Centre axle clamp for TARS_arm_mechanism: one half (body centre plane y = 0 toward the right arm, -Y), to be mirrored.

Split pillow block around the axle at the body centre: a thick bottom piece and a thinner top piece, each with a
half-circle cut for the tube, a gap between them so the screws squeeze the tube, and two M3 screws per half
(clearance through the top, pilot hole into the bottom). Two cheeks hang from the bottom piece on both sides of the
hinge sleeve (hinge_sleeve.py), each with a blind vertical groove that the sleeve's tongue slides in. Everything is
driven by user parameters. Running it again replaces the clamp.
"""

import adsk.core
import adsk.fusion

CLAMP_PARAMS = [
    # Sleeve-side numbers first: clamp_w depends on them. They mirror hinge_sleeve.py's constants.
    ("sleeve_half_w", "15.66 mm", "mm", "hinge sleeve centre to its side wall outer face (hinge_sleeve.py SIDE_OUT)"),
    ("sleeve_bot", "60.61 mm", "mm", "axle centre down to the hinge sleeve bottom; the cheeks end here"),
    ("cheek_t", "5 mm", "mm", "cheek thickness, inner face to outer face"),
    ("rail_w", "15 mm", "mm", "sleeve tongue width along Y (hinge_sleeve.py RAIL_W)"),
    ("rail_depth", "2 mm", "mm", "tongue height out of the sleeve wall (hinge_sleeve.py RAIL_DEPTH)"),
    ("rail_fit", "0.3 mm", "mm", "gap sleeve to cheek and tongue to groove, per side"),
    ("rail_y", "22.4 mm", "mm", "groove centre from the body centre plane (middle of the sleeve side wall)"),
    ("clamp_len", "84 mm", "mm", "full clamp length along Y (this half is clamp_len/2); stops 1 mm before the arm pulley"),
    ("clamp_w", "2 * (sleeve_half_w + rail_fit + cheek_t)", "mm", "clamp size along X: sleeve + fit + both cheeks"),
    ("clamp_bot_h", "9 mm", "mm", "bottom piece: axle centre down to its underside; the sleeve tops out here (47.4 mm centre distance)"),
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
    dimmed_rect(sk, adsk.core.Point3D.create(-hx, 0, z_lo), adsk.core.Point3D.create(hx, 0, z_hi),
                ((x_expr, z_lo_expr), (x_expr, z_hi_expr)), name)
    return sk


def dimmed_rect(sk, a, b, exprs, name):
    """Rectangle between model points a and b; exprs gives each corner's (horizontal, vertical) distance from the origin."""
    p0, p1 = sk.modelToSketchSpace(a), sk.modelToSketchSpace(b)
    lines = sk.sketchCurves.sketchLines.addTwoPointRectangle(p0, p1)
    dims, origin = sk.sketchDimensions, sk.originPoint
    hor = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    ver = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    pts = {}
    for ln in lines:
        for sp in (ln.startSketchPoint, ln.endSketchPoint):
            pts[(round(sp.geometry.x, 4), round(sp.geometry.y, 4))] = sp
    for p, (x_e, y_e) in zip((p0, p1), exprs, strict=True):
        sp = pts[(round(p.x, 4), round(p.y, 4))]
        g = sp.geometry
        dim(dims, origin, sp, hor, adsk.core.Point3D.create(g.x, g.y + 1, 0), expr=x_e, where=name + " h")
        dim(dims, origin, sp, ver, adsk.core.Point3D.create(g.x + 1, g.y, 0), expr=y_e, where=name + " v")


def cheeks(design, comp, body):
    """Cheeks on both sides of the hinge sleeve, from the bottom piece down to the sleeve bottom, with a blind groove each.

    The groove is open at the bottom (the sleeve slides in from below and down to tension the belt) and closed at the
    top by the clamp, and it stops rail_depth + rail_fit into the cheek instead of cutting through it.
    """
    join = adsk.fusion.FeatureOperations.JoinFeatureOperation
    cut = adsk.fusion.FeatureOperations.CutFeatureOperation
    to_minus_y = away_from_normal(comp.xZConstructionPlane, want_negative_axis=True)
    outer = xz_rect(design, comp, "clamp cheeks outer", "clamp_w / 2", "sleeve_bot", "clamp_bot_h", -1, -1)
    extrude(comp, outer.profiles.item(0), join, "clamp_len / 2", to_minus_y, [body])
    inner = xz_rect(design, comp, "clamp cheeks inner", "sleeve_half_w + rail_fit", "sleeve_bot + 1 mm", "clamp_bot_h", -1, -1)
    extrude(comp, inner.profiles.item(0), cut, "clamp_len", to_minus_y, [body])

    plane_in = comp.constructionPlanes.createInput()
    plane_in.setByOffset(comp.xYConstructionPlane, val("-sleeve_bot"))
    plane = comp.constructionPlanes.add(plane_in)
    plane.name = "sleeve bottom"
    sk = comp.sketches.add(plane)
    sk.name = "clamp grooves"
    # The groove starts 1 mm inside the cheek face so the cut never leaves a zero-thickness skin.
    x_in, x_out = "sleeve_half_w + rail_fit - 1 mm", "sleeve_half_w + 2 * rail_fit + rail_depth"
    y_near, y_far = "rail_y - rail_w / 2 - rail_fit", "rail_y + rail_w / 2 + rail_fit"
    z = -ev(design, "sleeve_bot")
    for sx in (-1, 1):
        a = adsk.core.Point3D.create(sx * ev(design, x_in), -ev(design, y_near), z)
        b = adsk.core.Point3D.create(sx * ev(design, x_out), -ev(design, y_far), z)
        dimmed_rect(sk, a, b, ((x_in, y_near), (x_out, y_far)), f"groove {sx}")
    grooves = adsk.core.ObjectCollection.create()
    for p in sk.profiles:
        grooves.add(p)
    extrude(comp, grooves, cut, "sleeve_bot - clamp_bot_h", away_from_normal(plane, want_negative_axis=False), [body])


def screw_circles(design, comp, plane, d_expr, name):
    """Sketch the two screw circles (x = +-clamp_screw_x, y = -clamp_len/4) of diameter d_expr on `plane`."""
    sk = comp.sketches.add(plane)
    sk.name = name
    d = sk.sketchDimensions
    for sx in (-1, 1):
        centre = sk.modelToSketchSpace(adsk.core.Point3D.create(
            sx * ev(design, "clamp_screw_x"), -ev(design, "clamp_len") / 4, plane.geometry.origin.z))
        c = sk.sketchCurves.sketchCircles.addByCenterRadius(centre, ev(design, d_expr) / 2)
        g = c.centerSketchPoint.geometry
        d.addDiameterDimension(c, adsk.core.Point3D.create(g.x + 0.5, g.y + 0.5, 0)).parameter.expression = d_expr
        dim(d, sk.originPoint, c.centerSketchPoint, adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation,
            adsk.core.Point3D.create(g.x, g.y - 0.8, 0), expr="clamp_screw_x", where=f"{name} {sx} h")
        if sx > 0:
            continue  # Fusion already keeps the second centre level with the first; a second Y dimension over-constrains
        dim(d, sk.originPoint, c.centerSketchPoint, adsk.fusion.DimensionOrientations.VerticalDimensionOrientation,
            adsk.core.Point3D.create(g.x + 0.8, g.y, 0), expr="clamp_len / 4", where=f"{name} {sx} v")
    return sk


def extrude_symmetric(comp, sk, op, dist_expr, body):
    profiles = adsk.core.ObjectCollection.create()
    for p in sk.profiles:
        profiles.add(p)
    ext = comp.features.extrudeFeatures
    inp = ext.createInput(profiles, op)
    inp.setSymmetricExtent(val(dist_expr), True)
    inp.participantBodies = [body]
    ext.add(inp)


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
        if part == "bottom":
            cheeks(design, comp, body)
        # Screw holes: two per half, at y = -clamp_len/4, x = +-clamp_screw_x, in from the top: clearance through the
        # top piece, pilot hole into the bottom piece. The cut runs twice the clamp height each way so it always leaves
        # both outer faces whatever the piece heights are.
        hole_d = "m3_clear" if part == "top" else "m3_pilot"
        sk_h = screw_circles(design, comp, comp.xYConstructionPlane, hole_d, f"clamp {part} screw holes")
        extrude_symmetric(comp, sk_h, cut, "2 * (clamp_bot_h + clamp_top_h)", body)
        pieces[part] = body

    for part, body in pieces.items():
        bb = body.boundingBox
        print(f"clamp {part}: x {bb.minPoint.x*10:.1f}..{bb.maxPoint.x*10:.1f}  y {bb.minPoint.y*10:.1f}..{bb.maxPoint.y*10:.1f}"
              f"  z {bb.minPoint.z*10:.1f}..{bb.maxPoint.z*10:.1f}  volume {body.volume*1000:.0f} mm3")
