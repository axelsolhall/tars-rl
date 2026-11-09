"""Belt tension for the hinge drive in TARS_arm_mechanism: two M3 jack screws through the axle clamp, M3 nuts sunk
into the clamp bottom's underside, and a replaceable steel plate in a pocket on the servo sleeve's top.

Turning the screws down pushes the plate, and with it the whole hinge drive, down the clamp rails, which tightens the
belt. The screws pass with clearance through both clamp pieces, so they never pull the clamp halves together; the clamp
screws at y = -clamp_len/4 do that.

Also carries later sleeve edits: the front M2 screws are dropped and filled, the corners above the drive pulley are
filled, and the arm pulley clearance follows the sleeve's highest position.

Adds features to the existing clamp and sleeve instead of rebuilding them, so Axel's ground, slider and axle joints
keep their references. Running it again removes only what it added before and adds it again. Rerunning axle_clamp.py
or hinge_sleeve.py deletes those parts (and the joints on them); run this one after them.
"""

import adsk.core
import adsk.fusion

PARAMS = [
    ("jack_x", "clamp_screw_x", "mm", "tension screw centre from the axle centre along X (same as the clamp screws)"),
    ("jack_y", "36 mm", "mm", "tension screw centre from the body centre plane, near the belt plane so the rails carry less moment"),
    ("m3_nut_af", "5.5 mm", "mm", "M3 hex nut across flats (DIN 934)"),
    ("m3_nut_h", "2.4 mm", "mm", "M3 hex nut height (DIN 934)"),
    ("nut_fit", "0.3 mm", "mm", "nut pocket play: across flats, and extra depth"),
    ("plate_w", "28 mm", "mm", "tension plate size along X (covers both screw tips)"),
    ("plate_d", "10 mm", "mm", "tension plate size along Y"),
    ("plate_t", "1 mm", "mm", "tension plate thickness (steel); its pocket is this deep so the plate sits flush"),
    ("plate_fit", "0.2 mm", "mm", "tension plate pocket play per side"),
]

# Sleeve geometry in the hinge servo's frame (hinge_sleeve.py): sleeve top face, and where the tongue centre sits.
SLEEVE_TOP_X = 9.61 + 0.3 + 3.0        # CASE_TOP_X + GAP + T
AXIS_X = -25.5                         # servo output axis
SIDE_FRONT_Y, FRONT_END_Y = 7.0, 30.1  # where the pulley plane starts, sleeve front end
ARM_PULLEY_CLEAR_R = 19.85 + 1.65      # arm pulley flange radius + clearance (hinge_sleeve.py)
# Front M2 screws dropped (Axel 2026-11-08): the servo is held by the four back screws only, and the front bores and
# the corners above the drive pulley are filled solid (hinge_sleeve.py still builds them; this fills them).
FRONT_HOLES = [(3.5, 10.25), (3.5, -10.25)]
FRONT_PAD_Y = 6.4
BLOCK_BOTTOM_X = AXIS_X + 19.85 + 1.15  # bottom of the block above the drive pulley
BEARING_Y0 = 23.1                      # upper bearing housing starts here
SIDE_OUT = 12.36 + 0.3 + 3.0           # side wall outer face
DRIVE_PULLEY_CLEAR_R = 19.85 + 1.0     # fill stays 1 mm off the drive pulley flange
FILL_START_Y = 7.5 + 0.3               # case front (label area) + gap: the fill must not touch the servo
RAIL_MID_Y = (-29.2 - 3.0 + 7.0) / 2   # middle of the side wall = tongue centre, which the slider puts at -rail_y

NAMES = ("jack screw holes", "jack screw holes sketch", "jack nut pockets", "jack nut pockets sketch", "jack nut plane",
         "tension plate pocket", "tension plate pocket tool", "arm pulley clearance",
         "arm pulley clearance tool", "sleeve fill", "sleeve fill tool")
TB = adsk.fusion.TemporaryBRepManager.get()


def val(expr):
    return adsk.core.ValueInput.createByString(expr)


def pt(x, y, z):
    return adsk.core.Point3D.create(x, y, z)


def remove_previous(comp):
    for coll in (comp.features.extrudeFeatures, comp.features.combineFeatures, comp.features.baseFeatures,
                 comp.sketches, comp.constructionPlanes):
        for item in [i for i in coll if i.name.split(" (")[0] in NAMES]:  # Fusion appends " (1)" on name clashes
            item.deleteMe()


def screw_holes(design, comp, body):
    ev = lambda e: design.unitsManager.evaluateExpression(e, "cm")  # noqa: E731
    sk = comp.sketches.add(comp.xYConstructionPlane)
    sk.name = "jack screw holes sketch"  # sketch and feature names must differ, or Fusion renames one
    d = sk.sketchDimensions
    hor = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    ver = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    for sx in (-1, 1):
        c = sk.sketchCurves.sketchCircles.addByCenterRadius(
            sk.modelToSketchSpace(pt(sx * ev("jack_x"), -ev("jack_y"), 0)), ev("m3_clear") / 2)
        g = c.centerSketchPoint.geometry
        d.addDiameterDimension(c, pt(g.x + 0.5, g.y + 0.5, 0)).parameter.expression = "m3_clear"
        d.addDistanceDimension(sk.originPoint, c.centerSketchPoint, hor, pt(g.x, g.y - 0.8, 0)).parameter.expression = "jack_x"
        if sx < 0:  # Fusion keeps the second centre level with the first; a second Y dimension over-constrains
            d.addDistanceDimension(sk.originPoint, c.centerSketchPoint, ver, pt(g.x + 0.8, g.y, 0)).parameter.expression = "jack_y"
    profiles = adsk.core.ObjectCollection.create()
    for p in sk.profiles:
        profiles.add(p)
    ext = comp.features.extrudeFeatures
    inp = ext.createInput(profiles, adsk.fusion.FeatureOperations.CutFeatureOperation)
    inp.setSymmetricExtent(val("2 * (clamp_bot_h + clamp_top_h)"), True)
    inp.participantBodies = [body]
    ext.add(inp).name = "jack screw holes"


def nut_pockets(design, comp, body):
    """Hex pockets up into the bottom piece's underside. The hex size is read from the parameters when this runs
    (sketch polygons are not dimensioned), so run the script again after changing m3_nut_af or nut_fit."""
    ev = lambda e: design.unitsManager.evaluateExpression(e, "cm")  # noqa: E731
    plane_in = comp.constructionPlanes.createInput()
    plane_in.setByOffset(comp.xYConstructionPlane, val("-clamp_bot_h"))
    plane = comp.constructionPlanes.add(plane_in)
    plane.name = "jack nut plane"
    plane.isLightBulbOn = False
    sk = comp.sketches.add(plane)
    sk.name = "jack nut pockets sketch"
    for sx in (-1, 1):
        centre = sk.modelToSketchSpace(pt(sx * ev("jack_x"), -ev("jack_y"), -ev("clamp_bot_h")))
        sk.sketchCurves.sketchLines.addScribedPolygon(centre, 6, 0, (ev("m3_nut_af") + ev("nut_fit")) / 2, False)
    profiles = adsk.core.ObjectCollection.create()
    for p in sk.profiles:
        profiles.add(p)
    up = adsk.fusion.ExtentDirections.PositiveExtentDirection if plane.geometry.normal.z > 0 else adsk.fusion.ExtentDirections.NegativeExtentDirection
    ext = comp.features.extrudeFeatures
    inp = ext.createInput(profiles, adsk.fusion.FeatureOperations.CutFeatureOperation)
    inp.setOneSideExtent(adsk.fusion.DistanceExtentDefinition.create(val("m3_nut_h + nut_fit")), up)
    inp.participantBodies = [body]
    ext.add(inp).name = "jack nut pockets"


def servo_box(x0, x1, y0, y1, z0, z1):
    """Box in the servo frame, mm in, cm out."""
    c = pt((x0 + x1) / 20, (y0 + y1) / 20, (z0 + z1) / 20)
    obb = adsk.core.OrientedBoundingBox3D.create(c, adsk.core.Vector3D.create(1, 0, 0), adsk.core.Vector3D.create(0, 1, 0),
                                                 (x1 - x0) / 10, (y1 - y0) / 10, (z1 - z0) / 10)
    return TB.createBox(obb)


def to_local(servo_occ, occ):
    m = servo_occ.transform2.copy()
    inv = occ.transform2.copy()
    inv.invert()
    m.transformBy(inv)
    return m


def add_base_body(comp, body, name):
    bf = comp.features.baseFeatures.add()
    bf.startEdit()
    comp.bRepBodies.add(body, bf)
    bf.finishEdit()
    bf.name = name
    return comp.bRepBodies.item(comp.bRepBodies.count - 1)


def run(_context: str):
    app = adsk.core.Application.get()
    if app.activeDocument.name.split(" v")[0] != "TARS_arm_mechanism":
        raise RuntimeError("wrong document: " + app.activeDocument.name)
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    ev = lambda e: design.unitsManager.evaluateExpression(e, "mm") * 10  # noqa: E731  (Fusion returns cm)
    for name, expr, unit, comment in PARAMS:
        if not design.userParameters.itemByName(name):
            design.userParameters.add(name, val(expr), unit, comment)

    occs = {o.fullPathName: o for o in root.allOccurrences}
    servo, sleeve_occ, hinge = occs["hinge drive:1+ST3215:1"], occs["hinge drive:1+servo sleeve:1"], occs["hinge drive:1"]
    bottom = design.allComponents.itemByName("axle clamp bottom")
    top = design.allComponents.itemByName("axle clamp top")
    sleeve = sleeve_occ.component
    # Axel mirrors the clamp halves after this script ran (2026-11-08). Removing and re-adding the clamp features would
    # then cut only the right half, so refuse instead of silently dropping the left-side holes.
    if bottom.bRepBodies.item(0).boundingBox.maxPoint.y > 0.1:
        raise RuntimeError("axle clamp is already mirrored; rerunning would drop the left-side jack holes. "
                           "Edit the features in place instead.")

    plate_old = occs.get("hinge drive:1+tension plate:1")
    if plate_old is not None:
        plate_old.deleteMe()
    for comp in (bottom, top, sleeve):
        remove_previous(comp)

    # Clamp: clearance through both pieces, nut pockets in the bottom piece's underside.
    screw_holes(design, top, top.bRepBodies.item(0))
    screw_holes(design, bottom, bottom.bRepBodies.item(0))
    nut_pockets(design, bottom, bottom.bRepBodies.item(0))

    # Sleeve: pocket for the plate, under the screw tips. World -Y is the servo's +Y here (the slider joint lines the
    # tongue centre up with -rail_y), and world X is the servo's Z, so the screws at +-jack_x land at servo z +-jack_x.
    plate_y = RAIL_MID_Y + (ev("jack_y") - ev("rail_y"))
    w, dpt, t, fit = ev("plate_w"), ev("plate_d"), ev("plate_t"), ev("plate_fit")
    tool = servo_box(SLEEVE_TOP_X - t, SLEEVE_TOP_X + 1, plate_y - dpt / 2 - fit, plate_y + dpt / 2 + fit, -w / 2 - fit, w / 2 + fit)
    TB.transform(tool, to_local(servo, sleeve_occ))
    sleeve_body = next(b for b in sleeve.bRepBodies if b.name == "hinge sleeve")
    vol_before = sleeve_body.volume
    tool_body = add_base_body(sleeve, tool, "tension plate pocket tool")
    tools = adsk.core.ObjectCollection.create()
    tools.add(tool_body)
    cin = sleeve.features.combineFeatures.createInput(sleeve_body, tools)
    cin.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
    cin.isKeepToolBodies = False
    sleeve.features.combineFeatures.add(cin).name = "tension plate pocket"

    # Fill: the front screw bores and bosses, and the corners above the drive pulley (inside the belt runs, |z| <= the
    # side walls). Joined before the arm pulley clearance so that cut still wins.
    fill = TB.createBox(adsk.core.OrientedBoundingBox3D.create(
        pt((AXIS_X + BLOCK_BOTTOM_X) / 20, (FILL_START_Y + BEARING_Y0) / 20, 0), adsk.core.Vector3D.create(1, 0, 0),
        adsk.core.Vector3D.create(0, 1, 0), (BLOCK_BOTTOM_X + 0.01 - AXIS_X) / 10, (BEARING_Y0 + 0.01 - FILL_START_Y) / 10,
        2 * SIDE_OUT / 10))
    TB.booleanOperation(fill, TB.createCylinderOrCone(pt(AXIS_X / 10, (SIDE_FRONT_Y - 1) / 10, 0), DRIVE_PULLEY_CLEAR_R / 10,
                                                      pt(AXIS_X / 10, (BEARING_Y0 + 1) / 10, 0), DRIVE_PULLEY_CLEAR_R / 10),
                        adsk.fusion.BooleanTypes.DifferenceBooleanType)
    for x, z in FRONT_HOLES:
        TB.booleanOperation(fill, TB.createCylinderOrCone(pt(x / 10, (FRONT_PAD_Y + 0.01) / 10, z / 10), 0.215,
                                                          pt(x / 10, FRONT_END_Y / 10, z / 10), 0.215),
                            adsk.fusion.BooleanTypes.UnionBooleanType)
    TB.transform(fill, to_local(servo, sleeve_occ))
    tools = adsk.core.ObjectCollection.create()
    tools.add(add_base_body(sleeve, fill, "sleeve fill tool"))
    cin = sleeve.features.combineFeatures.createInput(next(b for b in sleeve.bRepBodies if b.name == "hinge sleeve"), tools)
    cin.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
    cin.isKeepToolBodies = False
    sleeve.features.combineFeatures.add(cin).name = "sleeve fill"

    # Arm pulley clearance at the sleeve's highest position (top against the clamp underside, plate flush), the
    # smallest centre distance; lower positions only move the pulley further from the sleeve. hinge_sleeve.py's own
    # cut assumed 50.5 mm.
    axle_x = SLEEVE_TOP_X + ev("clamp_bot_h")
    tool = TB.createCylinderOrCone(pt(axle_x / 10, SIDE_FRONT_Y / 10, 0), ARM_PULLEY_CLEAR_R / 10,
                                   pt(axle_x / 10, (FRONT_END_Y + 1) / 10, 0), ARM_PULLEY_CLEAR_R / 10)
    TB.transform(tool, to_local(servo, sleeve_occ))
    tools = adsk.core.ObjectCollection.create()
    tools.add(add_base_body(sleeve, tool, "arm pulley clearance tool"))
    cin = sleeve.features.combineFeatures.createInput(next(b for b in sleeve.bRepBodies if b.name == "hinge sleeve"), tools)
    cin.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
    cin.isKeepToolBodies = False
    sleeve.features.combineFeatures.add(cin).name = "arm pulley clearance"
    print(f"arm pulley clearance cut for a {axle_x - AXIS_X:.2f} mm centre distance (sleeve at the top of the rails)")

    # The plate: its own component in hinge drive, locked to the sleeve.
    plate_occ = hinge.component.occurrences.addNewComponent(adsk.core.Matrix3D.create()).createForAssemblyContext(hinge)
    plate_occ.component.name = "tension plate"
    plate = servo_box(SLEEVE_TOP_X - t, SLEEVE_TOP_X, plate_y - dpt / 2, plate_y + dpt / 2, -w / 2, w / 2)
    TB.transform(plate, to_local(servo, plate_occ))
    plate_body = add_base_body(plate_occ.component, plate, "tension plate")
    plate_body.name = "tension plate"
    steel = None
    for lib in app.materialLibraries:
        steel = steel or lib.materials.itemByName("Steel")
    if steel:
        plate_body.material = steel
    inp = root.asBuiltJoints.createInput(plate_occ, sleeve_occ, None)
    inp.setAsRigidJointMotion()
    root.asBuiltJoints.add(inp)

    # Report: what was removed, and where the plate sits under the screws (world, mm).
    bb = plate_body.createForAssemblyContext(plate_occ).boundingBox if plate_body.assemblyContext is None else plate_body.boundingBox
    sleeve_body = next(b for b in sleeve.bRepBodies if b.name == "hinge sleeve")
    def health(comp, name):
        f = next((f for f in comp.features.extrudeFeatures if f.name == name), None)
        return "missing" if f is None else ("ok" if f.healthState == 0 else f.timelineObject.errorOrWarningMessage[:80])
    print(f"sleeve pocket removed {(vol_before - sleeve_body.volume) * 1000:.0f} mm3; top holes {health(top, 'jack screw holes')}, "
          f"bottom holes {health(bottom, 'jack screw holes')}, nut pockets {health(bottom, 'jack nut pockets')}")
    print(f"plate (world): x {bb.minPoint.x*10:.2f}..{bb.maxPoint.x*10:.2f} y {bb.minPoint.y*10:.2f}..{bb.maxPoint.y*10:.2f} "
          f"z {bb.minPoint.z*10:.2f}..{bb.maxPoint.z*10:.2f}; steel={'yes' if steel else 'not found'}; "
          f"screws at x +-{ev('jack_x'):.1f}, y -{ev('jack_y'):.1f}")
