"""Hinge servo sleeve + split bearing cap in TARS_arm_mechanism (component `servo sleeve` and `bearing cap`).

All coordinates are in the hinge servo's own frame (ST3215 model): output shaft along +Y (driving horn at +Y),
output axis at X = -25.5, Z = 0, case X -35.61..9.61 (output near the -X end), Z +-12.36. The numbers below were
measured from the servo STEP and from the pulley/bearing as modelled (2026-11-08), so the parts follow the servo
wherever it is placed. Not parametric: change a constant and run again to rebuild both parts.

Sleeve (one print): top plate over the case's +X end, back plate with 4 x M2 standoffs into the case's back holes,
a cable window at the connector and a cut-out round the driven horn, two side walls, a solid block above the pulley
(2 x M2 into the case's front holes), and the upper half of the bearing housing. A tongue on each side wall slides in
the groove of the axle clamp's cheeks (axle_clamp.py), and the top is cut round the arm pulley on the axle. Bearing cap
(second print): lower half of the housing, 2 x M3 up into the sleeve. Assembly: servo + pulley + bearing slide in from
-X, then the cap.
"""

import adsk.core
import adsk.fusion

# --- measured geometry, servo frame, mm ---------------------------------------------------------------------
AXIS_X = -25.5
CASE_TOP_X, CASE_BOTTOM_X, CASE_HALF_Z = 9.61, -35.61, 12.36
CASE_FRONT_Y = 7.5                   # highest point of the front face (label area)
FRONT_PAD_Y = 6.4                    # front face at the M2 holes
BACK_PAD_Y = -25.6                   # back face at the M2 holes
BACK_MOST_Y = -28.2                  # driven-horn boss; connector -27.5, driven horn -27.65
FRONT_HOLES = [(3.5, 10.25), (3.5, -10.25)]                                   # (x, z); the -17.2 pair is behind the pulley
BACK_HOLES = [(7.25, 10.25), (7.25, -10.25), (-17.2, 10.25), (-17.2, -10.25)]
CONNECTOR_X = (-7.0, 4.5)            # window round the connector shroud (-5.95..3.7) with margin
CONNECTOR_HALF_Z = 9.8
PULLEY_TOP_X = AXIS_X + 19.85        # pulley flange radius
PULLEY_OUTER_Y = 22.1                # outer guide face
BEARING_Y = (23.1, 28.1)             # 6801 as placed on the stub
BEARING_R = 10.5

# --- design choices, mm -------------------------------------------------------------------------------------
T = 3.0                              # wall
GAP = 0.3                            # sleeve to case
BACK_INNER_Y = BACK_MOST_Y - 1.0     # 1 mm clear of horn, boss and connector
BLOCK_BOTTOM_X = PULLEY_TOP_X + 1.15 # solid block stays 1.15 mm above the pulley flange
LIP_T, LIP_R = 2.0, 8.0              # outer lip holds the bearing's outer ring, clears the turning inner ring
HOUSING_HALF_Z = 17.5
CAP_SPLIT = 0.2                      # gap at the split so the screws clamp the outer ring
CAP_SCREW_Z, CAP_BELOW = 14.0, 4.0   # M3 positions, material under the bearing
M2_CLEAR, M2_HEAD = 2.4, 4.2
M3_CLEAR, M3_PILOT, M3_HEAD = 3.4, 2.8, 6.2
CENTRE_DIST = 50.5                   # servo output to hinge axle, along +X, as the hinge drive sits now (2026-11-08)
ARM_PULLEY_CLEAR_R = 19.85 + 1.65    # arm pulley flange radius (same 60T as the drive) + clearance
RAIL_W, RAIL_DEPTH = 15.0, 2.0       # tongue on each side wall; axle_clamp.py rail_w / rail_depth must match

TOP_IN, TOP_OUT = CASE_TOP_X + GAP, CASE_TOP_X + GAP + T
SIDE_IN, SIDE_OUT = CASE_HALF_Z + GAP, CASE_HALF_Z + GAP + T
BACK_OUT = BACK_INNER_Y - T
FRONT_END_Y = BEARING_Y[1] + LIP_T
SIDE_FRONT_Y = 7.0                   # side walls stop short of the pulley (pulley back face 8.1)
RAIL_MID_Y = (BACK_OUT + SIDE_FRONT_Y) / 2   # tongue centred on the side wall; axle_clamp.py rail_y follows it


def mm(v):
    return v / 10.0


def pt(x, y, z):
    return adsk.core.Point3D.create(mm(x), mm(y), mm(z))


TB = adsk.fusion.TemporaryBRepManager.get()


def box(x0, x1, y0, y1, z0, z1):
    c = pt((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    obb = adsk.core.OrientedBoundingBox3D.create(c, adsk.core.Vector3D.create(1, 0, 0), adsk.core.Vector3D.create(0, 1, 0),
                                                 mm(x1 - x0), mm(y1 - y0), mm(z1 - z0))
    return TB.createBox(obb)


def cyl_y(x, z, y0, y1, d):
    return TB.createCylinderOrCone(pt(x, y0, z), mm(d / 2), pt(x, y1, z), mm(d / 2))


def cyl_x(y, z, x0, x1, d):
    return TB.createCylinderOrCone(pt(x0, y, z), mm(d / 2), pt(x1, y, z), mm(d / 2))


def union(a, *bs):
    for b in bs:
        TB.booleanOperation(a, b, adsk.fusion.BooleanTypes.UnionBooleanType)
    return a


def cut(a, *bs):
    for b in bs:
        TB.booleanOperation(a, b, adsk.fusion.BooleanTypes.DifferenceBooleanType)
    return a


def sleeve_body():
    housing_y0 = BEARING_Y[0]
    s = box(TOP_IN, TOP_OUT, BACK_OUT, FRONT_END_Y, -SIDE_OUT, SIDE_OUT)                          # top plate
    union(s,
          box(CASE_BOTTOM_X, TOP_OUT, BACK_OUT, BACK_INNER_Y, -SIDE_OUT, SIDE_OUT),              # back plate
          box(CASE_BOTTOM_X, TOP_OUT, BACK_OUT, SIDE_FRONT_Y, SIDE_IN, SIDE_OUT),                # side walls
          box(CASE_BOTTOM_X, TOP_OUT, BACK_OUT, SIDE_FRONT_Y, -SIDE_OUT, -SIDE_IN),
          box(BLOCK_BOTTOM_X, TOP_OUT, SIDE_FRONT_Y, FRONT_END_Y, -SIDE_OUT, SIDE_OUT),          # block above the pulley
          box(AXIS_X, BLOCK_BOTTOM_X + 0.01, housing_y0, FRONT_END_Y, -HOUSING_HALF_Z, HOUSING_HALF_Z))  # upper housing
    # The block must not overlap the case front, which reaches CASE_FRONT_Y between the side walls.
    cut(s, box(BLOCK_BOTTOM_X - 1, TOP_IN, SIDE_FRONT_Y - 1, CASE_FRONT_Y + 0.05, -SIDE_IN, SIDE_IN))
    for sz in (1, -1):                                                                          # tongues for the clamp rails
        z0, z1 = sorted((sz * (SIDE_OUT - 0.01), sz * (SIDE_OUT + RAIL_DEPTH)))
        union(s, box(CASE_BOTTOM_X, TOP_OUT, RAIL_MID_Y - RAIL_W / 2, RAIL_MID_Y + RAIL_W / 2, z0, z1))
    for x, z in BACK_HOLES:                                                                     # standoffs to the back pads
        union(s, cyl_y(x, z, BACK_INNER_Y - 0.01, BACK_PAD_Y, 5.0))
    for x, z in FRONT_HOLES:                                                                    # bosses down to the front pads
        union(s, cyl_y(x, z, FRONT_PAD_Y, CASE_FRONT_Y + 0.1, 5.0))
    cut(s,
        cyl_y(AXIS_X, 0, BACK_OUT - 1, BACK_INNER_Y + 1, 22.0),                                   # driven horn clearance
        box(CONNECTOR_X[0], CONNECTOR_X[1], BACK_OUT - 1, BACK_INNER_Y + 1, -CONNECTOR_HALF_Z, CONNECTOR_HALF_Z),  # cables
        cyl_y(AXIS_X, 0, BEARING_Y[0] - 0.01, BEARING_Y[1], 2 * BEARING_R),                    # bearing seat
        cyl_y(AXIS_X, 0, BEARING_Y[1] - 0.01, FRONT_END_Y + 1, 2 * LIP_R))                     # lip opening
    for x, z in BACK_HOLES:
        cut(s, cyl_y(x, z, BACK_OUT - 1, BACK_PAD_Y + 0.01, M2_CLEAR), cyl_y(x, z, BACK_OUT - 1, BACK_OUT + 1.2, M2_HEAD))
    for x, z in FRONT_HOLES:   # long access bore from the front end down to the boss, then the M2 clearance
        cut(s, cyl_y(x, z, FRONT_PAD_Y - 0.01, FRONT_END_Y + 1, M2_CLEAR), cyl_y(x, z, CASE_FRONT_Y + 3.0, FRONT_END_Y + 1, M2_HEAD))
    mid_y = (BEARING_Y[0] + FRONT_END_Y) / 2
    for z in (CAP_SCREW_Z, -CAP_SCREW_Z):
        cut(s, cyl_x(mid_y, z, AXIS_X - 0.01, AXIS_X + 12.0, M3_PILOT))
    # The arm pulley turns round the axle in the same belt plane as the drive pulley.
    cut(s, cyl_y(AXIS_X + CENTRE_DIST, 0, SIDE_FRONT_Y, FRONT_END_Y + 1, 2 * ARM_PULLEY_CLEAR_R))
    return s


def cap_body():
    top = AXIS_X - CAP_SPLIT
    bottom = AXIS_X - BEARING_R - CAP_BELOW
    c = box(bottom, top, BEARING_Y[0], FRONT_END_Y, -HOUSING_HALF_Z, HOUSING_HALF_Z)
    cut(c,
        cyl_y(AXIS_X, 0, BEARING_Y[0] - 0.01, BEARING_Y[1], 2 * BEARING_R),
        cyl_y(AXIS_X, 0, BEARING_Y[1] - 0.01, FRONT_END_Y + 1, 2 * LIP_R))
    mid_y = (BEARING_Y[0] + FRONT_END_Y) / 2
    for z in (CAP_SCREW_Z, -CAP_SCREW_Z):
        cut(c, cyl_x(mid_y, z, bottom - 1, top + 1, M3_CLEAR), cyl_x(mid_y, z, bottom - 1, bottom + 3.0, M3_HEAD))
    return c


def place(comp, occ, servo_occ, body, name):
    """Move a body built in the servo frame into `comp`'s own space and add it as a base feature."""
    to_local = servo_occ.transform2.copy()   # servo frame -> world ...
    inv = occ.transform2.copy()
    inv.invert()
    to_local.transformBy(inv)                 # ... -> the component's own space
    TB.transform(body, to_local)
    bf = comp.features.baseFeatures.add()
    bf.startEdit()
    comp.bRepBodies.add(body, bf)
    bf.finishEdit()
    bf.name = name
    b = comp.bRepBodies.item(comp.bRepBodies.count - 1)
    b.name = name
    return b


def run(_context: str):
    app = adsk.core.Application.get()
    if app.activeDocument.name.split(" v")[0] != "TARS_arm_mechanism":
        raise RuntimeError("wrong document: " + app.activeDocument.name)
    design = adsk.fusion.Design.cast(app.activeProduct)
    occs = {o.fullPathName: o for o in design.rootComponent.allOccurrences}
    servo = occs["hinge drive:1+ST3215:1"]
    hinge = occs["hinge drive:1"]
    sleeve_occ = occs.get("hinge drive:1+servo sleeve:1")
    if sleeve_occ is None:
        sleeve_occ = hinge.component.occurrences.addNewComponent(adsk.core.Matrix3D.create()).createForAssemblyContext(hinge)
        sleeve_occ.component.name = "servo sleeve"
    sleeve = sleeve_occ.component
    if any(b.name != "hinge sleeve" for b in sleeve.bRepBodies):
        raise RuntimeError("servo sleeve already has bodies of its own; not touching them")
    for f in list(sleeve.features.baseFeatures):
        f.deleteMe()
    cap_occ = occs.get("hinge drive:1+bearing cap:1")
    if cap_occ is not None:
        cap_occ.deleteMe()
    cap_occ = hinge.component.occurrences.addNewComponent(adsk.core.Matrix3D.create()).createForAssemblyContext(hinge)
    cap_occ.component.name = "bearing cap"

    s = place(sleeve, sleeve_occ, servo, sleeve_body(), "hinge sleeve")
    c = place(cap_occ.component, cap_occ, servo, cap_body(), "bearing cap")

    # Lock both to the servo so they move with it (and it with them) when the drive gets placed.
    root = design.rootComponent
    for occ in (sleeve_occ, cap_occ):
        for j in list(root.allAsBuiltJoints):
            if {j.occurrenceOne.fullPathName, j.occurrenceTwo.fullPathName} == {occ.fullPathName, servo.fullPathName}:
                j.deleteMe()
        inp = root.asBuiltJoints.createInput(occ, servo, None)
        inp.setAsRigidJointMotion()
        root.asBuiltJoints.add(inp)

    for name, b in (("sleeve", s), ("cap", c)):
        mass = b.volume * 1.27
        print(f"{name}: volume {b.volume * 1000:.0f} mm3, PETG {mass:.1f} g at 100 % infill")

    # Interference against everything it must clear.
    ents = adsk.core.ObjectCollection.create()
    for path in ("hinge drive:1+servo sleeve:1", "hinge drive:1+bearing cap:1", "hinge drive:1+ST3215:1", "hinge drive:1+drive pulley:1"):
        o = {x.fullPathName: x for x in root.allOccurrences}.get(path)
        if o:
            ents.add(o)
    for o in root.allOccurrences:
        if "5972K276" in o.fullPathName:
            ents.add(o)
    res = design.analyzeInterference(design.createInterferenceInput(ents))
    if res.count == 0:
        print("interference: none")
    for i in range(res.count):
        r = res.item(i)
        print("interference:", r.entityOne.assemblyContext.name if r.entityOne.assemblyContext else r.entityOne.name,
              "x", r.entityTwo.assemblyContext.name if r.entityTwo.assemblyContext else r.entityTwo.name,
              f"{r.interferenceBody.volume * 1000:.2f} mm3")
