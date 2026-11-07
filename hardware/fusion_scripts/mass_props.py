"""Mass, centre of mass and inertia per simulator body, from the Fusion geometry.

Bodies: `central` (slab shell + everything mounted in it), and per arm `carrier` (carrier box +
slide servo; turns with the hinge) and `sleeve` (arm shell; turns and slides). Printed bodies get
PETG's density (petg_density); a bought part's block is rescaled to its stored real mass, so only
its shape is assumed uniform. Output: one JSON line per body, frame axes = world axes, inertia
about the body's own centre of mass in kg*m^2, positions in m relative to the body's hinge point.
"""
import json

import adsk.core
import adsk.fusion

HIGH = adsk.fusion.CalculationAccuracy.HighCalculationAccuracy


def body_props(body, mass_kg):
    """World-frame (m, c[3] in cm, J_origin 3x3 in kg*cm^2 as the inertia tensor) for one body proxy."""
    pp = body.getPhysicalProperties(HIGH)
    scale = mass_kg / pp.mass
    c = pp.centerOfMass
    _, xx, yy, zz, xy, yz, xz = pp.getXYZMomentsOfInertia()
    J = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    return mass_kg, [c.x, c.y, c.z], [[v * scale for v in row] for row in J], (xy, xz)


def run(_context: str):
    design = adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct)
    units = design.unitsManager
    rho = units.evaluateExpression("petg_density", "") / 1000.0  # kg per cm^3
    root = design.rootComponent

    groups = {}  # body name -> list of (mass, c, J)
    sign_check = None
    for occ in [None] + list(root.allOccurrences):
        comp = root if occ is None else occ.component
        for body in comp.bRepBodies:
            b = body if occ is None else body.createForAssemblyContext(occ)
            path = "" if occ is None else occ.fullPathName
            if body.attributes.itemByName("tars", "printed"):
                mass = body.getPhysicalProperties(HIGH).volume * rho
            else:
                attr = comp.attributes.itemByName("tars", "mass")
                if attr is None:
                    continue
                mass = units.evaluateExpression(attr.value, "kg")
            m, c, J, raw = body_props(b, mass)
            if path.startswith("central"):
                key = "central"
            else:
                side = "left" if path.startswith("arm:1") else "right"
                key = f"{side}_carrier" if ("carrier" in path or "slide_servo" in path) else f"{side}_sleeve"
            groups.setdefault(key, []).append((m, c, J, body.name))
            if body.name == "pi5":
                sign_check = (m, c, raw)

    # Sign convention, checked on the Pi block (offset in x and z, so its x-z product is m cx cz > 0):
    # Fusion returns tensor elements (-integral x z dm) when its value is negative; else flip them.
    m, c, (_, xz_raw) = sign_check
    product = m * c[0] * c[2]
    tensor_sign = 1.0 if xz_raw * product < 0 else -1.0

    hinge_y = units.evaluateExpression("arm_offset", "cm")
    out = {"sign_check": {"pi5_xz_product_formula": round(product, 4), "fusion_xz": round(xz_raw, 4)}}
    for key, items in sorted(groups.items()):
        M = sum(i[0] for i in items)
        C = [sum(i[0] * i[1][k] for i in items) / M for k in range(3)]
        J = [[0.0] * 3 for _ in range(3)]
        for mi, ci, Ji, _ in items:
            for r in range(3):
                for s in range(3):
                    v = Ji[r][s] if r == s else tensor_sign * Ji[r][s]
                    J[r][s] += v
        # Shift from the world origin to the body's centre of mass: J_c = J_o - M(|C|^2 E - C C^T).
        cc = sum(v * v for v in C)
        for r in range(3):
            for s in range(3):
                J[r][s] -= M * ((cc if r == s else 0.0) - C[r] * C[s])
        frame_y = 0.0 if key == "central" else (hinge_y if key.startswith("left") else -hinge_y)
        out[key] = {
            "mass_kg": round(M, 4),
            "com_m": [round(C[0] / 100, 4), round((C[1] - frame_y) / 100, 4), round(C[2] / 100, 4)],
            "inertia_kgm2": [[round(v * 1e-4, 7) for v in row] for row in J],
            "bodies": [f"{n} {mi * 1000:.0f} g" for mi, _, _, n in items],
        }
    print(json.dumps(out))
