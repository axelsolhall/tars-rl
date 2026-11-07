"""Check the placed parts: overlaps between parts, parts poking through the slab walls, and the
bought parts' total mass and centre of mass (from each part's stored mass, not its block volume)."""
import itertools

import adsk.core
import adsk.fusion


def run(_context: str):
    design = adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct)
    units, p = design.unitsManager, design.userParameters
    mm = lambda name: p.itemByName(name).value * 10  # noqa: E731
    wall = mm("wall_t")
    inner = {
        "central": (mm("depth") / 2 - wall, mm("central_w") / 2 - wall),
        "arm": (mm("depth") / 2 - wall, mm("arm_w") / 2 - wall),
    }
    half_h = mm("height") / 2 - wall
    boxes, total, moment = [], 0.0, [0.0, 0.0, 0.0]
    for occ in design.rootComponent.allOccurrences:
        attr = occ.component.attributes.itemByName("tars", "mass")
        if attr is None:
            continue
        parent = occ.assemblyContext.component.name
        bb = occ.bRepBodies.item(0).boundingBox
        lo = [v * 10 for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z)]
        hi = [v * 10 for v in (bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]
        boxes.append((occ.fullPathName, lo, hi))
        # Walls: compare in the parent's own frame (its occurrence is translated only along y).
        off = occ.assemblyContext.transform2.translation.y * 10
        hx, hy = inner[parent]
        out = []
        if lo[0] < -hx - 1e-6 or hi[0] > hx + 1e-6:
            out.append(f"x {lo[0]:.1f}..{hi[0]:.1f} vs +-{hx:.1f}")
        if lo[1] - off < -hy - 1e-6 or hi[1] - off > hy + 1e-6:
            out.append(f"y {lo[1]-off:.1f}..{hi[1]-off:.1f} vs +-{hy:.1f}")
        if lo[2] < -half_h - 1e-6 or hi[2] > half_h + 1e-6:
            out.append(f"z {lo[2]:.1f}..{hi[2]:.1f} vs +-{half_h:.1f}")
        if out:
            print(f"THROUGH WALL  {occ.fullPathName}: " + "; ".join(out))
        m = units.evaluateExpression(attr.value, "kg")
        total += m
        for i in range(3):
            moment[i] += m * (lo[i] + hi[i]) / 2
    for (a, alo, ahi), (b, blo, bhi) in itertools.combinations(boxes, 2):
        depth = [min(ahi[i], bhi[i]) - max(alo[i], blo[i]) for i in range(3)]
        if min(depth) > 1e-6:
            print(f"OVERLAP  {a} / {b}: {depth[0]:.1f} x {depth[1]:.1f} x {depth[2]:.1f} mm")
    cog = [c / total for c in moment]
    print(f"bought parts: {len(boxes)} blocks, {total * 1000:.0f} g, centre of mass "
          f"x {cog[0]:.1f}  y {cog[1]:.1f}  z {cog[2]:.1f} mm (hinge centre = 0; floor at {-mm('hinge_z'):.0f})")
