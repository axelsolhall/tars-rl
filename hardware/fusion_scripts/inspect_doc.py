"""Read-only: print the active design's component tree (grounding, links, bodies, joints, bounding boxes) and parameters."""

import adsk.core
import adsk.fusion


def mm(v):
    return round(v * 10, 1)


def walk(occs, depth, out):
    for occ in occs:
        comp = occ.component
        bb = occ.boundingBox
        flags = [f for f, on in (("grounded", occ.isGrounded), ("linked", occ.isReferencedComponent),
                                 ("hidden", not occ.isLightBulbOn)) if on]
        box = f"x {mm(bb.minPoint.x)}..{mm(bb.maxPoint.x)} y {mm(bb.minPoint.y)}..{mm(bb.maxPoint.y)} z {mm(bb.minPoint.z)}..{mm(bb.maxPoint.z)}" if bb else "empty"
        out.append(f"{'  ' * depth}{occ.name} [{', '.join(flags)}] bodies={comp.bRepBodies.count} sketches={comp.sketches.count} | {box}")
        if not occ.isReferencedComponent:
            walk(occ.childOccurrences, depth + 1, out)


def run(_context: str):
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    out = [f"document: {app.activeDocument.name}", f"root bodies={root.bRepBodies.count} sketches={root.sketches.count}"]
    walk(root.occurrences, 0, out)
    out.append("joints: " + ", ".join(f"{j.name}({j.occurrenceOne.name if j.occurrenceOne else '-'} / {j.occurrenceTwo.name if j.occurrenceTwo else '-'})" for j in root.allJoints) or "none")
    out.append("as-built joints: " + str(len(root.allAsBuiltJoints)))
    out.append("parameters: " + ", ".join(f"{p.name}={p.expression}" for p in design.userParameters))
    print("\n".join(out))
