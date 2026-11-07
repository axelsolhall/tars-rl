"""Print each body's world bounding box (mm) and the derived parameters; with BABY set, change `baby` first."""
import adsk.core, adsk.fusion

BABY = None


def boxes(root):
    out = {}
    for occ in root.allOccurrences:
        for b in occ.bRepBodies:
            bb = b.boundingBox
            out[occ.name] = tuple(round(v * 10, 1) for v in (bb.minPoint.x, bb.maxPoint.x, bb.minPoint.y, bb.maxPoint.y, bb.minPoint.z, bb.maxPoint.z))
    return out


def run(_context: str):
    design = adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct)
    if BABY is not None:
        design.userParameters.itemByName("baby").expression = str(BABY)
    p = design.userParameters
    print("baby", p.itemByName("baby").expression, "| depth", round(p.itemByName("depth").value * 10, 2),
          "| total_w", round(p.itemByName("total_w").value * 10, 2), "mm")
    for name, (x0, x1, y0, y1, z0, z1) in boxes(design.rootComponent).items():
        print(f"{name:10s} x {x0:7.1f}..{x1:6.1f}  y {y0:7.1f}..{y1:6.1f}  z {z0:7.1f}..{z1:6.1f}")
