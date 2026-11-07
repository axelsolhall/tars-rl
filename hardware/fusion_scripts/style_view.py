"""Make the printed slabs see-through so the parts inside show, and colour the parts by kind."""
import adsk.core
import adsk.fusion

SLAB_OPACITY = 0.25
COLOURS = {  # part name prefix -> RGB
    "battery": (220, 60, 50), "pi5": (40, 160, 60), "hinge_servo": (40, 90, 220), "slide_servo": (40, 90, 220),
    "bus_board": (230, 150, 30), "buck": (230, 150, 30), "imu": (160, 60, 200), "screen": (30, 30, 30),
    "respeaker": (200, 200, 60), "camera": (20, 180, 200),
}


def colour(design, name, rgb):
    lib = design.appearances
    app = lib.itemByName(name)
    if app is None:
        base = adsk.core.Application.get().materialLibraries.itemByName("Fusion Appearance Library").appearances.itemByName("Paint - Enamel Glossy (Yellow)")
        app = lib.addByCopy(base, name)
        for prop in app.appearanceProperties:
            if isinstance(prop, adsk.core.ColorProperty) and prop.id in ("opaque_albedo", "generic_diffuse"):
                prop.value = adsk.core.Color.create(*rgb, 255)
    return app


def run(_context: str):
    design = adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct)
    for comp in design.allComponents:
        for body in comp.bRepBodies:
            if body.name in ("central slab", "arm slab"):
                body.opacity = SLAB_OPACITY
                body.isLightBulbOn = True
            else:
                for prefix, rgb in COLOURS.items():
                    if body.name.startswith(prefix):
                        body.appearance = colour(design, "tars " + prefix, rgb)
    print("styled")
