import adsk.core, adsk.fusion


def run(_context: str):
    app = adsk.core.Application.get()
    print("Fusion", app.version)
    print("active document:", app.activeDocument.name if app.activeDocument else None)
    print("documents open:", [d.name for d in app.documents])
