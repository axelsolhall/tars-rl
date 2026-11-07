"""Save the active design as `TARS 50` in the hub's `TARS` project, creating the project if needed."""
import adsk.core

PROJECT, NAME = "TARS", "TARS 50"


def run(_context: str):
    app = adsk.core.Application.get()
    hub = app.data.activeHub
    project = next((p for p in hub.dataProjects if p.name == PROJECT), None) or hub.dataProjects.add(PROJECT)
    doc = app.activeDocument
    if doc.isSaved and doc.dataFile and doc.dataFile.parentProject.id == project.id:
        doc.save("Update from tars-rl")
    else:
        doc.saveAs(NAME, project.rootFolder, "Blocked-out TARS from hardware/tars_params.json", "")
    print("hub", hub.name, "| project", project.name, "| saved as", doc.name)
