"""Read a STEP file with OpenCASCADE and print what text-to-cad would see, as JSON.

Runs under a Python that has OCP (the build123d reference venv, .venv-ref), not under keel:

    .venv-ref/bin/python tests/occt_check.py part.step

Faces are listed in ``TopExp.MapShapes`` order, the same order text-to-cad numbers them in
(``label.f1``, ``label.f2``, ...), so a test can check that keel's face i is OCCT's face i. Edges are listed the same way,
with OCCT's own degenerate edges (sphere poles) flagged.
"""

import json
import sys

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedMapOfShape


def entities(shape, kind):
    found = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, found)
    return [found.FindKey(i) for i in range(1, found.Extent() + 1)]


def main(path):
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        return {"read": False}
    reader.TransferRoots()
    shape = reader.OneShape()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props, 1e-7)  # the default is only ~1e-4 on NURBS faces
    faces = []
    for face in entities(shape, TopAbs_FACE):
        fp = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, fp)
        c = fp.CentreOfMass()
        faces.append({"area": fp.Mass(), "center": [c.X(), c.Y(), c.Z()]})
    edges = []
    for edge in entities(shape, TopAbs_EDGE):
        edge = TopoDS.Edge_s(edge)
        if BRep_Tool.Degenerated_s(edge):
            edges.append({"degenerated": True})
            continue
        curve = BRepAdaptor_Curve(edge)
        mid = curve.Value(0.5 * (curve.FirstParameter() + curve.LastParameter()))
        edges.append({"degenerated": False, "type": curve.GetType().name, "mid": [mid.X(), mid.Y(), mid.Z()]})
    return {
        "read": True,
        "valid": BRepCheck_Analyzer(shape).IsValid(),
        "solids": len(entities(shape, TopAbs_SOLID)),
        "volume": props.Mass(),
        "edges": edges,
        "faces": faces,
    }


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1])))
