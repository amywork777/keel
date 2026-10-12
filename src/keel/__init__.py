"""keel: a CAD language on NVIDIA's usd-brep (SMLib) kernel.

    from keel import *

    plate = Box(60, 40, 6)
    plate -= Cylinder(3, 6).at(grid(2, 2, 46, 26))
    boss = Circle(10).on(plate.faces(">Z")).extrude(8)
    part = (plate + boss).fillet(">Z and %circle", 1)
    part.export("plate.usda")
"""

from ._kernel import KeelError
from .build import Build
from .geom import Axis, Plane
from .io import Mesh, export, export_mesh_usd, load, tessellate
from .part import BBox, Part, fuse
from .path import Arc, Helix, Line, Path, Pen, Polyline, Spline, ThreePointArc
from .patterns import grid, line_of, polar
from .sketch import (
    Circle,
    Ellipse,
    Polygon,
    Rect,
    RegularPolygon,
    RoundedRect,
    Sketch,
    Slot,
    extrude,
    loft,
    revolve,
    sweep,
)
from .solids import Box, Cone, Cylinder, Sphere, Torus
from .topo import Edge, Face, Selection, Vertex

__version__ = "0.1.0"

__all__ = [
    "Arc", "Axis", "BBox", "Box", "Build", "Circle", "Cone", "Cylinder", "Edge", "Ellipse", "Face",
    "Helix", "KeelError", "Line", "Mesh", "Part", "Path", "Pen", "Plane", "Polygon", "Polyline", "Rect",
    "RegularPolygon", "RoundedRect", "Selection", "Sketch", "Slot", "Sphere", "Spline", "ThreePointArc",
    "Torus", "Vertex", "export", "export_mesh_usd", "extrude", "fuse", "grid", "line_of", "load", "loft",
    "polar", "revolve", "sweep", "tessellate",
]
