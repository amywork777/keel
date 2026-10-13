"""keel123d: build123d's language on NVIDIA's usd-brep kernel.

Scripts written for build123d run here with one changed line, for the subset it covers:

    import keel123d as bd          # instead of: import build123d as bd

    with bd.BuildPart() as flange:
        bd.Cylinder(40, 10, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        with bd.Locations(*bd.PolarLocations(30, 6)):
            bd.Cylinder(3, 10, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN),
                        mode=bd.Mode.SUBTRACT)
    part = bd.fillet(flange.part.edges().filter_by(bd.GeomType.CIRCLE), radius=1)

Conventions follow build123d 0.11: rotations are degrees applied as Rx·Ry·Rz, shapes are
aligned before they are rotated, ``Polygon`` is not centered unless asked, and builder
objects are placed at every location of the active ``Locations``.

Not covered yet: joints and assemblies, text, ``make_hull``, ``offset``, importing files.
"""

from __future__ import annotations

import enum
import inspect
import math
from typing import Iterable, Sequence

from keel import geom as g
from keel import path as _kpath
from keel import sketch as _sk
from keel import solids as _so
from keel._kernel import KeelError, call, ub
from keel.part import Part as _KPart
from keel.sketch import Sketch as _KSketch
from keel.topo import Edge as _KEdge
from keel.topo import Face as _KFace

# ---------------------------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------------------------


class Align(enum.Enum):
    MIN = "min"
    CENTER = "center"
    MAX = "max"
    NONE = None


class Mode(enum.Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    INTERSECT = "intersect"
    REPLACE = "replace"
    PRIVATE = "private"


class GeomType(enum.Enum):
    LINE = "line"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    HYPERBOLA = "hyperbola"
    PARABOLA = "parabola"
    BEZIER = "bezier"
    BSPLINE = "spline"
    OFFSET = "offset"
    OTHER = "other"
    PLANE = "plane"
    CYLINDER = "cylinder"
    CONE = "cone"
    SPHERE = "sphere"
    TORUS = "torus"
    REVOLUTION = "revolution"
    EXTRUSION = "extrusion"

    def __str__(self) -> str:
        return f"GeomType.{self.name}"


class SortBy(enum.Enum):
    LENGTH = "length"
    RADIUS = "radius"
    AREA = "area"
    VOLUME = "volume"
    DISTANCE = "distance"


# ---------------------------------------------------------------------------------------------
# vectors, rotations, locations
# ---------------------------------------------------------------------------------------------


class Vector:
    """A 3D vector with build123d's X/Y/Z accessors."""

    __slots__ = ("X", "Y", "Z")

    def __init__(self, *args):
        if len(args) == 1 and not isinstance(args[0], (int, float)):
            args = tuple(args[0])
        if len(args) == 2:
            args = (args[0], args[1], 0.0)
        if len(args) == 0:
            args = (0.0, 0.0, 0.0)
        self.X, self.Y, self.Z = (float(a) for a in args[:3])

    def __iter__(self):
        return iter((self.X, self.Y, self.Z))

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return (self.X, self.Y, self.Z)[i]

    def __add__(self, o):
        o = Vector(o)
        return Vector(self.X + o.X, self.Y + o.Y, self.Z + o.Z)

    __radd__ = __add__

    def __sub__(self, o):
        o = Vector(o)
        return Vector(self.X - o.X, self.Y - o.Y, self.Z - o.Z)

    def __rsub__(self, o):
        return Vector(o) - self

    def __mul__(self, s: float):
        return Vector(self.X * s, self.Y * s, self.Z * s)

    __rmul__ = __mul__

    def __truediv__(self, s: float):
        return Vector(self.X / s, self.Y / s, self.Z / s)

    def __neg__(self):
        return Vector(-self.X, -self.Y, -self.Z)

    def __eq__(self, o):
        try:
            return (self - Vector(o)).length < 1e-9
        except (TypeError, ValueError):
            return NotImplemented

    __hash__ = None  # mutable-style value semantics, like build123d

    @property
    def length(self) -> float:
        return math.sqrt(self.X**2 + self.Y**2 + self.Z**2)

    def normalized(self) -> "Vector":
        return self / self.length

    def dot(self, o) -> float:
        return g.dot(tuple(self), tuple(Vector(o)))

    def cross(self, o) -> "Vector":
        return Vector(g.cross(tuple(self), tuple(Vector(o))))

    def to_tuple(self):
        return (self.X, self.Y, self.Z)

    def __repr__(self):
        return f"Vector: (X={self.X:g}, Y={self.Y:g}, Z={self.Z:g})"


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _mm(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _mv(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def _transpose(a):
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


_I3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _euler(rx, ry, rz):
    """build123d's rotation tuple: degrees, matrix Rx·Ry·Rz."""
    return _mm(_mm(_rx(math.radians(rx)), _ry(math.radians(ry))), _rz(math.radians(rz)))


class Location:
    """A rigid transform: rotation then translation (build123d semantics)."""

    __slots__ = ("R", "t")

    def __init__(self, *args):
        self.R, self.t = _I3, (0.0, 0.0, 0.0)
        if not args:
            return
        first = args[0]
        if isinstance(first, Location):
            self.R, self.t = first.R, first.t
            return
        if isinstance(first, Plane):
            self.R = tuple(zip(first.x, first.y, first.z))  # columns are the plane axes
            self.t = tuple(first.origin)
            if len(args) > 1:
                raise TypeError("Location(plane) takes no further arguments")
            return
        self.t = tuple(float(c) for c in Vector(first))
        if len(args) >= 2:
            rot = args[1]
            if len(args) == 3:  # Location(position, axis, angle)
                self.R = _axis_angle_matrix(tuple(Vector(rot)), float(args[2]))
            else:
                self.R = _euler(*rot)

    # composition --------------------------------------------------------------------------
    def __mul__(self, other):
        if isinstance(other, Location):
            loc = Location()
            loc.R = _mm(self.R, other.R)
            loc.t = tuple(a + b for a, b in zip(_mv(self.R, other.t), self.t))
            return loc
        if isinstance(other, (list, tuple)) and other and isinstance(other[0], (Shape, Location)):
            return [self * o for o in other]
        if isinstance(other, (Shape, Sketch, Edge, Wire)):
            return other.moved(self)
        if isinstance(other, Plane):
            return Plane(Location(self) * Location(other))
        return NotImplemented

    def inverse(self) -> "Location":
        loc = Location()
        loc.R = _transpose(self.R)
        loc.t = tuple(-c for c in _mv(loc.R, self.t))
        return loc

    def apply(self, p) -> tuple:
        return tuple(a + b for a, b in zip(_mv(self.R, tuple(p)), self.t))

    @property
    def position(self) -> Vector:
        return Vector(self.t)

    @property
    def x_axis(self):
        return tuple(r[0] for r in self.R)

    @property
    def z_axis(self):
        return tuple(r[2] for r in self.R)

    def __iter__(self):
        return iter((self.position, Vector(0, 0, 0)))

    def __repr__(self):
        return f"Location(position={self.t})"


def Pos(*args, X: float = 0.0, Y: float = 0.0, Z: float = 0.0) -> Location:
    if not args:
        return Location((X, Y, Z))
    return Location(Vector(*args))


def Rot(*args, X: float = 0.0, Y: float = 0.0, Z: float = 0.0) -> Location:
    if not args:
        return Location((0, 0, 0), (X, Y, Z))
    if len(args) == 1:
        args = tuple(args[0])
    return Location((0, 0, 0), tuple(args))


Rotation = Rot


def _axis_angle_matrix(axis, deg):
    x, y, z = g.norm(axis)
    a = math.radians(deg)
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return (
        (c + x * x * C, x * y * C - z * s, x * z * C + y * s),
        (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
        (z * x * C - y * s, z * y * C + x * s, c + z * z * C),
    )


class Axis(g.Axis):
    """An axis with build123d's ``position``/``direction`` vectors."""

    X: "Axis"
    Y: "Axis"
    Z: "Axis"

    @property
    def position(self) -> Vector:
        return Vector(self.origin)

    @property
    def direction(self) -> Vector:
        return Vector(self.dir)


Axis.X = Axis((0, 0, 0), (1, 0, 0))
Axis.Y = Axis((0, 0, 0), (0, 1, 0))
Axis.Z = Axis((0, 0, 0), (0, 0, 1))


class Plane(g.Plane):
    """keel's Plane with build123d's constructors and accessors."""

    def __init__(self, origin=(0, 0, 0), x_dir=None, z_dir=(0, 0, 1)):
        if isinstance(origin, Location):
            loc = origin
            super().__init__(loc.t, tuple(r[0] for r in loc.R), tuple(r[2] for r in loc.R))
            return
        if isinstance(origin, g.Plane):
            super().__init__(origin.origin, origin.x, origin.z)
            return
        if hasattr(origin, "_f"):  # a Face
            p = origin._f.plane()
            super().__init__(p.origin, p.x, p.z)
            return
        if x_dir is None:
            x_dir = g._default_x(g.direction(z_dir))
        super().__init__(tuple(Vector(origin)), tuple(Vector(x_dir)), tuple(Vector(z_dir)))

    @property
    def x_dir(self) -> Vector:
        return Vector(self.x)

    @property
    def y_dir(self) -> Vector:
        return Vector(self.y)

    @property
    def z_dir(self) -> Vector:
        return Vector(self.z)

    @property
    def location(self) -> Location:
        return Location(self)

    def offset(self, amount: float) -> "Plane":
        return Plane(g.Plane.offset(self, amount))

    def shift_origin(self, origin) -> "Plane":
        return Plane(tuple(Vector(origin)), self.x, self.z)

    def rotated(self, rotation=(0, 0, 0)) -> "Plane":
        return Plane(Location(self) * Location((0, 0, 0), tuple(rotation)))

    def move(self, loc: Location) -> "Plane":
        return Plane(loc * Location(self))

    def __mul__(self, other):
        if isinstance(other, Location):
            return Location(self) * other
        if isinstance(other, (Shape, Sketch, Edge, Wire)):
            return other.moved(Location(self))
        if isinstance(other, (list, tuple)):
            return [self * o for o in other]
        return NotImplemented

    def __neg__(self):
        return Plane(self.origin, self.x, tuple(-c for c in self.z))


for _name in ("XY", "YX", "XZ", "ZX", "YZ", "ZY"):
    setattr(Plane, _name, Plane(getattr(g.Plane, _name)))
Plane.front = Plane.XZ
Plane.top = Plane.XY
Plane.right = Plane.YZ


class Color:
    """Stores a color as given; keel writes RGB to USD displayColor."""

    def __init__(self, *args, alpha: float = 1.0):
        self.args = args
        self.rgba = _parse_color(args, alpha)

    def __iter__(self):
        return iter(self.rgba)

    def __repr__(self):
        return f"Color{tuple(round(c, 3) for c in self.rgba)}"


_NAMED = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "black": (0, 0, 0), "white": (1, 1, 1),
          "gray": (0.5, 0.5, 0.5), "grey": (0.5, 0.5, 0.5), "yellow": (1, 1, 0), "orange": (1, 0.5, 0)}


def _parse_color(args, alpha):
    if len(args) == 1:
        a = args[0]
        if isinstance(a, str):
            if a.startswith("#"):
                v = a[1:]
                v = "".join(c * 2 for c in v) if len(v) == 3 else v
                return tuple(int(v[i : i + 2], 16) / 255 for i in (0, 2, 4)) + (alpha,)
            return tuple(_NAMED.get(a.lower(), (0.5, 0.5, 0.5))) + (alpha,)
        if isinstance(a, Color):
            return a.rgba
        a = tuple(a)
        return a if len(a) == 4 else a + (alpha,)
    if len(args) in (3, 4):
        return tuple(float(c) for c in args) + (() if len(args) == 4 else (alpha,))
    return (0.5, 0.5, 0.5, alpha)


# ---------------------------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------------------------


def _place(part: _KPart, loc: Location) -> _KPart:
    """Apply a Location (rotation, then translation) to a keel Part."""
    cols = _transpose(loc.R)  # rows of the transpose = columns = images of the axes
    rot = g._axis_angle(cols[0], cols[1], cols[2])
    if rot is not None:
        part = part.rotate(rot[0], rot[1])
    if any(abs(c) > 0 for c in loc.t):
        part = part.move(loc.t)
    return part


class BoundBox:
    def __init__(self, lo, hi):
        self.min, self.max = Vector(lo), Vector(hi)

    @property
    def size(self) -> Vector:
        return self.max - self.min

    def center(self) -> Vector:
        return (self.min + self.max) * 0.5

    @property
    def diagonal(self) -> float:
        return self.size.length

    def __repr__(self):
        return f"BoundBox(min={tuple(self.min)}, max={tuple(self.max)})"


class Shape:
    """A solid body (or a compound of bodies) with build123d's methods."""

    def __init__(self, parts: Sequence[_KPart] | _KPart | None = None, label: str = "", color=None):
        if parts is None:
            parts = []
        self._parts = [parts] if isinstance(parts, _KPart) else list(parts)
        self.label = label
        self.color = color
        self.children: list = []

    # -- internals -----------------------------------------------------------------------
    def _one(self) -> _KPart:
        if not self._parts:
            raise ValueError("empty shape")
        if len(self._parts) == 1:
            return self._parts[0]
        return self._parts[0].fuse(*self._parts[1:])

    def _wrap(self, parts) -> "Shape":
        return Part(parts, self.label, self.color)

    # -- measures ------------------------------------------------------------------------
    @property
    def volume(self) -> float:
        return sum(p.volume for p in self._parts)

    @property
    def area(self) -> float:
        return sum(p.area for p in self._parts)

    @property
    def is_valid(self) -> bool:
        return bool(self._parts) and all(p.is_valid for p in self._parts)

    def bounding_box(self, tolerance=None, optimal: bool = True) -> BoundBox:
        boxes = [p.bbox for p in self._parts]
        lo = tuple(min(b.min[i] for b in boxes) for i in range(3))
        hi = tuple(max(b.max[i] for b in boxes) for i in range(3))
        return BoundBox(lo, hi)

    def center(self, center_of=None) -> Vector:
        vols = [(p.volume, p.center) for p in self._parts]
        total = sum(v for v, _ in vols) or 1.0
        return Vector(tuple(sum(v * c[i] for v, c in vols) / total for i in range(3)))

    @property
    def wrapped(self):
        return self._one().brep

    # -- topology ------------------------------------------------------------------------
    def edges(self) -> "ShapeList":
        return ShapeList(Edge(e, self) for p in self._parts for e in p.edges())

    def faces(self) -> "ShapeList":
        return ShapeList(Face(f, self) for p in self._parts for f in p.faces())

    def vertices(self) -> "ShapeList":
        return ShapeList(Vertex(v) for p in self._parts for v in p.vertices())

    def solids(self) -> "ShapeList":
        out = ShapeList()
        for p in self._parts:
            n = _lump_count(p)
            out.extend([Solid(p, self.label, self.color)] if n == 1 else [_Lump(p, i) for i in range(n)])
        return out

    # -- booleans ------------------------------------------------------------------------
    def fuse(self, *others, glue: bool = False, tol=None) -> "Shape":
        others = [o for o in _flatten(others) if o is not None]
        return Part(self._one().fuse(*[o._one() for o in others]), self.label, self.color)

    def cut(self, *others) -> "Shape":
        others = [o for o in _flatten(others) if o is not None]
        return Part(self._one().cut(*[o._one() for o in others]), self.label, self.color)

    def intersect(self, *others) -> "Shape":
        others = [o for o in _flatten(others) if o is not None]
        return Part(self._one().intersect(*[o._one() for o in others]), self.label, self.color)

    def __add__(self, other):
        return self.fuse(other)

    def __radd__(self, other):
        return self if other == 0 or other is None else NotImplemented

    def __sub__(self, other):
        return self.cut(other)

    def __and__(self, other):
        return self.intersect(other)

    # -- placement -----------------------------------------------------------------------
    def moved(self, loc: Location) -> "Shape":
        return self._wrap([_place(p, Location(loc)) for p in self._parts])

    def move(self, loc: Location) -> "Shape":
        self._parts = [_place(p, Location(loc)) for p in self._parts]
        return self

    def located(self, loc: Location) -> "Shape":
        return self.moved(Location(loc))

    def locate(self, loc: Location) -> "Shape":
        return self.move(Location(loc))

    @property
    def location(self) -> Location:
        return Location()

    def translate(self, vector) -> "Shape":
        return self.moved(Pos(vector))

    def rotate(self, axis: Axis, angle: float) -> "Shape":
        return self._wrap([p.rotate(axis.dir, angle, about=axis.origin) for p in self._parts])

    def mirror(self, mirror_plane: Plane | None = None) -> "Shape":
        return self._wrap([p.mirror(mirror_plane or Plane.XY) for p in self._parts])

    def clean(self) -> "Shape":
        return self

    def scale(self, factor: float) -> "Shape":
        return self._wrap([p.scale(factor) for p in self._parts])

    # -- features ------------------------------------------------------------------------
    def fillet(self, radius: float, edge_list: Iterable) -> "Shape":
        return Part(self._one().fillet([e._e for e in edge_list], radius), self.label, self.color)

    def chamfer(self, length: float, length2: float | None, edge_list: Iterable, face=None) -> "Shape":
        return Part(self._one().chamfer([e._e for e in edge_list], length), self.label, self.color)

    def __repr__(self):
        return f"<{type(self).__name__} {self.label!r} volume={self.volume:.6g}>"


class Part(Shape):
    pass


class Solid(Part):
    pass


class _Lump(Solid):
    """One solid of a multi-body part (keel can't split bodies yet: measures cover the whole part)."""

    def __init__(self, part, index):
        super().__init__(part)
        self.index = index


class Compound(Shape):
    def __init__(self, obj=None, label: str = "", color=None, children: Sequence | None = None, **_):
        items = list(children or []) or ([obj] if isinstance(obj, Shape) else list(obj or []))
        parts = [p for it in items for p in it._parts]
        super().__init__(parts, label, color)
        self.children = items


def _flatten(items) -> list:
    out = []
    for it in items:
        if isinstance(it, (list, tuple, ShapeList)):
            out.extend(_flatten(it))
        else:
            out.append(it)
    return out


def _lump_count(part: _KPart) -> int:
    faces = list(part.brep.faces())
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in part.brep.edges():
        fs = e.faces()
        idx = [next(i for i, f in enumerate(faces) if f == x) for x in fs]
        for a in idx[1:]:
            parent[find(a)] = find(idx[0])
    return len({find(i) for i in range(len(faces))})


class Edge:
    """An edge: either an edge of a part, or a free curve (built from a recipe of points)."""

    def __init__(self, e: _KEdge | None = None, owner: Shape | None = None, *, recipe: "_Recipe | None" = None):
        self._e = e
        self._owner = owner
        self._recipe = recipe

    # -- free-curve constructors (build123d's Edge.make_*) --------------------------------
    @classmethod
    def make_line(cls, point1, point2) -> "Edge":
        return cls(recipe=_Recipe("line", [point1, point2]))

    @classmethod
    def make_three_point_arc(cls, point1, point2, point3) -> "Edge":
        return cls(recipe=_Recipe("arc", [point1, point2, point3]))

    @classmethod
    def make_spline(cls, points, tangents=None, periodic: bool = False, **_) -> "Edge":
        return cls(recipe=_Recipe("spline", list(points), periodic=periodic))

    @classmethod
    def make_bezier(cls, *points, weights=None) -> "Edge":
        pts = list(points[0]) if len(points) == 1 and not isinstance(points[0][0], (int, float)) else list(points)
        return cls(recipe=_Recipe("bezier", pts))

    @classmethod
    def make_circle(cls, radius: float, plane: "Plane | None" = None, start_angle: float = 360,
                    end_angle: float = 360, angular_direction=None) -> "Edge":
        plane = plane or Plane.XY
        if abs((end_angle - start_angle) % 360) > 1e-9 and not (start_angle == 360 and end_angle == 360):
            a0, a1 = math.radians(start_angle), math.radians(end_angle)
            am = (a0 + a1) / 2 if a1 > a0 else (a0 + a1 + 2 * math.pi) / 2
            pts = [plane.to_world((radius * math.cos(a), radius * math.sin(a), 0)) for a in (a0, am, a1)]
            return cls(recipe=_Recipe("arc", pts))
        pts = [plane.to_world((radius * math.cos(a), radius * math.sin(a), 0)) for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)]
        return cls(recipe=_Recipe("circle", pts))

    # -- measures (part edges or free curves) ---------------------------------------------
    @property
    def geom_type(self) -> GeomType:
        if self._recipe is not None:
            return {"line": GeomType.LINE, "arc": GeomType.CIRCLE, "circle": GeomType.CIRCLE}.get(
                self._recipe.kind, GeomType.BSPLINE)
        return {"line": GeomType.LINE, "circle": GeomType.CIRCLE}.get(self._e.geom_type, GeomType.BSPLINE)

    def _circle(self):
        if self._recipe is not None:
            return self._recipe.circle()
        return self._e.circle

    @property
    def radius(self) -> float:
        c = self._circle()
        if c is None:
            raise ValueError("edge is not circular")
        return c[1]

    @property
    def arc_center(self) -> Vector:
        c = self._circle()
        if c is None:
            raise ValueError("edge is not circular")
        return Vector(c[0])

    def normal(self) -> Vector:
        c = self._circle()
        if c is None:
            raise ValueError("normal is only defined for circular edges here")
        return Vector(c[2])

    @property
    def length(self) -> float:
        return self._recipe.build().length() if self._recipe is not None else self._e.length

    def start_point(self) -> Vector:
        return Vector(self._recipe.start if self._recipe is not None else self._e.start)

    def end_point(self) -> Vector:
        return Vector(self._recipe.end if self._recipe is not None else self._e.end)

    def position_at(self, t: float) -> Vector:
        if self._recipe is not None:
            c = self._recipe.build()
            a, b = c.parameter_range()
            return Vector(c.evaluate(a + t * (b - a)))
        return Vector(self._e.point(t))

    def center(self, center_of=None) -> Vector:
        return self.position_at(0.5) if self._recipe is not None else Vector(self._e.mid)

    def tangent_at(self, t: float = 0.5) -> Vector:
        if self._recipe is not None:
            a, b = self.position_at(max(0.0, t - 1e-6)), self.position_at(min(1.0, t + 1e-6))
            return (b - a).normalized()
        return Vector(self._e.tangent(t))

    def bounding_box(self) -> BoundBox:
        lo, hi = (self._recipe.build() if self._recipe is not None else self._e._h).bounding_box()
        return BoundBox(lo, hi)

    # -- free-curve algebra --------------------------------------------------------------
    def edges(self) -> "ShapeList":
        return ShapeList([self])

    def moved(self, loc: Location) -> "Edge":
        return Edge(recipe=self._free().mapped(Location(loc).apply))

    def mirror(self, about: "Plane | None" = None) -> "Edge":
        return Edge(recipe=self._free().mapped(_reflector(about or Plane.XZ)))

    def __add__(self, other):
        return Wire([self, *other.edges()])

    def trim(self, start: float, end: float) -> "Edge":
        """The part of a free line, arc or Bezier between fractions ``start`` and ``end``."""
        r = self._free()
        if r.kind in ("line", "arc", "circle"):
            pts = [tuple(self.position_at(t)) for t in (start, (start + end) / 2, end)]
            return Edge(recipe=_Recipe("line" if r.kind == "line" else "arc", pts if r.kind != "line" else [pts[0], pts[2]]))
        if r.kind == "bezier":
            return Edge(recipe=_Recipe("bezier", _bezier_segment(r.points, start, end)))
        raise KeelError("trimming splines is not supported yet (the kernel doesn't expose their control points)")

    def _free(self) -> "_Recipe":
        if self._recipe is None:
            raise KeelError("this operation needs a free curve, not an edge of a part")
        return self._recipe

    def __repr__(self):
        return f"<Edge {self.geom_type.name} {tuple(round(c, 3) for c in self.start_point())}>"


class Wire:
    """A chain of free edges."""

    def __init__(self, edges=None):
        self._edges = [e for e in _flatten(edges or []) for e in (e.edges() if isinstance(e, Wire) else [e])]

    @classmethod
    def make_polygon(cls, points, close: bool = True) -> "Wire":
        pts = [tuple(Vector(p)) for p in points]
        if close and g.dist(pts[0], pts[-1]) > 1e-9:
            pts.append(pts[0])
        return cls([Edge.make_line(a, b) for a, b in zip(pts, pts[1:])])

    @classmethod
    def make_circle(cls, radius: float, plane: "Plane | None" = None) -> "Wire":
        return cls([Edge.make_circle(radius, plane)])

    def edges(self) -> "ShapeList":
        return ShapeList(self._edges)

    def moved(self, loc: Location) -> "Wire":
        return Wire([e.moved(loc) for e in self._edges])

    def mirror(self, about: "Plane | None" = None) -> "Wire":
        return Wire([e.mirror(about) for e in self._edges])

    def __add__(self, other):
        return Wire([*self._edges, *other.edges()])

    def start_point(self) -> Vector:
        return self._edges[0].start_point()

    def end_point(self) -> Vector:
        return self._edges[-1].end_point()

    @property
    def length(self) -> float:
        return sum(e.length for e in self._edges)


class _Recipe:
    """A free curve as its defining points, so rigid moves and mirrors stay exact."""

    def __init__(self, kind: str, points, periodic: bool = False):
        self.kind = kind
        self.points = [tuple(float(c) for c in Vector(p)) for p in points]
        self.periodic = periodic

    def mapped(self, f) -> "_Recipe":
        return _Recipe(self.kind, [f(p) for p in self.points], self.periodic)

    @property
    def start(self):
        return self.points[0]

    @property
    def end(self):
        if self.kind == "circle" or (self.kind == "spline" and self.periodic):
            return self.points[0]
        return self.points[-1]

    def circle(self):
        if self.kind not in ("arc", "circle"):
            return None
        a, b, c = self.points[:3]
        return _circumcircle(a, b, c)

    def build(self):
        """A fresh kernel curve."""
        k = self.kind
        if k == "line":
            return ub.create_line_segment(self.points[0], self.points[1])
        if k == "bezier":
            n = len(self.points)
            if n == 2:
                return ub.create_line_segment(*self.points)
            return ub.create_canonical_curve(self.points, [0.0, 1.0], [n, n], degree=n - 1)
        if k == "spline":
            pts = self.points + ([self.points[0]] if self.periodic and g.dist(self.points[0], self.points[-1]) > 1e-9 else [])
            return _kpath._Spline(pts).build()
        center, r, n = self.circle()
        x = g.norm(g.sub(self.points[0], center))
        frame = g.Plane(center, x, n)
        if k == "circle":
            return frame.place(ub.create_circle((0.0, 0.0, 0.0), r))
        end = frame.to_local(self.points[2])
        sweep = math.degrees(math.atan2(end[1], end[0])) % 360 or 360.0
        return frame.place(ub.create_arc((0.0, 0.0, 0.0), r, 0.0, sweep))


def _bezier_segment(pts, t0: float, t1: float):
    """Control points of a Bezier restricted to [t0, t1] (exact, by de Casteljau splits)."""
    def split(ctrl, t):
        left, right, level = [ctrl[0]], [ctrl[-1]], list(ctrl)
        while len(level) > 1:
            level = [g.add(g.mul(a, 1 - t), g.mul(b, t)) for a, b in zip(level, level[1:])]
            left.append(level[0])
            right.append(level[-1])
        return left, right[::-1]
    _, tail = split(pts, t0)
    head, _ = split(tail, 0.0 if t0 >= 1 else (t1 - t0) / (1 - t0))
    return head


def _circumcircle(a, b, c):
    """(center, radius, normal) of the circle through a, b, c, oriented so a -> b -> c runs
    counter-clockwise about the normal."""
    ab, ac = g.sub(b, a), g.sub(c, a)
    n = g.cross(ab, ac)
    nn = g.dot(n, n)
    if nn < 1e-24:
        raise KeelError("arc points are collinear")
    center = g.add(a, g.mul(g.add(g.mul(g.cross(n, ab), g.dot(ac, ac)), g.mul(g.cross(ac, n), g.dot(ab, ab))), 1 / (2 * nn)))
    return center, g.dist(center, a), g.norm(n)


def _reflector(plane: g.Plane):
    o, n = plane.origin, plane.z
    return lambda p: g.sub(p, g.mul(n, 2 * g.dot(g.sub(p, o), n)))


def _line_plane() -> Plane:
    """The plane line objects take tuple points in: the active BuildLine's (or BuildPart's)
    first workplane, else XY. A sketch's lines are drawn in its local XY."""
    builder = _innermost()
    if isinstance(builder, BuildLine):
        return builder.workplane
    if isinstance(builder, BuildPart):
        return builder.workplanes[0]
    return Plane.XY


def _localize(*pts) -> list[tuple]:
    """build123d's WorkplaneList.localize: tuples are workplane coordinates, Vectors are global."""
    plane = _line_plane()
    return [tuple(plane.to_world(tuple(Vector(p)))) if isinstance(p, (tuple, list)) else tuple(Vector(p))
            for p in pts]


def _point_list(pts) -> list:
    return list(pts[0]) if len(pts) == 1 and not isinstance(pts[0][0], (int, float)) else list(pts)


def _to_line(curve, mode: Mode):
    """Add a line object's edges to an active BuildLine, like build123d's line objects do."""
    builder = _innermost()
    if isinstance(builder, BuildLine):
        builder._combine(list(curve.edges()), mode)
    return curve


def Line(*pts, mode: Mode = Mode.ADD) -> Edge:
    pts = _point_list(pts)
    if len(pts) != 2:
        raise ValueError("Line requires two points")
    return _to_line(Edge.make_line(*_localize(*pts)), mode)


def Polyline(*pts, close: bool = False, mode: Mode = Mode.ADD) -> Wire:
    return _to_line(Wire.make_polygon(_localize(*_point_list(pts)), close=close), mode)


def Spline(*pts, tangents=None, tangent_scalars=None, periodic: bool = False, mode: Mode = Mode.ADD) -> Edge:
    return _to_line(Edge.make_spline(_localize(*_point_list(pts)), periodic=periodic), mode)


def Bezier(*cntl_pnts, weights=None, mode: Mode = Mode.ADD) -> Edge:
    return _to_line(Edge.make_bezier(_localize(*_point_list(cntl_pnts))), mode)


def ThreePointArc(*pts, mode: Mode = Mode.ADD) -> Edge:
    pts = _point_list(pts)
    if len(pts) != 3:
        raise ValueError("ThreePointArc requires three points")
    return _to_line(Edge.make_three_point_arc(*_localize(*pts)), mode)


def CenterArc(center, radius: float, start_angle: float, arc_size: float, mode: Mode = Mode.ADD) -> Edge:
    """An arc on a plane parallel to the workplane, through ``center``."""
    (c,) = _localize(center)
    frame = _line_plane()
    angles = [math.radians(start_angle + arc_size * f) for f in (0.0, 0.5, 1.0)]

    def at(a):
        return g.add(c, g.add(g.mul(frame.x, radius * math.cos(a)), g.mul(frame.y, radius * math.sin(a))))

    if abs(abs(arc_size) - 360) < 1e-9:
        return _to_line(Edge(recipe=_Recipe("circle", [at(angles[0] + k * 2 * math.pi / 3) for k in range(3)])), mode)
    return _to_line(Edge.make_three_point_arc(*[at(a) for a in angles]), mode)


def RadiusArc(start_point, end_point, radius: float, short_sagitta: bool = True, mode: Mode = Mode.ADD) -> Edge:
    """build123d's sagitta construction: a positive radius takes the short arc bulging to the
    left of start -> end (about the workplane normal); ``short_sagitta=False`` or a negative
    radius flips it."""
    a, b = _localize(start_point, end_point)
    half = g.dist(a, b) / 2
    if radius * radius < half * half:
        raise ValueError("Arc radius is not large enough to reach the end point.")
    h = math.sqrt(radius * radius - half * half)
    sagitta = abs(radius) - h if short_sagitta else -abs(radius) - h
    if radius < 0:
        sagitta = -sagitta
    left = g.norm(g.cross(_line_plane().z, g.sub(b, a)))
    mid = g.mul(g.add(a, b), 0.5)
    sag_point = g.add(mid, g.mul(left, abs(sagitta) if sagitta > 0 else -abs(sagitta)))
    return _to_line(Edge.make_three_point_arc(a, sag_point, b), mode)


# ---------------------------------------------------------------------------------------------
# faces from curves; revolve, loft, sweep
# ---------------------------------------------------------------------------------------------


def _free_edges(obj) -> list:
    if isinstance(obj, Wire):
        return list(obj.edges())
    if isinstance(obj, Edge):
        return [obj]
    if isinstance(obj, (list, tuple, ShapeList)):
        return [e for o in obj for e in _free_edges(o)]
    raise TypeError(f"expected edges or wires, got {type(obj).__name__}")


def _chain(edges):
    """Order free edges head to tail (reversing where needed); returns sample points along the loop."""
    rest = list(edges)
    first = rest.pop(0)
    pts = [first.position_at(t / 8) for t in range(9)]
    while rest:
        end = pts[-1]
        for i, e in enumerate(rest):
            if (e.start_point() - end).length < 1e-6:
                pts += [e.position_at(t / 8) for t in range(1, 9)]
                break
            if (e.end_point() - end).length < 1e-6:
                pts += [e.position_at(1 - t / 8) for t in range(1, 9)]
                break
        else:
            e = rest[0]
            i = 0
            pts += [e.position_at(t / 8) for t in range(9)]
        rest.pop(i)
    return [tuple(p) for p in pts]


def _newell(points) -> tuple:
    n = [0.0, 0.0, 0.0]
    for p, q in zip(points, points[1:] + points[:1]):
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return g.norm(tuple(n))


def _face(edges, face_up: bool) -> Sketch:
    """A planar face bounded by free edges. Its normal follows the loop's direction, or
    points up (+Z) when ``face_up`` and the loop runs clockwise seen from above."""
    es = _free_edges(edges)
    if not es:
        raise ValueError("No objects to create a face")
    pts = _chain(es)
    normal = _newell(pts)
    if face_up and normal[2] < -1e-12:
        normal = g.mul(normal, -1)
    center = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))
    plane = Plane(center, None, normal)
    sheet = call("make_face", ub.create_planar_faces, [e._free().build() for e in es])
    return Sketch(_KSketch(plane.unplace(sheet), plane))


def _local_face(edges) -> _KSketch:
    """A face from edges drawn in a sketch's local XY."""
    es = _free_edges(edges)
    if not es:
        raise ValueError("No objects to create a face")
    for e in es:
        if any(abs(e.position_at(t / 4).Z) > 1e-7 for t in range(5)):
            raise KeelError("make_face: a sketch's edges must lie in its local XY plane (z = 0)")
    return _KSketch(call("make_face", ub.create_planar_faces, [e._free().build() for e in es]))


def make_face(edges=None, mode: Mode = Mode.ADD) -> Sketch:
    """A planar face from perimeter edges (default: a BuildSketch's pending edges, which a
    BuildLine inside it leaves there). Faces point up (+Z) unless they are vertical."""
    builder = _innermost()
    if not isinstance(builder, BuildSketch):
        if edges is None:
            raise ValueError("No objects to create a face")
        return _face(edges, face_up=True)
    if edges is None:
        edges, builder.pending_edges = builder.pending_edges, []
    face = _local_face(edges)
    builder._combine([face], mode)
    return Sketch(face)


def _sections(obj) -> list[_KSketch]:
    return _sketches_of(obj)


def revolve(profiles=None, axis: Axis = Axis.Z, revolution_arc: float = 360.0, clean: bool = True,
            mode: Mode = Mode.ADD) -> Part:
    builder = _innermost()
    items = _sections(profiles)
    if not items and isinstance(builder, BuildPart):
        items, builder.pending = builder.pending, []
    if not items:
        raise ValueError("revolve needs a profile")
    bodies = []
    for it in items:
        o = it.plane.to_local(axis.origin)
        d = (g.dot(axis.dir, it.plane.x), g.dot(axis.dir, it.plane.y), g.dot(axis.dir, it.plane.z))
        if abs(o[2]) > 1e-6 or abs(d[2]) > 1e-6:
            raise KeelError("revolve: the axis must lie in the profile's plane")
        bodies.append(it.revolve(revolution_arc, axis=((o[0], o[1]), (d[0], d[1]))))
    if isinstance(builder, BuildPart):
        builder._combine(list(bodies), mode)
    return Part(bodies[0].fuse(*bodies[1:]) if len(bodies) > 1 else bodies[0])


def loft(sections=None, ruled: bool = False, clean: bool = True, mode: Mode = Mode.ADD) -> Part:
    builder = _innermost()
    items = _sections(sections)
    if not items and isinstance(builder, BuildPart):
        items, builder.pending = builder.pending, []
    body = _sk.loft(*items, smooth=not ruled)
    if isinstance(builder, BuildPart):
        builder._combine([body], mode)
    return Part(body)


def sweep(sections=None, path=None, multisection: bool = False, is_frenet: bool = False, transition=None,
          normal=None, binormal=None, clean: bool = True, mode: Mode = Mode.ADD) -> Part:
    """Sweep a profile along a path. The profile stays where it is (put it at the path start).

    Exact for round profiles (a pipe) and for profiles centered on a planar path's start;
    otherwise keel warns, because the kernel re-anchors the profile."""
    import warnings

    builder = _innermost()
    items = _sections(sections)
    if not items and isinstance(builder, BuildPart):
        items, builder.pending = builder.pending, []
    if path is None and isinstance(builder, BuildPart) and builder.pending_edges:
        path, builder.pending_edges = builder.pending_edges, []
    if len(items) != 1 or path is None:
        raise KeelError("sweep needs one profile and a path")
    item = items[0]
    edges = _free_edges(path)
    curves = [e._free().build() for e in edges]
    start = tuple(edges[0].start_point())
    loops = item._loop_curves()
    if loops[0][1]:
        raise KeelError("sweep profiles with holes are not supported yet")
    radius = _sk._circle_radius(loops[0][0])
    center = loops[0][0][0].center() if radius is not None else None
    path_pts = [tuple(e.position_at(t / 6)) for e in edges for t in range(7)]
    if radius is not None and g.dist(tuple(center), start) < 1e-6:
        if len(curves) == 1:
            single = curves[0] if type(curves[0]).__name__ != "Line" else _kpath._Line(*edges[0]._free().points).build_bspline()
            body = call("sweep", ub.pipe_sweep, single, radius=radius)
        else:
            body = call("sweep", ub.sweep_along_planar_path, loops[0][0], curves, move_profile=True)
    elif _kpath._coplanar(path_pts):
        (lo, hi) = item._placed().bounding_box(True)
        if g.dist(g.mul(g.add(lo, hi), 0.5), start) > 1e-6:
            warnings.warn("sweep: the kernel centers the profile on the path start", stacklevel=2)
        body = call("sweep", ub.sweep_along_planar_path, loops[0][0], curves, move_profile=True)
    else:
        warnings.warn("sweep: on a non-planar path the kernel anchors the profile's first point on the path",
                      stacklevel=2)
        body = call("sweep", ub.curve_sweep, loops[0][0], curves[0])
    part = _KPart(body)
    if isinstance(builder, BuildPart):
        builder._combine([part], mode)
    return Part(part)


class Face:
    def __new__(cls, *args, **kwargs):
        # build123d's Face(wire) / Face(outer_wire, inner_wires) builds a planar face
        if args and isinstance(args[0], (Wire, Edge, list, tuple)):
            face = _face(args[0], face_up=False)
            for hole in (args[1] if len(args) > 1 else kwargs.get("inner_wires", [])) or []:
                face = face - _face(hole, face_up=False)
            return face
        return super().__new__(cls)

    def __init__(self, f: _KFace, owner: Shape | None = None):
        self._f = f
        self._owner = owner

    def vertices(self) -> "ShapeList":
        seen, out = set(), ShapeList()
        for e in self._f.edges():
            for pnt in (e.start, e.end):
                key = tuple(round(c, 9) for c in pnt)
                if key not in seen:
                    seen.add(key)
                    out.append(Vector(pnt))
        return out

    @property
    def geom_type(self) -> GeomType:
        return GeomType.PLANE if self._f.is_planar else GeomType.BSPLINE

    @property
    def area(self) -> float:
        return self._f.area

    def center(self, center_of=None) -> Vector:
        return Vector(self._f.center)

    def normal_at(self, *args) -> Vector:
        if args and not isinstance(args[0], (int, float)):
            return Vector(self._f.normal_at(tuple(Vector(args[0]))))
        return Vector(self._f.normal)

    def edges(self) -> "ShapeList":
        return ShapeList(Edge(e, self._owner) for e in self._f.edges())

    def bounding_box(self) -> BoundBox:
        lo, hi = self._f._h.bounding_box()
        return BoundBox(lo, hi)


class Vertex:
    def __init__(self, v):
        self._v = v

    @property
    def X(self):
        return self._v.point[0]

    @property
    def Y(self):
        return self._v.point[1]

    @property
    def Z(self):
        return self._v.point[2]

    def center(self, center_of=None) -> Vector:
        return Vector(self._v.point)


class ShapeList(list):
    """build123d's filterable list of shapes."""

    def filter_by(self, filter_by, reverse: bool = False, tolerance: float = 1e-5) -> "ShapeList":
        if isinstance(filter_by, GeomType):
            keep = [s for s in self if s.geom_type == filter_by]
        elif isinstance(filter_by, g.Axis):
            d = filter_by.dir
            keep = [s for s in self if _parallel(s, d, tolerance)]
        elif callable(filter_by):
            keep = [s for s in self if filter_by(s)]
        else:
            raise TypeError(f"cannot filter by {filter_by!r}")
        if reverse:
            keep = [s for s in self if s not in keep]
        return ShapeList(keep)

    def filter_by_position(self, axis: Axis, minimum: float, maximum: float, inclusive=(True, True)) -> "ShapeList":
        d = axis.dir
        lo_in, hi_in = inclusive

        def ok(s):
            v = g.dot(tuple(s.center()), d)
            return (v >= minimum - 1e-7 if lo_in else v > minimum) and (v <= maximum + 1e-7 if hi_in else v < maximum)

        return ShapeList(s for s in self if ok(s))

    def sort_by(self, sort_by=Axis.Z, reverse: bool = False) -> "ShapeList":
        if isinstance(sort_by, g.Axis):
            key = lambda s: g.dot(tuple(s.center()), sort_by.dir)  # noqa: E731
        elif isinstance(sort_by, SortBy):
            key = {SortBy.LENGTH: lambda s: s.length, SortBy.RADIUS: lambda s: s.radius,
                   SortBy.AREA: lambda s: s.area, SortBy.VOLUME: lambda s: s.volume,
                   SortBy.DISTANCE: lambda s: s.center().length}[sort_by]
        else:
            key = sort_by
        return ShapeList(sorted(self, key=key, reverse=reverse))

    def group_by(self, group_by=Axis.Z, reverse: bool = False, tol_digits: int = 6) -> list:
        groups: dict = {}
        for s in self:
            k = round(g.dot(tuple(s.center()), group_by.dir), tol_digits) if isinstance(group_by, g.Axis) else group_by(s)
            groups.setdefault(k, ShapeList()).append(s)
        return [groups[k] for k in sorted(groups, reverse=reverse)]

    @property
    def first(self):
        return self[0]

    @property
    def last(self):
        return self[-1]

    def __getitem__(self, i):
        r = super().__getitem__(i)
        return ShapeList(r) if isinstance(i, slice) else r

    def __gt__(self, axis):  # build123d: shapes > Axis.Z sorts ascending
        return self.sort_by(axis)

    def __lt__(self, axis):
        return self.sort_by(axis, reverse=True)

    def __rshift__(self, axis):  # highest group along the axis
        return self.group_by(axis)[-1]

    def __lshift__(self, axis):
        return self.group_by(axis)[0]


def _parallel(s, d, tol) -> bool:
    if isinstance(s, Edge):
        v = s._e.direction
    elif isinstance(s, Face):
        v = s._f.normal if s._f.is_planar else None
    else:
        v = None
    return v is not None and abs(abs(g.dot(v, d)) - 1) < tol


# ---------------------------------------------------------------------------------------------
# sketches (2D faces placed on a plane)
# ---------------------------------------------------------------------------------------------


class Sketch:
    """Planar regions placed on planes (build123d's Sketch and Face, for profiles).

    Each item is a keel Sketch: a region in a plane's local XY plus that plane."""

    def __init__(self, items: Sequence[_KSketch] | _KSketch | None = None):
        self._items = [items] if isinstance(items, _KSketch) else list(items or [])

    @property
    def area(self) -> float:
        return sum(s.area for s in self._items)

    def moved(self, loc: Location) -> "Sketch":
        return Sketch([s.on(_plane_moved(s.plane, loc)) for s in self._items])

    located = moved

    def faces(self) -> "ShapeList":
        return ShapeList(Sketch(s) for s in self._items)

    def _boolean(self, other: "Sketch", op: str) -> "Sketch":
        if isinstance(other, (list, tuple)):
            out = self
            for o in _flatten(other):
                out = out._boolean(o, op)
            return out
        if not isinstance(other, Sketch):
            return NotImplemented
        if not self._items:
            return other if op == "+" else self
        result = list(self._items)
        leftovers = []
        for b in other._items:
            for i, a in enumerate(result):
                if _coplanar(a.plane, b.plane):
                    moved_b = _KSketch(a.plane.unplace(b.plane.place(b._sheet.copy())), a.plane)
                    result[i] = {"+": a + moved_b, "-": a - moved_b, "&": a & moved_b}[op]
                    break
            else:
                if op == "+":
                    leftovers.append(b)
        return Sketch(result + leftovers)

    def __add__(self, other):
        return self._boolean(other, "+")

    def __sub__(self, other):
        return self._boolean(other, "-")

    def __and__(self, other):
        return self._boolean(other, "&")

    def __radd__(self, other):
        return self if other == 0 or other is None else NotImplemented

    def mirror(self, about: "Plane | None" = None) -> "Sketch":
        raise KeelError("mirroring sketches is not supported yet; mirror the curves before make_face")

    def __repr__(self):
        return f"<Sketch regions={len(self._items)} area={self.area:.6g}>"


def _coplanar(p: g.Plane, q: g.Plane, tol: float = 1e-7) -> bool:
    return abs(abs(g.dot(p.z, q.z)) - 1) < tol and abs(g.dot(g.sub(q.origin, p.origin), p.z)) < tol * max(1.0, g.length(p.origin))


def _plane_moved(p: g.Plane, loc: Location) -> Plane:
    return Plane(Location(loc) * Location(Plane(p)))


def _align_sketch(s: _KSketch, align) -> _KSketch:
    if align is None:
        return s
    modes = (align, align) if isinstance(align, Align) else tuple(align)
    (x0, y0), (x1, y1) = s.bbox
    shift = []
    for mode, lo, hi in zip(modes, (x0, y0), (x1, y1)):
        m = mode.value if isinstance(mode, Align) else mode
        shift.append({"min": -lo, "center": -(lo + hi) / 2, "max": -hi, None: 0.0}[m])
    return s.move(*shift) if any(abs(v) > 0 for v in shift) else s


def _sketch_object(make, rotation: float, align, mode: Mode):
    s = _align_sketch(make(), align)
    if rotation:
        s = s.rotate(rotation)
    builder = _innermost()
    if isinstance(builder, BuildSketch):
        placed = []
        for loc in _locations(builder):
            angle = _z_angle(loc)
            placed.append((s.rotate(angle) if abs(angle) > 1e-12 else s).move(*loc.t[:2]))
        builder._combine(placed, mode)
    return Sketch(s)


def _z_angle(loc: Location) -> float:
    return math.degrees(math.atan2(loc.R[1][0], loc.R[0][0]))


_C2 = (Align.CENTER, Align.CENTER)


def Rectangle(width: float, height: float, rotation: float = 0, align=_C2, mode: Mode = Mode.ADD):
    return _sketch_object(lambda: _sk.Rect(width, height), rotation, align, mode)


def RectangleRounded(width: float, height: float, radius: float, rotation: float = 0, align=_C2, mode: Mode = Mode.ADD):
    return _sketch_object(lambda: _sk.RoundedRect(width, height, radius), rotation, align, mode)


def Circle(radius: float, align=_C2, mode: Mode = Mode.ADD):
    return _sketch_object(lambda: _sk.Circle(radius), 0, align, mode)


def Ellipse(x_radius: float, y_radius: float, rotation: float = 0, align=_C2, mode: Mode = Mode.ADD):
    return _sketch_object(lambda: _sk.Ellipse(x_radius, y_radius), rotation, align, mode)


def Polygon(*pts, rotation: float = 0, align=None, mode: Mode = Mode.ADD):
    pts = list(pts[0]) if len(pts) == 1 and not isinstance(pts[0][0], (int, float)) else list(pts)
    return _sketch_object(lambda: _sk.Polygon([tuple(Vector(p))[:2] for p in pts]), rotation, align, mode)


def RegularPolygon(radius: float, side_count: int, major_radius: bool = True, rotation: float = 0,
                   align=_C2, mode: Mode = Mode.ADD):
    r = radius if major_radius else radius / math.cos(math.pi / side_count)
    return _sketch_object(lambda: _sk.RegularPolygon(r, side_count), rotation, align, mode)


def SlotOverall(width: float, height: float, rotation: float = 0, align=_C2, mode: Mode = Mode.ADD):
    return _sketch_object(lambda: _sk.Slot(width, height), rotation, align, mode)


def SlotCenterToCenter(center_separation: float, height: float, rotation: float = 0, mode: Mode = Mode.ADD):
    return _sketch_object(lambda: _sk.Slot(center_separation + height, height), rotation, _C2, mode)


# ---------------------------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------------------------

_builders: list = []
_location_stack: list = []  # (builder, [Location, ...])


def _innermost():
    return _builders[-1] if _builders else None


def _locations(builder) -> list:
    for owner, locs in reversed(_location_stack):
        if owner is builder:
            return locs
    return [Location()]


class Locations:
    """Positions for objects created inside it (relative to enclosing Locations)."""

    def __init__(self, *pts):
        flat = []
        for p in pts:
            if isinstance(p, (list, tuple)) and p and isinstance(p[0], (Location, Plane, Vector, tuple, list)) and not isinstance(p[0], (int, float)):
                flat.extend(p)
            else:
                flat.append(p)
        self.locations = [p if isinstance(p, Location) else Location(p) if isinstance(p, Plane) else Location(Vector(p)) for p in flat]

    def __enter__(self):
        builder = _innermost()
        parent = _locations(builder)
        combined = [a * b for a in parent for b in self.locations]
        _location_stack.append((builder, combined))
        return self

    def __exit__(self, *exc):
        _location_stack.pop()

    def __iter__(self):
        return iter(self.locations)


def GridLocations(x_spacing: float, y_spacing: float, x_count: int, y_count: int, align=_C2) -> Locations:
    pts = [(i * x_spacing - (x_count - 1) * x_spacing / 2, j * y_spacing - (y_count - 1) * y_spacing / 2)
           for i in range(x_count) for j in range(y_count)]
    return Locations(*pts)


def PolarLocations(radius: float, count: int, start_angle: float = 0, angular_range: float = 360,
                   rotate: bool = True) -> Locations:
    step = angular_range / count if abs(angular_range - 360) < 1e-9 else angular_range / max(count - 1, 1)
    locs = []
    for i in range(count):
        a = start_angle + i * step
        r = math.radians(a)
        locs.append(Location((radius * math.cos(r), radius * math.sin(r), 0), (0, 0, a if rotate else 0)))
    return Locations(*locs)


class _Builder:
    def __enter__(self):
        # Like build123d, a builder nests under the active one only when both `with` blocks
        # are in the same Python function; a builder opened inside a helper is independent.
        # Shapes created in a helper still go to the active builder.
        self._python_frame = inspect.currentframe().f_back
        active = _innermost()
        same_scope = active is not None and getattr(active, "_python_frame", None) is self._python_frame
        self._parent = active if same_scope else None
        _builders.append(self)
        return self

    def __exit__(self, *exc):
        _builders.pop()
        if exc[0] is None:
            self._on_exit()

    def _on_exit(self):
        pass


class BuildPart(_Builder):
    def __init__(self, *workplanes, mode: Mode = Mode.ADD):
        self.workplanes = [Plane(w) if not isinstance(w, Plane) else w for w in workplanes] or [Plane.XY]
        self.mode = mode
        self._part: _KPart | None = None
        self.pending: list[_KSketch] = []
        self.pending_edges: list[Edge] = []

    @property
    def part(self) -> Part:
        return Part(self._part) if self._part is not None else Part([])

    def _combine(self, parts: list[_KPart], mode: Mode):
        if mode == Mode.PRIVATE or not parts:
            return
        parts = list(parts)
        if mode == Mode.REPLACE:
            self._part = parts[0].fuse(*parts[1:]) if len(parts) > 1 else parts[0]
        elif mode == Mode.ADD:
            base = self._part if self._part is not None else parts.pop(0)
            self._part = base.fuse(*parts) if parts else base
        elif mode == Mode.SUBTRACT:
            if self._part is not None:
                self._part = self._part.cut(*parts)
        elif mode == Mode.INTERSECT:
            if self._part is not None:
                self._part = self._part.intersect(*parts)

    def _placements(self) -> list[Location]:
        return [Location(wp) * loc for wp in self.workplanes for loc in _locations(self)]

    def _on_exit(self):
        if isinstance(self._parent, BuildPart) and self._part is not None:
            self._parent._combine([self._part], self.mode)


class BuildSketch(_Builder):
    def __init__(self, *workplanes, mode: Mode = Mode.ADD):
        self._explicit = [Plane(w) if not isinstance(w, Plane) else w for w in workplanes]
        self.mode = mode
        self._region: _KSketch | None = None
        self.pending_edges: list[Edge] = []

    def _combine(self, sketches: list[_KSketch], mode: Mode):
        if mode == Mode.PRIVATE or not sketches:
            return
        merged = sketches[0]
        for s in sketches[1:]:
            merged = merged + s
        if mode == Mode.REPLACE or self._region is None:
            self._region = merged if mode in (Mode.ADD, Mode.REPLACE) else self._region
        elif mode == Mode.ADD:
            self._region = self._region + merged
        elif mode == Mode.SUBTRACT:
            self._region = self._region - merged
        elif mode == Mode.INTERSECT:
            self._region = self._region & merged

    def _planes(self) -> list[Plane]:
        if self._explicit:
            return self._explicit
        if isinstance(self._parent, BuildPart):
            return [Plane(loc) for loc in self._parent._placements()]
        return [Plane.XY]

    @property
    def sketch_local(self) -> Sketch:
        return Sketch(self._region) if self._region is not None else Sketch()

    @property
    def sketch(self) -> Sketch:
        if self._region is None:
            return Sketch()
        return Sketch([self._region.on(p) for p in self._planes()])

    def _on_exit(self):
        if isinstance(self._parent, BuildPart) and self._region is not None:
            self._parent.pending.extend(self._region.on(p) for p in self._planes())


class BuildLine(_Builder):
    """Collects line objects (Line, RadiusArc, Spline, ...) into a path. Tuple points are
    coordinates on ``workplane``; Vectors are global. On exit the edges go to the enclosing
    BuildSketch (for ``make_face()``) or BuildPart (as a ``sweep()`` path)."""

    def __init__(self, workplane=Plane.XY, mode: Mode = Mode.ADD):
        self.workplane = workplane if isinstance(workplane, Plane) else Plane(workplane)
        self.mode = mode
        self._edges: list[Edge] = []

    def _combine(self, edges: list, mode: Mode):
        if mode == Mode.PRIVATE or not edges:
            return
        if mode == Mode.ADD:
            self._edges.extend(edges)
        elif mode == Mode.REPLACE:
            self._edges = list(edges)
        else:
            raise KeelError(f"BuildLine: mode={mode} is not supported yet")

    @property
    def line(self) -> "Wire | None":
        return Wire(self._edges) if self._edges else None

    def edges(self) -> "ShapeList":
        return ShapeList(self._edges)

    def edge(self) -> Edge:
        if len(self._edges) != 1:
            raise ValueError(f"BuildLine has {len(self._edges)} edges, not one")
        return self._edges[0]

    def wires(self) -> "ShapeList":
        return ShapeList(Wire(chain) for chain in _chains(self._edges))

    def wire(self) -> "Wire":
        wires = self.wires()
        if len(wires) != 1:
            raise ValueError(f"BuildLine has {len(wires)} wires, not one")
        return wires[0]

    def vertices(self) -> "ShapeList":
        out, seen = ShapeList(), []
        for e in self._edges:
            for p in (e.start_point(), e.end_point()):
                if all((p - q).length > 1e-9 for q in seen):
                    seen.append(p)
                    out.append(p)
        return out

    def _on_exit(self):
        if isinstance(self._parent, (BuildSketch, BuildPart)) and self.mode != Mode.PRIVATE:
            self._parent.pending_edges.extend(self._edges)


def _chains(edges: list) -> list[list]:
    """Group edges into connected chains (build123d's Wire.combine)."""
    rest, out = list(edges), []
    while rest:
        chain = [rest.pop(0)]
        grown = True
        while grown:
            grown = False
            for i, e in enumerate(rest):
                ends = (chain[0].start_point(), chain[-1].end_point())
                if any((p - q).length < 1e-6 for p in (e.start_point(), e.end_point()) for q in ends):
                    chain.append(rest.pop(i))
                    grown = True
                    break
        out.append(chain)
    return out


def _part_object(make, raw, rotation, align, mode: Mode) -> Part:
    """Build a primitive aligned per axis (None/Align.NONE = the shape's own placement,
    ``raw``), rotate it, and add it to an active BuildPart at every location."""
    modes = (align,) * 3 if align is None or isinstance(align, Align) else tuple(align)
    resolved = tuple(r if (m is None or m == Align.NONE) else (m.value if isinstance(m, Align) else m)
                     for m, r in zip(modes, raw))
    p = make(resolved)
    if rotation and any(rotation):
        p = _place(p, Location((0, 0, 0), tuple(rotation)))
    builder = _innermost()
    if isinstance(builder, BuildPart):
        builder._combine([_place(p, loc) for loc in builder._placements()], mode)
    return Part(p)


_C3 = (Align.CENTER, Align.CENTER, Align.CENTER)


def Box(length: float, width: float, height: float, rotation=(0, 0, 0), align=_C3, mode: Mode = Mode.ADD) -> Part:
    return _part_object(lambda a: _so.Box(length, width, height, align=a), ("min",) * 3, rotation, align, mode)


def Cylinder(radius: float, height: float, arc_size: float = 360, rotation=(0, 0, 0), align=_C3,
             mode: Mode = Mode.ADD) -> Part:
    if abs(arc_size - 360) > 1e-9:
        raise KeelError("Cylinder(arc_size=...) is not supported yet")
    return _part_object(lambda a: _so.Cylinder(radius, height, align=a), ("center", "center", "min"),
                        rotation, align, mode)


def Cone(bottom_radius: float, top_radius: float, height: float, arc_size: float = 360, rotation=(0, 0, 0),
         align=_C3, mode: Mode = Mode.ADD) -> Part:
    return _part_object(lambda a: _so.Cone(bottom_radius, top_radius, height, align=a),
                        ("center", "center", "min"), rotation, align, mode)


def Sphere(radius: float, arc_size1: float = -90, arc_size2: float = 90, arc_size3: float = 360,
           rotation=(0, 0, 0), align=_C3, mode: Mode = Mode.ADD) -> Part:
    return _part_object(lambda a: _so.Sphere(radius, align=a), ("center",) * 3, rotation, align, mode)


def Torus(major_radius: float, minor_radius: float, minor_start_angle: float = 0, minor_end_angle: float = 360,
          major_angle: float = 360, rotation=(0, 0, 0), align=_C3, mode: Mode = Mode.ADD) -> Part:
    return _part_object(lambda a: _so.Torus(major_radius, minor_radius, align=a), ("center",) * 3,
                        rotation, align, mode)


# ---------------------------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------------------------


def _sketches_of(obj) -> list[_KSketch]:
    if obj is None:
        return []
    if isinstance(obj, Sketch):
        return list(obj._items)
    if isinstance(obj, BuildSketch):
        return list(obj.sketch._items)
    if isinstance(obj, (list, tuple)):
        return [s for o in obj for s in _sketches_of(o)]
    raise TypeError(f"cannot extrude {type(obj).__name__}")


def extrude(to_extrude=None, amount: float | None = None, dir=None, until=None, target=None,  # noqa: A002
            both: bool = False, taper: float = 0.0, clean: bool = True, mode: Mode = Mode.ADD) -> Part:
    if until is not None:
        raise KeelError("extrude(until=...) is not supported yet")
    builder = _innermost()
    sketches = _sketches_of(to_extrude)
    if not sketches and isinstance(builder, BuildPart):
        sketches, builder.pending = builder.pending, []
    if not sketches:
        raise ValueError("extrude needs a sketch")
    if amount is None:
        raise ValueError("extrude needs an amount")
    if dir is not None:
        if taper:
            raise KeelError("extrude(dir=..., taper=...) is not supported yet")
        bodies = [_extrude_along(sk, tuple(Vector(dir)), amount, both) for sk in sketches]
    else:
        bodies = [s.extrude(amount, both=both, taper=taper) for s in sketches]
    if isinstance(builder, BuildPart):
        builder._combine(list(bodies), mode)
    return Part(bodies[0].fuse(*bodies[1:]) if len(bodies) > 1 else bodies[0])


def _extrude_along(sk: _KSketch, direction, amount: float, both: bool) -> _KPart:
    d = g.norm(direction)
    if amount < 0:
        d, amount = g.mul(d, -1), -amount
    sheet = sk._placed()
    if both:
        ub.translate(sheet, g.mul(d, -amount))
        amount *= 2
    return _KPart(call("extrude", ub.non_manifold_linear_sweep, sheet, d, amount, faces=sheet.faces()))


def add(objects, rotation=None, clean: bool = True, mode: Mode = Mode.ADD):
    builder = _innermost()
    if builder is None:
        raise RuntimeError("add must have an active builder context")
    items = _flatten(objects if isinstance(objects, (list, tuple)) else [objects])
    if isinstance(builder, BuildPart):
        parts = []
        for it in items:
            shape = it.part if isinstance(it, BuildPart) else it
            if rotation is not None:
                shape = shape.moved(Rot(*rotation)) if isinstance(rotation, tuple) else shape.moved(rotation)
            for loc in _locations(builder):
                parts.extend(shape.moved(loc)._parts)
        builder._combine(parts, mode)
        return
    if isinstance(builder, BuildSketch):
        builder._combine([s for it in items for s in _sketches_of(it)], mode)


def _edges_and_target(objects):
    edges = [e for e in _flatten(objects if isinstance(objects, (list, tuple, ShapeList)) else [objects])]
    if not edges:
        raise ValueError("no edges given")
    builder = _innermost()
    if isinstance(builder, BuildPart) and builder._part is not None:
        return edges, builder
    owner = edges[0]._owner
    if owner is None:
        raise ValueError("cannot tell which part these edges belong to")
    return edges, owner


def fillet(objects, radius: float) -> Part:
    edges, target = _edges_and_target(objects)
    if isinstance(target, BuildPart):
        target._part = target._part.fillet([e._e for e in edges], radius)
        return target.part
    return Part(target._one().fillet([e._e for e in edges], radius), target.label, target.color)


def chamfer(objects, length: float, length2: float | None = None, angle: float | None = None, reference=None) -> Part:
    edges, target = _edges_and_target(objects)
    if length2 is not None or angle is not None:
        raise KeelError("asymmetric chamfers are not supported yet")
    if isinstance(target, BuildPart):
        target._part = target._part.chamfer([e._e for e in edges], length)
        return target.part
    return Part(target._one().chamfer([e._e for e in edges], length), target.label, target.color)


def mirror(objects=None, about: Plane = Plane.XZ, mode: Mode = Mode.ADD):
    builder = _innermost()
    if objects is None and isinstance(builder, BuildPart):
        objects = builder.part
    shapes = _flatten(objects if isinstance(objects, (list, tuple)) else [objects])
    mirrored = [s.mirror(about) for s in shapes]
    if isinstance(builder, BuildPart):
        builder._combine([p for m in mirrored if isinstance(m, Shape) for p in m._parts], mode)
    return mirrored[0] if len(mirrored) == 1 else mirrored


# ---------------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------------


def _export_part(to_export) -> _KPart:
    if isinstance(to_export, BuildPart):
        to_export = to_export.part
    if not isinstance(to_export, Shape) or not to_export._parts:
        raise ValueError("nothing to export")
    rgb = tuple(to_export.color)[:3] if to_export.color is not None else None
    return to_export._one().named(to_export.label or None, rgb)


def export_step(to_export, file_path, unit=None, write_pcurves: bool = True, precision_mode=None) -> bool:
    """Exact B-rep as AP214 STEP, written by keel (one solid body per file)."""
    _export_part(to_export).export(str(file_path))
    return True


def export_stl(to_export, file_path, tolerance: float = 1e-3, angular_tolerance: float = 0.1,
               ascii_format: bool = False) -> bool:
    """A tessellated mesh. The kernel picks its own facet angle; ``tolerance`` is the chord error."""
    if ascii_format:
        raise KeelError("export_stl(ascii_format=True) is not supported yet")
    _export_part(to_export).export(str(file_path), tolerance=tolerance)
    return True


__all__ = [
    "Align", "Axis", "BoundBox", "Box", "BuildLine", "BuildPart", "BuildSketch", "Circle", "Color", "Compound", "Cone",
    "Cylinder", "Edge", "Ellipse", "Face", "GeomType", "GridLocations", "Location", "Locations", "Mode", "Part",
    "Plane", "PolarLocations", "Polygon", "Pos", "Rectangle", "RectangleRounded", "RegularPolygon", "Rot",
    "Rotation", "Shape", "ShapeList", "Sketch", "SlotCenterToCenter", "SlotOverall", "Solid", "SortBy", "Sphere",
    "Torus", "Vector", "Vertex", "add", "chamfer", "extrude", "fillet", "mirror",
    "Bezier", "CenterArc", "Line", "Polyline", "RadiusArc", "Spline", "ThreePointArc", "Wire",
    "loft", "make_face", "revolve", "sweep", "export_step", "export_stl",
]
