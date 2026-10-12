"""Faces, edges and vertices of a Part, and the Selection list used to pick them.

Selections can be narrowed with methods or with compact selector strings:

    ">Z"   highest along Z (all ties)         "<X"   lowest along X
    "|Z"   parallel to Z                      "#Z"   perpendicular to Z
    "+Z"   faces whose normal points +Z       "-X"   faces whose normal points -X
    "%plane" "%curved" "%line" "%circle" "%spline"   by geometry type
    "|Z and >X"   combine with "and"

Axes can be X/Y/Z or an explicit direction like "(1,1,0)".
"""

from __future__ import annotations

from typing import Callable, Iterable

from . import geom as g
from .geom import Plane

_ANGLE_TOL = 1e-4  # |dot| tolerance for parallel/perpendicular tests
_TIE_TOL = 1e-4  # distance tolerance when grouping ">Z" ties


class _Topo:
    """Base wrapper around a kernel topology handle owned by a Part's brep."""

    kind = "topo"

    def __init__(self, handle, owner, index: int):
        self._h = handle
        self._owner = owner  # the kernel Brep this belongs to
        self.index = index

    @property
    def center(self) -> tuple:
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) is type(other) and self._owner is other._owner and self.index == other.index

    def __hash__(self):
        return hash((type(self).__name__, id(self._owner), self.index))


class Vertex(_Topo):
    kind = "vertex"

    @property
    def point(self) -> tuple:
        return tuple(self._h.point())

    @property
    def center(self) -> tuple:
        return self.point

    def __repr__(self):
        return f"Vertex{g._fmt(self.point)}"


class Edge(_Topo):
    kind = "edge"

    @property
    def start(self) -> tuple:
        return tuple(self._h.start_vertex().point())

    @property
    def end(self) -> tuple:
        return tuple(self._h.end_vertex().point())

    @property
    def length(self) -> float:
        return self._h.length()

    @property
    def mid(self) -> tuple:
        a, b = self._h.parameter_range()
        return tuple(self._h.point_at(0.5 * (a + b)))

    @property
    def center(self) -> tuple:
        """Arc/circle center for circular edges, otherwise the midpoint."""
        c = self._h.curve()
        if hasattr(c, "center") and hasattr(c, "radius"):
            return tuple(c.center())
        return self.mid

    @property
    def geom_type(self) -> str:
        name = type(self._h.curve()).__name__
        return {"Line": "line", "Circle": "circle"}.get(name, "spline")

    @property
    def radius(self) -> float | None:
        c = self._h.curve()
        return c.radius() if hasattr(c, "radius") else None

    @property
    def direction(self) -> tuple | None:
        """Unit direction for straight edges, else None."""
        if self.geom_type != "line":
            return None
        return g.norm(g.sub(self.end, self.start))

    @property
    def is_seam(self) -> bool:
        """True where a periodic surface (cylinder, sphere, ...) meets itself."""
        return len(self._h.faces()) == 1

    @property
    def is_closed(self) -> bool:
        return g.dist(self.start, self.end) < 1e-9

    def tangent(self, t: float = 0.5) -> tuple:
        """Unit tangent at fraction ``t`` (0..1) along the edge."""
        a, b = self._h.parameter_range()
        return g.norm(tuple(self._h.tangent_at(a + t * (b - a))))

    def point(self, t: float = 0.5) -> tuple:
        a, b = self._h.parameter_range()
        return tuple(self._h.point_at(a + t * (b - a)))

    def __repr__(self):
        return f"Edge({self.geom_type} {g._fmt(self.start)} -> {g._fmt(self.end)})"


class Face(_Topo):
    kind = "face"

    def _mid_uv(self):
        (u0, v0), (u1, v1) = self._h.uv_domain()
        return 0.5 * (u0 + u1), 0.5 * (v0 + v1)

    @property
    def center(self) -> tuple:
        return tuple(self._h.centroid())

    @property
    def area(self) -> float:
        return self._h.area()

    @property
    def normal(self) -> tuple:
        """Outward normal (at the middle of the face's parameter range)."""
        return g.norm(tuple(self._h.outward_normal_at_uv(*self._mid_uv())))

    def normal_at(self, point) -> tuple:
        """Outward normal at the face point closest to ``point``."""
        res = self._h.closest_point(g.vec(point))
        uv = _find_uv(res)
        if uv is None:
            return self.normal
        return g.norm(tuple(self._h.outward_normal_at_uv(*uv)))

    @property
    def is_planar(self) -> bool:
        return bool(self._h.surface().is_planar(1e-6))

    @property
    def geom_type(self) -> str:
        return "plane" if self.is_planar else "curved"

    def edges(self) -> "Selection":
        owner_edges = self._owner.edges()
        index = {_edge_key(e): i for i, e in enumerate(owner_edges)}
        out = []
        for e in self._h.edges():
            if len(e.faces()) == 1:  # seam: see Part.edges
                continue
            i = index.get(_edge_key(e))
            out.append(Edge(e, self._owner, i if i is not None else -1))
        return Selection(out)

    def plane(self, x_dir=None) -> Plane:
        """A sketch plane on this face: origin at the face center, z along the outward normal."""
        if not self.is_planar:
            raise ValueError("can only sketch on a planar face")
        return Plane.at(self.center, self.normal, x_dir)

    def __repr__(self):
        return f"Face({self.geom_type} center={g._fmt(self.center)} normal={g._fmt(self.normal)})"


def _edge_key(e):
    a, b = e.parameter_range()
    return tuple(round(c, 7) for c in e.point_at(0.5 * (a + b))) + (round(e.length(), 7),)


def _find_uv(res):
    # closest_point returns a tuple whose shape varies; find a (u, v) pair in it.
    if isinstance(res, tuple):
        for item in res:
            if isinstance(item, tuple) and len(item) == 2 and all(isinstance(c, float) for c in item):
                return item
    return None


# -- selection -------------------------------------------------------------------------------


class Selection(list):
    """A list of faces/edges/vertices with filtering, sorting and selector strings.

    Calling a selection with a selector string narrows it: ``part.faces()(">Z")``.
    """

    def __call__(self, selector: str) -> "Selection":
        return select(self, selector)

    # -- narrowing ---------------------------------------------------------------------------
    def filter(self, pred: Callable) -> "Selection":
        return Selection(x for x in self if pred(x))

    def sort(self, key="Z", reverse: bool = False) -> "Selection":  # type: ignore[override]
        """Sort by an axis ("X", "-Z", (1,1,0)) or a key function."""
        if callable(key):
            return Selection(sorted(self, key=key, reverse=reverse))
        d = g.direction(key)
        return Selection(sorted(self, key=lambda o: g.dot(o.center, d), reverse=reverse))

    def max(self, axis="Z") -> "Selection":
        """All items whose center is furthest along ``axis`` (ties kept)."""
        return _extreme(self, g.direction(axis))

    def min(self, axis="Z") -> "Selection":
        return _extreme(self, g.mul(g.direction(axis), -1))

    def parallel(self, axis="Z") -> "Selection":
        """Straight edges parallel to ``axis``; planar faces whose normal is parallel to it."""
        d = g.direction(axis)
        return self.filter(lambda o: (v := _orient(o)) is not None and abs(abs(g.dot(v, d)) - 1) < _ANGLE_TOL)

    def perpendicular(self, axis="Z") -> "Selection":
        d = g.direction(axis)
        return self.filter(lambda o: (v := _orient(o)) is not None and abs(g.dot(v, d)) < _ANGLE_TOL)

    def facing(self, direction="Z") -> "Selection":
        """Faces whose outward normal points along ``direction``."""
        d = g.direction(direction)
        return self.filter(lambda f: isinstance(f, Face) and f.is_planar and g.dot(f.normal, d) > 1 - _ANGLE_TOL)

    def of_type(self, kind: str) -> "Selection":
        return self.filter(lambda o: getattr(o, "geom_type", None) == kind)

    def near(self, point, count: int = 1) -> "Selection":
        """The ``count`` items whose centers are closest to ``point``."""
        p = g.vec(point)
        return Selection(sorted(self, key=lambda o: g.dist(o.center, p))[:count])

    def largest(self) -> "Selection":
        return Selection([max(self, key=_size)]) if self else Selection()

    def smallest(self) -> "Selection":
        return Selection([min(self, key=_size)]) if self else Selection()

    # -- conversion --------------------------------------------------------------------------
    def edges(self) -> "Selection":
        """Edges of the selected faces (deduplicated)."""
        seen, out = set(), []
        for f in self:
            for e in f.edges() if isinstance(f, Face) else [f]:
                if e not in seen:
                    seen.add(e)
                    out.append(e)
        return Selection(out)

    @property
    def first(self):
        return self[0]

    @property
    def last(self):
        return self[-1]

    def one(self):
        """The single selected item; raises if the selection is not exactly one item."""
        if len(self) != 1:
            raise ValueError(f"expected exactly one item, selection has {len(self)}")
        return self[0]

    def plane(self, x_dir=None) -> Plane:
        """Sketch plane of the single selected face, so ``Rect(4, 4).on(part.faces(">Z"))`` works."""
        return self.one().plane(x_dir)

    def __getitem__(self, i):
        r = super().__getitem__(i)
        return Selection(r) if isinstance(i, slice) else r

    def __add__(self, other):
        return Selection(list(self) + [x for x in other if x not in self])

    def __sub__(self, other):
        return Selection(x for x in self if x not in other)

    def __repr__(self):
        return f"Selection({list.__repr__(self)})"


def _orient(o):
    if isinstance(o, Edge):
        return o.direction
    if isinstance(o, Face) and o.is_planar:
        return o.normal
    return None


def _size(o):
    if isinstance(o, Face):
        return o.area
    if isinstance(o, Edge):
        return o.length
    return 0.0


def _extreme(items: Iterable, d) -> Selection:
    items = list(items)
    if not items:
        return Selection()
    vals = [g.dot(o.center, d) for o in items]
    top = max(vals)
    return Selection(o for o, v in zip(items, vals) if top - v <= _TIE_TOL)


def select(items, selector: str) -> Selection:
    """Apply a selector string (see module docstring)."""
    sel = Selection(items)
    for term in (t.strip() for t in selector.split(" and ")):
        if not term:
            continue
        op, arg = term[0], term[1:].strip()
        if op == ">":
            sel = sel.max(arg)
        elif op == "<":
            sel = sel.min(arg)
        elif op == "|":
            sel = sel.parallel(arg)
        elif op == "#":
            sel = sel.perpendicular(arg)
        elif op in "+-":
            sel = sel.facing(op + arg)
        elif op == "%":
            sel = sel.of_type(arg.lower())
        else:
            raise ValueError(f"unknown selector {term!r} (use >Z <Z |Z #Z +Z -Z %type)")
    return sel


__all__ = ["Vertex", "Edge", "Face", "Selection", "select"]
