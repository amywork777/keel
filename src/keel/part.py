"""Part: an immutable 3D solid.

Every operation returns a new Part; the kernel's in-place and input-consuming calls are
always given private copies, so a Part you hold never changes underneath you.
"""

from __future__ import annotations

from typing import Iterable

from . import geom as g
from ._kernel import KeelError, call, ub
from .geom import Plane
from .topo import Edge, Face, Selection, Vertex, select


class BBox(tuple):
    """(min, max) corners. The kernel pads boxes by its tolerance (~2e-5)."""

    @property
    def min(self):
        return self[0]

    @property
    def max(self):
        return self[1]

    @property
    def size(self):
        return g.sub(self[1], self[0])

    @property
    def center(self):
        return g.mul(g.add(self[0], self[1]), 0.5)


class Part:
    """A 3D body. Combine with ``+`` (fuse), ``-`` (cut) and ``&`` (intersect)."""

    def __init__(self, brep, label: str | None = None, color=None):
        self._brep = brep
        self.label = label
        self.color = color

    # -- identity / copies --------------------------------------------------------------------
    @property
    def brep(self):
        """The underlying kernel Brep. Treat as read-only."""
        return self._brep

    def _copy(self):
        return self._brep.copy()

    def _new(self, brep) -> "Part":
        return Part(brep, self.label, self.color)

    def named(self, label: str, color=None) -> "Part":
        """Same geometry with a label (USD prim name) and optional RGB color (0..1)."""
        return Part(self._brep, label, color if color is not None else self.color)

    # -- measures -----------------------------------------------------------------------------
    @property
    def volume(self) -> float:
        return self._brep.volume()

    @property
    def area(self) -> float:
        return self._brep.area()

    @property
    def center(self) -> tuple:
        """Center of mass."""
        return tuple(self._brep.mass_properties()["centroid"])

    @property
    def bbox(self) -> BBox:
        mn, mx = self._brep.bounding_box()
        return BBox((tuple(mn), tuple(mx)))

    @property
    def is_valid(self) -> bool:
        """True for a closed manifold solid."""
        return bool(self._brep.is_manifold_solid())

    def info(self) -> dict:
        return self._brep.info()

    # -- topology -----------------------------------------------------------------------------
    def faces(self, selector: str | None = None) -> Selection:
        s = Selection(Face(f, self._brep, i) for i, f in enumerate(self._brep.faces()))
        return select(s, selector) if selector else s

    def edges(self, selector: str | None = None) -> Selection:
        s = Selection(Edge(e, self._brep, i) for i, e in enumerate(self._brep.edges()))
        return select(s, selector) if selector else s

    def vertices(self, selector: str | None = None) -> Selection:
        s = Selection(Vertex(v, self._brep, i) for i, v in enumerate(self._brep.vertices()))
        return select(s, selector) if selector else s

    # -- booleans -----------------------------------------------------------------------------
    def fuse(self, *others) -> "Part":
        return self._boolean(ub.boolean_union, "fuse", others)

    def cut(self, *others) -> "Part":
        return self._boolean(ub.boolean_difference, "cut", others)

    def intersect(self, *others) -> "Part":
        return self._boolean(ub.boolean_intersection, "intersect", others)

    def _boolean(self, fn, name, others) -> "Part":
        result = self._copy()
        for other in _flatten(others):
            if not isinstance(other, Part):
                raise TypeError(f"cannot {name} a Part with {type(other).__name__}")
            result = call(name, fn, result, other._copy())
        return self._new(result)

    __add__ = fuse
    __sub__ = cut
    __and__ = intersect

    def __radd__(self, other):  # lets sum([...]) work
        if other == 0:
            return self
        return NotImplemented

    # -- placement ----------------------------------------------------------------------------
    def move(self, x=0.0, y=0.0, z=0.0) -> "Part":
        """Translate. Accepts ``move(1, 2, 3)`` or ``move((1, 2, 3))``."""
        d = g.vec(x) if isinstance(x, (tuple, list)) else (float(x), float(y), float(z))
        return self._new(call("move", ub.translate, self._copy(), d))

    def at(self, *locations):
        """Copies moved to each location: ``at((x, y, z))`` -> Part, ``at(grid(...))`` -> list."""
        if len(locations) == 1 and isinstance(locations[0], (tuple, list)) and locations[0] and isinstance(
            locations[0][0], (int, float)
        ):
            return self.move(locations[0])
        locs = locations[0] if len(locations) == 1 else locations
        return [self.move(p) for p in locs]

    def rotate(self, axis="Z", degrees: float = 90.0, about=(0, 0, 0)) -> "Part":
        """Rotate about an axis direction through ``about`` (right-hand rule)."""
        if isinstance(axis, g.Axis):
            about, axis = axis.origin, axis.dir
        return self._new(call("rotate", ub.rotate, self._copy(), g.vec(about), g.direction(axis), float(degrees)))

    def scale(self, factor, about=(0, 0, 0)) -> "Part":
        f = (factor,) * 3 if isinstance(factor, (int, float)) else g.vec(factor)
        return self._new(call("scale", ub.scale_about_point, self._copy(), g.vec(about), f))

    def mirror(self, plane="YZ") -> "Part":
        """Mirror across a plane through its origin ("YZ" mirrors X, etc.)."""
        p = g.plane_of(plane)
        n = p.z
        b = self._copy()
        # reflect = move plane to origin, scale -1 along its normal, move back
        if g.length(p.origin) > 0:
            ub.translate(b, g.mul(p.origin, -1))
        axis_aligned = [i for i in range(3) if abs(abs(n[i]) - 1) < 1e-9]
        if not axis_aligned:
            raise KeelError("mirror currently supports axis-aligned planes only")
        f = [1.0, 1.0, 1.0]
        f[axis_aligned[0]] = -1.0
        call("mirror", ub.scale, b, tuple(f))
        if g.length(p.origin) > 0:
            ub.translate(b, p.origin)
        return self._new(b)

    def place(self, plane) -> "Part":
        """Move a part modeled around world XY onto ``plane`` (a Plane, face or name)."""
        return self._new(g.plane_of(plane).place(self._copy()))

    # -- features -----------------------------------------------------------------------------
    def fillet(self, edges, radius: float) -> "Part":
        """Round edges. ``edges``: a selector string, a Selection, an Edge, or a list of edges."""
        b, es = self._edges_on_copy(edges)
        return self._new(call("fillet", ub.fillet_edges, b, es, radius=float(radius), xsect_type=1))

    def chamfer(self, edges, distance: float) -> "Part":
        """Bevel edges, cutting each adjacent face back by ``distance`` (at 90° corners)."""
        b, es = self._edges_on_copy(edges)
        return self._new(call("chamfer", ub.fillet_edges, b, es, radius=float(distance), xsect_type=0))

    def shell(self, thickness: float, open=None) -> "Part":
        """Hollow the part with walls of ``thickness`` (inward), removing the ``open`` faces."""
        faces = []
        if open is not None:
            faces = [f._h for f in self._resolve(open, "faces")]
        return self._new(
            call("shell", ub.shell_brep, self._brep, -abs(float(thickness)), faces_to_shell=faces)
        )

    def offset(self, distance: float) -> "Part":
        """Grow (positive) or shrink (negative) every face."""
        return self._new(call("offset", ub.offset_brep, self._brep, float(distance)))

    # -- output -------------------------------------------------------------------------------
    def tessellate(self, tolerance: float = 0.0, angle: float = 15.0):
        from .io import tessellate

        return tessellate(self, tolerance=tolerance, angle=angle)

    def export(self, path: str, **kwargs) -> str:
        """Write by extension: .usda/.usdc/.usd (exact BRep), .stl (mesh)."""
        from .io import export

        return export(self, path, **kwargs)

    def __repr__(self):
        i = self._brep.info()
        name = f" {self.label!r}" if self.label else ""
        return f"<Part{name} faces={i['faces']} edges={i['edges']} volume={i['volume']:.6g}>"

    # -- helpers ------------------------------------------------------------------------------
    def _resolve(self, items, kind: str) -> list:
        if isinstance(items, str):
            return list(getattr(self, kind)(items))
        if isinstance(items, (Edge, Face, Vertex)):
            return [items]
        return list(items)

    def _edges_on_copy(self, edges):
        """Copy the brep and find the selected edges on the copy."""
        chosen = self._resolve(edges, "edges")
        if not chosen:
            raise KeelError("no edges selected")
        b = self._copy()
        copy_edges = b.edges()
        out = []
        for e in chosen:
            if not isinstance(e, Edge):
                raise TypeError(f"expected edges, got {type(e).__name__}")
            match = None
            if e._owner is self._brep and 0 <= e.index < len(copy_edges):
                cand = copy_edges[e.index]
                if _same_edge(cand, e._h):
                    match = cand
            if match is None:  # edge from another part or reordered: match by geometry
                match = next((c for c in copy_edges if _same_edge(c, e._h)), None)
            if match is None:
                raise KeelError(f"{e!r} is not an edge of this part")
            out.append(match)
        return b, out


def _same_edge(a, b, tol: float = 1e-6) -> bool:
    if abs(a.length() - b.length()) > tol * max(1.0, a.length()):
        return False
    pa, pb = a.parameter_range(), b.parameter_range()
    ma = a.point_at(0.5 * (pa[0] + pa[1]))
    mb = b.point_at(0.5 * (pb[0] + pb[1]))
    return g.dist(ma, mb) < 1e-5


def _flatten(items) -> list:
    out = []
    for it in items:
        if isinstance(it, (list, tuple)):
            out.extend(_flatten(it))
        else:
            out.append(it)
    return out


def fuse(parts: Iterable[Part]) -> Part:
    """Union of many parts."""
    parts = _flatten(list(parts))
    if not parts:
        raise ValueError("fuse() needs at least one part")
    return parts[0].fuse(*parts[1:])

