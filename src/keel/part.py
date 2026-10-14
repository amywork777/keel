"""Part: an immutable 3D solid.

Every operation returns a new Part; the kernel's in-place and input-consuming calls are
always given private copies, so a Part you hold never changes underneath you.
"""

from __future__ import annotations

import math
import re
import warnings

from typing import Iterable

from . import geom as g
from ._kernel import KeelError, call, ub
from .geom import Plane
from .topo import Edge, Face, Selection, Vertex, select


class BBox(tuple):
    """(min, max) corners."""

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


def _occt_shell_walk(face_edges: list[list[int]]) -> list[int]:
    """The face order OpenCASCADE (7.9, ShapeFix_Shell) gives a shell read from a file.

    ``face_edges[i]`` lists face i's edges in loop order. From the first face not yet reached,
    a stack walk: pop a face, and for each of its edges push the unvisited faces on that
    edge in file order. Separate shells come out one after another.
    """
    edge_faces: dict[int, list[int]] = {}
    for i, es in enumerate(face_edges):
        for e in es:
            faces = edge_faces.setdefault(e, [])
            if i not in faces:
                faces.append(i)
    order, seen = [], set()
    for start in range(len(face_edges)):
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        while stack:
            f = stack.pop()
            order.append(f)
            for e in face_edges[f]:
                for n in edge_faces[e]:
                    if n not in seen:
                        seen.add(n)
                        stack.append(n)
    return order


class Part:
    """A 3D body. Combine with ``+`` (fuse), ``-`` (cut) and ``&`` (intersect)."""

    def __init__(self, brep, label: str | None = None, color=None):
        self._brep = brep
        self.label = label
        self.color = color
        self._volumes: dict = {}  # accuracy -> volume; shared with rigidly moved copies

    # -- identity / copies --------------------------------------------------------------------
    @property
    def brep(self):
        """The underlying kernel Brep. Treat as read-only."""
        return self._brep

    def _copy(self):
        return self._brep.copy()

    def _new(self, brep) -> "Part":
        return Part(brep, self.label, self.color)

    def _moved(self, brep) -> "Part":
        """A new Part for a rigid move of this one: same volume, so the volume cache is shared."""
        out = Part(brep, self.label, self.color)
        out._volumes = self._volumes
        return out

    def named(self, label: str, color=None) -> "Part":
        """Same geometry with a label (USD prim name) and optional RGB color (0..1)."""
        out = Part(self._brep, label, color if color is not None else self.color)
        out._volumes = self._volumes
        return out

    # -- measures -----------------------------------------------------------------------------
    @property
    def volume(self) -> float:
        """Enclosed volume (relative accuracy ~1e-6; the kernel's quick estimate can be 1e-3 off)."""
        return self._volume_to(1e-6)

    def _quick_volume(self) -> float:
        """Volume to ~1e-4 relative, for plausibility checks."""
        return self._volume_to(1e-4)

    def _volume_to(self, accuracy: float) -> float:
        # cached: a Part never changes, and a rigid move keeps the volume
        if accuracy not in self._volumes:
            self._volumes[accuracy] = _volume(self._brep, accuracy)
        return self._volumes[accuracy]

    @property
    def area(self) -> float:
        """Surface area (relative accuracy ~1e-4, the kernel's finest)."""
        return self._brep.area(1e-4)

    @property
    def center(self) -> tuple:
        """Center of mass."""
        return tuple(self._brep.mass_properties()["centroid"])

    @property
    def bbox(self) -> BBox:
        """Tight bounding box of the trimmed geometry."""
        if getattr(self, "_bbox_cache", None) is None:
            mn, mx = self._brep.bounding_box(True)
            self._bbox_cache = BBox((tuple(mn), tuple(mx)))
        return self._bbox_cache

    @property
    def is_valid(self) -> bool:
        """True for a closed manifold solid."""
        return bool(self._brep.is_manifold_solid())

    def info(self) -> dict:
        return self._brep.info()

    # -- topology -----------------------------------------------------------------------------
    def ref(self, entity) -> str:
        """The name a STEP reader gives ``entity`` (a face or edge of this part), the way
        text-to-cad writes it: ``"f12"``, ``"e7"``, or ``"label.f12"`` for a labeled part."""
        from .step import step_names

        faces, edges = step_names(self)
        if isinstance(entity, Face) and entity._owner is self._brep:
            name = f"f{faces.index(entity.index) + 1}"
        elif isinstance(entity, Edge) and entity._owner is self._brep:
            name = f"e{edges.index(entity.index) + 1}"
        else:
            raise KeelError(f"{entity!r} is not a face or edge of this part")
        return f"{self.label}.{name}" if self.label else name

    def find(self, ref: str):
        """The face or edge a STEP name refers to: ``"f12"``, ``"e7"``, ``"label.e7"`` or
        text-to-cad's ``"#label.e7"``. Inverse of ``ref``."""
        from .step import step_names

        m = re.fullmatch(r"#?(?:(.+)\.)?([fe])(\d+)", ref.strip())
        if not m or (m.group(1) and m.group(1) != self.label):
            raise KeelError(f"{ref!r} doesn't name a face or edge of this part")
        faces, edges = step_names(self)
        kind, n = m.group(2), int(m.group(3))
        order = faces if kind == "f" else edges
        if not 1 <= n <= len(order):
            raise KeelError(f"{ref!r}: the part has {len(order)} {'faces' if kind == 'f' else 'edges'}")
        i = order[n - 1]
        if i is None:
            raise KeelError(f"{ref!r} is a zero-length edge STEP readers add at a sphere pole; keel has no edge there")
        if kind == "f":
            return Face(self._brep.faces()[i], self._brep, i)
        return Edge(self._brep.edges()[i], self._brep, i)

    def faces(self, selector: str | None = None) -> Selection:
        faces = self._brep.faces()
        s = Selection(Face(faces[i], self._brep, i) for i in self._face_order())
        return select(s, selector) if selector else s

    def _face_order(self) -> list[int]:
        """Kernel face indices in the order STEP files (and so text-to-cad) number them.

        OpenCASCADE's STEP reader rebuilds every shell with a depth-first walk over shared
        edges, so a STEP file's faces come back in walk order, not file order. keel writes
        faces in an order the walk leaves unchanged (repeating the walk until it settles),
        so ``part.faces()[i]`` is face ``i + 1`` in the file and in OCCT (``label.f12``).
        """
        if getattr(self, "_face_order_cache", None) is None:
            index = {e: i for i, e in enumerate(self._brep.edges())}
            edges = [[index[e] for loop in f.boundary_loops() for e, _ in loop] for f in self._brep.faces()]
            order = list(range(len(edges)))
            for _ in range(10):
                walked = [order[i] for i in _occt_shell_walk([edges[i] for i in order])]
                if walked == order:
                    break
                order = walked
            # file order, and the order a reader gets back (the same once the walk settles)
            self._face_order_cache = (order, [order[i] for i in _occt_shell_walk([edges[i] for i in order])])
        return self._face_order_cache[1]

    def _step_order(self) -> list[int]:
        """Kernel face indices in the order to write them to a STEP file."""
        self._face_order()
        return self._face_order_cache[0]

    def edges(self, selector: str | None = None, *, seams: bool = False) -> Selection:
        """Edges, optionally narrowed by a selector.

        Seam edges (where a cylinder, cone, sphere or torus wraps onto itself) are left out
        unless ``seams=True``: they aren't real edges, and fillets or chamfers on them fail.
        """
        edges = self._brep.edges()
        s = Selection(Edge(edges[i], self._brep, i) for i in self._edge_order() if seams or len(edges[i].faces()) != 1)
        return select(s, selector) if selector else s

    def _edge_order(self) -> list[int]:
        """Kernel edge indices in the order the faces (in ``faces()`` order) reach them, loop by loop.

        Close to how STEP readers number edges, but not the same: they also reverse some loops
        and add zero-length edges at poles. Use ``ref``/``find`` for STEP names.
        """
        if getattr(self, "_edge_order_cache", None) is None:
            index = {e: i for i, e in enumerate(self._brep.edges())}
            faces = self._brep.faces()
            order = dict.fromkeys(index[e] for i in self._face_order() for loop in faces[i].boundary_loops() for e, _ in loop)
            self._edge_order_cache = list(order) + [i for i in index.values() if i not in order]
        return self._edge_order_cache

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
        others = _flatten(others)
        for other in others:
            if not isinstance(other, Part):
                raise TypeError(f"cannot {name} a Part with {type(other).__name__}")
        if len(others) > 1 and name in _BATCH_OPS:
            batched = self._batched(name, others)
            if batched is not None:
                return batched
        result = self
        for other in others:
            before = result
            result = self._new(call(name, fn, before._copy(), other._copy()))
            _check_boolean(name, before, [other], result)
        return result if result is not self else self._new(self._copy())

    def _batched(self, name, others) -> "Part | None":
        """All of ``others`` in one kernel call, like build123d (and OpenCascade) combine every
        object made in one Locations block. The kernel then sorts out the tools in one pass
        instead of one boolean per tool, each on a body that grows with every hole.

        A cut the kernel can't do in one pass (it fails on some overlapping tools, also one at
        a time) is tried once more with the tools fused first. Returns None, and the caller
        falls back to one boolean per tool, each checked, when the kernel fails or splits the
        result into several bodies, or the result fails the volume check."""
        result = self._one_call(name, others)
        if result is None and name == "cut":
            try:
                tool = others[0].fuse(*others[1:])
                result = self._new(call(name, ub.boolean_difference, self._copy(), tool._copy()))
            except KeelError:
                return None
            others = [tool]
        if result is None:
            return None
        try:
            _check_boolean(name, self, others, result)
        except KeelError:
            return None
        return result

    def _one_call(self, name, others) -> "Part | None":
        try:
            breps, _ = ub.boolean_lists([self._copy()], [], [o._copy() for o in others], [],
                                        operation=_BATCH_OPS[name])
        except (RuntimeError, ValueError, TypeError):
            return None
        return self._new(breps[0]) if len(breps) == 1 else None

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
        return self._moved(call("move", ub.translate, self._copy(), d))

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
        return self._moved(call("rotate", ub.rotate, self._copy(), g.vec(about), g.direction(axis), float(degrees)))

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
        return self._moved(b)

    def place(self, plane) -> "Part":
        """Move a part modeled around world XY onto ``plane`` (a Plane, face or name)."""
        return self._moved(g.plane_of(plane).place(self._copy()))

    # -- features -----------------------------------------------------------------------------
    def fillet(self, edges, radius: float) -> "Part":
        """Round edges. ``edges``: a selector string, a Selection, an Edge, or a list of edges."""
        _require_manifold("fillet", self)
        b, es = self._edges_on_copy(edges)
        reach = _edge_reach(es, radius)
        result = call("fillet", ub.fillet_edges, b, es, radius=float(radius), xsect_type=1)
        return _check_feature("fillet", self, result, reach)

    def chamfer(self, edges, distance: float) -> "Part":
        """Bevel edges, cutting each adjacent face back by ``distance``.

        The kernel's flat cross-section is sized like a fillet (it cuts between the lines a
        round of that radius would touch), so each edge's size is converted from the
        setback using the angle between its faces: size = distance * tan(wedge / 2).
        """
        _require_manifold("chamfer", self)
        b, es = self._edges_on_copy(edges)
        sizes = [float(distance) * math.tan(_wedge_angle(e) / 2) for e in es]
        reach = _edge_reach(es, distance)
        try:
            result = call("chamfer", ub.fillet_edges_per_edge, b, es, sizes, [ub.FilletXSect.LINEAR] * len(es))
            return _check_feature("chamfer", self, result, reach)
        except KeelError:
            if all(abs(sz - distance) < 1e-9 * max(1.0, distance) for sz in sizes):
                raise
        # Very shallow edges (faces meeting at nearly 180 degrees) need a huge kernel size for
        # the requested setback, which the kernel can't build. Fall back to one size, exact
        # on 90-degree edges, and say which edges got a different setback.
        off = [e for e, sz in zip(es, sizes) if abs(sz - distance) > 1e-6 * max(1.0, distance)]
        warnings.warn(
            f"chamfer: the kernel couldn't build an exact {distance:g} setback on {len(off)} edge(s) "
            "that meet at shallow angles; those got a smaller setback",
            stacklevel=2,
        )
        b, es = self._edges_on_copy(edges)
        result = call("chamfer", ub.fillet_edges, b, es, radius=float(distance), xsect_type=0)
        return _check_feature("chamfer", self, result, reach)

    def shell(self, thickness: float, open=None, *, outward: bool = False, rounded: bool = False) -> "Part":
        """Hollow the part with walls of ``thickness`` (inward, or ``outward``), removing the
        ``open`` faces.

        ``rounded=True`` rounds the corners that outward walls open at convex edges, like
        OpenCascade's arc join (build123d's ``Kind.ARC``); by default the walls are extended
        until they meet."""
        faces = [f._h for f in self._own_faces(open)] if open is not None else []
        distance = abs(float(thickness)) * (1 if outward else -1)
        result = call("shell", ub.shell_brep, self._brep, distance, extended_offset=not rounded, faces_to_shell=faces)
        return self._new(_check_solid("shell", result))

    def offset(self, distance: float, *, rounded: bool = False) -> "Part":
        """Grow (positive) or shrink (negative) every face.

        ``rounded=True`` rounds the gaps a growing offset opens at convex edges (and a
        shrinking one at concave edges), like OpenCascade's arc join (build123d's
        ``Kind.ARC``); by default the offset faces are extended until they meet."""
        result = call("offset", ub.offset_brep, self._brep, float(distance), extended_offset=not rounded)
        return self._new(_check_solid("offset", result))

    def _own_faces(self, items) -> list:
        """Faces of this part's brep for ``items`` (a selector, Face or faces), matching faces
        that came from another copy of the part by their center and normal."""
        out = []
        faces = None
        for f in self._resolve(items, "faces"):
            if not isinstance(f, Face):
                raise TypeError(f"expected faces, got {type(f).__name__}")
            if f._owner is self._brep:
                out.append(f)
                continue
            if faces is None:
                faces = list(self.faces())
            match = next((c for c in faces if g.dist(c.center, f.center) < 1e-6 and g.dot(c.normal, f.normal) > 1 - 1e-9
                          and abs(c.area - f.area) < 1e-6 * max(1.0, f.area)), None)
            if match is None:
                raise KeelError(f"{f!r} is not a face of this part")
            out.append(match)
        return out

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
        return b, [e for e in _tangent_chain(copy_edges, out) if not _is_smooth(e)]


def _edge_reach(edges, size: float) -> float:
    """Upper bound on the volume a fillet/chamfer of ``size`` can remove or add along edges."""
    return sum(e.length() for e in edges) * float(size) ** 2


def _require_manifold(name: str, part: Part) -> None:
    """Fillets and chamfers need a closed manifold body: on one that isn't, the kernel fails,
    returns a broken body or crashes, and the size is not what's wrong."""
    if not part.is_valid:
        raise KeelError(
            f"{name} failed: the body was already non-manifold before the {name} (likely parts "
            "touching along an edge or a line, or at a point); overlap them or keep them apart, "
            f"then {name}"
        )


def _check_feature(name: str, before: Part, result, reach: float):
    """Reject an invalid or implausible fillet/chamfer result instead of returning it.

    The kernel sometimes "succeeds" with an empty or broken body where OpenCascade raises
    (e.g. a radius too large for the faces at the edge)."""
    v0 = before._quick_volume()
    v1 = _volume(result, 1e-4) if result.face_count() else 0.0
    if not result.is_manifold_solid() or v1 <= 0 or abs(v1 - v0) > reach * 1.01 + 1e-9 * v0:
        raise KeelError(
            f"{name} failed: the kernel returned an invalid result (volume {v0:.6g} -> {v1:.6g}); "
            "try a smaller size or fewer edges"
        )
    out = before._new(result)
    out._volumes[1e-4] = v1
    return out


def _check_solid(name: str, result):
    if not result.face_count() or not result.is_manifold_solid():
        raise KeelError(f"{name} failed: the kernel returned an invalid body; try a smaller distance")
    return result


def _wedge_angle(e) -> float:
    """Angle (radians) of the open wedge at an edge where a round or chamfer sits:
    pi minus the angle between the two faces' outward normals (pi/2 at a box edge)."""
    from .topo import face_normal_at

    faces = e.faces()
    if len(faces) != 2:
        return math.pi / 2
    t0, t1 = e.parameter_range()
    mid = tuple(e.point_at(0.5 * (t0 + t1)))
    c = max(-1.0, min(1.0, g.dot(face_normal_at(faces[0], mid), face_normal_at(faces[1], mid))))
    return math.pi - math.acos(c)


def _is_smooth(e, tol: float = 1e-6) -> bool:
    """True when the two faces at an edge continue each other (same normal there), so the
    edge is not a corner: e.g. where a slot's flat wall turns into its round end."""
    from .topo import face_normal_at

    faces = e.faces()
    if len(faces) != 2:
        return False
    t0, t1 = e.parameter_range()
    mid = tuple(e.point_at(0.5 * (t0 + t1)))
    return g.dot(face_normal_at(faces[0], mid), face_normal_at(faces[1], mid)) > 1 - tol


def _edge_ends(e):
    """((start point, unit tangent leaving it), (end point, unit tangent leaving it))."""
    t0, t1 = e.parameter_range()
    a, b = tuple(e.point_at(t0)), tuple(e.point_at(t1))
    da = g.norm(tuple(e.tangent_at(t0)))
    db = g.mul(g.norm(tuple(e.tangent_at(t1))), -1)
    return (a, da), (b, db)


def _tangent_chain(all_edges, chosen, tol: float = 1e-4) -> list:
    """Grow an edge selection along edges that continue it smoothly (tangent at a shared vertex).

    The kernel's fillet/chamfer needs a whole tangent chain in one call (for example the
    straight sides of a slot together with its round ends); OpenCascade, and so build123d,
    propagate a selection along such chains automatically. Seams are never added.
    """
    candidates = [e for e in all_edges if len(e.faces()) != 1]
    ends = [(e, _edge_ends(e)) for e in candidates]
    selected = list(chosen)
    frontier = list(chosen)
    while frontier:
        cur = frontier.pop()
        for p, t in _edge_ends(cur):
            for e, e_ends in ends:
                # tangent continuation: same point, leaving in opposite directions. Test this
                # first: it is cheap, and _same_edge asks the kernel for lengths and points.
                if not any(g.dist(p, q) < 1e-6 and g.dot(t, u) < -1 + tol for q, u in e_ends):
                    continue
                if any(e is x or _same_edge(e, x) for x in selected):
                    continue
                selected.append(e)
                frontier.append(e)
    return selected


_BATCH_OPS = {"cut": 3, "fuse": 1}  # boolean_lists selectors: A minus B, A or B


def _check_boolean(name: str, a: Part, tools: list, result: Part) -> None:
    """Catch the kernel returning a geometrically impossible boolean result.

    The kernel occasionally drops an operand without reporting an error (seen when a
    mirrored part shares partially trimmed faces with the original). A fuse must contain
    every operand, be at least as big as each and no bigger than all of them together; a cut
    can't add volume or remove more than its tools hold; an intersect can't add volume.
    """
    va = a._quick_volume()
    vbs = [t._quick_volume() for t in tools]
    tol_v = 1e-3 * max(va, *vbs, 1e-9)
    if name != "fuse" and (name == "intersect" or va - sum(vbs) - tol_v <= 0):
        # the result's bounding box bounds its volume from above: when that already meets
        # the upper limit (and the lower limit is vacuous), the checks hold without measuring
        # the result, which is the slow part on a body with many features
        (lo, hi) = result.brep.bounding_box()
        box = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
        if box <= (va if name == "cut" else min(va, *vbs)) + tol_v:
            return
    vr = result._quick_volume()
    if name == "fuse":
        ok = (max(va, *vbs) - tol_v <= vr <= va + sum(vbs) + tol_v) and _box_contains(result, a, *tools)
    elif name == "cut":
        ok = va - sum(vbs) - tol_v <= vr <= va + tol_v
    else:
        ok = vr <= min(va, *vbs) + tol_v
    if not ok:
        vb = f"{vbs[0]:.6g}" if len(vbs) == 1 else f"{sum(vbs):.6g} in {len(vbs)} tools"
        raise KeelError(
            f"{name} failed: the kernel returned an impossible result (volumes {va:.6g}, {vb} -> "
            f"{vr:.6g}). This usually means the parts share coincident faces the kernel can't "
            "resolve; overlap them by a small amount or build the shape another way."
        )


def _volume(brep, accuracy: float) -> float:
    return ub.compute_volume(brep, accuracy)


def _box_contains(result: Part, *parts: Part) -> bool:
    rmin, rmax = result.bbox
    boxes = [p.bbox for p in parts]
    size = max(mx[i] - mn[i] for mn, mx in boxes for i in range(3))
    tol = max(1e-4, 1e-6 * size)
    return all(rmin[i] <= mn[i] + tol and rmax[i] >= mx[i] - tol for mn, mx in boxes for i in range(3))


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

