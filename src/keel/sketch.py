"""2D: sketches are planar regions (with holes) drawn in a plane's local XY.

Combine with ``+`` (union), ``-`` (difference), ``&`` (intersection) and ``^`` (xor), put
them somewhere with ``.on(plane_or_face)``, then turn them into Parts with
``extrude``, ``revolve``, ``sweep`` or ``loft``.
"""

from __future__ import annotations

import math

from . import geom as g
from ._kernel import KeelError, call, ub
from .geom import Plane
from .part import Part


class Sketch:
    def __init__(self, sheet, plane: Plane = Plane.XY):
        self._sheet = sheet  # kernel sheet Brep in local XY (z = 0)
        self.plane = plane

    # -- measures -----------------------------------------------------------------------------
    @property
    def area(self) -> float:
        return self._sheet.area()

    @property
    def region_count(self) -> int:
        return self._sheet.face_count()

    @property
    def bbox(self):
        """Local-coordinate bounding box ((xmin, ymin), (xmax, ymax))."""
        mn, mx = self._sheet.bounding_box()
        return (mn[0], mn[1]), (mx[0], mx[1])

    # -- 2D booleans --------------------------------------------------------------------------
    def _bool(self, other: "Sketch", op, name: str) -> "Sketch":
        if not isinstance(other, Sketch):
            return NotImplemented
        plane = self.plane
        if other.plane != self.plane:
            if self.plane == Plane.XY:
                plane = other.plane
            elif other.plane != Plane.XY:
                raise ValueError("cannot combine sketches on different planes; combine first, then .on()")
        return Sketch(call(name, ub.boolean_2d, self._sheet.copy(), other._sheet.copy(), operation=op), plane)

    def __add__(self, other):
        return self._bool(other, ub.BooleanOp2D.UNION, "sketch union")

    def __sub__(self, other):
        return self._bool(other, ub.BooleanOp2D.DIFFERENCE, "sketch difference")

    def __and__(self, other):
        return self._bool(other, ub.BooleanOp2D.INTERSECTION, "sketch intersection")

    def __xor__(self, other):
        return self._bool(other, ub.BooleanOp2D.EXCLUSIVE_OR, "sketch xor")

    def __radd__(self, other):
        return self if other == 0 else NotImplemented

    # -- placement (local, within the sketch plane) --------------------------------------------
    def move(self, x: float = 0.0, y: float = 0.0) -> "Sketch":
        if isinstance(x, (tuple, list)):
            x, y = x[0], x[1]
        return Sketch(ub.translate(self._sheet.copy(), (float(x), float(y), 0.0)), self.plane)

    def at(self, *locations):
        """Copies moved to each (x, y): one location -> Sketch, many -> their union."""
        locs = locations[0] if len(locations) == 1 and not isinstance(locations[0][0], (int, float)) else locations
        moved = [self.move(p[0], p[1]) for p in locs]
        out = moved[0]
        for s in moved[1:]:
            out = out + s
        return out

    def rotate(self, degrees: float, about=(0, 0)) -> "Sketch":
        c = (float(about[0]), float(about[1]), 0.0)
        return Sketch(ub.rotate(self._sheet.copy(), c, (0.0, 0.0, 1.0), float(degrees)), self.plane)

    def scale(self, factor: float) -> "Sketch":
        return Sketch(ub.scale(self._sheet.copy(), (factor, factor, 1.0)), self.plane)

    def mirror(self, axis: str = "Y") -> "Sketch":
        """Mirror across the local Y axis ("Y", flips x) or X axis ("X", flips y)."""
        f = (-1.0, 1.0, 1.0) if axis.upper() == "Y" else (1.0, -1.0, 1.0)
        return Sketch(call("sketch mirror", ub.scale, self._sheet.copy(), f), self.plane)

    def on(self, target, x: float = 0.0, y: float = 0.0, rotation: float = 0.0) -> "Sketch":
        """Put the sketch on a Plane, planar Face, face Selection or plane name ("XZ").

        ``x``/``y`` offset the sketch origin within that plane; ``rotation`` spins it.
        """
        p = g.plane_of(target).shifted(x, y)
        if rotation:
            p = p.rotated(rotation)
        return Sketch(self._sheet, p)

    # -- to 3D --------------------------------------------------------------------------------
    def _placed(self):
        return self.plane.place(self._sheet.copy())

    def extrude(self, amount: float, *, both: bool = False, taper: float = 0.0) -> Part:
        """Extrude along the plane normal (negative goes the other way).

        ``both=True`` extrudes ``amount`` to each side. ``taper`` (degrees, > 0 narrows) needs
        regions without holes.
        """
        amount = float(amount)
        if abs(amount) < 1e-12:
            raise ValueError("extrude amount must be non-zero")
        if taper:
            return self._taper_extrude(amount, float(taper), both)
        d = self.plane.z if amount > 0 else g.mul(self.plane.z, -1)
        sheet = self._placed()
        dist = abs(amount)
        if both:
            ub.translate(sheet, g.mul(d, -dist))
            dist *= 2
        body = call("extrude", ub.non_manifold_linear_sweep, sheet, d, dist, faces=sheet.faces())
        return Part(body)

    def _taper_extrude(self, amount, taper, both) -> Part:
        if both:
            raise KeelError("taper with both=True is not supported")
        bodies = []
        for outer, holes in self._loop_curves(local=True):
            if holes:
                raise KeelError("taper needs regions without holes")
            b = call("tapered extrude", ub.draft_sweep, outer, amount, taper)
            bodies.append(Part(self.plane.place(b)))
        out = bodies[0]
        return out.fuse(*bodies[1:]) if len(bodies) > 1 else out

    def revolve(self, degrees: float = 360.0, axis="Y") -> Part:
        """Revolve about an in-plane axis: "X" or "Y" through the sketch origin,
        or ``((x, y), (dx, dy))`` for any local line."""
        if isinstance(axis, str):
            base, dir_local = (0.0, 0.0, 0.0), g.direction(axis)
        else:
            base, dir_local = g.vec(axis[0]), g.norm(g.vec(axis[1]))
        base_w = self.plane.to_world(base)
        dir_w = self.plane.dir_to_world(dir_local)
        result = None
        for outer, holes in self._loop_curves():
            body = Part(call("revolve", ub.rotational_sweep, outer, base_w, dir_w, float(degrees)))
            for h in holes:
                body = body - Part(call("revolve", ub.rotational_sweep, h, base_w, dir_w, float(degrees)))
            result = body if result is None else result + body
        return result

    def sweep(self, path) -> Part:
        """Sweep along a Path; the sketch's local origin follows the path start."""
        return path.sweep(self)

    def _loop_curves(self, local: bool = False):
        """[(outer_curves, [hole_curves, ...]), ...] per region, as kernel curves."""
        sheet = self._sheet.copy() if local else self._placed()
        out = []
        for f in sheet.faces():
            outer_edges = f.outer_loop_edges()
            outer_ids = {_edge_key(e) for e in outer_edges}
            holes = []
            for loop in f.boundary_loops():
                edges = [e for e, _ in loop]
                if {_edge_key(e) for e in edges} != outer_ids:
                    holes.append([e.curve() for e in edges])
            out.append(([e.curve() for e in outer_edges], holes))
        # keep the sheet alive as long as its borrowed curves are in use
        self._keepalive = sheet
        return out

    def __repr__(self):
        return f"<Sketch regions={self.region_count} area={self.area:.6g} on {self.plane}>"


def _circle_radius(curves):
    """Radius if a loop is one full circle, else None."""
    if len(curves) == 1 and hasattr(curves[0], "radius") and curves[0].is_closed():
        return curves[0].radius()
    return None


def _edge_key(e):
    a, b = e.parameter_range()
    return tuple(round(c, 7) for c in e.point_at(0.5 * (a + b)))


def _from_curves(curves, name: str) -> Sketch:
    return Sketch(call(name, ub.create_planar_faces, list(curves)))


# -- 2D primitives (all centered on the local origin) ------------------------------------------


def Rect(width: float, height: float) -> Sketch:
    return _from_curves(ub.create_rectangle((0.0, 0.0, 0.0), float(width), float(height)), "Rect")


def Circle(radius: float) -> Sketch:
    return _from_curves([ub.create_circle((0.0, 0.0, 0.0), float(radius))], "Circle")


def Ellipse(x_radius: float, y_radius: float) -> Sketch:
    return _from_curves([ub.create_ellipse((0.0, 0.0, 0.0), float(x_radius), float(y_radius))], "Ellipse")


def RegularPolygon(radius: float, sides: int, rotation: float = 0.0) -> Sketch:
    """Polygon with ``sides`` corners on a circle of ``radius`` (first corner on +X)."""
    s = _from_curves(ub.create_regular_polygon((0.0, 0.0, 0.0), int(sides), float(radius)), "RegularPolygon")
    return s.rotate(rotation) if rotation else s


def Polygon(*pts) -> Sketch:
    """Closed polygon through the given (x, y) points."""
    pts = list(pts[0]) if len(pts) == 1 and not isinstance(pts[0][0], (int, float)) else list(pts)
    pts = [g.vec(p) for p in pts]
    if g.dist(pts[0], pts[-1]) < 1e-9:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("a polygon needs at least three points")
    loop = pts + [pts[0]]
    return _from_curves([ub.create_line_segment(a, b) for a, b in zip(loop, loop[1:])], "Polygon")


def Slot(length: float, width: float) -> Sketch:
    """Stadium shape: overall ``length`` along X, ``width`` across, round ends."""
    r = width / 2.0
    half = length / 2.0 - r
    if half <= 0:
        return Circle(r)
    curves = [
        ub.create_line_segment((-half, -r, 0.0), (half, -r, 0.0)),
        ub.create_arc((half, 0.0, 0.0), r, -90.0, 90.0),
        ub.create_line_segment((half, r, 0.0), (-half, r, 0.0)),
        ub.create_arc((-half, 0.0, 0.0), r, 90.0, 270.0),
    ]
    return _from_curves(curves, "Slot")


def RoundedRect(width: float, height: float, radius: float) -> Sketch:
    """Rectangle with rounded corners."""
    w, h, r = width / 2.0, height / 2.0, float(radius)
    if r <= 0:
        return Rect(width, height)
    if r > min(w, h) + 1e-12:
        raise ValueError("corner radius larger than half the rectangle")
    curves = []
    corners = [(w - r, h - r, 0), (-w + r, h - r, 90), (-w + r, -h + r, 180), (w - r, -h + r, 270)]
    for i, (cx, cy, a) in enumerate(corners):
        curves.append(ub.create_arc((cx, cy, 0.0), r, a, a + 90))
        nx, ny, na = corners[(i + 1) % 4]
        p0 = (cx + r * math.cos(math.radians(a + 90)), cy + r * math.sin(math.radians(a + 90)), 0.0)
        p1 = (nx + r * math.cos(math.radians(na)), ny + r * math.sin(math.radians(na)), 0.0)
        if g.dist(p0, p1) > 1e-9:
            curves.append(ub.create_line_segment(p0, p1))
    return _from_curves(curves, "RoundedRect")


# -- functional forms -----------------------------------------------------------------------------


def extrude(sketch: Sketch, amount: float, **kw) -> Part:
    return sketch.extrude(amount, **kw)


def revolve(sketch: Sketch, degrees: float = 360.0, axis="Y") -> Part:
    return sketch.revolve(degrees, axis)


def sweep(sketch: Sketch, path) -> Part:
    return sketch.sweep(path)


def loft(*sketches: Sketch, smooth: bool = False) -> Part:
    """Skin a solid through single-region sketches (each placed with ``.on``)."""
    sketches = list(sketches[0]) if len(sketches) == 1 and isinstance(sketches[0], (list, tuple)) else list(sketches)
    if len(sketches) < 2:
        raise ValueError("loft needs at least two sketches")
    counts, curves = [], []
    for s in sketches:
        loops = s._loop_curves()
        if len(loops) != 1 or loops[0][1]:
            raise KeelError("loft sections must be single regions without holes")
        counts.append(len(loops[0][0]))
        curves += loops[0][0]
    degree = min(3, len(sketches) - 1) if smooth else 1
    return Part(call("loft", ub.create_skin_primitive, counts, curves, degree))
