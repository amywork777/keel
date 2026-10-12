"""1D: paths made of line, arc, spline and helix segments.

Planar paths live in a Plane's local XY (2D points); closing one gives a Sketch. Any path
can be a sweep or pipe spine.

    p = Pen(0, 0).line_to(30, 0).line(0, 10).tangent_arc_to(20, 20).close()
    profile = p.to_sketch()
"""

from __future__ import annotations

import math
from typing import Sequence

from . import geom as g
from ._kernel import KeelError, call, ub
from .geom import Plane

_CLOSE_TOL = 1e-7


# -- segments (recipes, built into kernel curves on demand) -----------------------------------


class _Seg:
    planar = True

    @property
    def start(self):
        raise NotImplementedError

    @property
    def end(self):
        raise NotImplementedError

    def build(self):
        """A fresh kernel curve in local coordinates."""
        raise NotImplementedError

    def moved(self, d):
        raise NotImplementedError

    def end_tangent(self):
        raise NotImplementedError


class _Line(_Seg):
    def __init__(self, a, b):
        self.a, self.b = g.vec(a), g.vec(b)
        if g.dist(self.a, self.b) < 1e-12:
            raise ValueError("zero-length line")
        self.planar = abs(self.a[2]) < 1e-12 and abs(self.b[2]) < 1e-12

    start = property(lambda s: s.a)
    end = property(lambda s: s.b)

    def build(self):
        return ub.create_line_segment(self.a, self.b)

    def build_bspline(self):
        pts = [g.add(self.a, g.mul(g.sub(self.b, self.a), t)) for t in (0, 1 / 3, 2 / 3, 1)]
        return ub.create_curve_interp_points(pts)

    def moved(self, d):
        return _Line(g.add(self.a, d), g.add(self.b, d))

    def end_tangent(self):
        return g.norm(g.sub(self.b, self.a))


class _Arc(_Seg):
    """Circular arc in local XY; counter-clockwise if ``a1 > a0`` (degrees)."""

    def __init__(self, center, radius, a0, a1):
        self.c, self.r, self.a0, self.a1 = g.vec(center), float(radius), float(a0), float(a1)
        if abs(self.c[2]) > 1e-12:
            raise ValueError("arcs must lie in the sketch plane (z = 0)")
        if self.r <= 0 or abs(self.a1 - self.a0) < 1e-9:
            raise ValueError("arc needs a positive radius and a non-zero sweep")

    def _pt(self, a):
        t = math.radians(a)
        return (self.c[0] + self.r * math.cos(t), self.c[1] + self.r * math.sin(t), 0.0)

    start = property(lambda s: s._pt(s.a0))
    end = property(lambda s: s._pt(s.a1))

    def build(self):
        if self.a1 > self.a0:
            return ub.create_arc(self.c, self.r, self.a0, self.a1)
        return ub.create_arc(self.c, self.r, self.a1, self.a0)  # same arc, drawn the other way

    build_bspline = build

    def moved(self, d):
        return _Arc(g.add(self.c, d), self.r, self.a0, self.a1)

    def end_tangent(self):
        t = math.radians(self.a1)
        s = 1.0 if self.a1 > self.a0 else -1.0
        return (-math.sin(t) * s, math.cos(t) * s, 0.0)


class _Spline(_Seg):
    def __init__(self, pts):
        pts = [g.vec(p) for p in pts]
        if len(pts) < 2:
            raise ValueError("a spline needs at least two points")
        if len(pts) < 4:  # the kernel's interpolator needs four; add midpoints (still passes through pts)
            dense = [pts[0]]
            for a, b in zip(pts, pts[1:]):
                dense += [g.mul(g.add(a, b), 0.5), b]
            pts = dense if len(dense) >= 4 else [pts[0], *(g.add(pts[0], g.mul(g.sub(pts[1], pts[0]), t)) for t in (1 / 3, 2 / 3)), pts[1]]
        self.pts = pts
        self.planar = all(abs(p[2]) < 1e-12 for p in pts)

    start = property(lambda s: s.pts[0])
    end = property(lambda s: s.pts[-1])

    def build(self):
        return ub.create_curve_interp_points(self.pts, parameterization=ub.CurveParameterization.CHORD_LENGTH)

    build_bspline = build

    def moved(self, d):
        return _Spline([g.add(p, d) for p in self.pts])

    def end_tangent(self):
        return g.norm(g.sub(self.pts[-1], self.pts[-2]))


class _Helix(_Seg):
    planar = False

    def __init__(self, radius, height, turns, radius_end=None, left=False, origin=(0, 0, 0)):
        self.r0, self.h, self.turns = float(radius), float(height), float(turns)
        self.r1 = float(radius_end if radius_end is not None else radius)
        self.left, self.o = left, g.vec(origin)

    @property
    def start(self):
        return g.add(self.o, (self.r0, 0.0, 0.0))

    @property
    def end(self):
        a = 2 * math.pi * self.turns * (-1 if self.left else 1)
        return g.add(self.o, (self.r1 * math.cos(a), self.r1 * math.sin(a), self.h))

    def build(self):
        return ub.create_helix(self.o, self.h, self.r0, self.r1, self.turns, right_handed=not self.left)

    build_bspline = build

    def moved(self, d):
        return _Helix(self.r0, self.h, self.turns, self.r1, self.left, g.add(self.o, d))


# -- Path -------------------------------------------------------------------------------------


class Path:
    """An ordered chain of segments in a plane's local coordinates."""

    def __init__(self, segments: Sequence[_Seg], plane: Plane = Plane.XY):
        self.segments = list(segments)
        self.plane = plane
        for a, b in zip(self.segments, self.segments[1:]):
            if g.dist(a.end, b.start) > 1e-6:
                raise ValueError(f"path segments do not connect: {g._fmt(a.end)} -> {g._fmt(b.start)}")

    @property
    def start(self):
        return self.segments[0].start

    @property
    def end(self):
        return self.segments[-1].end

    @property
    def is_closed(self) -> bool:
        return g.dist(self.start, self.end) < 1e-6

    @property
    def is_planar(self) -> bool:
        return all(s.planar for s in self.segments)

    def __add__(self, other: "Path") -> "Path":
        if other.plane != self.plane:
            raise ValueError("cannot join paths on different planes")
        return Path(self.segments + other.segments, self.plane)

    def close(self) -> "Path":
        """Add a straight segment back to the start (no-op if already closed)."""
        if self.is_closed:
            return self
        return Path(self.segments + [_Line(self.end, self.start)], self.plane)

    def on(self, plane, x: float = 0, y: float = 0) -> "Path":
        return Path(self.segments, g.plane_of(plane).shifted(x, y))

    def move(self, x=0.0, y=0.0, z=0.0) -> "Path":
        d = g.vec(x) if isinstance(x, (tuple, list)) else (float(x), float(y), float(z))
        return Path([s.moved(d) for s in self.segments], self.plane)

    @property
    def length(self) -> float:
        return sum(c.length() for c in self.curves())

    def curves(self, bspline: bool = False) -> list:
        """Fresh kernel curves placed in world space."""
        out = []
        for s in self.segments:
            c = s.build_bspline() if bspline else s.build()
            out.append(self.plane.place(c))
        return out

    def to_sketch(self):
        """The region enclosed by this closed planar path."""
        from .sketch import Sketch

        if not self.is_closed:
            raise ValueError("path must be closed to make a sketch; call .close()")
        if not self.is_planar:
            raise ValueError("only planar paths can make a sketch")
        curves = [s.build() for s in self.segments]
        return Sketch(call("sketch from path", ub.create_planar_faces, curves), self.plane)

    def _is_coplanar(self) -> bool:
        if self.is_planar:
            return True
        pts = [self.plane.to_world(p) for seg in self.segments for p in _sample(seg)]
        return _coplanar(pts)

    def pipe(self, radius: float):
        """A round tube of ``radius`` centered on this path."""
        from .part import Part
        from .sketch import Circle

        if len(self.segments) == 1:
            curve = self.curves(bspline=True)[0]
            return Part(call("pipe", ub.pipe_sweep, curve, radius=float(radius)))
        return self.sweep(Circle(radius))

    def start_tangent(self) -> tuple:
        """Unit direction of the path at its start (world space)."""
        c = self.curves(bspline=True)[0]
        t0, t1 = c.parameter_range()
        a = c.evaluate(t0)
        b = c.evaluate(t0 + 1e-6 * (t1 - t0))
        return g.norm(g.sub(b, a))

    def sweep(self, profile):
        """Sweep a Sketch along this path.

        The profile is stood up perpendicular to the path at its start (its own plane is
        ignored) and its bounding-box center follows the path. Its local x maps to world X
        projected into that cross-section (world Y if the path starts along X).
        """
        import warnings

        from .part import Part
        from .sketch import _circle_radius

        start = self.plane.to_world(self.start)
        loops = profile._loop_curves(local=True)
        if len(loops) != 1 or loops[0][1]:
            raise KeelError("sweep profiles must be a single region without holes")
        if self._is_coplanar():
            prof = profile.on(Plane.at(start, self.start_tangent()))._loop_curves()[0][0]
            return Part(call("sweep", ub.sweep_along_planar_path, prof, self.curves(), move_profile=True))
        if len(self.segments) > 1:
            raise KeelError("multi-segment sweep paths must be planar")
        r = _circle_radius(loops[0][0])
        if r is not None:
            return Part(call("sweep", ub.pipe_sweep, self.curves(bspline=True)[0], radius=r))
        warnings.warn(
            "on a non-planar path the kernel anchors the profile's first boundary point (not its "
            "center) on the path",
            stacklevel=2,
        )
        prof = profile.on(Plane.at(start, self.start_tangent()))._loop_curves()[0][0]
        return Part(call("sweep", ub.curve_sweep, prof, self.curves(bspline=True)[0]))

    def __repr__(self):
        kinds = ",".join(type(s).__name__.strip("_").lower() for s in self.segments)
        return f"Path({kinds}{', closed' if self.is_closed else ''})"


# -- constructors -----------------------------------------------------------------------------


def Line(a, b) -> Path:
    return Path([_Line(a, b)])


def Polyline(*pts, close: bool = False) -> Path:
    pts = list(pts[0]) if len(pts) == 1 and not isinstance(pts[0][0], (int, float)) else list(pts)
    p = Path([_Line(a, b) for a, b in zip(pts, pts[1:])])
    return p.close() if close else p


def Arc(center, radius: float, start: float, end: float) -> Path:
    """Arc by center, radius and start/end angles in degrees (counter-clockwise when end > start)."""
    return Path([_Arc(center, radius, start, end)])


def ThreePointArc(p0, p1, p2) -> Path:
    """Arc from p0 through p1 to p2."""
    return Path([_arc_through(g.vec(p0), g.vec(p1), g.vec(p2))])


def Spline(*pts) -> Path:
    pts = list(pts[0]) if len(pts) == 1 and not isinstance(pts[0][0], (int, float)) else list(pts)
    return Path([_Spline(pts)])


def Helix(radius: float, height: float, turns: float, *, radius_end=None, left: bool = False) -> Path:
    """Helix around +Z starting on +X (world coordinates)."""
    return Path([_Helix(radius, height, turns, radius_end, left)])


class Pen:
    """Draw a planar path step by step.

        Pen(0, 0).line_to(20, 0).line(0, 5).tangent_arc_to(10, 15).close()
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, plane=Plane.XY):
        self._segs: list[_Seg] = []
        self._start = (float(x), float(y), 0.0)
        self._at = self._start
        self._plane = g.plane_of(plane)

    def _push(self, seg: _Seg) -> "Pen":
        self._segs.append(seg)
        self._at = seg.end
        return self

    def line_to(self, x: float, y: float) -> "Pen":
        return self._push(_Line(self._at, (x, y, 0)))

    def line(self, dx: float, dy: float) -> "Pen":
        return self.line_to(self._at[0] + dx, self._at[1] + dy)

    def h(self, dx: float) -> "Pen":
        return self.line(dx, 0)

    def v(self, dy: float) -> "Pen":
        return self.line(0, dy)

    def arc_to(self, x: float, y: float, through) -> "Pen":
        """Arc to (x, y) passing through ``through``."""
        return self._push(_arc_through(self._at, g.vec(through), (float(x), float(y), 0.0)))

    def tangent_arc_to(self, x: float, y: float) -> "Pen":
        """Arc to (x, y) leaving tangent to the previous segment."""
        if not self._segs:
            raise ValueError("tangent_arc_to needs a previous segment")
        t = self._segs[-1].end_tangent()
        return self._push(_tangent_arc(self._at, t, (float(x), float(y), 0.0)))

    def spline_to(self, *pts) -> "Pen":
        """Spline through the given points, ending at the last one."""
        return self._push(_Spline([self._at, *pts]))

    @property
    def path(self) -> Path:
        return Path(self._segs, self._plane)

    def close(self) -> Path:
        if g.dist(self._at, self._start) > _CLOSE_TOL:
            self.line_to(self._start[0], self._start[1])
        return Path(self._segs, self._plane)

    def to_sketch(self):
        return self.close().to_sketch()


# -- arc construction ---------------------------------------------------------------------------


def _angle(c, p) -> float:
    return math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))


def _arc_from(c, r, p0, p1, ccw: bool) -> _Arc:
    a0, a1 = _angle(c, p0), _angle(c, p1)
    if ccw:
        while a1 <= a0:
            a1 += 360.0
    else:
        while a1 >= a0:
            a1 -= 360.0
    return _Arc(c, r, a0, a1)


def _arc_through(p0, pm, p1) -> _Arc:
    ax, ay, bx, by, cx, cy = p0[0], p0[1], pm[0], pm[1], p1[0], p1[1]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        raise ValueError("three-point arc points are collinear")
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    c = (ux, uy, 0.0)
    r = g.dist(c, p0)
    # counter-clockwise when the middle point is to the right of the chord p0 -> p1
    ccw = ((p1[0] - p0[0]) * (pm[1] - p0[1]) - (p1[1] - p0[1]) * (pm[0] - p0[0])) < 0
    return _arc_from(c, r, p0, p1, ccw)


def _tangent_arc(p0, t, p1) -> _Seg:
    chord = g.sub(p1, p0)
    n = (-t[1], t[0], 0.0)  # left normal of the tangent
    k = g.dot(chord, n)
    if abs(k) < 1e-12:  # end point straight ahead: it's a line
        return _Line(p0, p1)
    r = g.dot(chord, chord) / (2 * k)  # signed: > 0 turns left (ccw)
    c = g.add(p0, g.mul(n, r))
    return _arc_from(c, abs(r), p0, p1, ccw=r > 0)


def _sample(seg, n: int = 8) -> list:
    if isinstance(seg, _Spline):
        return seg.pts
    if isinstance(seg, _Arc):
        return [seg._pt(seg.a0 + (seg.a1 - seg.a0) * k / n) for k in range(n + 1)]
    if isinstance(seg, _Helix):
        out = []
        for k in range(n + 1):
            f = k / n
            a = 2 * math.pi * seg.turns * f * (-1 if seg.left else 1)
            r = seg.r0 + (seg.r1 - seg.r0) * f
            out.append(g.add(seg.o, (r * math.cos(a), r * math.sin(a), seg.h * f)))
        return out
    return [seg.start, seg.end]


def _coplanar(pts, tol: float = 1e-7) -> bool:
    """True if all points lie on one plane (collinear sets count as planar)."""
    p0 = pts[0]
    normal = None
    for i in range(1, len(pts)):
        for j in range(i + 1, len(pts)):
            n = g.cross(g.sub(pts[i], p0), g.sub(pts[j], p0))
            if g.length(n) > 1e-9:
                normal = g.norm(n)
                break
        if normal:
            break
    if normal is None:
        return True
    scale = max(1.0, max(g.dist(p, p0) for p in pts))
    return all(abs(g.dot(g.sub(p, p0), normal)) <= tol * scale for p in pts)
