"""Vectors, axes and planes.

Points and vectors are plain tuples everywhere in keel's API; these helpers do the math.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from ._kernel import ub

TOL = 1e-6

Vec = tuple  # (x, y, z)


def vec(p: Sequence[float]) -> tuple[float, float, float]:
    """Coerce a 2- or 3-sequence to a 3-tuple of floats."""
    if len(p) == 2:
        return (float(p[0]), float(p[1]), 0.0)
    if len(p) == 3:
        return (float(p[0]), float(p[1]), float(p[2]))
    raise ValueError(f"expected a 2D or 3D point, got {p!r}")


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, s: float):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def length(a) -> float:
    return math.sqrt(dot(a, a))


def norm(a):
    n = length(a)
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return mul(a, 1.0 / n)


def dist(a, b) -> float:
    return length(sub(a, b))


_AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def direction(d) -> tuple[float, float, float]:
    """Parse a direction: "X", "-Z", "+Y", an ``Axis`` or a 3-tuple."""
    if isinstance(d, Axis):
        return d.dir
    if isinstance(d, str):
        s = d.strip().upper()
        sign = -1.0 if s.startswith("-") else 1.0
        s = s.lstrip("+-")
        if s in _AXES:
            return mul(_AXES[s], sign)
        if s.startswith("(") and s.endswith(")"):
            return mul(norm(vec([float(v) for v in s[1:-1].split(",")])), sign)
        raise ValueError(f"unknown direction {d!r}")
    return norm(vec(d))


class Axis:
    """An infinite line: a point and a unit direction."""

    def __init__(self, origin=(0, 0, 0), dir=(0, 0, 1)):
        self.origin = vec(origin)
        self.dir = direction(dir)

    def __repr__(self):
        return f"Axis(origin={_fmt(self.origin)}, dir={_fmt(self.dir)})"


Axis.X = Axis((0, 0, 0), (1, 0, 0))
Axis.Y = Axis((0, 0, 0), (0, 1, 0))
Axis.Z = Axis((0, 0, 0), (0, 0, 1))


class Plane:
    """A local coordinate frame. Sketches are drawn in a plane's local XY.

    ``x`` and ``z`` are orthonormalized; ``y = z × x``.
    """

    XY: "Plane"
    XZ: "Plane"
    YZ: "Plane"
    YX: "Plane"
    ZX: "Plane"
    ZY: "Plane"

    def __init__(self, origin=(0, 0, 0), x_dir=(1, 0, 0), z_dir=(0, 0, 1)):
        z = direction(z_dir)
        x = vec(x_dir)
        x = sub(x, mul(z, dot(x, z)))
        if length(x) < 1e-9:  # x parallel to z: pick any perpendicular
            x = _any_perpendicular(z)
        self.origin = vec(origin)
        self.x = norm(x)
        self.z = z
        self.y = cross(self.z, self.x)

    @classmethod
    def at(cls, origin=(0, 0, 0), normal="Z", x_dir=None) -> "Plane":
        """A plane through ``origin`` facing ``normal``; ``x_dir`` defaults to world X projected."""
        z = direction(normal)
        return cls(origin, x_dir if x_dir is not None else _default_x(z), z)

    # -- frame math -------------------------------------------------------------------------
    def to_world(self, p) -> tuple[float, float, float]:
        """Local (x, y[, z]) -> world point."""
        p = vec(p)
        return add(self.origin, add(mul(self.x, p[0]), add(mul(self.y, p[1]), mul(self.z, p[2]))))

    def to_local(self, p) -> tuple[float, float, float]:
        d = sub(vec(p), self.origin)
        return (dot(d, self.x), dot(d, self.y), dot(d, self.z))

    def dir_to_world(self, d) -> tuple[float, float, float]:
        d = vec(d)
        return add(mul(self.x, d[0]), add(mul(self.y, d[1]), mul(self.z, d[2])))

    # -- derived planes ---------------------------------------------------------------------
    def offset(self, distance: float) -> "Plane":
        """The same plane moved ``distance`` along its normal."""
        return Plane(add(self.origin, mul(self.z, distance)), self.x, self.z)

    def shifted(self, x: float = 0, y: float = 0) -> "Plane":
        """The same plane with its origin moved within the plane."""
        return Plane(self.to_world((x, y, 0)), self.x, self.z)

    def rotated(self, degrees: float) -> "Plane":
        """The same plane spun about its normal."""
        a = math.radians(degrees)
        return Plane(self.origin, add(mul(self.x, math.cos(a)), mul(self.y, math.sin(a))), self.z)

    def flipped(self) -> "Plane":
        """The plane facing the other way (x kept, normal reversed)."""
        return Plane(self.origin, self.x, mul(self.z, -1))

    # -- placing kernel objects -------------------------------------------------------------
    def _rotation(self):
        """Axis-angle (axis, degrees) that turns world XY onto this frame, or None."""
        return _axis_angle(self.x, self.y, self.z)

    def place(self, obj):
        """Move a kernel object built in world XY onto this plane (mutates ``obj``)."""
        rot = self._rotation()
        if rot is not None:
            ub.rotate(obj, (0.0, 0.0, 0.0), rot[0], rot[1])
        if length(self.origin) > 0:
            ub.translate(obj, self.origin)
        return obj

    def is_parallel(self, other: "Plane", tol: float = TOL) -> bool:
        return abs(abs(dot(self.z, other.z)) - 1) < tol

    def __eq__(self, other):
        if not isinstance(other, Plane):
            return NotImplemented
        return (
            dist(self.origin, other.origin) < TOL
            and dist(self.x, other.x) < TOL
            and dist(self.z, other.z) < TOL
        )

    def __hash__(self):
        return hash(tuple(round(v, 6) for v in self.origin + self.x + self.z))

    def __repr__(self):
        return f"Plane(origin={_fmt(self.origin)}, x={_fmt(self.x)}, z={_fmt(self.z)})"


Plane.XY = Plane((0, 0, 0), (1, 0, 0), (0, 0, 1))
Plane.YX = Plane((0, 0, 0), (0, 1, 0), (0, 0, -1))
Plane.XZ = Plane((0, 0, 0), (1, 0, 0), (0, -1, 0))  # local y = world Z
Plane.ZX = Plane((0, 0, 0), (0, 0, 1), (0, 1, 0))
Plane.YZ = Plane((0, 0, 0), (0, 1, 0), (1, 0, 0))  # local y = world Z
Plane.ZY = Plane((0, 0, 0), (0, 0, 1), (-1, 0, 0))


def plane_of(target) -> Plane:
    """Resolve a plane argument: a Plane, a planar Face, a name like "XZ", or a Selection of one face."""
    if isinstance(target, Plane):
        return target
    if isinstance(target, str):
        try:
            return getattr(Plane, target.upper())
        except AttributeError:
            raise ValueError(f"unknown plane {target!r}") from None
    to_plane = getattr(target, "plane", None)
    if callable(to_plane):
        return to_plane()
    raise TypeError(f"cannot use {target!r} as a plane")


# -- internals -----------------------------------------------------------------------------


def _fmt(v) -> str:
    return "(" + ", ".join(f"{c:g}" for c in v) + ")"


def _any_perpendicular(z):
    helper = (1.0, 0.0, 0.0) if abs(z[0]) < 0.9 else (0.0, 1.0, 0.0)
    return norm(cross(cross(z, helper), z))


def _default_x(z):
    """World X projected into the plane, else world Y, so face sketches have a predictable x."""
    for cand in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        p = sub(cand, mul(z, dot(cand, z)))
        if length(p) > 1e-6:
            return norm(p)
    return _any_perpendicular(z)


def _axis_angle(x, y, z):
    # Columns of R are the frame axes: R maps world XY to this frame.
    r = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    trace = r[0][0] + r[1][1] + r[2][2]
    c = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    angle = math.acos(c)
    if angle < 1e-12:
        return None
    if math.pi - angle < 1e-6:
        # 180 degrees: axis from the symmetric part, R = 2 a a^T - I
        diag = [(r[i][i] + 1.0) / 2.0 for i in range(3)]
        i = max(range(3), key=lambda k: diag[k])
        a = [0.0, 0.0, 0.0]
        a[i] = math.sqrt(max(diag[i], 0.0))
        for j in range(3):
            if j != i:
                a[j] = r[i][j] / (2.0 * a[i])
        return norm(tuple(a)), 180.0
    s = 2.0 * math.sin(angle)
    axis = ((r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s)
    return norm(axis), math.degrees(angle)


def points(seq: Iterable) -> list[tuple[float, float, float]]:
    return [vec(p) for p in seq]
