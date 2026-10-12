"""3D primitives. All are centered on the origin unless ``align`` says otherwise.

``align`` is "center" (default), "min" or "max", or one of those per axis:
``Box(10, 10, 5, align=("center", "center", "min"))`` sits on the XY plane.
"""

from __future__ import annotations

from ._kernel import call, ub
from .part import Part

_MODES = {"center": 0.5, "min": 0.0, "max": 1.0}


def _shift(size, align) -> tuple:
    """Translation that moves a box occupying [0, size] to the requested alignment."""
    modes = (align,) * 3 if isinstance(align, str) else tuple(align)
    if len(modes) != 3 or any(m not in _MODES for m in modes):
        raise ValueError('align must be "center", "min", "max" or a 3-tuple of those')
    return tuple(-s * _MODES[m] for s, m in zip(size, modes))


def _placed(brep, offset, label=None) -> Part:
    if any(abs(c) > 0 for c in offset):
        ub.translate(brep, offset)
    return Part(brep, label)


def Box(length: float, width: float, height: float, *, align="center") -> Part:
    """Box with ``length`` along X, ``width`` along Y, ``height`` along Z."""
    size = (float(length), float(width), float(height))
    return _placed(call("Box", ub.create_box, (0.0, 0.0, 0.0), *size), _shift(size, align))


def Cylinder(radius: float, height: float, *, align="center") -> Part:
    """Cylinder along Z."""
    r, h = float(radius), float(height)
    off = _shift((2 * r, 2 * r, h), align)
    return _placed(call("Cylinder", ub.create_cylinder, (r, r, 0.0), r, h), off)


def Cone(bottom_radius: float, top_radius: float, height: float, *, align="center") -> Part:
    """Cone or frustum along Z (either radius may be 0)."""
    r0, r1, h = float(bottom_radius), float(top_radius), float(height)
    r = max(r0, r1)
    off = _shift((2 * r, 2 * r, h), align)
    return _placed(call("Cone", ub.create_cone, (r, r, 0.0), r0, r1, h), off)


def Sphere(radius: float, *, align="center") -> Part:
    r = float(radius)
    off = _shift((2 * r, 2 * r, 2 * r), align)
    return _placed(call("Sphere", ub.create_sphere, (r, r, r), r), off)


def Torus(major_radius: float, minor_radius: float, *, align="center") -> Part:
    """Torus around Z: tube of ``minor_radius`` on a circle of ``major_radius``."""
    R, r = float(major_radius), float(minor_radius)
    off = _shift((2 * (R + r), 2 * (R + r), 2 * r), align)
    return _placed(call("Torus", ub.create_torus, (R + r, R + r, r), R, r), off)
