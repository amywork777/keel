"""Location patterns: lists of (x, y, z) offsets to feed ``.at(...)``.

    plate - Cylinder(2, 10).at(grid(3, 2, 15, 10))
    disc - Cylinder(1, 5).at(polar(6, 20))
"""

from __future__ import annotations

import math


def grid(nx: int, ny: int, dx: float, dy: float, *, center: bool = True) -> list[tuple[float, float, float]]:
    """``nx`` × ``ny`` points spaced ``dx``/``dy`` apart, centered on the origin by default."""
    ox = -(nx - 1) * dx / 2 if center else 0.0
    oy = -(ny - 1) * dy / 2 if center else 0.0
    return [(ox + i * dx, oy + j * dy, 0.0) for j in range(ny) for i in range(nx)]


def polar(count: int, radius: float, *, start: float = 0.0, sweep: float = 360.0) -> list[tuple[float, float, float]]:
    """``count`` points on a circle; ``sweep`` < 360 spreads them over an arc including both ends."""
    step = sweep / count if abs(sweep - 360.0) < 1e-9 else sweep / max(count - 1, 1)
    pts = []
    for i in range(count):
        a = math.radians(start + i * step)
        pts.append((radius * math.cos(a), radius * math.sin(a), 0.0))
    return pts


def line_of(count: int, spacing: float, direction=(1.0, 0.0, 0.0), *, center: bool = False):
    """``count`` points along ``direction``."""
    o = -(count - 1) * spacing / 2 if center else 0.0
    return [tuple((o + i * spacing) * c for c in direction) for i in range(count)]
