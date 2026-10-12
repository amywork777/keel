"""Builder mode: accumulate a part step by step inside a ``with`` block.

    with Build() as b:
        b += Box(40, 20, 5)
        b -= Cylinder(3, 5).at(grid(2, 1, 25, 0))
        b.fillet("|Z", 2)
    bracket = b.part
"""

from __future__ import annotations

from .part import Part
from .topo import Selection


class Build:
    def __init__(self, start: Part | None = None):
        self.part: Part | None = start

    def __enter__(self) -> "Build":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def _need(self) -> Part:
        if self.part is None:
            raise ValueError("add something first (b += Box(...))")
        return self.part

    def __iadd__(self, other) -> "Build":
        items = other if isinstance(other, (list, tuple)) else [other]
        if self.part is None:
            self.part, items = items[0], list(items[1:])
        if items:
            self.part = self.part.fuse(*items)
        return self

    def __isub__(self, other) -> "Build":
        self.part = self._need().cut(other)
        return self

    def __iand__(self, other) -> "Build":
        self.part = self._need().intersect(other)
        return self

    # features apply to the current part
    def fillet(self, edges, radius: float) -> "Build":
        self.part = self._need().fillet(edges, radius)
        return self

    def chamfer(self, edges, distance: float) -> "Build":
        self.part = self._need().chamfer(edges, distance)
        return self

    def shell(self, thickness: float, open=None) -> "Build":
        self.part = self._need().shell(thickness, open)
        return self

    def faces(self, selector: str | None = None) -> Selection:
        return self._need().faces(selector)

    def edges(self, selector: str | None = None) -> Selection:
        return self._need().edges(selector)

    def vertices(self, selector: str | None = None) -> Selection:
        return self._need().vertices(selector)
