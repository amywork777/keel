"""Files: exact BRep USD (BrepArray prims), meshes (STL, mesh USD), and loading.

USD output uses NVIDIA's omniSolid ``BrepArray`` schema, so other usd-brep tools (and
NVIDIA's Hydra plugin) read it as exact geometry. ``.usda`` files also get prim names from
part labels, ``displayColor`` from part colors, and millimeter/Z-up stage metadata.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import tempfile
from . import geom as g
from ._kernel import KeelError, call, ub, usd_call
from .part import Part, _flatten

_USD_EXT = (".usda", ".usdc", ".usd")


class Mesh:
    """Triangle mesh: ``points`` [(x, y, z)], ``triangles`` [(i, j, k)]."""

    def __init__(self, points, triangles):
        self.points = [tuple(p) for p in points]
        self.triangles = [tuple(t) for t in triangles]

    def __add__(self, other: "Mesh") -> "Mesh":
        n = len(self.points)
        return Mesh(self.points + other.points, self.triangles + [(a + n, b + n, c + n) for a, b, c in other.triangles])

    def write_stl(self, path: str) -> str:
        with open(path, "wb") as f:
            f.write(b"keel".ljust(80, b" "))
            f.write(struct.pack("<I", len(self.triangles)))
            for a, b, c in self.triangles:
                pa, pb, pc = self.points[a], self.points[b], self.points[c]
                n = g.cross(g.sub(pb, pa), g.sub(pc, pa))
                ln = g.length(n)
                n = g.mul(n, 1 / ln) if ln > 0 else (0.0, 0.0, 0.0)
                f.write(struct.pack("<12fH", *n, *pa, *pb, *pc, 0))
        return path

    def write_obj(self, path: str) -> str:
        with open(path, "w") as f:
            for p in self.points:
                f.write(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
            for a, b, c in self.triangles:
                f.write(f"f {a + 1} {b + 1} {c + 1}\n")
        return path

    def __repr__(self):
        return f"<Mesh points={len(self.points)} triangles={len(self.triangles)}>"


def tessellate(part: Part, tolerance: float = 0.0, angle: float = 15.0) -> Mesh:
    """Mesh a part. ``tolerance``: max chord deviation (0 = automatic), ``angle``: degrees."""
    poly = call(
        "tessellate",
        ub.tessellate,
        part.brep,
        chord_height_tolerance=float(tolerance),
        curve_angle_tolerance_deg=float(angle),
        surface_angle_tolerance_deg=float(angle),
    )
    arrays = poly.to_mesh_arrays()
    flat, tris, i = arrays["faces"], [], 0
    while i < len(flat):  # [n, i0, ..., in-1, n, ...]; fan-triangulate any polygon
        n = flat[i]
        idx = flat[i + 1 : i + 1 + n]
        tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, n - 1)]
        i += n + 1
    return Mesh(arrays["points"], tris)


def _parts(items) -> list[Part]:
    if isinstance(items, Part):
        return [items]
    if isinstance(items, dict):
        return [p.named(name) if p.label is None else p for name, p in items.items()]
    parts = _flatten(list(items))
    for p in parts:
        if not isinstance(p, Part):
            raise TypeError(f"can only export Parts, got {type(p).__name__}")
    return parts


def export(items, path: str, *, uv_curves: bool = True, tolerance: float = 0.0) -> str:
    """Write one or more Parts. Format from the extension:

    ``.usda``/``.usdc``/``.usd``  exact BRep, one BrepArray prim per part under /World
    ``.stl`` / ``.obj``           tessellated mesh of all parts
    """
    parts = _parts(items)
    if not parts:
        raise ValueError("nothing to export")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".stl", ".obj"):
        mesh = tessellate(parts[0], tolerance)
        for p in parts[1:]:
            mesh = mesh + tessellate(p, tolerance)
        return mesh.write_stl(path) if ext == ".stl" else mesh.write_obj(path)
    if ext not in _USD_EXT:
        raise ValueError(f"unsupported export format {ext!r} (use .usda .usdc .usd .stl .obj)")

    breps = [p.brep.copy() for p in parts]  # export may cache trims on its input
    tmp_dir = tempfile.mkdtemp(prefix="keel-")
    try:
        tmp = os.path.join(tmp_dir, "out.usda")
        usd_call("export", ub.usd.export_brep, breps[0], tmp, export_uv_curves=uv_curves)
        if len(breps) > 1:
            usd_call("export", ub.usd.append_breps, breps[1:], tmp, export_uv_curves=uv_curves)
        with open(tmp) as f:
            text = _decorate_usda(f.read(), parts)
        if ext == ".usda":
            with open(path, "w") as f:
                f.write(text)
        else:
            # No pxr in the kernel wheel to re-encode the decorated layer; write crate directly.
            usd_call("export", ub.usd.export_brep, breps[0].copy(), path, export_uv_curves=uv_curves)
            if len(breps) > 1:
                usd_call("export", ub.usd.append_breps, [b.copy() for b in breps[1:]], path, export_uv_curves=uv_curves)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return path


def export_mesh_usd(items, path: str, *, tolerance: float = 0.01) -> str:
    """Write a USD file of tessellated ``UsdGeom.Mesh`` prims (for viewers without BrepArray support)."""
    tmp_dir = tempfile.mkdtemp(prefix="keel-")
    try:
        src = export(items, os.path.join(tmp_dir, "brep.usda"))
        usd_call("mesh export", ub.usd.tessellate_file, src, path, chord_height_tolerance=float(tolerance))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return path


def load(path: str) -> list[Part]:
    """Read every BRep in a USD file as Parts (labels come from prim names in .usda files)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    names = []
    if path.lower().endswith(".usda"):
        with open(path) as f:
            names = re.findall(r'def BrepArray "([^"]+)"', f.read())
    parts = []
    if names:
        try:
            for name in names:
                parts.append(Part(usd_call("load", ub.usd.import_brep, path, f"/World/{name}"), label=name))
            return parts
        except KeelError:
            parts = []  # not keel's /World/<name> layout; fall back to unnamed import
    return [Part(b) for b in usd_call("load", ub.usd.import_breps, path)]


# -- .usda decoration -------------------------------------------------------------------------


def _prim_name(label: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", label)
    return name if name and not name[0].isdigit() else f"_{name}"


def _decorate_usda(text: str, parts: list[Part]) -> str:
    lines = text.split("\n")
    if lines and lines[0].startswith("#usda") and not any(l.lstrip().startswith("(") for l in lines[1:3]):
        lines[1:1] = ["(", '    defaultPrim = "World"', "    metersPerUnit = 0.001", '    upAxis = "Z"', ")"]
    out, used, k = [], set(), 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'(\s*)def BrepArray "([^"]+)"', line)
        if m and k < len(parts):
            part = parts[k]
            k += 1
            name = _prim_name(part.label) if part.label else m.group(2)
            base, n = name, 1
            while name in used:
                name, n = f"{base}_{n}", n + 1
            used.add(name)
            out.append(line.replace(f'"{m.group(2)}"', f'"{name}"', 1))
            if part.color is not None:
                # copy through the prim's metadata block up to its opening brace, then add the color
                i += 1
                while i < len(lines) and lines[i].strip() != "{":
                    out.append(lines[i])
                    i += 1
                if i < len(lines):
                    out.append(lines[i])
                    r, gg, b = (float(c) for c in part.color[:3])
                    out.append(f"{m.group(1)}    color3f[] primvars:displayColor = [({r:g}, {gg:g}, {b:g})]")
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)
