"""STEP (AP214) export, written directly from the kernel's exact B-rep.

The kernel has no STEP writer, and its Python API doesn't expose exact surface data. Its USD
export does: typed surfaces, 3D NURBS edge curves and the full face/loop/edge topology with
orientations. This module reads that ``BrepArray`` record and writes the same geometry as
STEP, so no OpenCASCADE is needed to export.

Names survive: ``part.faces()[i]`` is face ``i + 1`` in the file and in OpenCASCADE (so
text-to-cad's ``label.f12``), and ``part.ref(entity)`` / ``part.find("e7")`` translate any face
or edge to and from those names, using the same plan the writer follows. The file has no
timestamp, so the same part always gives the same bytes.
"""

from __future__ import annotations

import ast
import math
import os
import re
import shutil
import tempfile
from types import SimpleNamespace

from . import geom as g
from ._kernel import KeelError

_NO_VERTEX = 9999999  # loop:vertexIndex for a loop made of edges


# -- reading the kernel's BrepArray record ----------------------------------------------------


def _read_breps(usda_text: str) -> list[dict]:
    """Every BrepArray prim's attributes, as {name: python value}."""
    breps = []
    for block in re.split(r'\n\s*def BrepArray "', usda_text)[1:]:
        attrs = {}
        for m in re.finditer(r"^\s*uniform [\w\[\]]+ ([\w:]+) = (.*)$", block, re.M):
            attrs[m.group(1)] = ast.literal_eval(m.group(2))
        breps.append(attrs)
    return breps


class _Nurbs:
    """Consecutive NURBS records under one attribute prefix, read one at a time."""

    def __init__(self, a: dict, prefix: str):
        self.a, self.p = a, prefix
        self.cv = self.knot = self.n = 0

    def next_curve(self):
        a, p = self.a, self.p
        count, order = a[p + "vertexCount"][self.n], a[p + "order"][self.n]
        cvs = a[p + "controlVertices"][self.cv : self.cv + count]
        weights = a[p + "weights"][self.cv : self.cv + count]
        knots = a[p + "knots"][self.knot : self.knot + count + order]
        self.cv, self.knot, self.n = self.cv + count, self.knot + count + order, self.n + 1
        return order - 1, cvs, weights, knots


def _nurbs_point(degree, cvs, weights, knots, t):
    """Point on a (rational) NURBS curve at ``t``, by de Boor in homogeneous coordinates."""
    n = len(cvs) - 1
    span = degree
    while span < n and knots[span + 1] <= t:
        span += 1
    d = [(*g.mul(cvs[j], weights[j]), weights[j]) for j in range(span - degree, span + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            i = span - degree + j
            denom = knots[i + degree - r + 1] - knots[i]
            a = (t - knots[i]) / denom if denom else 0.0
            d[j] = tuple((1 - a) * x + a * y for x, y in zip(d[j - 1], d[j]))
    x, y, z, w = d[degree]
    return (x / w, y / w, z / w)


def _canonical(degree, cvs, weights, knots, t0, t1):
    """("line", origin, direction) or ("circle", center, normal, x_dir, radius) when the edge's
    curve is exactly one, else None. Exact matters: STEP readers measure and recognize holes on
    true lines and circles, and integrate NURBS circles only approximately."""
    scale = max(1.0, max(abs(c) for p in cvs for c in p))
    tol = 1e-9 * scale
    samples = [_nurbs_point(degree, cvs, weights, knots, t0 + (t1 - t0) * k / 16) for k in range(17)]
    start = samples[0]
    far = max(samples, key=lambda p: g.length(g.sub(p, start)))
    span = g.length(g.sub(far, start))
    if span > tol:
        u = g.mul(g.sub(far, start), 1 / span)
        if all(g.length(g.cross(g.sub(p, start), u)) < tol for p in samples):
            return ("line", start, g.norm(g.sub(samples[1], start)))
    if degree != 2:
        return None
    a, b, c = samples[0], samples[5], samples[11]
    normal = g.cross(g.sub(b, a), g.sub(c, a))
    if g.length(normal) < tol:
        return None
    normal = g.norm(normal)
    # circumcenter of a, b, c
    ab, ac = g.sub(b, a), g.sub(c, a)
    w = g.cross(ab, ac)
    center = g.add(a, g.mul(g.add(g.mul(g.cross(w, ab), g.dot(ac, ac)), g.mul(g.cross(ac, w), g.dot(ab, ab))), 1 / (2 * g.dot(w, w))))
    radius = g.length(g.sub(a, center))
    if not all(abs(g.length(g.sub(p, center)) - radius) < tol and abs(g.dot(g.sub(p, center), normal)) < tol for p in samples):
        return None
    return ("circle", center, normal, g.norm(g.sub(a, center)), radius)


# -- writing STEP -----------------------------------------------------------------------------


def _real(x: float) -> str:
    s = repr(float(x)).upper()
    if s == "-0.0":
        return "0."
    mantissa, _, exp = s.partition("E")
    if "." not in mantissa:
        mantissa += "."
    return mantissa + ("E" + exp if exp else "")


def _text(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"


def _knot_runs(knots):
    values, mults = [], []
    for k in knots:
        if values and abs(k - values[-1]) <= 1e-12 * max(1.0, abs(k)):
            mults[-1] += 1
        else:
            values.append(k)
            mults.append(1)
    return values, mults


def _rational(weights) -> bool:
    return any(abs(w - 1.0) > 1e-15 for w in weights)


class _Writer:
    def __init__(self):
        self.lines: list[str] = []

    def add(self, body: str) -> str:
        self.lines.append(body)
        return f"#{len(self.lines)}"

    def point(self, p) -> str:
        return self.add(f"CARTESIAN_POINT('',({','.join(_real(c) for c in p)}))")

    def direction(self, d) -> str:
        return self.add(f"DIRECTION('',({','.join(_real(c) for c in d)}))")

    def placement(self, origin, axis, ref) -> str:
        return self.add(f"AXIS2_PLACEMENT_3D('',{self.point(origin)},{self.direction(axis)},{self.direction(ref)})")

    def refs(self, items) -> str:
        return "(" + ",".join(items) + ")"

    def curve(self, degree, cvs, weights, knots, t0, t1) -> str:
        shape = _canonical(degree, cvs, weights, knots, t0, t1)
        if shape and shape[0] == "line":
            _, origin, direction = shape
            vector = self.add(f"VECTOR('',{self.direction(direction)},1.)")
            return self.add(f"LINE('',{self.point(origin)},{vector})")
        if shape:
            _, center, normal, x_dir, radius = shape
            return self.add(f"CIRCLE('',{self.placement(center, normal, x_dir)},{_real(radius)})")
        pts = self.refs(self.point(p) for p in cvs)
        values, mults = _knot_runs(knots)
        closed = ".T." if all(abs(a - b) < 1e-9 for a, b in zip(cvs[0], cvs[-1])) else ".F."
        kn = f"({','.join(map(str, mults))}),({','.join(_real(k) for k in values)}),.UNSPECIFIED."
        if not _rational(weights):
            return self.add(f"B_SPLINE_CURVE_WITH_KNOTS('',{degree},{pts},.UNSPECIFIED.,{closed},.F.,{kn})")
        return self.add(
            f"(BOUNDED_CURVE() B_SPLINE_CURVE({degree},{pts},.UNSPECIFIED.,{closed},.F.) "
            f"B_SPLINE_CURVE_WITH_KNOTS({kn}) CURVE() GEOMETRIC_REPRESENTATION_ITEM() "
            f"RATIONAL_B_SPLINE_CURVE(({','.join(_real(w) for w in weights)})) REPRESENTATION_ITEM(''))"
        )


def _surface(w: _Writer, a: dict, kind: str, k: int, nurbs: dict) -> str:
    def v(name):
        return a[f"brep:surface:{kind}:{name}"][k]

    if kind == "plane":
        return w.add(f"PLANE('',{w.placement(v('origin'), v('axis'), v('refDirection'))})")
    if kind == "cylinder":
        return w.add(f"CYLINDRICAL_SURFACE('',{w.placement(v('origin'), v('axis'), v('refDirection'))},{_real(v('radius'))})")
    if kind == "cone":
        angle, axis = v("semiAngle"), v("axis")
        if abs(angle) < 1e-12:  # the kernel stores some cylinders as zero-angle cones
            return w.add(f"CYLINDRICAL_SURFACE('',{w.placement(v('origin'), axis, v('refDirection'))},{_real(v('radius'))})")
        if angle < 0:  # STEP needs a positive half-angle: flip the axis (normals still point outward)
            angle, axis = -angle, tuple(-c for c in axis)
        place = w.placement(v("origin"), axis, v("refDirection"))
        return w.add(f"CONICAL_SURFACE('',{place},{_real(v('radius'))},{_real(angle)})")
    if kind == "sphere":
        return w.add(f"SPHERICAL_SURFACE('',{w.placement(v('center'), v('axis'), v('refDirection'))},{_real(v('radius'))})")
    if kind == "torus":
        place = w.placement(v("origin"), v("axis"), v("refDirection"))
        return w.add(f"TOROIDAL_SURFACE('',{place},{_real(v('majorRadius'))},{_real(v('minorRadius'))})")
    if kind == "nurb":
        return _nurbs_surface(w, a, nurbs)
    raise KeelError(f"STEP export: surface type {kind!r} is not supported yet")


def _step_normal(a: dict, kind: str, k: int, p) -> tuple | None:
    """Direction of the STEP surface's normal at point ``p`` (None for NURBS, which keep the
    kernel's parametrization and so its normal). STEP fixes these: planes along their axis,
    cylinders and cones away from the axis, spheres away from the center, tori away from the
    tube's center circle. The kernel's own canonical surfaces may point either way."""

    def v(name):
        return a[f"brep:surface:{kind}:{name}"][k]

    if kind == "plane":
        return tuple(v("axis"))
    if kind in ("cylinder", "cone", "torus"):
        axis = g.norm(v("axis"))
        d = g.sub(p, v("origin"))
        radial = g.sub(d, g.mul(axis, g.dot(d, axis)))
        if kind != "torus":
            return radial
        return g.sub(radial, g.mul(g.norm(radial), v("majorRadius")))
    if kind == "sphere":
        return g.sub(p, v("center"))
    return None


def _nurbs_surface(w: _Writer, a: dict, state: dict) -> str:
    p, n = "brep:surface:nurb:", state["n"]
    nu, nv = a[p + "uVertexCount"][n], a[p + "vVertexCount"][n]
    uo, vo = a[p + "uOrder"][n], a[p + "vOrder"][n]
    cvs = a[p + "controlVertices"][state["cv"] : state["cv"] + nu * nv]
    weights = a[p + "weights"][state["cv"] : state["cv"] + nu * nv]
    uk = a[p + "uKnots"][state["uk"] : state["uk"] + nu + uo]
    vk = a[p + "vKnots"][state["vk"] : state["vk"] + nv + vo]
    state.update(n=n + 1, cv=state["cv"] + nu * nv, uk=state["uk"] + nu + uo, vk=state["vk"] + nv + vo)
    # The kernel lists control vertices in rows of constant u (v varies fastest), as STEP does.
    grid = w.refs(w.refs(w.point(cvs[i * nv + j]) for j in range(nv)) for i in range(nu))
    (uvals, umult), (vvals, vmult) = _knot_runs(uk), _knot_runs(vk)
    kn = (
        f"({','.join(map(str, umult))}),({','.join(map(str, vmult))}),"
        f"({','.join(_real(x) for x in uvals)}),({','.join(_real(x) for x in vvals)}),.UNSPECIFIED."
    )
    if not _rational(weights):
        return w.add(f"B_SPLINE_SURFACE_WITH_KNOTS('',{uo - 1},{vo - 1},{grid},.UNSPECIFIED.,.F.,.F.,.F.,{kn})")
    wgrid = "(" + ",".join("(" + ",".join(_real(weights[i * nv + j]) for j in range(nv)) + ")" for i in range(nu)) + ")"
    return w.add(
        f"(BOUNDED_SURFACE() B_SPLINE_SURFACE({uo - 1},{vo - 1},{grid},.UNSPECIFIED.,.F.,.F.,.F.) "
        f"B_SPLINE_SURFACE_WITH_KNOTS({kn}) GEOMETRIC_REPRESENTATION_ITEM() "
        f"RATIONAL_B_SPLINE_SURFACE({wgrid}) REPRESENTATION_ITEM('') SURFACE())"
    )


def _edge_map(a: dict, curves: list, brep) -> dict:
    """The kernel edge each USD edge is: same end points, and the USD curve's midpoint on it."""
    points = a.get("brep:vertexPoint:point:position", [])
    kernel = brep.edges()
    scale = max([1.0] + [abs(c) for p in points for c in p])
    tol = 1e-6 * scale
    ends = [(tuple(e.start_vertex().point()), tuple(e.end_vertex().point())) for e in kernel]
    found = {}
    for k, ((s, e), (t0, t1, curve)) in enumerate(zip(a["edge:vertexIndices"], curves)):
        mid = _nurbs_point(*curve, 0.5 * (t0 + t1))
        best = None
        for i, (ks, ke) in enumerate(ends):
            if g.length(g.sub(ks, points[s])) < tol and g.length(g.sub(ke, points[e])) < tol:
                d = kernel[i].closest_point(mid)[2]
                if d < tol and (best is None or d < best[1]):
                    best = (i, d)
        if best is None or best[0] in found.values():
            raise KeelError("STEP export: couldn't match the kernel's edges to its exported curves")
        found[k] = best[0]
    return found


def _plan(a: dict, brep, face_order: list[int]) -> SimpleNamespace:
    """How one brep goes into STEP: which USD record items are which kernel faces and edges,
    each face's loops and sense as written, and the edge numbers a STEP reader will give.

    Faces go in ``face_order``; loops and edge uses in the kernel's ``boundary_loops()`` order,
    except that a loop is written reversed when the kernel's surface normal is opposite to the
    STEP surface's (readers reverse such loops themselves, which renumbers edges).
    """
    plan = SimpleNamespace(a=a)
    points = a.get("brep:vertexPoint:point:position", [])
    nurbs_curves = _Nurbs(a, "brep:edge3dNurb:curve3d:nurb:")
    ranges = a["edge:range"]
    plan.curves = []
    for k, kind in enumerate(a["edge:curveType"]):
        if kind != "BrepCurve3dNurbAPI":
            raise KeelError(f"STEP export: edge curve type {kind!r} is not supported yet")
        plan.curves.append((ranges[2 * k], ranges[2 * k + 1], nurbs_curves.next_curve()))
    kernel_edge_of = _edge_map(a, plan.curves, brep)
    plan.usd_edge_of = {kernel: usd for usd, kernel in kernel_edge_of.items()}

    # USD faces: their surface kind, plus the (kernel edge, sense) uses that identify them.
    plan.kinds = [t[len("BrepSurface") : -len("API")].lower() for t in a["face:surfaceType"]]
    seen: dict[str, int] = {}
    loop = eu = 0
    plan.kind_index, plan.vertex_loops, signature_to_face = [], [], {}
    for f, (kind, n_loops) in enumerate(zip(plan.kinds, a["face:loopCount"])):
        plan.kind_index.append((kind, seen.get(kind, 0)))
        seen[kind] = seen.get(kind, 0) + 1
        uses, vertex_loops = [], []
        for _ in range(n_loops):
            count, vertex = a["loop:edgeuseCount"][loop], a["loop:vertexIndex"][loop]
            if vertex != _NO_VERTEX:
                vertex_loops.append(vertex)
            else:
                uses += [
                    (kernel_edge_of[a["edgeuse:edgeIndex"][i]], a["edgeuse:orientationType"][i] == "same")
                    for i in range(eu, eu + count)
                ]
                eu += count
            loop += 1
        plan.vertex_loops.append(vertex_loops)
        signature_to_face[tuple(sorted(uses))] = f

    # Regions own shells, shells own face uses.
    shell_of, plan.shell_count, shell, fu = {}, 0, 0, 0
    for region, n_shells in zip(a["region:type"], a["region:shellCount"]):
        for s in range(n_shells):
            n = a["shell:faceuseCount"][shell]
            if region == "solidRegion":
                if s > 0:
                    raise KeelError("STEP export: solids with internal voids are not supported yet")
                for f in a["faceuse:faceIndex"][fu : fu + n]:
                    shell_of[f] = plan.shell_count
                plan.shell_count += 1
            shell, fu = shell + 1, fu + n

    kernel_edges = brep.edges()
    kernel_index = {e: i for i, e in enumerate(kernel_edges)}
    kernel_faces = brep.faces()
    plan.faces = []
    for kf in face_order:
        face = kernel_faces[kf]
        loops = [[(kernel_index[e], bool(sense)) for e, sense in lp] for lp in face.boundary_loops()]
        f = signature_to_face.get(tuple(sorted(u for lp in loops for u in lp)))
        if f is None or f not in shell_of:
            raise KeelError("STEP export: couldn't match the kernel's faces to its exported surfaces")
        # The kernel runs loops around its surface's normal; STEP's canonical surfaces have
        # fixed normals (see _step_normal) that can point the other way.
        surface = face.surface()
        (u0, v0), (u1, v1) = surface.uv_domain()
        u, v = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        outward = tuple(face.outward_normal_at_uv(u, v))
        kernel_normal = tuple(surface.normal(u, v))
        step_normal = _step_normal(a, *plan.kind_index[f], tuple(surface.evaluate(u, v))) or kernel_normal
        if g.dot(kernel_normal, step_normal) < 0:
            loops = [[(i, not sense) for i, sense in reversed(lp)] for lp in loops]
        plan.faces.append({
            "kernel": kf,
            "usd": f,
            "shell": shell_of[f],
            "loops": loops,
            "sense": ".T." if g.dot(step_normal, outward) > 0 else ".F.",  # face and loop flags alike
        })

    # Edge numbers: readers number edges in the order faces, loops and uses first reach them,
    # and OCCT adds a zero-length edge before every use that starts at a sphere's pole.
    plan.edge_names = []
    seen_edges = set()
    for face in plan.faces:
        poles = _poles(a, *plan.kind_index[face["usd"]])
        for lp in face["loops"]:
            for i, sense in lp:
                e = kernel_edges[i]
                start = tuple((e.start_vertex() if sense else e.end_vertex()).point())
                if any(g.length(g.sub(start, p)) < 1e-6 * max(1.0, g.length(p)) for p in poles):
                    plan.edge_names.append(None)
                if i not in seen_edges:
                    seen_edges.add(i)
                    plan.edge_names.append(i)
    return plan


def _poles(a: dict, kind: str, k: int) -> list:
    """Points where the surface collapses (a sphere's poles, a NURBS side shrunk to a point).
    STEP readers put a zero-length edge there."""
    if kind == "sphere":
        center, r = a["brep:surface:sphere:center"][k], a["brep:surface:sphere:radius"][k]
        axis = g.norm(a["brep:surface:sphere:axis"][k])
        return [g.add(center, g.mul(axis, r)), g.sub(center, g.mul(axis, r))]
    if kind != "nurb":
        return []
    p = "brep:surface:nurb:"
    start = sum(u * v for u, v in zip(a[p + "uVertexCount"][:k], a[p + "vVertexCount"][:k]))
    nu, nv = a[p + "uVertexCount"][k], a[p + "vVertexCount"][k]
    cvs = a[p + "controlVertices"][start : start + nu * nv]  # rows of constant u
    sides = [
        [cvs[j] for j in range(nv)],
        [cvs[(nu - 1) * nv + j] for j in range(nv)],
        [cvs[i * nv] for i in range(nu)],
        [cvs[i * nv + nv - 1] for i in range(nu)],
    ]
    scale = max(1.0, max(abs(c) for q in cvs for c in q))
    return [side[0] for side in sides if all(g.length(g.sub(q, side[0])) < 1e-9 * scale for q in side)]


def _solids(w: _Writer, plan: SimpleNamespace) -> list[str]:
    """Write one planned brep's solids; return their MANIFOLD_SOLID_BREP ids."""
    a = plan.a
    vertices = [w.add(f"VERTEX_POINT('',{w.point(p)})") for p in a.get("brep:vertexPoint:point:position", [])]
    edges = []
    for (s, e), (t0, t1, curve) in zip(a["edge:vertexIndices"], plan.curves):
        edges.append(w.add(f"EDGE_CURVE('',{vertices[s]},{vertices[e]},{w.curve(*curve, t0, t1)},.T.)"))
    nurbs = {"n": 0, "cv": 0, "uk": 0, "vk": 0}
    surfaces = [_surface(w, a, kind, k, nurbs) for kind, k in plan.kind_index]

    shells: list[list[str]] = [[] for _ in range(plan.shell_count)]
    for face in plan.faces:
        sense, bounds = face["sense"], []
        for lp in face["loops"]:
            oriented = [w.add(f"ORIENTED_EDGE('',*,*,{edges[plan.usd_edge_of[i]]},{'.T.' if s else '.F.'})") for i, s in lp]
            edge_loop = w.add(f"EDGE_LOOP('',{w.refs(oriented)})")
            bounds.append(w.add(f"FACE_BOUND('',{edge_loop},{sense})"))
        for vertex in plan.vertex_loops[face["usd"]]:
            vertex_loop = w.add(f"VERTEX_LOOP('',{vertices[vertex]})")
            bounds.append(w.add(f"FACE_BOUND('',{vertex_loop},{sense})"))
        shells[face["shell"]].append(w.add(f"ADVANCED_FACE('',{w.refs(bounds)},{surfaces[face['usd']]},{sense})"))

    solids = []
    for faces in shells:
        closed_shell = w.add(f"CLOSED_SHELL('',{w.refs(faces)})")
        solids.append(w.add(f"MANIFOLD_SOLID_BREP('',{closed_shell})"))
    return solids


def step_plans(part) -> list[SimpleNamespace]:
    """The part's STEP plans (one per brep), computed once and cached on the part."""
    from .io import export

    if getattr(part, "_step_plans", None) is None:
        tmp_dir = tempfile.mkdtemp(prefix="keel-")
        try:
            usda = export(part, os.path.join(tmp_dir, "part.usda"), uv_curves=False)
            with open(usda) as f:
                breps = _read_breps(f.read())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        part._step_plans = [_plan(a, part.brep, part._step_order()) for a in breps]
    return part._step_plans


def step_names(part) -> tuple[list[int], list[int | None]]:
    """Kernel face and edge indices in the order STEP readers number them (f1, e1, ...).

    Edge entries are None where OCCT inserts a zero-length edge of its own (sphere poles).
    """
    (plan,) = step_plans(part)
    return part._face_order(), plan.edge_names


def write_step(parts, path: str) -> str:
    """Write one Part as an AP214 STEP file, named from the part's label."""
    if len(parts) != 1:
        raise ValueError("STEP export writes one part per file for now")
    part = parts[0]
    name = part.label or os.path.splitext(os.path.basename(path))[0]
    w = _Writer()
    ctx_app = w.add("APPLICATION_CONTEXT('automotive design')")
    w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{ctx_app})")
    product_ctx = w.add(f"PRODUCT_CONTEXT('',{ctx_app},'mechanical')")
    product = w.add(f"PRODUCT({_text(name)},{_text(name)},'',({product_ctx}))")
    formation = w.add(f"PRODUCT_DEFINITION_FORMATION('','',{product})")
    pd_ctx = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{ctx_app},'design')")
    definition = w.add(f"PRODUCT_DEFINITION('design','',{formation},{pd_ctx})")
    pd_shape = w.add(f"PRODUCT_DEFINITION_SHAPE('','',{definition})")
    length = w.add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))")
    angle = w.add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))")
    solid_angle = w.add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())")
    tol = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),{length},'distance_accuracy_value','')")
    rep_ctx = w.add(
        f"(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({tol})) "
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({length},{angle},{solid_angle})) REPRESENTATION_CONTEXT('',''))"
    )
    solids = [s for plan in step_plans(part) for s in _solids(w, plan)]
    if not solids:
        raise KeelError("STEP export: the part has no solid")
    origin = w.placement((0, 0, 0), (0, 0, 1), (1, 0, 0))
    rep = w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION({_text(name)},{w.refs([origin] + solids)},{rep_ctx})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION({pd_shape},{rep})")

    header = (
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('keel'),'2;1');\n"
        f"FILE_NAME({_text(name)},'',(''),(''),'keel','keel','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n"
    )
    with open(path, "w") as f:
        f.write(header)
        f.writelines(f"#{i}={body};\n" for i, body in enumerate(w.lines, start=1))
        f.write("ENDSEC;\nEND-ISO-10303-21;\n")
    return path
