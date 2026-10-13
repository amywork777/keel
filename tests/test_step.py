"""STEP export, checked by reading the file back with OpenCASCADE (what text-to-cad uses).

The kernel's own volume is the ground truth. text-to-cad names faces by their position in
OCCT's face map (``label.f12``), so face order has to match keel's and stay the same run to run.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from keel import Box, Cone, Cylinder, KeelError, Sphere, Torus

HERE = Path(__file__).resolve().parent
OCCT_PYTHON = os.environ.get("KEEL_OCCT_PYTHON", str(HERE.parent / ".venv-ref" / "bin" / "python"))

pytestmark = pytest.mark.skipif(
    not os.path.exists(OCCT_PYTHON), reason="needs a Python with OCP to read STEP back (.venv-ref)"
)


def occt(path):
    out = subprocess.run([OCCT_PYTHON, str(HERE / "occt_check.py"), str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def plate_with_hole():
    return Box(40, 30, 6) - Cylinder(5, 6)


PARTS = {
    "box": lambda: Box(10, 20, 30),
    "cylinder": lambda: Cylinder(5, 10),
    "cone": lambda: Cone(4, 2, 10),
    "sphere": lambda: Sphere(5),
    "torus": lambda: Torus(10, 2),
    "plate_with_hole": plate_with_hole,
    "tube": lambda: Cylinder(15, 28) - Cylinder(5, 30),
    "dome": lambda: Cylinder(10, 20) + Sphere(10).move(0, 0, 10),
    "filleted_box": lambda: Box(20, 20, 10).fillet("|Z", 3),
    "filleted_cylinder": lambda: Cylinder(15, 28).fillet("%circle", 1.1),  # rational NURBS fillet faces
}


@pytest.mark.parametrize("name", PARTS)
def test_step_reads_back_as_one_valid_solid_with_the_kernel_volume(name, tmp_path):
    part = PARTS[name]()
    got = occt(part.export(str(tmp_path / f"{name}.step")))
    assert got["read"] and got["valid"] and got["solids"] == 1
    assert got["volume"] == pytest.approx(part.volume, rel=1e-4)


def near(a, b, tol):
    return all(abs(x - y) < tol for x, y in zip(a, b))


@pytest.mark.parametrize("name", PARTS)
def test_step_faces_come_back_in_keel_order(name, tmp_path):
    part = PARTS[name]()
    got = occt(part.export(str(tmp_path / f"{name}.step")))
    tol = 2e-3 * max(part.bbox.size)  # keel's face centroids are good to ~1e-3 of the part size
    assert len(got["faces"]) == len(part.faces())
    for i, (mine, theirs) in enumerate(zip(part.faces(), got["faces"]), start=1):
        assert near(mine.center, theirs["center"], tol), f"f{i}"
        assert theirs["area"] == pytest.approx(mine.area, rel=2e-3), f"f{i} area"  # keel's face areas are ~1e-3


def written_faces(path):
    """(surface entity, same_sense) per ADVANCED_FACE, in shell order, read from the file text."""
    text = Path(path).read_text()
    ent = dict(re.findall(r"^(#\d+)=(.*);$", text, re.M))
    shell = re.search(r"CLOSED_SHELL\('',\(([^)]*)\)\)", text).group(1).split(",")
    out = []
    for ref in shell:
        surface, sense = re.search(r",(#\d+),\.([TF])\.\)$", ent[ref]).groups()
        out.append((ent, ent[surface], sense == "T"))
    return out


def declared_normal(ent, surface, point):
    """The outward normal a STEP plane or cylinder face declares at ``point`` (None otherwise)."""
    def vec(ref):
        return [float(x) for x in re.search(r"\(([^()]*)\)\)$", ent[ref]).group(1).split(",")]

    kind = surface.split("(")[0]
    if kind not in ("PLANE", "CYLINDRICAL_SURFACE"):
        return None
    origin, axis, _ = (vec(r) for r in re.findall(r"#\d+", ent[re.search(r"#\d+", surface).group(0)]))
    if kind == "PLANE":
        return axis
    d = [p - o for p, o in zip(point, origin)]
    along = sum(a * b for a, b in zip(d, axis))
    return [x - along * a for x, a in zip(d, axis)]


@pytest.mark.parametrize("name", PARTS)
def test_step_faces_are_written_facing_outward(name, tmp_path):
    """Checked on the file itself: OCCT silently repairs flipped faces on read (and reorders them)."""
    part = PARTS[name]()
    faces = written_faces(part.export(str(tmp_path / f"{name}.step")))
    for i, (mine, (ent, surface, same_sense)) in enumerate(zip(part.faces(), faces), start=1):
        point = mine._h.closest_point(mine.center)[0]
        normal = declared_normal(ent, surface, point)
        if normal is not None:
            facing = sum(a * b for a, b in zip(normal, mine.normal_at(point)))
            assert (facing > 0) == same_sense, f"f{i} {surface.split('(')[0]}"


def check_refs(part, got):
    """part.find("f<j>") / ("e<j>") is OCCT's face j / edge j, the way text-to-cad numbers them."""
    tol = 2e-3 * max(part.bbox.size)
    for j, theirs in enumerate(got["faces"], start=1):
        assert near(part.find(f"f{j}").center, theirs["center"], tol), f"f{j}"
    for j, theirs in enumerate(got["edges"], start=1):
        if theirs["degenerated"]:  # OCCT's zero-length edge at a sphere pole: keel has no edge there
            with pytest.raises(KeelError):
                part.find(f"e{j}")
            continue
        edge = part.find(f"e{j}")
        assert edge._h.closest_point(tuple(theirs["mid"]))[2] < 1e-4 * max(part.bbox.size), f"e{j}"
        assert part.ref(edge) == f"e{j}"


@pytest.mark.parametrize("name", PARTS)
def test_step_refs_name_faces_and_edges_like_occt(name, tmp_path):
    part = PARTS[name]()
    check_refs(part, occt(part.export(str(tmp_path / f"{name}.step"))))


def test_step_refs_use_the_part_label():
    part = (Box(40, 30, 6) - Cylinder(5, 6)).named("mounting_plate")
    face = part.find("mounting_plate.f2")
    assert part.ref(face) == "mounting_plate.f2"
    assert part.find("#mounting_plate.f2") == face  # text-to-cad's selector form


def test_step_writes_lines_and_circles_as_lines_and_circles(tmp_path):
    got = occt((Box(40, 30, 6) - Cylinder(5, 6)).export(str(tmp_path / "plate.step")))
    types = sorted(e["type"] for e in got["edges"])
    assert types == ["GeomAbs_Circle"] * 2 + ["GeomAbs_Line"] * 13


def test_step_export_is_byte_identical_run_to_run(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    a = plate_with_hole().export(str(tmp_path / "a" / "plate.step"))
    b = plate_with_hole().export(str(tmp_path / "b" / "plate.step"))
    assert Path(a).read_bytes() == Path(b).read_bytes()


def test_step_product_is_named_from_the_part_label(tmp_path):
    path = Box(10, 10, 10).named("mounting_plate").export(str(tmp_path / "p.step"))
    assert "PRODUCT('mounting_plate'" in Path(path).read_text()
