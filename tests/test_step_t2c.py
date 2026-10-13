"""STEP export of real text-to-cad example parts, built with keel.build123d.

These are the 17 single-part examples from text-to-cad's models/examples that the 2026-10-12
feasibility study pushed through NVIDIA's usd_to_occt (2 of 17 came out right). Each must
read back in OpenCASCADE as a valid solid with the kernel's volume and keel's faces and edges.

Needs a text-to-cad checkout at .ref/t2c-feasibility/t2c-main (or KEEL_T2C_EXAMPLES) and the
OCP venv used by test_step.py; skipped otherwise.
"""

import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

from test_step import OCCT_PYTHON, check_refs, declared_normal, near, occt, written_faces

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(os.environ.get("KEEL_T2C_EXAMPLES", ROOT / ".ref/t2c-feasibility/t2c-main/models/examples/src"))
MODELS = [
    "cam_follower_roller", "centrifugal_impeller", "circular_flange", "cylindrical_spacer_sleeve",
    "electronics_enclosure_base", "flywheel_disk", "gusset_plate", "keyed_shaft_hub", "l_bracket",
    "mounting_plate", "pulley_wheel", "radial_engine_cylinder", "rectangular_calibration_block",
    "retainer_plate", "shaft_collar", "spur_gear_blank", "stepped_shaft_keyway",
]

pytestmark = pytest.mark.skipif(
    not EXAMPLES.is_dir() or not os.path.exists(OCCT_PYTHON), reason="needs text-to-cad's examples and OCP"
)


@pytest.fixture(scope="module")
def cadgen():
    sys.path.insert(0, str(ROOT / "compat"))
    import t2c_harness

    standin = tempfile.mkdtemp(prefix="t2c-cadgen-")
    t2c_harness._write_standin(Path(standin))
    os.environ["T2C_BACKEND"] = "keel"
    sys.path[:0] = [standin, str(EXAMPLES)]
    import cadgen

    return cadgen


@pytest.mark.parametrize("name", MODELS)
def test_t2c_example_exports_to_step(name, cadgen, tmp_path):
    importlib.import_module(name)
    (part,) = next(fn for _, n, fn in cadgen._REGISTRY if n == name)()._parts
    path = part.export(str(tmp_path / f"{name}.step"))
    got = occt(path)
    assert got["valid"] and got["solids"] == 1
    assert got["volume"] == pytest.approx(part.volume, rel=1e-4)
    assert [near(f.center, g["center"], 2e-3 * max(part.bbox.size)) for f, g in zip(part.faces(), got["faces"])] == [True] * len(part.faces())
    assert len(got["faces"]) == len(part.faces())
    assert sum(not e["degenerated"] for e in got["edges"]) == len(part.edges(seams=True))
    for i, (mine, (ent, surface, same_sense)) in enumerate(zip(part.faces(), written_faces(path)), start=1):
        point = mine._h.closest_point(mine.center)[0]
        normal = declared_normal(ent, surface, point)
        if normal is not None:
            assert (sum(a * b for a, b in zip(normal, mine.normal_at(point))) > 0) == same_sense, f"f{i} faces inward"


@pytest.mark.parametrize("name", MODELS)
def test_t2c_example_refs_name_faces_and_edges_like_occt(name, cadgen, tmp_path):
    importlib.import_module(name)
    (part,) = next(fn for _, n, fn in cadgen._REGISTRY if n == name)()._parts
    check_refs(part, occt(part.export(str(tmp_path / f"{name}.step"))))
