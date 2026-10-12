import math
import os

import pytest

from keel import *

PI = math.pi


def close(a, b, rel=1e-4):
    return a == pytest.approx(b, rel=rel)


def close_pt(a, b, tol=1e-3):
    return all(abs(x - y) < tol for x, y in zip(a, b))


# -- 3D primitives ----------------------------------------------------------------------------


def test_primitives_volumes_and_centering():
    assert close(Box(10, 20, 30).volume, 6000)
    assert close(Cylinder(3, 10).volume, PI * 9 * 10)
    assert close(Cone(4, 2, 10).volume, PI * 10 / 3 * (16 + 8 + 4))
    assert close(Sphere(5).volume, 4 / 3 * PI * 125)
    assert close(Torus(10, 2).volume, 2 * PI * PI * 10 * 4)
    for p in (Box(10, 20, 30), Cylinder(3, 10), Sphere(5), Torus(10, 2)):
        assert close_pt(p.bbox.center, (0, 0, 0))
        assert p.is_valid


def test_align():
    b = Box(10, 10, 4, align=("center", "center", "min"))
    assert close_pt(b.bbox.min, (-5, -5, 0))
    c = Cylinder(2, 6, align="min")
    assert close_pt(c.bbox.min, (0, 0, 0))


# -- booleans and placement -------------------------------------------------------------------


def test_booleans_do_not_consume_operands():
    a, b = Box(10, 10, 10), Sphere(4).move(5, 5, 5)
    va, vb = a.volume, b.volume
    u, d, i = a + b, a - b, a & b
    assert close(a.volume, va) and close(b.volume, vb)  # operands untouched
    assert close(u.volume + i.volume, va + vb)
    assert close(d.volume, va - i.volume)


def test_cut_many_with_pattern():
    plate = Box(60, 40, 6)
    holes = Cylinder(3, 6).at(grid(2, 2, 46, 26))
    assert len(holes) == 4
    out = plate - holes
    assert close(out.volume, 60 * 40 * 6 - 4 * PI * 9 * 6)
    assert out.is_valid


def test_move_rotate_mirror_scale():
    b = Box(10, 2, 2, align="min")
    assert close_pt(b.move(1, 2, 3).bbox.min, (1, 2, 3))
    r = b.rotate("Z", 90)
    assert close_pt(r.bbox.min, (-2, 0, 0)) and close_pt(r.bbox.max, (0, 10, 2))
    m = b.mirror("YZ")
    assert close_pt(m.bbox.min, (-10, 0, 0)) and close(m.volume, 40)
    s = b.scale(2)
    assert close(s.volume, 320)


def test_sum_fuses():
    parts = [Box(2, 2, 2).move(x, 0, 0) for x in (0, 1.5, 3)]
    assert close(sum(parts).volume, 2 * 2 * 2 + 1.5 * 4 * 2)


# -- selectors --------------------------------------------------------------------------------


def test_face_selectors():
    b = Box(10, 20, 30)
    top = b.faces(">Z")
    assert len(top) == 1 and close_pt(top[0].normal, (0, 0, 1))
    assert len(b.faces("<X")) == 1
    assert len(b.faces("|Z")) == 2
    assert len(b.faces("#Z")) == 4
    assert len(b.faces("+Y")) == 1
    assert b.faces().largest()[0].area == pytest.approx(600)


def test_edge_selectors():
    b = Box(10, 20, 30)
    assert len(b.edges("|Z")) == 4
    assert len(b.edges(">Z")) == 4
    assert len(b.edges(">Z and |X")) == 2
    assert len(b.faces(">Z").edges()) == 4
    c = Cylinder(3, 10)
    assert len(c.edges("%circle")) == 2
    assert c.edges("%circle and >Z")[0].radius == pytest.approx(3)


# -- features ---------------------------------------------------------------------------------


def test_fillet_selected_edges():
    b = Box(10, 10, 10)
    f = b.fillet("|Z", 1)
    # each vertical edge loses (1 - pi/4) * r^2 * length
    assert close(f.volume, 1000 - 4 * (1 - PI / 4) * 10)
    assert f.is_valid and len(f.faces()) == 10


def test_fillet_with_selection_object():
    b = Box(10, 10, 10)
    f = b.fillet(b.edges(">Z"), 2)
    assert len(f.faces()) == 10 and f.is_valid


def test_chamfer():
    b = Box(10, 10, 10)
    c = b.chamfer(b.edges("|Z").first, 1)
    assert close(c.volume, 1000 - 0.5 * 1 * 1 * 10, rel=1e-3)


def test_shell_open_top():
    b = Box(10, 10, 10)
    s = b.shell(1, open=">Z")
    assert close(s.volume, 1000 - 8 * 8 * 9)
    closed = b.shell(1)
    assert close(closed.volume, 1000 - 8 * 8 * 8)


# -- sketches ---------------------------------------------------------------------------------


def test_sketch_booleans_and_extrude_with_hole():
    s = Rect(20, 10) - Circle(2).move(3, 0)
    assert close(s.area, 200 - PI * 4)
    p = s.extrude(5)
    assert close(p.volume, (200 - PI * 4) * 5)
    assert close_pt(p.bbox.min, (-10, -5, 0), 1e-3)
    assert p.is_valid


def test_extrude_both_and_negative():
    assert close_pt(Rect(2, 2).extrude(3, both=True).bbox.min, (-1, -1, -3))
    assert close_pt(Rect(2, 2).extrude(-3).bbox.min, (-1, -1, -3))


def test_sketch_on_planes():
    p = Rect(4, 2).on("XZ").extrude(5)  # XZ normal is -Y; local y is world Z
    assert close_pt(p.bbox.min, (-2, -5, -1)) and close_pt(p.bbox.max, (2, 0, 1))
    q = Rect(4, 2).on(Plane.at((0, 0, 10), "X")).extrude(1)
    assert close_pt(q.bbox.center, (0.5, 0, 10))


def test_sketch_on_face():
    base = Box(20, 20, 4, align=("center", "center", "min"))
    boss = Circle(3).on(base.faces(">Z")).extrude(6)
    assert close_pt(boss.bbox.min, (-3, -3, 4)) and close_pt(boss.bbox.max, (3, 3, 10))
    pocket = base - Rect(6, 6).on(base.faces(">Z")).extrude(-2)
    assert close(pocket.volume, 1600 - 72)


def test_sketch_primitives():
    assert close(Ellipse(4, 2).area, PI * 8)
    assert close(RegularPolygon(1, 6).area, 3 * math.sqrt(3) / 2)
    assert close(Polygon((0, 0), (4, 0), (0, 3)).area, 6)
    assert close(Slot(10, 2).area, 8 * 2 + PI)
    assert close(RoundedRect(10, 6, 1).area, 60 - (4 - PI))


def test_sketch_union_disjoint_and_at():
    s = Circle(1).at([(0, 0), (5, 0)])
    assert s.region_count == 2 and close(s.area, 2 * PI)
    assert close(s.extrude(1).volume, 2 * PI)


def test_revolve_with_hole():
    ring = (Rect(4, 4) - Circle(1)).move(10, 0)
    p = ring.revolve(360, "Y")
    assert close(p.volume, (16 - PI) * 2 * PI * 10)
    quarter = Rect(2, 2).move(10, 0).revolve(90, "Y")
    assert close(quarter.volume, 4 * 2 * PI * 10 / 4)


def test_taper():
    p = Rect(10, 10).extrude(5, taper=10)
    assert p.volume < 500 and p.is_valid


def test_loft():
    p = loft(Rect(10, 10), Circle(3).on(Plane.XY.offset(10)))
    assert p.is_valid and 10 * PI * 9 < p.volume < 1000


# -- 1D paths ---------------------------------------------------------------------------------


def test_pen_profile():
    s = Pen(0, 0).line_to(10, 0).line(0, 5).line_to(0, 5).close().to_sketch()
    assert close(s.area, 50)
    arc = Pen(0, 0).h(10).tangent_arc_to(10, 10).h(-10).close().to_sketch()
    assert close(arc.area, 100 + PI * 25 / 2)


def test_three_point_arc_and_polyline():
    p = (Polyline((0, 0), (10, 0)) + ThreePointArc((10, 0), (15, 5), (10, 10)) + Polyline((10, 10), (0, 10))).close()
    assert close(p.to_sketch().area, 100 + PI * 25 / 2)


def test_pipe_and_sweep():
    helix = Helix(10, 20, 3)
    tube = helix.pipe(1)
    assert tube.is_valid and tube.volume > 0
    spine = Polyline((0, 0), (20, 0), (20, 20))
    bent = spine.pipe(1)
    assert bent.is_valid and close(bent.volume, PI * 40, rel=0.05)


def test_sweep_profile_center_follows_planar_path():
    p = Rect(2, 1).sweep(Polyline((0, 0), (20, 0), (20, 20)))
    assert close_pt(p.bbox.min, (0, -1, -0.5)) and close_pt(p.bbox.max, (21, 20, 0.5))
    q = Rect(2, 1).sweep(Arc((0, 0), 10, 0, 90))
    assert close_pt(q.bbox.min, (0, 0, -0.5)) and close_pt(q.bbox.max, (11, 11, 0.5))
    assert close(q.volume, 2 * PI * 10 / 4 * 2)


def test_round_profile_on_helix_is_centered():
    p = Circle(1).sweep(Helix(10, 20, 3))
    assert abs(p.bbox.center[0]) < 0.2 and abs(p.bbox.center[1]) < 0.2


def test_noncircular_profile_on_3d_path_warns():
    with pytest.warns(UserWarning):
        Rect(1, 1).sweep(Helix(10, 20, 3))


# -- builder ----------------------------------------------------------------------------------


def test_build_context():
    with Build() as b:
        b += Box(40, 20, 5)
        b -= Cylinder(3, 5).at(grid(2, 1, 25, 0))
        b.fillet("|Z and >X", 2)
    assert b.part.is_valid
    assert close(b.part.volume, 4000 - 2 * PI * 9 * 5 - 2 * (1 - PI / 4) * 4 * 5)


# -- files ------------------------------------------------------------------------------------


def test_usd_roundtrip_with_labels(tmp_path):
    a = Box(10, 10, 10).named("block", color=(0.8, 0.2, 0.2))
    c = Cylinder(2, 5).move(20, 0, 0).named("pin")
    path = export([a, c], str(tmp_path / "out.usda"))
    text = open(path).read()
    assert 'def BrepArray "block"' in text and 'def BrepArray "pin"' in text
    assert "primvars:displayColor" in text and "metersPerUnit = 0.001" in text
    parts = load(path)
    assert [p.label for p in parts] == ["block", "pin"]
    assert close(parts[0].volume, 1000) and close(parts[1].volume, PI * 4 * 5)


def test_usdc_roundtrip(tmp_path):
    path = Sphere(3).export(str(tmp_path / "s.usdc"))
    (p,) = load(path)
    assert close(p.volume, 4 / 3 * PI * 27)


def test_stl_export(tmp_path):
    path = Box(1, 2, 3).export(str(tmp_path / "b.stl"))
    size = os.path.getsize(path)
    assert size == 84 + 50 * ((size - 84) // 50) and (size - 84) // 50 >= 12


def test_mesh_usd(tmp_path):
    path = export_mesh_usd(Box(1, 1, 1), str(tmp_path / "m.usda"))
    assert "def Mesh" in open(path).read()


def test_errors_are_keel_errors():
    with pytest.raises(KeelError):
        Box(10, 10, 10).fillet("|Z", 50)
    with pytest.raises(KeelError):
        Box(1, 1, 1).fillet(Box(5, 5, 5).edges("|Z"), 0.1)


def test_sweep_path_with_clockwise_arc():
    path = Pen(0, 0).v(30).tangent_arc_to(30, 60).h(40).path.on("XZ")
    p = Circle(2).sweep(path)
    assert p.is_valid and close(p.volume, PI * 4 * (30 + 15 * PI + 40))


def test_impossible_boolean_raises_instead_of_dropping_geometry():
    arm = Slot(40, 10).move(15, 0).extrude(6) + Cylinder(8, 6, align=("center", "center", "min"))
    with pytest.raises(KeelError, match="impossible result"):
        arm + arm.mirror("YZ")
    # the same shape built by rotation fuses fine
    assert close((arm + arm.rotate("Z", 180)).volume, 2 * arm.volume - PI * 64 * 6, rel=5e-3)


def test_loft_sections_are_aligned_not_twisted():
    # a straight prism lofted between identical hexagons, one built rotated a full turn of
    # vertices: an untwisted loft has the prism's exact volume
    hexagon = RegularPolygon(10, 6)
    p = loft(hexagon, RegularPolygon(10, 6, rotation=60).on(Plane.XY.offset(20)))
    assert close(p.volume, hexagon.area * 20)
    q = loft(Circle(5), RegularPolygon(5, 4, rotation=45).on(Plane.XY.offset(10)), Circle(5).on(Plane.XY.offset(20)))
    assert q.is_valid


def test_loft_polygon_to_circle_is_not_twisted():
    # hexagon (corner on +X) to circle of the same circumradius. Untwisted, the mid-height
    # slice is widest along X (corner direction); a twisted loft turns that bulge away.
    p = loft(RegularPolygon(10, 6), Circle(10).on(Plane.XY.offset(20)))
    mid = p & Box(40, 40, 0.5).move(0, 0, 10)
    size = mid.bbox.size
    assert size[0] > size[1] + 0.5
