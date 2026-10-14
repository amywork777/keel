"""keel123d follows build123d 0.11's conventions (checked against real build123d)."""

import math

import pytest

import keel123d as bd
from keel import KeelError

PI = math.pi
C, MIN = bd.Align.CENTER, bd.Align.MIN


def close(a, b, rel=1e-4):
    return a == pytest.approx(b, rel=rel)


def bb(shape):
    b = shape.bounding_box()
    return tuple(round(v, 4) for v in (*b.min, *b.max))


def test_rotation_is_rx_ry_rz():
    # values printed by build123d 0.11.1 for Location((0, 0, 0), (30, 40, 50)) * (1, 0, 0) etc.
    rot = bd.Location((0, 0, 0), (30, 40, 50))
    assert [round(c, 4) for c in (rot * bd.Location((1, 0, 0))).position] == [0.4924, 0.87, 0.0252]
    assert [round(c, 4) for c in (rot * bd.Location((0, 1, 0))).position] == [-0.5868, 0.3105, 0.7478]
    assert [round(c, 4) for c in (bd.Pos(5, 0, 0) * bd.Rot(0, 0, 90) * bd.Location((1, 0, 0))).position] == [5, 1, 0]


def test_align_before_rotation():
    b = bd.Box(10, 2, 1, rotation=(0, 0, 90), align=(MIN, MIN, MIN))
    assert bb(b) == (-2, 0, 0, 0, 10, 1)


def test_builder_locations_and_modes():
    with bd.BuildPart() as p:
        bd.Cylinder(40, 10, align=(C, C, MIN))
        with bd.Locations(bd.Location((0, 0, -1))):
            bd.Cylinder(15, 12, align=(C, C, MIN), mode=bd.Mode.SUBTRACT)
        with bd.PolarLocations(30, 6):
            bd.Cylinder(3, 10, align=(C, C, MIN), mode=bd.Mode.SUBTRACT)
    assert close(p.part.volume, PI * (1600 - 225 - 6 * 9) * 10)


def test_sketch_on_plane_and_extrude():
    with bd.BuildPart() as p:
        with bd.BuildSketch(bd.Plane.XZ):
            bd.Polygon((0, 0), (4, 0), (0, 3))  # not centered by default
        bd.extrude(amount=2)
    assert close(p.part.volume, 12)
    assert bb(p.part) == (0, -2, 0, 4, 0, 3)  # XZ extrudes toward -Y


def test_sketch_shapes_center_by_default():
    with bd.BuildPart() as p:
        with bd.BuildSketch():
            bd.RectangleRounded(20, 10, 2)
        bd.extrude(amount=1)
    assert bb(p.part) == (-10, -5, 0, 10, 5, 1)


def test_builder_in_helper_function_is_independent():
    def lug():
        with bd.BuildPart() as inner:
            bd.Box(2, 2, 2)
        return inner.part.moved(bd.Location((10, 0, 0)))

    with bd.BuildPart() as outer:
        bd.Box(2, 2, 2)
        bd.add(lug())
    assert close(outer.part.volume, 16)  # the unmoved lug is not merged into `outer`


def test_shapes_created_in_a_helper_join_the_active_builder():
    def hole():
        bd.Cylinder(1, 4, mode=bd.Mode.SUBTRACT)

    with bd.BuildPart() as p:
        bd.Box(4, 4, 2)
        hole()
    assert close(p.part.volume, 32 - PI * 2)


def test_fillet_grows_tangent_chains_and_skips_smooth_edges():
    with bd.BuildPart() as p:
        bd.Box(82, 34, 6, align=(C, C, MIN))
        with bd.Locations((0, 0, -1)):
            bd.Box(26, 10, 8, align=(C, C, MIN), mode=bd.Mode.SUBTRACT)
            for x in (-13, 13):
                with bd.Locations((x, 0, 0)):
                    bd.Cylinder(5, 8, align=(C, C, MIN), mode=bd.Mode.SUBTRACT)
    lines = p.part.edges().filter_by(bd.GeomType.LINE)
    out = bd.chamfer(lines, 0.8)
    assert out.is_valid and out.volume < p.part.volume


def test_impossible_fillet_raises():
    box = bd.Box(10, 10, 10)
    with pytest.raises(Exception):
        bd.fillet(box.edges().filter_by(bd.Axis.Z), 6)


def test_spline_circles_are_circles():
    with bd.BuildPart() as p:
        bd.Cylinder(35, 16, align=(C, C, MIN))
        with bd.Locations((0, 0, 8)):
            bd.Torus(33.8, 3, mode=bd.Mode.SUBTRACT)
    circles = p.part.edges().filter_by(bd.GeomType.CIRCLE)
    groove = [e for e in circles if abs(e.radius - 35) < 1e-6 and 0.1 < e.arc_center.Z < 15.9]
    assert len(groove) == 2 and all(abs(abs(e.normal().Z) - 1) < 1e-9 for e in groove)


def test_algebra_mode():
    plate = bd.Box(40, 20, 5) - bd.Pos(10, 0, 0) * bd.Cylinder(3, 5)
    assert close(plate.volume, 4000 - PI * 9 * 5)
    assert close(plate.moved(bd.Rot(0, 0, 90)).volume, plate.volume)


def test_old_import_name_is_the_same_module():
    from keel import build123d

    assert build123d is bd


# Expected values below were printed by build123d 0.11.1 for the same scripts.


def test_buildline_profile_in_a_sketch_on_xz():
    with bd.BuildSketch(bd.Plane.XZ) as sk:
        with bd.BuildLine():
            bd.Line((0.001, -1.6), (0.55, -1.6))
            bd.Line((0.55, -1.6), (0.55, 3.10))
            bd.Line((0.55, 3.10), (2.0, 3.10))
            bd.Line((2.0, 3.10), (2.0, 4.06))
            bd.RadiusArc((2.0, 4.06), (0.001, 4.6), -5.0)
            bd.Line((0.001, 4.6), (0.001, -1.6))
        bd.make_face()
    assert close(bd.revolve(sk.sketch, bd.Axis.Z).volume, 19.7464)


def test_buildline_splines_make_a_face_with_a_hole():
    with bd.BuildSketch() as sk:
        with bd.BuildLine():
            bd.Spline((0, -2), (1.5, 0), (0, 2))
            bd.Spline((0, 2), (-1, 0), (0, -2))
        bd.make_face()
        bd.Circle(0.4, mode=bd.Mode.SUBTRACT)
    assert close(bd.extrude(sk.sketch, amount=1).volume, 6.1641)


def test_three_point_spline_is_the_quadratic_through_them():
    # OpenCASCADE fits degree 2 with its middle pole at (3, 0): the curve passes (1.5, 0) at t = 0.5.
    # Lengths agree to the kernel's length integration (~1e-4).
    spline = bd.Spline((0, -2), (1.5, 0), (0, 2))
    assert [round(c, 9) for c in spline.position_at(0.5)] == [1.5, 0, 0]
    assert close(spline.length, 5.198038, rel=2e-4)
    assert close(bd.Spline((0, 2), (-1, 0), (0, -2)).length, 4.591141, rel=2e-4)


def test_radius_arc_sides_on_a_workplane():
    with bd.BuildLine(bd.Plane.YZ) as line:
        bd.Line((0, 0), (1, 2))
        bd.RadiusArc((1, 2), (3, 0), 3)
        bd.RadiusArc((3, 0), (0, 0), 2, short_sagitta=False)
    short, long_ = line.edges()[1], line.edges()[2]
    assert [round(c, 4) for c in short.position_at(0.5)] == [0, 2.2505, 1.2505]
    assert [round(c, 4) for c in long_.position_at(0.5)] == [0, 1.5, 3.3229]
    assert len(line.wires()) == 1
    assert bd.RadiusArc((1, 2), (3, 0), 3).position_at(0.5).X == pytest.approx(2.2505, abs=1e-4)
    with pytest.raises(ValueError, match="not large enough"):
        bd.RadiusArc((0, 0), (10, 0), 1)


def test_buildline_in_a_part_is_the_sweep_path():
    with bd.BuildPart() as part:
        with bd.BuildLine():
            bd.Line((0, 0, 0), (0, 0, 10))
        with bd.BuildSketch():
            bd.Circle(1)
        bd.sweep()
    assert close(part.part.volume, 10 * PI)


def test_private_line_objects_stay_out_of_the_line():
    with bd.BuildLine() as line:
        bd.Line((0, 0), (1, 0))
        bd.Line((1, 0), (1, 1), mode=bd.Mode.PRIVATE)
    assert len(line.edges()) == 1


def test_make_face_points_up_but_face_follows_the_loop():
    clockwise = bd.Polyline((0, 0), (0, 10), (10, 10), (10, 0), close=True)
    assert bb(bd.extrude(bd.make_face(clockwise.edges()), 5))[5] == 5
    assert bb(bd.extrude(bd.Face(clockwise), 5))[2] == -5


def test_export_step_and_stl(tmp_path):
    part = bd.Box(10, 20, 5)
    part.label = "block"
    assert bd.export_step(part, tmp_path / "block.step")
    text = (tmp_path / "block.step").read_text()
    assert "ISO-10303-21" in text and "'block'" in text
    assert bd.export_stl(part, tmp_path / "block.stl")
    assert (tmp_path / "block.stl").stat().st_size > 84


def test_make_hull_of_circles_is_exact():
    # build123d samples each edge, so it agrees to ~1e-6 here, and keel's tangents are exact.
    pin, lobe = bd.Pos(0, 10) * bd.Circle(4), bd.Circle(6)
    assert close(bd.extrude(bd.make_hull(pin.edges() + lobe.edges()), 1).volume, 183.688)
    three = bd.Circle(12.5) + bd.Pos(-20, 30) * bd.Circle(4.8) + bd.Pos(20, 30) * bd.Circle(4.8)
    assert close(bd.extrude(bd.make_hull(three.edges()), 1).volume, 1633.339)
    lines_and_circle = bd.Rectangle(20, 4).edges() + (bd.Pos(0, 12) * bd.Circle(3)).edges()
    assert close(bd.extrude(bd.make_hull(lines_and_circle), 1).volume, 230.453)
    with bd.BuildSketch() as sk:
        bd.Circle(5)
        with bd.Locations((15, 0)):
            bd.Circle(2)
        bd.make_hull()
    assert close(sk.sketch.area, 152.66)
    with pytest.raises(KeelError, match="not supported yet"):
        bd.make_hull([bd.Spline((0, 0), (3, 4), (6, 0), (10, 2))])


def test_sketch_edges_and_vertices():
    assert len(bd.RectangleRounded(6, 4, 1).edges()) == 8
    assert {v.X for v in bd.Rectangle(4, 2).vertices()} == {-2, 2}
    assert len(bd.Rectangle(4, 2).vertices()) == 4


def test_positions_along_edges_and_wires_go_by_length():
    wire = bd.Polyline((0, 0), (10, 0), (10, 5))
    assert tuple(wire @ 0.5) == pytest.approx((7.5, 0, 0))
    assert tuple(wire % 0.9) == pytest.approx((0, 1, 0))
    spline = bd.Spline((0, 0), (3, 4), (6, 0), (10, 2))
    assert tuple(spline @ 0.3) == pytest.approx((2.0417, 4.2876, 0), abs=1e-4)
    assert tuple(spline % 0.3) == pytest.approx((0.9249, 0.3803, 0), abs=1e-4)


def test_holes_follow_build123d():
    # volumes from build123d 0.11.1 (OpenCascade) for the same script
    with bd.BuildPart() as p:
        bd.Box(60, 40, 10, align=(C, C, MIN))
        with bd.Locations((0, 0, 10)):
            with bd.GridLocations(20, 0, 3, 1):
                bd.CounterBoreHole(2, 4, 3)
        with bd.Locations((0, 12, 10), (0, -12, 10)):
            bd.CounterSinkHole(1.5, 3, depth=6)
        with bd.Locations((25, 15, 10)):
            bd.Hole(1, depth=4)  # centered on its location: 4 mm each way, so 4 mm into the part
        with bd.Locations(bd.Plane.XZ.offset(-20)):
            with bd.Locations((-25, 5)):
                bd.Hole(1.5)  # through everything
    assert close(p.part.volume, 22871.058263463747, rel=1e-6)
    assert len(p.faces()) == len(p.part.faces()) and len(p.edges()) == len(p.part.edges())
    assert len(p.vertices()) == len(p.part.vertices())
    with pytest.raises(ValueError, match="No depth provided"):
        bd.Hole(1)


def test_hole_grid_cuts_in_one_boolean():
    # every hole of a Locations block goes into one kernel call (build123d cuts them together
    # too); 100 holes take about half a second instead of tens of seconds
    import time

    t0 = time.perf_counter()
    with bd.BuildPart() as p:
        bd.Box(60, 60, 4, align=(C, C, MIN))
        with bd.Locations((0, 0, 4)):
            with bd.GridLocations(5.5, 5.5, 10, 10):
                bd.Hole(1.5)
    assert close(p.part.volume, 60 * 60 * 4 - 100 * PI * 1.5**2 * 4, rel=1e-6)
    assert time.perf_counter() - t0 < 5


def test_intersect_mode_keeps_what_any_new_object_covers():
    # build123d intersects with the union of the objects made at all locations
    with bd.BuildPart() as p:
        bd.Box(20, 20, 20)
        with bd.Locations((7, 0, 0), (-7, 0, 0)):
            bd.Sphere(6, mode=bd.Mode.INTERSECT)
    assert close(p.part.volume, 1526.8140119640652, rel=1e-5)


@pytest.mark.parametrize(
    "amount, open_top, kind, expected",
    [
        (-1.5, True, bd.Kind.ARC, 2803.5),
        (1.5, True, bd.Kind.ARC, 3439.8119223689014),
        (1.5, True, bd.Kind.INTERSECTION, 3523.5),
        (-1.5, False, bd.Kind.ARC, 5508.0),
        (1.5, False, bd.Kind.ARC, 13523.595092691128),
        (1.5, False, bd.Kind.INTERSECTION, 13662.0),
    ],
)
def test_offset_and_shell_match_build123d(amount, open_top, kind, expected):
    with bd.BuildPart() as p:
        bd.Box(30, 20, 15, align=(C, C, MIN))
        bd.offset(amount=amount, openings=p.faces().sort_by(bd.Axis.Z)[-1] if open_top else None, kind=kind)
    assert p.part.is_valid
    assert close(p.part.volume, expected, rel=1e-6)


def test_offset_of_a_cylinder_shell():
    with bd.BuildPart() as p:
        bd.Cylinder(10, 15, align=(C, C, MIN))
        bd.offset(amount=-1.5, openings=p.faces().sort_by(bd.Axis.Z)[-1])
    assert close(p.part.volume, 1648.158045889545, rel=1e-5)


def test_text_matches_build123d():
    # build123d 0.11.1 resolves "Arial" through OpenCascade's font aliases; keel finds the
    # same font, glyph curves and layout. Areas from Compound.make_text, boxes from Text.
    font = __import__("keel.text", fromlist=["find_font"]).find_font("Arial")[0]
    if not font.endswith("FreeSans.ttf"):
        pytest.skip("reference values were taken with FreeSans standing in for Arial")
    with bd.BuildPart() as p:
        bd.Box(50, 15, 3, align=(C, C, MIN))
        with bd.BuildSketch(bd.Plane.XY.offset(3)):
            bd.Text("KEEL", font_size=9)
        bd.extrude(amount=1)
    assert close(p.part.volume, 2298.271934205416, rel=1e-9)
    TA = bd.TextAlign
    cases = [
        ("KEEL", 9, {}, 48.271934205, (-10.1205, -2.25, 10.7955, 4.311)),
        ("Ag Oq%", 10, dict(text_align=(TA.LEFT, TA.BOTTOM)), 82.507835056, (0.15, -2.18, 36.58, 7.41)),
        ("two\nlines", 8, dict(text_align=(TA.RIGHT, TA.TOP)), 52.068858918, (-15.288, -15.384, -0.232, -1.056)),
        ("abc\nlonger line\nx", 6, {}, 52.446155259, (-12.843, -8.1, 13.023, 9.474)),
        ("Bold", 9, dict(font_style=bd.FontStyle.BOLD), 64.515427895, (-8.973, -2.457, 9.288, 4.311)),
        ("Ital", 9, dict(font_style=bd.FontStyle.ITALIC), 27.191414974, (-4.9905, -2.457, 6.6555, 4.311)),
    ]
    for txt, size, kw, area, box in cases:
        with bd.BuildSketch() as s:
            bd.Text(txt, size, **kw)
        assert close(s.sketch.area, area, rel=1e-8), txt
        b = bd.extrude(s.sketch, amount=1).bounding_box()
        assert (b.min.X, b.min.Y, b.max.X, b.max.Y) == pytest.approx(box, abs=1e-3), txt


def test_loft_square_to_circle_matches_build123d():
    with bd.BuildPart() as p:
        with bd.BuildSketch():
            bd.Rectangle(20, 20)
        with bd.BuildSketch(bd.Plane.XY.offset(20)):
            bd.Circle(6)
        bd.loft()
    # OpenCascade's ThruSections pairs each square side with a quarter of the circle by angle
    assert close(p.part.volume, 4861.155009398977, rel=1e-6)


def test_fillet_on_a_non_manifold_body_says_so():
    touching = bd.Box(20, 20, 10, align=(C, C, MIN)) + bd.Pos(0, 0, 15) * bd.Rot(90, 0, 0) * bd.Cylinder(5, 20)
    assert not touching.is_valid  # the cylinder rests on the box along a line
    with pytest.raises(KeelError, match="already non-manifold before the fillet"):
        bd.fillet(touching.edges().filter_by(bd.Axis.Z), radius=1)
    corner = bd.Box(10, 10, 10) + bd.Pos(10, 10, 0) * bd.Box(10, 10, 10)  # boxes sharing an edge
    with pytest.raises(KeelError, match="already non-manifold before the chamfer"):
        bd.chamfer(corner.edges().group_by(bd.Axis.Z)[-1], length=1)
