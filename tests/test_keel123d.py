"""keel123d follows build123d 0.11's conventions (checked against real build123d)."""

import math

import pytest

import keel123d as bd

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
