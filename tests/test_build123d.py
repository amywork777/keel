"""keel.build123d follows build123d 0.11's conventions (checked against real build123d)."""

import math

import pytest

from keel import build123d as bd

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
