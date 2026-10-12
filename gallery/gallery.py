"""Realistic parts that exercise keel end to end. Each prints valid/volume or the failure."""
import math, os, sys, traceback, warnings
from keel import *

os.makedirs("out", exist_ok=True)
results = {}

def case(fn):
    name = fn.__name__
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            part = fn()
        ok = part.is_valid
        part.export(f"out/{name}.stl"); part.named(name).export(f"out/{name}.usda")
        back = load(f"out/{name}.usda")[0]
        rt = abs(back.volume - part.volume) / part.volume < 1e-4
        results[name] = ("OK" if ok and rt else "BAD", f"valid={ok} roundtrip={rt} faces={len(part.faces())} vol={part.volume:.1f}" + (f" warn={w[0].message}" if w else ""))
    except Exception as e:
        results[name] = ("FAIL", f"{type(e).__name__}: {str(e)[:160]}")
    return fn

@case
def flange():
    disc = Cylinder(40, 10, align=("center","center","min"))
    hub = Circle(18).on(disc.faces(">Z")).extrude(15)
    p = disc + hub
    p -= Cylinder(10, 30, align=("center","center","min"))
    p -= Cylinder(4, 10, align=("center","center","min")).at(polar(6, 30))
    return p.fillet(p.edges("%circle").filter(lambda e: abs(e.radius-18)<1e-6 and abs(e.center[2]-10)<1e-6), 3)

@case
def enclosure():
    box = Box(100, 60, 40, align=("center","center","min")).fillet("|Z", 8)
    shell = box.shell(2, open=">Z")
    return shell - Rect(20, 8).on(shell.faces("-X").min("X"), y=5).extrude(-2)

@case
def phone_stand():
    prof = Polygon((0,0),(60,0),(60,5),(15,5),(35,60),(30,62),(8,5),(0,5)).on("XZ")
    return prof.extrude(40, both=True).fillet("|Y", 1)

@case
def knob():
    body = RegularPolygon(15, 8).extrude(12, taper=5)
    return body - Cylinder(3, 12, align=("center","center","min"))

@case
def pulley():
    prof = (Pen(5,0).h(15).v(3).line(-4,4).line(4,4).v(3).h(-15).close().to_sketch()).on("XZ")
    return prof.revolve(360, "Y")

@case
def bolt():
    head = RegularPolygon(8, 6).extrude(5)
    shank = Cylinder(4, 30, align=("center","center","max"))
    thread = Circle(0.5).sweep(Helix(4, 25, 12).move(0,0,-28))
    return head + shank + thread

@case
def handle():
    path = Pen(0,0).v(30).tangent_arc_to(30, 60).h(40).path.on("XZ")
    return RoundedRect(12, 8, 3).sweep(path)

@case
def lofted_vase():
    s = [Circle(20), RegularPolygon(28, 6).on(Plane.XY.offset(40)), Circle(12).on(Plane.XY.offset(90))]
    return loft(*s, smooth=True)

@case
def bracket_builder():
    with Build() as b:
        b += Box(50, 30, 8)
        b += Box(8, 30, 40, align=("min","center","min")).move(-25, 0, -4)
        b -= [c.move(8, 0, 0) for c in Cylinder(3, 8).at(grid(2, 2, 25, 15))]
        b.chamfer(b.edges(">Z and |Y"), 1)
    return b.part

@case
def rotated_boss():
    base = Box(40, 40, 10)
    tilted = Cylinder(5, 30).rotate("X", 45).move(0, 0, 10)
    return (base + tilted).fillet(base.edges(">Z"), 1)

@case
def mirrored_arm():
    arm = Slot(40, 10).move(15, 0).extrude(6) + Cylinder(8, 6, align=("center","center","min"))
    return arm + arm.rotate("Z", 180)  # mirror() + fuse hits a kernel boolean bug here (now raises)

@case
def ring_with_gems():
    ring = Torus(10, 1.5)
    gems = [Sphere(2.2).move(p[0], p[1], 1.2) for p in polar(5, 10, start=60, sweep=60)]
    return ring + gems

for k, (s, msg) in results.items():
    print(f"{s:4} {k:16} {msg}")
