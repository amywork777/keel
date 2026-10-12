"""A bottle: draw a 1D profile with Pen, close it into a 2D sketch, revolve and shell to 3D."""

import os

from keel import *

os.makedirs("out", exist_ok=True)

# half-profile in the XZ plane (x = radius, y = height), revolved about local Y
profile = (
    Pen(0, 0)
    .line_to(30, 0)
    .line_to(30, 90)
    .tangent_arc_to(12, 120)   # shoulder
    .line_to(12, 140)          # neck
    .line_to(0, 140)
    .close()
    .to_sketch()
    .on("XZ")
)
body = profile.revolve(360, "Y")

# hollow it, leaving the neck open
bottle = body.shell(2, open=body.faces(">Z"))

# a label recess wrapped around isn't a kernel feature, so add grip rings instead
for z in (25, 40, 55):
    bottle = bottle - Torus(31, 1.5).move(0, 0, z)

print(bottle, "valid:", bottle.is_valid)
bottle.named("bottle", color=(0.35, 0.6, 0.45)).export("out/bottle.usda")
bottle.export("out/bottle.stl")

# 1D -> 3D directly: a spring and a bent rail
spring = Helix(radius=8, height=40, turns=6).pipe(1.2)
rail = Rect(6, 3).sweep(Pen(0, 0).h(40).tangent_arc_to(60, 20).v(30).path)
print(spring, rail)
export([spring.named("spring"), rail.named("rail").move(30, 0, 0)], "out/spring_and_rail.usda")
export([spring, rail.move(30, 0, 0)], "out/spring_and_rail.stl")
