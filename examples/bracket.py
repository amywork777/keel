"""An L-bracket: 3D primitives, selectors, sketches on faces, patterns, fillets."""

import os

from keel import *

os.makedirs("out", exist_ok=True)

base = Box(80, 40, 6, align=("center", "center", "min"))
wall = Box(6, 40, 50, align=("min", "center", "min")).move(-40, 0, 0)
bracket = base + wall

# four mounting holes through the base, clear of the wall
hole = Cylinder(3, 6, align=("center", "center", "min"))
bracket -= [h.move(6, 0, 0) for h in hole.at(grid(2, 2, 50, 24))]

# a slot through the wall, sketched on its inner face
wall_face = bracket.faces("+X").min("X")
bracket -= Slot(24, 8).on(wall_face, y=8).extrude(-6)

# round the outer corners and the wall top
bracket = bracket.fillet(bracket.edges("|Z").max("X"), 6)
bracket = bracket.fillet(bracket.edges(">Z"), 2)

print(bracket, "valid:", bracket.is_valid)
bracket.named("bracket", color=(0.75, 0.78, 0.82)).export("out/bracket.usda")
bracket.export("out/bracket.stl")
