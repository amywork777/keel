"""Render every STL in gallery/out (and examples/out) into gallery/out/gallery.png.

Runs in a plain Python image (no kernel needed):
    docker run --rm -v "$PWD/..":/k -w /k/gallery python:3.12-slim \
      sh -c "pip install -q matplotlib numpy && python render.py"
"""
import glob
import struct

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

files = sorted(glob.glob("out/*.stl")) + sorted(glob.glob("../examples/out/*.stl"))
cols = 4
rows = (len(files) + cols - 1) // cols
fig = plt.figure(figsize=(cols * 4, rows * 4))
light = np.array([0.4, -0.5, 0.75])
light /= np.linalg.norm(light)
for i, f in enumerate(files):
    data = open(f, "rb").read()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84 : 84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    tri = rec["v"].reshape(-1, 3, 3)
    shade = np.clip(rec["n"] @ light, 0, 1) * 0.7 + 0.25
    ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.c_[shade * 0.55, shade * 0.65, shade * 0.8], edgecolor="none"))
    lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(25, -55)
    ax.set_axis_off()
    ax.set_title(f.split("/")[-1][:-4], fontsize=11)
plt.tight_layout()
plt.savefig("out/gallery.png", dpi=80)
print(f"rendered {len(files)} parts")
