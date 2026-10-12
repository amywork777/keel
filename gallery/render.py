"""Render STL meshes as clean CAD-style images (no visible triangles).

A small numpy rasterizer: z-buffer, crease-aware smooth normals (curved faces shade smoothly,
real edges stay sharp), 3x supersampling, and outlines drawn where depth or normals jump.

Gallery sheet of every STL in gallery/out and examples/out:
    docker run --rm -v "$PWD/..":/k -w /k/gallery python:3.12-slim \\
      sh -c "pip install -q matplotlib numpy && python render.py"

Also importable: ``render(meshes, ...)`` returns an RGB array.
"""

from __future__ import annotations

import glob
import struct

import numpy as np

CREASE_DEG = 35.0  # adjacent triangles bent more than this keep separate normals
LIGHT = np.array([0.35, -0.55, 0.76])
LIGHT /= np.linalg.norm(LIGHT)
FILL = np.array([-0.6, 0.3, 0.2])
FILL /= np.linalg.norm(FILL)


def load_stl(path: str) -> np.ndarray:
    data = open(path, "rb").read()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84 : 84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3, 3).astype(np.float64)


def _corner_normals(tri: np.ndarray) -> np.ndarray:
    """Per-corner normals: average of adjacent face normals within the crease angle."""
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(fn, axis=1)
    keep = area > 1e-12
    fn = np.where(keep[:, None], fn / np.maximum(area[:, None], 1e-30), 0)
    span = np.ptp(tri.reshape(-1, 3), axis=0).max() or 1.0
    key = np.round(tri.reshape(-1, 3) / (span * 1e-6)).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(-1)
    corner_face = np.repeat(np.arange(len(tri)), 3)
    order = np.argsort(vid, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(vid[order])) + 1, len(order)]
    cos_crease = np.cos(np.radians(CREASE_DEG))
    out = np.repeat(fn, 3, axis=0)
    weighted = fn * area[:, None]
    for s, e in zip(starts[:-1], starts[1:]):
        corners = order[s:e]
        if len(corners) < 2:
            continue
        faces = corner_face[corners]
        n_f = fn[faces]
        sim = n_f @ n_f.T >= cos_crease
        acc = sim.astype(float) @ weighted[faces]
        norm = np.linalg.norm(acc, axis=1, keepdims=True)
        out[corners] = np.where(norm > 1e-30, acc / np.maximum(norm, 1e-30), n_f)
    return out.reshape(-1, 3, 3)


def _smooth_groups(tri: np.ndarray) -> np.ndarray:
    """Label triangles so each smooth surface (no crease inside it) gets one id."""
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    span = np.ptp(tri.reshape(-1, 3), axis=0).max() or 1.0
    key = np.round(tri.reshape(-1, 3) / (span * 1e-6)).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    edges = np.concatenate([np.sort(vid[:, [i, (i + 1) % 3]], axis=1) for i in range(3)])
    owner = np.tile(np.arange(len(tri)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    e, o = edges[order], owner[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    parent = np.arange(len(tri))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cos_crease = np.cos(np.radians(CREASE_DEG))
    for k in np.flatnonzero(same):
        a, b = o[k], o[k + 1]
        if fn[a] @ fn[b] >= cos_crease:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return np.array([find(i) for i in range(len(tri))])


def _view_basis(elev: float, azim: float):
    e, a = np.radians(elev), np.radians(azim)
    fwd = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # camera looks along fwd
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def render(meshes, colors, size=(480, 400), elev=26.0, azim=-58.0, ss=3, bounds=None) -> np.ndarray:
    """Render a list of triangle arrays (N,3,3) with RGB colors to an (H, W, 3) float image."""
    W, H = size[0] * ss, size[1] * ss
    right, up, fwd = _view_basis(elev, azim)
    allpts = np.concatenate([m.reshape(-1, 3) for m in meshes])
    lo, hi = (allpts.min(0), allpts.max(0)) if bounds is None else bounds
    center = (lo + hi) / 2
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - center
    extent = max(np.abs(corners @ right).max() / W, np.abs(corners @ up).max() / H) * 2.25
    zbuf = np.full((H, W), np.inf)
    nbuf = np.zeros((H, W, 3))
    cbuf = np.ones((H, W, 3))
    idbuf = np.full((H, W), -1, dtype=np.int64)
    group_offset = 0
    # later meshes win depth ties, so coincident surfaces don't speckle
    bias = 1e-3 * float(np.ptp(allpts, axis=0).max())
    for k, (mesh, color) in enumerate(zip(meshes, colors)):
        cn = _corner_normals(mesh)
        groups = _smooth_groups(mesh) + group_offset
        group_offset = groups.max() + 1
        rel = mesh - center
        sx = (rel @ right) / extent + W / 2
        sy = H / 2 - (rel @ up) / extent
        sz = rel @ fwd - k * bias
        color = np.asarray(color, dtype=float)
        for t in range(len(mesh)):
            x, y, z = sx[t], sy[t], sz[t]
            x0, x1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()), W - 1))
            y0, y1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()), H - 1))
            if x0 > x1 or y0 > y1:
                continue
            den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(den) < 1e-12:
                continue
            px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / den
            w1 = ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            if not inside.any():
                continue
            depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
            region = zbuf[y0 : y1 + 1, x0 : x1 + 1]
            win = inside & (depth < region)
            if not win.any():
                continue
            region[win] = depth[win]
            n = w0[win, None] * cn[t, 0] + w1[win, None] * cn[t, 1] + w2[win, None] * cn[t, 2]
            nbuf[y0 : y1 + 1, x0 : x1 + 1][win] = n / np.linalg.norm(n, axis=1, keepdims=True)
            cbuf[y0 : y1 + 1, x0 : x1 + 1][win] = color
            idbuf[y0 : y1 + 1, x0 : x1 + 1][win] = groups[t]

    hit = np.isfinite(zbuf)
    n = nbuf
    facing = np.where((n @ -fwd)[..., None] < 0, -n, n)  # light both sides of thin/open parts
    diffuse = np.clip(facing @ LIGHT, 0, 1)
    fill = np.clip(facing @ FILL, 0, 1)
    half = LIGHT - fwd
    half /= np.linalg.norm(half)
    spec = np.clip(facing @ half, 0, 1) ** 40
    shade = 0.32 + 0.58 * diffuse + 0.15 * fill
    img = cbuf * shade[..., None] + 0.22 * spec[..., None]
    img = np.where(hit[..., None], img, 1.0)

    # outlines: silhouettes, depth jumps, and boundaries between smooth surfaces (real edges)
    z = np.where(hit, zbuf, np.nan)
    edge = np.zeros((H, W), dtype=bool)
    dz_tol = extent * 6
    for dy, dx in ((0, 1), (1, 0)):
        shift = lambda a: np.roll(np.roll(a, -dy, 0), -dx, 1)  # noqa: E731
        b_hit = shift(hit)
        silhouette = hit ^ b_hit
        with np.errstate(invalid="ignore"):
            depth_jump = np.nan_to_num(np.abs(z - shift(z)) > dz_tol, nan=False)
        surface_change = (idbuf != shift(idbuf)) & hit & b_hit
        edge |= silhouette | depth_jump | surface_change
    edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
    img = np.where(edge[..., None], img * 0.35, img)
    img = img.reshape(H // ss, ss, W // ss, ss, 3).mean(axis=(1, 3))
    return np.clip(img, 0, 1)


def gallery(files, out_path, cols=4, tile=(420, 340), color=(0.55, 0.66, 0.82)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * tile[0] / 100, rows * (tile[1] + 40) / 100), dpi=100)
    for ax in np.ravel(axes):
        ax.set_axis_off()
    for ax, f in zip(np.ravel(axes), files):
        ax.imshow(render([load_stl(f)], [color], size=tile))
        ax.set_title(f.split("/")[-1][:-4].replace("_", " "), fontsize=12, color="#333")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    print(f"rendered {len(files)} parts -> {out_path}")


if __name__ == "__main__":
    gallery(sorted(glob.glob("out/*.stl")) + sorted(glob.glob("../examples/out/*.stl")), "out/gallery.png")
