"""Text: glyph outlines from font files, laid out the way OpenCASCADE does it.

build123d's ``Text`` is OpenCASCADE's ``Font_BRepTextBuilder``: FreeType loads each glyph
unhinted at 4800 pixels per em, its outline becomes lines and quadratic or cubic Bezier
curves, and ``Font_TextFormatter`` places the glyphs (float32 pen positions, kerning from the
font's ``kern`` table). keel reads the font with fontTools and repeats each step, including
FreeType's fixed-point scaling, so the same font gives the same faces.

Fonts are found like OpenCASCADE's ``Font_FontMgr``: by family name (plus any style words
that are not Regular/Bold/Italic), then through its aliases ("Arial" is "arial", "helvetica",
"freesans" or "dejavu sans" on Linux, whichever is installed first), then any sans font.
"""

from __future__ import annotations

import os
import struct
import sys
from functools import lru_cache
from pathlib import Path

from ._kernel import KeelError

REGULAR, BOLD, ITALIC, BOLDITALIC = "regular", "bold", "italic", "bolditalic"

_PPEM = 4800  # OpenCASCADE: 72 pt at 4800 dpi
_EXTENSIONS = (".ttf", ".otf", ".ttc")

# Font_FontMgr's alias lists (OCCT 7.9), per platform
if sys.platform.startswith("win") or sys.platform == "darwin":
    _MONO, _SERIF, _SANS = ["courier new"], ["times new roman"], ["arial"]
else:
    _MONO = ["courier new", "freemono", "dejavu sans mono"]
    _SERIF = ["times new roman", "times", "freeserif", "dejavu serif"]
    _SANS = ["arial", "helvetica", "freesans", "dejavu sans"]
_ALIASES = {
    "mono": (_MONO, None), "courier": (_MONO, None), "monospace": (_MONO, None),
    "rock": (_SANS, None), "sansserif": (_SANS, None), "sans-serif": (_SANS, None), "sans": (_SANS, None),
    "arial": (_SANS, None), "times": (_SERIF, None), "serif": (_SERIF, None), "times-roman": (_SERIF, None),
    "times-bold": (_SERIF, BOLD), "times-italic": (_SERIF, ITALIC), "times-bolditalic": (_SERIF, BOLDITALIC),
    "symbol": (["symbol"], None), "iris": (["lucida console"], None),
}


def _f32(x: float) -> float:
    """Round to float32, the precision of OpenCASCADE's text layout."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _mulfix(a: int, b: int) -> int:
    """FreeType's FT_MulFix: a * b / 65536, rounded half away from zero."""
    s = -1 if (a < 0) != (b < 0) else 1
    return s * ((abs(a) * abs(b) + 0x8000) >> 16)


def _divfix(a: int, b: int) -> int:
    """FreeType's FT_DivFix: a * 65536 / b, rounded."""
    s = -1 if (a < 0) != (b < 0) else 1
    a, b = abs(a), abs(b)
    return s * (((a << 16) + (b >> 1)) // b)


# -- finding fonts ------------------------------------------------------------------------------


def _font_dirs() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        dirs = [Path("/System/Library/Fonts"), Path("/Library/Fonts"), home / "Library/Fonts"]
    elif sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs = [Path(windir) / "Fonts", Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts"]
    else:
        dirs = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".local/share/fonts", home / ".fonts"]
    return [d for d in dirs if d.is_dir()]


def _sfnt_tables(data: bytes, offset: int) -> dict[bytes, tuple[int, int]]:
    num = struct.unpack_from(">H", data, offset + 4)[0]
    tables = {}
    for i in range(num):
        tag, _, off, length = struct.unpack_from(">4sLLL", data, offset + 12 + 16 * i)
        tables[tag] = (off, length)
    return tables


def _sfnt_names(data: bytes, off: int) -> dict[int, str]:
    """name ID -> string, preferring Windows Unicode English names as FreeType does."""
    _, count, strings = struct.unpack_from(">HHH", data, off)
    found: dict[int, tuple[int, str]] = {}
    for i in range(count):
        plat, enc, lang, name_id, length, str_off = struct.unpack_from(">HHHHHH", data, off + 6 + 12 * i)
        if name_id not in (1, 2, 16, 17):
            continue
        raw = data[off + strings + str_off: off + strings + str_off + length]
        if plat == 3 and enc in (0, 1, 10):
            rank, text = (0 if lang == 0x409 else 1), raw.decode("utf-16-be", "replace")
        elif plat == 0:
            rank, text = 2, raw.decode("utf-16-be", "replace")
        elif plat == 1 and enc == 0:
            rank, text = (3 if lang == 0 else 4), raw.decode("mac_roman", "replace")
        else:
            continue
        if name_id not in found or rank < found[name_id][0]:
            found[name_id] = (rank, text)
    return {k: v[1] for k, v in found.items()}


def _describe(data: bytes, offset: int) -> tuple[str, str] | None:
    """(OpenCASCADE font key, aspect) of the font at ``offset`` in a font file, from the names
    and style flags FreeType reports. Reads the raw tables: fontTools is slow at this."""
    t = _sfnt_tables(data, offset)
    if b"name" not in t or b"cmap" not in t or b"head" not in t:
        return None
    cmap = t[b"cmap"][0]
    n_sub = struct.unpack_from(">H", data, cmap + 2)[0]
    subtables = [struct.unpack_from(">HH", data, cmap + 4 + 8 * i) for i in range(n_sub)]
    if not any(p == 0 or (p == 3 and e in (1, 10)) for p, e in subtables):  # FT_Select_Charmap(unicode)
        return None
    names = _sfnt_names(data, t[b"name"][0])
    if 16 in names:
        family, style = names[16], names.get(17, names.get(2, ""))
    elif 1 in names:
        family, style = names[1], names.get(2, "")
    else:
        return None
    if b"OS/2" in t and (b"glyf" in t or b"CFF " in t or b"CFF2" in t):
        sel = struct.unpack_from(">H", data, t[b"OS/2"][0] + 62)[0]
        bold, italic = bool(sel & 32), bool(sel & (1 | 512))
    else:
        mac = struct.unpack_from(">H", data, t[b"head"][0] + 44)[0]
        bold, italic = bool(mac & 1), bool(mac & 2)
    aspect = BOLDITALIC if bold and italic else BOLD if bold else ITALIC if italic else REGULAR

    def drop(word):
        nonlocal style
        i = style.find(word)
        if i >= 0:
            style = style[:i] + style[i + len(word):]
            return True
        return False

    if italic:
        drop("Italic") or drop("Oblique")
    if bold:
        drop("Bold")
    drop("Regular") or drop("Book")
    style = " ".join(style.split())
    return (f"{family} {style}" if style else family).lower(), aspect


def _faces_in(path: Path):
    """(face index, (key, aspect) or None) for each font in a font file."""
    import mmap

    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:  # reads only the tables used
        if data[:4] == b"ttcf":
            faces = [(i, _describe(data, struct.unpack_from(">L", data, 12 + 4 * i)[0]))
                     for i in range(struct.unpack_from(">L", data, 8)[0])]
        else:
            faces = [(0, _describe(data, 0))]
    return faces


@lru_cache(maxsize=1)
def _font_map() -> dict[str, dict[str, tuple[str, int]]]:
    """Every installed font: key -> {aspect: (path, face index)}; the first file wins."""
    fonts: dict[str, dict[str, tuple[str, int]]] = {}
    for d in _font_dirs():
        for root, dirnames, files in os.walk(d):
            dirnames.sort()
            for fname in sorted(files):
                path = Path(root) / fname
                if path.suffix.lower() not in _EXTENSIONS:
                    continue
                try:
                    for index, described in _faces_in(path):
                        if described:
                            fonts.setdefault(described[0], {}).setdefault(described[1], (str(path), index))
                except Exception:  # noqa: BLE001 - unreadable font files are skipped, as OCCT does
                    continue
    return fonts


def find_font(name: str = "Arial", aspect: str = REGULAR) -> tuple[str, int, bool]:
    """(path, face index, synthesize italic) of the font OpenCASCADE picks for ``name``."""
    fonts = _font_map()
    key = name.lower()
    found = fonts.get(key)
    if found is None:
        for alias_pass in (_ALIASES.get(key), (_SANS, None)):
            if alias_pass is None:
                continue
            names, alias_aspect = alias_pass
            for alias in names:
                if alias in fonts:
                    if found is None:
                        found = fonts[alias]
                    if alias_aspect is not None and alias_aspect in fonts[alias]:
                        found, aspect = fonts[alias], alias_aspect
                        break
                    if alias_aspect is None and aspect in fonts[alias]:
                        found = fonts[alias]
                        break
            if found is not None:
                break
    if found is None:
        if not fonts:
            raise KeelError("Text: no fonts found on this system; pass font_path= to a .ttf or .otf file")
        found = fonts[sorted(fonts)[0]]
    return _path_for_aspect(found, aspect)


def _path_for_aspect(found: dict, aspect: str) -> tuple[str, int, bool]:
    """Font_SystemFont::FontPathAny: the aspect, else italic synthesized from bold or regular."""
    if aspect in found:
        return (*found[aspect], False)
    if aspect in (ITALIC, BOLDITALIC):
        if aspect == BOLDITALIC and BOLD in found:
            return (*found[BOLD], True)
        if REGULAR in found:
            return (*found[REGULAR], True)
    if REGULAR in found:
        return (*found[REGULAR], False)
    for a in (REGULAR, BOLD, ITALIC, BOLDITALIC):
        if a in found:
            return (*found[a], False)
    raise KeelError("Text: font has no faces")


# -- glyphs -------------------------------------------------------------------------------------


class _Font:
    """A font loaded the way OpenCASCADE's Font_FTFont loads it (unhinted, 4800 ppem)."""

    def __init__(self, path: str, index: int = 0, synthesize_italic: bool = False):
        from fontTools.ttLib import TTCollection, TTFont

        self.font = TTCollection(path).fonts[index] if path.lower().endswith(".ttc") else TTFont(path, fontNumber=index)
        head = self.font["head"]
        self.upem = head.unitsPerEm
        scaled = (72 * 64 * _PPEM + 36) // 72  # FT_REQUEST_WIDTH of 72 pt at 4800 dpi
        self.scale = _divfix(scaled, self.upem)
        if head.flags & 8:  # integer ppem fonts: FreeType rescales from the rounded ppem
            self.scale = _divfix(((scaled + 32) >> 6) << 6, self.upem)
        self.cmap = self.font.getBestCmap() or {}
        self.glyph_set = self.font.getGlyphSet()
        # FreeType's face->ascender and face->height: typographic metrics when the font says
        # USE_TYPO_METRICS, else hhea (falling back to OS/2 when hhea has none)
        hhea, os2 = self.font["hhea"], self.font["OS/2"] if "OS/2" in self.font else None
        if os2 is not None and os2.fsSelection & 128:
            asc, desc, gap = os2.sTypoAscender, os2.sTypoDescender, os2.sTypoLineGap
        else:
            asc, desc, gap = hhea.ascent, hhea.descent, hhea.lineGap
            if not (asc or desc) and os2 is not None:
                if os2.sTypoAscender or os2.sTypoDescender:
                    asc, desc, gap = os2.sTypoAscender, os2.sTypoDescender, os2.sTypoLineGap
                else:
                    asc, desc, gap = os2.usWinAscent, -os2.usWinDescent, 0
        ppem_per_unit = _f32(_f32(_PPEM) / _f32(self.upem))
        self.ascender = _f32(_f32(asc) * ppem_per_unit)
        self.line_spacing = _f32(_f32(asc - desc + gap) * ppem_per_unit)
        self.kern = self._kern_pairs()
        self.matrix = None
        if synthesize_italic:  # Font_FTFont: shear 10 degrees through FT_Set_Transform
            import math

            xx = int(math.cos(-math.radians(10)) * (1 << 16))
            self.matrix = (xx, _mulfix(int(math.tan(math.radians(10)) * (1 << 16)), xx), 0, xx)

    def _kern_pairs(self) -> dict:
        """The ``kern`` table summed over its horizontal format 0 subtables, as FreeType's
        FT_Get_Kerning reads it (an override subtable replaces the sum)."""
        if "kern" not in self.font:
            return {}
        pairs: dict = {}
        for sub in self.font["kern"].kernTables:
            if getattr(sub, "format", 0) != 0 or not hasattr(sub, "kernTable"):
                continue
            coverage = getattr(sub, "coverage", 1)
            if getattr(sub, "version", 0) == 0 and not coverage & 1:  # vertical
                continue
            override = getattr(sub, "version", 0) == 0 and coverage & 8
            for pair, value in sub.kernTable.items():
                pairs[pair] = value if override else pairs.get(pair, 0) + value
        return pairs

    def glyph_name(self, char: str) -> str:
        return self.cmap.get(ord(char), ".notdef")

    def _x(self, v: int) -> int:
        return _mulfix(v, self.scale)

    def _transform(self, x: int, y: int) -> tuple[int, int]:
        if self.matrix is None:
            return x, y
        xx, xy, yx, yy = self.matrix
        return _mulfix(x, xx) + _mulfix(y, xy), _mulfix(x, yx) + _mulfix(y, yy)

    def advance(self, char: str, next_char: str | None) -> float:
        """Font_FTFont::AdvanceX in pixels (float32), kerning with the next character."""
        name = self.glyph_name(char)
        width, lsb = self.font["hmtx"][name]
        shift = self._lsb_shift(name, lsb)
        adv = self._x(shift + width) - self._x(shift)
        if self.matrix is not None:
            adv = self._transform(adv, 0)[0]
        kern = 0
        if next_char is not None and self.kern:
            a, b = self.cmap.get(ord(char)), self.cmap.get(ord(next_char))
            if a and b:
                kern = self._x(self.kern.get((a, b), 0))
        return _f32(_f32(adv + kern) / _f32(64.0))

    def _lsb_shift(self, name: str, lsb: int) -> int:
        """Font units FreeType shifts a TrueType outline by so the glyph origin is at x = 0."""
        if "glyf" not in self.font:
            return 0
        g = self.font["glyf"][name]
        return getattr(g, "xMin", lsb) - lsb if g.numberOfContours else 0

    def outline(self, char: str) -> list[tuple[list[tuple[int, int]], list[int]]]:
        """The glyph's contours as FreeType gives them: 26.6 points and tags (1 on the curve,
        0 quadratic control point, 2 cubic control point)."""
        name = self.glyph_name(char)
        contours = []
        if "glyf" in self.font:
            glyf = self.font["glyf"]
            g = glyf[name]
            if not g.numberOfContours:
                return []
            coords, ends, flags = g.getCoordinates(glyf)
            shift = self._lsb_shift(name, self.font["hmtx"][name][1])
            start = 0
            for end in ends:
                pts = [self._transform(self._x(x - shift), self._x(y)) for x, y in coords[start:end + 1]]
                contours.append((pts, [f & 1 for f in flags[start:end + 1]]))
                start = end + 1
            return contours
        from fontTools.pens.recordingPen import RecordingPen

        pen = RecordingPen()
        self.glyph_set[name].draw(pen)
        pts, tags = [], []

        def add(p, tag):
            pts.append(self._transform(round(p[0] * self.scale / 65536), round(p[1] * self.scale / 65536)))
            tags.append(tag)

        for op, args in pen.value:
            if op == "moveTo":
                if pts:
                    contours.append((pts, tags))
                pts, tags = [], []
                add(args[0], 1)
            elif op == "lineTo":
                add(args[0], 1)
            elif op == "curveTo":
                add(args[0], 2)
                add(args[1], 2)
                add(args[2], 1)
            elif op == "qCurveTo":
                for p in args[:-1]:
                    add(p, 0)
                add(args[-1], 1)
            elif op in ("closePath", "endPath") and pts:
                if len(pts) > 1 and pts[-1] == pts[0] and tags[-1] == tags[0] == 1:
                    pts.pop()
                    tags.pop()
                contours.append((pts, tags))
                pts, tags = [], []
        if pts:
            contours.append((pts, tags))
        return contours


def glyph_segments(font: _Font, char: str, size: float) -> list[list[tuple]]:
    """StdPrs_BRepFont::renderGlyph: each contour as segments ("line", p0, p1) or
    ("bezier", poles...), in model units for a font of ``size``."""
    unit = size / 72.0 * 72.0 / _PPEM  # StdPrs_BRepFont's scale, same rounding

    def xy(v):
        return (unit * v[0] / 64.0, unit * v[1] / 64.0)

    out = []
    for pts, tags in font.outline(char):
        n = len(pts)
        if n < 3:
            continue
        segs = []
        cur, nxt = xy(pts[n - 1]), xy(pts[0])
        line_seg = tags[n - 1] == 1
        line_start = cur
        for i in range(n):
            prev, cur, nxt = cur, nxt, xy(pts[(i + 1) % n])
            tag = tags[i]
            if tag == 1:
                if not line_seg:
                    line_start, line_seg = cur, True
                    continue
                if ((cur[0] - line_start[0]) ** 2 + (cur[1] - line_start[1]) ** 2) ** 0.5 <= 1e-7:
                    line_start, line_seg = cur, True
                    continue
                segs.append(("line", line_start, cur))
                line_start = cur
            elif tag == 0:
                line_seg = False
                p0 = ((cur[0] + prev[0]) * 0.5, (cur[1] + prev[1]) * 0.5) if tags[(i - 1) % n] == 0 else prev
                p2 = ((cur[0] + nxt[0]) * 0.5, (cur[1] + nxt[1]) * 0.5) if tags[(i + 1) % n] == 0 else nxt
                segs.append(("bezier", p0, cur, p2))
            elif tag == 2 and tags[(i + 1) % n] == 2:
                line_seg = False
                segs.append(("bezier", prev, cur, nxt, xy(pts[(i + 2) % n])))
        if not segs:
            continue
        first, last = segs[0][1], segs[-1][-1]
        if ((first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2) ** 0.5 > 1e-7:
            segs.append(("line", last, first))
        out.append(segs)
    return out


def layout(font: _Font, txt: str, halign: str = "center", valign: str = "center") -> list[tuple[str, float, float]]:
    """Font_TextFormatter: (character, x, y) pen positions in pixels for the visible glyphs."""
    chars = [c for c in txt if c not in "\r\a\f\b\v"]
    pen = _f32(0.0)
    corners, new_lines = [], []
    for i, c in enumerate(chars):
        # OCCT's iterator leaves the last character as the "next" one at the end of the text
        # (so it kerns with itself), unless the text is one character long
        nxt = chars[i + 1] if i + 1 < len(chars) else (c if len(chars) > 1 else None)
        if c == "\n":
            new_lines.append(pen)
            adv = _f32(0.0)
        elif c == "\t":
            raise KeelError("Text: tabs are not supported yet")
        else:
            adv = font.advance(c, nxt)
        corners.append([pen, _f32(0.0)])
        pen = _f32(pen + adv)
    if not corners:
        return []
    last_width = _f32(pen - corners[-1][0])

    def line_width(i):
        if i < len(new_lines):
            return new_lines[0] if i == 0 else _f32(new_lines[i] - new_lines[i - 1])
        return pen if i == 0 else _f32(pen - new_lines[i - 1])

    max_width = pen if not new_lines else max(line_width(i) for i in range(len(new_lines) + 1))
    pen_line = _f32(-font.ascender)
    lines, start = 0, 0

    def new_line(last):
        nonlocal pen_line, lines, start
        if start >= len(corners):
            lines += 1
            pen_line = _f32(pen_line - font.line_spacing)
            return
        x_min = corners[start][0]
        if last + 1 < len(corners) and abs(corners[last][1] - corners[last + 1][1]) < 1e-7:
            x_max = corners[last + 1][0]
        elif last + 1 < len(corners):
            x_max = {"left": line_width(lines), "right": max_width,
                     "center": _f32(_f32(0.5) * _f32(max_width + line_width(lines)))}[halign]
        else:
            x_max = _f32(corners[last][0] + last_width)
        if halign == "left":
            dx = _f32(-x_min)
        elif halign == "right":
            dx = _f32(_f32(-x_min + _f32(max_width - _f32(x_max - x_min))) - max_width)
        else:
            dx = _f32(_f32(-x_min + _f32(_f32(0.5) * _f32(max_width - _f32(x_max - x_min)))) - _f32(_f32(0.5) * max_width))
        for k in range(start, last + 1):
            corners[k][0] = _f32(corners[k][0] + dx)
            corners[k][1] = _f32(corners[k][1] + pen_line)
        lines += 1
        pen_line = _f32(pen_line - font.line_spacing)
        start = last + 1

    for i, c in enumerate(chars):
        if c == "\n":
            new_line(i)
    new_line(len(corners) - 1)
    top = {"bottom": _f32(-font.line_spacing - pen_line), "center": _f32(_f32(0.5) * _f32(font.line_spacing * lines)),
           "topfirstline": font.ascender, "top": _f32(0.0)}[valign]
    if valign != "top":
        for corner in corners:
            corner[1] = _f32(corner[1] + top)
    return [(c, x, y) for c, (x, y) in zip(chars, corners) if c not in " \t\n"]


def text_segments(txt: str, size: float, font: str = "Arial", font_path: str | None = None, aspect: str = REGULAR,
                  halign: str = "center", valign: str = "center") -> list[list[list[tuple]]]:
    """Glyph by glyph, the contours of ``txt`` (see ``glyph_segments``), placed in model units."""
    if size <= 1e-7 * 100:
        raise KeelError("Text: font_size is too small")
    if font_path is not None:
        path, index, italic = str(font_path), 0, False
    else:
        path, index, italic = find_font(font, aspect)
    f = _load(path, index, italic)
    unit = size / 72.0 * 72.0 / _PPEM
    glyphs = []
    for c, x, y in layout(f, txt, halign, valign):
        dx, dy = x * unit, y * unit
        contours = [[(kind, *((p[0] + dx, p[1] + dy) for p in pts)) for kind, *pts in segs]
                    for segs in glyph_segments(f, c, size)]
        if contours:
            glyphs.append(contours)
    return glyphs


@lru_cache(maxsize=8)
def _load(path: str, index: int, italic: bool) -> _Font:
    return _Font(path, index, italic)


def _curve(seg):
    from ._kernel import ub

    kind, *pts = seg
    p3 = [(float(x), float(y), 0.0) for x, y in pts]
    if kind == "line":
        return ub.create_line_segment(*p3)
    n = len(p3)
    return ub.create_canonical_curve(p3, [0.0, 1.0], [n, n], degree=n - 1)


def _box(glyph):
    pts = [p for contour in glyph for seg in contour for p in seg[1:]]
    return min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)


def text_sketch(txt: str, size: float, font: str = "Arial", font_path: str | None = None, aspect: str = REGULAR,
                halign: str = "center", valign: str = "center"):
    """``txt`` as a Sketch in local XY: exact faces bounded by the font's lines and Beziers."""
    from ._kernel import call, ub
    from .sketch import Sketch

    glyphs = text_segments(txt, size, font, font_path, aspect, halign, valign)
    if not glyphs:
        raise KeelError(f"Text: {txt!r} has no visible glyphs")
    boxes = [_box(gl) for gl in glyphs]
    overlap = any(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
                  for i, a in enumerate(boxes) for b in boxes[i + 1:])
    if not overlap:  # separate glyphs: one call makes every face, holes found by nesting
        return Sketch(call("Text", ub.create_planar_faces, [_curve(s) for gl in glyphs for c in gl for s in c]))
    sketch = None
    for gl in glyphs:  # glyphs that may touch: one face set each, then their union
        part = Sketch(call("Text", ub.create_planar_faces, [_curve(s) for c in gl for s in c]))
        sketch = part if sketch is None else sketch + part
    return sketch
