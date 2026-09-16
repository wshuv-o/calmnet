"""Render a draw.io (mxGraph) file to vector PDF with matplotlib.

Why this exists: draw.io desktop is a 109 MB Electron app whose portable Windows
build does not run headless here (it ships an incomplete V8 snapshot), and
drawio-batch has been unpublished from npm. Rather than hand-drawing the figure
in Python and keeping two sources that drift apart, this renders the .drawio
file itself, so paper/fig_arch.drawio stays the single editable source of truth
and the PDF in the paper regenerates from it.

It supports the mxGraph subset used by paper/fig_arch.drawio: rounded and square
rectangles, ellipses, cubes, parallelograms, lines, text, and orthogonal edges
with optional waypoints. Styling honoured: fillColor, strokeColor, strokeWidth,
dashed/dashPattern, fontSize, fontColor, fontStyle (bold/italic), align, arcSize
and shape size. Labels may contain <b>, <i>, <br>, <sup>, <font> and HTML
entities; tags are stripped and <br> becomes a line break.

Usage:  python render_drawio.py ../paper/fig_arch.drawio ../results/fig_arch.pdf
"""
from __future__ import annotations

import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch, Polygon, Rectangle)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

S.use()

# draw.io units are pixels at 96 dpi; the paper wants a fixed physical width.
PX_PER_IN = 96.0

# Set by render(): points per draw.io pixel, so that a label declared at F px in
# the .drawio renders at the size it would have if the canvas were printed at
# the requested physical width. Without this, text is drawn at absolute point
# sizes on a shrunken canvas and swamps the shapes.
FSCALE = 1.0


def parse_style(s):
    """'rounded=1;fillColor=#fff;' -> dict, with bare tokens mapped to True."""
    out = {}
    for part in (s or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[part] = True
    return out


def clean_label(raw):
    """HTML label -> plain text with newlines; returns (text, bold, italic)."""
    if not raw:
        return "", False, False
    t = raw
    bold = bool(re.search(r"<b>|<strong>|font-weight:\s*bold", t, re.I))
    italic = bool(re.search(r"<i>|<em>", t, re.I))
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"</?div[^>]*>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = "\n".join(line.strip() for line in t.split("\n"))
    return t.strip("\n"), bold, italic


def col(style, key, default=None):
    v = style.get(key)
    if v in (None, "none", "None"):
        return None if v is not None else default
    return v


def dash_of(style):
    if str(style.get("dashed", "0")) != "1":
        return None
    pat = style.get("dashPattern")
    if pat:
        try:
            nums = [float(x) for x in pat.split()]
            return (0, tuple(nums))
        except ValueError:
            pass
    return (0, (4, 3))


class Doc:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        model = root.find(".//mxGraphModel")
        self.W = float(model.get("pageWidth", 1600))
        self.H = float(model.get("pageHeight", 800))
        self.cells = {}
        self.order = []
        for c in model.find("root"):
            cid = c.get("id")
            self.cells[cid] = c
            self.order.append(cid)

    def geom(self, cell):
        g = cell.find("mxGeometry")
        if g is None:
            return None
        try:
            return (float(g.get("x", 0)), float(g.get("y", 0)),
                    float(g.get("width", 0)), float(g.get("height", 0)))
        except (TypeError, ValueError):
            return None

    def y(self, v):
        """Flip draw.io's y-down origin to matplotlib's y-up."""
        return self.H - v


def draw_vertex(ax, doc, cell):
    st = parse_style(cell.get("style"))
    g = doc.geom(cell)
    if g is None:
        return
    x, y, w, h = g
    yb = doc.y(y + h)                      # bottom edge in flipped coords
    fill = col(st, "fillColor")
    stroke = col(st, "strokeColor")
    lw = float(st.get("strokeWidth", 1))
    dash = dash_of(st)
    is_text = "text" in st or st.get("shape") == "text"

    if not is_text and (fill or stroke):
        fc = fill if fill else "none"
        ec = stroke if stroke else "none"
        kw = dict(facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
        if dash:
            kw["linestyle"] = dash

        if "ellipse" in st:
            ax.add_patch(Ellipse((x + w / 2, yb + h / 2), w, h, **kw))
        elif st.get("shape") == "parallelogram":
            d = float(st.get("size", 20))
            ax.add_patch(Polygon([(x + d, yb), (x + w, yb),
                                  (x + w - d, yb + h), (x, yb + h)],
                                 closed=True, **kw))
        elif st.get("shape") == "cube":
            d = float(st.get("size", 12))
            # front face, then the two visible side faces, lightly shaded
            ax.add_patch(Rectangle((x, yb), w - d, h - d, **kw))
            side = dict(kw)
            side["facecolor"] = fc
            side["alpha"] = 0.55
            ax.add_patch(Polygon([(x + w - d, yb), (x + w, yb + d),
                                  (x + w, yb + h), (x + w - d, yb + h - d)],
                                 closed=True, **side))
            ax.add_patch(Polygon([(x, yb + h - d), (x + w - d, yb + h - d),
                                  (x + w, yb + h), (x + d, yb + h)],
                                 closed=True, **side))
        elif st.get("shape") == "line" or "line" in st:
            ax.plot([x, x + w], [yb + h / 2, yb + h / 2],
                    color=ec if ec != "none" else "#BBBBBB", lw=lw, zorder=2)
        elif str(st.get("rounded", "0")) == "1":
            r = min(float(st.get("arcSize", 12)) / 100.0 * min(w, h), min(w, h) / 2)
            ax.add_patch(FancyBboxPatch(
                (x + r, yb + r), w - 2 * r, h - 2 * r,
                boxstyle="round,pad=%f,rounding_size=%f" % (r, r), **kw))
        else:
            ax.add_patch(Rectangle((x, yb), w, h, **kw))

    text, b, i = clean_label(cell.get("value"))
    if not text:
        return
    fs = float(st.get("fontSize", 11)) * FSCALE
    fcol = col(st, "fontColor", "#000000") or "#000000"
    fstyle = int(st.get("fontStyle", 0) or 0)
    weight = "bold" if (b or fstyle & 1) else "normal"
    slant = "italic" if (i or fstyle & 2) else "normal"
    align = st.get("align", "center")
    ha = {"left": "left", "right": "right", "center": "center"}.get(align, "center")
    tx = {"left": x + 4, "right": x + w - 4, "center": x + w / 2}[ha]
    ax.text(tx, yb + h / 2, text, ha=ha, va="center", fontsize=fs,
            color=fcol, fontweight=weight, style=slant, zorder=6,
            linespacing=1.35)


def anchor(doc, cell, side):
    g = doc.geom(cell)
    if g is None:
        return None
    x, y, w, h = g
    yb = doc.y(y + h)
    return {"c": (x + w / 2, yb + h / 2), "l": (x, yb + h / 2),
            "r": (x + w, yb + h / 2), "t": (x + w / 2, yb + h),
            "b": (x + w / 2, yb)}[side]


def draw_edge(ax, doc, cell):
    st = parse_style(cell.get("style"))
    src = doc.cells.get(cell.get("source"))
    tgt = doc.cells.get(cell.get("target"))
    g = cell.find("mxGeometry")
    pts = []
    if g is not None:
        arr = g.find("Array")
        if arr is not None:
            for p in arr.findall("mxPoint"):
                pts.append((float(p.get("x", 0)), doc.y(float(p.get("y", 0)))))

    def loose(el, tag):
        if el is None or g is None:
            return None
        p = g.find("mxPoint[@as='%s']" % tag)
        return (float(p.get("x")), doc.y(float(p.get("y")))) if p is not None else None

    p0 = loose(None, "sourcePoint") if src is None else None
    p1 = loose(None, "targetPoint") if tgt is None else None
    if src is not None and tgt is not None:
        sc, tc = anchor(doc, src, "c"), anchor(doc, tgt, "c")
        if sc is None or tc is None:
            return
        # pick the faces that face each other
        dx, dy = tc[0] - sc[0], tc[1] - sc[1]
        if abs(dx) >= abs(dy):
            p0 = anchor(doc, src, "r" if dx > 0 else "l")
            p1 = anchor(doc, tgt, "l" if dx > 0 else "r")
        else:
            p0 = anchor(doc, src, "t" if dy > 0 else "b")
            p1 = anchor(doc, tgt, "b" if dy > 0 else "t")
    if p0 is None or p1 is None:
        return

    color = col(st, "strokeColor", "#333333") or "#333333"
    lw = float(st.get("strokeWidth", 1.2))
    dash = dash_of(st)
    chain = [p0] + pts + [p1]
    for a, b in zip(chain[:-2], chain[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                linestyle=dash if dash else "-", zorder=3,
                solid_capstyle="round")
    a, b = chain[-2], chain[-1]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=color, zorder=3,
                                 linestyle=dash if dash else "-",
                                 shrinkA=0, shrinkB=0))
    text, _, _ = clean_label(cell.get("value"))
    if text:
        mx = (chain[len(chain) // 2 - 1][0] + chain[len(chain) // 2][0]) / 2
        my = (chain[len(chain) // 2 - 1][1] + chain[len(chain) // 2][1]) / 2
        ax.text(mx, my, text, ha="center", va="bottom",
                fontsize=float(st.get("fontSize", 10)) * FSCALE,
                color=col(st, "fontColor", color), zorder=6, linespacing=1.3)


def render(src, out, width_in=S.FULL):
    global FSCALE
    doc = Doc(src)
    # px -> pt (72/96), then the shrink from the natural canvas width to the
    # requested physical width.
    FSCALE = 0.75 * (width_in / (doc.W / PX_PER_IN))
    print("  canvas %.0f x %.0f px, natural %.2f in, rendering at %.2f in, "
          "font scale %.3f" % (doc.W, doc.H, doc.W / PX_PER_IN, width_in, FSCALE))
    fig, ax = plt.subplots(figsize=(width_in, width_in * doc.H / doc.W))
    ax.set_xlim(0, doc.W)
    ax.set_ylim(0, doc.H)
    ax.set_aspect("equal")
    ax.axis("off")
    for cid in doc.order:
        c = doc.cells[cid]
        if c.get("vertex") == "1":
            draw_vertex(ax, doc, c)
    for cid in doc.order:
        c = doc.cells[cid]
        if c.get("edge") == "1":
            draw_edge(ax, doc, c)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=400,
                bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("rendered %s -> %s (+ .png)" % (Path(src).name, Path(out).name))


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "../paper/fig_arch.drawio"
    b = sys.argv[2] if len(sys.argv) > 2 else "../results/fig_arch.pdf"
    render(a, b)
