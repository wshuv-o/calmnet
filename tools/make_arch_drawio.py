r"""Emit the architecture figure as an editable draw.io file.

Same content, colours and left-to-right layout as results/fig_arch.pdf, written
as mxGraph XML so the figure can be edited by hand in diagrams.net rather than
only regenerated from matplotlib.

Open paper/fig_arch.drawio at app.diagrams.net or in the VS Code Draw.io
extension. Exporting to PDF from there replaces results/fig_arch.pdf directly;
nothing in the manuscript needs changing.

    python tools/make_arch_drawio.py
"""
from __future__ import annotations

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "fig_arch.drawio")

BAND_STEM, BAND_BR, BAND_OUT = "#F4F4F4", "#DCE6F1", "#FDF2CC"
G, B, Y, ORANGE = "#70AD47", "#2E75B6", "#BF9000", "#C55A11"
FILL_M, FILL_EMB, FILL_Y = "#9DC3E6", "#FBE5D6", "#FFE699"
INK, NOTE = "#262626", "#7F7F7F"

BAND = ("rounded=1;arcSize=6;fillColor=%s;strokeColor=none;"
        "verticalAlign=top;align=left;spacingLeft=14;spacingTop=10;"
        "fontSize=15;fontStyle=1;fontColor=" + INK + ";html=1;")
BOX = ("rounded=1;arcSize=18;fillColor=%s;strokeColor=%s;strokeWidth=2;"
       "fontSize=13;fontColor=" + INK + ";html=1;whiteSpace=wrap;")
NOTE_S = ("text;html=1;align=%s;verticalAlign=middle;fontSize=11;"
          "fontStyle=2;fontColor=" + NOTE + ";")
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;"
        "endFill=1;strokeColor=" + INK + ";strokeWidth=1.6;")
EDGE_D = EDGE + "dashed=1;dashPattern=6 4;"

cells = []


def node(id_, x, y, w, h, style, label=""):
    cells.append(
        '        <mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">\n'
        '          <mxGeometry x="%d" y="%d" width="%d" height="%d" '
        'as="geometry"/>\n        </mxCell>' % (id_, label, style, x, y, w, h))


def edge(id_, src, dst, style=EDGE):
    cells.append(
        '        <mxCell id="%s" style="%s" edge="1" parent="1" source="%s" '
        'target="%s">\n          <mxGeometry relative="1" as="geometry"/>\n'
        '        </mxCell>' % (id_, style, src, dst))


def main():
    # ------------------------------------------------------------- bands
    node("bandStem", 170, 40, 700, 235, BAND % BAND_STEM,
         "Multi-Scale Power Stem")
    node("bandBr", 170, 300, 700, 285, BAND % BAND_BR, "Tangent-Space Branch")
    node("bandOut", 900, 40, 505, 545, BAND % BAND_OUT, "Fusion and Decision")

    # ------------------------------------------------------------- input
    node("eeg", 20, 250, 125, 105,
         "rounded=0;fillColor=#FFFFFF;strokeColor=" + INK + ";strokeWidth=2;"
         "fontSize=11;html=1;", "EEG")
    node("eegNote", 20, 360, 125, 40, NOTE_S % "center",
         "60 &#215; 400&lt;br&gt;4 s @ 100 Hz")

    # -------------------------------------------------------- stem blocks
    for i, ms in enumerate(("64 ms", "128 ms", "256 ms")):
        node("sc%d" % i, 195, 95 + i * 52, 80, 42, BOX % ("#FFFFFF", G), ms)
    node("dw", 300, 108, 115, 85, BOX % ("#FFFFFF", G),
         "depthwise&lt;br&gt;&lt;font style='font-size:11px'&gt;spatial"
         "&lt;/font&gt;")
    node("scNote", 185, 205, 190, 30, NOTE_S % "left",
         "temporal conv, three scales")
    node("sq", 440, 122, 78, 58, BOX % ("#FFFFFF", INK), "square")
    node("po", 535, 122, 64, 58, BOX % ("#FFFFFF", INK), "pool")
    node("lg", 615, 122, 56, 58, BOX % ("#FFFFFF", INK), "log")
    node("emb", 690, 108, 160, 85, BOX % (FILL_EMB, ORANGE),
         "frame embedding&lt;br&gt;&lt;font style='font-size:11px'&gt;"
         "&#8594; d = 128&lt;/font&gt;&lt;br&gt;"
         "&lt;font style='font-size:11px;color:#7F7F7F'&gt;19,505&lt;/font&gt;")

    # ------------------------------------------------------ branch blocks
    node("cov", 195, 370, 145, 85, BOX % ("#FFFFFF", B),
         "covariance&lt;br&gt;&lt;font style='font-size:11px'&gt;"
         "C = XX&#8868;/T&lt;/font&gt;")
    node("shr", 365, 370, 120, 85, BOX % ("#FFFFFF", B),
         "shrinkage&lt;br&gt;&lt;font style='font-size:11px'&gt;"
         "&#955; = 0.1&lt;/font&gt;")
    node("tan", 510, 370, 155, 85, BOX % ("#FFFFFF", B),
         "tangent map&lt;br&gt;&lt;font style='font-size:11px'&gt;"
         "at reference M&lt;/font&gt;")
    node("prj", 690, 370, 160, 85, BOX % ("#FFFFFF", B),
         "project&lt;br&gt;&lt;font style='font-size:11px'&gt;"
         "1830 &#8594; 128&lt;/font&gt;&lt;br&gt;"
         "&lt;font style='font-size:11px;color:#7F7F7F'&gt;238,028&lt;/font&gt;")
    node("M", 545, 490, 85, 60, BOX % (FILL_M, B), "&lt;b&gt;M&lt;/b&gt;")
    node("mNote", 650, 495, 210, 45, NOTE_S % "left",
         "unsupervised update,&lt;br&gt;active at inference")

    # ------------------------------------------------------ fusion blocks
    node("cat", 930, 270, 125, 85, BOX % (FILL_Y, Y),
         "concat&lt;br&gt;&lt;font style='font-size:11px'&gt;256&lt;/font&gt;")
    node("fus", 1090, 270, 115, 85, BOX % (FILL_Y, Y),
         "fuse&lt;br&gt;&lt;font style='font-size:11px'&gt;&#8594; 128"
         "&lt;/font&gt;&lt;br&gt;&lt;font style='font-size:11px;"
         "color:#7F7F7F'&gt;32,896&lt;/font&gt;")
    node("cls", 1240, 270, 145, 85, BOX % (FILL_Y, Y),
         "classifier&lt;br&gt;&lt;font style='font-size:11px'&gt;Walk / Stop"
         "&lt;/font&gt;&lt;br&gt;&lt;font style='font-size:11px;"
         "color:#7F7F7F'&gt;514&lt;/font&gt;")
    node("outNote", 1240, 360, 150, 30, NOTE_S % "center",
         "posterior, ECE 0.049")

    node("total", 20, 605, 900, 30, NOTE_S % "left",
         "290,943 parameters: 19,505 stem path, 238,028 branch, "
         "32,896 fusion, 514 output.")

    # -------------------------------------------------------------- edges
    for i in range(3):
        edge("e_sc%d" % i, "sc%d" % i, "dw")
    for a, b_ in (("eeg", "sc1"), ("eeg", "cov"), ("dw", "sq"), ("sq", "po"),
                  ("po", "lg"), ("lg", "emb"), ("cov", "shr"),
                  ("shr", "tan"), ("tan", "prj"), ("M", "tan"),
                  ("emb", "cat"), ("prj", "cat"), ("cat", "fus"),
                  ("fus", "cls")):
        edge("e_%s_%s" % (a, b_), a, b_)
    edge("e_upd", "cov", "M", EDGE_D)

    xml = ('<mxfile host="app.diagrams.net" type="device">\n'
           '  <diagram name="Architecture" id="arch">\n'
           '    <mxGraphModel dx="1600" dy="900" grid="0" gridSize="10" '
           'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
           'pageScale="1" pageWidth="1450" pageHeight="680" math="0" '
           'shadow="0">\n      <root>\n'
           '        <mxCell id="0"/>\n'
           '        <mxCell id="1" parent="0"/>\n'
           + "\n".join(cells) + "\n"
           '      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n')
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(xml)
    print("wrote %s  (%d cells, %.1f KB)"
          % (os.path.relpath(OUT, ROOT), len(cells), len(xml) / 1024))


if __name__ == "__main__":
    main()
