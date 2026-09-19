"""Figure 1: the architecture, in the slide deck's design language.

Follows paper/Presentation "Proposed Architecture": pastel stage bands with a
bold title at the top left, rounded boxes with coloured strokes inside them,
small italic grey annotations underneath, dashed arrows for the label-free
update path, and the input drawn as a waveform thumbnail with its shape beneath.
Flow is left to right.

Content is the reported model: a window enters at the left and splits into the
multi-scale power stem, which reads the diagonal of its spatial covariance as
log band power, and the tangent-space branch, which reads the off-diagonal
structure at a running reference. The two rejoin into one linear classifier.

Parameter counts are printed per block and sum to the total in the footer.

Writes results/fig_arch.pdf and .png.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
try:
    import fontTools.varLib  # noqa: F401
    matplotlib.rcParams["pdf.fonttype"] = 42
except Exception:                 # blocked TrueType subsetter on this machine
    matplotlib.rcParams["pdf.fonttype"] = 3
matplotlib.rcParams["ps.fonttype"] = matplotlib.rcParams["pdf.fonttype"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS = Path(__file__).resolve().parent.parent / "results"

INK = "#262626"
NOTE = "#7F7F7F"
# stage bands, as pastel fills with no stroke
BAND_STEM = "#F4F4F4"
BAND_BR = "#DCE6F1"
BAND_OUT = "#FDF2CC"
# box strokes
G = "#70AD47"      # stem convolutional blocks
B = "#2E75B6"      # branch blocks
Y = "#BF9000"      # output blocks
FILL_M = "#9DC3E6"    # the running reference, filled as M was in the slide
FILL_EMB = "#FBE5D6"  # the embedding box, peach as in the slide
FILL_Y = "#FFE699"

FS, FSS, FST = 7.2, 6.1, 8.0


def band(ax, x, y, w, h, title, colour):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.016",
        linewidth=0, facecolor=colour, zorder=1))
    ax.text(x + 0.014, y + h - 0.052, title, ha="left", va="center",
            fontsize=FST, color=INK, fontweight="bold", zorder=4)


def box(ax, x, y, w, h, label, sub=None, params=None, stroke=INK,
        fill="white", dashed=False, r=0.010):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=%g" % r,
        linewidth=1.1, edgecolor=stroke, facecolor=fill, zorder=3,
        linestyle=(0, (4, 2)) if dashed else "solid"))
    cy = y + h / 2 + (0.020 if sub else 0) + (0.017 if params else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center", fontsize=FS,
            color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, cy - 0.036, sub, ha="center", va="center",
                fontsize=FSS, color=INK, zorder=4)
    if params:
        ax.text(x + w / 2, cy - (0.064 if sub else 0.034),
                "{:,}".format(params), ha="center", va="center", fontsize=FSS,
                color=NOTE, zorder=4)


def note(ax, x, y, text, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=FSS, color=NOTE,
            style="italic", zorder=4)


def arrow(ax, x0, y0, x1, y1, colour=INK, dashed=False, lw=1.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, linewidth=lw,
        color=colour, zorder=5, shrinkA=0, shrinkB=0,
        linestyle=(0, (4, 2)) if dashed else "solid"))


def waveform(ax, x, y, w, h, n=7):
    rng = np.random.default_rng(3)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0.003", linewidth=0.9,
        edgecolor=INK, facecolor="white", zorder=3))
    t = np.linspace(0, 1, 220)
    for i in range(n):
        s = rng.standard_normal(220).cumsum()
        s = s / (np.abs(s).max() + 1e-9)
        ax.plot(x + 0.012 + t * (w - 0.024),
                y + h * (i + 0.5) / n + s * (h / n) * 0.34,
                color="#5B9BD5", lw=0.45, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(6.84, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    TOPY, BOTY = 0.570, 0.075          # band bottoms
    BH_T, BH_B = 0.350, 0.430          # band heights (branch carries M below)
    ROW_T, ROW_B = 0.630, 0.285        # box bottoms
    H = 0.145

    # ------------------------------------------------------------- input
    waveform(ax, 0.006, 0.395, 0.086, 0.150)
    note(ax, 0.049, 0.366, r"EEG   $60 \times 400$")
    note(ax, 0.049, 0.336, "4 s @ 100 Hz")
    for yy in (ROW_T + H / 2, ROW_B + H / 2):
        ax.plot([0.094, 0.110, 0.110], [0.470, 0.470, yy], color=INK, lw=1.0,
                zorder=5)
        arrow(ax, 0.110, yy, 0.128, yy)

    # ------------------------------------------- stage 1: the power stem
    band(ax, 0.120, TOPY, 0.500, BH_T, "Multi-Scale Power Stem", BAND_STEM)
    for i, ms in enumerate(("64 ms", "128 ms", "256 ms")):
        box(ax, 0.134, ROW_T + 0.086 - i * 0.052, 0.054, 0.044, ms, stroke=G)
        arrow(ax, 0.188, ROW_T + 0.108 - i * 0.052, 0.206, ROW_T + H / 2)
    note(ax, 0.166, ROW_T - 0.052, "temporal conv, three scales")
    box(ax, 0.206, ROW_T, 0.074, H, "depthwise", "spatial", stroke=G)
    for j, lab in enumerate(("square", "pool", "log")):
        w = (0.060, 0.046, 0.040)[j]
        x = (0.292, 0.362, 0.418)[j]
        box(ax, x, ROW_T + 0.028, w, 0.090, lab)
        arrow(ax, x - 0.014, ROW_T + H / 2, x, ROW_T + H / 2)
    box(ax, 0.464, ROW_T, 0.140, H, "frame embedding", r"$\rightarrow d = 128$",
        19505, stroke="#C55A11", fill=FILL_EMB)
    arrow(ax, 0.458, ROW_T + H / 2, 0.464, ROW_T + H / 2)

    # ----------------------------------------- stage 2: the tangent branch
    band(ax, 0.120, BOTY, 0.500, BH_B, "Tangent-Space Branch", BAND_BR)
    box(ax, 0.134, ROW_B, 0.098, H, "covariance", r"$C = XX^{\top}\!/T$",
        stroke=B)
    box(ax, 0.248, ROW_B, 0.078, H, "shrinkage", r"$\lambda = 0.1$", stroke=B)
    box(ax, 0.342, ROW_B, 0.098, H, "tangent map", "at reference $M$",
        stroke=B)
    box(ax, 0.456, ROW_B, 0.148, H, "project", r"$1830 \rightarrow 128$",
        238028, stroke=B)
    for a_, b_ in ((0.232, 0.248), (0.326, 0.342), (0.440, 0.456)):
        arrow(ax, a_, ROW_B + H / 2, b_, ROW_B + H / 2)

    # the running reference sits below the map it feeds
    box(ax, 0.357, 0.120, 0.068, 0.090, r"$\mathbf{M}$", stroke=B,
        fill=FILL_M)
    arrow(ax, 0.391, 0.210, 0.391, ROW_B)
    ax.plot([0.183, 0.183, 0.357], [ROW_B, 0.165, 0.165], color=INK, lw=0.9,
            ls=(0, (4, 2)), zorder=4)
    note(ax, 0.442, 0.184, "unsupervised update,", ha="left")
    note(ax, 0.442, 0.154, "active at inference", ha="left")

    # --------------------------------------- stage 3: fusion and decision
    band(ax, 0.645, BOTY, 0.348, TOPY - BOTY + BH_T, "Fusion and Decision",
         BAND_OUT)
    box(ax, 0.664, 0.395, 0.084, 0.150, "concat", "$256$", stroke=Y,
        fill=FILL_Y)
    box(ax, 0.772, 0.395, 0.076, 0.150, "fuse", r"$\rightarrow 128$", 32896,
        stroke=Y, fill=FILL_Y)
    box(ax, 0.872, 0.395, 0.104, 0.150, "classifier", "Walk / Stop", 514,
        stroke=Y, fill=FILL_Y)
    for yy in (ROW_T + H / 2, ROW_B + H / 2):
        ax.plot([0.604, 0.632, 0.632], [yy, yy, 0.470], color=INK, lw=1.0,
                zorder=5)
    arrow(ax, 0.632, 0.470, 0.664, 0.470)
    arrow(ax, 0.748, 0.470, 0.772, 0.470)
    arrow(ax, 0.848, 0.470, 0.872, 0.470)
    note(ax, 0.924, 0.366, "posterior, ECE 0.049")

    ax.text(0.006, 0.020,
            "290,943 parameters: 19,505 stem path, 238,028 branch, "
            "32,896 fusion, 514 output.",
            ha="left", fontsize=FSS, color=NOTE)

    fig.tight_layout(pad=0.15)
    for ext in ("pdf", "png"):
        out = RESULTS / ("fig_arch.%s" % ext)
        fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=220)
        print("wrote %s" % out)


if __name__ == "__main__":
    main()
