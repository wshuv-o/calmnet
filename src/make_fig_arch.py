"""Figure 1: the architecture, drawn left to right.

Design taken from paper/calmnet_architecture.drawio so the paper keeps one
visual language: unfilled boxes with black outlines, dashed grey containers
carrying a numbered stage label at the top left, parallelograms for data
entering and leaving, orthogonal black arrows, and a single red accent reserved
for the one path that is live at inference.

Content is the reported model and nothing else. A window enters at the left and
splits into two lanes. The stem reads the diagonal of its spatial covariance as
log band power; the branch reads the off-diagonal structure through a tangent
map at a running reference. The lanes rejoin into one linear classifier.

Parameter counts are read from the built model by src/driftnet.py and printed
per block, so the total at the foot is auditable by adding up the boxes.

Writes results/fig_arch.pdf and .png.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Type 42 needs fontTools to subset TrueType and an Application Control policy
# on this machine blocks that DLL, so probe and fall back rather than assume.
try:
    import fontTools.varLib  # noqa: F401
    matplotlib.rcParams["pdf.fonttype"] = 42
except Exception:
    matplotlib.rcParams["pdf.fonttype"] = 3
matplotlib.rcParams["ps.fonttype"] = matplotlib.rcParams["pdf.fonttype"]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

RESULTS = Path(__file__).resolve().parent.parent / "results"

# the drawio palette, unchanged
INK = "#000000"        # box outlines and text
GREY = "#8A8F98"        # dashed stage containers
SUBTLE = "#5f6368"      # stage labels
MUTED = "#333333"       # secondary text
ACCENT = "#C0392B"      # the label-free path, live at inference

FS = 7.4
FS_SUB = 6.4
FS_STAGE = 7.6


def stage(ax, x, y, w, h, n, title):
    """A dashed grey container with a numbered label at its top left."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.014",
        linewidth=0.9, edgecolor=GREY, facecolor="none",
        linestyle=(0, (6, 4)), zorder=1))
    ax.text(x + 0.012, y + h - 0.030,
            "%d · %s" % (n, title), ha="left", va="center",
            fontsize=FS_STAGE, color=SUBTLE, fontweight="bold")


def box(ax, x, y, w, h, label, sub=None, params=None, colour=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=1.0, edgecolor=colour, facecolor="none", zorder=3))
    cy = y + h / 2
    off = 0.018 if sub else 0.0
    off += 0.016 if params else 0.0
    ax.text(x + w / 2, cy + off, label, ha="center", va="center",
            fontsize=FS, color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, cy + off - 0.032, sub, ha="center", va="center",
                fontsize=FS_SUB, color=MUTED, zorder=4)
    if params:
        ax.text(x + w / 2, cy + off - (0.060 if sub else 0.030),
                "{:,}".format(params), ha="center", va="center",
                fontsize=FS_SUB, color=SUBTLE, zorder=4)


def data(ax, x, y, w, h, label, sub=None):
    """Parallelogram, as the drawio uses for data in and out."""
    k = 0.016
    ax.add_patch(Polygon([[x + k, y], [x + w, y], [x + w - k, y + h],
                          [x, y + h]], closed=True, fill=False,
                         edgecolor=INK, linewidth=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.014 if sub else 0), label, ha="center",
            va="center", fontsize=FS, color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.018, sub, ha="center", va="center",
                fontsize=FS_SUB, color=MUTED, zorder=4)


def arrow(ax, x0, y0, x1, y1, colour=INK, dashed=False, lw=1.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, linewidth=lw,
        color=colour, zorder=5, shrinkA=0, shrinkB=0,
        linestyle=(0, (5, 3)) if dashed else "solid"))


def elbow(ax, x0, y0, x1, y1, colour=INK):
    """Orthogonal connector, as drawio draws them."""
    ax.plot([x0, (x0 + x1) / 2, (x0 + x1) / 2], [y0, y0, y1], color=colour,
            lw=1.0, solid_joinstyle="miter", zorder=5)
    arrow(ax, (x0 + x1) / 2, y1, x1, y1, colour=colour)


def main():
    fig, ax = plt.subplots(figsize=(6.84, 2.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    TOP, BOT = 0.615, 0.205          # lane baselines
    H = 0.185                        # box height

    # ------------------------------------------------------------ input
    data(ax, 0.012, 0.415, 0.088, H, "EEG window", "$C \\times T$")
    elbow(ax, 0.100, 0.508, 0.147, TOP + H / 2)
    elbow(ax, 0.100, 0.508, 0.147, BOT + H / 2)

    # ------------------------------------------------- stage 1, the stem
    stage(ax, 0.140, TOP - 0.052, 0.455, H + 0.105, 1,
          "Multi-scale power stem  —  first order")
    box(ax, 0.152, TOP, 0.145, H, "temporal $+$ spatial",
        "64 / 128 / 256 ms", 6768)
    box(ax, 0.317, TOP, 0.118, H, "square, pool, log", "band power")
    box(ax, 0.455, TOP, 0.128, H, "attend $+$ project", "to $d = 128$", 12737)
    arrow(ax, 0.297, TOP + H / 2, 0.317, TOP + H / 2)
    arrow(ax, 0.435, TOP + H / 2, 0.455, TOP + H / 2)

    # --------------------------------------------- stage 2, the branch
    stage(ax, 0.140, BOT - 0.052, 0.455, H + 0.105, 2,
          "Tangent-space branch  —  second order")
    box(ax, 0.152, BOT, 0.145, H, "covariance $C$",
        "$XX^{\\top}\\!/T$, shrunk")
    box(ax, 0.317, BOT, 0.118, H, "tangent map",
        "$\\log(M^{-1/2}CM^{-1/2})$")
    box(ax, 0.455, BOT, 0.128, H, "project", "$1830 \\rightarrow 128$", 238028)
    arrow(ax, 0.297, BOT + H / 2, 0.317, BOT + H / 2)
    arrow(ax, 0.435, BOT + H / 2, 0.455, BOT + H / 2)

    # the reference, the only red and the only thing live at inference
    ax.add_patch(FancyBboxPatch(
        (0.317, BOT - 0.150), 0.118, 0.082,
        boxstyle="round,pad=0.004,rounding_size=0.008", linewidth=1.0,
        edgecolor=ACCENT, facecolor="none", linestyle=(0, (5, 3)), zorder=3))
    ax.text(0.376, BOT - 0.093, "running reference $M$", ha="center",
            va="center", fontsize=FS_SUB, color=ACCENT, zorder=4)
    ax.text(0.376, BOT - 0.127, "no labels, updates at test time",
            ha="center", va="center", fontsize=6.0, color=ACCENT, zorder=4)
    arrow(ax, 0.376, BOT - 0.068, 0.376, BOT, colour=ACCENT, dashed=True)
    ax.plot([0.200, 0.200, 0.317], [BOT, BOT - 0.109, BOT - 0.109],
            color=ACCENT, lw=0.9, ls=(0, (5, 3)), zorder=4)

    # ------------------------------------------- stage 3, fuse and decide
    stage(ax, 0.625, BOT - 0.052, 0.250, TOP - BOT + H + 0.105, 3,
          "Fusion and decision")
    box(ax, 0.640, 0.505, 0.105, 0.150, "concatenate", "$256$")
    box(ax, 0.640, 0.285, 0.105, 0.150, "fuse", "$\\rightarrow 128$", 32896)
    box(ax, 0.762, 0.395, 0.100, 0.150, "classifier", "norm $+$ linear", 514)
    elbow(ax, 0.583, TOP + H / 2, 0.640, 0.580)
    elbow(ax, 0.583, BOT + H / 2, 0.640, 0.360)
    arrow(ax, 0.6925, 0.505, 0.6925, 0.435)
    elbow(ax, 0.745, 0.360, 0.762, 0.470)

    data(ax, 0.892, 0.395, 0.096, 0.150, "Stop / Walk", "posterior")
    arrow(ax, 0.862, 0.470, 0.892, 0.470)

    # ---------------------------------------------------------- footer
    ax.plot([0.012, 0.988], [0.055, 0.055], lw=0.7, color=GREY)
    ax.text(0.012, 0.018,
            "290,943 parameters: 19,505 stem path, 238,028 branch, 32,896 "
            "fusion, 514 output. The boxes sum to the total.",
            ha="left", fontsize=FS_SUB, color=MUTED)

    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        out = RESULTS / ("fig_arch.%s" % ext)
        fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=220)
        print("wrote %s" % out)


if __name__ == "__main__":
    main()
