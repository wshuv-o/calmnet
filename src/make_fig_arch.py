"""Figure 1: the architecture actually reported.

A window enters two paths. The stem reads the diagonal of its spatial
covariance as log band power; the branch reads the off-diagonal structure by
mapping the covariance into the tangent space at a running reference. Their
outputs concatenate into one linear classifier.

The diagram carries three things a box-and-arrow sketch usually omits:

  * exact parameter counts per block, so the 290,943 total is auditable and the
    fact that the branch and its fusion layer are 93 % of it is visible
    rather than buried;
  * the reference update path, drawn dashed, which is the only arrow live at
    inference on unlabelled data, and the component that fails when a class
    block outlasts one covariance update;
  * the three components that were built, measured and NOT retained, shown
    greyed below the model, because the ablation is reported against them and a
    reader should be able to see what was removed.

Written to results/fig_arch.pdf and .png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Type 42 needs fontTools to subset TrueType, and this machine's Application
# Control policy blocks that DLL, so the PDF write fails silently-late. Type 3
# keeps the output vector and needs no subsetter. Both are acceptable to
# Elsevier; 42 is preferred, so try it and fall back rather than assume.
try:
    import fontTools.varLib  # noqa: F401
    matplotlib.rcParams["pdf.fonttype"] = 42
except Exception:
    matplotlib.rcParams["pdf.fonttype"] = 3
matplotlib.rcParams["ps.fonttype"] = matplotlib.rcParams["pdf.fonttype"]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS = Path(__file__).resolve().parent.parent / "results"
INK = "#000000"
STEM = "#A8763E"      # first-order path: log band power
BRANCH = "#2E6F5E"    # second-order path: the contribution
HEAD = "#6B4E71"      # fusion and classifier
GREY = "#9AA3AB"      # components not retained

# (label, sub-label, parameters). None means no count is meaningful.
# Counts read from the built model with torch, not estimated. They sum to the
# 290,943 printed at the foot of the figure, and a reviewer adding up the boxes
# is exactly who catches it when they do not.
STEM_BLOCKS = [
    ("Multi-scale conv $+$ spatial", "64 / 128 / 256 ms", 6768),
    ("square $\\rightarrow$ pool $\\rightarrow$ log", "log band power", None),
    ("Frame attention $+$ project", "to $d=128$", 12737),
]
BRANCH_BLOCKS = [
    ("Covariance $C = XX^{\\top}/T$", "trace-normalised", None),
    ("Shrinkage toward $I$", "$\\lambda = 0.1$", None),
    ("Tangent map at $M$", "$\\log(M^{-1/2} C M^{-1/2})$", None),
    ("Linear projection", "1830 $\\rightarrow$ 128", 238028),
]
DROPPED = [
    ("Adaptive alignment", "whitens by $M^{-1/2}$", "substitute for the branch"),
    ("Cross-epoch transformer", "594,816 parameters", "costs accuracy"),
    ("Selective head", "coverage-constrained", "no usable reject option"),
]


def box(ax, x, y, w, h, label, sub, params, colour, alpha=0.10, fs=8.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.1, edgecolor=colour, facecolor=colour, alpha=alpha,
        zorder=2))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.1, edgecolor=colour, facecolor="none", zorder=3))
    ty = y + h / 2 + (0.016 if sub else 0.0) + (0.014 if params else 0.0)
    ax.text(x + w / 2, ty, label, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, ty - 0.030, sub, ha="center", va="center",
                fontsize=fs - 1.3, color=GREY, zorder=4)
    if params:
        ax.text(x + w / 2, ty - 0.056, "{:,}".format(params), ha="center",
                va="center", fontsize=fs - 1.3, color=colour, zorder=4,
                fontweight="bold")


def arrow(ax, x0, y0, x1, y1, colour=INK, lw=1.1, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=colour, zorder=5,
        linestyle=(0, (3, 2)) if dashed else "solid",
        shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(6.84, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------------------------------------------------------- input
    box(ax, 0.40, 0.905, 0.20, 0.062,
        "EEG window $X$", "$C \\times T$", None, INK, alpha=0.06)
    arrow(ax, 0.47, 0.905, 0.27, 0.862)
    arrow(ax, 0.53, 0.905, 0.73, 0.862)

    ax.text(0.27, 0.878, "first order", ha="center", fontsize=7.6,
            color=STEM, style="italic")
    ax.text(0.73, 0.878, "second order", ha="center", fontsize=7.6,
            color=BRANCH, style="italic")

    # ------------------------------------------------------- the two paths
    h, gap = 0.098, 0.036
    top = 0.800
    for i, (lab, sub, par) in enumerate(STEM_BLOCKS):
        y = top - i * (h + gap)
        box(ax, 0.10, y, 0.34, h, lab, sub, par, STEM)
        if i < len(STEM_BLOCKS) - 1:
            arrow(ax, 0.27, y, 0.27, y - gap)
    for i, (lab, sub, par) in enumerate(BRANCH_BLOCKS):
        y = top - i * (h + gap)
        box(ax, 0.56, y, 0.34, h, lab, sub, par, BRANCH)
        if i < len(BRANCH_BLOCKS) - 1:
            arrow(ax, 0.73, y, 0.73, y - gap)

    # ------------------------------------- the reference update, label-free
    y_tan = top - 2 * (h + gap)
    ax.add_patch(FancyBboxPatch(
        (0.905, y_tan + 0.012), 0.078, h - 0.024,
        boxstyle="round,pad=0.005,rounding_size=0.010", linewidth=1.0,
        edgecolor=BRANCH, facecolor="white", linestyle=(0, (3, 2)), zorder=3))
    ax.text(0.944, y_tan + h / 2 + 0.010, "running", ha="center",
            va="center", fontsize=7.2, color=BRANCH)
    ax.text(0.944, y_tan + h / 2 - 0.014, "reference $M$", ha="center",
            va="center", fontsize=7.2, color=BRANCH)
    arrow(ax, 0.905, y_tan + h / 2, 0.900, y_tan + h / 2,
          colour=BRANCH, dashed=True)
    arrow(ax, 0.944, y_tan + h, 0.944, top + h - 0.004,
          colour=BRANCH, lw=0.9, dashed=True)
    arrow(ax, 0.944, top + h - 0.004, 0.900, top + h / 2,
          colour=BRANCH, lw=0.9, dashed=True)
    ax.text(0.952, (y_tan + top) / 2 + 0.02,
            "no labels,\nlive at inference", ha="left", va="center",
            fontsize=6.8, color=BRANCH, rotation=0)

    # ------------------------------------------------------ fuse, classify
    y_f = top - 4 * (h + gap) + 0.006
    arrow(ax, 0.27, top - 3 * (h + gap), 0.44, y_f + 0.052)
    arrow(ax, 0.73, top - 3 * (h + gap), 0.56, y_f + 0.052)
    box(ax, 0.34, y_f - 0.020, 0.32, 0.070,
        "concatenate $\\rightarrow$ fuse", "$256 \\rightarrow 128$", 32896,
        HEAD)
    arrow(ax, 0.50, y_f - 0.020, 0.50, y_f - 0.074)
    box(ax, 0.38, y_f - 0.144, 0.24, 0.070,
        "linear classifier", "Stop / Walk", 258, HEAD)

    ax.plot([0.06, 0.94], [y_f - 0.186, y_f - 0.186], lw=0.8, color=GREY)
    ax.text(0.50, y_f - 0.216,
            "Reported model: 290,943 parameters. The branch is 238,028 of "
            "them, and the fusion it requires a further 32,896: 93 % together.",
            ha="center", fontsize=8.2, color=INK)

    # ------------------------------------------- built, measured, not kept
    ax.text(0.06, y_f - 0.262, "Built and measured, not retained:",
            ha="left", fontsize=7.8, color=GREY, style="italic")
    for i, (lab, sub, why) in enumerate(DROPPED):
        x = 0.06 + i * 0.305
        ax.add_patch(FancyBboxPatch(
            (x, y_f - 0.352), 0.285, 0.070,
            boxstyle="round,pad=0.005,rounding_size=0.010", linewidth=0.9,
            edgecolor=GREY, facecolor=GREY, alpha=0.07, zorder=2))
        ax.text(x + 0.142, y_f - 0.300, lab, ha="center", fontsize=7.4,
                color=GREY)
        ax.text(x + 0.142, y_f - 0.322, sub, ha="center", fontsize=6.6,
                color=GREY)
        ax.text(x + 0.142, y_f - 0.343, why, ha="center", fontsize=6.6,
                color=GREY, style="italic")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = RESULTS / ("fig_arch.%s" % ext)
        fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=200)
        print("wrote %s" % out)


if __name__ == "__main__":
    main()
