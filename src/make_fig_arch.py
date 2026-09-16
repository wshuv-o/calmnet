"""Figure 1: the architecture.

A block diagram of the proposed decoder, drawn to carry three things a reader
needs and a box-and-arrow sketch usually omits:

  * exact parameter counts per block, so the 24,181 total is auditable and the
    cost of the optional context pathway (594,816) is visible rather than stated;
  * the test-time update path of the alignment layer, which is what makes it an
    adaptation layer rather than a normalisation layer -- it is the only arrow
    that is live at inference on unlabelled data;
  * the three outputs, since the claim is that the model emits calibrated
    confidence and an abstention decision, not only a class.

Written at 200 dpi to results/fig_arch.png.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RESULTS = Path(__file__).resolve().parent.parent / "results"
INK = "#161D24"
ALIGN = "#2E6F5E"     # the novel component
STEM = "#A8763E"      # feature extraction
CTX = "#4A6FA5"       # optional context pathway
HEAD = "#6B4E71"      # outputs
GREY = "#9AA3AB"

plt.rcParams.update({
    "font.size": 9, "text.color": INK,
    "axes.edgecolor": INK, "figure.dpi": 200,
})

NL = chr(10)


def box(ax, x, y, w, h, label, sub, params, color, dashed=False, fs=9.5):
    """One architecture block: title band, body band, parameter band."""
    for fill, edge in ((color, "none"), ("none", color)):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.6, edgecolor=edge if edge != "none" else color,
            facecolor=fill if fill != "none" else "none",
            alpha=0.11 if fill != "none" else 1.0,
            linestyle="--" if dashed else "-",
            zorder=2 if fill != "none" else 3))
    ax.text(x + w / 2, y + h - 0.042, label, ha="center", va="top",
            fontsize=fs, fontweight="bold", color=color, zorder=4)
    ax.text(x + w / 2, y + h / 2 - 0.018, sub, ha="center", va="center",
            fontsize=7.5, color=INK, zorder=4, linespacing=1.5)
    if params:
        ax.text(x + w / 2, y + 0.030, params, ha="center", va="bottom",
                fontsize=7.7, color=color, style="italic", zorder=4)


def arrow(ax, x1, y1, x2, y2, color=INK, style="-|>", dashed=False, lw=1.5,
          rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, zorder=5,
        linestyle="--" if dashed else "-",
        connectionstyle="arc3,rad=%s" % rad))


def main():
    fig, ax = plt.subplots(figsize=(13.2, 5.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    Y, H = 0.30, 0.34
    MID = Y + H / 2

    # ---- input --------------------------------------------------------------
    ax.text(0.034, MID + 0.020, "EEG" + NL + "window", ha="center",
            va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(0.034, MID - 0.072, "60 ch x 4 s" + NL + "@ 100 Hz", ha="center",
            va="center", fontsize=7.6, color=GREY)
    arrow(ax, 0.070, MID, 0.096, MID)

    # ---- 1. adaptive alignment (the contribution) ---------------------------
    box(ax, 0.098, Y, 0.176, H, "Adaptive Alignment",
        "running spatial covariance M" + NL
        + "whiten by $M^{-1/2}$" + NL
        + "learned raw / aligned blend",
        "1 parameter", ALIGN)
    arrow(ax, 0.274, MID, 0.302, MID)

    # the update loop -- what makes this adaptation, not normalisation
    arrow(ax, 0.256, Y, 0.130, Y, color=ALIGN, dashed=True, rad=-0.70, lw=1.4)
    ax.text(0.186, Y - 0.150, "unsupervised update, active at inference",
            ha="center", fontsize=7.6, color=ALIGN, style="italic")
    ax.text(0.186, Y - 0.196, "(eval mode, no labels, no target data needed)",
            ha="center", fontsize=7.0, color=GREY)

    # ---- 2. multi-scale power stem ------------------------------------------
    box(ax, 0.304, Y, 0.176, H, "Multi-Scale Power",
        "64 / 128 / 256 ms branches" + NL
        + "depthwise spatial filters" + NL
        + r"square $\rightarrow$ pool $\rightarrow$ log",
        "6,768 parameters", STEM)
    ax.text(0.392, Y + H + 0.036,
            "normalisation precedes squaring," + NL
            + "so log-power ratios survive",
            ha="center", fontsize=7.2, color=STEM, style="italic",
            linespacing=1.4)
    arrow(ax, 0.480, MID, 0.506, MID)

    # ---- 3. frame embedding -------------------------------------------------
    box(ax, 0.508, Y, 0.126, H, "Frame Embed",
        "norm + projection" + NL + "+ within-window" + NL + "attention pooling",
        "12,737 parameters", STEM)
    arrow(ax, 0.634, MID, 0.646, MID)

    # ---- 4. optional context pathway ----------------------------------------
    box(ax, 0.648, 0.745, 0.208, 0.180, "Cross-Epoch Context  (optional)",
        "causal transformer over" + NL + "K = 8 epochs (14.5 s)",
        "594,816 parameters", CTX, dashed=True, fs=8.6)
    ax.text(0.752, 0.958,
            "safety mode: $-$0.023 accuracy, $-$23 % false activations",
            ha="center", fontsize=7.4, color=CTX, style="italic")
    arrow(ax, 0.662, Y + H, 0.692, 0.745, color=CTX, dashed=True, rad=0.28)
    arrow(ax, 0.818, 0.745, 0.848, Y + H, color=CTX, dashed=True, rad=0.28)

    # ---- 5. selective head --------------------------------------------------
    box(ax, 0.648, Y, 0.208, H, "Selective Head",
        "classifier + abstention gate" + NL + "under a coverage constraint",
        "4,419 parameters", HEAD, fs=9.0)
    arrow(ax, 0.856, MID, 0.880, MID)

    # ---- outputs ------------------------------------------------------------
    outs = [("Walk / Stop", "balanced acc. 0.901", MID + 0.118),
            ("calibrated confidence", "ECE 0.039", MID),
            ("abstain / act", "0.876 @ 90 % coverage", MID - 0.118)]
    for name, val, yy in outs:
        ax.add_patch(FancyBboxPatch(
            (0.886, yy - 0.036), 0.108, 0.072,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=1.3, edgecolor=HEAD, facecolor=HEAD, alpha=0.10,
            zorder=2))
        ax.text(0.940, yy + 0.012, name, ha="center", fontsize=7.8,
                fontweight="bold", color=HEAD, zorder=4)
        ax.text(0.940, yy - 0.020, val, ha="center", fontsize=7.0, color=INK,
                zorder=4)
    ax.plot([0.880, 0.880], [MID - 0.118, MID + 0.118], lw=1.2, color=HEAD,
            zorder=4)
    for _, _, yy in outs:
        arrow(ax, 0.880, yy, 0.886, yy, color=HEAD, lw=1.2)

    # ---- parameter budget ---------------------------------------------------
    ax.plot([0.10, 0.90], [0.078, 0.078], lw=0.8, color=GREY)
    ax.text(0.50, 0.034,
            "Default configuration: 24,181 parameters   "
            "(53 % of ATCNet, 5 % of EEG Conformer).   "
            "Enabling the optional context pathway raises this to 618,997.",
            ha="center", fontsize=8.4, color=INK)

    fig.tight_layout()
    out = RESULTS / "fig_arch.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
