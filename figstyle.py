"""Shared style for every figure: serif text, Okabe-Ito colours, exact column widths."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# iopjournal two-column measures, in inches
COL = 2.94
FULL = 6.02

# Okabe-Ito
OURS = "#000000"
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
INK = "#000000"
GREY = "#666666"
FAINT = "#BBBBBB"

# one series = one colour + marker + dash across the paper; ours solid and filled
SERIES = {
    "ours":     dict(color=OURS,   marker="o", ls="-",               mfc=OURS,   lw=2.0, ms=4.2),
    "ours_ctx": dict(color=PURPLE, marker="D", ls=(0, (3, 1, 1, 1)), mfc="none", lw=1.1, ms=3.6),
    "ours_alt": dict(color=BLUE,   marker="^", ls=(0, (4, 2)),       mfc="none", lw=1.1, ms=4.0),
    "noalign":  dict(color=VERM,   marker="s", ls=(0, (4, 2)),       mfc="none", lw=1.1, ms=3.8),
    "stem":     dict(color=GREEN,  marker="v", ls=(0, (1, 1)),       mfc="none", lw=1.1, ms=4.0),
    "baseline": dict(color=ORANGE, marker="P", ls=(0, (4, 2)),       mfc="none", lw=1.1, ms=4.2),
}


def use():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Latin Modern Roman", "Times New Roman",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "axes.grid": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,          # TrueType, not Type 3
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "figure.dpi": 200,
    })


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, "(%s)" % letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", color=INK)


def note(ax, x, y, text, **kw):
    """Small grey annotation."""
    kw.setdefault("fontsize", 6.5)
    kw.setdefault("color", GREY)
    ax.text(x, y, text, **kw)


def save(fig, name, png_dpi=400, pdf_dpi="figure"):
    """figures/<name>.pdf and .png. pdf_dpi only affects rasterised artists."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / (name + ".pdf"), dpi=pdf_dpi)
    fig.savefig(FIGURES / (name + ".png"), dpi=png_dpi)
    plt.close(fig)
    print("wrote figures/%s.pdf (+ .png)" % name)
