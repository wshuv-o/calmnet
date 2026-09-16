"""Shared figure style for every plot in the paper (FIGURE_RULES.md §11).

Imported by src/figures.py; never edit styling in an individual figure.

Key decisions and why:

  serif + STIX      The document is Elsevier CAS, which sets Times. A sans-serif
                    axis label under a serif paragraph is the single loudest
                    signal that a figure was made elsewhere and pasted in
                    (FIGURE_RULES §1).
  Okabe-Ito         Colour-blind-safe. Every series additionally differs in
                    marker shape and dash pattern, so the encoding survives
                    greyscale printing (§4).
  pdf.fonttype 42   Embeds TrueType rather than emitting Type 3, which some
                    publishers reject outright (§12).
  exact widths      Figures are generated at the real CAS column measure so
                    point sizes in the figure equal point sizes in the body
                    text. Never draw large and scale down in LaTeX (§12).
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Measured from the compiled document: \the\columnwidth and \the\textwidth
# under \documentclass[a4paper,fleqn]{cas-dc}.
COL = 3.30       # inches, single column
FULL = 6.84      # inches, spans both columns

# Okabe-Ito, colour-blind safe.
OURS = "#000000"      # our method: black, solid, heavy, filled (tifs_style)
BLUE = "#0072B2"      # secondary blue
VERM = "#D55E00"      # the alignment-free comparison
GREEN = "#009E73"     # the bare stem
PURPLE = "#CC79A7"    # the context pathway
ORANGE = "#E69F00"    # published baselines
INK = "#000000"
GREY = "#666666"
FAINT = "#BBBBBB"

# One series = one colour + one marker + one dash, for the whole document (§11).
# Our method is solid and filled where others are dashed and hollow (§4).
SERIES = {
    "ours":     dict(color=OURS,   marker="o", ls="-",              mfc=OURS,   lw=2.0, ms=4.2),
    "ours_ctx": dict(color=PURPLE, marker="D", ls=(0, (3, 1, 1, 1)), mfc="none", lw=1.1, ms=3.6),
    "ours_alt": dict(color=BLUE,   marker="^", ls=(0, (4, 2)),       mfc="none", lw=1.1, ms=4.0),
    "noalign":  dict(color=VERM,   marker="s", ls=(0, (4, 2)),       mfc="none", lw=1.1, ms=3.8),
    "stem":     dict(color=GREEN,  marker="v", ls=(0, (1, 1)),       mfc="none", lw=1.1, ms=4.0),
    "baseline": dict(color=ORANGE, marker="P", ls=(0, (4, 2)),       mfc="none", lw=1.1, ms=4.2),
}


def use():
    plt.rcParams.update({
        # -- typography (§1): match the document's serif body face
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
        # -- frame and ink (§3)
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
        # -- export (§12)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "figure.dpi": 200,
    })


def panel(ax, letter, x=-0.16, y=1.04):
    """Panel identifier in the corner -- never a sentence title (§2)."""
    ax.text(x, y, "(%s)" % letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", color=INK)


def note(ax, x, y, text, **kw):
    """Commentary: small and grey so it reads as annotation, not data (§9)."""
    kw.setdefault("fontsize", 6.5)
    kw.setdefault("color", GREY)
    ax.text(x, y, text, **kw)


def save(fig, name):
    """Vector PDF for LaTeX; PNG alongside only for the HTML preview (§12)."""
    out = RESULTS / (name + ".pdf")
    fig.savefig(out)
    fig.savefig(RESULTS / (name + ".png"), dpi=400)
    plt.close(fig)
    print("wrote %s (+ .png)" % out.name)
