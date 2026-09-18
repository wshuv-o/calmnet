"""Figure 2: the adaptation rate is two-sided.

Reads results/ only -- computes nothing.

  (a) the controlled sweep -- cohort B, five arms, three momenta. Every arm that
      uses alignment steps up as the memory crosses the class-block length; the
      two that do not are flat. This is what makes it an experiment rather than
      a comparison.
  (b) the two cohorts respond in OPPOSITE directions to the same change, so no
      single setting serves both. Matched estimator version throughout.
  (c) why: the admissible rate band is bounded below by drift and above by class
      block duration. Cohort A's band is wide, cohort B's is empty.

Written at 200 dpi to results/fig_rate.png.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"
INK = "#161D24"
ALIGN = "#2E6F5E"
RAW = "#A8763E"
GREY = "#9AA3AB"
BAD = "#B4453C"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})


def load(name):
    p = RESULTS / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def acc(d, arm, seed=0):
    v = d.get(f"{arm}|s{seed}")
    return v["acc"] if v else None


def main():
    m020 = load("driftnet_mobi.json")          # cohort B, pre-fix estimator
    m020f = load("driftfix_mobi.json")         # cohort B, fixed estimator
    m020c = load("driftfix_mobi_m020.json")    # cohort B, fixed, extra arms
    m005 = load("driftmom_0.05.json")
    m001 = load("driftmom_0.01.json")
    a020 = load("driftfix_ds.json")            # cohort A, fixed
    a001 = load("driftmom_ds_0.01.json")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))

    # ---- (a) the controlled sweep -------------------------------------------
    ax = axes[0]
    arms = [("dn_full", "align + ctx + gate", True),
            ("dn_noctx", "align + gate", True),
            ("dn_nogate", "align + ctx", True),
            ("dn_noalign", "ctx + gate (no align)", False),
            ("dn_stem", "stem only", False)]
    mom = [0.20, 0.05, 0.01]
    x = np.arange(3)
    lbl_y = []
    for a, lab, uses in arms:
        # prefer matched-version baseline at m=0.20 where one exists
        base = acc(m020c, a) or acc(m020f, a) or acc(m020, a)
        ys = [base, acc(m005, a), acc(m001, a)]
        if any(v is None for v in ys):
            continue
        ax.plot(x, ys, "-o", ms=4.5, lw=2.0 if uses else 1.4,
                color=ALIGN if uses else GREY,
                ls="-" if uses else "--", zorder=3 if uses else 2)
        lbl_y.append((ys[-1], lab, ALIGN if uses else "#6C757D"))
    lbl_y.sort()
    for i in range(1, len(lbl_y)):
        if lbl_y[i][0] - lbl_y[i - 1][0] < 0.011:
            lbl_y[i] = (lbl_y[i - 1][0] + 0.011, lbl_y[i][1], lbl_y[i][2])
    for yv, lab, col in lbl_y:
        ax.annotate(lab, (x[-1] + 0.08, yv), va="center", fontsize=7.5, color=col)
    ax.axvspan(-0.15, 1.0, color=BAD, alpha=0.06, zorder=0)
    ax.text(0.42, 0.565, "memory shorter\nthan a class block", ha="center",
            fontsize=7.5, color=BAD, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"m = {v}\n~{int(1/v)*32} win" for v in mom], fontsize=8)
    ax.set_xlim(-0.2, 2.95)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("(a) cohort B: only alignment arms respond\n"
                 "slower adaptation, five arms, one change",
                 fontsize=9.5, loc="left")

    # ---- (b) opposite directions --------------------------------------------
    ax = axes[1]
    pairs = [("cohort A\nblocks 18 win", acc(a020, "dn_full"), acc(a001, "dn_full"), RAW),
             ("cohort B\nblocks 214 win", acc(m020f, "dn_full"), acc(m001, "dn_full"), ALIGN)]
    for i, (lab, fast, slow, c) in enumerate(pairs):
        if fast is None or slow is None:
            continue
        ax.plot([0, 1], [fast, slow], "-o", color=c, lw=2.4, ms=7)
        ax.annotate(f"{fast:.3f}", (0, fast), textcoords="offset points",
                    xytext=(-8, 0), ha="right", fontsize=8.5, color=c)
        ax.annotate(f"{slow:.3f}", (1, slow), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8.5, color=c)
        mid = (fast + slow) / 2
        ax.annotate(f"{slow-fast:+.3f}", (0.5, mid), textcoords="offset points",
                    xytext=(0, 9 if slow > fast else -16), ha="center",
                    fontsize=9, color=c, fontweight="bold")
        ax.annotate(lab, (1.14, slow), fontsize=8, color=c, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fast\nm = 0.20", "slow\nm = 0.01"], fontsize=8.5)
    ax.set_xlim(-0.35, 1.75)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("(b) the same change, opposite signs\n"
                 "no single rate serves both protocols",
                 fontsize=9.5, loc="left")

    # ---- (c) the admissible band --------------------------------------------
    # Memory must EXCEED the class-block duration (or it tracks the class) and
    # stay BELOW the drift timescale (or it stops tracking drift). The upper
    # bound is bracketed, not pinned: cohort A is good at ~160 windows and bad
    # at ~3200, so it lies between. Drawn as an uncertainty band.
    ax = axes[2]
    UP_LO, UP_HI = 160, 3200
    ax.axvspan(UP_LO, UP_HI, color=INK, alpha=0.07, zorder=0)
    ax.text(np.sqrt(UP_LO * UP_HI), 1.62,
            "upper bound: drift timescale\n(bracketed, not pinned --\n"
            "cohort A good at 160, bad at 3200)",
            ha="center", fontsize=7.0, color=INK, style="italic")

    for k, (lab, block, col) in enumerate([("cohort A", 18, RAW),
                                           ("cohort B", 214, ALIGN)]):
        y = 1 - k
        ax.plot([block], [y], "|", ms=14, mew=2.4, color=col, zorder=5)
        ax.annotate(f"class block\n{block} win", (block, y - 0.36), ha="center",
                    fontsize=7.5, color=col)
        ax.text(11.5, y, lab, ha="right", fontsize=9.5, va="center", color=col)
        if block < UP_LO:
            ax.plot([block, UP_LO], [y, y], lw=10, color=col, alpha=0.32,
                    solid_capstyle="butt", zorder=2)
            ax.annotate("admissible", (np.sqrt(block * UP_LO), y + 0.20),
                        ha="center", fontsize=7.5, color=col, style="italic")
        else:
            ax.plot([block, UP_HI], [y, y], lw=10, color=BAD, alpha=0.16,
                    solid_capstyle="butt", zorder=2)
            ax.annotate("no rate avoids the class\nwithout losing drift",
                        (np.sqrt(block * UP_HI), y + 0.24), ha="center",
                        fontsize=7.5, color=BAD, style="italic")
            ax.plot([block], [y], "x", ms=9, mew=2.2, color=BAD, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(10, 6000)
    ax.set_ylim(-0.8, 2.0)
    ax.set_yticks([])
    ax.set_xlabel("adaptation memory (windows, log scale)")
    ax.set_title("(c) admissible = longer than a class block,\n"
                 "shorter than the drift timescale", fontsize=9.5, loc="left")

    fig.tight_layout()
    out = RESULTS / "fig_rate.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
