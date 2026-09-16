"""Figures 4-7. Reads results/ only -- computes nothing.

  fig_efficiency.png  accuracy against parameter count, harness C only. This is
                      the claim the paper actually makes, so it gets a figure:
                      parity with the field at half the parameters.
  fig_inversion.png   the same five arms on both cohorts. The ordering does not
                      compress between cohorts, it inverts, and a slope chart is
                      the only presentation that makes that unmissable.
  fig_metrics.png     the five counterexamples: distribution-level metric moved
                      one way, the task moved the other. The paper's thesis.
  fig_operating.png   calibration and the abstention operating point per arm,
                      since "more than accuracy" is a claim and needs evidence.

Written at 200 dpi to results/.
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
OURS = "#2E6F5E"
BASE = "#A8763E"
CTX = "#4A6FA5"
HEAD = "#6B4E71"
BAD = "#B4453C"
GREY = "#9AA3AB"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})

NL = chr(10)


def L(name):
    p = RESULTS / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def acc(d, arm, field="acc"):
    v = d.get(arm + "|s0")
    return v[field] if v and field in v else None


# ---------------------------------------------------------------- figure 4
def fig_efficiency():
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    pts = [
        ("ATCNet",              45280,  0.913, BASE,  True),
        ("ATCNet, rate-matched", 45280, 0.908, BASE,  False),
        ("Ours, align + gate",  24181,  0.901, OURS,  True),
        ("Ours, + context",    618997,  0.878, CTX,   True),
        ("Ours, stem only",      6768,  0.827, GREY,  True),
    ]
    for name, p, a, c, lab in pts:
        mine = name.startswith("Ours, align")
        ax.scatter([p], [a], s=190 if mine else 90, color=c,
                   zorder=4, edgecolor="white", linewidth=1.4)
        if lab:
            dx, dy, ha = (1.28, 0.0, "left") if not mine else (0.72, -0.011, "right")
            ax.annotate(name, (p * dx, a + dy), fontsize=8, color=c, ha=ha,
                        va="center", fontweight="bold" if mine else "normal")
    # the efficiency frontier the claim rests on
    ax.annotate("", xy=(24181, 0.901), xytext=(45280, 0.913),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1, ls=":"))
    ax.text(33000, 0.9235, "53 % of the parameters," + NL + "0.012 less accuracy",
            ha="center", fontsize=7.8, color=INK, style="italic")
    ax.set_xscale("log")
    ax.set_xlim(4000, 1.5e6)
    ax.set_ylim(0.805, 0.945)
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("balanced accuracy, cohort A")
    ax.set_title("Accuracy against model size" + NL
                 + "harness C only, seed 0 -- no cross-harness comparison",
                 fontsize=9.5, loc="left")
    ax.grid(alpha=0.13, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_efficiency.png", bbox_inches="tight",
                facecolor="white")
    print("wrote fig_efficiency.png")


# ---------------------------------------------------------------- figure 5
def fig_inversion():
    a, b = L("driftnet_ds.json"), L("driftnet_mobi.json")
    arms = [("dn_noctx", "align + gate", OURS),
            ("dn_full", "align + ctx + gate", CTX),
            ("dn_nogate", "align + ctx", HEAD),
            ("dn_noalign", "ctx + gate", BASE),
            ("dn_stem", "stem only", GREY)]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for k, lab, c in arms:
        ya, yb = acc(a, k), acc(b, k)
        if ya is None or yb is None:
            continue
        uses = k in ("dn_noctx", "dn_full", "dn_nogate")
        ax.plot([0, 1], [ya, yb], "-o", color=c, lw=2.4 if uses else 1.6,
                ms=7, alpha=1.0 if uses else 0.75,
                ls="-" if uses else "--", zorder=3)
        ax.annotate(f"{ya:.3f}", (-0.035, ya), ha="right", va="center",
                    fontsize=8, color=c)
        ax.annotate(f"  {lab}  {yb:.3f}", (1.035, yb), ha="left", va="center",
                    fontsize=8, color=c)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cohort A" + NL + "exoskeleton, blocks 18 win",
                        "cohort B" + NL + "treadmill, blocks 309 win"],
                       fontsize=8.5)
    ax.set_xlim(-0.30, 1.72)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("The ordering does not compress between cohorts -- it inverts"
                 + NL + "solid = uses alignment; the best arm on A is the worst on B",
                 fontsize=9.5, loc="left")
    ax.grid(axis="y", alpha=0.13)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_inversion.png", bbox_inches="tight",
                facecolor="white")
    print("wrote fig_inversion.png")


# ---------------------------------------------------------------- figure 6
def fig_metrics():
    """Each row: a distribution-level metric improved; the task did not follow."""
    rows = [
        ("Rate tuned to maximise\ndrift removal",
         "drift removed\n25 % -> 60 %", -0.092),
        ("Cohort where MORE drift\nwas removed (84 % vs 52 %)",
         "more drift removed", -0.166),
        ("Verified fix to a real\ncovariance-estimation bias",
         "estimator corrected", -0.016),
        ("Prediction registered from\nour own diagnosis",
         "diagnosis applied", -0.060),
        ("Nuisance-leakage probe\non a decoder that cannot\nsee the nuisance",
         "R2 = +0.863", 0.0),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y = np.arange(len(rows))[::-1]
    for yy, (label, metric, delta) in zip(y, rows):
        if delta == 0.0:
            # No bar: this row has no accuracy delta, and drawing one on an
            # accuracy axis would imply a cost that does not exist.
            ax.text(-0.004, yy, "no accuracy change -- the probe rises"
                    + NL + "with accuracy regardless of the decoder",
                    va="center", ha="right", fontsize=7.6, color=GREY,
                    style="italic")
        else:
            ax.barh(yy, delta, color=BAD, alpha=0.75, height=0.52, zorder=3)
            ax.text(delta - 0.006, yy, f"{delta:+.3f}", va="center", ha="right",
                    fontsize=8.6, color=BAD, fontweight="bold")
        ax.text(0.008, yy, metric, va="center", ha="left", fontsize=7.8,
                color=OURS, style="italic")
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(-0.20, 0.115)
    ax.set_xlabel("change in balanced accuracy")
    ax.set_title("Five times the metric improved and the task did not follow"
                 + NL
                 + "green = the distribution-level quantity that improved;"
                   " red = what it cost",
                 fontsize=9.5, loc="left")
    ax.grid(axis="x", alpha=0.13)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_metrics.png", bbox_inches="tight",
                facecolor="white")
    print("wrote fig_metrics.png")


# ---------------------------------------------------------------- figure 7
def fig_operating():
    d = L("driftnet_ds.json")
    arms = [("dn_noctx", "align + gate", OURS),
            ("dn_full", "align + ctx + gate", CTX),
            ("dn_noalign", "ctx + gate", BASE),
            ("dn_stem", "stem only", GREY)]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))

    # (a) calibration
    ax = axes[0]
    names = [l for _, l, _ in arms]
    vals = [acc(d, k, "ece") for k, _, _ in arms]
    cols = [c for _, _, c in arms]
    ax.barh(range(len(names))[::-1], vals, color=cols, alpha=0.82, height=0.6)
    for i, v in zip(range(len(names))[::-1], vals):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8.4,
                color=INK)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("expected calibration error (lower better)")
    ax.set_title("(a) confidence means something", fontsize=9.5, loc="left")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.grid(axis="x", alpha=0.13)

    # (b) accuracy gained by declining 10 % of windows
    ax = axes[1]
    for k, lab, c in arms:
        a0, a90 = acc(d, k), acc(d, k, "acc_at_90")
        if a90 is None:
            continue
        ax.plot([1.0, 0.9], [a0, a90], "-o", color=c, lw=2.2, ms=6)
        ax.annotate(f"  {lab}", (0.9, a90), fontsize=7.8, color=c,
                    ha="right", va="bottom", xytext=(-6, 4),
                    textcoords="offset points")
    ax.set_xlim(1.035, 0.862)
    ax.set_xticks([1.0, 0.9])
    ax.set_xticklabels(["100 % coverage", "90 % coverage"], fontsize=8.5)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("(b) declining 10 % of windows", fontsize=9.5, loc="left")
    ax.grid(axis="y", alpha=0.13)

    # (c) what the wearer experiences
    ax = axes[2]
    for k, lab, c in arms:
        fa, a0 = acc(d, k, "false_onsets_per_min"), acc(d, k)
        if fa is None:
            continue
        ax.scatter([fa], [a0], s=120, color=c, zorder=4, edgecolor="white",
                   linewidth=1.3)
        ax.annotate(lab, (fa, a0), fontsize=7.8, color=c,
                    xytext=(7, -3), textcoords="offset points")
    ax.set_xlabel("spurious activations per minute of standing")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("(c) the trade the wearer feels", fontsize=9.5, loc="left")
    ax.set_xlim(0.45, 2.35)
    ax.grid(alpha=0.13)

    fig.tight_layout()
    fig.savefig(RESULTS / "fig_operating.png", bbox_inches="tight",
                facecolor="white")
    print("wrote fig_operating.png")


if __name__ == "__main__":
    fig_efficiency()
    fig_inversion()
    fig_metrics()
    fig_operating()
