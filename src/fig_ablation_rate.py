"""Regenerate the two figures that still plotted the old architecture.

fig_ablation drew five arms built from the alignment layer, the gate and the
cross-epoch transformer, with the align-plus-gate arm marked as the proposal.
None of those is in the reported model. It is redrawn as the ablation of the
model we report: the stem, the stem plus the branch, and every component that
was tested and left out, so a reader can see what each one bought.

fig_rate drew the alignment layer's response to the adaptation rate. The rate
condition now governs the branch's reference, and it has been measured by
moving the estimator memory directly rather than by comparing cohorts. It is
redrawn as that manipulation: the branch effect against the memory on both
cohorts, with each cohort's class-block length marked. The effect appears and
disappears as the memory crosses the block length.

Writes results/fig_ablation.pdf and results/fig_rate.pdf (+ .png).
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
try:
    import fontTools.varLib  # noqa: F401
    matplotlib.rcParams["pdf.fonttype"] = 42
except Exception:                      # blocked TrueType subsetter, see fig_arch
    matplotlib.rcParams["pdf.fonttype"] = 3
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
import figstyle as S                                          # noqa: E402
from figstyle import FULL, COL, OURS, VERM, GREEN, GREY       # noqa: E402

S.use()


def L(f):
    p = os.path.join(RES, f)
    if not os.path.exists(p):
        sys.exit("missing: %s" % f)
    return json.load(io.open(p, encoding="utf-8"))


def cells(f, arm):
    d = L(f)
    ks = sorted(k for k in d if k.startswith(arm + "|s"))
    subs = sorted(d[ks[0]]["per_subject"])
    return np.array([[d[k]["per_subject"][s]["acc"] for s in subs]
                     for k in ks]), subs


# ------------------------------------------------------------------ ablation
ARMS = [
    ("stem + branch", "a_tangent_3seed.json", "dn_stem_tan", True),
    ("stem", "a_ablation_s12.json", "dn_stem", False),
    ("stem + align + gate + branch", "a_tangent_3seed.json", "dn_tan", False),
    ("stem + align + gate", "a_aligngate_3seed.json", "dn_noctx", False),
    ("stem + gate + ctx", "a_ablation_s12_2060.json", "dn_noalign", False),
    ("stem + align", "a_align_s12.json", "dn_align", False),
    ("stem + align + gate + ctx", "a_ablation_s12.json", "dn_full", False),
    ("stem + align + ctx", "a_ablation_s12.json", "dn_nogate", False),
    ("stem + gate", "a_gate_3seed.json", "dn_gate", False),
]


def fig_ablation():
    ctl, subs = cells("a_ablation_s12.json", "dn_stem")
    base = ctl.mean(axis=0)
    rows = []
    for lab, f, arm, star in ARMS:
        M, ss = cells(f, arm)
        d = M.mean(axis=0) - base
        p = (None if arm == "dn_stem"
             else stats.wilcoxon(M.mean(axis=0), base).pvalue)
        rows.append((lab, M.mean(), np.std(M.mean(axis=1), ddof=1),
                     d.mean(), p, int((d > 0).sum()), len(ss), star))
    rows.sort(key=lambda r: -r[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 3.1),
                                   gridspec_kw=dict(width_ratios=[1.0, 0.85]))
    y = np.arange(len(rows))[::-1]
    for i, (lab, acc, sd, d, p, w, n, star) in enumerate(rows):
        col = OURS if star else (GREY if lab == "stem" else VERM)
        ax1.barh(y[i], acc, height=0.62, color=col,
                 alpha=1.0 if star else 0.45, zorder=2)
        ax1.plot([acc - sd, acc + sd], [y[i], y[i]], color="white", lw=1.2,
                 zorder=3)
        ax1.text(acc + 0.004, y[i], "%.3f" % acc, va="center", fontsize=6.8,
                 color=col if star else GREY,
                 fontweight="bold" if star else "normal")
    ax1.set_yticks(y)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=6.9)
    ax1.set_xlim(0.80, 0.94)
    ax1.set_xlabel("balanced accuracy, cohort A, 3 seeds")
    for s_ in ("top", "right"):
        ax1.spines[s_].set_visible(False)
    S.panel(ax1, "a", x=-0.62)

    for i, (lab, acc, sd, d, p, w, n, star) in enumerate(rows):
        if lab == "stem":
            continue
        col = OURS if star else VERM
        ax2.plot([0, d], [y[i], y[i]], color=col, lw=2.2 if star else 1.4,
                 alpha=1.0 if star else 0.5, solid_capstyle="round", zorder=2)
        ax2.plot([d], [y[i]], "o", color=col, ms=5.5 if star else 4,
                 zorder=3, markeredgecolor="white", markeredgewidth=0.7)
        ax2.text(d + (0.004 if d > 0 else -0.004), y[i], "%d/%d" % (w, n),
                 va="center", ha="left" if d > 0 else "right", fontsize=6.4,
                 color=GREY)
    ax2.axvline(0, color=GREY, lw=0.9, zorder=1)
    ax2.axvspan(-0.02, 0.02, color=GREY, alpha=0.10, zorder=0)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(-0.055, 0.075)
    ax2.set_xlabel("change against the stem alone")
    for s_ in ("top", "right", "left"):
        ax2.spines[s_].set_visible(False)
    S.note(ax2, 0.019, y[-1] - 0.2, "interpretation\nthreshold", ha="right",
           fontsize=6.2, color=GREY)
    S.panel(ax2, "b", x=-0.08)
    S.save(fig, "fig_ablation")
    for r in rows:
        print("  %-30s %.4f  %s" % (r[0], r[1],
                                    "--" if r[4] is None
                                    else "%+.4f p=%.3f" % (r[3], r[4])))


# ---------------------------------------------------------------- rate/mu
MU = [
    ("A", 18, 8, "a_batch8_m1_3seed.json", "a_batch8_3seed.json"),
    ("A", 18, 40, "a_batch8_3seed.json", "a_batch8_3seed.json"),
    ("A", 18, 160, "a_tangent_3seed.json", "a_ablation_s12.json"),
    ("B", 214, 160, "b_tangent_3seed.json", "b_tangent_3seed.json"),
    ("B", 214, 1280, "b_batch256_3seed.json", "b_batch256_3seed.json"),
    ("B", 214, 3200, "b_tangent_m001_3seed.json", "b_tangent_3seed.json"),
]


def fig_rate():
    out = {}
    for coh, tau, mu, bf, cf in MU:
        B, subs = cells(bf, "dn_stem_tan")
        C, _ = cells(cf, "dn_stem")
        d = B.mean(axis=0) - C.mean(axis=0)
        out.setdefault(coh, []).append(
            (mu, tau, d.mean(), stats.wilcoxon(B.mean(axis=0),
                                               C.mean(axis=0)).pvalue,
             int((d > 0).sum()), len(subs)))

    fig, ax = plt.subplots(figsize=(COL * 1.55, 2.7))
    for coh, col, mk in (("A", OURS, "o"), ("B", VERM, "s")):
        pts = sorted(out[coh])
        ax.plot([p[0] for p in pts], [p[2] for p in pts], mk + "-", color=col,
                lw=1.5, ms=5.5, markeredgecolor="white", markeredgewidth=0.8,
                label="cohort %s  ($\\tau_{\\mathrm{blk}}=%d$)"
                      % (coh, pts[0][1]), zorder=3)
        tau = pts[0][1]
        ax.axvline(tau, color=col, lw=0.9, ls=(0, (3, 2)), alpha=0.7, zorder=1)
        ax.text(tau, 0.085, " $\\tau_{\\mathrm{blk}}$", color=col,
                fontsize=6.6, ha="left")
        for mu, _, d, p, w, n in pts:
            ax.annotate("%d/%d" % (w, n), (mu, d), textcoords="offset points",
                        xytext=(0, 8 if d > 0 else -12), ha="center",
                        fontsize=6.0, color=col)
    ax.axhline(0, color=GREY, lw=0.9, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("estimator memory $\\mu = N/m$ (windows)")
    ax.set_ylabel("change in balanced accuracy")
    ax.set_ylim(-0.22, 0.12)
    ax.legend(fontsize=6.6, loc="lower right", frameon=False)
    S.save(fig, "fig_rate")
    for coh in ("A", "B"):
        for mu, tau, d, p, w, n in sorted(out[coh]):
            print("  cohort %s  mu=%-5d tau=%-4d %+.4f  p=%.3f  %d/%d"
                  % (coh, mu, tau, d, p, w, n))


if __name__ == "__main__":
    print("ABLATION")
    fig_ablation()
    print("\nRATE")
    fig_rate()
