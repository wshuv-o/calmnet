"""The paper's argument in one figure: five cohorts, one boundary.

Panel (a) is the effect of the branch on every cohort, paired across
participants, ordered by the class-block length of the protocol. Panel (b) is
the same five numbers against that block length, with the covariance update size
drawn as a vertical line: the branch helps on every cohort whose blocks are
shorter than one update and fails on the only one whose blocks are longer.

Both quantities on the x axis of (b) are read from the recording protocol before
any model is trained, so the separation in the figure uses no accuracy.

Writes results/fig_cohorts.pdf and .png.
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
except Exception:                       # blocked TrueType subsetter, see fig_arch
    matplotlib.rcParams["pdf.fonttype"] = 3
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
import figstyle as S                                            # noqa: E402
from figstyle import FULL, OURS, VERM, GREY                     # noqa: E402

S.use()
MU = 160          # estimator memory N/m at the settings used throughout
COHORTS = [
    ("D", "BCI IV-2a", 1, "d_bnci_3seed.json", "d_bnci_3seed.json"),
    ("C", "motor execution", 5, "c_stem_tangent_3seed.json",
     "c_stem_tangent_3seed.json"),
    ("E", "exoskeleton", 13, "e_decoded_3seed.json", "e_decoded_3seed.json"),
    ("A", "exoskeleton", 18, "a_tangent_3seed.json", "a_ablation_s12.json"),
    ("B", "treadmill", 214, "b_tangent_3seed.json", "b_tangent_3seed.json"),
]


def paired(bf, cf):
    b = json.load(io.open(os.path.join(RES, bf), encoding="utf-8"))
    c = json.load(io.open(os.path.join(RES, cf), encoding="utf-8"))
    subs = sorted(c["dn_stem|s0"]["per_subject"])

    def g(d, k):
        return np.array([d[k]["per_subject"][s]["acc"] for s in subs])

    X = np.mean([g(b, "dn_stem_tan|s%d" % i) for i in range(3)], axis=0)
    Y = np.mean([g(c, "dn_stem|s%d" % i) for i in range(3)], axis=0)
    d = X - Y
    # paired bootstrap over participants, which is the unit of replication
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean()
                   for _ in range(4000)])
    return (d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
            stats.wilcoxon(X, Y).pvalue, int((d > 0).sum()), len(d))


def main():
    rows = [(k, t, tau) + paired(bf, cf) for k, t, tau, bf, cf in COHORTS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.9),
                                   gridspec_kw=dict(width_ratios=[1.15, 1.0]))

    # ---------------------------------------------- (a) effect per cohort
    y = np.arange(len(rows))[::-1]
    for i, (k, t, tau, m, lo, hi, p, w, n) in enumerate(rows):
        col = OURS if m > 0 else VERM
        ax1.plot([lo, hi], [y[i], y[i]], color=col, lw=1.6,
                 solid_capstyle="round", zorder=2)
        ax1.plot([m], [y[i]], "o", color=col, ms=5.5, zorder=3,
                 markeredgecolor="white", markeredgewidth=0.8)
        ax1.text(hi + 0.012, y[i], "%d/%d" % (w, n), va="center",
                 fontsize=6.6, color=GREY)
    ax1.axvline(0, color=GREY, lw=0.8, zorder=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(["%s  %s\n$\\tau_{\\mathrm{blk}}=%d$" % (k, t, tau)
                         for k, t, tau, *_ in rows], fontsize=7)
    ax1.set_xlabel("change in balanced accuracy from the branch")
    ax1.set_xlim(-0.23, 0.13)
    S.panel(ax1, "a", x=-0.40)

    # ------------------------------------- (b) effect against block length
    tau = np.array([r[2] for r in rows], float)
    eff = np.array([r[3] for r in rows])
    ax2.axvspan(0.6, MU, color=OURS, alpha=0.05, zorder=0)
    ax2.axvline(MU, color=GREY, lw=1.0, ls=(0, (4, 2)), zorder=1)
    ax2.axhline(0, color=GREY, lw=0.8, zorder=1)
    for i, (k, t, tu, m, lo, hi, p, w, n) in enumerate(rows):
        col = OURS if m > 0 else VERM
        ax2.plot([tu, tu], [lo, hi], color=col, lw=1.4, zorder=2)
        ax2.plot([tu], [m], "o", color=col, ms=5.5, zorder=3,
                 markeredgecolor="white", markeredgewidth=0.8)
        ax2.annotate(k, (tu, m), textcoords="offset points",
                     xytext=(0, 9 if m > 0 else -13), ha="center",
                     fontsize=7.2, color=col)
    ax2.set_xscale("log")
    ax2.set_xlabel("class-block length $\\tau_{\\mathrm{blk}}$ (windows)")
    ax2.set_ylabel("change in balanced accuracy")
    ax2.set_xlim(0.6, 400)
    S.note(ax2, MU * 0.82, 0.085,
           "estimator memory\n$\\mu = N/m = 160$", ha="right", fontsize=6.6,
           color=GREY)
    S.panel(ax2, "b", x=-0.22)

    S.save(fig, "fig_cohorts")
    print("%-3s %-18s %-7s %-9s %-16s %s"
          % ("", "cohort", "tau_blk", "delta", "95% CI", "p"))
    for k, t, tu, m, lo, hi, p, w, n in rows:
        print("  %-1s %-16s %-7d %+.4f   [%+.3f, %+.3f]   %.4f  %d/%d"
              % (k, t, tu, m, lo, hi, p, w, n))


if __name__ == "__main__":
    main()
