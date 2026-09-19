"""The strongest result in the paper, drawn per participant.

Twenty of twenty participants improve on the held-out motor-execution cohort at
p below 1e-4, and until now that appeared only as a row in a table. A paired
plot shows it directly: one line per participant, stem on the left, stem plus
branch on the right, every line sloping the same way.

The second panel gives the per-participant differences sorted, so the size of
the effect and its consistency are both visible rather than summarised.

Writes results/fig_cohortC.pdf and .png.
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
except Exception:
    matplotlib.rcParams["pdf.fonttype"] = 3
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
import figstyle as S                                   # noqa: E402
from figstyle import FULL, OURS, GREY                  # noqa: E402

S.use()
SRC = "c_stem_tangent_3seed.json"


def main():
    d = json.load(io.open(os.path.join(RES, SRC), encoding="utf-8"))
    subs = sorted(d["dn_stem|s0"]["per_subject"])

    def g(arm):
        return np.mean([[d["%s|s%d" % (arm, i)]["per_subject"][s]["acc"]
                         for s in subs] for i in range(3)], axis=0)

    stem, branch = g("dn_stem"), g("dn_stem_tan")
    diff = branch - stem
    p = stats.wilcoxon(branch, stem).pvalue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.9),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.25]))

    # ------------------------------------------------- (a) paired lines
    for i in range(len(subs)):
        ax1.plot([0, 1], [stem[i], branch[i]], color=OURS, lw=0.9, alpha=0.55,
                 zorder=2, solid_capstyle="round")
        ax1.plot([0, 1], [stem[i], branch[i]], "o", color=OURS, ms=3.0,
                 alpha=0.75, zorder=3)
    ax1.plot([0, 1], [stem.mean(), branch.mean()], color=OURS, lw=2.6,
             zorder=4)
    ax1.plot([0, 1], [stem.mean(), branch.mean()], "o", color=OURS, ms=6.5,
             markeredgecolor="white", markeredgewidth=1.1, zorder=5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["stem", "stem $+$ branch"], fontsize=7.4)
    ax1.set_xlim(-0.28, 1.28)
    ax1.set_ylabel("balanced accuracy")
    S.note(ax1, 0.5, stem.mean() - 0.055,
           "%.3f $\\rightarrow$ %.3f" % (stem.mean(), branch.mean()),
           ha="center", fontsize=7.2, color=OURS)
    for s_ in ("top", "right"):
        ax1.spines[s_].set_visible(False)
    S.panel(ax1, "a", x=-0.26)

    # -------------------------------------------- (b) sorted differences
    order = np.argsort(diff)
    y = np.arange(len(subs))
    ax2.barh(y, diff[order], height=0.68, color=OURS, alpha=0.82, zorder=2)
    ax2.axvline(0, color=GREY, lw=0.9, zorder=1)
    ax2.axvline(diff.mean(), color=OURS, lw=1.0, ls=(0, (4, 2)), zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([subs[i].replace("S0", "S") for i in order],
                        fontsize=5.6)
    ax2.set_xlabel("change in balanced accuracy from the branch")
    ax2.set_ylim(-0.8, len(subs) - 0.2)
    for s_ in ("top", "right", "left"):
        ax2.spines[s_].set_visible(False)
    S.note(ax2, diff.mean() + 0.004, len(subs) - 3.4,
           "mean $%+.3f$" % diff.mean(), ha="left", fontsize=6.6, color=OURS)
    S.note(ax2, diff.max() * 0.98, 0.6,
           "%d of %d improve,  $p = %.1f\\times10^{-5}$"
           % (int((diff > 0).sum()), len(subs), p * 1e5),
           ha="right", fontsize=6.8, color=GREY)
    S.panel(ax2, "b", x=-0.14)

    S.save(fig, "fig_cohortC")
    print("  stem %.4f -> branch %.4f   delta %+.4f"
          % (stem.mean(), branch.mean(), diff.mean()))
    print("  improved %d/%d   p=%.6f   range %+.3f to %+.3f"
          % (int((diff > 0).sum()), len(subs), p, diff.min(), diff.max()))


if __name__ == "__main__":
    main()
