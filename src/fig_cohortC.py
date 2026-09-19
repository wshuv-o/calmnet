"""The held-out cohort, and the three things that make its number credible.

Cohort C was held out from every design decision, so it carries most of the
paper's weight. An accuracy on its own does not support that weight: a reader
needs to know whether the gain is consistent, whether it is larger than the
noise the measurement already has, and what it is large relative to.

  (a) every participant improves, in absolute balanced accuracy
  (b) each participant's change against its own spread across seeds, so the
      effect is read next to the noise rather than separately from it
  (c) where the result sits among eight published decoders trained in the same
      pipeline on the same cohort, which is what makes 0.843 mean anything

The earlier version of this figure drew the per-participant change twice, once
as paired lines and once as sorted bars, and gave no context for the accuracy.

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
from figstyle import FULL, OURS, GREY, FAINT, ORANGE   # noqa: E402

S.use()
SRC = "c_stem_tangent_3seed.json"
PUB = "bd_cohort_c.json"
N_SEED = 3


def load():
    d = json.load(io.open(os.path.join(RES, SRC), encoding="utf-8"))
    subs = sorted(d["dn_stem|s0"]["per_subject"])

    def arm(name, field="acc"):
        return np.array([[d["%s|s%d" % (name, i)]["per_subject"][s][field]
                          for s in subs] for i in range(N_SEED)])

    b = json.load(io.open(os.path.join(RES, PUB), encoding="utf-8"))
    pub = {}
    for m in sorted({k.split("|")[0] for k in b}):
        pub[m] = np.array([[b["%s|s%d" % (m, i)]["per_subject"][s]["acc"]
                            for s in subs] for i in range(N_SEED)])
    return subs, arm("dn_stem"), arm("dn_stem_tan"), arm, pub


def main():
    subs, stem, branch, arm, pub = load()
    s_mu, b_mu = stem.mean(0), branch.mean(0)
    dif = branch - stem                      # seeds x participants
    d_mu, d_sd = dif.mean(0), dif.std(0, ddof=1)
    p_pair = stats.wilcoxon(b_mu, s_mu).pvalue
    beats_noise = int((d_mu > d_sd).sum())

    fig = plt.figure(figsize=(FULL, 3.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.80, 1.18, 1.22], wspace=0.55)

    # ------------------------------------------------ (a) every one improves
    ax = fig.add_subplot(gs[0, 0])
    for i in range(len(subs)):
        ax.plot([0, 1], [s_mu[i], b_mu[i]], color=OURS, lw=0.8, alpha=0.40,
                zorder=2, solid_capstyle="round")
    ax.plot([0, 1], [s_mu.mean(), b_mu.mean()], color=OURS, lw=2.6, zorder=4)
    ax.plot([0, 1], [s_mu.mean(), b_mu.mean()], "o", color=OURS, ms=6.0,
            markeredgecolor="white", markeredgewidth=1.1, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["stem", "stem\n$+$ branch"], fontsize=7.0)
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylabel("balanced accuracy")
    S.note(ax, 0.5, s_mu.mean() - 0.052,
           "%.3f $\\rightarrow$ %.3f" % (s_mu.mean(), b_mu.mean()),
           ha="center", fontsize=7.0, color=OURS)
    S.note(ax, 0.5, b_mu.max() + 0.012,
           "%d of %d improve" % (int((d_mu > 0).sum()), len(subs)),
           ha="center", fontsize=6.6, color=GREY)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    S.panel(ax, "a", x=-0.34)

    # ------------------------------- (b) the change beside its own seed noise
    ax = fig.add_subplot(gs[0, 1])
    o = np.argsort(d_mu)
    y = np.arange(len(subs))
    ax.barh(y, d_mu[o], height=0.70, color=OURS, alpha=0.80, zorder=3)
    ax.errorbar(d_mu[o], y, xerr=d_sd[o], fmt="none", ecolor=FAINT,
                elinewidth=0.9, capsize=1.6, zorder=4)
    ax.axvline(0, color=GREY, lw=0.9, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([subs[i].replace("S0", "S") for i in o], fontsize=5.2)
    ax.set_xlabel("change in balanced accuracy")
    ax.set_ylim(-0.9, len(subs) - 0.1)
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    # the sort puts the small changes at the bottom, so the lower right of
    # the axes is the only region no bar or whisker reaches
    xr = ax.get_xlim()[1]
    S.note(ax, xr * 0.98, 5.4,
           "larger than its own seed\nspread in %d of %d participants"
           % (beats_noise, len(subs)), ha="right", fontsize=6.0, color=OURS)
    S.note(ax, xr * 0.98, 2.0,
           "bars: mean over %d seeds\nwhiskers: SD over seeds" % N_SEED,
           ha="right", fontsize=5.6, color=GREY)
    S.panel(ax, "b", x=-0.20)

    # ------------------------------------------ (c) the field on this cohort
    ax = fig.add_subplot(gs[0, 2])
    rows = [(m, M.mean(), M.mean(0)) for m, M in pub.items()]
    rows.append(("stem", s_mu.mean(), s_mu))
    rows.append(("stem $+$ branch", b_mu.mean(), b_mu))
    rows.sort(key=lambda r: r[1])
    best = max((r for r in rows if r[0] not in ("stem", "stem $+$ branch")),
               key=lambda r: r[1])
    w = stats.wilcoxon(b_mu, best[2])

    ypos = np.arange(len(rows))
    for i, (m, mu, per) in enumerate(rows):
        ours = m == "stem $+$ branch"
        col = OURS if ours else (GREY if m == "stem" else ORANGE)
        sd = per.std(ddof=1) / np.sqrt(1)      # across participants
        ax.plot([mu - per.std(ddof=1) / np.sqrt(len(subs)),
                 mu + per.std(ddof=1) / np.sqrt(len(subs))], [i, i],
                color=col, lw=1.2, alpha=0.85, zorder=3)
        ax.plot(mu, i, "o", color=col, ms=5.2 if ours else 3.4,
                markeredgecolor="white" if ours else "none",
                markeredgewidth=0.9 if ours else 0, zorder=4)
    ax.axvline(0.5, color=FAINT, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.8)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("balanced accuracy on cohort C")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    S.note(ax, 0.507, -0.4, "chance", fontsize=5.6, color=GREY, ha="left")
    S.note(ax, ax.get_xlim()[1], 0.7,
           "$+%.3f$ over the best published,\nhigher in %d of %d, "
           "$p=%.0f{\\times}10^{-4}$"
           % (rows[-1][1] - best[1], int((b_mu > best[2]).sum()), len(subs),
              w.pvalue * 1e4),
           ha="right", fontsize=6.0, color=OURS)
    S.panel(ax, "c", x=-0.34)

    S.save(fig, "fig_cohortC")
    print("  stem %.4f -> branch %.4f   delta %+.4f  p=%.2e"
          % (s_mu.mean(), b_mu.mean(), d_mu.mean(), p_pair))
    print("  improved %d/%d;  larger than own seed spread in %d/%d"
          % (int((d_mu > 0).sum()), len(subs), beats_noise, len(subs)))
    print("  best published: %s %.4f -> branch leads by %+.4f (%d/%d, p=%.2e)"
          % (best[0], best[1], rows[-1][1] - best[1],
             int((b_mu > best[2]).sum()), len(subs), w.pvalue))
    e_s = arm("dn_stem", "ece").mean(0)
    e_b = arm("dn_stem_tan", "ece").mean(0)
    print("  ECE %.4f -> %.4f  lower in %d/%d  p=%.4f"
          % (e_s.mean(), e_b.mean(), int((e_b < e_s).sum()), len(subs),
             stats.wilcoxon(e_b, e_s).pvalue))


if __name__ == "__main__":
    main()
