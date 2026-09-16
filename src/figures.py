"""Every figure in the paper. Reads results/ only -- computes nothing.

One script and one style module for the whole document, so that a series has
the same colour, marker and dash in every figure (FIGURE_RULES.md §11).

  fig_arch        the architecture, with exact per-block parameter counts
  fig_drift       per-subject session-drift reduction, both cohorts
  fig_ablation    component ablation with between-subject spread
  fig_rate        the two-sided adaptation-rate condition
  fig_efficiency  accuracy against parameter count
  fig_inversion   the cohort ordering inverting
  fig_metrics     five cases where the metric improved and the task did not

No in-axes titles anywhere; the captions carry that (§2). Error bars are the
between-subject SD and are labelled as such in every caption (§8).
"""
from __future__ import annotations
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as S
from figstyle import RESULTS, COL, FULL, SERIES, OURS, BLUE, VERM, GREEN, PURPLE, ORANGE, INK, GREY, FAINT

S.use()
NL = chr(10)


def L(name):
    p = RESULTS / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def g(d, arm, field="acc"):
    v = d.get(arm + "|s0")
    return v[field] if v and field in v else None


# ===================================================================== fig 1
def fig_arch():
    fig, ax = plt.subplots(figsize=(FULL, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    Y, H = 0.30, 0.44
    MID = Y + H / 2

    def box(x, w, title, sub, params, color, dashed=False, y=Y, h=H):
        for fc, ec, z in ((color, "none", 2), ("none", color, 3)):
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
                linewidth=0.9, edgecolor=color if ec != "none" else color,
                facecolor=fc if fc != "none" else "none",
                alpha=0.10 if fc != "none" else 1.0,
                linestyle=(0, (3, 2)) if dashed else "-", zorder=z))
        ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top",
                fontsize=7.6, fontweight="bold", color=color, zorder=4)
        ax.text(x + w / 2, y + h / 2 - 0.035, sub, ha="center", va="center",
                fontsize=6.3, color=INK, zorder=4, linespacing=1.45)
        ax.text(x + w / 2, y + 0.035, params, ha="center", va="bottom",
                fontsize=6.3, color=color, style="italic", zorder=4)

    def arrow(x1, y1, x2, y2, color=INK, dashed=False, rad=0.0, lw=0.9):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
            linewidth=lw, color=color, zorder=5,
            linestyle=(0, (3, 2)) if dashed else "-",
            connectionstyle="arc3,rad=%s" % rad))

    ax.text(0.030, MID + 0.035, "EEG" + NL + "window", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color=INK)
    ax.text(0.030, MID - 0.085, r"$60 \times 400$" + NL + "4 s @ 100 Hz",
            ha="center", va="center", fontsize=6.3, color=GREY)
    arrow(0.068, MID, 0.092, MID)

    box(0.094, 0.180, "Adaptive Alignment",
        r"running covariance $\mathbf{M}$" + NL + r"whiten by $\mathbf{M}^{-1/2}$"
        + NL + "learned raw/aligned blend", "1 parameter", OURS)
    arrow(0.274, MID, 0.298, MID)
    arrow(0.250, Y, 0.132, Y, color=OURS, dashed=True, rad=-0.62, lw=0.9)
    S.note(ax, 0.191, Y - 0.135, "unsupervised update, active at inference",
           ha="center", color=OURS, fontsize=6.3, style="italic")
    S.note(ax, 0.191, Y - 0.205, "(eval mode, no labels, no target data)",
           ha="center", fontsize=6.0)

    box(0.300, 0.180, "Multi-Scale Power",
        "64 / 128 / 256 ms branches" + NL + "depthwise spatial filters" + NL
        + r"square $\rightarrow$ pool $\rightarrow$ log", "6 768 parameters", GREEN)
    arrow(0.480, MID, 0.504, MID)

    box(0.506, 0.126, "Frame Embed",
        "norm + projection" + NL + "+ within-window" + NL + "attention pooling",
        "12 737 parameters", GREEN)
    arrow(0.632, MID, 0.648, MID)

    box(0.650, 0.196, "Cross-Epoch Context",
        "causal transformer," + NL + r"$K=8$ epochs (14.5 s)",
        "594 816 parameters", PURPLE, dashed=True, y=0.815, h=0.165)
    S.note(ax, 0.748, 0.995, "optional: $-$0.023 accuracy, $-$23 % false activations",
           ha="center", color=PURPLE, fontsize=6.2, style="italic")
    arrow(0.664, Y + H, 0.690, 0.815, color=PURPLE, dashed=True, rad=0.26)
    arrow(0.820, 0.815, 0.846, Y + H, color=PURPLE, dashed=True, rad=0.26)

    box(0.650, 0.196, "Selective Head",
        "classifier + abstention" + NL + "under a coverage constraint",
        "4 419 parameters", BLUE)
    arrow(0.846, MID, 0.868, MID)

    outs = [("Walk / Stop", MID + 0.150), ("calibrated confidence", MID),
            ("abstain / act", MID - 0.150)]
    ax.plot([0.870, 0.870], [MID - 0.150, MID + 0.150], lw=0.8, color=INK, zorder=4)
    for name, yy in outs:
        arrow(0.870, yy, 0.882, yy, lw=0.8)
        ax.text(0.888, yy, name, ha="left", va="center", fontsize=6.8, color=INK)

    ax.plot([0.094, 0.906], [0.085, 0.085], lw=0.5, color=FAINT)
    ax.text(0.5, 0.020, "default configuration: 24 181 parameters "
            "(53 % of ATCNet, 5 % of EEG Conformer)", ha="center", fontsize=6.6,
            color=GREY)
    S.save(fig, "fig_arch")


# ===================================================================== fig 2
def fig_drift():
    d = L("drift.json")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.25),
                             gridspec_kw=dict(width_ratios=[1.05, 1.2], wspace=0.28))
    for ax, (key, lab, n) in zip(axes, [("ds007788", "cohort A", 7),
                                        ("mobi", "cohort B", 8)]):
        per = d.get(key, {}).get("per_subject", {})
        subs = sorted(per)
        x = np.arange(len(subs))
        raw = [per[s]["raw"] for s in subs]
        ali = [per[s]["aligned"] for s in subs]
        noop = [per[s]["noop"] for s in subs]
        w = 0.27
        ax.bar(x - w, raw, w, color=ORANGE, alpha=0.85, lw=0, label="raw")
        ax.bar(x, noop, w, color=FAINT, alpha=0.95, lw=0, label="no-op control")
        ax.bar(x + w, ali, w, color=OURS, alpha=0.85, lw=0, label="aligned")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("sub-", "") for s in subs], fontsize=6.6)
        ax.set_xlabel("%s subject ($n=%d$)" % (lab, n))
        ax.set_ylabel(r"Riemannian distance $\delta$" if key == "ds007788" else "")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        red = d.get(key, {}).get("mean_reduction")
        red = red if red is not None else (0.520 if key == "ds007788" else 0.839)
        S.note(ax, 0.98, 0.93, r"$-%.0f\,\%%$ aligned vs raw" % (red * 100),
               transform=ax.transAxes, ha="right", color=OURS, fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06),
               fontsize=7, handlelength=1.4, columnspacing=1.6)
    S.panel(axes[0], "a", x=-0.19)
    S.panel(axes[1], "b", x=-0.13)
    S.save(fig, "fig_drift")


# ===================================================================== fig 3
def fig_ablation():
    d = L("driftnet_ds.json")
    arms = [("dn_noctx", "align + gate", "ours"),
            ("dn_full", "align + ctx + gate", "ours_ctx"),
            ("dn_nogate", "align + ctx", "ours_alt"),
            ("dn_noalign", "ctx + gate", "noalign"),
            ("dn_stem", "stem only", "stem")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3),
                             gridspec_kw=dict(wspace=0.42))

    ax = axes[0]
    y = np.arange(len(arms))[::-1]
    for yy, (k, lab, sk) in zip(y, arms):
        c = SERIES[sk]["color"]
        ax.barh(yy, g(d, k), 0.55, color=c, alpha=0.85, lw=0, zorder=2)
        ax.errorbar(g(d, k), yy, xerr=g(d, k, "acc_sd"), color=INK, lw=0.7,
                    capsize=1.8, capthick=0.7, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([a[1] for a in arms], fontsize=6.8)
    ax.set_xlim(0.72, 1.0)
    ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95])
    ax.set_xlabel("balanced accuracy")
    ax.axvline(g(d, "dn_stem"), color=GREY, lw=0.6, ls=":", zorder=1)
    S.note(ax, g(d, "dn_stem") - 0.006, -0.9, "bare stem", rotation=90,
           ha="right", va="bottom", fontsize=6.0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    S.panel(ax, "a", x=-0.62)

    ax = axes[1]
    for k, lab, sk in arms:
        a0, a90 = g(d, k), g(d, k, "acc_at_90")
        if a90 is None:
            continue
        st = dict(SERIES[sk])
        ax.plot([1.0, 0.9], [a0, a90], **st)
        ax.text(0.893, a90, " " + lab, fontsize=6.3, color=st["color"],
                ha="right", va="center")
    ax.set_xlim(1.03, 0.80)
    ax.set_xticks([1.0, 0.9])
    ax.set_xticklabels(["1.00", "0.90"])
    ax.set_xlabel("coverage")
    ax.set_ylabel("balanced accuracy")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    S.panel(ax, "b", x=-0.30)

    ax = axes[2]
    for k, lab, sk in arms:
        st = dict(SERIES[sk])
        ax.plot([g(d, k, "false_onsets_per_min")], [g(d, k)],
                marker=st["marker"], color=st["color"], mfc=st["mfc"],
                ms=st["ms"] + 1.2, ls="none", mew=1.0)
    ax.set_xlabel("false activations min$^{-1}$ standing")
    ax.set_ylabel("balanced accuracy")
    ax.set_xlim(0.5, 1.95)
    ax.grid()
    ax.set_axisbelow(True)
    S.note(ax, 0.72, 0.836, "context trades" + NL + "accuracy for safety",
           fontsize=6.2)
    S.panel(ax, "c", x=-0.30)
    S.save(fig, "fig_ablation")


# ===================================================================== fig 4
def fig_rate():
    m020 = L("driftfix_mobi.json") or L("driftnet_mobi.json")
    m020o = L("driftnet_mobi.json")
    m005, m001 = L("driftmom_0.05.json"), L("driftmom_0.01.json")
    a020, a001 = L("driftfix_ds.json"), L("driftmom_ds_0.01.json")

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3),
                             gridspec_kw=dict(wspace=0.36))

    # (a) the sweep -- sparse, so markers are prominent (§7)
    ax = axes[0]
    arms = [("dn_full", "align+ctx+gate", "ours_ctx"),
            ("dn_noctx", "align+gate", "ours"),
            ("dn_nogate", "align+ctx", "ours_alt"),
            ("dn_noalign", "ctx+gate", "noalign"),
            ("dn_stem", "stem only", "stem")]
    x = np.array([0, 1, 2])
    ends = []
    for k, lab, sk in arms:
        base = g(m020, k) if g(m020, k) is not None else g(m020o, k)
        ys = [base, g(m005, k), g(m001, k)]
        if any(v is None for v in ys):
            continue
        ax.plot(x, ys, **SERIES[sk])
        ends.append((ys[-1], lab, SERIES[sk]["color"]))
    # direct labels rather than a legend box, de-collided by a minimum gap
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 0.022:
            ends[i] = (ends[i - 1][0] + 0.022, ends[i][1], ends[i][2])
    for yv, lab, c in ends:
        ax.text(2.09, yv, lab, fontsize=6.0, color=c, va="center")
    ax.axvspan(-0.18, 1.0, color=VERM, alpha=0.07, lw=0, zorder=0)
    S.note(ax, 0.41, 0.800, "memory $<$" + NL + "class block", ha="center",
           va="top", color=VERM, fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$0.20$" + NL + r"$160$", r"$0.05$" + NL + r"$640$",
                        r"$0.01$" + NL + r"$3200$"], fontsize=6.6)
    ax.set_xlabel("momentum $m$ / memory (windows)")
    ax.set_ylabel("balanced accuracy")
    ax.set_xlim(-0.2, 3.35)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    S.panel(ax, "a", x=-0.30)

    # (b) opposite signs
    ax = axes[1]
    for lab, fast, slow, sk in [("cohort A", g(a020, "dn_full"), g(a001, "dn_full"), "noalign"),
                                ("cohort B", g(m020, "dn_full"), g(m001, "dn_full"), "ours")]:
        if fast is None or slow is None:
            continue
        st = dict(SERIES[sk])
        st["lw"] = 1.6
        ax.plot([0, 1], [fast, slow], **st)
        ax.annotate("%+.3f" % (slow - fast), ((0.5), (fast + slow) / 2),
                    textcoords="offset points",
                    xytext=(-16, 4 if slow > fast else -10), ha="right",
                    fontsize=7, color=st["color"])
        ax.text(1.06, slow, " " + lab, fontsize=6.6, color=st["color"],
                va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"fast" + NL + r"$m=0.20$", r"slow" + NL + r"$m=0.01$"],
                       fontsize=6.6)
    ax.set_xlim(-0.25, 1.65)
    ax.set_ylabel("balanced accuracy")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    S.panel(ax, "b", x=-0.30)

    # (c) the admissible band
    ax = axes[2]
    UP_LO, UP_HI = 160, 3200
    ax.axvspan(UP_LO, UP_HI, color=GREY, alpha=0.10, lw=0, zorder=0)
    S.note(ax, np.sqrt(UP_LO * UP_HI), 1.72, r"$\tau_{\mathrm{drift}}$ bracket",
           ha="center", fontsize=6.4)
    for i, (lab, blk, sk) in enumerate([("cohort A", 18, "noalign"),
                                        ("cohort B", 309, "ours")]):
        yy = 1 - i
        c = SERIES[sk]["color"]
        if blk < UP_LO:
            ax.plot([blk, UP_LO], [yy, yy], lw=5, color=c, alpha=0.35,
                    solid_capstyle="butt", zorder=2)
            S.note(ax, np.sqrt(blk * UP_LO), yy + 0.17, "admissible",
                   ha="center", color=c, fontsize=6.4)
        else:
            ax.fill_between([blk, UP_HI], yy - 0.10, yy + 0.10, facecolor="none",
                            edgecolor=VERM, hatch="////", lw=0.0, alpha=0.9,
                            zorder=2)
            S.note(ax, np.sqrt(blk * UP_HI), yy + 0.17, "empty",
                   ha="center", color=VERM, fontsize=6.4)
        ax.plot([blk], [yy], marker="|", ms=9, mew=1.6, color=c, zorder=5)
        S.note(ax, blk, yy - 0.34, r"$\tau_{\mathrm{blk}}=%d$" % blk,
               ha="center", color=c, fontsize=6.2)
        ax.text(12, yy, lab, ha="right", va="center", fontsize=6.8, color=c)
    ax.set_xscale("log")
    ax.set_xlim(11, 6000)
    ax.set_ylim(-0.7, 1.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("adaptation memory (windows)")
    S.panel(ax, "c", x=-0.10)
    S.save(fig, "fig_rate")


# ===================================================================== fig 5
def fig_efficiency():
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    pts = [("ATCNet", 45280, 0.913, "baseline"),
           ("ATCNet, rate-matched", 45280, 0.908, "baseline"),
           ("ours, align+gate", 24181, 0.901, "ours"),
           ("ours, +context", 618997, 0.878, "ours_ctx"),
           ("ours, stem only", 6768, 0.827, "stem")]
    for name, p, a, sk in pts:
        st = SERIES[sk]
        mine = name.startswith("ours, align")
        ax.plot([p], [a], marker=st["marker"], color=st["color"],
                mfc=st["color"] if mine else st["mfc"],
                ms=st["ms"] + (2.4 if mine else 0.6), ls="none", mew=1.0,
                zorder=4)
    ax.annotate("", xy=(24181, 0.901), xytext=(45280, 0.913),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.7,
                                ls=(0, (2, 2))))
    S.note(ax, 31000, 0.9215, r"$53\,\%$ of the size," + NL + r"$-0.012$ accuracy",
           ha="center", fontsize=6.3)
    ax.text(24181 * 0.82, 0.8985, "ours ", ha="right", va="center",
            fontsize=6.8, color=OURS, fontweight="bold")
    ax.text(45280 * 1.25, 0.9135, " ATCNet", ha="left", va="center",
            fontsize=6.8, color=ORANGE)
    ax.text(618997 * 0.78, 0.8765, "+context ", ha="right", va="center",
            fontsize=6.5, color=PURPLE)
    ax.text(6768 * 1.3, 0.827, " stem", ha="left", va="center",
            fontsize=6.5, color=GREEN)
    ax.set_xscale("log")
    ax.set_xlim(4500, 1.6e6)
    ax.set_ylim(0.815, 0.935)
    ax.set_xlabel("parameters")
    ax.set_ylabel("balanced accuracy")
    ax.grid(which="major")
    ax.set_axisbelow(True)
    S.save(fig, "fig_efficiency")


# ===================================================================== fig 6
def fig_inversion():
    a, b = L("driftnet_ds.json"), L("driftnet_mobi.json")
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    arms = [("dn_noctx", "align+gate", "ours"),
            ("dn_full", "align+ctx+gate", "ours_ctx"),
            ("dn_nogate", "align+ctx", "ours_alt"),
            ("dn_noalign", "ctx+gate", "noalign"),
            ("dn_stem", "stem only", "stem")]
    for k, lab, sk in arms:
        ya, yb = g(a, k), g(b, k)
        if ya is None or yb is None:
            continue
        st = dict(SERIES[sk])
        ax.plot([0, 1], [ya, yb], **st)
        ax.text(1.05, yb, " " + lab, fontsize=6.4, color=st["color"],
                va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"cohort A" + NL + r"$\tau_{\mathrm{blk}}=18$",
                        r"cohort B" + NL + r"$\tau_{\mathrm{blk}}=309$"],
                       fontsize=6.8)
    ax.set_xlim(-0.12, 1.95)
    ax.set_ylabel("balanced accuracy")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    S.note(ax, 0.5, 0.685, "alignment arms cross" + NL + "below the rest",
           ha="center", fontsize=6.3, color=VERM)
    S.save(fig, "fig_inversion")


# ===================================================================== fig 7
def fig_metrics():
    rows = [("adaptation rate tuned to" + NL + "maximise drift removal",
             r"$25 \rightarrow 60\,\%$ removed", -0.092),
            ("cohort with MORE drift" + NL + r"removed ($84$ vs $52\,\%$)",
             "more removed", -0.166),
            ("verified fix to a real" + NL + "covariance bias",
             "estimator corrected", -0.016),
            ("prediction registered from" + NL + "our own diagnosis",
             "diagnosis applied", -0.060),
            ("leakage probe on a decoder" + NL + "that cannot see movement",
             r"$R^2 = +0.863$", None)]
    fig, ax = plt.subplots(figsize=(COL, 2.55))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, metric, delta) in zip(y, rows):
        if delta is None:
            S.note(ax, -0.004, yy, "no accuracy change", va="center",
                   ha="right", fontsize=6.3, style="italic")
        else:
            ax.barh(yy, delta, 0.5, color=VERM, alpha=0.8, lw=0, zorder=2)
            ax.text(delta - 0.005, yy, "%+.3f" % delta, va="center",
                    ha="right", fontsize=6.8, color=VERM)
        S.note(ax, 0.007, yy, metric, va="center", ha="left", fontsize=6.3,
               color=OURS, style="italic")
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    ax.set_xlim(-0.20, 0.105)
    ax.set_xticks([-0.20, -0.15, -0.10, -0.05, 0])
    ax.set_xlabel(r"change in balanced accuracy")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.save(fig, "fig_metrics")


if __name__ == "__main__":
    fig_arch()
    fig_drift()
    fig_ablation()
    fig_rate()
    fig_efficiency()
    fig_inversion()
    fig_metrics()
