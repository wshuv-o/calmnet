"""Every figure in the paper. Reads results/ only -- computes nothing.

One script and one style module for the whole document, so that a series has
the same colour, marker and dash in every figure (FIGURE_RULES.md §11).

  (fig_arch is NOT built here: it is drawn in paper/fig_arch.drawio and
   rendered by src/render_drawio.py, so the diagram stays editable in
   draw.io rather than living as matplotlib coordinates.)
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


# ===================================================================== fig 2
def fig_drift():
    # drift_fixed.json: no trace normalisation after the map, so the no-op
    # control is exactly null (drift.json is kept only for audit).
    d = L("drift_fixed.json")
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
        red = d[key]["summary"]["reduction"]
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
        if ends[i][0] - ends[i - 1][0] < 0.016:
            ends[i] = (ends[i - 1][0] + 0.016, ends[i][1], ends[i][2])
    for yv, lab, c in ends:
        ax.text(2.14, yv, lab, fontsize=6.0, color=c, va="center")
    ax.axvspan(-0.18, 1.0, color=VERM, alpha=0.07, lw=0, zorder=0)
    S.note(ax, 0.70, 0.574, "memory $<$" + NL + "class block", ha="center",
           va="top", color=VERM, fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$0.20$" + NL + r"$160$", r"$0.05$" + NL + r"$640$",
                        r"$0.01$" + NL + r"$3200$"], fontsize=6.6)
    ax.set_xlabel("momentum $m$ / memory (windows)")
    ax.set_ylabel("balanced accuracy")
    ax.set_xlim(-0.2, 3.35)
    ax.set_ylim(0.53, 0.80)
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

    # (c) measured effect of alignment against estimator memory, three seeds,
    # current estimator: align + gate minus gate only, for every cohort. No
    # upper bound is drawn: tau_drift was not measured.
    ax = axes[2]

    def m3(files, arm):
        vs = []
        for f in files:
            vs += [v["acc"] for k, v in L(f).items() if k.split("|")[0] == arm]
        return float(np.mean(vs)) if len(vs) >= 3 else None

    cohorts = [
        ("cohort A", 18, "noalign", m3(["a_gate_3seed.json"], "dn_gate"),
         [(160, m3(["a_aligngate_3seed.json"], "dn_noctx")),
          (3200, m3(["a_slow_noctx_3seed.json"], "dn_noctx"))]),
        ("cohort B", 309, "ours", m3(["b_gate_aligngate_3seed.json"], "dn_gate"),
         [(160, m3(["b_gate_aligngate_3seed.json"], "dn_noctx")),
          (320, m3(["b_floor_m0.10.json"], "dn_noctx")),
          (1600, m3(["b_floor_m0.02.json"], "dn_noctx")),
          (3200, m3(["b_aligngate_m001_3seed.json"], "dn_noctx"))]),
        ("cohort C", 5, "ours_alt", m3(["cohort3_m0.2.json", "cohort3_rep_m0.2.json"], "dn_gate"),
         [(160, m3(["cohort3_m0.2.json", "cohort3_rep_m0.2.json"], "dn_noctx")),
          (3200, m3(["cohort3_m0.01.json", "cohort3_rep_m0.01.json"], "dn_noctx"))]),
    ]
    ax.axhline(0.0, color=GREY, lw=0.8, zorder=1)
    ends = []
    for lab, blk, sk, gate, pts in cohorts:
        pts = [(mem, v - gate) for mem, v in pts if v is not None and gate is not None]
        if not pts:
            continue
        st = dict(SERIES[sk])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], **st)
        ax.axvline(blk, color=st["color"], ls=":", lw=0.9, zorder=1)
        ends.append((pts[-1][1], lab, st["color"]))
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 0.022:
            ends[i] = (ends[i - 1][0] + 0.022, ends[i][1], ends[i][2])
    for yv, lab, c in ends:
        ax.text(3900, yv, " " + lab, fontsize=6.4, color=c, va="center")
    ax.set_xscale("log")
    ax.set_xlim(3, 12000)
    ax.set_xlabel("estimator memory (windows)")
    ax.set_ylabel("effect of alignment")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
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
    fig_drift()
    fig_ablation()
    fig_rate()
    fig_efficiency()
    fig_inversion()
    fig_metrics()
