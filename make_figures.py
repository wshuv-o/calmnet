"""Regenerate every paper and supplementary figure into figures/ (PDF + PNG).

    python make_figures.py                  all figures
    python make_figures.py fig_cone ...     named figures
    python make_figures.py --recompute      rebuild caches from raw data (CALMNET_DATA)
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figstyle as S
from analysis import figure_data as F

STYLE = {"C": dict(color=S.OURS, marker="o"), "D": dict(color=S.PURPLE, marker="D"),
         "E": dict(color=S.ORANGE, marker="P"), "A": dict(color=S.BLUE, marker="^"),
         "B": dict(color=S.VERM, marker="s")}


def _pane(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#DDDDDD")
        axis._axinfo["grid"]["linewidth"] = 0.3
        axis._axinfo["grid"]["color"] = "#EEEEEE"


# ------------------------------------------------------- model internals --
def fig_activations(recompute=False):
    from sklearn.decomposition import PCA
    A = F.activations_data(recompute)
    y = A["y"]
    stop, walk = y == 0, y == 1
    n_chan = int(round((np.sqrt(8 * A["tangent"].shape[1] + 1) - 1) / 2))

    def unvec(v):
        """Fold a tangent vector back into its symmetric matrix."""
        iu = np.triu_indices(n_chan)
        M = np.zeros((n_chan, n_chan))
        M[iu] = v * np.where(iu[0] == iu[1], 1.0, 1.0 / np.sqrt(2.0))
        return M + M.T - np.diag(np.diag(M))

    Ms, Mw = unvec(A["tangent"][stop].mean(0)), unvec(A["tangent"][walk].mean(0))
    D = Mw - Ms
    S.use()
    fig = plt.figure(figsize=(S.FULL, 3.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], hspace=0.55, wspace=0.46,
                          width_ratios=[1, 1, 1, 1.5])
    for k, (M, ttl) in enumerate(((Ms, "Stop"), (Mw, "Walk"), (D, "Walk $-$ Stop"))):
        ax = fig.add_subplot(gs[0, k])
        lim = np.abs(D).max() if k == 2 else max(np.abs(Ms).max(), np.abs(Mw).max())
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(ttl, fontsize=7.4, pad=4)
        ax.set_xticks([]), ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5.4)
        S.panel(ax, "abc"[k], x=-0.10, y=1.20)
    fig.text(0.055, 0.540, "tangent map, folded to $%d \\times %d$ (mean over held-out windows)"
             % (n_chan, n_chan), fontsize=6.6, color=S.GREY)

    ax = fig.add_subplot(gs[0, 3])
    names = ["tangent", "project", "stem", "fuse"]
    lbl = ["tangent\nmap", "project", "frame\nembed", "fuse"]
    dvals = []
    for n in names:
        Z = A[n]
        s = np.sqrt(0.5 * (Z[walk].var(0) + Z[stop].var(0))) + 1e-9
        dvals.append(float(np.abs((Z[walk].mean(0) - Z[stop].mean(0)) / s).mean()))
    ax.bar(range(4), dvals, color=[S.GREY, S.OURS, S.GREY, S.OURS], width=0.62)
    ax.set_xticks(range(4))
    ax.set_xticklabels(lbl, fontsize=5.6)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel("mean $|d|$ per unit", fontsize=6.8)
    ax.tick_params(axis="y", labelsize=6.0)
    for s_ in ("top", "left"):
        ax.spines[s_].set_visible(False)
    S.panel(ax, "d", x=-0.16, y=1.20)

    for k, (n, ttl) in enumerate(zip(names, lbl)):
        ax = fig.add_subplot(gs[1, k])
        Z = PCA(n_components=2, random_state=0).fit_transform(A[n])
        ax.scatter(Z[stop, 0], Z[stop, 1], s=4, color=S.GREY, alpha=0.55, linewidths=0, label="Stop")
        ax.scatter(Z[walk, 0], Z[walk, 1], s=4, color=S.VERM, alpha=0.55, linewidths=0, label="Walk")
        ax.set_title(ttl.replace("\n", " "), fontsize=7.0, pad=3)
        ax.set_xticks([]), ax.set_yticks([])
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        S.panel(ax, "efgh"[k], x=-0.10, y=1.16)
        if k == 0:
            ax.legend(fontsize=5.8, frameon=False, loc="upper left", handletextpad=0.2,
                      borderpad=0.1)
    S.save(fig, "fig_activations")


def _info(names, kind="standard_1020"):
    import mne
    info = mne.create_info(list(names), 100.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage(kind), on_missing="ignore")
    return info


def fig_topography(recompute=False):
    import mne
    D = F.topography_data(recompute)
    info = _info([str(c) for c in D["ch"]])
    S.use()
    fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 2.4))
    band = F.cfg("A")["data"]["band"]
    for ax, vals, ttl, cmap in (
            (axes[0], D["power"], "channel power at the reference", "viridis"),
            (axes[1], D["gain"], r"gain of $\mathbf{M}^{-1/2}$ in the tangent map", "magma"),
            (axes[2], D["diff"], "%g–%g Hz power, Walk $-$ Stop (dB)" % tuple(band), "RdBu_r")):
        vv = np.asarray(vals, float)
        lim = np.abs(vv).max() if cmap == "RdBu_r" else None
        im, _ = mne.viz.plot_topomap(vv, info, axes=ax, show=False, cmap=cmap, contours=4,
                                     vlim=(-lim, lim) if lim else (None, None), sensors=True)
        ax.set_title(ttl, fontsize=7.2, pad=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cb.ax.tick_params(labelsize=5.8)
    for ax_, L_ in zip(axes, "abc"):
        S.panel(ax_, L_, x=-0.20, y=1.16)
    S.save(fig, "fig_topography")


def fig_topography_group(recompute=False):
    import mne
    res = F.topography_group(recompute)
    mean_d = np.array(res["mean_diff_db"])
    p_d = np.array(res["signflip_p_maxstat"])
    agree = np.array(res["sign_agreement_of_7"], float)
    Dd = np.array(res["per_subject_diff_db"])
    n_sub = len(res["subjects"])
    info = _info(res["channels"])
    S.use()
    fig = plt.figure(figsize=(S.FULL, 3.1))
    gs = fig.add_gridspec(2, n_sub, height_ratios=[1.6, 0.8], hspace=0.28,
                          left=0.03, right=0.97, top=0.93, bottom=0.06)
    row = gs[0, :].subgridspec(1, 3, wspace=0.55)
    top = [fig.add_subplot(row[0, i]) for i in range(3)]
    md = np.abs(mean_d).max()
    for ax, vals, lab, cmap, vl, msk in (
            (top[0], mean_d, "Walk $-$ Stop, 8–30 Hz (dB)", "RdBu_r", (-md, md), p_d < 0.05),
            (top[1], agree, "participants sharing the sign (of %d)" % n_sub, "Greys",
             (n_sub / 2.0, n_sub), None),
            (top[2], np.array(res["mean_gain"]), r"gain of $\mathbf{M}^{-1/2}$", "magma",
             (None, None), None)):
        im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=cmap, contours=4,
                                     vlim=vl, sensors=True, mask=msk,
                                     mask_params=dict(marker="o", markerfacecolor="w",
                                                      markeredgecolor="k", markersize=3))
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cb.ax.tick_params(labelsize=5.8)
        ax.text(0.5, -0.08, lab, transform=ax.transAxes, ha="center", va="top", fontsize=6.4)
    for ax_, L_ in zip(top, "abc"):
        S.panel(ax_, L_, x=-0.12, y=1.02)
    lim = np.abs(Dd).max()
    first = None
    for j, s in enumerate(res["subjects"]):
        ax = fig.add_subplot(gs[1, j])
        first = first or ax
        mne.viz.plot_topomap(Dd[j], info, axes=ax, show=False, cmap="RdBu_r", contours=0,
                             vlim=(-lim, lim), sensors=False)
        ax.text(0.5, -0.12, s, transform=ax.transAxes, ha="center", fontsize=6)
    S.panel(first, "d", x=-0.25, y=1.0)
    S.save(fig, "fig_topography_group")


# ------------------------------------------------------ reference geometry --
def fig_cone(recompute=False):
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    D = F.cone_data(recompute)
    tau = F.tau_blk()

    def xyz(C):
        a, b, c = C[..., 0, 0], C[..., 0, 1], C[..., 1, 1]
        return (a - c) / 2, b, (a + c) / 2

    S.use()
    fig = plt.figure(figsize=(S.FULL, 3.3))
    rng = np.random.default_rng(0)
    for k, tag in enumerate(("A", "B")):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.computed_zorder = False
        x, y, z = xyz(D[tag + "_pts"])
        cls = D[tag + "_y"]
        # zoom to the bulk; the top 2% fall outside the view (stated in the caption)
        zmax = float(np.quantile(z, 0.98))
        R = float(np.quantile(np.abs(np.r_[x, y]), 0.98))
        keep = (z <= zmax) & (np.abs(x) <= R) & (np.abs(y) <= R)
        hr = min(zmax, R)
        th, h = np.meshgrid(np.linspace(0, 2 * np.pi, 72), np.linspace(0, hr, 10))
        ax.plot_surface(h * np.cos(th), h * np.sin(th), h, color="#999999", alpha=0.06,
                        linewidth=0, shade=False)
        ax.plot_wireframe(h * np.cos(th), h * np.sin(th), h, color=S.FAINT, linewidth=0.3,
                          rstride=3, cstride=6, alpha=0.9)
        t_ = np.linspace(0, 2 * np.pi, 120)
        ax.plot(hr * np.cos(t_), hr * np.sin(t_), np.full_like(t_, hr), color="#AAAAAA", lw=0.6)
        for c, col, mk in ((0, S.BLUE, "o"), (1, S.VERM, "s")):
            m = np.flatnonzero((cls == c) & keep)
            m = rng.choice(m, min(len(m), 700), replace=False)
            ax.scatter(x[m], y[m], z[m], s=1.2, c=col, marker=mk, alpha=0.35, linewidths=0,
                       depthshade=False, rasterized=True)
        px, py, pz = xyz(D[tag + "_path"])
        P = np.stack([px, py, pz], 1)
        seg = np.stack([P[:-1], P[1:]], 1)
        nb = len(seg)
        # each step coloured by the majority class of the batch behind it
        maj = np.array([cls[i * F.BATCH:(i + 1) * F.BATCH].mean() >= 0.5 for i in range(nb)])
        ax.add_collection3d(Line3DCollection(seg, colors=[S.OURS] * nb, linewidths=1.9, zorder=11))
        ax.add_collection3d(Line3DCollection(seg, colors=np.where(maj, S.VERM, S.BLUE),
                                             linewidths=0.9, zorder=12))
        ax.scatter([px[0]], [py[0]], [pz[0]], s=22, facecolors="white", edgecolors=S.OURS,
                   linewidths=1.0, depthshade=False, zorder=13)
        for key, col, mk in (("_stopmean", S.BLUE, "o"), ("_walkmean", S.VERM, "s")):
            cx, cy, cz = xyz(D[tag + key])
            ax.scatter([cx], [cy], [cz], s=46, c=col, marker=mk, edgecolors="white",
                       linewidths=0.9, depthshade=False, zorder=20)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(0, zmax)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel(r"$(a-c)/2$", labelpad=5, fontsize=7)
        ax.set_ylabel(r"$b$", labelpad=1)
        ax.set_zlabel(r"$(a+c)/2$", labelpad=1)
        ax.tick_params(pad=0, labelsize=5.5)
        _pane(ax)
        ax.view_init(elev=18, azim=-58)
        ax.set_box_aspect((1, 1, 0.8))
        ax.text2D(0.02, 0.97, r"(%s) cohort %s, $\tau_{\mathrm{blk}} = %d$" % ("ab"[k], tag, tau[tag]),
                  transform=ax.transAxes, fontsize=8, va="top")
    h = [Line2D([], [], ls="", marker="o", ms=2.5, color=S.BLUE, alpha=0.6),
         Line2D([], [], ls="", marker="s", ms=2.5, color=S.VERM, alpha=0.6),
         Line2D([], [], ls="", marker="o", ms=5.5, color=S.BLUE, mec="white"),
         Line2D([], [], ls="", marker="s", ms=5.5, color=S.VERM, mec="white"),
         Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec=S.OURS),
         (Line2D([], [], color=S.OURS, lw=1.9), Line2D([], [], color=S.BLUE, lw=0.9)),
         (Line2D([], [], color=S.OURS, lw=1.9), Line2D([], [], color=S.VERM, lw=0.9))]
    fig.legend(h, ["Stop window", "Walk window", "Stop mean", "Walk mean",
                   "reference at test start", "reference, Stop streaming",
                   "reference, Walk streaming"],
               loc="lower center", ncol=4, fontsize=6.5, handlelength=1.6, columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.12, top=1.0, wspace=0.02)
    S.save(fig, "fig_cone", pdf_dpi=300)


def fig_refmaps(recompute=False):
    import mne
    D = F.refmaps_data(recompute)
    head_r = 0.12      # outermost electrode of either montage sits at 0.116
    mont = mne.channels.make_standard_montage("standard_1005")
    S.use()
    fig = plt.figure(figsize=(S.FULL, 3.5))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 0.26, 1, 1, 0.26], wspace=0.10, hspace=0.20,
                          left=0.05, right=0.93, top=0.86, bottom=0.05)
    # colour limits at the 98th percentile of the maps compared
    lim, ims = {}, {}
    for q in ("drift", "contrast"):
        lim[q] = float(np.quantile(np.concatenate(
            [np.abs(D["%s_%s_%s" % (c, q, r)]) for c in "AB" for r in ("frozen", "running")]), 0.98))
    for row, tag in enumerate(("A", "B")):
        names = [str(c) for c in D[tag + "_ch"]]
        keep = [i for i, c in enumerate(names)
                if c in mont.ch_names and not (tag == "B" and c in F.MOBI_EOG)]
        info = mne.create_info([names[i] for i in keep], 100.0, "eeg")
        info.set_montage(mont, on_missing="ignore", verbose="ERROR")
        for col, (q, r) in enumerate((("drift", "frozen"), ("drift", "running"),
                                      ("contrast", "frozen"), ("contrast", "running"))):
            ax = fig.add_subplot(gs[row, col + (1 if col >= 2 else 0)])
            v = D["%s_%s_%s" % (tag, q, r)][keep]
            im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r",
                                         vlim=(-lim[q], lim[q]), contours=3, sensors=False,
                                         outlines="head", sphere=(0.0, 0.0, 0.0, head_r),
                                         extrapolate="head")
            ims[q] = im
            if row == 0:
                ax.set_title("%s reference" % r, fontsize=7.5, pad=3)
            if col == 0:
                ax.text(-0.18, 0.5, "cohort %s" % tag, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=8)
            S.note(ax, 0.5, -0.06, "%.2f" % np.abs(v).mean(), ha="center", va="top",
                   transform=ax.transAxes)
            if row == 0 and col in (0, 2):
                ax.text(1.07, 1.24, "session drift" if q == "drift" else "Walk $-$ Stop contrast",
                        transform=ax.transAxes, ha="center", fontsize=8)
    for q, c in (("drift", 2), ("contrast", 5)):
        cax = fig.add_subplot(gs[:, c])
        box = cax.get_position()
        cax.set_position([box.x0 + 0.012, box.y0 + 0.20, 0.010, box.height * 0.56])
        cb = fig.colorbar(ims[q], cax=cax)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)
        cb.set_label("SD of the fitting split", fontsize=6.5, labelpad=2)
    S.save(fig, "fig_refmaps", pdf_dpi=300)


# label offsets (points) and alignment on the crowded plateau
RATE_PLACE = {("D", 160): (0, 6, "center", "bottom"), ("C", 160): (0, -6, "center", "top"),
              ("B", 3200): (2, -6, "left", "top"), ("E", 160): (-2, 6, "right", "bottom"),
              ("A", 160): (2, 6, "left", "bottom"), ("B", 1280): (2, -6, "left", "top"),
              ("A", 40): (0, -6, "center", "top"), ("C", 8): (-3, 6, "right", "bottom"),
              ("C", 4): (6, 0, "left", "center"), ("B", 160): (2, -6, "left", "top"),
              ("A", 8): (-6, 0, "right", "center"), ("C", 2): (6, 0, "left", "center")}


def fig_rate_collapse(recompute=False):
    D = F.rate_data(recompute)
    tau, dacc = F.tau_blk(), F.sweep_delta()
    S.use()
    fig, ax = plt.subplots(figsize=(S.FULL, 2.7))
    u = np.logspace(-2.6, 1.4, 400)
    ax.plot(u, np.tanh(u) / u, color=S.GREY, lw=1.0, zorder=2)
    pts = sorted((tau[t] / (2.0 * m), t, m) for t in F.RATE_CONFIG
                 for m in (int(k.split("_")[-1]) for k in D if k.startswith(t + "_meas")))
    ax.axvline(0.5, color=S.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    S.note(ax, 0.5, 0.02, r"$\mu = \tau_{\mathrm{blk}}$", ha="center", va="bottom",
           transform=ax.get_xaxis_transform())
    for uu, tag, mu in pts:
        z = float(D["%s_meas_%d" % (tag, mu)].mean())
        da = dacc[(tag, mu)]
        helps = da > 0
        col = S.OURS if helps else S.VERM
        ax.plot([uu], [z], marker="o" if helps else "s", ms=5.0, color=col, mec="white",
                mew=0.7, zorder=4)
        dx, dy, ha, va = RATE_PLACE.get((tag, mu), (2, 6, "left", "bottom"))
        ax.annotate("%s $\\mu{=}%d$\n$%+.3f$" % (tag, mu, da), (uu, z), (dx, dy),
                    textcoords="offset points", fontsize=5.8, color=S.GREY if helps else col,
                    ha=ha, va=va, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau_{\mathrm{blk}} / 2\mu$   (block length per unit of memory)")
    ax.set_ylabel("surviving class contrast")
    ax.set_xlim(1.5e-3, 25); ax.set_ylim(0, 1.64)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    S.note(ax, 0.985, 0.93, r"$s=\tanh u\,/\,u$", ha="right", transform=ax.transAxes, fontsize=7.5)
    S.note(ax, 0.015, 0.05, "label: cohort, memory, change in balanced accuracy", ha="left",
           transform=ax.transAxes)
    fig.tight_layout(pad=0.3)
    S.save(fig, "fig_rate_collapse", pdf_dpi=300)


# --------------------------------------------------- per-participant results --
def _sweep_panel(ax, rows, letter, cohorts, tau):
    rng = np.random.default_rng(0)
    for coh, x, dlt in rows:
        st = STYLE[coh]
        xj = x * np.exp(rng.uniform(-0.07, 0.07, len(dlt)))
        ax.scatter(xj, dlt, s=7, facecolor="none", edgecolor=st["color"], marker=st["marker"],
                   lw=0.6, alpha=0.75, zorder=2)
        ax.plot([x / 1.25, x * 1.25], [dlt.mean()] * 2, color=st["color"], lw=1.8, zorder=3)
    ax.axhline(0, color=S.GREY, lw=0.6, zorder=1)
    ax.axvline(1, color=S.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"memory / median class block, $\mu/\tau_{\mathrm{blk}}$")
    for coh in cohorts:
        st = STYLE[coh]
        ax.scatter([], [], facecolor="none", edgecolor=st["color"], marker=st["marker"], s=12,
                   label=r"%s ($\tau_{\mathrm{blk}}=%d$)" % (coh, tau[coh]))
    ax.legend(loc="lower right", fontsize=6.5, frameon=False, handletextpad=0.2)
    S.panel(ax, letter)


def fig_r_sweep(recompute=False):
    tau = F.tau_blk()
    S.use()
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL, 2.5), sharey=True)
    rows = [(coh, (n / m) / tau[coh], F.delta(fb, fc)[1])
            for coh, n, m, fb, fc in F.SWEEP if coh in ("C", "A", "B")]
    _sweep_panel(axs[0], rows, "a", ("C", "A", "B"), tau)
    axs[0].set_ylabel(r"$\Delta$ balanced accuracy")
    rows = [(coh, mu / tau[coh], F.delta(fb, F.CAUSAL_STEM[coh])[1]) for coh, mu, fb in F.CAUSAL]
    _sweep_panel(axs[1], rows, "b", F.COHORTS, tau)
    fig.tight_layout(w_pad=1.0)
    S.save(fig, "fig_r_sweep")


def fig_r_cohorts(recompute=False):
    S.use()
    fig, ax = plt.subplots(figsize=(S.FULL, 2.4))
    rng = np.random.default_rng(0)
    for j, coh in enumerate(F.COHORTS):
        st = STYLE[coh]
        subs, off = F.delta(*F.OFFLINE[coh])
        _, cau = F.delta(F.CAUSAL_STEM[coh], F.CAUSAL_STEM[coh])
        for k, (d, filled) in enumerate(((off, False), (cau, True))):
            x0 = 2.6 * j + 0.9 * k
            ax.scatter(x0 + rng.uniform(-0.18, 0.18, len(d)), d, s=8, marker=st["marker"], lw=0.6,
                       zorder=2, facecolor=st["color"] if filled else "none", edgecolor=st["color"])
            lo, hi = np.percentile(rng.choice(d, (10000, len(d))).mean(1), [2.5, 97.5])
            ax.plot([x0 + 0.3] * 2, [lo, hi], color=st["color"], lw=1.0, zorder=3)
            ax.plot([x0 + 0.22, x0 + 0.38], [d.mean()] * 2, color=st["color"], lw=1.8, zorder=3)
        ax.text(2.6 * j + 0.45, 1.0, "%s (n = %d)" % (coh, len(subs)), ha="center", va="bottom",
                fontsize=7, transform=ax.get_xaxis_transform())
    ax.axhline(0, color=S.GREY, lw=0.6, zorder=1)
    ax.set_xticks([2.6 * j + 0.9 * k for j in range(5) for k in range(2)])
    ax.set_xticklabels(["submitted", "causal"] * 5, fontsize=6.5)
    ax.set_xlim(-0.5, 2.6 * 4 + 1.5)
    ax.set_ylabel(r"$\Delta$ balanced accuracy")
    S.save(fig, "fig_r_cohorts")


def fig_r_deploy(recompute=False):
    S.use()
    fig, axs = plt.subplots(2, 5, figsize=(S.FULL, 3.0), sharex=True)
    for j, coh in enumerate(("A", "B", "C", "D", "E")):
        d = F.J(F.CAUSAL_STEM[coh])
        for i, (metric, lab) in enumerate((("false_onsets_per_min", "false activations / min"),
                                           ("missed_onsets", "missed onsets (fraction)"))):
            subs, tan = F.seedavg(d, "stem_tangent", metric)
            _, stem = F.seedavg(d, "stem", metric, subs)
            ax = axs[i, j]
            for a, b in zip(stem, tan):
                ax.plot([0, 1], [a, b], color=S.FAINT, lw=0.6, zorder=1)
            ax.scatter(np.zeros_like(stem), stem, s=8, facecolor="none", edgecolor=S.GREEN,
                       marker="v", lw=0.7, zorder=2)
            ax.scatter(np.ones_like(tan), tan, s=8, color=S.OURS, marker="o", zorder=2)
            ax.plot([-0.2, 0.2], [np.nanmean(stem)] * 2, color=S.GREEN, lw=1.6)
            ax.plot([0.8, 1.2], [np.nanmean(tan)] * 2, color=S.OURS, lw=1.6)
            if metric == "missed_onsets":
                top = np.nanmax(np.r_[stem, tan])
                ax.set_ylim(-0.03 * max(top, 0.1), 1.08 * max(top, 0.1))
                if top == 0:
                    S.note(ax, 0.5, 0.55, "none missed", ha="center", transform=ax.transAxes)
            ax.set_xlim(-0.45, 1.45)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["stem", "+ branch"])
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
            if i == 0:
                ax.text(0.5, 1.03, "cohort %s (n = %d)" % (coh, len(subs)),
                        transform=ax.transAxes, ha="center", fontsize=7)
    fig.tight_layout(w_pad=0.6, h_pad=0.4)
    S.save(fig, "fig_r_deploy")


def fig_s_blocks(recompute=False):
    ps = F.J("revision/protocol_stats.json")
    mu = int(round(F.BATCH / F.MOM))
    S.use()
    fig, axs = plt.subplots(1, 5, figsize=(S.FULL, 1.9), sharey=True)
    for ax, coh in zip(axs, F.COHORTS):
        for split, kw in (("fit", dict(color=S.OURS, lw=1.3, ls="-")),
                          ("test", dict(color=S.VERM, lw=1.0, ls=(0, (3, 1.5))))):
            c = ps[coh][split]["counts"]
            L = np.array(sorted(int(k) for k in c))
            n = np.array([c[str(k)] for k in L], float)
            cum = np.cumsum(L * n) / (L * n).sum()      # fraction of windows in blocks <= L
            ax.step(np.r_[0.8, L, 4000], np.r_[0, cum, 1], where="post", label=split, **kw)
        ax.axvline(mu, color=S.GREY, lw=0.6, ls=(0, (1, 1)))
        ax.set_xscale("log")
        ax.set_xlim(0.8, 4000)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1000"])
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("block length (windows)", fontsize=7)
        ax.text(0.5, 1.03, "cohort %s" % coh, transform=ax.transAxes, ha="center", fontsize=7)
    axs[0].set_ylabel("fraction of windows\nin blocks $\\leq$ length", fontsize=7)
    S.note(axs[0], mu * 0.94, 0.5, r"$\mu=%d$ " % mu, ha="right")
    axs[0].legend(loc="upper right", fontsize=6.5, frameon=False, handlelength=1.6)
    fig.tight_layout(w_pad=0.5)
    S.save(fig, "fig_s_blocks")


def fig_s1_delta(recompute=False):
    allD, tau = F.supplement_deltas(), F.tau_blk()
    S.use()
    fig, ax = plt.subplots(figsize=(S.FULL, 2.3))
    rng = np.random.default_rng(0)
    for i, coh in enumerate(F.COHORTS):
        d, dca = allD[coh]
        col, mk = (S.VERM, "s") if coh == "B" else (S.OURS, "o")
        jit = rng.uniform(-0.13, 0.13, len(d))
        ax.plot(i - 0.17 + jit, d, mk, ms=3.6, color=col, mec="white", mew=0.4, ls="", zorder=3)
        ax.plot(i + 0.17 + jit, dca, mk, ms=3.6, mfc="none", mec=col, mew=0.7, ls="", zorder=3)
        ax.plot([i - 0.33, i - 0.01], [d.mean()] * 2, color=col, lw=1.2, zorder=4)
        ax.plot([i + 0.01, i + 0.33], [dca.mean()] * 2, color=col, lw=1.2, ls=(0, (2, 1.2)), zorder=4)
    ax.axhline(0, color=S.GREY, lw=0.6, zorder=1)
    ax.set_xticks(range(len(F.COHORTS)))
    ax.set_xticklabels(["%s\n$n=%d$, $\\tau_{\\mathrm{blk}}=%d$" % (c, len(allD[c][0]), tau[c])
                        for c in F.COHORTS])
    ax.set_ylabel(r"$\Delta$ balanced accuracy")
    ax.set_xlim(-0.6, len(F.COHORTS) - 0.4)
    ax.legend([Line2D([], [], ls="", marker="o", ms=3.6, color=S.OURS, mec="white"),
               Line2D([], [], ls="", marker="o", ms=3.6, mfc="none", mec=S.OURS)],
              ["reported update order", "causal update order"],
              loc="lower left", fontsize=7, frameon=False, handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    S.save(fig, "fig_s1_delta", png_dpi=300)


FIGURES = {f.__name__: f for f in (
    fig_activations, fig_topography, fig_cone, fig_refmaps, fig_rate_collapse,
    fig_r_sweep, fig_r_cohorts, fig_r_deploy, fig_s1_delta, fig_topography_group, fig_s_blocks)}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    recompute = "--recompute" in sys.argv
    unknown = [a for a in args if a not in FIGURES]
    if unknown:
        sys.exit("unknown figure(s): %s\navailable: %s" % (", ".join(unknown), ", ".join(FIGURES)))
    for name in args or FIGURES:
        FIGURES[name](recompute)
