"""The figure the paper's central claim has been missing.

Every other figure shows an outcome (accuracy). This one shows the mechanism:
the running covariance estimate walking onto whichever class is currently
streaming.

Why this is exact rather than a reconstruction
----------------------------------------------
AdaptiveAlignment is the first layer, so its input is the raw window, and its
update is

    c = self._cov(x.detach().float())
    run_cov = (1 - m) * run_cov + m * c

with no gradient and no label. M therefore depends only on the input data and
the momentum, NOT on any trained weight, so replaying the EMA reproduces the
layer's internal state without training anything, and shows the failure is
structural rather than a training artefact.

The replay must use AdaptiveAlignment._cov, which trace-normalises each window
before averaging. An earlier version used exp_drift.cov (no per-window
normalisation) and diverged from the layer by 7 %. Verified against a live
layer: max abs difference 2.4e-07 over 12 batches.

Panels
------
(a) Trajectory. Session covariances and the two class covariances projected to
    2D by MDS on the affine-invariant Riemannian distance. M's path is drawn
    through that space in recording order, coloured by the streaming class.
(b) Distance from M to each class covariance against window index, with the
    background shaded by the streaming class. If the estimate tracks the class,
    the two lines swap whenever the streaming class changes.
(c) The same distances at a slow rate, where the memory exceeds a class block.

Writes results/fig_class_tracking.pdf (+ .png). CPU only.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S                                    # noqa: E402
from figstyle import FULL, OURS, VERM, GREEN, INK, GREY  # noqa: E402
from exp_drift import cov, riemannian                   # noqa: E402

S.use()
BATCH = 32


def _layer_cov(probe, chunk):
    """The covariance the layer itself forms.

    AdaptiveAlignment._cov trace-normalises EACH window before averaging;
    exp_drift.cov does not. Using the wrong one changes the running estimate by
    about 7 % and the replay no longer matches the layer, so this calls the
    layer's own method. Verified against a live layer: max abs difference
    2.4e-07 over 12 batches, i.e. float32 rounding.
    """
    import torch
    with torch.no_grad():
        return probe._cov(torch.as_tensor(chunk).float()).numpy()


def ema_trajectory(X, momentum, batch=BATCH):
    """Replay the layer's running covariance, one update per batch."""
    from driftnet import AdaptiveAlignment
    probe = AdaptiveAlignment(X.shape[1], momentum=momentum).eval()
    M, out = None, []
    for i in range(0, len(X) - batch + 1, batch):
        c = _layer_cov(probe, X[i:i + batch])
        M = c.copy() if M is None else (1 - momentum) * M + momentum * c
        out.append((i + batch, M.copy()))
    return out


def class_covs(X, y):
    """Class covariances in the SAME estimator as the running one, so the
    distances in panels b, d, f are commensurable with the trajectory."""
    from driftnet import AdaptiveAlignment
    probe = AdaptiveAlignment(X.shape[1], momentum=0.2).eval()
    return _layer_cov(probe, X[y == 0]), _layer_cov(probe, X[y == 1])


def streaming_class(y, batch=BATCH):
    """Majority class of each batch, in recording order."""
    return np.array([np.bincount(y[i:i + batch], minlength=2).argmax()
                     for i in range(0, len(y) - batch + 1, batch)])


def panel_distances(ax, traj, Cs, Cw, strm, title, show_ylab=True,
                    **kwargs):
    idx = [t[0] for t in traj]
    ds = [riemannian(M, Cs) for _, M in traj]
    dw = [riemannian(M, Cw) for _, M in traj]
    # Shade at WINDOW resolution rather than batch majority. Cohort A's class
    # blocks (18 windows) are shorter than the 32-window batch, so a
    # batch-majority shading erases the alternation, which is precisely the
    # contrast this panel exists to show.
    yw = kwargs.get("y_windows")
    if yw is not None:
        start = 0
        for k in range(1, len(yw) + 1):
            if k == len(yw) or yw[k] != yw[start]:
                ax.axvspan(start, k - 1,
                           color=(GREEN if yw[start] == 0 else VERM),
                           alpha=0.13, lw=0, zorder=0)
                start = k
    ax.plot(idx, ds, color=GREEN, lw=1.3, label=r"to $\Sigma_{\mathrm{stop}}$")
    ax.plot(idx, dw, color=VERM, lw=1.3, ls=(0, (4, 2)),
            label=r"to $\Sigma_{\mathrm{walk}}$")
    ax.set_xlabel("window index (recording order)")
    if show_ylab:
        ax.set_ylabel(r"Riemannian distance from $\mathbf{M}$")
    S.note(ax, 0.02, 0.04, title, transform=ax.transAxes, fontsize=6.6)
    cross = int(np.sum(np.diff(np.sign(np.array(ds) - np.array(dw))) != 0))
    S.note(ax, 0.98, 0.93, "%d crossings" % cross, transform=ax.transAxes,
           ha="right", fontsize=7,
           color=(VERM if cross else GREEN))
    return cross


def build(sub_a="sub-01", sub_b=None):
    from dataio import build_epochs, list_sessions
    fig = plt.figure(figsize=(FULL, 4.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.55,
                          wspace=0.38)

    # ---------------- cohort A, fast rate
    es = build_epochs(subject=sub_a, win=4.0, step=0.5, zscore=False)
    X = es.X / (es.X.std() + 1e-30)
    g = es.session.astype(int)
    pres = sorted(set(int(v) for v in np.unique(g)))
    sess = [s for s in list_sessions(sub_a) if s in pres]
    m0 = np.isin(g, sess[:3])
    Xa, ya = X[m0], es.y[m0].astype(int)
    Cs, Cw = class_covs(Xa, ya)
    strm = streaming_class(ya)
    traj = ema_trajectory(Xa, 0.2)

    # (a) trajectory in MDS space
    ax = fig.add_subplot(gs[0, 0])
    step = max(1, len(traj) // 90)
    pts = [Cs, Cw] + [M for _, M in traj[::step]]
    n = len(pts)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = riemannian(pts[i], pts[j])
    Y = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
            normalized_stress=False).fit_transform(D)
    sc = strm[::step][:len(Y) - 2]
    ax.plot(Y[2:, 0], Y[2:, 1], color=GREY, lw=0.6, alpha=0.7, zorder=1)
    ax.scatter(Y[2:, 0][sc == 0], Y[2:, 1][sc == 0], s=9, color=GREEN,
               zorder=3, label="Stop streaming")
    ax.scatter(Y[2:, 0][sc == 1], Y[2:, 1][sc == 1], s=9, color=VERM,
               zorder=3, label="Walk streaming")
    ax.scatter(Y[0, 0], Y[0, 1], marker="X", s=90, color=GREEN,
               edgecolor="white", linewidth=1.2, zorder=5)
    ax.scatter(Y[1, 0], Y[1, 1], marker="X", s=90, color=VERM,
               edgecolor="white", linewidth=1.2, zorder=5)
    S.note(ax, Y[0, 0], Y[0, 1], r"  $\Sigma_{\mathrm{stop}}$", color=GREEN,
           fontsize=6.6)
    S.note(ax, Y[1, 0], Y[1, 1], r"  $\Sigma_{\mathrm{walk}}$", color=VERM,
           fontsize=6.6)
    ax.set_xticks([]), ax.set_yticks([])
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_visible(False)
    S.panel(ax, "a", x=-0.04)
    S.note(ax, 0.5, -0.09, "cohort A, $m=0.2$: the estimate never approaches "
           "either class", transform=ax.transAxes, ha="center", fontsize=6.4)

    # (b) distances, cohort A
    axb = fig.add_subplot(gs[1, 0])
    cA = panel_distances(axb, traj, Cs, Cw, strm,
                         r"cohort A, $\tau_{\mathrm{blk}}=18$, $m=0.2$",
                         y_windows=ya)
    S.panel(axb, "b", x=-0.30)

    # ---------------- cohort B, fast then slow
    from dataio_mobi import build_subject, subjects
    sb = sub_b or list(subjects())[0]
    eb = build_subject(sb, win=2.0, step=0.5, zscore=False)
    Xb = eb.X / (eb.X.std() + 1e-30)
    yb = eb.y.astype(int)
    gb = (eb.session if hasattr(eb, "session") else eb.trial).astype(int)
    k0 = sorted(set(int(v) for v in np.unique(gb)))[0]
    Xb, yb = Xb[gb == k0], yb[gb == k0]
    Csb, Cwb = class_covs(Xb, yb)
    strmb = streaming_class(yb)

    for col, mom, lab in ((1, 0.2, r"cohort B, $\tau_{\mathrm{blk}}=309$, "
                                   r"$m=0.2$ (memory 160)"),
                          (2, 0.01, r"cohort B, $m=0.01$ (memory 3200)")):
        tb = ema_trajectory(Xb, mom)
        axt = fig.add_subplot(gs[0, col])
        stepb = max(1, len(tb) // 90)
        ptsb = [Csb, Cwb] + [M for _, M in tb[::stepb]]
        nb = len(ptsb)
        Db = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(i + 1, nb):
                Db[i, j] = Db[j, i] = riemannian(ptsb[i], ptsb[j])
        Yb = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                 normalized_stress=False).fit_transform(Db)
        scb = strmb[::stepb][:len(Yb) - 2]
        axt.plot(Yb[2:, 0], Yb[2:, 1], color=GREY, lw=0.6, alpha=0.7, zorder=1)
        axt.scatter(Yb[2:, 0][scb == 0], Yb[2:, 1][scb == 0], s=9, color=GREEN,
                    zorder=3)
        axt.scatter(Yb[2:, 0][scb == 1], Yb[2:, 1][scb == 1], s=9, color=VERM,
                    zorder=3)
        axt.scatter(Yb[0, 0], Yb[0, 1], marker="X", s=90, color=GREEN,
                    edgecolor="white", linewidth=1.2, zorder=5)
        axt.scatter(Yb[1, 0], Yb[1, 1], marker="X", s=90, color=VERM,
                    edgecolor="white", linewidth=1.2, zorder=5)
        axt.set_xticks([]), axt.set_yticks([])
        for s_ in ("left", "bottom"):
            axt.spines[s_].set_visible(False)
        S.panel(axt, "c" if col == 1 else "e", x=-0.04)
        S.note(axt, 0.5, -0.09,
               "the estimate walks onto the streaming class" if col == 1
               else "slowed: the walk disappears",
               transform=axt.transAxes, ha="center", fontsize=6.4)

        axd = fig.add_subplot(gs[1, col])
        c = panel_distances(axd, tb, Csb, Cwb, strmb, lab, show_ylab=False,
                            y_windows=yb)
        S.panel(axd, "d" if col == 1 else "f", x=-0.18)
        print("  cohort B m=%.2f: %d crossings of the two distance lines"
              % (mom, c), flush=True)

    h, l = axb.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03),
               fontsize=7, handlelength=1.6)
    print("  cohort A m=0.20: %d crossings" % cA, flush=True)
    S.save(fig, "fig_class_tracking")


if __name__ == "__main__":
    build()
