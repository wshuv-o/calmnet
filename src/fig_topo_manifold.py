"""Two figures that show what the alignment layer does, not how well it scores.

fig_topography.pdf
    Scalp maps of the whitening transform. Channel power before alignment, the
    per-channel gain M^-1/2 applies, and the gain split by class. This is the
    direct answer to the motion-artefact objection (Castermans 2014, Kline
    2015): if the layer is reweighting sensorimotor cortex the ERD account
    holds, and if it is dominated by peripheral, EMG-prone sites that is worth
    knowing before a reviewer finds it.

fig_manifold.pdf
    Each session's covariance as a point, positioned by MDS on the
    affine-invariant Riemannian distance, joined in recording order. Raw, the
    momentum-0 control, and aligned. Makes the 45.9 % reduction visible, and
    shows the control sitting exactly on the raw points as affine invariance
    requires.

Both use AdaptiveAlignment's own covariance estimator, so they are
commensurable with the layer's internal state (see fig_class_tracking.py: using
exp_drift.cov instead diverges by 7 %).

CPU only. No training.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S                                      # noqa: E402
from figstyle import FULL, COL, VERM, GREEN, GREY, INK    # noqa: E402
from exp_drift import riemannian, cov as plain_cov          # noqa: E402
from driftnet import AdaptiveAlignment                    # noqa: E402

S.use()


def layer_cov(probe, X):
    with torch.no_grad():
        return probe._cov(torch.as_tensor(X).float()).numpy()


def inv_sqrt(M):
    w, v = np.linalg.eigh(M.astype(np.float64))
    w = np.clip(w, 1e-6, None)
    return (v @ np.diag(w ** -0.5) @ v.T)


def make_info(ch_names, sfreq=100.0):
    import mne
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    mont = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(mont, on_missing="ignore")
    return info


# ===================================================================== topo
def fig_topography(sub="sub-01"):
    import mne
    from dataio import build_epochs, list_sessions
    es = build_epochs(subject=sub, win=4.0, step=0.5, zscore=False)
    X = (es.X / (es.X.std() + 1e-30)).astype(np.float32)
    y = es.y.astype(int)
    g = es.session.astype(int)
    pres = sorted(set(int(v) for v in np.unique(g)))
    sess = [s for s in list_sessions(sub) if s in pres]
    m0 = np.isin(g, sess[:3])
    X, y = X[m0], y[m0]

    probe = AdaptiveAlignment(X.shape[1], momentum=0.2).eval()
    M = layer_cov(probe, X)
    W = inv_sqrt(M)

    info = make_info(es.ch_names)
    picks = [i for i, c in enumerate(es.ch_names)
             if c in info.ch_names and info.get_montage() is not None]
    power = np.diag(M)                       # channel power before alignment
    gain = np.sqrt(np.sum(W ** 2, axis=1))   # per-channel gain of the whitener
    # ABSOLUTE 8-30 Hz band power, not the layer's covariance diagonal.
    # AdaptiveAlignment._cov trace-normalises each window, so its diagonal is
    # RELATIVE power: a decrease anywhere is forced by an increase elsewhere,
    # and a central decrease then says nothing on its own. Computed here by
    # band-pass filtering and taking variance per channel.
    from scipy.signal import butter, filtfilt
    b_, a_ = butter(4, [8.0 / 50.0, 30.0 / 50.0], btype="band")
    Xf = filtfilt(b_, a_, X.astype(np.float64), axis=-1)
    pw_stop = Xf[y == 0].var(axis=-1).mean(axis=0)
    pw_walk = Xf[y == 1].var(axis=-1).mean(axis=0)
    diff = 10.0 * np.log10((pw_walk + 1e-30) / (pw_stop + 1e-30))

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.5))
    for ax, vals, title, cmap in (
            (axes[0], power, "channel power before alignment", "viridis"),
            (axes[1], gain, r"gain applied by $\mathbf{M}^{-1/2}$", "magma"),
            (axes[2], diff, "8-30 Hz power, Walk vs Stop (dB)", "RdBu_r")):
        v = np.asarray(vals, float)
        lim = np.max(np.abs(v)) if cmap == "RdBu_r" else None
        im, _ = mne.viz.plot_topomap(
            v, info, axes=ax, show=False, cmap=cmap, contours=4,
            vlim=(-lim, lim) if lim else (None, None), sensors=True)
        ax.set_title(title, fontsize=7.5, pad=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cb.ax.tick_params(labelsize=6)
    for ax_, L_ in zip(axes, "abc"):
        S.panel(ax_, L_, x=-0.22, y=1.16)
    S.save(fig, "fig_topography")

    order = np.argsort(-gain)
    print("  highest whitening gain: " +
          ", ".join("%s %.2f" % (es.ch_names[i], gain[i]) for i in order[:6]),
          flush=True)
    print("  lowest  whitening gain: " +
          ", ".join("%s %.2f" % (es.ch_names[i], gain[i]) for i in order[-6:]),
          flush=True)
    od = np.argsort(diff)
    print("  8-30 Hz, largest DECREASE during walk: " +
          ", ".join("%s %+.2f dB" % (es.ch_names[i], diff[i]) for i in od[:6]),
          flush=True)
    print("  8-30 Hz, largest INCREASE during walk: " +
          ", ".join("%s %+.2f dB" % (es.ch_names[i], diff[i]) for i in od[-6:]),
          flush=True)

    # Persist, so the text can quote these instead of describing the maps.
    import json
    stats = {
        "_note": ("Single participant, cohort A, first three sessions. Panel "
                  "(c) is ABSOLUTE 8-30 Hz band power (4th-order Butterworth, "
                  "per-channel variance), reported as 10*log10(walk/stop). It "
                  "is NOT the layer's covariance diagonal, which is "
                  "trace-normalised per window and therefore relative: under "
                  "that normalisation a decrease anywhere is forced by an "
                  "increase elsewhere. The sensorimotor decrease and the "
                  "posterior increase are spatially separable, which is all "
                  "this shows; posterior sites are also where neck EMG "
                  "appears, and optic flow during walking accounts for an "
                  "occipital increase equally well."),
        "subject": sub,
        "sessions": [int(x) for x in sess[:3]],
        "n_windows": int(len(X)),
        "band_hz": [8.0, 30.0],
        "walk_minus_stop_db": {es.ch_names[i]: round(float(diff[i]), 3)
                               for i in range(len(diff))},
        "whitening_gain": {es.ch_names[i]: round(float(gain[i]), 3)
                           for i in range(len(gain))},
        "largest_decrease": [[es.ch_names[i], round(float(diff[i]), 2)]
                             for i in od[:8]],
        "largest_increase": [[es.ch_names[i], round(float(diff[i]), 2)]
                             for i in od[-8:][::-1]],
        "highest_gain": [[es.ch_names[i], round(float(gain[i]), 2)]
                         for i in np.argsort(-gain)[:8]],
        "lowest_gain": [[es.ch_names[i], round(float(gain[i]), 2)]
                        for i in np.argsort(-gain)[-8:][::-1]],
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "topography_stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=1)
    print("  wrote results/topography_stats.json", flush=True)


# ================================================================= manifold
def fig_manifold(sub="sub-01"):
    from dataio import build_epochs, list_sessions
    es = build_epochs(subject=sub, win=4.0, step=0.5, zscore=False)
    X = (es.X / (es.X.std() + 1e-30)).astype(np.float32)
    g = es.session.astype(int)
    pres = sorted(set(int(v) for v in np.unique(g)))
    sess = [s for s in list_sessions(sub) if s in pres]

    def covs_for(momentum):
        """Session covariances after running the layer at this momentum."""
        probe = AdaptiveAlignment(X.shape[1], momentum=momentum).eval()
        out = []
        with torch.no_grad():
            for s in sess:
                Xi = X[g == s]
                o = []
                for i in range(0, len(Xi), 32):
                    o.append(probe(torch.as_tensor(Xi[i:i + 32])).numpy())
                out.append(plain_cov(np.concatenate(o)))
        return out

    # Measure with the un-normalised covariance used in drift_fixed.json, not
    # the layer's per-window-normalised one. The latter breaks affine
    # invariance and leaves the momentum-0 control 3 % off instead of null.
    raw = [plain_cov(X[g == s]) for s in sess]
    noop = covs_for(0.0)
    ali = covs_for(0.2)

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.4))
    coords = {}
    for ax, C, name, col in ((axes[0], raw, "raw", GREY),
                             (axes[1], noop, "no-op control ($m=0$)", VERM),
                             (axes[2], ali, "aligned ($m=0.2$)", GREEN)):
        n = len(C)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = riemannian(C[i], C[j])
        Y = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                normalized_stress=False).fit_transform(D)
        ax.plot(Y[:, 0], Y[:, 1], color=GREY, lw=0.6, alpha=0.6, zorder=1)
        ax.scatter(Y[:, 0], Y[:, 1], s=55, color=col, edgecolor="white",
                   linewidth=1.0, zorder=3)
        for k, (x_, y_) in enumerate(Y):
            ax.text(x_, y_, str(k + 1), fontsize=5.6, ha="center",
                    va="center", color="white", zorder=4, fontweight="bold")
        sd = float(np.mean(D[np.triu_indices(n, 1)]))
        ax.set_title("%s\nmean pairwise $\\delta$ = %.2f" % (name, sd),
                     fontsize=7.5, pad=5)
        coords[name] = Y
        ax.set_xticks([]), ax.set_yticks([])
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_visible(False)
        print("  %-22s mean pairwise distance %.3f" % (name, sd), flush=True)
    # MDS coordinates carry the units of the input distances, so a shared
    # limit makes the contraction visible. Auto-scaling each panel hides it:
    # all three looked equally scattered while delta fell from 8.0 to 3.9.
    lim = max(np.abs(Y_).max() for Y_ in coords.values()) * 1.15
    for ax in axes:
        ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
    for ax, L in zip(axes, "abc"):
        S.panel(ax, L, x=-0.04, y=1.16)
    S.save(fig, "fig_manifold")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("topo", "both"):
        print("TOPOGRAPHY", flush=True)
        fig_topography()
    if which in ("manifold", "both"):
        print("MANIFOLD", flush=True)
        fig_manifold()
