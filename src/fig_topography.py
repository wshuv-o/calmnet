"""Where on the scalp the branch's reference acts, and what the classes differ by.

Regenerated from the reported model's own estimator. The branch computes
log(M^-1/2 C M^-1/2), so M^-1/2 is applied inside the tangent map and its
per-channel gain is a property of the model we report, not of any earlier one.
The reference here is built by TangentBranch itself, with its own shrinkage and
per-window trace normalisation, so the figure is commensurate with what the
layer holds at inference.

  (a) channel power the reference sees
  (b) per-channel gain of M^-1/2, i.e. which sites the tangent map amplifies
  (c) absolute 8-30 Hz power, Walk minus Stop

Panel (c) is the only evidence in the paper bearing on the artefact objection,
since the ICA control could not be run for this configuration. It is not an
artefact rejection: the honest reading is that a central decrease is the
direction expected of mu and beta desynchronisation, while gross movement
artefact would raise power during walking rather than lower it.

Writes results/fig_topography.pdf and .png, and results/topography_stats.json.
"""
from __future__ import annotations

import io
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.signal import butter, filtfilt

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
import figstyle as S                                        # noqa: E402
from figstyle import FULL                                   # noqa: E402
from driftnet import TangentBranch                          # noqa: E402

S.use()
SUB = os.environ.get("TOPO_SUB", "sub-01")
RES = os.path.join(HERE, "..", "results")


def main():
    import mne
    from dataio import build_epochs, list_sessions

    es = build_epochs(subject=SUB, win=4.0, step=0.5, zscore=False)
    X = (es.X / (es.X.std() + 1e-30)).astype(np.float32)
    y = es.y.astype(int)
    g = es.session.astype(int)
    pres = sorted(set(int(v) for v in np.unique(g)))
    sess = [s for s in list_sessions(SUB) if s in pres][:3]
    m = np.isin(g, sess)
    X, y = X[m], y[m]

    # the reference exactly as the branch builds it
    br = TangentBranch(X.shape[1], 128).eval()
    with torch.no_grad():
        br.freeze_reference(torch.as_tensor(X))
        M = br.run_cov.double().numpy()
    w, v = np.linalg.eigh(M)
    W = v @ np.diag(np.clip(w, 1e-6, None) ** -0.5) @ v.T

    power = np.diag(M)
    gain = np.sqrt(np.sum(W ** 2, axis=1))

    # absolute band power, not the reference diagonal: that is trace-normalised
    # per window and therefore relative, so a decrease anywhere would be forced
    # by an increase elsewhere and a central decrease would mean nothing.
    b_, a_ = butter(4, [8.0 / 50.0, 30.0 / 50.0], btype="band")
    Xf = filtfilt(b_, a_, X.astype(np.float64), axis=-1)
    pw_stop = Xf[y == 0].var(axis=-1).mean(axis=0)
    pw_walk = Xf[y == 1].var(axis=-1).mean(axis=0)
    diff = 10.0 * np.log10((pw_walk + 1e-30) / (pw_stop + 1e-30))

    info = mne.create_info(list(es.ch_names), 100.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     on_missing="ignore")

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.4))
    for ax, vals, ttl, cmap in (
            (axes[0], power, "channel power at the reference", "viridis"),
            (axes[1], gain, r"gain of $\mathbf{M}^{-1/2}$ in the tangent map",
             "magma"),
            (axes[2], diff, "8–30 Hz power, Walk $-$ Stop (dB)", "RdBu_r")):
        vv = np.asarray(vals, float)
        lim = np.max(np.abs(vv)) if cmap == "RdBu_r" else None
        im, _ = mne.viz.plot_topomap(
            vv, info, axes=ax, show=False, cmap=cmap, contours=4,
            vlim=(-lim, lim) if lim else (None, None), sensors=True)
        ax.set_title(ttl, fontsize=7.2, pad=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cb.ax.tick_params(labelsize=5.8)
    for ax_, L_ in zip(axes, "abc"):
        S.panel(ax_, L_, x=-0.20, y=1.16)
    S.save(fig, "fig_topography")

    od = np.argsort(diff)
    og = np.argsort(-gain)
    stats = {
        "_note": ("Reference built by TangentBranch with its own shrinkage and "
                  "per-window trace normalisation, so it matches what the "
                  "reported model holds. Panel (c) is ABSOLUTE 8-30 Hz band "
                  "power (4th-order Butterworth, per-channel variance) as "
                  "10*log10(walk/stop), NOT the reference diagonal, which is "
                  "relative. One participant, three sessions. The sign of a "
                  "power contrast is not an artefact rejection."),
        "subject": SUB, "sessions": [int(x) for x in sess],
        "n_windows": int(len(X)),
        "highest_gain": [[es.ch_names[i], round(float(gain[i]), 2)]
                         for i in og[:6]],
        "lowest_gain": [[es.ch_names[i], round(float(gain[i]), 2)]
                        for i in og[-6:][::-1]],
        "largest_decrease": [[es.ch_names[i], round(float(diff[i]), 2)]
                             for i in od[:6]],
        "largest_increase": [[es.ch_names[i], round(float(diff[i]), 2)]
                             for i in od[-6:][::-1]],
    }
    io.open(os.path.join(RES, "topography_stats.json"), "w",
            encoding="utf-8").write(json.dumps(stats, indent=1))
    print("  highest gain : %s" % ", ".join(
        "%s %.1f" % (es.ch_names[i], gain[i]) for i in og[:5]))
    print("  lowest  gain : %s" % ", ".join(
        "%s %.1f" % (es.ch_names[i], gain[i]) for i in og[-5:]))
    print("  largest decrease in walk: %s" % ", ".join(
        "%s %+.1f dB" % (es.ch_names[i], diff[i]) for i in od[:5]))
    print("  largest increase in walk: %s" % ", ".join(
        "%s %+.1f dB" % (es.ch_names[i], diff[i]) for i in od[-5:][::-1]))


if __name__ == "__main__":
    main()
