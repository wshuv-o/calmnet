"""Generate CALM-Net result figures from results/longitudinal.json (+ a fresh
model for the reliability / risk-coverage panels)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataio import build_epochs, session_days
from splits import grouped_split
from train import train_model, predict
from calibrate import (fit_temperature, softmax_np, reliability_curve,
                       expected_calibration_error)
from abstain import risk_coverage_curve

RESULTS = Path(__file__).resolve().parent.parent / "results"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3})
C = {"eegnet": "#2563eb", "conformer": "#dc2626", "imu": "#6b7280",
     "motor": "#059669", "frontal": "#d97706"}


def fig_longitudinal(R):
    days = {int(k): v for k, v in R["session_days"].items()}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, lab, col in [("eegnet_full", "EEGNet (60ch)", C["eegnet"]),
                          ("conformer_full", "Conformer (60ch)", C["conformer"])]:
        s = R[key]["sessions"]
        xs = [s[k]["day"] for k in sorted(s, key=int)]
        ax[0].plot(xs, [s[k]["bal_acc"] for k in sorted(s, key=int)], "o-", color=col, label=lab)
        ax[1].plot(xs, [s[k]["ece_raw"] for k in sorted(s, key=int)], "o--", color=col, alpha=.6,
                   label=f"{lab} raw")
        ax[1].plot(xs, [s[k]["ece_cal"] for k in sorted(s, key=int)], "s-", color=col,
                   label=f"{lab} +temp")
    # IMU-only baseline
    iu = R["imu_only"]
    xs = [days[int(k)] for k in sorted(iu, key=int)]
    ax[0].plot(xs, [iu[k] for k in sorted(iu, key=int)], "^:", color=C["imu"],
               label="IMU-motion only")
    ax[0].set(xlabel="days since session 1", ylabel="balanced accuracy",
              title="Longitudinal decoding (train ses 1-3)"); ax[0].legend(fontsize=8)
    ax[0].axhline(0.5, color="k", lw=.7, ls=":")
    ax[1].set(xlabel="days since session 1", ylabel="Expected Calibration Error",
              title="Calibration vs session gap"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_longitudinal.png"); plt.close(fig)


def fig_confound(R):
    keys = [("imu_only", "IMU motion\nonly", C["imu"], lambda v: v),
            ("eegnet_frontal", "EEG frontal\nonly", C["frontal"], lambda v: v["bal_acc"]),
            ("eegnet_motor", "EEG motor\nonly", C["motor"], lambda v: v["bal_acc"]),
            ("eegnet_full", "EEG all 60ch", C["eegnet"], lambda v: v["bal_acc"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (key, lab, col, f) in enumerate(keys):
        if key == "imu_only":
            vals = [f(v) for v in R[key].values()]
        else:
            vals = [f(v) for v in R[key]["sessions"].values()]
        ax.bar(i, np.mean(vals), color=col, yerr=np.std(vals), capsize=4, alpha=.85)
        ax.text(i, np.mean(vals) + .01, f"{np.mean(vals):.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k[1] for k in keys])
    ax.axhline(0.5, color="k", lw=.7, ls=":")
    ax.set(ylabel="balanced accuracy (mean over test sessions)",
           title="Movement confound: what carries the Walk/Stop signal?", ylim=(0.4, 1.0))
    fig.tight_layout(); fig.savefig(RESULTS / "fig_confound.png"); plt.close(fig)


def fig_reliability_and_rc(es):
    """Train the Conformer (the model with usable confidence) on ses1-3,
    evaluate calibration + abstention on the far session (ses9)."""
    tr = es.by_sessions([1, 2, 3]); te = es.by_sessions([9])
    ti, vi = grouped_split(tr.segment, tr.y, frac=0.2, seed=0)
    model = train_model("conformer", tr.X[ti], tr.y[ti], tr.X[vi], tr.y[vi], epochs=80, seed=0)
    vlog, _, _ = predict(model, tr.X[vi]); T = fit_temperature(vlog, tr.y[vi])
    logit, _, _ = predict(model, te.X)
    p_raw, p_cal = softmax_np(logit, 1.0), softmax_np(logit, T)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for p, lab, col in [(p_raw, f"raw (ECE {expected_calibration_error(p_raw, te.y):.3f})", C["conformer"]),
                        (p_cal, f"+temp T={T:.2f} (ECE {expected_calibration_error(p_cal, te.y):.3f})", C["eegnet"])]:
        bc, ba, cnt = reliability_curve(p, te.y, n_bins=10)
        good = ~np.isnan(ba)
        ax[0].plot(bc[good], ba[good], "o-", color=col, label=lab)
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="confidence", ylabel="accuracy", title="Reliability (ses-9, 21 days later)")
    ax[0].legend(fontsize=8)

    conf = p_cal.max(1); correct = (p_cal.argmax(1) == te.y).astype(int)
    cov, risk = risk_coverage_curve(conf, correct)
    ax[1].plot(cov, risk, "-", color=C["eegnet"], lw=2)
    ax[1].axvline(0.8, color="k", ls=":", lw=.8)
    ax[1].set(xlabel="coverage (fraction executed)", ylabel="selective risk (error on executed)",
              title="Risk-coverage: abstention buys accuracy")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_reliability_riskcoverage.png"); plt.close(fig)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    R = json.loads((RESULTS / "longitudinal.json").read_text())
    es = build_epochs()
    fig_longitudinal(R); print("wrote fig_longitudinal.png")
    fig_confound(R); print("wrote fig_confound.png")
    fig_reliability_and_rc(es); print("wrote fig_reliability_riskcoverage.png")
