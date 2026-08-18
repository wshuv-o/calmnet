"""Cross-subject figures from results/multi_subject.json (incl. CALMNet)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3})
COL = {"imu": "#6b7280", "eegnet": "#2563eb", "conformer": "#dc2626", "calmnet": "#7c3aed"}


def _m(sessions, f):
    return float(np.mean([x[f] for x in sessions.values()]))


def main():
    R = json.loads((RESULTS / "multi_subject.json").read_text())
    subs = sorted(R)
    has_calm = all("calmnet_full" in R[s] for s in subs)

    acc = {
        "imu": [float(np.mean(list(R[s]["imu_only"].values()))) for s in subs],
        "eegnet": [_m(R[s]["eegnet_full"]["sessions"], "bal_acc") for s in subs],
        "conformer": [_m(R[s]["conformer_full"]["sessions"], "bal_acc") for s in subs],
    }
    auroc = {
        "eegnet": [_m(R[s]["eegnet_full"]["sessions"], "conf_auroc") for s in subs],
        "conformer": [_m(R[s]["conformer_full"]["sessions"], "conf_auroc") for s in subs],
    }
    if has_calm:
        acc["calmnet"] = [_m(R[s]["calmnet_full"]["sessions"], "bal_acc") for s in subs]
        auroc["calmnet"] = [_m(R[s]["calmnet_full"]["sessions"], "conf_auroc") for s in subs]

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.7))
    x = np.arange(len(subs))
    order_a = ["imu", "eegnet", "conformer"] + (["calmnet"] if has_calm else [])
    lab_a = {"imu": "IMU motion only", "eegnet": "EEGNet", "conformer": "Conformer",
             "calmnet": "CALMNet (ours)"}
    w = 0.8 / len(order_a)
    for i, k in enumerate(order_a):
        ax[0].bar(x + (i - (len(order_a) - 1) / 2) * w, acc[k], w, label=lab_a[k], color=COL[k])
    ax[0].axhline(0.5, color="k", lw=.7, ls=":")
    ax[0].set_xticks(x); ax[0].set_xticklabels([s.replace("sub-", "S") for s in subs])
    ax[0].set(ylabel="balanced accuracy", ylim=(0.4, 1.0),
              title="Per-subject decoding vs the movement (IMU) baseline")
    ax[0].legend(fontsize=8, ncol=2)

    order_b = ["eegnet", "conformer"] + (["calmnet"] if has_calm else [])
    w = 0.8 / len(order_b)
    for i, k in enumerate(order_b):
        ax[1].bar(x + (i - (len(order_b) - 1) / 2) * w, auroc[k], w, label=lab_a[k], color=COL[k])
    ax[1].axhline(0.5, color="k", lw=.7, ls=":")
    ax[1].set_xticks(x); ax[1].set_xticklabels([s.replace("sub-", "S") for s in subs])
    ax[1].set(ylabel="confidence-vs-correctness AUROC", ylim=(0.4, 1.0),
              title="Usable uncertainty for abstention (higher = better)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_multi_subject.png"); plt.close(fig)
    print("wrote fig_multi_subject.png")

    def mean(d, k): return np.mean(d[k])
    line = "MEAN acc  " + "  ".join(f"{k} {mean(acc,k):.3f}" for k in order_a)
    print(line)
    print("MEAN AUROC  " + "  ".join(f"{k} {mean(auroc,k):.3f}" for k in order_b))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
