"""MID validation figure: invariance gained + honest movement-invariant neural decode."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.loads((Path(__file__).resolve().parent.parent / "results" / "mid_validation.json").read_text())
subs = sorted(R)
imu = [R[s]["imu_only"] for s in subs]
no_acc = [R[s]["noMID"]["bal_acc"] for s in subs]
mid_acc = [R[s]["MID"]["bal_acc"] for s in subs]
no_r2 = [R[s]["noMID"]["intent_imu_r2_mlp"] for s in subs]
mid_r2 = [R[s]["MID"]["intent_imu_r2_mlp"] for s in subs]
lab = [s.replace("sub-", "S") for s in subs]
x = np.arange(len(subs))

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

w = 0.26
ax[0].bar(x - w, imu, w, label="IMU-only (movement)", color="#6b7280")
ax[0].bar(x, no_acc, w, label="EEG, no MID (movement-inflated)", color="#9ca3af")
ax[0].bar(x + w, mid_acc, w, label="EEG + MID (movement-invariant)", color="#7c3aed")
ax[0].axhline(0.5, color="k", lw=.8, ls=":")
ax[0].set_xticks(x); ax[0].set_xticklabels(lab)
ax[0].set(ylabel="balanced accuracy", ylim=(0.4, 1.0),
          title="MID isolates the honest neural decode\n(drop = movement contribution removed)")
ax[0].legend(fontsize=8, loc="upper right")

ax[1].bar(x - w / 2, no_r2, w, label="no MID", color="#9ca3af")
ax[1].bar(x + w / 2, mid_r2, w, label="+ MID", color="#7c3aed")
ax[1].axhline(0, color="k", lw=.8)
ax[1].set_xticks(x); ax[1].set_xticklabels(lab)
ax[1].set(ylabel="intent → IMU  R² (nonlinear probe)",
          title="MID makes the intent code movement-invariant\n(R² → 0 = motion no longer recoverable)")
ax[1].legend(fontsize=8)
fig.tight_layout()
out = Path(__file__).resolve().parent
fig.savefig(out / "fig_mid.pdf"); fig.savefig(out / "fig_mid.png", dpi=150)
print("wrote fig_mid.pdf/.png")
print(f"mean honest neural decode (MID): {np.mean(mid_acc):.3f}  |  movement-inflated: {np.mean(no_acc):.3f}")
print(f"mean intent->IMU R2: {np.mean(no_r2):.3f} -> {np.mean(mid_r2):.3f}")
