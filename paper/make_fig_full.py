"""Full CALM-Net figure + LaTeX results table (LSC: calibration + adaptive conformal)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results"
R = json.loads((RES / "calmnet_full.json").read_text())
subs = sorted(R)
S = {s: R[s]["summary"] for s in subs}
lab = [s.replace("sub-", "S") for s in subs]
x = np.arange(len(subs))

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))

# Panel A: conformal coverage static vs adaptive vs target
w = 0.35
ax[0].bar(x - w / 2, [S[s]["conformal_cov_static"] for s in subs], w,
          label="static conformal", color="#9ca3af")
ax[0].bar(x + w / 2, [S[s]["conformal_cov_adaptive"] for s in subs], w,
          label="adaptive conformal (LSC)", color="#7c3aed")
ax[0].axhline(0.90, color="#cc0000", lw=1.2, ls="--", label="target 0.90")
ax[0].set_xticks(x); ax[0].set_xticklabels(lab)
ax[0].set(ylabel="prediction-set coverage", ylim=(0.7, 1.0),
          title="Adaptive conformal holds coverage under drift\n(static conformal misses the target)")
ax[0].legend(fontsize=8, loc="lower right")

# Panel B: ECE raw vs temperature-scaled
ax[1].bar(x - w / 2, [S[s]["ece_raw"] for s in subs], w, label="ECE raw", color="#f4a3a3")
ax[1].bar(x + w / 2, [S[s]["ece_cal"] for s in subs], w, label="ECE + temperature", color="#dc2626")
ax[1].set_xticks(x); ax[1].set_xticklabels(lab)
ax[1].set(ylabel="Expected Calibration Error",
          title="Temperature scaling improves calibration\nacross the session gap")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "fig_full.pdf")
fig.savefig(Path(__file__).resolve().parent / "fig_full.png", dpi=150)
print("wrote fig_full.pdf/.png")

# LaTeX table rows
print("\n% ---- table rows ----")
for s in subs:
    d = S[s]
    print(f"{s.replace('sub-','S')} & {d['bal_acc']:.3f} & {d['ece_raw']:.3f} & {d['ece_cal']:.3f} & "
          f"{d['conf_auroc']:.3f} & {d['exec_acc@80']:.3f} & {d['conformal_cov_static']:.3f} & "
          f"{d['conformal_cov_adaptive']:.3f} \\\\")
m = lambda k: np.nanmean([S[s][k] for s in subs])
print("\\midrule")
print(f"\\textbf{{Mean}} & {m('bal_acc'):.3f} & {m('ece_raw'):.3f} & {m('ece_cal'):.3f} & "
      f"{m('conf_auroc'):.3f} & {m('exec_acc@80'):.3f} & {m('conformal_cov_static'):.3f} & "
      f"{m('conformal_cov_adaptive'):.3f} \\\\")
