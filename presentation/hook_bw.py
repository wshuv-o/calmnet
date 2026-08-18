"""Grayscale confound chart (old-school, no colour)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"]})
fig, ax = plt.subplots(figsize=(6.3, 4.6), dpi=200)
labels = ["Chance", "Best EEG\ndecoder", "Motion sensor\n(no brain data)"]
vals = [0.50, 0.84, 0.92]
shades = ["0.80", "0.55", "0.25"]
bars = ax.bar(labels, vals, color=shades, edgecolor="black", linewidth=0.9, width=0.6)
ax.axhline(0.5, ls=":", color="0.4", lw=1.0)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("balanced accuracy")
ax.set_title("A motion sensor beats every EEG decoder", fontweight="bold")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
            ha="center", va="bottom", fontweight="bold", color="black")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig("hook_bw.png", dpi=200, facecolor="white")
print("saved hook_bw.png")
