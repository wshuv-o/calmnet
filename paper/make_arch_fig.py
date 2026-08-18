"""CALM-Net framework schematic (Fig. 1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 8.2, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.3, 4.3)); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 60)

C = {"eeg": "#2563eb", "kin": "#059669", "core": "#7c3aed", "cal": "#d97706",
     "safe": "#dc2626", "in": "#374151"}


def box(x, y, w, h, text, color, fc=None, fs=8.2, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                 linewidth=1.2, edgecolor=color, facecolor=fc or color, alpha=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, weight="bold", wrap=True)


def arrow(x1, y1, x2, y2, color="#374151", style="-|>", lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=11, linewidth=lw, color=color, linestyle=ls,
                 shrinkA=1, shrinkB=1))


# ---- inputs ----
box(1, 46, 17, 9, "EEG 60ch\n(8-30 Hz)", C["eeg"])
box(1, 33, 17, 9, "Head + Exo\nIMU", C["kin"])
box(1, 20, 17, 9, "EOG 4ch", C["kin"], fc="#10b981")
# ---- encoders ----
box(24, 44, 20, 12, "Multi-band encoder\n$\\mu$ / low-$\\beta$ / high-$\\beta$\n+ spatial covariance", C["eeg"])
box(24, 31, 20, 9, "Kinematic\nencoder", C["kin"])
# ---- XFCA ----
box(48, 44, 15, 12, "XFCA\ncross-freq\ncoupling attn", C["core"])
# ---- MID ----
box(66, 46, 17, 10, "MID: intent\nsubspace  (IMU-\ninvariant, GRL)", C["core"])
box(66, 31, 17, 10, "MID: artifact\nsubspace\n($\\to$ predict IMU)", C["core"], fc="#a78bfa")
# ---- heads ----
box(86, 47, 13, 9, "Selective\nclassifier", C["safe"])
box(86, 32, 13, 9, "Kinematic\ncontamination", C["kin"])
# ---- LSC bottom ----
box(30, 6, 40, 10, "Longitudinal self-calibration (LSC):\nper-session feature alignment +\nadaptive conformal risk control", C["cal"])
# ---- SAS output ----
box(84, 15, 15, 12, "SAS commit:\nWalk / Stop\nelse $\\to$ STOP", C["safe"], fc="#ef4444")

# arrows
arrow(18, 50.5, 24, 50); arrow(18, 37.5, 24, 35.5); arrow(18, 24.5, 23, 33)
arrow(44, 50, 48, 50); arrow(44, 35.5, 66, 36, color=C["kin"])
arrow(63, 50, 66, 51)
arrow(83, 51, 86, 51.5); arrow(83, 36, 86, 36.5)
arrow(75, 41, 79, 36, color=C["core"], ls="--")   # artifact informs contamination
arrow(92.5, 47, 92, 27); arrow(92.5, 32, 92, 27, color=C["kin"])
arrow(50, 16, 89, 47, color=C["cal"], ls=":", lw=1.1)   # LSC to classifier
arrow(50, 16, 90, 27, color=C["cal"], ls=":", lw=1.1)   # LSC to SAS
# GRL adversary annotation
ax.text(74.5, 44.4, "adversarial", fontsize=6.5, style="italic", color=C["core"])

ax.text(50, 58.5, "CALM-Net: multimodal, motion-disentangled, longitudinally self-calibrating decoder",
        ha="center", fontsize=9, weight="bold")
fig.tight_layout()
fig.savefig("fig_architecture.pdf", bbox_inches="tight")
fig.savefig("fig_architecture.png", dpi=150, bbox_inches="tight")
print("wrote fig_architecture.pdf/.png")
