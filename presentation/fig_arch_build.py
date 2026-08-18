"""Professional CALM-Net architecture figure with mathematical annotation.
Renders presentation/fig_arch.png (matplotlib mathtext). A matching editable
draw.io source is written separately."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "cm"})

INK = "#1E1B33"; ACC = "#5A4AE0"; RED = "#B0392B"; MUT = "#5A586E"
FACE = "#FFFFFF"; TINT = "#F3F2FB"; RTINT = "#FBF1F0"

fig, ax = plt.subplots(figsize=(15.2, 5.2), dpi=220)
ax.set_xlim(0, 18.2); ax.set_ylim(0, 6.0); ax.axis("off")


def box(cx, cy, w, h, edge=INK, face=FACE, lw=1.4, r=0.10):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={r}",
                 linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))


def txt(x, y, s, size=11, color=INK, weight="normal", style="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            weight=weight, style=style, zorder=4)


def arrow(x1, y1, x2, y2, color=INK, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=13, linewidth=lw, color=color, ls=ls,
                 shrinkA=0, shrinkB=0, zorder=3))


# ---- Stage label band (top) ----
for x, lab in [(1.75, "INPUTS"), (5.15, "STAGE 1  ENCODER"), (9.15, "STAGE 2  DISENTANGLE"),
               (12.85, "STAGE 3  CALIBRATE"), (15.75, "STAGE 4  DECIDE")]:
    txt(x, 5.72, lab, size=10.5, color=ACC, weight="bold")

# ---- INPUTS ----
box(1.75, 4.55, 2.9, 0.86, edge=INK)
txt(1.75, 4.72, "EEG", size=12.5, weight="bold")
txt(1.75, 4.40, r"$X \in \mathbb{R}^{60 \times 200}$", size=11, color=MUT)
box(1.75, 3.45, 2.9, 0.86, edge=RED)
txt(1.75, 3.62, "Head + Exo IMU", size=11.5, weight="bold", color=RED)
txt(1.75, 3.30, r"$\varphi(M) \in \mathbb{R}^{12}$", size=11, color=MUT)
box(1.75, 2.45, 2.9, 0.70, edge=MUT)
txt(1.75, 2.45, r"EOG  (4 ch)", size=11, color=MUT)

# ---- STAGE 1: ENCODER ----
box(5.15, 3.55, 3.0, 2.7, edge=INK, face=TINT)
txt(5.15, 4.55, "Neuro-kinematic", size=12.5, weight="bold")
txt(5.15, 4.22, "encoder  " + r"$f_\theta$", size=12.5, weight="bold")
txt(5.15, 3.72, "multi-band temporal conv", size=9.6, color=MUT)
txt(5.15, 3.46, "spatial covariance (SPD)", size=9.6, color=MUT)
txt(5.15, 3.20, "cross-frequency attention", size=9.6, color=MUT)
txt(5.15, 2.72, r"$z = f_\theta(X) \in \mathbb{R}^{d}$", size=11, color=INK)

# ---- STAGE 2: MID (two subspaces) ----
box(9.15, 4.35, 3.2, 1.14, edge=ACC, face=TINT)
txt(9.15, 4.60, "Intent code  " + r"$z_i$", size=11.5, weight="bold", color=ACC)
txt(9.15, 4.14, r"$\min\ I(z_i;\, M)$   via GRL + HSIC", size=9.8, color=MUT)
box(9.15, 2.70, 3.2, 1.14, edge=RED, face=RTINT)
txt(9.15, 2.95, "Artifact code  " + r"$z_a$", size=11.5, weight="bold", color=RED)
txt(9.15, 2.49, r"$z_a \rightarrow \varphi(M)$   (absorbs motion)", size=9.8, color=MUT)
# adversary tag
txt(9.15, 3.52, r"adversary $g_\psi$  /  gradient reversal", size=9.0, color=MUT, style="italic")

# ---- STAGE 3: LSC ----
box(12.85, 3.55, 2.9, 2.7, edge=INK, face=TINT)
txt(12.85, 4.55, "Longitudinal", size=12, weight="bold")
txt(12.85, 4.24, "self-calibration", size=12, weight="bold")
txt(12.85, 3.74, r"$p = \mathrm{softmax}(\ell / T^\ast)$", size=10.2, color=INK)
txt(12.85, 3.30, r"$q_{t+1} = q_t + \eta(\alpha - \mathbf{1}[y \notin C_t])$", size=9.6, color=MUT)
txt(12.85, 2.78, r"$P(y \in C(x)) \geq 1-\alpha$", size=10.2, color=ACC, weight="bold")

# ---- STAGE 4: SAS + output ----
box(15.75, 3.55, 2.6, 2.7, edge=INK, face=TINT)
txt(15.75, 4.55, "Safety-asymmetric", size=11, weight="bold")
txt(15.75, 4.24, "selection", size=11, weight="bold")
txt(15.75, 3.74, r"threshold $\tau_c$ per class", size=9.6, color=MUT)
txt(15.75, 3.30, r"$P(\hat{Y}{=}\mathrm{Walk}\,|\,Y{=}\mathrm{Stop}) \leq \beta$", size=9.2, color=RED)
txt(15.75, 2.80, "Walk / Stop / abstain", size=10.2, weight="bold", color=INK)
txt(15.75, 2.52, r"$\rightarrow$ STOP", size=10.2, weight="bold", color=RED)

# ---- ARROWS ----
arrow(3.25, 4.55, 3.65, 3.9)         # EEG -> encoder
arrow(3.25, 3.45, 3.65, 3.4)         # IMU -> encoder
arrow(3.25, 2.45, 3.65, 3.0)         # EOG -> encoder
arrow(6.65, 3.55, 7.55, 4.2, color=ACC)      # encoder -> intent
arrow(6.65, 3.55, 7.55, 2.85, color=RED)     # encoder -> artifact
arrow(10.75, 4.35, 11.4, 3.7)        # intent -> LSC
arrow(14.30, 3.55, 14.45, 3.55)      # LSC -> SAS
# IMU descriptor feeds the disentanglement target (dashed supervision path)
ax.add_patch(FancyArrowPatch((3.25, 3.28), (7.55, 2.55), connectionstyle="arc3,rad=-0.22",
             arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=RED, ls=(0, (4, 3)), zorder=1))
txt(5.15, 1.95, r"$\varphi(M)$  supervises the artifact subspace", size=8.8, color=RED, style="italic")

# ---- LOSS STRIP (bottom) ----
box(9.0, 0.86, 17.2, 0.98, edge=INK, face="#FAFAFE", lw=1.2, r=0.08)
txt(2.0, 0.86, "Training\nobjective", size=10, weight="bold", color=INK)
txt(10.0, 0.86,
    r"$\mathcal{L} = \mathcal{L}_{\mathrm{cls}}(h(z_i),\, y)\; +\; \lambda_{\mathrm{adv}}\,\mathcal{L}_{\mathrm{adv}}"
    r"\; +\; \lambda_{\mathrm{hsic}}\,\mathrm{HSIC}(z_i,\, \varphi(M))\; +\; \lambda_{\mathrm{art}}\,\| g(z_a) - \varphi(M) \|^2$",
    size=12.5, color=INK)

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
out = HERE / "fig_arch.png"
fig.savefig(out, dpi=220, facecolor="white")
print("saved", out.name)
